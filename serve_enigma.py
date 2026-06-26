#!/usr/bin/env python
"""Serve the local Enigma model as an OpenAI-compatible /v1 endpoint, so Odysseus
(or any OpenAI client) can chat with it. This is Modkit's thin serving glue —
the model runs here; the front-end is Odysseus.

  python serve_enigma.py                  # serves models/enigma-8b on :8000
  # then, in Odysseus chat:  /setup local http://127.0.0.1:8000/v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:  # Windows consoles default to cp1252 and crash printing unicode.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
MODEL_ID = "enigma"

_p = argparse.ArgumentParser()
_p.add_argument("--model", default=str(ROOT / "models" / "enigma-8b"))
_p.add_argument("--host", default="127.0.0.1")
_p.add_argument("--port", type=int, default=8000)
_p.add_argument("--quant", choices=["4bit", "bf16"], default="4bit",
                help="4bit = ~5-6 GB NF4 (default); bf16 = ~16 GB max quality")
ARGS, _ = _p.parse_known_args()

print(f"Loading Enigma from {ARGS.model} ({ARGS.quant}) ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(ARGS.model)
if ARGS.quant == "4bit":
    from transformers import BitsAndBytesConfig
    _bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model, quantization_config=_bnb, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model, dtype=torch.bfloat16).to("cuda")
model.eval()
model.config.use_cache = True
# 4-bit quantization stages full bf16 weights on the GPU, then drops them; torch's
# caching allocator keeps that memory reserved. Release it so the steady-state
# footprint reflects the real ~6 GB model, not the ~16 GB load transient.
import gc

gc.collect()
torch.cuda.empty_cache()
print("Enigma loaded.", flush=True)

app = FastAPI(title="Modkit · Enigma")


class Msg(BaseModel):
    role: str
    content: str | None = None
    # OpenAI tool-calling shapes: assistant turns may carry `tool_calls`, and a
    # tool result comes back as role="tool" with the originating `tool_call_id`.
    # Pass these through so multi-turn tool conversations round-trip.
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: list[Msg]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    tools: list | None = None              # OpenAI function schemas, if offered
    tool_choice: str | dict | None = None  # accepted; the model decides


def _encode(messages, tools=None):
    msgs = [m.model_dump(exclude_none=True) for m in messages]
    text = tokenizer.apply_chat_template(
        msgs, tools=tools or None, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _split_tool_calls(text, created):
    """Pull Qwen3 ``<tool_call>{...}</tool_call>`` blocks out of generated text and
    convert them to OpenAI ``tool_calls``. Returns (tool_calls, remaining_text)."""
    calls = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue  # malformed block — leave it in the text rather than guess
        args = obj.get("arguments", {})
        calls.append({
            "id": f"call_{created}_{i}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": args if isinstance(args, str) else json.dumps(args),
            },
        })
    clean = _TOOL_CALL_RE.sub("", text).strip()
    return calls, clean


@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "modkit"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    ids = _encode(req.messages, req.tools)
    gen = dict(max_new_tokens=req.max_tokens, do_sample=True, temperature=req.temperature,
               top_p=req.top_p, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
    created = int(time.time())
    cid = f"chatcmpl-{created}"

    # Plain chat streams token-by-token. When tools are offered we must read the
    # whole output to extract <tool_call> blocks, so those requests generate fully
    # and emit a single delta afterwards (handled below).
    if req.stream and not req.tools:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        threading.Thread(target=model.generate,
                         kwargs=dict(**ids, streamer=streamer, **gen), daemon=True).start()

        def event_stream():
            for chunk in streamer:
                payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": MODEL_ID,
                           "choices": [{"index": 0, "delta": {"content": chunk},
                                        "finish_reason": None}]}
                yield f"data: {json.dumps(payload)}\n\n"
            done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    with torch.no_grad():
        out = model.generate(**ids, **gen)
    n_prompt = ids.input_ids.shape[1]
    text = tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
    tool_calls, clean = _split_tool_calls(text, created) if req.tools else ([], text)
    finish = "tool_calls" if tool_calls else "stop"
    message = {"role": "assistant", "content": (clean or None) if tool_calls else clean}
    if tool_calls:
        message["tool_calls"] = tool_calls

    if req.stream:  # tool-enabled stream: emit the parsed result as one delta
        def event_stream():
            delta = {"role": "assistant"}
            if tool_calls:
                delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(tool_calls)]
            else:
                delta["content"] = clean
            yield ("data: " + json.dumps(
                {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": MODEL_ID,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n")
            done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": n_prompt,
                      "completion_tokens": out.shape[1] - n_prompt,
                      "total_tokens": out.shape[1]}}


if __name__ == "__main__":
    print(f"Enigma OpenAI-compatible API → http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    print(f"In Odysseus:  /setup local http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
