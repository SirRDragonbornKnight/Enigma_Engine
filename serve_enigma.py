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
ARGS, _ = _p.parse_known_args()

print(f"Loading Enigma from {ARGS.model} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(ARGS.model)
model = AutoModelForCausalLM.from_pretrained(ARGS.model, dtype=torch.bfloat16).to("cuda")
model.eval()
model.config.use_cache = True
print("Enigma loaded.", flush=True)

app = FastAPI(title="Modkit · Enigma")


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: list[Msg]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False


def _encode(messages):
    text = tokenizer.apply_chat_template(
        [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)


@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "modkit"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    ids = _encode(req.messages)
    gen = dict(max_new_tokens=req.max_tokens, do_sample=True, temperature=req.temperature,
               top_p=req.top_p, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
    created = int(time.time())
    cid = f"chatcmpl-{created}"

    if req.stream:
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
    return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_prompt,
                      "completion_tokens": out.shape[1] - n_prompt,
                      "total_tokens": out.shape[1]}}


if __name__ == "__main__":
    print(f"Enigma OpenAI-compatible API → http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    print(f"In Odysseus:  /setup local http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
