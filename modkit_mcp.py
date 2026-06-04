"""Modkit MCP server — exposes Modkit mod capabilities as MCP tools.

This is the bridge that lets Odysseus (and therefore Enigma) actually *use*
the mods, instead of them sitting orphaned. It follows Odysseus's own native
pattern (see odysseus/mcp_servers/): a small in-process stdio MCP server that
calls each capability directly — no TCP router, no launcher, no two-protocol
mess. Heavy capabilities lazy-load their models only on first call.

Register in Odysseus as a stdio MCP server:
    command: python   args: ["<repo>/modkit_mcp.py"]

Add a capability: write a `_thunk(args) -> str`, then add a Tool in list_tools()
and a branch in call_tool(). That's the whole pattern.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import websockets

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

server = Server("modkit")

_service_cache: dict = {}

# The on-screen avatar listens on this local relay (mods/avatar/bus.py). We send
# it commands as a one-shot client — never bind the port — so there's no conflict
# with whoever hosts the bus (the desktop overlay's launcher does).
_AVATAR_BUS = "ws://127.0.0.1:8765"

# Heavy native deps that must be imported BEFORE the asyncio loop starts.
# A first-time ``import easyocr`` (which pulls in torch) — or any large
# C-extension import — deadlocks once the stdio event loop is live, whether
# it runs in a worker thread (``asyncio.to_thread``) OR synchronously on the
# main thread mid-request: the import machinery and the running loop fight on
# Windows. Hoisting the import to before ``asyncio.run`` (see __main__) is the
# only thing that reliably avoids it. Model *weights* still load lazily inside
# the mods on first call — only the module import is pre-warmed here.
_WARM_ON_START = ["easyocr", "kokoro"]  # vision OCR + avatar voice; weights still lazy-load (missing dep is ignored)


def _warm_imports() -> None:
    """Pre-import heavy native deps on the main thread before the event loop."""
    import importlib
    for mod in _WARM_ON_START:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # a missing optional dep must not kill the server
            print(f"modkit_mcp: warm import {mod!r} failed: {exc}",
                  file=sys.stderr, flush=True)


def _load_service(mod_file: str, class_name: str):
    """Import a mod's service class in-process (cached, path-based so it works
    regardless of whether `mods` is a package)."""
    if class_name in _service_cache:
        return _service_cache[class_name]
    spec = importlib.util.spec_from_file_location(
        f"modkit_{class_name}", ROOT / mod_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # register before exec, or a mod's @dataclass fails
    spec.loader.exec_module(module)   # (dataclasses resolves types via sys.modules[cls.__module__])
    cls = getattr(module, class_name)
    _service_cache[class_name] = cls
    return cls


# --------------------------------------------------------------------------
# Capability: code generation (codegen mod, offline template provider)
# --------------------------------------------------------------------------
def _generate_code(prompt: str, language: str) -> str:
    CodeGen = _load_service("mods/codegen/codegen.py", "CodeGen")
    svc = CodeGen(default_provider="template")
    r = svc.handle_command(
        "generate",
        {"prompt": prompt, "language": language, "provider": "template"},
    )
    if r.get("success"):
        return r.get("code", "") or "(no code returned)"
    return f"Error: {r.get('error', 'unknown error')}"


# --------------------------------------------------------------------------
# Capability: screen sight (vision mod — capture via PIL + read via OCR)
# --------------------------------------------------------------------------
_vision_singleton = None


def _get_vision():
    """One Vision instance so the OCR model stays resident across calls."""
    global _vision_singleton
    if _vision_singleton is None:
        Vision = _load_service("mods/vision/vision.py", "Vision")
        _vision_singleton = Vision()
    return _vision_singleton


def _see_screen() -> str:
    r = _get_vision().handle_command("ocr", {})  # capture (PIL) + read (easyocr)
    if not r.get("success"):
        return f"Error: {r.get('error', 'unknown error')}"
    text = (r.get("text") or "").strip()
    if text:
        return text
    return ("(captured the screen but read no text — "
            f"{r.get('note', 'no OCR backend available')})")


# --------------------------------------------------------------------------
# Capability: avatar speech (voice mod's Kokoro TTS → WAV → overlay lip-sync)
# --------------------------------------------------------------------------
import time as _time


def _avatar_speak_wav(text: str, voice: str, speed: float) -> Path:
    """Synthesize `text` to a WAV with the voice mod's Kokoro pipeline. No fallback."""
    LocalTTS = _load_service("mods/voice/voice.py", "LocalTTS")
    tts = LocalTTS(voice=voice, speed=speed)
    if not tts.load():
        raise RuntimeError("Kokoro TTS unavailable — pip install --user kokoro (no fallback)")
    out = ROOT / "outputs" / "voice" / f"avatar_{int(_time.time() * 1000)}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    path = tts.generate_to_file(text, out_path=out)
    if not path:
        raise RuntimeError("Kokoro synthesis produced no audio")
    return Path(path)


# --------------------------------------------------------------------------
# MCP wiring
# --------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_code",
            description=(
                "Generate code from a natural-language description using "
                "Modkit's codegen capability (offline templates)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What the code should do",
                    },
                    "language": {
                        "type": "string",
                        "description": "python or javascript (default python)",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="see_screen",
            description=(
                "Capture the user's screen and read the text on it (OCR). "
                "Use this whenever the user asks about — or refers to — what "
                "is on their screen right now."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="avatar_express",
            description=(
                "Make the on-screen avatar (the desktop companion) perform an "
                "emotion / gesture as body language while you reply — bring your "
                "answer to life. Use 'talk' while explaining, 'happy' for good "
                "news, 'wag' (a tail wag) when pleased, 'nod' for yes, 'shake' "
                "for no, 'sad', 'alert' to get attention. No-op if the avatar "
                "isn't running, so it's always safe to call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "talk, happy, wag, nod, shake, sad, or alert",
                    },
                    "duration": {
                        "type": "number",
                        "description": "seconds to hold it (default 2.5)",
                    },
                },
                "required": ["emotion"],
            },
        ),
        Tool(
            name="avatar_say",
            description=(
                "Make the on-screen avatar SPEAK your text aloud: it synthesizes "
                "speech with the local Kokoro voice and the avatar lip-syncs (the "
                "mouth/jaw tracks the audio) with talking body language. Use for "
                "spoken replies. 100% local, no cloud. No-op if the avatar/bus "
                "isn't running, so it's safe to call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "what to say aloud"},
                    "voice": {"type": "string", "description": "Kokoro voice id (default af_heart; e.g. af_bella, am_adam, bf_emma)"},
                    "speed": {"type": "number", "description": "speech speed, 0.5–2.0 (default 1.0)"},
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="avatar_command",
            description=(
                "Full control of the on-screen avatar, beyond express/say. ONE action "
                "per call. Actions:\n"
                "  load — switch model (url, e.g. ./models/glados/scene.gltf)\n"
                "  size — resize (value, ~0.3–1.5)\n"
                "  moveTo — reposition (px, py in screen pixels)\n"
                "  recolor — tint a part (name = material e.g. hair/body/Eye, color = #rrggbb)\n"
                "  attach — add a prop/clothing mesh (url to .glb/.fbx, optional bone like "
                "righthand/head/back, category prop|clothes|furniture)\n"
                "  detach — remove an attachment (id, or omit for all)\n"
                "  springTune — hair/tail feel (stiffness, drag, gravity)\n"
                "  stop — stop speaking\n"
                "No-op if the overlay/bus isn't running, so it's safe to call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "load, size, moveTo, recolor, attach, detach, springTune, stop"},
                    "url": {"type": "string", "description": "model/prop path for load/attach"},
                    "value": {"type": "number", "description": "scale for size"},
                    "px": {"type": "number"}, "py": {"type": "number"},
                    "name": {"type": "string", "description": "material name (recolor)"},
                    "color": {"type": "string", "description": "#rrggbb (recolor)"},
                    "bone": {"type": "string", "description": "attach target (righthand/lefthand/head/back/hips…)"},
                    "category": {"type": "string", "description": "prop | clothes | furniture (attach)"},
                    "id": {"type": "string", "description": "attachment id (detach)"},
                    "stiffness": {"type": "number"}, "drag": {"type": "number"}, "gravity": {"type": "number"},
                },
                "required": ["action"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    args = arguments or {}
    try:
        if name == "generate_code":
            prompt = args.get("prompt", "")
            if not prompt:
                return [TextContent(type="text", text="Error: prompt is required")]
            language = args.get("language", "python")
            code = await asyncio.to_thread(_generate_code, prompt, language)
            return [TextContent(type="text", text=code)]
        if name == "see_screen":
            text = await asyncio.to_thread(_see_screen)
            return [TextContent(type="text", text=text)]
        if name == "avatar_express":
            emotion = (args.get("emotion") or args.get("name") or "talk").strip()
            duration = float(args.get("duration") or 2.5)
            try:
                async with websockets.connect(_AVATAR_BUS, open_timeout=2) as ws:
                    await ws.send(json.dumps(
                        {"action": "express", "name": emotion, "dur": duration}))
                return [TextContent(type="text",
                                    text=f"avatar: expressed '{emotion}' for {duration:g}s")]
            except Exception as exc:
                return [TextContent(type="text",
                                    text=f"avatar not reachable (overlay/bus not running): {exc}")]
        if name == "avatar_say":
            text = (args.get("text") or "").strip()
            if not text:
                return [TextContent(type="text", text="Error: text is required")]
            voice = (args.get("voice") or "af_heart").strip()
            speed = float(args.get("speed") or 1.0)
            try:
                wav = await asyncio.to_thread(_avatar_speak_wav, text, voice, speed)
            except Exception as exc:
                return [TextContent(type="text", text=f"avatar_say: TTS failed — {exc}")]
            try:
                async with websockets.connect(_AVATAR_BUS, open_timeout=2) as ws:
                    await ws.send(json.dumps({"action": "say", "url": wav.as_uri()}))
                return [TextContent(type="text", text=f"avatar: speaking ({len(text)} chars)")]
            except Exception as exc:
                return [TextContent(type="text",
                                    text=f"avatar not reachable (overlay/bus not running): {exc}")]
        if name == "avatar_command":
            action = (args.get("action") or "").strip()
            if not action:
                return [TextContent(type="text", text="Error: action is required")]
            cmd = {"action": action}
            for k in ("url", "value", "px", "py", "name", "color", "bone", "category",
                      "id", "stiffness", "drag", "gravity", "breeze", "scale", "dur", "pos", "rot"):
                if args.get(k) is not None:
                    cmd[k] = args[k]
            try:
                async with websockets.connect(_AVATAR_BUS, open_timeout=2) as ws:
                    await ws.send(json.dumps(cmd))
                return [TextContent(type="text", text=f"avatar: '{action}' sent")]
            except Exception as exc:
                return [TextContent(type="text",
                                    text=f"avatar not reachable (overlay/bus not running): {exc}")]
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as exc:  # never crash the server on a bad call
        return [TextContent(type="text", text=f"Error: {exc}")]


async def run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    _warm_imports()  # MUST run before asyncio.run — see _WARM_ON_START
    asyncio.run(run())
