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
import os
import subprocess
import sys
import uuid
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
# Capability: avatar model introspection (headless glTF scan → capability profile)
# --------------------------------------------------------------------------
def _avatar_node() -> str:
    """The portable Node the avatar overlay ships with; fall back to PATH `node`.
    Resolved via %LOCALAPPDATA% (no hardcoded user path) so it travels per-machine."""
    base = os.environ.get("LOCALAPPDATA")
    if base:
        cand = Path(base) / "node-portable" / "node.exe"
        if cand.exists():
            return str(cand)
    return "node"


def _describe_model(model: str | None) -> str:
    """Run the headless profiler (mods/avatar/tools/describe.mjs) and return its JSON.
    `model` = a model id or path; omit for a brief menu of every installed model. The
    profile is name-agnostic: parts are addressed by role / material index / VRM preset."""
    script = ROOT / "mods" / "avatar" / "tools" / "describe.mjs"
    if not script.exists():
        return f"Error: avatar scanner not found at {script}"
    cmd = [_avatar_node(), str(script)]
    if model:
        cmd.append(model)
    try:
        proc = subprocess.run(
            cmd, cwd=str(script.parent.parent),  # mods/avatar, so ./models + overrides resolve
            capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    except Exception as exc:
        return f"avatar_describe: could not run the scanner — {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        return f"avatar_describe: scanner exited {proc.returncode} — {err or 'no output'}"
    return (proc.stdout or "").strip() or "(scanner produced no output)"


# --------------------------------------------------------------------------
# Capability: live avatar self-report (two-way bus RPC — ground truth from the overlay)
# --------------------------------------------------------------------------
async def _avatar_rpc(cmd: dict, timeout: float = 3.0):
    """Send a command to the avatar bus and AWAIT the overlay's reply (two-way). Returns the
    reply's `result`, or None if none arrives in time. The bus broadcasts the overlay's reply
    to every other client, so we tag a reqId and filter for our own."""
    req_id = uuid.uuid4().hex
    payload = json.dumps({**cmd, "reqId": req_id})
    async with websockets.connect(_AVATAR_BUS, open_timeout=2) as ws:
        await ws.send(payload)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.05, deadline - loop.time()))
            except asyncio.TimeoutError:
                break
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("type") == "reply" and msg.get("reqId") == req_id:
                return msg.get("result")
    return None


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
                "  setMorph — drive a morph target by index (probe to find the mouth → face.mouthMorph)\n"
                "  gesture — animated whole-body MOTION (name: jump|flip|laydown|getup|clap; optional dur) — fires LIVE/instantly\n"
                "  express — emote (name: happy|sad|talk|wag|nod|alert|shake; optional dur)\n"
                "  rotate — turn her (x,y,z degrees, or axis+deg)\n"
                "  lookAt — gaze at a screen point (px,py)   ·   lookMode — head|eyes|both (mode)\n"
                "  goTo — move by name (to: center|topleft|cursor|…) or px,py\n"
                "  regionWeight — soft-body jiggle per area (region, weight)   ·   morph — set a shape key (index, value)\n"
                "  setMesh — show/hide a part (index, on)   ·   monitor — hop screens (value: next|prev|index)\n"
                "  stop — stop speaking\n"
                "These run LIVE — the overlay is its own process, so commands take effect the instant they "
                "arrive; fire them in short bursts mid-task to drive her like a body.\n"
                "No-op if the overlay/bus isn't running, so it's safe to call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "load|size|moveTo|goTo|rotate|gesture|express|lookAt|lookMode|recolor|attach|detach|springTune|setMorph|regionWeight|morph|setMesh|monitor|stop"},
                    "url": {"type": "string", "description": "model/prop path for load/attach"},
                    "value": {"type": ["number", "string"], "description": "scale for size · morph value 0..1 · monitor: next|prev|<index>"},
                    "px": {"type": "number"}, "py": {"type": "number"},
                    "name": {"type": "string", "description": "material name (recolor)"},
                    "color": {"type": "string", "description": "#rrggbb (recolor)"},
                    "bone": {"type": "string", "description": "attach target (righthand/lefthand/head/back/hips…)"},
                    "category": {"type": "string", "description": "prop | clothes | furniture (attach)"},
                    "id": {"type": "string", "description": "attachment id (detach)"},
                    "stiffness": {"type": "number"}, "drag": {"type": "number"}, "gravity": {"type": "number"},
                    "index": {"type": "number", "description": "morph index (setMorph) or material index (recolor)"},
                    "idx": {"type": "number", "description": "alias for index"},
                    "on": {"type": "boolean", "description": "toggle on/off (showBones / setMesh)"},
                    "deg": {"type": "number", "description": "degrees (hue / single-axis rotate)"},
                    "dur": {"type": "number", "description": "seconds (gesture / express duration)"},
                    "x": {"type": "number", "description": "rotate pitch°"},
                    "y": {"type": "number", "description": "rotate yaw°"},
                    "z": {"type": "number", "description": "rotate roll°"},
                    "axis": {"type": "string", "description": "x|y|z for a single-axis rotate"},
                    "mode": {"type": "string", "description": "head|eyes|both (lookMode)"},
                    "to": {"type": "string", "description": "anchor name for goTo (center|topleft|topright|cursor|…)"},
                    "region": {"type": "string", "description": "soft-body area (regionWeight)"},
                    "weight": {"type": "number", "description": "jiggle amount 0..2 (regionWeight)"},
                    "breeze": {"type": "number", "description": "springTune ambient sway"},
                    "scale": {"type": "number", "description": "attach fit: prop scale"},
                    "pos": {"type": "array", "items": {"type": "number"}, "description": "attach fit: [x,y,z] offset"},
                    "rot": {"type": "array", "items": {"type": "number"}, "description": "attach fit: [x,y,z] degrees"},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="avatar_describe",
            description=(
                "Scan a 3D avatar model BEFORE driving it and return its capability "
                "profile as JSON: which rig roles resolved (found by STRUCTURE, not by "
                "name), the facial / lip-sync system, recolorable materials (by index), "
                "morph targets, spring/dynamic chains (hair/tail/wires), health, and "
                "which avatar_express / say / command actions are actually supported on "
                "it. Models share NO naming and some have corrupted names, so address "
                "parts by the returned ROLE / material INDEX / VRM PRESET — never by a "
                "guessed name. Omit `model` for a brief menu of all installed models. "
                "Call this before avatar_command load/recolor/attach so your commands "
                "fit the actual model."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "model id or path (e.g. 'roxanne_wolf' or './models/x/scene.gltf'); omit for the menu of all models",
                    },
                },
            },
        ),
        Tool(
            name="avatar_status",
            description=(
                "Report the LIVE avatar overlay's ACTUAL state (ground truth from the running "
                "overlay, not a static scan): current model, facial/lip-sync mode (whether the "
                "loaded model even HAS a mouth), recolorable materials BY INDEX (the authority "
                "for avatar_command recolor — get indices here; names are unreliable), resolved "
                "rig roles, attachments, toggles. Call before recolor, or to check whether the "
                "loaded model can lip-sync. `what`: state (default) | materials | facial | model. "
                "Empty if the overlay/bus isn't running."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "what": {"type": "string", "description": "state (default) | materials | facial | model"},
                },
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
                      "id", "stiffness", "drag", "gravity", "breeze", "scale", "dur", "pos", "rot",
                      "index", "idx", "on", "deg", "x", "y", "z", "axis", "mode", "to", "anchor",
                      "region", "weight", "gesture", "open", "what"):
                if args.get(k) is not None:
                    cmd[k] = args[k]
            try:
                async with websockets.connect(_AVATAR_BUS, open_timeout=2) as ws:
                    await ws.send(json.dumps(cmd))
                return [TextContent(type="text", text=f"avatar: '{action}' sent")]
            except Exception as exc:
                return [TextContent(type="text",
                                    text=f"avatar not reachable (overlay/bus not running): {exc}")]
        if name == "avatar_describe":
            out = await asyncio.to_thread(_describe_model, args.get("model"))
            return [TextContent(type="text", text=out)]
        if name == "avatar_status":
            what = (args.get("what") or "state").strip()
            try:
                result = await _avatar_rpc({"action": "query", "what": what})
            except Exception as exc:
                return [TextContent(type="text", text=f"avatar not reachable (overlay/bus not running): {exc}")]
            if result is None:
                return [TextContent(type="text", text="(no reply — the avatar overlay/bus isn't running)")]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
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
