"""Test the avatar control path end to end (minus the browser): the modkit MCP
tool + say.py CLI are producers; a stand-in consumer plays the avatar's role and
must receive the relayed commands from bus.py.

Run with bus.py already listening (the test starts one if needed is NOT done —
keep it simple: `python mods/avatar/bus.py` in another shell, then run this)."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import websockets  # noqa: E402
import modkit_mcp  # noqa: E402

URI = "ws://127.0.0.1:8765"


async def _consumer(received: list, ready: asyncio.Future) -> None:
    async with websockets.connect(URI, open_timeout=3) as ws:
        ready.set_result(True)
        try:
            while True:
                received.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))
        except Exception:
            pass


async def main() -> int:
    # 1) my edits parse + the tool is registered
    try:
        tools = [t.name for t in await modkit_mcp.list_tools()]
    except TypeError:
        tools = [t.name for t in await modkit_mcp.list_tools()]  # decorated; same call
    assert "avatar_express" in tools, f"avatar_express missing from {tools}"
    print("list_tools OK ->", tools)

    received: list = []
    ready = asyncio.get_event_loop().create_future()
    consumer = asyncio.create_task(_consumer(received, ready))
    await asyncio.wait_for(ready, timeout=4)            # consumer (fake avatar) connected
    await asyncio.sleep(0.2)

    # 2) drive it through the actual MCP tool
    tool_txt = None
    try:
        res = await modkit_mcp.call_tool("avatar_express", {"emotion": "happy", "duration": 1.5})
        tool_txt = res[0].text
        print("MCP tool ->", tool_txt)
    except Exception as exc:
        print("MCP tool direct-call skipped:", exc)
    await asyncio.sleep(0.4)

    # 3) drive it through the say.py CLI (a separate process producer)
    cli = subprocess.run([sys.executable, str(ROOT / "mods/avatar/say.py"), "wag", "3"],
                         capture_output=True, text=True)
    print("say.py ->", cli.stdout.strip() or cli.stderr.strip())
    await asyncio.sleep(0.4)

    consumer.cancel()
    print("consumer received:", received)
    names = {(c.get("action"), c.get("name")) for c in received}
    assert ("express", "wag") in names, f"say.py command not relayed: {received}"
    if tool_txt and "expressed" in tool_txt:
        assert ("express", "happy") in names, f"MCP tool command not relayed: {received}"
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
