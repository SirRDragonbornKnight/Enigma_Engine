"""Smoke test: drive modkit_mcp.py as a real MCP client (like Odysseus does)."""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["modkit_mcp.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])

            res = await session.call_tool(
                "generate_code",
                {"prompt": "sort a list of numbers", "language": "python"})
            print("=== generate_code ===")
            print(res.content[0].text[:200])

            print("=== see_screen (capturing + reading your screen) ===")
            res2 = await session.call_tool("see_screen", {})
            print(res2.content[0].text[:1000])


if __name__ == "__main__":
    asyncio.run(main())
