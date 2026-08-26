import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools from MCP server:")
            for tool in tools.tools:
                print(f" - {tool.name}: {tool.description}")

            result1 = await session.call_tool("list_directory", {"path": "."})
            print("\nTool 1 result (list_directory):")
            print(result1.content[0].text)

            result2 = await session.call_tool("read_file", {"path": "hello.py"})
            print("\nTool 2 result (read_file):")
            print(result2.content[0].text)

asyncio.run(main())
