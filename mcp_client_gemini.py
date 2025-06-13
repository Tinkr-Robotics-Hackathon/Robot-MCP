import asyncio
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


def get_genai_client():
    return genai.Client(
        vertexai=False,
        api_key=os.environ["GOOGLE_AI_STUDIO"],
    )


async def main():
    client = get_genai_client()

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_robot_server.py", "--transport", "stdio"],
        env=os.environ,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_meta = await session.list_tools()

            tools = [
                types.Tool(
                    function_declarations=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                k: v
                                for k, v in tool.inputSchema.items()
                                if k not in ["additionalProperties", "$schema"]
                            },
                        }
                    ]
                )
                for tool in tools_meta.tools
            ]

            prompt = "Rotate the gripper joint by 45 degrees."
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=tools,
                ),
            )

            if response.function_calls:
                first_call = response.function_calls[0]
                name = first_call.name
                args = first_call.args
                print(f"> Gemini calls: {name}({args})")

                result = await session.call_tool(name, arguments=args)
                print("> Tool result:", result)
            else:
                print("> Gemini says:", response.text)


if __name__ == "__main__":
    asyncio.run(main())
