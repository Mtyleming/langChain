from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import os
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langgraph.config import get_stream_writer
import asyncio


@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    writer = get_stream_writer()
    writer(f"正在获取{city}的天气...")
    writer(f"已获取到{city}的天气是阳光明媚")
    return f"阳光明媚 {city}!"


agent = create_agent(
    model="deepseek-v4-pro",
    tools=[get_weather]
)


# for chunk in agent.stream(
#     {"messages": [HumanMessage("日本天气怎么样")]},
#     stream_mode="messages",
#     version="v2",
# ):
#     if chunk["type"] == "messages":
#         token, metadata = chunk["data"]
#         print(f"node: {metadata['langgraph_node']}")
#         print(f"content: {token.content_blocks}")
#         print("\n")

async def main():
    async for chunk in agent.astream(
            {"messages": [HumanMessage("日本天气怎么样")]},
            stream_mode=["updates", "messages", "custom"],
            version="v2",
    ):
        chunk_type = chunk["type"]
        data = chunk["data"]

        if chunk_type == "messages":
            token, metadata = data
            if getattr(token, "content", None):
                print(token.content, end="", flush=True)
            elif getattr(token, "additional_kwargs", None):
                print(token.additional_kwargs.get("reasoning_content"), end="", flush=True)
        elif chunk_type == "updates":
            print(f"\n update: {data}")
        elif chunk_type == "custom":
            print(f"\n custom: {data}")

if __name__ == "__main__":
    asyncio.run(main())
