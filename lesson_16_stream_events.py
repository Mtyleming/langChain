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

stream = agent.stream_events(
    {"messages": [HumanMessage("日本天气怎么样")]},
    version="v3",
)
for message in stream.messages:
    # for token in message.reasoning:
    #     print(f"{token}", end="")
   for token in message.text:
        print(f"{token}", end="", flush=True)
