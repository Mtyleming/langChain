from typing import Any

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk, AIMessage, AnyMessage,ToolMessage,HumanMessage
from langchain_core.tools import tool

from lesson_18_tool_guardrail import safety_guardrail


@tool
def get_weather(city: str) -> str:
    """获取某个城市的天气。"""

    return f"这里总是阳光明媚 {city}!"


agent = create_agent(
    model="deepseek-v4-pro",
    tools=[get_weather],
    middleware=[safety_guardrail]
)

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")



for chunk in agent.stream(
    {"messages": [HumanMessage("日本天气怎么样")]},
    stream_mode=["messages", "updates", "custom"],
    version="v2",
):
    # if chunk["type"] == "messages":
    #     token, metadata = chunk["data"]
    #     if isinstance(token, AIMessageChunk):
    #         _render_message_chunk(token)
    if chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
    elif chunk["type"] == "custom":
        # 在流中访问已完成的消息
        print(f"custom Tool calls: {chunk['data'].tool_calls}")