import os
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool

api_key = os.getenv("DEEPSEEK_API_KEY")
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

@tool
def get_weather(query: str) -> str:
    """获取天气"""
    return f"{query}城市的天气目前是晴天"

llm = model.bind_tools([get_weather])

reasoning_content = ""
content = ""
tool_calls = []
for chunk in llm.stream("重庆天气如何"):
    print(chunk)
    if chunk.additional_kwargs:
        reasoning_content += chunk.additional_kwargs.get("reasoning_content")
    if chunk.content:
        content += chunk.content
    if chunk.tool_calls:
        for index, tool_call in enumerate(chunk.tool_calls):
            if tool_call["name"]: tool_calls.append({"tool_name": tool_call["name"]})

print(f"Agent在思考：{reasoning_content}")
print(f"Agent的回复：{content}")
print(f"Agent调用的工具：{tool_calls}")

