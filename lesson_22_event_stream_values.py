from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime


@tool
def get_weather(city: str,runtime: ToolRuntime) -> str:
    """获取某个城市的天气。"""
    runtime.emit_output_delta("正在获取{city}的天气...\n")
    runtime.emit_output_delta(f"已获取到{city}的天气是阳光明媚\n")
    return f"这里总是阳光明媚 {city}!"

agent = create_agent(
    model="deepseek-v4-pro",
    tools=[get_weather],
)

stream = agent.stream_events(
    {
        "messages": [HumanMessage("重庆天气怎么样")],
    },
    version="v3"
)

print(stream.output)