from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime


@tool
def sub_sub_call_weather(city: str,runtime: ToolRuntime) -> str:
    """孙子agent真正的获取某个城市的天气。"""
    runtime.emit_output_delta("正在获取{city}的天气...\n")
    runtime.emit_output_delta(f"已获取到{city}的天气是阳光明媚\n")
    return f"这里总是阳光明媚 {city}!"

@tool
def sub_call_weather(city: str,runtime: ToolRuntime) -> str:
    """子智能体的工具 获取某个城市的天气。"""
    res = sub_sub_agent.invoke({"messages":[HumanMessage(city)]})
    return res["messages"][-1].text

@tool
def agent_call_weather(city: str,runtime: ToolRuntime) -> str:
    """父智能体获取某个城市的天气。"""
    res = sub_agent.invoke({"messages":[HumanMessage(city)]})
    return res["messages"][-1].text

agent = create_agent(
    model="deepseek-v4-pro",
    tools=[agent_call_weather],
)

sub_agent = create_agent(
    model="deepseek-v4-pro",
    tools=[sub_call_weather],
    name="weatherAgent",
)

sub_sub_agent = create_agent(
    model="deepseek-v4-pro",
    tools=[sub_sub_call_weather],
    name="subWeatherAgent",
)



stream = agent.stream_events(
    {
        "messages": [HumanMessage("日本天气怎么样")],
    },
    version="v3"
)

# for message in stream.messages:
#     print(f"[{message.node}] ", end="")
#     for delta in message.text:
#         print(delta, end="", flush=True)

for sb_agent in stream.subagents:
    print(f"[{sb_agent.name}] ", end="")
    for message in sb_agent.messages:
        print("------[sb_agent]---------")
        print(message.text, end="", flush=True)
    print(f"------{sb_agent.subagents.name}---------")
    # for sb_sb_agent in sb_agent.subagents:
    #     print("-------sb_sb_agent--------")
    #     for message in sb_sb_agent.messages:
    #         print(message.text, end="", flush=True)