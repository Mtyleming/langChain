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

stream = agent.stream_events({
    "messages": [HumanMessage("日本天气怎么样")],
}, version="v3")

for message in stream.messages:
    # print(f"[{message.node}] ", end="")
    # for delta in message.text:
    #     print(delta, end="", flush=True)

    # for delta in message.reasoning:
    #     print(f"[thinking] {delta}", end="", flush=True)
    #
    # for delta in message.text:
    #     print(delta, end="", flush=True)

    for chunk in message.tool_calls:
        print(f"tool call chunk: {chunk}")

# final_state = stream.output
# print("-------------------------\n")
# print(final_state)

# for call in stream.tool_calls:
#     print(f"{call.tool_name}({call.input})")
#     print("--------------------\n")
#     for delta in call.output_deltas:
#         print(delta, end="", flush=True)
#     print("--------------------\n")
#     print(call.output, call.error)