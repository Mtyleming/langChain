from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
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
        "messages": [SystemMessage("简短的给出结果 不要多说"),HumanMessage("重庆天气怎么样")],
    },
    version="v3"
)

printed_reasoning = False
printed_chunk = False

for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        printed_reasoning = False
        printed_chunk = False
        for reasoning in item.reasoning:
            if not printed_reasoning:
                printed_reasoning = True
                print(f"\n[思考]:{reasoning} ", end="", flush=True)
            print(reasoning, end="",flush= True)
        # print("\n---------")
        for chunk in item.text:
            if not printed_chunk:
                printed_chunk = True
                print(f"\n[LLM]:{chunk} ", end="", flush=True)
            print(chunk, end="",flush= True)
        finalized = item.tool_calls.get()
        if finalized:
            print(f"\n[最终工具调用]: {finalized}")
    elif name == "tool_calls":

        print(f"\n[工具调用前]：{item.tool_name}({item.input})")
        for delta in item.output_deltas:
            print(f"\n[工具执行]：{delta}", end="", flush=True)
        print(f"\n[工具调用后]：{item.output},{item.error}")
    elif name == "values":
        print(f"\n[状态更新]{item}")
print("-------------------------\n")
print("f[最终状态]：{stream.output}")