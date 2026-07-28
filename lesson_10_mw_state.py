from langchain.agents.middleware import before_agent,after_model, AgentState
from langgraph.runtime import Runtime
from typing import Any
from typing_extensions import NotRequired
from pprint import pprint

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent

class TrackingState(AgentState):
    model_call_count: NotRequired[int]

@after_model(state_schema=TrackingState)
def increment_after_model(state: TrackingState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"after_model--- {state}",state)
    return {"model_call_count": state.get("model_call_count", 0) + 1}


@before_agent()
def increment_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"before_agent--- {state}",state)
    return {"model_call_count": state.get("model_call_count", 0) + 1}


agent = create_agent(
    model="deepseek-v4-pro",
    system_prompt=SystemMessage("你是一个旅游助手，会输出剪短的旅行建议"),
    middleware=[increment_before_agent, increment_after_model]
)

res = agent.invoke({
    "messages": HumanMessage("重庆怎么样"),
    "model_call_count": 0
})
pprint(res)