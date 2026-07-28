from pprint import pprint

from langchain.agents import create_agent

from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents.middleware import AgentState, AgentMiddleware, hook_config
from langchain_core.messages import AIMessage
from langgraph.typing import ContextT
from typing_extensions import NotRequired
from typing import Any
from langgraph.runtime import Runtime


class CustomState(AgentState):
    # 记录模型回答次数
    rewrite_count: NotRequired[int]


class MessageLimitMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    @hook_config(can_jump_to=["model"])
    def after_model(
            self,
            state: CustomState,
            runtime: Runtime
    ) -> dict[str, Any] | None:
        last_message = state.get("messages")[-1]
        rewrite_count = state.get("rewrite_count", 0)
        print("after_model")

        if not isinstance(last_message, AIMessage):
            return None

        if len(last_message.text) > 50 and rewrite_count < 1:
            return {
                "messages": [
                    HumanMessage("刚刚的回答太复杂，请简短一点")
                ],
                "jump_to": "model",
                "rewrite_count": rewrite_count + 1
            }
        return None


agent = create_agent(
    model="deepseek-v4-pro",
    system_prompt=SystemMessage("你是一个旅游助手，会输出完整旅行建议"),
    middleware=[MessageLimitMiddleware()],
    state_schema=CustomState
)

# Invoke with custom state
result = agent.invoke({
    "messages": [HumanMessage("我要去日本")]
})

pprint(result)
