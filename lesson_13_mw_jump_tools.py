from pprint import pprint

from langchain.agents import create_agent

from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.middleware import AgentState, AgentMiddleware, hook_config
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from typing import Any
from langgraph.runtime import Runtime

@tool
def search(param: str):
    """这是一个测试工具"""
    return f"Result.s for: {param} 测试"

class MessageLimitMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["tools"])
    def before_model(
            self,
            state: AgentState,
            runtime: Runtime
    ) -> dict[str, Any] | None:
        last_message = state.get("messages")[-1]
        if not isinstance(last_message, HumanMessage):
            return None

        return {
            "messages": [
                AIMessage(
                    content="",
                    id = "f45749ed-c238-49c5-b40c-ba61c9103274",
                    tool_calls=[
                        {
                            "id": "call_00_UuEfeDY30T0O1SMjIXeY7166",
                            "name": "search",
                            "args": {
                                "param": "东京"
                            },
                            "type":"tool_call"
                        }
                    ]
                )
            ],
            "jump_to": "tools"
        }


agent = create_agent(
    model="deepseek-v4-pro",
    system_prompt=SystemMessage("你是一个旅游助手，会输出完整旅行建议"),
    middleware=[MessageLimitMiddleware()],
    tools=[search]
)

# Invoke with custom state
result = agent.invoke({
    "messages": [HumanMessage("我要去大阪")]
})

pprint(result)
