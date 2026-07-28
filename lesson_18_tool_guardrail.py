from typing import Any, Literal

from langchain.agents.middleware import after_agent, AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel
import os
from langchain.chat_models import init_chat_model

class ResponseSafety(BaseModel):
    """评估一个反应是否安全。"""
    evaluation: Literal["safe", "unsafe"]

api_key = os.getenv("DEEPSEEK_API_KEY")
safety_model = init_chat_model(
    model="deepseek-v4-pro",
    api_key=api_key,
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """基于模型的护栏：使用大型语言模型（LLM）来评估响应安全性。"""
    stream_writer = get_stream_writer()
    # 获得模型响应
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    #使用其他模型评估安全性
    model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
    result = model_with_tools.invoke(
        [
            SystemMessage("评估这种AI反应是否总体上安全。"),
            HumanMessage(f"AI 响应: {last_message.text}")
        ]
    )
    stream_writer(result)

    tool_call = result.tool_calls[0]
    if tool_call["args"]["evaluation"] == "unsafe":
        last_message.content = "我无法给出这样的回答。请重新表述您的请求。"

    return None