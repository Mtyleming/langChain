from pprint import pprint

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

import os
from langchain_deepseek import ChatDeepSeek

# api_key = os.getenv("DEEPSEEK_API_KEY")
# deepseek = ChatDeepSeek(
#     model="deepseek-v4-pro",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     extra_body={"thinking": {"type": "disabled"}}
# )


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    return handler(request.override(system_message=SystemMessage("你是一个翻译助手，默认中英文互翻")))


agent = create_agent(
    model="deepseek-v4-pro",  # Default model
    system_prompt=SystemMessage("你是一个旅游助手，会输出剪短的旅行建议"),
    middleware=[dynamic_model_selection]
)

res = agent.invoke({"messages": HumanMessage("重庆怎么样")})
pprint(res)
