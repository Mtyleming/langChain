# import pprint
#
# from langchain.agents import create_agent
#
# from langchain_core.messages import HumanMessage
#
# # agent = create_agent(model="deepseek:deepseek-v4-pro")
# import os
# from langchain.agents import create_agent
# from langchain_core.messages import HumanMessage
# from langchain_deepseek import ChatDeepSeek
#
# # 1. 先创建模型对象
# api_key = os.getenv("DEEPSEEK_API_KEY")
#
# model = ChatDeepSeek(
#     model="deepseek-v4-pro",
#     api_key=api_key,
#     temperature=0.7,
#     max_tokens=1000,
# )
#
# # 2. 将模型对象传入 agent
# agent = create_agent(
#     model=model,  # 直接传入模型对象
#     tools=[],  # 可选：传入工具
#     system_prompt="你是一个智能助手",  # 可选：系统提示
# )
#
# #res = agent.invoke({"messages": HumanMessage("重庆天气如何")})
#
#
# res = agent.invoke({"messages": HumanMessage("重庆天气如何")})
# pprint.pprint(res)
#

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_ollama import ChatOllama

import os
from langchain_deepseek import ChatDeepSeek

api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    extra_body={"thinking": {"type": "disabled"}}
)
ollama = ChatOllama(
    model="qwen3.5:4b",
    temperature=0
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = deepseek
    else:
        model = ollama

    return handler(request.override(model=model))

agent = create_agent(
    model=deepseek,  # Default model
    middleware=[dynamic_model_selection]
)