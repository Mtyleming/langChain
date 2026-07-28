import os

from langchain_deepseek import ChatDeepSeek

from langchain.chat_models import init_chat_model

from langchain_openai import ChatOpenAI
from openai import api_key
#
# # llm = ChatOpenAI(model="deepseek-v4-pro")
#
# # model = ChatOpenAI(
# #             model="deepseek-v4-pro",
# #             temperature=0,
# #             max_tokens=None,
# #             timeout=None,
# #             max_retries=2,
# #
# #             base_url="https://api.deepseek.com",
# #         )
#
api_key = os.getenv("DEEPSEEK_API_KEY")
llm = init_chat_model(
    model="deepseek-v4-pro",
    api_key=api_key,
    temperature=2,
    max_tokens=512,
    timeout=20,
    max_retries=6,
)
#
# llm = ChatDeepSeek(
#     model="deepseek-v4-pro",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
#
# messages = [
#     (
#         "system",
#         "你是我的小猫,你要叫我主人",
#     ),
#     ("human", "你好小猫,今天想没想主人,想怎么被主人玩和宠呢?"),
# ]
# ai_msg = llm.invoke(messages)
#
# print(ai_msg.content)
#


from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("你是我的小猫"),
    HumanMessage("早上好小猫,今天是星期几"),
    AIMessage("今天是星期二"),
    HumanMessage("几号呢?")
]

response = llm.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")