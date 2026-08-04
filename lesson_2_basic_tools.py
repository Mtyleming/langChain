import os
from langchain.chat_models import init_chat_model


api_key = os.getenv("DEEPSEEK_API_KEY")
model = init_chat_model(
    model="deepseek-v4-pro",
    api_key=api_key,
    temperature=2,
    max_tokens=512,
    timeout=20,
    max_retries=6,
)
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = model.bind_tools([get_weather])
#
# response = model_with_tools.invoke("What's the weather like in Boston?")
# for tool_call in response.tool_calls:
#     # View tool calls made by the model
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")
#
# Bind (potentially multiple) tools to the model
# model_with_tools = model.bind_tools([get_weather])
#
# # Step 1: Model generates tool calls
# messages = [{"role": "user", "content": "What's the weather in Boston?"}]
# ai_msg = model_with_tools.invoke(messages)
# print(ai_msg)
# messages.append(ai_msg)
#
# # Step 2: Execute tools and collect results
# for tool_call in ai_msg.tool_calls:
#     # Execute the tool with the generated arguments
#     tool_result = get_weather.invoke(tool_call)
#     messages.append(tool_result)
#
# # Step 3: Pass results back to model for final response
# final_response = model_with_tools.invoke(messages)
# print(final_response)
# print(final_response.text)
# # "The current weather in Boston is 72°F and sunny."


# --------------------------------------强制调用------------
# any是指模型强制一定要调用某个工具
#model_with_tools = model.bind_tools([get_weather], tool_choice="any")

# 写死调用某个工具
#model_with_tools = model.bind_tools([get_weather], tool_choice="get_weather")


#----------------------------------------并行调用------------
#提示词里面 模型推理出可能多次调用工具
response = model_with_tools.invoke(
    "What's the weather in Boston and Tokyo?"
)


# The model may generate multiple tool calls
print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]


# Execute all tools (can be done in parallel with async)
# results = []
# for tool_call in response.tool_calls:
#     if tool_call['name'] == 'get_weather':
#         result = get_weather.invoke(tool_call)
#     ...
#     results.append(result)