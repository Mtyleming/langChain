import pprint

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


agent = create_agent(model="deepseek:deepseek-v4-pro", tools=[search])

res = agent.invoke({"messages": HumanMessage("重庆天气如何")})
pprint.pprint(res)

