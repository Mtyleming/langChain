########结构化输出
import os
from langchain_deepseek import ChatDeepSeek

api_key = os.getenv("DEEPSEEK_API_KEY")
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    extra_body={"thinking": {"type": "disabled"}}
)

#1.最基础的是自己用提示词 让ai赋予一个结构化的能力



##2.langChian自带的结构化
# ---------  可以用pydantic 系统在内部的时候对参数有校验和类型要求的时候可以用
# from pydantic import BaseModel, Field
#
#
#
#
# class Movie(BaseModel):
#     """A movie with details."""
#     title: str = Field(description="The title of the movie")
#     year: int = Field(description="The year the movie was released")
#     director: str = Field(description="The director of the movie")
#     rating: float = Field(description="The movie's rating out of 10")
#
# model_with_structure = model.with_structured_output(Movie)
# response = model_with_structure.invoke("《绝命毒师》信息")
# print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)

#---------  可以用json 在对外或者接入外部系统的时候 可以用json
import json

json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "director": {
            "type": "string",
            "description": "The director of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The movie's rating out of 10"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # {'title': 'Inception', 'year': 2010, ...}