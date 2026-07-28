import base64
from io import BytesIO
from PIL import Image
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama



llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0
)

# messages = [
#     (
#         "system",
#         "你是一个中英文翻译助手，请将用户句子翻译成中文。",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg.text)



def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str



file_path = "C:\\Users\\Administrator\\Desktop\\1c8b-888d0fdb29259acdeba83191cb08c79e.jpg"
pil_image = Image.open(file_path)

image_b64 = convert_to_base64(pil_image)

def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


from langchain_core.output_parsers import StrOutputParser

chain = prompt_func | llm | StrOutputParser()

query_chain = chain.invoke(
    {"text": "这个图片中是有什么角色什么动漫角色 ", "image": image_b64}
)

print(query_chain)