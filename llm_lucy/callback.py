import os
import re

import chromadb
from langchain.document_loaders import TextLoader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from dto import ChatbotRequest
import aiohttp
import time
import logging
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다. context를 참고해서 질문에 응답해 주세요."
logger = logging.getLogger("Callback")

db_client = chromadb.PersistentClient()
collection = db_client.get_or_create_collection(
    name="kakao",
    metadata={"hnsw:space": "cosine"}
)
'''
1. 사용할 데이터(project_data_카카오소셜.txt, project_data_카카오싱크.txt, project_data_카카오톡채널.txt)를 preprocessing을 하여 사용한다.
2. 데이터는 VectorDB에 업로드하여 사용한다.
3. Embedding과 VectorDB를 사용하여 데이터 쿼리를 구성한다.
4. LLM과 다양한 모듈을 위해 Langchain 또는 semantic-kernel 둘 중 하나를 사용한다.
5. ChatMemory 기능을 사용하여 history를 가질 수 있게 구성한다.
6. 서비스 형태는 카카오톡을 이용하여 구현한다.
7. 최적화를 위해 외부 application을 이용하여 구현한다..(예: web search 기능)
8. 다양한 prompt engineering 기법을 활용하여 최대한 일관성 있는 대답이 나오도록 유도한다.

이거참고..
https://github.com/kakao-aicoursework/finn.h/tree/main/llm_finn/third_step

쿼리하는거는 내일 예제참고

'''

def save_data_to_vectorDB():
    #data_types = ["project_data_channel", "project_data_social", "project_data_sync"]
    data_types = ["project_data_channel"]

    for data_type in data_types:
        file_path = f"./file/{data_type}.txt"
        documents = TextLoader(file_path).load()

        text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        ids = [f"{data_type}-{i}" for i in range(len(docs))]

        Chroma.from_documents(
            docs,
            OpenAIEmbeddings(),
            collection_name="kakao",
            persist_directory="./chromadb",
            ids=ids,
        )


async def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": request.userRequest.utterance},
        ],
        temperature=0,
    )
    # focus
    output_text = response.choices[0].message.content

    print(output_text)

    # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()


def main():
    message_log = [
        {
            "role": "system",
            "content": '''
                당신은 고객의 질문에 대답하는 챗봇입니다.\
                제안된 답변은 신뢰할 수 있고, 관심 있는 주제에 기반해야 합니다.\
                답변을 위해 다음의 정보를 이용해주세요. \
                고객에게 친절하고 상세한 답변을 제공하는 챗봇이 되어주세요.
                '''
        }
    ]




if __name__ == "__main__":
    # Vector DB로 데이터를 저장
    save_data_to_vectorDB()

    
    main()
