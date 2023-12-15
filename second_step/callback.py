import os

import chromadb
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from dto import ChatbotRequest
import aiohttp
import time
import logging
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
logger = logging.getLogger("Callback")

db_client = chromadb.PersistentClient()
collection = db_client.get_or_create_collection(
    name="kakao",
    metadata={"hnsw:space": "cosine"}
)

llm = ChatOpenAI(temperature=0.8)

'''
<문제> 

1. 제공된 데이터(project_data_카카오싱크.txt)는 LLM library에 이용한다.
2. prompt engineering 기법을 이용하여 chatGPT api에 해당 데이터를 적용한다. 
3. LLM + prompt engineering은 LangChain library를 이용하여 구성한다.
4. 여러 반복을 통해 카카오싱크 api 사용법을 잘 설명 해주는 챗봇을 완성한다.

todo. 여러반복을 통한다는것은 무슨의미? 

'''

def save_data_to_vectorDB():
    data_type = "project_data_sync"
    data = get_data(data_type)

    ids = []
    docs = []

    for chunk in data.split("\n#")[2:]:
        title = chunk.split("\n")[0].replace(" ", "-").strip()
        _id = f"{data_type}-{title}"
        _doc = chunk.strip()

        ids.append(_id)
        docs.append(_doc)

    collection.add(
        documents=docs,
        ids=ids,
    )


def get_data(data_type):
    with open(f"file/{data_type}.txt", "r") as fin:
        return fin.read()

async def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================

    SYSTEM_MSG = "You are Provider for kakao corperation. Answer the question with context. response for korean"

    test = collection.query(
        query_texts=["what is kakao Sync's Procedure"],
        n_results=3,
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"{test}"},
            {"role": "user", "content": request.userRequest.utterance}
        ],
        temperature=0,
    )

    # focus
    output_text = response.choices[0].message.content

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


def ask_langchain():
    system_message = ("You are Provider for kakao corperation. "
                      "Answer the question with context. "
                      "response for korean")
    system_message_prompt = SystemMessage(content=system_message)
    #print(system_message_prompt)

    human_template = ("what is kakao Sync's Procedure.")
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    #print(human_message_prompt)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    #print(chat_prompt)

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    return chain

if __name__ == "__main__":
    # Vector DB로 데이터를 저장
    #save_data_to_vectorDB()

    # vector DB에 질의하는 방법
    #query_result = collection.query(query_texts=["기능"], n_results=3)
    #chat_source = query_result['documents'][0]
    #print(chat_source)

    # chatGPT api에 요청하기
    template= ask_langchain()
    result = template.run(text=collection)
    print(result)
