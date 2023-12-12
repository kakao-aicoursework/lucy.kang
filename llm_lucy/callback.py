import os

import chromadb

from dto import ChatbotRequest
from samples import list_card
import aiohttp
import time
import logging
import openai

# 환경 변수 처리 필요!
openai.api_key = os.environ["API_KEY"]
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다. context를 참고해서 질문에 응답해 주세요."
logger = logging.getLogger("Callback")

# ChromaDB에 연결
client = chromadb.PersistentClient()

# ChromaDB 컬렉션 생성
collection = client.get_or_create_collection(
    name="kakao_channel_collection",
    metadata={
        "hnsw:space": "cosine"})  # "hnsw:space"라는 메타데이터 키를 사용하여 해당 컬렉션의 HNSW(하이 디멘전얼 스케일 유사도) 인덱스의 공간을 "cosine"으로 설정

def save_data():
    print('read sync data')

    with open('./project_data_sync.txt', 'r') as f:
        sync_data = f.read()
        print(sync_data)

    # 데이터를 읽어와서 데이터를 정형화한다.
    ids = []
    docs = []

    for chunk in sync_data.split("\n#")[2:]:
        title = chunk.split("\n")[0].replace(" ", "-").strip()
        data_type = 'talk_channel'
        _id = f"{data_type}-{title}"
        _doc = chunk.strip()

        ids.append(_id)
        docs.append(_doc)

    print(docs)
    print(ids)

    # 수정된 부분: collection 변수를 이용하여 데이터를 추가
    collection.add(
        documents=docs,
        ids=ids,
    )

    print('여기를 지나면 성공한 것입니다.')

async def callback_handler(request: ChatbotRequest) -> dict:
    print('callback 실행')
    save_data()

    print('callback 종료')



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