from dto import ChatbotRequest
from samples import list_card
import aiohttp
import time
import logging
import openai

# 환경 변수 처리 필요!
openai.api_key = ''
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

def query_by_langchain(docs, query) -> str:

    prompt_template = """
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer in Korean:
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['context','question']
    )
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT, verbose=True)
    result = chain.run(input_documents=docs, question=query)

    return result
    # return chain.run(input_documents=docs, question=query, return_only_outputs=True)


async def callback_handler(request: ChatbotRequest, docs) -> dict:
    # client = OpenAI(
    #     # This is the default and can be omitted
    #     api_key=os.environ.get("OPENAI_API_KEY"),
    # )

    print(request)
    query = request.userRequest.utterance
    related_docs = vector_db.get_relevant_documents(docs, 3, query)
    output_text = query_by_langchain(related_docs, query)
    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "user", "content": request.userRequest.utterance},
    #         {"role": "assistant", "content": SYSTEM_MSG},
    #     ]
    # )
    # output_text = completion.choices[0].message.content
    print(output_text)

    url = request.userRequest.callbackUrl

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

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload) as resp:
                print(resp.json())
                await resp.json()

# def callback_handler2(request: ChatbotRequest):
#     url = request.userRequest.callbackUrl
#     ic(url)
#     payload = {
#         "version": "2.0",
#         "template": {
#             "outputs": [
#                 {
#                     "simpleText": {
#                         "text": "콜백 응답~"
#                     }
#                 }
#             ]
#         }
#     }

#     try:
#         headers = { 'Content-Type': 'application/json' }
#         res = requests.post(url, headers=headers, data=json.dumps(payload))
#         ic(res.text)
#     except Exception as e:
#         ic(e)