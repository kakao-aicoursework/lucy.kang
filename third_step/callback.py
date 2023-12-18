import os

from langchain.chains import LLMChain, ConversationChain, MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Dict

from dto import ChatbotRequest
import aiohttp
import time
import logging
import openai

from third_step.db import getDB
from third_step.history import load_conversation_history, log_user_message, log_bot_message, get_chat_history

openai.api_key = os.environ["OPENAI_API_KEY"]

SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다. context를 참고해서 질문에 응답해 주세요."
logger = logging.getLogger("Callback")

CUR_DIR = os.path.dirname(os.path.abspath('./template/'))
db = ""

'''
4. LLM과 다양한 모듈을 위해 Langchain 또는 semantic-kernel 둘 중 하나를 사용한다.
5. ChatMemory 기능을 사용하여 history를 가질 수 있게 구성한다.
6. 서비스 형태는 카카오톡을 이용하여 구현한다.
7. 최적화를 위해 외부 application을 이용하여 구현한다..(예: web search 기능)
8. 다양한 prompt engineering 기법을 활용하여 최대한 일관성 있는 대답이 나오도록 유도한다.

langchain을 활용해서 한개씩 찾도록 수구성하고
websearch하기를 붙이고 
memory기능 추가해서 hisotry기능을 추가한다.  
'''


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

## 체인관련된 부분
def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )


PROMPT_CHANNEL_TEMPLATE = os.path.join(CUR_DIR, "template/channel.txt")
PROMPT_SOCIAL_TEMPLATE = os.path.join(CUR_DIR, "template/social.txt")
PROMPT_SYNC_TEMPLATE = os.path.join(CUR_DIR, "template/sync.txt")
PROMPT_GENERATOR_TEMPLATE = os.path.join(CUR_DIR, "template/generator.txt")
PROMPT_RESPONSE_TEMPLATE = os.path.join(CUR_DIR, "template/response.txt")
INTENT_LIST_TXT = os.path.join(CUR_DIR, "template/parse_intent.txt")

llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

prompt_channel_chain = create_chain(
    llm=llm,
    template_path=PROMPT_CHANNEL_TEMPLATE,
    output_key="channel-step",
)
prompt_social_chain = create_chain(
    llm=llm,
    template_path=PROMPT_SOCIAL_TEMPLATE,
    output_key="social-step",
)
prompt_sync_chain = create_chain(
    llm=llm,
    template_path=PROMPT_SYNC_TEMPLATE,
    output_key="sync-step",
)

prompt_generator_chain = create_chain(
    llm=llm,
    template_path=PROMPT_GENERATOR_TEMPLATE,
    output_key="gen-step",
)

prompt_response_chain = create_chain(
    llm=llm,
    template_path=PROMPT_RESPONSE_TEMPLATE,
    output_key="res-step",
)

parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_LIST_TXT,
    output_key="intent",
)

default_chain = ConversationChain(llm=llm, output_key="text")

destinations = [
    "channel: kakaotalk channel info",
    "social: kakaotalk social info",
    "sync: kakaotalk sync info",
    "question: A specific question about the codebase, product, project, or how to use a feature"
]

destinations = "\n".join(destinations)

router_prompt_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations)
router_prompt = PromptTemplate.from_template(
    template=router_prompt_template, output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt, verbose=True)

multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={
        "channel": prompt_channel_chain,
        "social": prompt_social_chain,
        "sync": prompt_sync_chain,
    },
    default_chain=ConversationChain(llm=llm, output_key="text"))


def generate_answer(user_message) -> Dict[str, str]:
    conversation_id = 1
    INTENT_LIST = os.path.join(CUR_DIR, "template/intent_list.txt")

    context = dict(user_message=user_message, chat_history=get_chat_history(conversation_id))
    query = context["user_message"]
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST)

    intent = parse_intent_chain.run(context)
    print(f"추출된 키워드값은..{intent}입니다")

    if intent in ["sync", "channel", "social"]:

        # Vector DB에 데이터 입력해서 가져오기
        db = getDB(intent)

        print(">>>>>>>>>>>>>>")
        # generate prompt using related documents
        context["related_documents"] = context["user_message"]
        context["target_topic"] = intent
        generator = prompt_generator_chain.run(context)
        print(generator)

        # 관련된 문구들의 문서를 가져오게됨
        docs = db.similarity_search(generator)
        print(docs)

        context["topic"] = intent
        context["content"] = docs
        output = prompt_response_chain.run(context)
    else:
        output = default_chain.run(context)

    # save history
    history_file = load_conversation_history(conversation_id)
    log_user_message(history_file, query)
    log_bot_message(history_file, output)

    return output


if __name__ == "__main__":
    generate_answer("카카오톡 채널 구성이 궁금하네")
