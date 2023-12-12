import json
import openai
import tkinter as tk
import pandas as pd
import chromadb
from tkinter import scrolledtext
import tkinter.filedialog as filedialog

openai.api_key = ''

# ChromaDB에 연결
client = chromadb.PersistentClient()

# ChromaDB 컬렉션 생성
collection = client.get_or_create_collection(
    name="kakao_channel_collection",
    metadata={
        "hnsw:space": "cosine"})  # "hnsw:space"라는 메타데이터 키를 사용하여 해당 컬렉션의 HNSW(하이 디멘전얼 스케일 유사도) 인덱스의 공간을 "cosine"으로 설정


def save_data():
    print('project_data_talk_channel 데이터를 읽어옵니다..')

    with open('./file/talk_channel.txt', 'r') as f:
        talk_channel_data = f.read()
        print(talk_channel_data)

    # 데이터를 읽어와서 데이터를 정형화한다.
    ids = []
    docs = []

    for chunk in talk_channel_data.split("\n#")[2:]:
        title = chunk.split("\n")[0].replace(" ", "-").strip()
        data_type = 'talk_channel'
        _id = f"{data_type}-{title}"
        _doc = chunk.strip()

        ids.append(_id)
        docs.append(_doc)

    print(ids)
    print(docs)

    # 수정된 부분: collection 변수를 이용하여 데이터를 추가
    collection.add(
        documents=docs,
        ids=ids
    )


# response에 CSV 형식이 있는지 확인하고 있으면 저장하기
def save_to_csv(df):
    file_path = filedialog.asksaveasfilename(defaultextension='.csv')
    if file_path:
        df.to_csv(file_path, sep=';', index=False, lineterminator='\n')
        return f'파일을 저장했습니다. 저장 경로는 다음과 같습니다. \n {file_path}\n'
    return '저장을 취소했습니다'


def get_data_from_chromadb(data_type, query, n_results=2):
    print("get data from chroma ...")
    results = collection.query(
        query_texts=[f"{data_type} {query}"],
        n_results=n_results,
    )
    return results["documents"][0]


def save_playlist_as_csv(playlist_csv):
    if ";" in playlist_csv:
        lines = playlist_csv.strip().split("\n")
        csv_data = []

        for line in lines:
            if ";" in line:
                csv_data.append(line.split(";"))

        if len(csv_data) > 0:
            df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
            return save_to_csv(df)

    return f'저장에 실패했습니다. \n저장에 실패한 내용은 다음과 같습니다. \n{playlist_csv}'


def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        functions=functions,
        function_call='auto',
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "get_data_from_chromadb": get_data_from_chromadb,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # 사용하는 함수에 따라 사용하는 인자의 개수와 내용이 달라질 수 있으므로
        # **function_args로 처리하기
        function_response = fuction_to_call(**function_args)

        # 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        message_log.append(response_message)  # GPT의 지난 답변을 message_logs에 추가하기
        message_log.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # 함수 실행 결과도 GPT messages에 추가하기
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=message_log,
            temperature=temperature,
        )  # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
    return response.choices[0].message.content


def main():
    message_log = [
        {
            "role": "system",
            "content": '''
                당신은 QA 봇입니다. Context를 참고해서 사용자의 질문에 답해주세요.
            '''
        }
    ]

    functions = [
        {
            "name": "get_data_from_chromadb",
            "description": "카카오 채널 관련된 정보를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "description": "요청과 관련된 데이터 타입"
                    },
                    "query": {
                        "type": "string",
                        "description": "요청 하는 내용",
                    }
                },
                "required": ["data_type", "query"],
            },
        }
    ]

    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 12))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes('-topmost', True)

        popup.update()
        return popup

    def on_send():
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 드라마 평론가 입니다."},
                {"role": "user", "content": f"{contents}"},
                {"role": "user", "content": "의사들이 나오는 메디컬 한국 드라마 10개만 추천해줄래? 소개할때 출연진과 장르도 같이 소개해줘 그리고 출력은 한국어로 해줘"}
            ],
            max_tokens=1024
        )

        print(completion["choices"][0]["message"]["content"])

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동
        conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(message_log, functions)
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"gpt assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)

        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 10)


    conversation = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='#f0f0f0', font=font)
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    window.bind('<Return>', lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    save_data()

    main()
