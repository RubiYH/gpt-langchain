# Reference
# https://github.com/teddylee777/openai-api-kr/blob/main/05-Assistant-API.ipynb
#

import json

def show_json(obj):
    # obj의 모델을 JSON 형태로 변환한 후 출력합니다.
    print(json.loads(obj.model_dump_json()))

# API KEY 정보를 불러옵니다
from dotenv import load_dotenv

load_dotenv()

import os

# os.environ["OPENAI_API_KEY"] = "API KEY를 입력해 주세요"
# OPENAI_API_KEY 를 설정합니다.
api_key = os.environ.get("OPENAI_API_KEY")

from openai import OpenAI

# OpenAI API를 사용하기 위한 클라이언트 객체를 생성합니다.
client = OpenAI(api_key=api_key)

# # 수학 과외 선생님 역할을 하는 챗봇을 생성합니다.
# # 이 챗봇은 간단한 문장이나 한 문장으로 질문에 답변합니다.
# assistant = client.beta.assistants.create(
#     name="Math Tutor",
#     instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
#     model="gpt-4o",
# )
# # 생성된 챗봇의 정보를 JSON 형태로 출력합니다.
# show_json(assistant)

ASSISTANT_ID = "asst_fjUB9Ur23wcISBWq3QLKjwE7"

def create_new_thread():
    # 새로운 스레드를 생성합니다.
    thread = client.beta.threads.create()
    return thread

import time
def submit_message(assistant_id, thread, user_message):
    # 사용자 입력 메시지를 스레드에 추가합니다.
    client.beta.threads.messages.create(
        # Thread ID가 필요합니다.
        # 사용자 입력 메시지 이므로 role은 "user"로 설정합니다.
        # 사용자 입력 메시지를 content에 지정합니다.
        thread_id=thread.id,
        role="user",
        content=user_message,
    )
    # 스레드에 메시지가 입력이 완료되었다면,
    # Assistant ID와 Thread ID를 사용하여 실행을 준비합니다.
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    return run


def wait_on_run(run, thread):
    # 주어진 실행(run)이 완료될 때까지 대기합니다.
    # status 가 "queued" 또는 "in_progress" 인 경우에는 계속 polling 하며 대기합니다.
    while run.status == "queued" or run.status == "in_progress":
        # run.status 를 업데이트합니다.
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        # API 요청 사이에 잠깐의 대기 시간을 두어 서버 부하를 줄입니다.
        time.sleep(0.5)
    return run


def get_response(thread):
    # 스레드에서 메시지 목록을 가져옵니다.
    # 메시지를 오름차순으로 정렬할 수 있습니다. order="asc"로 지정합니다.
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

def print_message(response):
    for res in response:
        print(f"[{res.role.upper()}]\n{res.content[0].text.value}\n")


def ask(assistant_id, thread, user_message):
    run = submit_message(
        assistant_id,
        thread,
        user_message,
    )
    # 실행이 완료될 때까지 대기합니다.
    run = wait_on_run(run, thread)
    print_message(get_response(thread).data[-2:])
    return run
