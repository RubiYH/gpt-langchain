# Reference
# https://github.com/teddylee777/openai-api-kr/blob/main/05-Assistant-API.ipynb
#
# Migration to V2
# https://platform.openai.com/docs/assistants/migration/what-has-changed
# https://platform.openai.com/docs/assistants/tools/file-search/quickstart
#

import json

from main import wait_on_run, ask, create_new_thread, print_message, get_response


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

# Create a vector store called "Financial Statements"
vector_store = client.beta.vector_stores.create(name="Quiz Generator")

# Ready the files for upload to OpenAI
file_paths = [
    "data/language_models_are_unsupervised_multitask_learners.pdf",
    "data/SPRi_AI_Brief_5월호_산업동향_최종.pdf",
]
file_streams = [open(path, "rb") for path in file_paths]

# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
)

# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

# 스키마를 정의합니다.
function_schema = {
    "name": "generate_quiz",
    "description": "Generate a quiz to the student, and returns the student's response. A single quiz has multiple questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "questions": {
                "type": "array",
                "description": "An array of questions, each with a title and multiple choice options.",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_text": {"type": "string"},
                        "choices": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["question_text", "choices"],
                },
            },
        },
        "required": ["title", "questions"],
    },
}

# 퀴즈를 출제하는 역할을 하는 챗봇을 생성합니다.
assistant = client.beta.assistants.create(
    name="Quiz Generator",
    instructions="You are an expert in generating multiple choice quizzes. Create quizzes based on uploaded files.",
    model="gpt-4o",
    tools=[
        {"type": "file_search"},
        {"type": "function", "function": function_schema},
    ],
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
)

# ASSISTANT_ID = "asst_ZN61sfxB5q2XlCewEQdevYaj"

# 생성된 챗봇의 정보를 JSON 형태로 출력합니다.
show_json(assistant)
ASSISTANT_ID = assistant.id

# 새로운 스레드를 생성한 뒤 진행합니다.
thread = create_new_thread()

# 퀴즈를 만들도록 요청합니다.
run = ask(
    ASSISTANT_ID,
    thread,
    # 객관식 퀴즈에 대한 구체적인 지시사항을 기입합니다.
    "3개의 객관식 퀴즈(multiple choice questions)를 만들어 주세요. "
    "객관식 퀴즈의 선택지에 번호를 표기해주세요. 1~4까지 숫자로 시작하여야 합니다. "
    "퀴즈는 내가 업로드한 파일에 관한 내용이어야 합니다. "
    "내가 제출한 responses에 대한 피드백을 주세요. "
    "내가 기입한 답, 정답, 제출한 답이 오답이라면 오답에 대한 피드백을 모두 포함해야 합니다. "
    "모든 내용은 한글로 작성해 주세요. ",
)

# 퀴즈를 사용자에게 표시하는 함수를 정의합니다.
def display_quiz(title, questions, show_numeric=False):
    print(f"제목: {title}\n")
    responses = []

    for q in questions:
        # 질문을 출력합니다.
        print(q["question_text"])
        response = ""

        # 각 선택지를 출력합니다.
        for i, choice in enumerate(q["choices"]):
            if show_numeric:
                print(f"{i+1} {choice}")
            else:
                print(f"{choice}")

        response = input("정답을 선택해 주세요: ")
        responses.append(response)
        print()

    return responses


# requires_action 상태는 사용자의 응답 제출해야 합니다.
# 제출이 완료될 때까지 Assistant 는 최종 답변을 대기합니다.
# 늦게 제출시 만료(expired) 상태가 될 수 있습니다.
if run.status == "requires_action":
    # 단일 도구 호출 추출
    tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    responses = display_quiz(arguments["title"], arguments["questions"])
    # 퀴즈를 표시하고 사용자의 응답을 반환합니다.
    print("기입한 답(순서대로)")
    print(responses)

    # 사용자 답변을 제출하기 위한 Run 을 생성합니다.
    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=[
            {
                "tool_call_id": tool_call.id,  # 대기중인 tool_call 의 ID
                "output": json.dumps(responses),  # 사용자 답변
            }
        ],
    )

    # 스레드에서 실행을 기다립니다.
    run = wait_on_run(run, thread)

    # 실행이 완료되면, 실행의 상태를 출력합니다.
    if run.status == "completed":
        print("퀴즈를 제출했습니다.")
        # 전체 대화내용 출력
        print_message(get_response(thread).data[-2:])
