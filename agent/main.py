# Reference
# https://teddylee777.github.io/langchain/langchain-agent/#-%EB%A9%94%EB%AA%A8%EB%A6%AC-%EC%B6%94%EA%B0%80%ED%95%98%EA%B8%B0
#

# 필요한 모듈 import
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# API KEY 정보를 불러옵니다
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

########## 1. 도구를 정의합니다 ##########

### 1-1. Search 도구 ###
# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=5은 검색 결과를 5개까지 가져오겠다는 의미입니다
search = TavilySearchResults(k=5)

### 1-2. PDF 문서 검색 도구 (Retriever) ###
# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("data/SPRi_AI_Brief_5월호_산업동향_최종.pdf")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할합니다.
split_docs = loader.load_and_split(text_splitter)

# VectorStore를 생성합니다.
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성합니다.
retriever = vector.as_retriever()

# langchain 패키지의 tools 모듈에서 retriever 도구를 생성
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",
    # 도구에 대한 설명을 자세히 기입해야 합니다!!!
    description="2024년 5월 AI 관련 정보를 PDF 문서에서 검색합니다. '2024년 5월 AI 산업동향' 과 관련된 질문은 이 도구를 사용해야 합니다!",
)

### 1-3. tools 리스트에 도구 목록을 추가합니다 ###
# tools 리스트에 search와 retriever_tool을 추가합니다.
tools = [search, retriever_tool]

########## 2. LLM 을 정의합니다 ##########
# LLM 모델을 생성합니다.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

########## 3. Prompt 를 정의합니다 ##########

# hub에서 prompt를 가져옵니다 - 이 부분을 수정할 수 있습니다!
prompt = hub.pull("hwchase17/openai-functions-agent")

########## 4. Agent 를 정의합니다 ##########

# OpenAI 함수 기반 에이전트를 생성합니다.
# llm, tools, prompt를 인자로 사용합니다.
agent = create_openai_functions_agent(llm, tools, prompt)

########## 5. AgentExecutor 를 정의합니다 ##########

# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정합니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

########## 6. 채팅 기록을 수행하는 메모리를 추가합니다. ##########

# 채팅 메시지 기록을 관리하는 객체를 생성합니다.
message_history = ChatMessageHistory()

# 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 대부분의 실제 시나리오에서 세션 ID가 필요하기 때문에 이것이 필요합니다
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    lambda session_id: message_history,
    # 프롬프트의 질문이 입력되는 key: "input"
    input_messages_key="input",
    # 프롬프트의 메시지가 입력되는 key: "chat_history"
    history_messages_key="chat_history",
)

########## 7. 질의-응답 테스트를 수행합니다. ##########

# 질의에 대한 답변을 출력합니다.
response = agent_with_chat_history.invoke(
    {
        "input": "엔비디아의 Blackwell에 대한 내용을 PDF 문서에서 알려줘"
    },
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "MyTestSessionID"}},
)
print(f"답변: {response['output']}")

response = agent_with_chat_history.invoke(
    {"input": "Hi! I'm Teddy. Glad to meet you."},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "MyTestSessionID"}},
)
print(f"답변: {response['output']}")

response = agent_with_chat_history.invoke(
    {"input": "What's my name?"},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "MyTestSessionID"}},
)
print(f"답변: {response['output']}")

response = agent_with_chat_history.invoke(
    {"input": "판교 카카오 프렌즈샵 아지트점의 전화번호를 검색하여 결과를 알려주세요."},
    # 세션 ID를 설정합니다.
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
    config={"configurable": {"session_id": "MyTestSessionID"}},
)
print(f"답변: {response['output']}")