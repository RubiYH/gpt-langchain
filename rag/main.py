# Reference
# https://teddylee777.github.io/langchain/rag-tutorial/#txt-%ED%8C%8C%EC%9D%BC
#

from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

import os
os.environ["LANGCHAIN_PROJECT"] = "RAG TUTORIAL"

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import nltk
nltk.download('punkt')

# 단계 1: 문서 로드(Load Documents)
loader = UnstructuredFileLoader(["./data/summary.txt"])
docs = loader.load()

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

splits = text_splitter.split_documents(docs)

# 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 단계 4: 검색(Search)
# 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 5: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = hub.pull("rlm/rag-prompt")

# 단계 6: 언어모델 생성(Create LLM)
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=3000)

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 단계 7: 체인 생성(Create Chain)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 단계 8: 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "문서를 요약해주세요."
response = rag_chain.invoke(question)

# 결과 출력
print(f"문서의 수: {len(docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")