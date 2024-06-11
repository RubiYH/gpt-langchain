# Reference
# https://teddylee777.github.io/langchain/rag-tutorial/#txt-%ED%8C%8C%EC%9D%BC
# https://www.youtube.com/watch?v=J2AsmUODBak&list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v&index=13
# https://colab.research.google.com/drive/1PALBOJ-vXgKe3LOIbKoqeD8kuZYx_Lnr#scrollTo=VpJhZWThn7gG

from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain.prompts import PromptTemplate

import nltk

nltk.download('punkt')

# 단계 1: 문서 로드(Load Documents)
# loaders = [
#     PyPDFLoader("./data/[복지이슈 FOCUS 15ȣ] 경기도 극저신용대출심사모형 개발을 위한 국내 신용정보 활용가능성 탐색.pdf"),
#     PyPDFLoader("./data/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf"),
# ]

loader = YoutubeLoader.from_youtube_url("YOUTUBE-VIDEO-URL", language=["en"], translation="en")

docs = []
docs.extend(loader.load_and_split())

# 단계 2: 문서 분할(Split Documents)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
#
# splits = text_splitter.split_documents(docs)

# Parent Document Retriever
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = child_splitter.split_documents(docs)

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 2

# 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Parent splitter
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

# 단계 4: 검색(Search)
# 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
# retriever = vectorstore.as_retriever()

# Parent Document Retriever
store = InMemoryStore()
parent_doc_retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter,
                                               parent_splitter=parent_splitter)

# Retriever에 저장
parent_doc_retriever.add_documents(docs, ids=None)

# initialize the ensemble retriever
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever, parent_doc_retriever], weights=[0.2, 0.3, 0.5]
)

# Long Context Reorder
docs = ensemble_retriever.invoke("Summarize the content.")
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)
# print(reordered_docs)

# 단계 5: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
# prompt = hub.pull("rlm/rag-prompt")
prompt = PromptTemplate.from_template(
    """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the 
    question. If you don't know the answer, just say that you don't know.
    Use maximum 10 setences for the answer.

    Question:
    {question}
    Context:
    {context}
    """
)

# 단계 6: 언어모델 생성(Create LLM)
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=3000)


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 단계 7: 체인 생성(Create Chain)
rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# 단계 8: 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "내용을 요악해줘."
response = rag_chain.invoke(question)

# 결과 출력
print(f"문서의 수: {len(reordered_docs)}")
print("===" * 20)
print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")