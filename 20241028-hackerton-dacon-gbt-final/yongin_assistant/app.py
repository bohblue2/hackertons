# app.py
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
import os
from typing import Dict, List

# 환경 변수 설정
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]

# Pinecone 초기화
embeddings = OpenAIEmbeddings()

# 저장된 데이터를 로드
from langchain.vectorstores.faiss import FAISS

from constant import DEFAULT_FAISS_FOLDER_PATH, DEFAULT_FAISS_INDEX_NAME
vectorstore = FAISS.load_local(
    folder_path=DEFAULT_FAISS_FOLDER_PATH,
    index_name=DEFAULT_FAISS_INDEX_NAME,
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)
# 검색을 위한 retriever 생성
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 프롬프트 템플릿 정의
template = """당신은 용인시의 행정 정보와 조례를 알려주는 AI 도우미입니다.
주어진 context를 바탕으로 사용자의 질문에 친절하게 답변해주세요.

Context:
{context}

Question:
{question}

Answer in Korean:"""

prompt = ChatPromptTemplate.from_template(template)

# RAG 체인 구성
model = ChatOpenAI(temperature=0.3)
output_parser = StrOutputParser()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

# FastAPI 앱 생성
app = FastAPI(
    title="용인시 정보 검색 서비스",
    description="용인시의 행정 정보와 조례를 쉽게 검색할 수 있는 서비스입니다.",
    version="1.0.0",
)

# Langserve 라우트 추가
add_routes(
    app,
    chain,
    path="/yongin-info",
)

# 건강 체크 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 데이터 수집 및 벡터 DB 업데이트를 위한 함수
def update_vector_db(texts: List[Dict[str, str]]):
    """
    새로운 텍스트를 벡터 DB에 추가하는 함수
    
    Args:
        texts (List[Dict[str, str]]): 문서 텍스트와 메타데이터를 포함하는 딕셔너리 리스트
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    documents = []
    for text_dict in texts:
        chunks = text_splitter.split_text(text_dict["text"])
        for chunk in chunks:
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": text_dict.get("source", "unknown"),
                    "date": text_dict.get("date", "unknown"),
                    "category": text_dict.get("category", "unknown")
                }
            })
    
    # Pinecone에 문서 추가
    vectorstore.add_texts(
        texts=[doc["text"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)