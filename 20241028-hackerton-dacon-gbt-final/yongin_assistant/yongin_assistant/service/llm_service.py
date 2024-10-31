
import os
from typing import Optional

from yongin_assistant.vectorstore import get_collection, get_client
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Write the answer in korean.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
from langchain_milvus import Milvus, Zilliz


# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class LlmService:
    def __init__(self):
        client = get_client()
        collection = get_collection()
        milvus_store = Zilliz(
            embedding_function = OpenAIEmbeddings(model="text-embedding-3-large"),
            collection_name = collection.name,
            connection_args = {
                "uri": os.getenv("MILVUS_URI"),
                "token": os.getenv("MILVUS_API_KEY")
            },
            primary_field="case_id",
            vector_field="content_embedding",
            text_field="content",
            drop_old=False
        )
        retriever = milvus_store.as_retriever()
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        self._rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def generate_answer(self, title: str, content: str, prompt_template: Optional[str] = None) -> str:
        return self._rag_chain.invoke(f"제목: {title}, 내용: {content} 에 대한 답변을 작성해주세요.")
    
    def close(self): ... 

if __name__ == "__main__":
    service = LlmService()
    print(service.generate_answer("교통신호등 설치 요청", "교통신호등 설치 요청합니다."))