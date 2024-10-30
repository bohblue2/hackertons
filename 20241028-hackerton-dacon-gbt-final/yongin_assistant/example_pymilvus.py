import os
import time
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, utility
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from constant import DEFAULT_EMBEDDING_DIM, MILVUS_METRIC_TYPE, MILVUS_NLIST, MILVUS_NPROBE, MILVUS_TOP_K, MILVUS_INDEX_TYPE

client = MilvusClient(uri="https://in03-d3fcec5282ea165.serverless.gcp-us-west1.cloud.zilliz.com", token=os.getenv("MILVUS_API_KEY"))
if client.has_collection("epeople_cases"):
    client.drop_collection("epeople_cases")
from pymilvus import connections
connections.connect(uri="https://in03-d3fcec5282ea165.serverless.gcp-us-west1.cloud.zilliz.com", token=os.getenv("MILVUS_API_KEY"))
print(client.list_collections())

case_id = FieldSchema(
    name="case_id", 
    dtype=DataType.INT64, 
    description="case_id", 
    is_primary=True
)

content = FieldSchema(
    name="content", 
    dtype=DataType.VARCHAR, 
    description="content",
    max_length=5000,
)

content_embedding = FieldSchema(
    name="content_embedding", 
    dtype=DataType.FLOAT_VECTOR, 
    description="content_embedding",
    dim=DEFAULT_EMBEDDING_DIM
)

question_datetime = FieldSchema(
    name="question_datetime", 
    dtype=DataType.VARCHAR, 
    description="question_datetime",
    max_length=30,
)

answer_datetime = FieldSchema(
    name="answer_datetime", 
    dtype=DataType.VARCHAR, 
    description="answer_datetime",
    max_length=30,
)

schema = CollectionSchema(fields=[
    case_id, 
    content, 
    content_embedding, 
    question_datetime, 
    answer_datetime
])

collection = Collection(
    name="epeople_cases", 
    schema=schema
)

index = collection.create_index(
    field_name="content_embedding", 
    index_params={
        "metric_type": MILVUS_METRIC_TYPE,
        "index_type": MILVUS_INDEX_TYPE,
        "params": { "nlist": MILVUS_NLIST }
    }
)
utility.wait_for_index_building_complete(collection.name, index_name="content_embedding")

# OpenAI 클라이언트 초기화
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
)

# 프롬프트 템플릿 정의
prompt_template = ChatPromptTemplate.from_template("""
다음은 민원에 대한 답변 내용입니다. 이 답변을 바탕으로 원래 민원인이 어떤 내용의 민원을 제기했는지 추론해주세요.
답변은 간단명료하게 작성해주세요.

추론된 민원 내용을 작성할때 제약사항:
- "민원인은 ~" 이라고 시작하지 말아주세요.
- 모든 문장은 ~입니다 로 끝내야 합니다.

답변 내용:
{content}

추론된 민원 내용:
""")

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

import pandas as pd 
import sqlite3
conn = sqlite3.connect('datasets/yongin.db')
df = pd.read_sql_query("SELECT * FROM epeople_cases", conn)
for row in df.itertuples():
    messages = prompt_template.format_messages(content=row.content)
    inferred_complaint = llm.invoke(messages).content
    print(f"원본 답변: {row.content[:50]}...")
    print(f"추론된 민원 내용: {inferred_complaint}\n")
    collection.insert([
        {
            "case_id": int(row.case_id),
            "content": row.content,
            "content_embedding": embedding.embed_query(row.content),
            "question_datetime": row.question_date,
            "answer_datetime": row.answer_date
        }
    ])
    


conn.close()
