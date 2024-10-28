from pymilvus import MilvusClient
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

client = MilvusClient("datasets/milvus_demo.db")

# OpenAI 클라이언트 초기화
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# 프롬프트 템플릿 정의
prompt_template = ChatPromptTemplate.from_template("""
다음은 민원에 대한 답변 내용입니다. 이 답변을 바탕으로 원래 민원인이 어떤 내용의 민원을 제기했는지 추론해주세요.
답변은 간단명료하게 작성해주세요.

답변 내용:
{content}

추론된 민원 내용:
""")

import pandas as pd 
import sqlite3
conn = sqlite3.connect('datasets/yongin.db')
df = pd.read_sql_query("SELECT * FROM epeople_cases", conn)
for row in df.itertuples():
    messages = prompt_template.format_messages(content=row.content)
    inferred_complaint = llm.invoke(messages).content
    print(f"원본 답변: {row.content[:50]}...")
    print(f"추론된 민원 내용: {inferred_complaint}\n")
conn.close()
