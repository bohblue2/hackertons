import datetime

from sqlalchemy import Column, Integer, String, Boolean, DateTime, LargeBinary
from yongin_assistant.database.base import Base  

class EpeopleCaseOrm(Base):
    __tablename__ = 'epeople_cases'

    id = Column(Integer, primary_key=True, index=True)
    # 민원 ID
    case_id = Column(String, unique=True)
    # 질문 제목
    title = Column(String, nullable=False)
    # 질문 날짜
    question_date = Column(DateTime, nullable=False)
    # 답변 내용
    content = Column(String, nullable=True)
    # 담당부서(예: 국토교통부 공공주택본부 공공주택추진단 도심주택공급협력과)
    department = Column(String, nullable=True)
    # 관련법령(예: 빈집 및 소규모주택 정비에 관한 특례법 / 제49조(임대주택 건설에 따른 특례))
    related_laws = Column(String, nullable=True)
    # 답변 날짜
    answer_date = Column(DateTime, nullable=False)
    # 백터화 여부 
    vectorized = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.datetime.now)
    updated_at = Column(DateTime, default=datetime.datetime.now)