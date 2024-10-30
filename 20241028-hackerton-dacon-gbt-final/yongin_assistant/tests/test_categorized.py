from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from yongin_assistant.database.base import Base
from yongin_assistant.database.models import EpeopleCaseOrm
from main import app, get_db
from yongin_assistant.schemas import CategoryCases, SimilarCaseGroup

# In-memory SQLite 데이터베이스 설정
SQLALCHEMY_DATABASE_URL = "sqlite://"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 테스트용 데이터베이스 의존성 오버라이드
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    # 테스트 전에 테이블 생성
    Base.metadata.create_all(bind=engine)
    
    # 테스트 데이터 생성
    db = TestingSessionLocal()
    test_cases = [
        EpeopleCaseOrm(
            case_id="CASE001",
            title="도로 보수 요청",
            content="도로가 파손되어 보수가 필요합니다.",
            department="도로관리과",
            question_date=datetime.now(),
            answer_date=datetime.now()
        ),
        EpeopleCaseOrm(
            case_id="CASE002",
            title="신호등 설치 요청",
            content="교차로에 신호등 설치가 필요합니다.",
            department="교통행정과",
            question_date=datetime.now(),
            answer_date=datetime.now()
        ),
    ]
    db.add_all(test_cases)
    db.commit()
    
    yield
    
    # 테스트 후 테이블 삭제
    Base.metadata.drop_all(bind=engine)

def test_get_categories():
    response = client.get("/categories")
    assert response.status_code == 200
    categories = response.json()
    
    # 카테고리 목록 검증
    assert isinstance(categories, list)
    assert "도로관리과" in categories
    assert "교통행정과" in categories

def test_get_categorized_cases():
    response = client.get("/categorized")
    assert response.status_code == 200
    data = response.json()
    
    # 응답 구조 검증
    assert isinstance(data, list)
    assert len(data) > 0
    
    # 첫 번째 카테고리 검증
    first_category = data[0]
    assert isinstance(first_category, dict)
    assert "category_name" in first_category
    assert "similar_case_groups" in first_category
    
    # 유사 케이스 그룹 검증
    groups = first_category["similar_case_groups"]
    assert isinstance(groups, list)
    if len(groups) > 0:
        first_group = groups[0]
        assert "cases" in first_group
        assert isinstance(first_group["cases"], list)
        
        # 개별 케이스 검증
        if len(first_group["cases"]) > 0:
            first_case = first_group["cases"][0]
            assert "case" in first_case
            assert "recommended_answer" in first_case

def test_get_cases_with_categories_from_db():
    categories = client.get("/categories").json()
    # print(categories.json())
    db = TestingSessionLocal()
    # categories = db.query(EpeopleCaseOrm.department)\
    #     .distinct()\
    #     .filter(EpeopleCaseOrm.department.isnot(None))\
    #     .order_by(EpeopleCaseOrm.department)\
    #     .all()
    
    # categories = [category[0] for category in categories if category[0] is not None]

    for category in categories:
        cases = db.query(EpeopleCaseOrm)\
            .filter(EpeopleCaseOrm.department == category)\
            .all()

        assert len(cases) > 0
        for case in cases:
            assert case.department == category

# def test_get_categories_empty_db():
#     # DB 비우기
#     db = TestingSessionLocal()
#     db.query(EpeopleCaseOrm).delete()
#     db.commit()
    
#     response = client.get("/categories")
#     assert response.status_code == 200
#     categories = response.json()
    
#     # mock 데이터 검증
#     assert isinstance(categories, list)
#     assert len(categories) > 0
#     assert "도로관리과" in categories

# def test_get_categorized_cases_empty_db():
#     # DB 비우기
#     db = TestingSessionLocal()
#     db.query(EpeopleCaseOrm).delete()
#     db.commit()
    
#     response = client.get("/cases/categorized")
#     assert response.status_code == 200
#     data = response.json()
    
#     # mock 데이터 검증
#     assert isinstance(data, list)
#     assert len(data) > 0
    
#     # mock 데이터 구조 검증
#     first_category = data[0]
#     assert "category_name" in first_category
#     assert "similar_case_groups" in first_category
#     assert len(first_category["similar_case_groups"]) > 0

# def test_category_structure():
#     response = client.get("/cases/categorized")
#     data = response.json()
    
#     for category in data:
#         # 카테고리 구조 검증
#         assert "category_name" in category
#         assert "similar_case_groups" in category
        
#         # 유사 케이스 그룹 구조 검증
#         for group in category["similar_case_groups"]:
#             assert "cases" in group
            
#             # 개별 케이스 구조 검증
#             for case_with_answer in group["cases"]:
#                 assert "case" in case_with_answer
#                 assert "recommended_answer" in case_with_answer
                
#                 case = case_with_answer["case"]
#                 assert "case_id" in case
#                 assert "title" in case
#                 assert "content" in case
#                 assert "department" in case