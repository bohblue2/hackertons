from datetime import datetime
from typing import List, Optional
from fastapi import Body, FastAPI, Depends, HTTPException, Path, Query
from yongin_assistant.database.session import SessionLocal
from yongin_assistant.database.models import EpeopleCaseOrm 
from yongin_assistant.schemas import CaseWithAnswer, CategoryCases, EpeopleCase, EpeopleCaseCreate, HealthCheck, SimilarCase, RecommendAnswerResponse, RecommendAnswerRequest, SimilarCaseGroup
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_app() -> FastAPI:
    app = FastAPI()
    return app  

app = get_app()

@app.get(
    "/health", 
    response_model=HealthCheck,
    summary="서버 상태 확인",
    description="서버의 현재 상태와 타임스탬프를 반환합니다.",
    responses={
        200: {
            "description": "서버가 정상적으로 동작 중",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-01T00:00:00"
                    }
                }
            }
        }
    }
)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now()
    }

@app.post(
    "/cases/", 
    response_model=EpeopleCase,
    summary="민원 사례 생성",
    description="새로운 민원 사례를 데이터베이스에 등록합니다.",
    responses={
        201: {
            "description": "민원 사례가 성공적으로 생성됨",
            "content": {
                "application/json": {
                    "example": {
                        "case_id": "CASE123",
                        "title": "도로 보수 요청",
                        "content": "도로에 파손이 심각하여 보수가 필요합니다.",
                        "answer_date": "2024-01-01T00:00:00",
                        "question_date": "2024-01-01T00:00:00",
                        "created_at": "2024-01-01T00:00:00"
                    }
                }
            }
        },
        400: {
            "description": "이미 등록된 민원 사례",
            "content": {
                "application/json": {
                    "example": {"detail": "Case already registered"}
                }
            }
        }
    }
)
def create_case(
    case: EpeopleCaseCreate = Body(
        ...,
        example={
            "case_id": "CASE123",
            "title": "도로 보수 요청",
            "content": "도로에 파손이 심각하여 보수가 필요합니다.",
            "answer_date": "2024-01-01T00:00:00",
            "question_date": "2024-01-01T00:00:00",
            "created_at": "2024-01-01T00:00:00"
        }
    ), 
    db: Session = Depends(get_db)
):
    db_case = db.query(EpeopleCaseOrm).filter(
        EpeopleCaseOrm.case_id == case.case_id
    ).first()
    if db_case:
        raise HTTPException(status_code=400, detail="Case already registered")
    
    db_case = EpeopleCaseOrm(**case.model_dump())
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    return db_case


@app.get(
    "/cases/{case_id}", 
    response_model=EpeopleCase,
    summary="민원 상세 조회",
    description="민원 ID를 통해 민원 상세 정보를 조회합니다.",
    responses={
        200: {
            "description": "민원 상세 정보",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "case_id": "CASE123",
                        "title": "도로 보수 요청",
                        "content": "도로에 파손이 심각하여 보수가 필요합니다.",
                        "department": "도로관리과",
                        "related_laws": "도로법 제31조",
                        "answer_date": "2024-01-01T00:00:00",
                        "question_date": "2024-01-01T00:00:00",
                        "vectorized": False,
                        "created_at": "2024-01-01T00:00:00",
                        "updated_at": "2024-01-01T00:00:00"
                    }
                }
            }
        },
        404: {
            "description": "민원을 찾을 수 없음",
            "content": {
                "application/json": {
                    "example": {"detail": "Case not found"}
                }
            }
        }
    }
)
def read_case(case_id: str, db: Session = Depends(get_db)):
    db_case = db.query(EpeopleCaseOrm).filter(
        EpeopleCaseOrm.case_id == case_id
    ).first()
    if db_case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return db_case

# 민원 리스트 조회
@app.get(
    "/cases/",
    response_model=List[EpeopleCase],
    summary="민원 목록 조회",
    description="민원 목록을 조회합니다. 부서, 기간으로 필터링이 가능하며 페이징을 지원합니다.",
    responses={
        200: {
            "description": "민원 목록",
            "content": {
                "application/json": {
                    "example": [{
                        "id": 1,
                        "case_id": "CASE123",
                        "title": "도로 보수 요청",
                        "content": "도로에 파손이 심각하여 보수가 필요합니다.",
                        "department": "도로관리과",
                        "related_laws": "도로법 제31조",
                        "answer_date": "2024-01-01T00:00:00",
                        "question_date": "2024-01-01T00:00:00",
                        "vectorized": False,
                        "created_at": "2024-01-01T00:00:00",
                        "updated_at": "2024-01-01T00:00:00"
                    }]
                }
            }
        }
    }
)
async def get_cases(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    department: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    query = db.query(EpeopleCaseOrm)
    
    if department:
        query = query.filter(EpeopleCaseOrm.department == department)
    if start_date:
        query = query.filter(EpeopleCaseOrm.question_date >= start_date)
    if end_date:
        query = query.filter(EpeopleCaseOrm.question_date <= end_date)
    
    total = query.count()
    cases = query.order_by(EpeopleCaseOrm.question_date.desc())\
                .offset(skip)\
                .limit(limit)\
                .all()
    
    return cases

@app.post(
    "/cases/{case_id}/request_recommended_answer", 
    response_model=RecommendAnswerResponse,
    summary="민원 답변 추천",
    description="특정 민원에 대한 AI 기반 답변을 생성하여 추천합니다. 답변 생성 시 prompt instruction과 temperature 값을 조정할 수 있습니다.",
    responses={
        200: {
            "description": "생성된 답변 추천",
            "content": {
                "application/json": {
                    "example": {
                        "case_id": "CASE123",
                        "content": "귀하의 민원에 대해 답변 드립니다...",
                        "department": "도로관리과",
                        "created_at": "2024-01-01T00:00:00",
                        "temperature": 0.7
                    }
                }
            }
        },
        404: {
            "description": "민원을 찾을 수 없음",
            "content": {
                "application/json": {
                    "example": {"detail": "Case not found"}
                }
            }
        }
    }
)
async def create_recommended_answer(
    case_id: str = Path(..., description="답변을 생성할 민원의 고유 ID"),
    request: RecommendAnswerRequest = Body(
        ...,
        example={
            "case_id": "CASE123",
            "prompt_instruction": "단정하고 공손한 어조로 답변해주세요",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        description="답변 생성을 위한 파라미터"
    ),
    db: Session = Depends(get_db)
) -> RecommendAnswerResponse:
    case = db.query(EpeopleCaseOrm)\
        .filter(EpeopleCaseOrm.case_id == case_id)\
        .first()
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # TODO: 일단은 mock 데이터를 사용하고 나중에 고도화함
    # llm_service = LLMService()
    # new_answer = llm_service.generate_answer(case.title, case.content, request.prompt_template)
    
    new_answer = "LLM이 생성한 새로운 답변 내용(MOCK DATA 임니다.)"  # 실제 구현 시 LLM 응답으로 대체

    return RecommendAnswerResponse(
        case_id=case_id,
        content=new_answer,
        department=case.department,
        created_at=datetime.now(),
        temperature=request.temperature
    )

# 유사 민원 조회
@app.get(
    "/cases/{case_id}/similar",
    response_model=List[SimilarCase],
    summary="유사 민원 조회",
    description="특정 민원과 유사한 민원들을 조회합니다. 유사도 점수를 기준으로 정렬된 결과를 반환합니다.",
    responses={
        200: {
            "description": "유사 민원 목록",
            "content": {
                "application/json": {
                    "example": [{
                        "case_id": "2024-00001",
                        "title": "도로 보수 요청",
                        "similarity_score": 0.95,
                        "question_date": "2024-01-01T00:00:00",
                        "department": "도로관리과"
                    }]
                }
            }
        },
        404: {
            "description": "요청한 민원을 찾을 수 없음",
            "content": {
                "application/json": {
                    "example": {"detail": "Case not found"}
                }
            }
        }
    }
)
async def get_similar_cases(
    case_id: str = Path(..., description="조회할 민원의 고유 ID"),
    limit: int = Query(5, ge=1, le=20, description="반환할 유사 민원의 최대 개수"),
    min_similarity: float = Query(0.7, ge=0, le=1, description="최소 유사도 점수 (0~1 사이)"),
    db: Session = Depends(get_db)
):
    case = db.query(EpeopleCaseOrm)\
            .filter(EpeopleCaseOrm.case_id == case_id)\
            .first()
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # TODO: 벡터 유사도 계산 및 정렬
    # vector_service = VectorService()
    # NOTE: 아마도 많은 데이터가 있을 경우 persistent-queue 를 사용해서 queuing 하는 것이 좋을 듯
    # similar_cases = vector_service.find_similar(case.content, limit, min_similarity)
    
    # TODO: 실제 구현 시 벡터 유사도 검색으로 대체
    similar_cases = [
        {
            "case_id": f"2024-{i:05d}",
            "title": f"유사 민원 제목 {i}",
            "similarity_score": 0.9 - (i * 0.05),
            "question_date": datetime.now(),
            "department": case.department
        }
        for i in range(limit)
    ]
    
    return similar_cases

@app.get(
    "/categories",
    response_model=List[str],
    summary="민원 카테고리 목록 조회",
    description="등록된 민원의 부서 정보를 기반으로 고유한 카테고리 목록을 반환합니다.",
    responses={
        200: {
            "description": "카테고리 목록",
            "content": {
                "application/json": {
                    "example": ["도로관리과", "교통행정과", "환경과"]
                }
            }
        }
    }
)
async def get_categories(db: Session = Depends(get_db)) -> List[str]:
    categories = db.query(EpeopleCaseOrm.department)\
        .distinct()\
        .filter(EpeopleCaseOrm.department.isnot(None))\
        .order_by(EpeopleCaseOrm.department)\
        .all()
    
    # 튜플 형태의 결과를 리스트로 변환
    return [category[0] for category in categories if category[0]]

@app.get(
    "/cases/categorized",
    response_model=List[CategoryCases],
    summary="카테고리별 민원 그룹 조회",
    description="민원을 카테고리별로 분류하고, 각 카테고리 내에서 유사한 민원끼리 그룹화하여 반환합니다.",
    responses={
        200: {
            "description": "카테고리별 민원 그룹 목록",
            "content": {
                "application/json": {
                    "example": [{
                        "category_name": "도로",
                        "similar_case_groups": [{
                            "cases": [{
                                "case": {
                                    "case_id": "CASE123",
                                    "title": "도로 보수 요청",
                                    "content": "도로에 파손이 심각하여 보수가 필요합니다."
                                },
                                "recommended_answer": "귀하의 도로 보수 요청 민원에 답변드립니다..."
                            }]
                        }]
                    }]
                }
            }
        }
    }
)
async def get_categorized_cases(
    db: Session = Depends(get_db)
) -> List[CategoryCases]:
    categories = await get_categories(db)
    
    result = []
    for category in categories:
        cases = db.query(EpeopleCaseOrm)\
            .filter(EpeopleCaseOrm.department.like(f"%{category}%"))\
            .all()
        
        # TODO: 실제 구현시에는 벡터 유사도 기반으로 그룹화해야 함, 임시로 2개씩 그룹화
        similar_groups = []
        for i in range(0, len(cases), 2):
            group_cases = []
            for case in cases[i:i+2]:
                # TODO: 실제 구현시에는 LLM으로 추천 답변 생성
                mock_answer = f"이것은 {case.case_id}에 대한 추천 답변입니다."
                group_cases.append(CaseWithAnswer(
                    case=case,
                    recommended_answer=mock_answer
                ))
            if group_cases:
                similar_groups.append(SimilarCaseGroup(cases=group_cases))
        
        result.append(CategoryCases(
            category_name=category,
            similar_case_groups=similar_groups
        ))
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)