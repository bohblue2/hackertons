from fastapi import Body, FastAPI, Depends, HTTPException, Query
from yongin_assistant.database.session import SessionLocal
from yongin_assistant.database.models import EpeopleCaseOrm 
from yongin_assistant.schemas import EpeopleCase, EpeopleCaseCreate 
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


@app.get("/cases/{case_id}", response_model=EpeopleCase)
def read_case(case_id: str, db: Session = Depends(get_db)):
    db_case = db.query(EpeopleCaseOrm).filter(
        EpeopleCaseOrm.case_id == case_id
    ).first()
    if db_case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return db_case

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)