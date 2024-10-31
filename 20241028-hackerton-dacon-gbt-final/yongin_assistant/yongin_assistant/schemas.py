from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional   

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime

class EpeopleCaseBase(BaseModel):
    case_id: str
    title: str
    question_date: datetime
    content: Optional[str]  = None 
    department: Optional[str]  = None 
    related_laws: Optional[str]  = None 
    answer_date: datetime
    vectorized: bool = False

class EpeopleCaseCreate(EpeopleCaseBase): ...

class EpeopleCase(EpeopleCaseBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class EpeopleCaseWithAnswer(EpeopleCase):
    recommended_answer: str

    class Config:
        from_attributes = True
    
class SimilarCase(BaseModel):
    case_id: str 
    title: str



class RecommendAnswerResponse(BaseModel):
    case_id: str
    content: str
    department: str
    created_at: datetime
    temperature: Optional[float] = 0.0


class RecommendAnswerRequest(BaseModel):
    case_id: str
    prompt_instruction: Optional[str] = None # NOTE: 단정한 어조를 사용해줘
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 1000

# TODO: refactor me (these object are duplicated with RecommendAnswerResponse)
class CaseWithAnswer(BaseModel):
    case: EpeopleCase
    recommended_answer: str

class SimilarCaseGroup(BaseModel):
    cases: List[CaseWithAnswer]

class CategoryCases(BaseModel):
    category_name: str
    target_case: EpeopleCaseWithAnswer
    similar_case_groups: List[SimilarCaseGroup]