from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional   

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
    
    