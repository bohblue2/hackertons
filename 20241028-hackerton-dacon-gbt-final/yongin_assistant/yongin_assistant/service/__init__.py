
from typing import List, Optional


class VectorService:
    def __init__(self):
        pass
    
    def find_similar(self, content: str, limit: int, min_similarity: float) -> List[dict]:
        # 실제 구현에서는 벡터 DB나 유사도 검색 엔진을 사용
        pass

class LLMService:
    def __init__(self):
        pass
    
    def generate_answer(self, title: str, content: str, prompt_template: Optional[str] = None) -> str:
        # 실제 구현에서는 LLM API 호출
        pass