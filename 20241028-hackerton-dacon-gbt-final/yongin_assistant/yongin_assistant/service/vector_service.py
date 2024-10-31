
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from pymilvus import Hit, Hits, SearchResult
from yongin_assistant.vectorstore import MilvusSearchParams, get_client, search, get_collection


class VectorService:
    def __init__(self):
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._client = get_client() 
        self._collection = get_collection()
    
    def find_similar(self, content: str, limit: int) -> List[dict]:
        data = self._embeddings.embed_query(content)
        search_params = MilvusSearchParams(
            data=data,
            anns_field="content_embedding",
            metric_type="L2",
            nprobe=16,
            limit=limit,
            output_fields=["case_id", "content"]
        )
        results: SearchResult = self._collection.search(**search_params.to_dict())
        assert len(results) == 1
        hits: Hits = results[0]
        
        similar_cases = []
        for hit in hits:
            hit: Hit
            similar_cases.append(
                {
                    "case_id": hit.get("case_id"),
                    "content": hit.get("content"),
                    "similarity_score": hit
                }
            )
        return similar_cases
    
    def close(self): ...
    
if __name__ == "__main__":
    service = VectorService()
    service.find_similar("교통신호등 설치 요청", limit=10, min_similarity=0.5)

