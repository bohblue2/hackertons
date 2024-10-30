
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from pymilvus import SearchResult
from yongin_assistant.vectorstore import MilvusSearchParams, get_client, search, get_collection


class VectorService:
    def __init__(self):
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._client = get_client() 
        self._collection = get_collection()
    
    def find_similar(self, content: str, limit: int, min_similarity: float) -> List[dict]:
        data = self._embeddings.embed_query(content)
        search_params = MilvusSearchParams(
            data=data,
            anns_field="content_embedding",
            metric_type="L2",
            nprobe=16,
            limit=10
        )
        results: SearchResult = self._collection.search(**search_params.to_dict())

