
from dataclasses import dataclass
import os
from typing import Any, List, Optional
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, utility
from yongin_assistant.constant import DEFAULT_EMBEDDING_DIM, MILVUS_METRIC_TYPE, MILVUS_NLIST, MILVUS_NPROBE, MILVUS_TOP_K, MILVUS_INDEX_TYPE
from pymilvus import connections

def get_client() -> MilvusClient:   
    client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_API_KEY"))
    connections.connect(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_API_KEY"))
    return client

def get_collection() -> Collection:
    fields = [
        FieldSchema(
            name="case_id",
            dtype=DataType.INT64,
            description="case_id",
            is_primary=True
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            description="content", 
            max_length=5000
        ),
        FieldSchema(
            name="content_embedding",
            dtype=DataType.FLOAT_VECTOR,
            description="content_embedding",
            dim=DEFAULT_EMBEDDING_DIM
        ),
        FieldSchema(
            name="question_datetime",
            dtype=DataType.VARCHAR,
            description="question_datetime",
            max_length=30
        ),
        FieldSchema(
            name="answer_datetime", 
            dtype=DataType.VARCHAR,
            description="answer_datetime",
            max_length=30
        )
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(
        name="epeople_cases",
        schema=schema,
        description="epeople_cases",
    )
    return collection

def create_index(collection: Collection, wait_for_building:bool=True) -> None:
    collection.create_index(
        field_name="content_embedding", 
        index_params={
            "metric_type": MILVUS_METRIC_TYPE,
            "index_type": MILVUS_INDEX_TYPE,
            "params": { "nlist": MILVUS_NLIST }
        }
    )
    if wait_for_building:
        utility.wait_for_index_building_complete(
            collection.name, 
            index_name="content_embedding"
        )

@dataclass
class MilvusSearchParams:
    data: Any
    anns_field: str
    metric_type: str
    nprobe: str
    limit: int
    expr: str = ""
    output_fields: Optional[List[str]] = None 
    
    def to_dict(self) -> dict:
        search_params = {
            "data": [self.data],
            "anns_field": self.anns_field,
            "param": {
                "metric_type": self.metric_type,
                "params": {"nprobe": self.nprobe}
            },
            "limit": self.limit,
            "output_fields": self.output_fields
        }
        
        if self.expr:
            search_params["expr"] = self.expr
            
        return search_params

def search(collection: Collection, search_params: MilvusSearchParams):
    return collection.search(**search_params.to_dict())