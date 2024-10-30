from typing import Any
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

from dataclasses import dataclass

@dataclass
class MilvusSearchParams:
    data: Any
    anns_field: str
    metric_type: str
    nprobe: str
    limit: int
    expr: str = ""
    
    def to_dict(self) -> dict:
        search_params = {
            "data": self.data,
            "anns_field": self.anns_field,
            "param": {
                "metric_type": self.metric_type,
                "params": {"nprobe": self.nprobe}
            },
            "limit": self.limit
        }
        
        if self.expr:
            search_params["expr"] = self.expr
            
        return search_params

from example_pymilvus_llm import 