# case id
# original question title/answer
# inferred question content
from pymilvus import FieldSchema, CollectionSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    # configure default value `25` for field `age`
    # FieldSchema(name="age", dtype=DataType.INT64, default_value=25, description="age"),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="vector")
]