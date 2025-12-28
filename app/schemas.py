# app/schemas.py
from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    features: List[List[float]]  # shape: (seq_len, input_dim)
    use_cache: bool = True

class PredictResponse(BaseModel):
    prediction: float
    cache_hit: bool
    latency_ms: float
