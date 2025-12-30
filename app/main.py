# app/main.py

import time
import hashlib
import torch
from fastapi import FastAPI
from app.model import Model
from app.cache import LRUWithTTLCache
from app.metrics import LatencyTracker
from app.schemas import PredictRequest, PredictResponse

app = FastAPI()

model = Model()
cache = LRUWithTTLCache(max_size=256, ttl_seconds=30)
latency_tracker = LatencyTracker()


def make_cache_key(features, prefix_k=4):
    """
    Cache key based on prefix (KV-cache inspired).
    """
    prefix = features[:prefix_k]
    raw = str(prefix).encode()
    return hashlib.sha256(raw).hexdigest()


def agent_policy(risk_score):
    """
    Simple agent decision logic.

    This is intentionally lightweight:
    - High risk → disable cache
    - Low risk → enable cache
    """
    if risk_score > 0.8:
        cache.enabled = False
    else:
        cache.enabled = True


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    x = torch.tensor(req.features).unsqueeze(0).float()
    cache_key = make_cache_key(req.features)

    cache_hit = False
    h = None

    # Try cache
    if req.use_cache:
        h = cache.get(cache_key)
        if h is not None:
            cache_hit = True

    # Encoder execution
    if h is None:
        h = model.encode(x)
        if req.use_cache:
            cache.set(cache_key, h)

    # Prediction
    y = model.predict(h)

    # --- Correctness Risk Proxy (embedding magnitude) ---
    risk_score = float(torch.norm(h).item() / 10.0)  # normalized proxy

    # Agent acts
    agent_policy(risk_score)

    latency_ms = (time.time() - start) * 1000
    latency_tracker.record(latency_ms)

    return PredictResponse(
        prediction=float(y.item()),
        cache_hit=cache_hit,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/stats")
def stats():
    return {
        "cache": cache.stats(),
        "latency": latency_tracker.summary(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cache_enabled": cache.enabled,
    }
