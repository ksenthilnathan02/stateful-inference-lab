# app/main.py

import time
import hashlib
import torch
from fastapi import FastAPI

from app.model import Model
from app.cache import LRUWithTTLCache
from app.metrics import LatencyTracker
from app.schemas import PredictRequest, PredictResponse
from app.config import (
    CACHE_MAX_SIZE,
    CACHE_TTL_SECONDS,
    CACHE_ENABLED,
    PREFIX_K,
)


app = FastAPI()

# -----------------------------
# Core components
# -----------------------------
model = Model()
cache = LRUWithTTLCache(
    max_size=CACHE_MAX_SIZE,
    ttl_seconds=CACHE_TTL_SECONDS
)
latency_tracker = LatencyTracker()

# -----------------------------
# Circuit breaker state
# -----------------------------
CACHE_ENABLED = True
RISK_THRESHOLD = 0.9  # conservative
rolling_risk = []    # simple rolling buffer
ROLLING_WINDOW = 20


def make_cache_key(features):
    prefix = [row[:PREFIX_K] for row in features]
    raw = str(prefix).encode()
    return hashlib.sha256(raw).hexdigest()



@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global CACHE_ENABLED

    start = time.time()

    x = torch.tensor(req.features).unsqueeze(0).float()
    cache_key = make_cache_key(req.features)

    cache_hit = False
    h = None

    # -----------------------------
    # Cache lookup (if enabled)
    # -----------------------------
    if req.use_cache and CACHE_ENABLED:
        h = cache.get(cache_key)
        if h is not None:
            cache_hit = True

    # -----------------------------
    # Encoder run if needed
    # -----------------------------
    if h is None:
        h = model.encode(x)
        if req.use_cache and CACHE_ENABLED:
            cache.set(cache_key, h)

    y = model.predict(h)

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
        "cache_enabled": CACHE_ENABLED,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
