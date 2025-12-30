````md
# Stateful Inference Lab (KV-Cache–Inspired)

A production-style machine learning inference system that explores **performance–correctness tradeoffs in stateful inference**, inspired by **KV caching** techniques used in modern ML systems.

This project focuses on **inference infrastructure**, not model accuracy: how to make predictions **faster**, **safer**, and **reproducible** under load and data drift.

---

## 🚀 Project Overview

Modern ML systems often serve **similar or repeated requests** at high throughput. Recomputing expensive intermediate representations for every request is wasteful, but caching introduces risks when inputs drift over time.

This project answers the question:

> **How can we reuse expensive inference computation while bounding correctness risk under drift?**

To study this, we build a **stateful inference service** with:
- prefix-based caching (KV-inspired)
- hybrid eviction policies
- latency and correctness evaluation
- production-style deployment using Docker

---

## 🧠 What the Model Does (Simple Explanation)

The model is intentionally simple and consists of **two stages**:

1. **Encoder (slow)**  
   Converts a 32-dimensional input vector into an internal embedding (a compact representation).

2. **Prediction Head (fast)**  
   Maps the embedding to a single score between **0 and 1**.

### Example

**Input**
```json
{
  "features": [
    [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  ]
}
````

**Output**

```json
{
  "prediction": 0.33,
  "cache_hit": false,
  "latency_ms": 1.1
}
```

The **exact meaning of the score is not important**.
The project treats the model as a black box and focuses on **how inference is served**, not how the model is trained.

---

## 🏗️ System Architecture

```
Request
  ↓
[ Encoder (expensive) ]  ←─ cached by prefix
  ↓
[ Prediction Head (cheap) ]
  ↓
Response
```

**Key idea**

* Cache the **encoder output**, not the final prediction
* Reuse work when requests share a common prefix

---

## 🔑 Prefix-Based (KV-Inspired) Caching

Only the **first `K` input features** are used to construct the cache key.

This mirrors **KV caching** in transformer models:

* shared context → reused computation
* request-specific details → recomputed cheaply

**Benefit:** reduced latency and compute
**Risk:** stale embeddings under drift

---

## 🧰 Caching Policies

The system implements a **hybrid cache**.

### LRU (Least Recently Used)

* Bounds memory usage
* Evicts old, unused entries

### TTL (Time-to-Live)

* Expires entries after a fixed duration
* Bounds correctness risk under drift

**Together**

> LRU bounds memory, TTL bounds staleness

---

## 📊 Experiments

### 1️⃣ Load Testing

* Async request generator
* Steady and bursty traffic
* Metrics:

  * mean latency
  * p95 / p99 latency
  * cache hit rate

**Finding:**
Caching reduces tail latency under steady traffic but can degrade under bursts.

---

### 2️⃣ Drift Simulation

We simulate **input drift** by:

* keeping the prefix fixed
* gradually changing the remaining features

This models real-world distribution shift.

---

### 3️⃣ Correctness Risk Metric

Because labels are unavailable at inference time, we define a **proxy correctness risk**:

> **Embedding divergence between cached and fresh representations**

Measured using:

* cosine distance
* L2 distance

**Interpretation:**
Larger divergence ⇒ higher risk of stale predictions

---

## 🛡️ Production Safeguards

* **Hybrid LRU + TTL cache**
* **Observability**

  * hit rate
  * evictions
  * expirations
  * latency percentiles
* Caching treated as an **optimization**, not a dependency

---

## 🤖 Agentic Infrastructure Extension (Correctness-Aware Recovery)

Beyond static caching policies, the system includes a **lightweight agentic control loop** that adapts inference behavior at runtime based on observed signals.

This extension does **not** change the core model or caching design.
Instead, it treats caching as a *controllable optimization* rather than a fixed assumption.

The agent runs **synchronously inside the inference path**, ensuring low-latency recovery without external orchestration.

---

### 🔍 Motivation

In real production systems:

* Cache hit rate alone is not a sufficient health signal
* High cache reuse under drift can silently degrade correctness
* Systems must be able to **recover automatically**, not just observe failures

This motivates an **agentic inference component** that can:

* observe runtime signals
* diagnose correctness risk
* take corrective action

---

### 🧠 Agent Design

The agent follows a simple but principled **Observe → Decide → Act** loop:

```
Observe:
  - embedding divergence (correctness risk proxy)
  - cache hit / miss statistics

Decide:
  - is cached reuse still safe?

Act:
  - disable cache reuse
  - force fresh computation
```

The agent is intentionally **rule-based**, not learned, to ensure:

* interpretability
* low latency
* production safety

---

### 🧩 Implementation Overview

The agent is split into two clean layers.

#### 1️⃣ Policy (Decision Logic)

```python
def should_disable_cache(risk_score, threshold):
    return risk_score > threshold
```

* Pure function
* No side effects
* Easy to test and extend

#### 2️⃣ Controller (System Action)

```python
if risk_exceeds_threshold:
    cache.enabled = False
```

* Applies decisions to the inference system
* Keeps cache implementation independent of policy logic

This separation mirrors production ML infrastructure patterns where
**decision logic is decoupled from system mechanics**.

---

### 🔒 Safety Properties

* Caching can be **fully disabled at runtime**
* System always falls back to **fresh inference**
* No single optimization is a correctness dependency

This ensures that:

> Performance optimizations never compromise system correctness.

---

## 🐳 Dockerized Deployment

The entire inference service is packaged with Docker to ensure:

* reproducibility
* environment isolation
* production-like behavior

### Run with Docker

```bash
docker compose up --build
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## 📈 Key Findings

* Prefix caching significantly improves latency for repeated requests
* Drift can silently increase correctness risk even when cache hit rates are high
* TTL expiration effectively bounds staleness at the cost of recomputation
* Inference systems must balance **performance, correctness, and safety**

---

## 📂 Project Structure

```
stateful-inference-lab/
├── app/                # FastAPI inference service
│   ├── main.py
│   ├── model.py
│   ├── cache.py
│   ├── metrics.py
│   ├── schemas.py
│   └── config.py
├── agent/              # Agent policy and controller
├── experiments/        # Load, drift, correctness experiments
├── docker/             # Dockerfile & compose
├── reports/            # Experiment outputs
└── README.md
```

---

## 🔮 Future Work

* Adaptive TTL based on observed correctness risk
* Canary cache policies
* Prometheus / Grafana integration
* Multi-model versioning

````

---
