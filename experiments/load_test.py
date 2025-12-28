# experiments/load_test.py

import asyncio
import time
import httpx
import statistics
import json

URL = "http://localhost:8000/predict"

# Two feature patterns with SAME prefix (first 4 dims)
FEATURE_A = [1, 1, 1, 1] + [0] * 28
FEATURE_B = [1, 1, 1, 1] + [9] * 28


async def send_request(client, features, use_cache=True):
    start = time.time()
    response = await client.post(
        URL,
        json={
            "features": [features],
            "use_cache": use_cache
        }
    )
    latency_ms = (time.time() - start) * 1000
    return latency_ms


async def run_load_test(
    num_requests=100,
    concurrency=10,
    reuse_prefix=True,
    use_cache=True
):
    latencies = []

    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def task(i):
            async with semaphore:
                features = FEATURE_A if reuse_prefix else FEATURE_B
                latency = await send_request(client, features, use_cache)
                latencies.append(latency)

        await asyncio.gather(*[task(i) for i in range(num_requests)])

    return latencies


def summarize(latencies):
    return {
        "count": len(latencies),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 2),
        "max_latency_ms": round(max(latencies), 2),
    }


if __name__ == "__main__":
    print("\n==============================")
    print(" DAY 3 — LOAD TEST EXPERIMENTS")
    print("==============================\n")

    # -----------------------------
    # Experiment 1: NO CACHE
    # -----------------------------
    print("Experiment 1: Baseline (NO CACHE)")

    baseline_latencies = asyncio.run(
        run_load_test(
            num_requests=100,
            concurrency=10,
            reuse_prefix=True,
            use_cache=False
        )
    )

    baseline_summary = summarize(baseline_latencies)
    print(baseline_summary)

    # -----------------------------
    # Experiment 2: CACHE ON
    # -----------------------------
    print("\nExperiment 2: Cached (PREFIX REUSE)")

    cached_latencies = asyncio.run(
        run_load_test(
            num_requests=100,
            concurrency=10,
            reuse_prefix=True,
            use_cache=True
        )
    )

    cached_summary = summarize(cached_latencies)
    print(cached_summary)

    # -----------------------------
    # Experiment 3: BURST TRAFFIC
    # -----------------------------
    print("\nExperiment 3: Bursty Traffic (CACHE ON)")

    bursty_latencies = asyncio.run(
        run_load_test(
            num_requests=200,
            concurrency=50,
            reuse_prefix=True,
            use_cache=True
        )
    )

    bursty_summary = summarize(bursty_latencies)
    print(bursty_summary)

    # -----------------------------
    # Save results
    # -----------------------------
    results = {
        "baseline_no_cache": baseline_summary,
        "cached_prefix_reuse": cached_summary,
        "bursty_cached": bursty_summary,
    }

    with open("reports/day3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to reports/day3_results.json")
