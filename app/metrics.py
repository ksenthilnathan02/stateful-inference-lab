# app/metrics.py
import time

class LatencyTracker:
    def __init__(self):
        self.latencies = []

    def record(self, ms):
        self.latencies.append(ms)

    def summary(self):
        if not self.latencies:
            return {}
        l = sorted(self.latencies)
        n = len(l)
        return {
            "count": n,
            "p50": l[int(0.5 * n)],
            "p95": l[int(0.95 * n)],
            "p99": l[int(0.99 * n) if n > 1 else -1],
        }
