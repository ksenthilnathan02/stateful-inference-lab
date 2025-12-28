# experiments/plot_day3.py
import json
import matplotlib.pyplot as plt

with open("reports/day3_results.json") as f:
    results = json.load(f)

labels = ["Mean", "P95"]
baseline = [
    results["baseline_no_cache"]["mean_latency_ms"],
    results["baseline_no_cache"]["p95_latency_ms"]
]
cached = [
    results["cached_prefix_reuse"]["mean_latency_ms"],
    results["cached_prefix_reuse"]["p95_latency_ms"]
]

x = range(len(labels))

plt.bar(x, baseline, width=0.4, label="No Cache")
plt.bar([i + 0.4 for i in x], cached, width=0.4, label="Cached")

plt.xticks([i + 0.2 for i in x], labels)
plt.ylabel("Latency (ms)")
plt.title("Day 3: Baseline vs Cached Latency")
plt.legend()

plt.savefig("reports/day3_latency_comparison.png")
plt.show()
