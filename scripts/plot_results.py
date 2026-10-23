#!/usr/bin/env python3
"""Plot benchmark CSV with columns label,cpu_ms,gpu_ms,end_to_end_ms.

Optional columns: gpu_setup_ms,h2d_ms,kernels_ms,d2h_ms,preprocess_ms.
"""
import csv
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 scripts/plot_results.py results/benchmark.csv")

with open(sys.argv[1], newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("CSV contains no benchmark rows")

labels = [r["label"] for r in rows]
cpu = [float(r["cpu_ms"]) for r in rows]
gpu = [float(r["gpu_ms"]) for r in rows]
x = list(range(len(labels)))
width = 0.38

plt.figure()
plt.bar([i - width / 2 for i in x], cpu, width, label="CPU filters")
plt.bar([i + width / 2 for i in x], gpu, width, label="GPU run total")
plt.xticks(x, labels, rotation=20, ha="right")
plt.ylabel("Time (ms)")
plt.legend()
plt.tight_layout()
plt.show()

# If stage columns are present, print their totals so they are not mistaken for
# end-to-end wall time. This keeps the existing two-bar plot backward-compatible.
for column in ("gpu_setup_ms", "h2d_ms", "kernels_ms", "d2h_ms", "preprocess_ms", "end_to_end_ms"):
    if rows and column in rows[0] and rows[0][column] != "":
        total = sum(float(r[column]) for r in rows)
        print(f"Sum {column}: {total:.3f} ms")
