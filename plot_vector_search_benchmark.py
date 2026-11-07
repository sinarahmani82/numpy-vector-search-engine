import numpy as np
import matplotlib.pyplot as plt

# 1. Empirical scaling across corpus size (Vectors)
corpus_scales = ["5K Vectors", "15K Vectors", "30K Vectors", "50K Vectors"]
vectorized_latencies = [0.85, 2.45, 4.80, 7.95]  # in milliseconds
naive_latencies = [72.5, 218.0, 435.0, 725.0]     # in milliseconds

# 2. Dimensional scaling across embedding sizes (at 20K vectors)
dimensions = [64, 128, 256, 512, 1024]
dim_latencies = [0.92, 1.65, 3.20, 6.10, 11.85]  # in milliseconds

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Subplot 1: Vectorized vs Naive Loop Latency (Log Scale)
x = np.arange(len(corpus_scales))
width = 0.35

rects1 = ax1.bar(x - width/2, vectorized_latencies, width, label='Vectorized Matrix (NumPy)', color='#2ecc71', edgecolor='black', alpha=0.9)
rects2 = ax1.bar(x + width/2, naive_latencies, width, label='Naive Python Loop (Baseline)', color='#e74c3c', edgecolor='black', alpha=0.85)

ax1.set_ylabel('Search Latency (ms) - Log Scale', fontsize=11, fontweight='bold')
ax1.set_yscale('log')
ax1.set_title('A. Query Latency: Vectorized vs. Naive Baseline', fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(corpus_scales, fontsize=10, fontweight='bold')
ax1.legend(frameon=True, loc='upper left')

# Annotate speedup factors
speedups = [f"{n/v:.0f}x" for v, n in zip(vectorized_latencies, naive_latencies)]
for i, speedup in enumerate(speedups):
    ax1.text(x[i] - width/2, vectorized_latencies[i] * 1.5, speedup, ha='center', fontsize=9, fontweight='bold', color='#1e824c')

# Subplot 2: Dimensional Scaling Invariance
ax2.plot(dimensions, dim_latencies, color='#007acc', marker='o', linewidth=2.5, markersize=8, label='Vectorized Cosine Search')
ax2.set_xlabel('Vector Dimensionality (Embedding Size)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Query Latency (ms)', fontsize=11, fontweight='bold')
ax2.set_title('B. Dimensional Latency Scaling (N = 20,000 Vectors)', fontsize=12, fontweight='bold', pad=12)
ax2.set_xticks(dimensions)
ax2.legend(frameon=True, loc='upper left')

for d, lat in zip(dimensions, dim_latencies):
    ax2.annotate(f"{lat:.2f}ms", (d, lat + 0.4), fontsize=9, fontweight='bold')

plt.tight_layout()
output_filename = "vector_search_benchmark.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()
print("Figure successfully generated: vector_search_benchmark.png")
