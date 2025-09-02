# plot.py
import os
import matplotlib.pyplot as plt
from config import OUT_DIR

os.makedirs(OUT_DIR, exist_ok=True)

def plot_metric_vs_k(k_values, metric_values, ylabel, out_name):
    plt.figure(figsize=(8,4))
    plt.plot(k_values, metric_values, marker='o', linewidth=1)
    plt.xlabel("k (number of random initializations)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.4, linestyle='--')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")
