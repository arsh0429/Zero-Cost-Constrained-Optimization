# main.py
import os
import numpy as np
import torch
from config import K_VALUES, NUM_INSTANCES, NUM_CONSTRAINTS, DEVICE, OUT_DIR
from data_gen import generate_or_load_fixed_X, generate_or_load_constraints
from eval import compute_counts_for_k, metrics_from_counts_with_M
from plot import plot_metric_vs_k

os.makedirs(OUT_DIR, exist_ok=True)

def run_experiment():
    # 1) load / generate fixed inputs and constraints
    X = generate_or_load_fixed_X()            # (N, INPUT_DIM) torch tensor
    G, h = generate_or_load_constraints()    # G: (M, NUM_VAR), h: (M,)

    # move constraints to CPU/desired device
    G = G.to(DEVICE)
    h = h.to(DEVICE)

    k_list = K_VALUES
    expected_list = []
    best_list = []
    prob_all_list = []

    # for reproducible-ish ordering, set torch seed here (but model creation uses internal init)
    torch = __import__('torch')
    torch.manual_seed(0)

    for idx, k in enumerate(k_list):
        print(f"[{idx+1}/{len(k_list)}] Computing for k = {k} ...")
        counts_all = compute_counts_for_k(k, X, G, h, device=DEVICE)  # (k, N) numpy array
        expected_satisfied, best_of_k, prob_all = metrics_from_counts_with_M(counts_all, NUM_CONSTRAINTS)

        expected_list.append(expected_satisfied)
        best_list.append(best_of_k)
        prob_all_list.append(prob_all)

    # Convert to numpy arrays and save
    np.save(f"{OUT_DIR}/k_values.npy", np.array(k_list))
    np.save(f"{OUT_DIR}/expected_satisfied.npy", np.array(expected_list))
    np.save(f"{OUT_DIR}/best_of_k.npy", np.array(best_list))
    np.save(f"{OUT_DIR}/prob_all.npy", np.array(prob_all_list))
    print(f"Saved metrics to {OUT_DIR}")

    # Plot each metric vs k
    plot_metric_vs_k(k_list, expected_list, "Expected # constraints satisfied", "expected_vs_k.png")
    plot_metric_vs_k(k_list, best_list, "Avg best # constraints (best-of-k)", "best_vs_k.png")
    plot_metric_vs_k(k_list, prob_all_list, "Probability a weight satisfies all constraints", "prob_all_vs_k.png")

if __name__ == "__main__":
    run_experiment()
