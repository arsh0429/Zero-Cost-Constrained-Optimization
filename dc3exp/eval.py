# eval.py
import numpy as np
import torch
from models import MLP
from config import INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYERS, DEVICE, NUM_CONSTRAINTS

def compute_counts_for_k(k, X, G, h, device=DEVICE):
    """
    For a given k:
      - sample k independent model initializations (fresh MLP instances),
      - run each on the fixed X (shape N x INPUT_DIM),
      - compute integer counts c_{j,i} = number of constraints satisfied by model_j(X[i]),
    Returns:
      counts_all : numpy int array shape (k, N)
    """
    N = X.shape[0]
    counts_all = np.zeros((k, N), dtype=np.int32)

    # keep tensors on CPU by default for reproducibility
    X_t = X.to(device)
    G_t = G.to(device)
    h_t = h.to(device)

    for j in range(k):
        # create a fresh model with random init
        model = MLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYERS).to(device)
        model.eval()
        with torch.no_grad():
            preds = model(X_t)  # shape (N, NUM_VAR)
            # compute G @ preds.T  -> (NUM_CONSTRAINTS, N)
            lhs = G_t @ preds.T  # shape (M, N)
            sat = (lhs <= h_t.unsqueeze(1))  # boolean tensor (M, N)
            counts = sat.sum(dim=0).cpu().numpy()  # length N
            counts_all[j, :] = counts
    return counts_all

def metrics_from_counts(counts_all):
    """
    Input counts_all: shape (k, N) ints
    Returns the three metrics:
      - expected_satisfied: average over j,i of counts
      - best_of_k: for each instance i, max_j counts[j,i], then average over instances
      - prob_all: for each instance i, fraction of j where counts[j,i]==M; then average over instances
    """
    k, N = counts_all.shape
    M = int(counts_all.max())  # not strictly needed; user-specific NUM_CONSTRAINTS is available
    expected_satisfied = counts_all.mean()  # scalar

    best_per_instance = counts_all.max(axis=0)  # length N
    best_of_k = best_per_instance.mean()

    # per-instance fraction of inits that are fully-feasible (satisfy all constraints)
    # Note: we need the actual NUM_CONSTRAINTS value, but assuming counts equals number satisfied:
    # if a row equals NUM_CONSTRAINTS then it's fully feasible
    # we will infer NUM_CONSTRAINTS = max possible in counts_all or require explicit arg; here we compute:
    # However to be correct, the caller should pass NUM_CONSTRAINTS; instead we compute using counts_all.max() when possible.
    # A safer caller-provided NUM_CONSTRAINTS would be better, but for now:
    # We'll compute per-instance fraction where counts==counts_all.max_possible if caller passed NUM_CONSTRAINTS separately.
    # So instead, define prob_all as fraction across (j,i) where counts==max_over_all_rows? That can be wrong.
    # To be precise we assume caller will pass NUM_CONSTRAINTS separately; but to keep API simple the main will pass correctly.
    raise RuntimeError("Use metrics_from_counts_with_M(counts_all, M) instead")

def metrics_from_counts_with_M(counts_all, M):
    k, N = counts_all.shape
    expected_satisfied = counts_all.mean()
    best_of_k = counts_all.max(axis=0).mean()
    # per-instance fraction of inits that satisfy all constraints:
    frac_per_instance = (counts_all == M).sum(axis=0) / float(k)  # length N
    prob_all = frac_per_instance.mean()
    return float(expected_satisfied), float(best_of_k), float(prob_all)
