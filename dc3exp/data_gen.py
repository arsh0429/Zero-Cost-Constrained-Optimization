# data_gen.py
import os
import torch
import numpy as np
from config import NUM_INSTANCES, INPUT_DIM, NUM_CONSTRAINTS, DATA_DIR, X_PATH, CONSTRAINTS_PATH, SEED

os.makedirs(DATA_DIR, exist_ok=True)

def generate_or_load_fixed_X(force_regen=False):
    """
    Creates (or loads) a fixed set of NUM_INSTANCES inputs X of shape (N, INPUT_DIM).
    Saves X to disk so the exact same X can be reused later.
    """
    if (not force_regen) and os.path.exists(X_PATH):
        X = torch.load(X_PATH)
        print(f"Loaded fixed X from {X_PATH} (shape {X.shape})")
        return X
    torch.manual_seed(SEED)
    X = torch.randn(NUM_INSTANCES, INPUT_DIM, dtype=torch.float32)
    torch.save(X, X_PATH)
    print(f"Saved fixed X to {X_PATH} (shape {X.shape})")
    return X

def generate_or_load_constraints(force_regen=False):
    """
    Generate or load constraints (G, h):
      G: (NUM_CONSTRAINTS, NUM_VAR)
      h: (NUM_CONSTRAINTS,)
    These are also saved to disk so experiments are reproducible.
    """
    if (not force_regen) and os.path.exists(CONSTRAINTS_PATH):
        d = torch.load(CONSTRAINTS_PATH)
        print(f"Loaded constraints from {CONSTRAINTS_PATH}")
        return d['G'], d['h']

    torch.manual_seed(SEED + 1)
    from config import NUM_VAR
    G = torch.randn(NUM_CONSTRAINTS, NUM_VAR, dtype=torch.float32)
    # Make h positive-ish and of a magnitude comparable to G*x typical values:
    h = torch.randn(NUM_CONSTRAINTS, dtype=torch.float32) * 2.0 + 1.0
    torch.save({'G': G, 'h': h}, CONSTRAINTS_PATH)
    print(f"Saved constraints to {CONSTRAINTS_PATH}")
    return G, h
