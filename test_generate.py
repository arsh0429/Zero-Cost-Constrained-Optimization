import numpy as np
import pickle
import torch
import os
from utils.simple_problem import SimpleProblem

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_dtype(torch.float64)

def test_generate_synthetic_dataset(dataset_path, num_var=100, num_ineq=50, num_eq=50, num_examples=10000):
    print(f"Function called with: num_var={num_var}, num_eq={num_eq}, num_ineq={num_ineq}, num_examples={num_examples}")
    np.random.seed(17)
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    print(f"A created with shape: {A.shape}")
    X = np.random.uniform(-1, 1, size=(num_examples, num_var))
    print(f"X created with shape: {X.shape}")
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)

    print(f"Debug: A shape: {A.shape}, X shape: {X.shape}, X.T shape: {X.T.shape}")
    print(f"About to create SimpleProblem...")
    
    problem = SimpleProblem(Q, p, A, G, h, X)
    print("SimpleProblem created, calling calc_Y()")
    problem.calc_Y()
    print("calc_Y() completed successfully!")

    with open(dataset_path, 'wb') as f:
        pickle.dump(problem, f)
    print("File saved successfully!")

if __name__ == "__main__":
    test_generate_synthetic_dataset('test_new.pkl', num_var=100, num_eq=50, num_ineq=50, num_examples=10000)
