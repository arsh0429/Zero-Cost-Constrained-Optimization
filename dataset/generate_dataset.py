import numpy as np
import pickle
import torch
import os
from utils.simple_problem import SimpleProblem

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_dtype(torch.float64)

def generate_synthetic_dataset(dataset_path, num_var=100, num_ineq=50, num_eq=50, num_examples=10000):
    """Generate a synthetic optimization dataset with linear constraints.
    
    Args:
        dataset_path (str): Path where to save the dataset
        num_var (int): Number of variables
        num_ineq (int): Number of inequality constraints  
        num_eq (int): Number of equality constraints
        num_examples (int): Number of data examples
    """
    print(f"Generating dataset with: {num_var} vars, {num_eq} eq constraints, {num_ineq} ineq constraints, {num_examples} examples")
    
    np.random.seed(17)
    
    # Generate problem matrices
    Q = np.diag(np.random.random(num_var))  # Quadratic cost matrix
    p = np.random.random(num_var)           # Linear cost vector
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))    # Equality constraint matrix
    X = np.random.uniform(-1, 1, size=(num_examples, num_var))       # Input data matrix
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))  # Inequality constraint matrix
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)               # Inequality constraint vector

    print(f"Created matrices - A: {A.shape}, X: {X.shape}, G: {G.shape}")
    
    # Create the problem instance
    problem = SimpleProblem(Q, p, A, G, h, X)
    problem.calc_Y()

    # Create directory if it doesn't exist
    dirname = os.path.dirname(dataset_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    
    # Save the problem
    with open(dataset_path, 'wb') as f:
        pickle.dump(problem, f)
    
    print(f"Dataset saved to {dataset_path}")
