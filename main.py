# Install required packages if not already installed
import subprocess
import sys

def install_requirements():
    """Install required packages if they are not already installed."""
    required_packages = ['numpy>=1.21.0', 'torch>=1.9.0']
    
    for package in required_packages:
        try:
            package_name = package.split('>=')[0].split('==')[0]
            __import__(package_name)
            print(f"✓ {package_name} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")

# Install requirements before importing
install_requirements()

import os
import torch
import pickle
from dataset.generate_dataset import generate_synthetic_dataset
from models.search_space import generate_search_space
from zero_cost.zero_cost_score import randomly_sample_architectures

DATASET_PATH = "./data/synthetic.pkl"
INPUT_DIM = 50
OUTPUT_DIM = 50

if not os.path.exists(DATASET_PATH):
    generate_synthetic_dataset(DATASET_PATH, num_var=100, num_eq=50, num_ineq=50, num_examples=10000)

with open(DATASET_PATH, 'rb') as f:
    problem = pickle.load(f)

search_space = generate_search_space(INPUT_DIM, OUTPUT_DIM, samples=100)
filtered_pool = randomly_sample_architectures(search_space, constraints={"max_params": 50000}, score_threshold=0.0, max_samples=10)

print("Top filtered architectures and their scores:")
for i, (model, score) in enumerate(filtered_pool):
    print(f"Model {i+1}: Score = {score:.4f}, Params = {sum(p.numel() for p in model.parameters())}")