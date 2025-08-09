# Zero Cost Neural Architecture Search

This project implements zero-cost proxy methods for Neural Architecture Search (NAS) using synthetic datasets.

## Dependencies

The project requires the following Python packages:
- `numpy>=1.21.0` - For numerical computations and array operations
- `torch>=1.9.0` - PyTorch for neural network models and tensor operations

## Installation

### Option 1: Automatic Installation
The main script will automatically install missing dependencies when you run it:

```bash
python main.py
```

### Option 2: Manual Installation
Install dependencies manually using pip:

```bash
pip install -r requirements.txt
```

Or install individual packages:

```bash
pip install numpy>=1.21.0 torch>=1.9.0
```

### Option 3: Using Setup Script
Run the setup script to install all dependencies:

```bash
python setup.py
```

## Project Structure

```
├── main.py                    # Main execution script with automatic dependency installation
├── requirements.txt           # List of required packages
├── setup.py                  # Setup script for easy installation
├── data/                     # Directory for generated datasets
├── dataset/
│   ├── __init__.py
│   └── generate_dataset.py   # Synthetic dataset generation
├── models/
│   ├── __init__.py
│   └── search_space.py       # Neural network architecture definitions
├── utils/
│   ├── __init__.py
│   └── simple_problem.py     # Problem definition utilities
└── zero_cost/
    ├── __init__.py
    └── zero_cost_score.py     # Zero-cost proxy scoring methods
```

## Usage

Once dependencies are installed, simply run:

```bash
python main.py
```

The script will:
1. Check and install any missing dependencies
2. Generate a synthetic dataset (if not already exists)
3. Create a search space of neural network architectures
4. Apply zero-cost proxy methods to filter architectures
5. Display the top architectures and their scores

## Features

- **Automatic Dependency Management**: The main script automatically checks and installs missing packages
- **Synthetic Dataset Generation**: Creates optimization problems with linear constraints
- **Neural Architecture Search**: Generates and evaluates MLP architectures
- **Zero-Cost Proxy Methods**: Uses NAS-WOT (Neural Architecture Search Without Training) scoring
- **Constraint Filtering**: Filters architectures based on parameter count limits
