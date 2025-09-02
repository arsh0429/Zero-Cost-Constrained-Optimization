# config.py
# Global configuration / hyperparameters

INPUT_DIM = 10       # feature-dimension of fixed X
NUM_VAR = 20         # dimension of the decision variable x (model output dim)
HIDDEN_LAYERS = [64, 64, 16]  # three hidden layers as requested
OUTPUT_DIM = NUM_VAR

NUM_CONSTRAINTS = 12  # number of inequalities (G x <= h)

NUM_INSTANCES = 1000  # number of fixed inputs X (store these for reuse)
SEED = 42

# k sweep: start at 1 then steps of 5 until 1000
K_MAX = 1000
K_STEP = 5
K_VALUES = [1] + list(range(5, K_MAX + 1, K_STEP))

DEVICE = "cpu"  # keep CPU by default, change to "cuda" if you want (and have GPU)

# file paths
DATA_DIR = "data"
X_PATH = f"{DATA_DIR}/X_fixed.pt"
CONSTRAINTS_PATH = f"{DATA_DIR}/constraints.pt"

# plotting / saving
OUT_DIR = "out"
