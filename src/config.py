import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data params
DATA_PATH = "data/02_processed/comprehensive_master_data_universal.parquet"
PROCESSED_DATA_DIR = "data/02_processed"
GROUPS_COLS = ["subjName", "trialN"]
TARGET_COL = "grip_strategy_label"

FEATURES_TO_DROP = ["pathLength", "MGA", "timeMGA", "movTime"]

# Model params
MAX_SEQ_LEN = 1200
BATCH_SIZE = 32
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.1
NUM_CLASSES = 72

# Training params
LEARNING_RATE = 1e-4
EPOCHS = 50
NUM_FOLDS = 5
