# Controller Parameter
NUM_PARALLELS = 32
NUM_STEP = 34
MIN_SAMPLE_SIZE = 128 * 10

# Learning Parameter
LEARNING_RATE = 1e-5
BATCH_SIZE = 128
NUM_EPOCH = 30
CLIP_EPS = 0.2
GAMMA = 0.99
LAMBDA = 1.0
ENTROPY_COEFFICIENT = 0.1

# Env Parameter
REWARD_FUNC = "RAW"  # or "RANK"
REWARD_LIST = [1.0, 0.5, -0.5, -1]

# Model Parameter
NUM_LAYERS = 12
NUM_FILTERS = 32
KERNEL_SIZE = (3, 3)
BATCH_NORMALIZATION = True

# GPU (Only CPU for now)
USE_GPU = True

# Save Info
SAVE_FREQ = 1000
SAVE_DIR = "./trained_models"

# Training Mode
AGAINST_GREEDY = True
