# Configuration settings for huBERT training

# Model hyperparameters
MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"
NUM_LABELS = 5
MAX_SEQUENCE_LENGTH = 512
HIDDEN_SIZE = 128
DROPOUT_RATE = 0.4

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "CrossEntropyLoss"

# Data split parameters
TEST_SIZE = 0.33
VAL_SIZE = 0.2
RANDOM_STATE_SPLIT = 33
RANDOM_STATE_VAL = 40

# Class weights for imbalanced dataset
CLASS_WEIGHTS = [656/95, 656/261, 656/439, 656/641, 656/656]

# Paths
DATA_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"
TRAINING_DATA_PATH = "./data/training.csv"
TEST_DATA_PATH = "./data/test.csv"
RAW_DATA_PATH = "./data/raw_data.csv"
MODEL_SAVE_PATH = "./model/deephubert_state.bin"
TOKENIZER_SAVE_PATH = "./tokenizer/leanhubert"

# Logging
LOG_INTERVAL = 10  # Log every N batches
