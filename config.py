import os
import torch

from dataclasses import dataclass
from pathlib import Path


# --- Paths Configuration (DEV MUST SET THE BASE PATH AS ENV VAR ON THEIR MACHINE) ---
COMMON_VOICE_BASE_DATA_DIR = Path(os.getenv("COMMON_VOICE_BASE_PREPROCESSED_DATA_DIR")).expanduser()
COVOST_TSV_PATH = COMMON_VOICE_BASE_DATA_DIR / "covost_v2.en_de.tsv"
OUTPUT_DIR = COMMON_VOICE_BASE_DATA_DIR
PROCESSED_DATA_DIR = OUTPUT_DIR / "processed"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"


# --- Parameters from data_processor.py by @sygrace---
@dataclass
class AudioParams:
    """Parameters for audio processing."""
    sr_target: int = 16000
    min_dur: float = 0.2
    max_dur: float = 20.0
    snr_db_thresh: float = 5.0
    sil_start_db: int = 20
    sil_stop_db: int = 20
    sil_pad_s: float = 0.05
    n_mels: int = 80
    win_ms: int = 25
    hop_ms: int = 10
    workers: int = 8

@dataclass
class TextParams:
    """Parameters for text processing and filtering."""
    spm_vocab_size: int = 16000
    spm_model_type: str = "unigram"
    max_tok_en: int = 200
    max_tok_de: int = 200
    max_char_en: int = 1200
    max_char_de: int = 1200
    max_len_ratio: float = 3.0
    min_tok: int = 1


@dataclass
class DatasetParams:
    """Parameters for dataset loading and splitting."""
    use_subset: bool = True
    subset_fraction: float = 1 # Use 100% of training data, 0.01 = 1% of training data, 0.1 = 10% of training data
    subset_size: int = None  # Another way to do it is to specify exact train subset size
    random_seed: int = 42
    split_method: str = "random" #"first_n", "random", or "stratified"

    # split train data into train/val if needed
    create_val_split: bool = False
    val_split_ratio: float = 0.1  # 10% for validation

    # For subsets reproducibility
    shuffle_before_split: bool = True


# --- Model Config ---
EMBED_DIM = 256
NUM_HEADS = 4 # reduced from 8 to 4 to decrease mmodel size and stability
NUM_ENCODER_LAYERS = 3 # reduced from 4 to 3 to decrease model size and training time
NUM_DECODER_LAYERS = 3 # reduced from 4 to 3 to decrease model size and training time
D_FF = 512 # reduced from 1024 to 512 to decrease model size and training time
DROPOUT = 0.2 # 0.1 increased to 0.2 to reduce overfitting

# --- Sequence Length Constraints ---
MAX_AUDIO_SEQUENCE_LENGTH = 1024  # Limit audio features
MAX_TEXT_SEQUENCE_LENGTH = 256    # Limit text tokens
ATTENTION_CHUNK_SIZE = 512


# --- Training Config ---
TRAINING_OUTPUT_DIR = Path("./models")  # OUTPUT_DIR / "training_results"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# --- Sample Hyperparameters (Just initials, not all are used) ---
BATCH_SIZE = 1 # redudced from 4 to 1 to fit in memory
NUM_TRAIN_EPOCHS = 10 # Increase to 10 from 5 for better results
# LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 16 # increased from 4 to simulate larger batch size
WARMUP_STEPS = 1000
FP16 = True if DEVICE == "cuda" else False
EVAL_STRATEGY = "steps" # or "epoch" but must adjust save strategy
EVAL_STEPS = 2000
SAVE_STEPS = 2000
LOGGING_STEPS = 200
SAVE_TOTAL_LIMIT = 2
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "bleu"
GREATER_IS_BETTER = True

# Adds stability and prevents exploding gradients
MAX_GRAD_NORM = 1.0              # gradient clipping
LEARNING_RATE = 1e-4             # Reduced from 2e-4
WARMUP_RATIO = 0.1


# --- Loss and Generation Config ---
LABEL_SMOOTHING = 0.1
PAD_TOKEN_ID = 0                 # Padding token
BOS_TOKEN_ID = 1                 # start of sequence token
EOS_TOKEN_ID = 2


# --- Dataset Config ---
DATASET_PARAMS = DatasetParams()
CHUNK_SIZE_GB = 0.5  # Maximum memory per chunk in GB


# Quick access variables for backward compatibility
USE_SUBSET = DATASET_PARAMS.use_subset
SUBSET_FRACTION = DATASET_PARAMS.subset_fraction
SUBSET_SIZE = DATASET_PARAMS.subset_size
RANDOM_SEED = DATASET_PARAMS.random_seed
SPLIT_METHOD = DATASET_PARAMS.split_method

