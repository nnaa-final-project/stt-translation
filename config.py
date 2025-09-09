import os
import torch

from dataclasses import dataclass
from pathlib import Path


# --- Paths Configuration (USER MUST SET THESE) ---
COMMON_VOICE_BASE_DATA_DIR = Path(os.getenv("COMMON_VOICE_BASE_PREPROCESSED_DATA_DIR")).expanduser()
COVOST_TSV_PATH = COMMON_VOICE_BASE_DATA_DIR / "covost_v2.en_de.tsv"
OUTPUT_DIR = COMMON_VOICE_BASE_DATA_DIR # / "processed_output", using base dir because output of offline preprocessor is saved there
FEATURES_DIR = OUTPUT_DIR / "features"


# --- Parameters from data_processor.py by sygrace---
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


# --- Model Config ---
EMBED_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1


# --- Training Config ---
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training_results"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# --- Sample Hyperparameters (Just initials, not all are used) ---
BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 1000
FP16 = True if DEVICE == "cuda" else False
EVAL_STRATEGY = "steps" # or "epoch" but must adjust save strategy
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 200
SAVE_TOTAL_LIMIT = 2
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "bleu"
GREATER_IS_BETTER = True
