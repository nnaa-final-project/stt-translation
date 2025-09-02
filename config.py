import torch

# Constants and env variables

TARGET_SAMPLE_RATE = 16_000
DATASET_PATH = "path/to/your/dataset"
DATASET_NAME = "covost2"
DATASET_CONFIG_NAME = "en-de"
TEXT_TRANSLATION_COLUMN = "translation"
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "sentence"

TOKENIZER_PATH = "path/to/your/tokenizer"
TOKENIZER_VOCAB_SIZE = 10_000
ENCODER_MODEL = "facebook/wav2vec2-base"
DECODER_MODEL = "facebook/bart-base"

# Hugging Face Hub settings
HF_USERNAME = "your-username"
HF_HUB_MODEL_ID = f"{HF_USERNAME}/stt_translation_en-de" # tentative name

# Training Config
OUTPUT_DIR = "./results"


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
