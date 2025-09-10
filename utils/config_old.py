from dataclasses import dataclass

@dataclass
class AudioParams:
    # Step 2 EDA & Filtering
    sr_target: int = 16000
    min_dur: float = 0.2
    max_dur: float = 20.0
    snr_db_thresh: float = 5.0

    # Step 4 Standardization (ffmpeg)
    sil_start_db: int = 20
    sil_stop_db: int = 20
    sil_pad_s: float = 0.05

    # Step 5 Feature extraction
    n_mels: int = 80
    win_ms: int = 25
    hop_ms: int = 10

    # Parallelism
    workers: int = 8

@dataclass
class TextParams:
    # SentencePiece
    spm_vocab_size: int = 16000
    spm_model_type: str = "unigram"  # or "bpe"

    # Filters
    max_tok_en: int = 200
    max_tok_de: int = 200
    max_char_en: int = 1200
    max_char_de: int = 1200
    max_len_ratio: float = 3.0
    min_tok: int = 1
