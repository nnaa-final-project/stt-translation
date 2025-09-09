import torch
import librosa
import numpy as np
import json
import sentencepiece as spm
import config
from pathlib import Path

from scripts.encoder_decoder_transformer import SpeechToTextTranslationModel


class InferenceEngine:
    def __init__(self, model: Path):
        pass

    def _extract_features(self, audio_path: str) -> torch.Tensor:
        pass

    def translate_audio(self) -> str:
        pass

