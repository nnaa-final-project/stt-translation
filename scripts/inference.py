# scripts/inference.py
import math
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch

import config
from scripts.encoder_decoder_transformer import SpeechToTextTranslationModel

try:
    from safetensors.torch import load_file as safe_load
except Exception:
    safe_load = None


# ---- Special tokens (adjust if your SPM uses different ids) ----
BOS_ID = 1   # often <s>
EOS_ID = 2   # often </s>


def wav_to_logmels(wav: np.ndarray, sr: int,
                   n_mels: int = 80, win_ms: int = 25, hop_ms: int = 10) -> np.ndarray:
    """Compute log-mel features shaped [T, n_mels]."""
    win_length = int(sr * (win_ms / 1000.0))
    hop_length = int(sr * (hop_ms / 1000.0))
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=win_length, hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )
    logmel = librosa.power_to_db(mel + 1e-10, ref=1.0)
    logmel = logmel.T.astype(np.float32)  # [T, n_mels]
    return logmel


class InferenceEngine:
    def __init__(self, ckpt_path: Path, spm_model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # ---- build model ----
        self.model = SpeechToTextTranslationModel(
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            embed_dim=config.EMBED_DIM,
            num_attn_heads=config.NUM_HEADS,
            tgt_vocab_size=config.TextParams.spm_vocab_size,
            d_ff=config.D_FF,
            dropout=config.DROPOUT,
            input_feat_dim=config.AudioParams.n_mels
        ).to(self.device).eval()

        # ---- load checkpoint weights ----
        self._load_checkpoint_into_model(self.model, Path(ckpt_path))

        # ---- sentencepiece tokenizer ----
        import sentencepiece as spm
        self.spm = spm.SentencePieceProcessor(model_file=str(spm_model_path))

    # ---------------------------------------------------------

    def _load_checkpoint_into_model(self, model: torch.nn.Module, ckpt_path: Path):
        ext = ckpt_path.suffix.lower()
        if ext == ".safetensors":
            if safe_load is None:
                raise RuntimeError("Install safetensors in your env:  pip install safetensors")
            # safetensors requires device as a string
            device_str = "cuda" if self.device.type == "cuda" else "cpu"
            state = safe_load(str(ckpt_path), device=device_str)
        else:
            state = torch.load(str(ckpt_path), map_location=self.device)

        if isinstance(state, dict):
            # common patterns
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]

        model.load_state_dict(state, strict=False)

    # ---------------------------------------------------------

    @torch.no_grad()
    def translate_audio(self, audio_path: str, max_len: int = 100) -> str:
        # 1) load wav -> log-mels -> tensor [1, T, 80]
        wav, _ = librosa.load(audio_path, sr=config.AudioParams.sr_target)
        feats = wav_to_logmels(
            wav, sr=config.AudioParams.sr_target,
            n_mels=config.AudioParams.n_mels,
            win_ms=config.AudioParams.win_ms,
            hop_ms=config.AudioParams.hop_ms
        )
        feats_t = torch.from_numpy(feats).unsqueeze(0).to(self.device)  # [1, T, 80]

        # 2) encoder (mimic model.forward() pre-encoder)
        #    feature_projection -> pos_encoder -> encoder_stack
        src = self.model.feature_projection(feats_t) * math.sqrt(self.model.config["embed_dim"])
        src = self.model.pos_encoder(src)
        memory = src
        for layer in self.model.encoder_stack:
            memory = layer(memory, None)  # no src mask for now
        # memory: [1, Tenc, embed_dim]

        # 3) greedy decode with the model's decoder + generator
        generated: List[int] = [BOS_ID]
        for _ in range(max_len):
            tgt = torch.tensor(generated, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, L]
            # causal mask + pad mask like training
            L = tgt.size(1)
            pad_mask = (tgt != 1).unsqueeze(1).unsqueeze(2)  # pad id assumed=1
            look_ahead = torch.triu(torch.ones((L, L), device=self.device), diagonal=1).bool()
            tgt_mask = pad_mask & ~look_ahead

            tgt_emb = self.model.tgt_embedding(tgt) * math.sqrt(self.model.config["embed_dim"])
            tgt_emb = self.model.pos_encoder(tgt_emb)

            dec_out = tgt_emb
            for layer in self.model.decoder_stack:
                dec_out = layer(dec_out, memory, tgt_mask, None)

            logits = self.model.generator(dec_out[:, -1, :])  # [1, vocab]
            next_id = int(torch.argmax(logits, dim=-1).item())

            generated.append(next_id)
            if next_id == EOS_ID:
                break

        # strip BOS/EOS if present
        ids = [i for i in generated if i not in (BOS_ID,)]
        if ids and ids[-1] == EOS_ID:
            ids = ids[:-1]

        # 4) detokenize with sentencepiece
        try:
            text = self.spm.decode(ids)
        except Exception:
            text = "<unexpected model output>"
        return text
