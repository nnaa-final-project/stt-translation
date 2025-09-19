"""
Data processors matching preprocess_audio.ipynb and preprocess_text.ipynb
exact pipeline (Common Voice v4 + CoVoST2 EN→DE)

This file mirrors the notebooks' steps 1–5 (audio) and 0–11 (text)
with the same intermediate artifacts and column names.

Outputs (same as notebooks):
- merged_en_de.csv
- merged_en_de_metrics.csv
- merged_en_de_filtered.csv
- split_map.csv, manifest_seed.csv, tiny_dev.csv
- manifest_wav.csv (standardized WAVs)
- features/logmel/*.npy + cmvn_train.json + manifest_features.csv
- text_outputs/spm/{model,vocab} + cleaned/*_en-de_clean.tsv + encoded/*.csv

Requirements:
 pandas numpy tqdm regex sentencepiece librosa soundfile matplotlib (optional) ftfy unicodedata2
 ffmpeg must be installed and on PATH.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import os, re, json, hashlib, subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

# parameters
from config import AudioParams, TextParams

# audio libs
import librosa
import soundfile as sf

# text libs
try:
    import regex as re2  # better unicode classes
    _RE = re2
except Exception:
    _RE = re

try:
    import sentencepiece as spm
    HAS_SPM = True
except Exception:
    HAS_SPM = False

import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# =====================
# Config
# =====================
@dataclass
class Paths:
    base: Path
    clips: Path
    covost_tsv: Path
    out_dir: Path

    @classmethod
    def make(cls, base: str | Path, covost: str | Path, out: Optional[str | Path] = None) -> "Paths":
        base = Path(base)
        return cls(
            base=base,
            clips=base / "clips",
            covost_tsv=Path(covost),
            out_dir=Path(out) if out else base,
        )


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


CV_SPLITS_DEFAULT = [
    "validated.tsv", "train.tsv", "dev.tsv", "test.tsv", "invalidated.tsv", "other.tsv"
]


# =====================
# Audio Processor (matches preprocess_audio.ipynb)
# =====================
class AudioProcessor:
    def __init__(self, paths: Paths, a: AudioParams, splits: Iterable[str] = CV_SPLITS_DEFAULT):
        self.P = paths
        self.A = a
        self.splits = list(splits)
        # derived dirs/files
        self.metrics_csv = self.P.base / "merged_en_de_metrics.csv"
        self.merged_csv = self.P.base / "merged_en_de.csv"
        self.filtered_csv = self.P.base / "merged_en_de_filtered.csv"
        self.wav_manifest_csv = self.P.base / "manifest_wav.csv"
        self.seed_manifest_csv = self.P.base / "manifest_seed.csv"
        self.split_map_csv = self.P.base / "split_map.csv"
        self.tiny_dev_csv = self.P.base / "tiny_dev.csv"
        self.features_dir = self.P.base / "processed" / "features" / "logmel"
        self.manifest_features_csv = self.P.base / "manifest_features.csv"
        self.cmvn_json = self.P.base / "cmvn_train.json"
        self.wav_dir = self.P.base / "processed" / "wav"
        _ensure(self.features_dir)
        _ensure(self.wav_dir)

    # ---------- 1. Load & Link ----------
    def load_and_link(self) -> pd.DataFrame:
        total_possible = 0
        merged_dfs: List[pd.DataFrame] = []

        # Report CV split sizes
        for file in self.splits:
            path = self.P.base / file
            try:
                df = pd.read_csv(path, sep="\t", dtype={"accent": str})
                print(f"{file:<20}: {len(df):>8} rows")
                total_possible += len(df)
            except Exception as e:
                print(f"{file:<20}: Error - {e}")
        print()

        # Load CoVoST translations
        covost_df = pd.read_csv(self.P.covost_tsv, sep="\t")
        assert "path" in covost_df.columns, "CoVoST must contain 'path'"

        # Detect translation column
        cands = [c for c in covost_df.columns if c.lower().startswith("translation") or c.endswith("_de") or c.lower()=="de"]
        translation_col = cands[0] if cands else "translation"
        if translation_col not in covost_df.columns:
            raise ValueError("Could not find translation column in CoVoST TSV.")

        # Merge each split with CoVoST
        for file_name in self.splits:
            path = self.P.base / file_name
            if not path.exists():
                continue
            try:
                cv_df = pd.read_csv(path, sep="\t", dtype={"accent": str})
                matched = pd.merge(cv_df, covost_df, on="path")
                print(f"{file_name:<20}: matched {len(matched):>6} / {len(cv_df):>6} rows ({100 * len(matched)/max(len(cv_df),1):.2f}%)")
                # Construct audio_path + select columns
                matched["audio_path"] = matched["path"].apply(lambda q: str((self.P.clips / q).resolve()))
                en_col = "sentence" if "sentence" in matched.columns else ("text" if "text" in matched.columns else "en_text")
                matched = matched[["audio_path", en_col, translation_col]].rename(columns={en_col:"en_text", translation_col:"de_text"})
                merged_dfs.append(matched)
            except Exception as e:
                print(f"{file_name:<20}: merge error - {e}")

        merged_df = pd.concat(merged_dfs, ignore_index=True) if merged_dfs else pd.DataFrame(columns=["audio_path","en_text","de_text"])
        merged_df.to_csv(self.merged_csv, index=False)
        print(f"💾 Total matched samples written to: {self.merged_csv}\n")
        return merged_df

    # ---------- 2. EDA & Filtering (Duration, SNR) ----------
    @staticmethod
    def safe_duration(path: str) -> Optional[float]:
        try:
            import soundfile as sf
            with sf.SoundFile(path) as f:
                return float(len(f) / f.samplerate)
        except Exception:
            return None

    @staticmethod
    def estimate_snr_librosa(filename: str, sr: int, top_db: int = 20) -> Optional[float]:
        try:
            # 1) if WAV, use soundfile
            y, fs = None, None
            try:
                y, fs = sf.read(filename, always_2d=False)
                if isinstance(y, np.ndarray) and y.ndim == 2:
                    y = y.mean(axis=1)
            except Exception:
                pass
            if y is None:
                # 2) if not, use librosa
                y, fs = librosa.load(filename, sr=sr, mono=True)
            if y is None or y.size == 0:
                return None
            if fs != sr:
                y = librosa.resample(np.asarray(y).astype(np.float32), orig_sr=fs, target_sr=sr)

            total_e = float(np.mean(y**2) + 1e-9)
            if total_e <= 1e-12:
                return None
            intervals = librosa.effects.split(y, top_db=top_db)
            if intervals.size > 0:
                voiced = np.concatenate([y[s:e] for s, e in intervals])
                signal_e = float(np.mean(voiced**2) + 1e-9)
            else:
                signal_e = 0.0
            noise_e = max(total_e - signal_e, 1e-9)
            return 10.0 * np.log10(signal_e / noise_e)
        except Exception:
            return None

    def compute_metrics_duration_snr(self, merged_csv: Optional[Path] = None) -> pd.DataFrame:
        df = pd.read_csv(merged_csv or self.merged_csv)
        if "duration_sec" not in df.columns:
            durs = []
            for ap in tqdm(df["audio_path"], desc="Compute duration (all)"):
                durs.append(self.safe_duration(str(ap)))
            df["duration_sec"] = durs
        if "snr_db" not in df.columns:
            snrs = []
            for ap in tqdm(df["audio_path"], desc="Estimate SNR (all)"):
                snrs.append(self.estimate_snr_librosa(str(ap), self.A.sr_target))
            df["snr_db"] = snrs
        df.to_csv(self.metrics_csv, index=False)
        print(f"💾 Saved metrics → {self.metrics_csv}\n")
        return df

    def filter_by_duration_snr(self, metrics_csv: Optional[Path] = None) -> pd.DataFrame:
        df = pd.read_csv(metrics_csv or self.metrics_csv)
        before = len(df)
        mask = (
            df["duration_sec"].between(self.A.min_dur, self.A.max_dur, inclusive="both") &
            (df["snr_db"] >= self.A.snr_db_thresh)
        )
        filtered = df[mask].dropna(subset=["duration_sec","snr_db"]).copy()
        filtered.to_csv(self.filtered_csv, index=False)
        print(f"Kept {len(filtered)} / {before} rows ({len(filtered)/max(before,1)*100:.2f}%) → {self.filtered_csv.name}")
        print("[Kept] Duration stats:\n", filtered["duration_sec"].describe())
        print("\n[Kept] SNR dB stats:\n", filtered["snr_db"].describe())
        print("")
        return filtered

    # ---------- 3. Speaker‑Disjoint Split ----------
    @staticmethod
    def assign_split_by_speaker(client_id: str, ratio=(98,1,1)) -> str:
        assert sum(ratio) == 100
        train_r, dev_r, test_r = ratio
        bucket = int(hashlib.md5(client_id.encode("utf-8")).hexdigest(), 16) % 100
        if bucket < train_r:
            return "train"
        elif bucket < train_r + dev_r:
            return "dev"
        else:
            return "test"

    def build_client_map(self) -> Dict[str, str]:
        client_map: Dict[str, str] = {}
        for file in self.splits:
            p = self.P.base / file
            if not p.exists():
                continue
            try:
                head = pd.read_csv(p, sep="\t", nrows=0)
                usecols = [c for c in ["path","client_id"] if c in head.columns]
                if not usecols:
                    continue
                t = pd.read_csv(p, sep="\t", usecols=usecols)
                if t.empty or "path" not in t.columns:
                    continue
                for r in t.itertuples(index=False):
                    path = getattr(r, "path")
                    client_id = getattr(r, "client_id") if hasattr(r, "client_id") else None
                    client_map[path] = client_id
            except Exception:
                continue
        return client_map

    def attach_speaker_and_split(self, filtered_csv: Optional[Path] = None, split_ratio=(98,1,1), tiny_dev_target: int = 200, max_per_spk_tiny: int = 5) -> pd.DataFrame:
        fdf = pd.read_csv(filtered_csv or self.filtered_csv)
        fdf["path"] = fdf["audio_path"].map(lambda q: os.path.basename(str(q)))
        client_map = self.build_client_map()
        fdf["client_id"] = fdf["path"].map(client_map).fillna("unknown_client")
        fdf["split"] = fdf["client_id"].astype(str).map(lambda cid: self.assign_split_by_speaker(cid, split_ratio))

        split_map = fdf[["client_id","split"]].drop_duplicates().sort_values("client_id")
        split_map.to_csv(self.split_map_csv, index=False)
        seed = fdf.copy()
        seed.to_csv(self.seed_manifest_csv, index=False)

        dev = seed[seed["split"]=="dev"].copy()
        tiny = dev.groupby("client_id", as_index=False).head(max_per_spk_tiny).head(tiny_dev_target)
        tiny.to_csv(self.tiny_dev_csv, index=False)
        print("💾 Saved: split_map.csv, manifest_seed.csv, tiny_dev.csv\n")
        return seed

    # ---------- 4. Audio Standardization (FFmpeg) ----------
    def _ffmpeg_standardize(self, src_path: str) -> Optional[Path]:
        src = Path(src_path)
        if not src.exists():
            return None
        out_wav = (self.wav_dir / src.name).with_suffix(".wav")
        _ensure(out_wav.parent)
        af = [
            "dynaudnorm=f=150:g=15",
            "highpass=f=40",
            "lowpass=f=7500",
            f"silenceremove=start_periods=1:start_threshold={self.A.sil_start_db}dB:start_silence={self.A.sil_pad_s}:stop_periods=1:stop_threshold={self.A.sil_stop_db}dB:stop_silence={self.A.sil_pad_s}",
        ]
        cmd = [
            "ffmpeg","-nostdin","-y","-i", str(src),
            "-ac","1","-ar", str(self.A.sr_target), "-sample_fmt","s16",
            "-af", ",".join(af),
            str(out_wav)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out_wav
        except subprocess.CalledProcessError:
            return None

    def standardize_all_to_wav(self, seed_csv: Optional[Path] = None) -> pd.DataFrame:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        seed = pd.read_csv(seed_csv or self.seed_manifest_csv)
        paths = seed["audio_path"].astype(str).tolist()
        results: List[Optional[Path]] = [None]*len(paths)
        with ThreadPoolExecutor(max_workers=self.A.workers) as ex:
            futs = {ex.submit(self._ffmpeg_standardize, p): i for i, p in enumerate(paths)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Standardizing WAVs"):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception:
                    results[i] = None
        wav_manifest = seed.copy()
        wav_manifest["audio_wav"] = [r.as_posix() if r else "" for r in results]
        wav_manifest.to_csv(self.wav_manifest_csv, index=False)
        print("💾 Saved: manifest_wav.csv\n")
        return wav_manifest

    # ---------- 5. Feature Extraction + Train-only CMVN ----------
    def _logmel_from_wav(self, path_wav: Path) -> Optional[np.ndarray]:
        try:
            y, sr = sf.read(str(path_wav), always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim == 2:
                y = y.mean(axis=1)
            if y is None or len(y) == 0:
                return None
            if sr != self.A.sr_target:
                y = librosa.resample(np.asarray(y).astype(np.float32), orig_sr=sr, target_sr=self.A.sr_target)
                sr = self.A.sr_target
            n_fft = int(sr * self.A.win_ms / 1000)
            hop = int(sr * self.A.hop_ms / 1000)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.A.n_mels,
                                               n_fft=n_fft, hop_length=hop, win_length=n_fft,
                                               center=True, power=2.0)
            logmel = np.log(np.maximum(S, 1e-10)).astype(np.float32)
            return logmel
        except Exception:
            return None

    def extract_features_and_cmvn(self, wav_manifest_csv: Optional[Path] = None) -> pd.DataFrame:
        dfm = pd.read_csv(wav_manifest_csv or self.wav_manifest_csv)
        out_dir = self.features_dir
        out_paths, shapes = [], []
        for _, row in tqdm(dfm.iterrows(), total=len(dfm), desc=f"Extract feats -> {out_dir.name}"):
            wav_p = Path(str(row["audio_wav"]))
            if not wav_p.exists():
                out_paths.append(""); shapes.append(""); continue
            F = self._logmel_from_wav(wav_p)
            if F is None:
                out_paths.append(""); shapes.append(""); continue
            out_path = out_dir / (wav_p.stem + ".npy")
            _ensure(out_dir)
            np.save(out_path, F)
            out_paths.append(out_path.as_posix())
            shapes.append(f"{F.shape[0]}x{F.shape[1]}")
        dfm["feat_npy"], dfm["feat_shape"] = out_paths, shapes

        # Train-only CMVN
        train_mask = dfm.get("split","train").astype(str).eq("train")
        train_feat_paths = dfm.loc[train_mask, "feat_npy"].replace("", np.nan).dropna().tolist()
        sum_feat = None; sum_sq = None; count = 0
        for pth in tqdm(train_feat_paths, desc="CMVN (train) accumulate"):
            F = np.load(pth)
            if sum_feat is None:
                sum_feat = F.sum(axis=1)
                sum_sq = (F**2).sum(axis=1)
            else:
                sum_feat += F.sum(axis=1)
                sum_sq += (F**2).sum(axis=1)
            count += F.shape[1]
        cmvn = {}
        if count > 0:
            mean = (sum_feat / count).astype(np.float32)
            var = (sum_sq / count - mean**2).astype(np.float32)
            std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
            cmvn = {"mean": mean.tolist(), "std": std.tolist()}
            with open(self.cmvn_json, "w") as f:
                json.dump(cmvn, f)
            print("💾 Saved: cmvn_train.json")
        if cmvn:
            mean = np.array(cmvn["mean"], dtype=np.float32)[:, None]
            std = np.array(cmvn["std"], dtype=np.float32)[:, None]
            for pth in tqdm(dfm["feat_npy"].replace("", np.nan).dropna().tolist(), desc="Apply CMVN"):
                F = np.load(pth)
                Fn = (F - mean) / std
                np.save(pth, Fn)

        dfm.to_csv(self.manifest_features_csv, index=False)
        print("💾 Saved: manifest_features.csv\n")
        return dfm


# =====================
# Text Processor (matches preprocess_text.ipynb)
# =====================
class TextProcessor:
    def __init__(self, paths: Paths, t: TextParams):
        self.P = paths
        self.T = t
        # dirs
        self.text_out = self.P.base / "text_outputs"
        self.vocab_dir = self.text_out / "spm"
        self.cleaned_dir = self.text_out / "cleaned"
        self.encoded_dir = self.text_out / "encoded"
        for d in [self.text_out, self.vocab_dir, self.cleaned_dir, self.encoded_dir]:
            _ensure(d)
        self.model_prefix = (self.vocab_dir / f"spm_shared_{self.T.spm_model_type}_{self.T.spm_vocab_size}").as_posix()
        self.corpus_txt = self.vocab_dir / "spm_corpus.txt"

    # ---------- Cleaning utilities ----------
    _RE_ZW = _RE.compile(r"[\p{Cf}]", _RE.UNICODE)
    _RE_WS = _RE.compile(r"\s+", _RE.UNICODE)
    _RE_NOISE = _RE.compile(r"\[(noise|music|laughter|unk)\]|<unk>|<noise>|<music>|<laughter>", _RE.IGNORECASE)
    _RE_MULTI_PUNCT = _RE.compile(r"([!?.,;:])\1{2,}")

    @staticmethod
    def _html_unescape(s: str) -> str:
        import html
        return html.unescape(s)

    def uclean(self, s: str) -> str:
        if s is None:
            return ""
        out = self._html_unescape(str(s))
        try:
            import unicodedata2 as ud
        except Exception:
            import unicodedata as ud
        out = ud.normalize("NFKC", out)
        out = self._RE_NOISE.sub(" ", out)
        out = self._RE_ZW.sub("", out)
        out = self._RE_WS.sub(" ", out)
        out = self._RE_MULTI_PUNCT.sub(r"\1\1", out)
        return out.strip()

    def clean_en(self, s: str) -> str:
        return self.uclean(s).lower()

    def clean_de(self, s: str) -> str:
        o = self.uclean(s)
        # optional: fix German quotes
        o = o.replace("\u201e", '"').replace("\u201c", '"').replace("\u201f", '"')
        return o

    # ---------- Load & Merge (text view) ----------
    def load_merge_pairs(self, audio_seed_csv: Optional[Path] = None) -> pd.DataFrame:
        # Use manifest_seed.csv from audio pipeline to ensure identical rows/order
        seed_csv = audio_seed_csv or (self.P.base / "manifest_seed.csv")
        if not seed_csv.exists():
            raise FileNotFoundError("Run AudioProcessor.attach_speaker_and_split() first to create manifest_seed.csv")
        seed = pd.read_csv(seed_csv)
        # Keep only columns we need
        return seed[["audio_path","en_text","de_text","split"]].copy()

    # ---------- 6) Train shared SentencePiece (train-only) ----------
    def train_shared_spm(self, merged_seed: Optional[pd.DataFrame] = None) -> Tuple[str, str]:
        if not HAS_SPM:
            print("[WARN] sentencepiece not installed; skipping training")
            return self.model_prefix + ".model", self.model_prefix + ".vocab"
        df = merged_seed if merged_seed is not None else self.load_merge_pairs()
        train = df[df["split"]=="train"].copy()
        # build corpus with language tags
        with open(self.corpus_txt, "w", encoding="utf-8") as f:
            for _, r in tqdm(train.iterrows(), total=len(train), desc="Build SPM corpus"):
                f.write("<en> " + self.clean_en(r["en_text"]) + "\n")
                f.write("<de> " + self.clean_de(r["de_text"]) + "\n")
        spm.SentencePieceTrainer.Train(
            input=self.corpus_txt.as_posix(),
            model_prefix=self.model_prefix,
            vocab_size=self.T.spm_vocab_size,
            model_type=self.T.spm_model_type,
            user_defined_symbols=["<en>","<de>"]
        )
        print("Trained SPM:", self.model_prefix + ".model\n")
        return self.model_prefix + ".model", self.model_prefix + ".vocab"

    def _sp(self):
        sp = spm.SentencePieceProcessor()
        sp.load(self.model_prefix + ".model")
        return sp

    # ---------- 7) Tokenize & Filter by length/ratio, save cleaned TSVs ----------
    def tokenize_and_filter(self, df_in: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
        df = df_in if df_in is not None else self.load_merge_pairs()
        sp = self._sp() if HAS_SPM and (Path(self.model_prefix + ".model").exists()) else None
        out_paths: Dict[str, Path] = {}
        for split, sdf in df.groupby("split"):
            rows = []
            for _, r in tqdm(sdf.iterrows(), total=len(sdf), desc=f"Clean/measure {split}"):
                en_c = self.clean_en(r["en_text"])
                de_c = self.clean_de(r["de_text"])
                if sp is not None:
                    en_ids = sp.encode(en_c, out_type=int)
                    de_ids = sp.encode(de_c, out_type=int)
                else:
                    en_ids = en_c.split()
                    de_ids = de_c.split()
                en_tok = len(en_ids); de_tok = len(de_ids)
                en_char = len(en_c); de_char = len(de_c)
                ratio = max(en_tok/max(de_tok,1), de_tok/max(en_tok,1))
                keep = (
                    (self.T.min_tok <= en_tok <= self.T.max_tok_en) and
                    (self.T.min_tok <= de_tok <= self.T.max_tok_de) and
                    (en_char <= self.T.max_char_en) and (de_char <= self.T.max_char_de) and
                    (ratio <= self.T.max_len_ratio)
                )
                if keep:
                    rows.append({
                        "audio_path": r["audio_path"],
                        "en_text": r["en_text"],
                        "de_text": r["de_text"],
                        "en_tok": en_tok, "de_tok": de_tok, "len_ratio": ratio,
                    })
            out = pd.DataFrame(rows)
            out_p = self.cleaned_dir / f"{split}_en-de_clean.tsv"
            out.to_csv(out_p, sep="\t", index=False)
            out_paths[str(split)] = out_p
            print(f"💾 Saved cleaned → {out_p}\n")
        return out_paths

    # ---------- 10) Encode each split to ID sequences ----------
    def encode_splits(self, df_in: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
        if not HAS_SPM:
            print("[WARN] sentencepiece not installed; skipping encoding")
            return {}
        sp = self._sp()
        df = df_in if df_in is not None else self.load_merge_pairs()
        out: Dict[str, Path] = {}
        for split, sdf in df.groupby("split"):
            rows = []
            for _, r in tqdm(sdf.iterrows(), total=len(sdf), desc=f"Encode {split}"):
                en_c = self.clean_en(r["en_text"])
                de_c = self.clean_de(r["de_text"])
                src_ids = sp.encode("<en> " + en_c, out_type=int)
                tgt_ids = sp.encode("<de> " + de_c, out_type=int)
                rows.append({
                    "audio_path": r["audio_path"],
                    "en_clean": en_c,
                    "de_clean": de_c,
                    "src_ids": " ".join(map(str, src_ids)),
                    "tgt_ids": " ".join(map(str, tgt_ids)),
                    "src_len_tok": len(src_ids),
                    "tgt_len_tok": len(tgt_ids),
                })
            out_df = pd.DataFrame(rows)
            out_p = self.encoded_dir / f"{split}.csv"
            out_df.to_csv(out_p, index=False)
            out[str(split)] = out_p
            print(f"💾 Saved encoded → {out_p}\n")
        return out


# =====================
# Orchestration example
# =====================
if __name__ == "__main__":
    # ---- set your paths here (Windows-style example) ----
    P = Paths.make(
        base="../data",
        covost="../data/covost_v2.en_de.tsv",
    )
    A = AudioParams()
    T = TextParams()

    # AUDIO
    print("========== AUDIO PREPROCESSING ==========")
    ap = AudioProcessor(P, A)
    merged = ap.load_and_link()
    metrics = ap.compute_metrics_duration_snr()
    filtered = ap.filter_by_duration_snr()
    seed = ap.attach_speaker_and_split(filtered_csv=ap.filtered_csv, split_ratio=(98,1,1), tiny_dev_target=200, max_per_spk_tiny=5)
    wav_manifest = ap.standardize_all_to_wav()
    feats = ap.extract_features_and_cmvn()

    # TEXT
    print("========== TEXT PREPROCESSING ==========")
    tp = TextProcessor(P, T)
    seed_pairs = tp.load_merge_pairs()
    tp.train_shared_spm(seed_pairs)
    tp.tokenize_and_filter(seed_pairs)
    tp.encode_splits(seed_pairs)