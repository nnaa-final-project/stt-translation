# ====================
# Data Manager for Audio-Text Preprocessing Pipeline from data_processor.py
# ====================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict
import os, re, json, hashlib, subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm
from config import AudioParams, TextParams

# audio libraries
import librosa
import soundfile as sf

# Choose regex library
try:
    import regex as re2

    _RE = re2
except ImportError:
    _RE = re

try:
    import sentencepiece as spm

    HAS_SPM = True
except ImportError:
    HAS_SPM = False

import warnings

# warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


# =====================
# Paths Class
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


CV_SPLITS_DEFAULT = ["validated.tsv", "train.tsv", "dev.tsv", "test.tsv"]


# =====================
# Audio Processor
# =====================
class AudioProcessor:
    def __init__(self, paths: Paths, a: AudioParams, splits: Iterable[str] = CV_SPLITS_DEFAULT):
        self.P = paths
        self.A = a
        self.splits = list(splits)
        # Derived paths using the central output directory
        self.artifacts_dir = self.P.out_dir
        self.metrics_csv = self.artifacts_dir / "merged_en_de_metrics.csv"
        self.merged_csv = self.artifacts_dir / "merged_en_de.csv"
        self.filtered_csv = self.artifacts_dir / "merged_en_de_filtered.csv"
        self.wav_manifest_csv = self.artifacts_dir / "manifest_wav.csv"
        self.seed_manifest_csv = self.artifacts_dir / "manifest_seed.csv"
        self.manifest_features_csv = self.artifacts_dir / "manifest_features.csv"
        self.cmvn_json = self.artifacts_dir / "cmvn_train.json"
        self.wav_dir = self.artifacts_dir / "wav"
        self.features_dir = self.artifacts_dir / "features" / "logmel"
        _ensure(self.artifacts_dir)
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
        cands = [c for c in covost_df.columns if
                 c.lower().startswith("translation") or c.endswith("_de") or c.lower() == "de"]
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
                print(
                    f"{file_name:<20}: matched {len(matched):>6} / {len(cv_df):>6} rows ({100 * len(matched) / max(len(cv_df), 1):.2f}%)")
                # Construct audio_path + select columns
                matched["audio_path"] = matched["path"].apply(lambda q: str((self.P.clips / q).resolve()))
                en_col = "sentence" if "sentence" in matched.columns else (
                    "text" if "text" in matched.columns else "en_text")
                matched = matched[["audio_path", en_col, translation_col]].rename(
                    columns={en_col: "en_text", translation_col: "de_text"})
                merged_dfs.append(matched)
            except Exception as e:
                print(f"{file_name:<20}: merge error - {e}")

        merged_df = pd.concat(merged_dfs, ignore_index=True) if merged_dfs else pd.DataFrame(
            columns=["audio_path", "en_text", "de_text"])
        merged_df.to_csv(self.merged_csv, index=False)
        print(f"💾 Total matched samples written to: {self.merged_csv}\n")
        return merged_df

    # ---------- 2. EDA & Filtering (Duration, SNR) ----------
    @staticmethod
    def safe_duration(path: str) -> Optional[float]:
        try:
            with sf.SoundFile(path) as f:
                return float(len(f) / f.samplerate)
        except Exception:
            return None

    @staticmethod
    def estimate_snr_librosa(filename: str, sr: int, top_db: int = 20) -> Optional[float]:
        try:
            y, fs = librosa.load(filename, sr=sr, mono=True)
            if y is None or y.size == 0: return None
            total_e = float(np.mean(y ** 2) + 1e-9)
            intervals = librosa.effects.split(y, top_db=top_db)
            signal_e = float(
                np.mean(np.concatenate([y[s:e] for s, e in intervals]) ** 2) + 1e-9) if intervals.size > 0 else 0.0
            noise_e = max(total_e - signal_e, 1e-9)
            return 10.0 * np.log10(signal_e / noise_e)
        except Exception:
            return None

    def compute_metrics_duration_snr(self, merged_csv: Optional[Path] = None) -> pd.DataFrame:
        df = pd.read_csv(merged_csv or self.merged_csv)
        df["duration_sec"] = [self.safe_duration(p) for p in tqdm(df["audio_path"], desc="Compute duration")]
        df["snr_db"] = [self.estimate_snr_librosa(p, self.A.sr_target) for p in
                        tqdm(df["audio_path"], desc="Estimate SNR")]
        df.to_csv(self.metrics_csv, index=False)
        print(f"💾 Saved metrics → {self.metrics_csv}\n")
        return df

    def filter_by_duration_snr(self, metrics_csv: Optional[Path] = None) -> pd.DataFrame:
        df = pd.read_csv(metrics_csv or self.metrics_csv)
        mask = (
                df["duration_sec"].between(self.A.min_dur, self.A.max_dur) &
                (df["snr_db"] >= self.A.snr_db_thresh)
        )
        filtered = df[mask].dropna(subset=["duration_sec", "snr_db"]).copy()
        filtered.to_csv(self.filtered_csv, index=False)
        print(f"Filtered {len(filtered)} / {len(df)} rows → {self.filtered_csv.name}")
        return filtered

    # ---------- 3. Speaker‑Disjoint Split ----------
    def attach_speaker_and_split(self, filtered_csv: Optional[Path] = None, split_ratio=(98, 1, 1)) -> pd.DataFrame:
        fdf = pd.read_csv(filtered_csv or self.filtered_csv)
        client_map = {}
        for file in self.splits:
            p = self.P.base / file
            if p.exists():
                t = pd.read_csv(p, sep="\t", usecols=["path", "client_id"])
                client_map.update(dict(zip(t.path, t.client_id)))
        fdf["path"] = fdf["audio_path"].map(lambda q: os.path.basename(str(q)))
        fdf["client_id"] = fdf["path"].map(client_map).fillna("unknown")
        fdf["split"] = fdf["client_id"].map(lambda cid: self.assign_split_by_speaker(cid, split_ratio))
        fdf.to_csv(self.seed_manifest_csv, index=False)
        print("💾 Saved speaker-disjoint manifest → manifest_seed.csv\n")
        return fdf

    @staticmethod
    def assign_split_by_speaker(client_id: str, ratio=(98, 1, 1)) -> str:
        bucket = int(hashlib.md5(client_id.encode("utf-8")).hexdigest(), 16) % 100
        train_r, dev_r, _ = ratio
        if bucket < train_r:
            return "train"
        elif bucket < train_r + dev_r:
            return "dev"
        else:
            return "test"

    # ---------- 4. Audio Standardization (FFmpeg) ----------
    def standardize_all_to_wav(self, seed_csv: Optional[Path] = None) -> pd.DataFrame:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        seed = pd.read_csv(seed_csv or self.seed_manifest_csv)
        paths = seed["audio_path"].tolist()
        results = [""] * len(paths)
        with ThreadPoolExecutor(max_workers=self.A.workers) as ex:
            futs = {ex.submit(self._ffmpeg_standardize, p): i for i, p in enumerate(paths)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Standardizing WAVs"):
                results[futs[fut]] = fut.result()
        wav_manifest = seed.copy()
        wav_manifest["audio_wav"] = [r.as_posix() if r else "" for r in results]
        wav_manifest.to_csv(self.wav_manifest_csv, index=False)
        print("💾 Saved WAV manifest → manifest_wav.csv\n")
        return wav_manifest

    def _ffmpeg_standardize(self, src_path: str) -> Optional[Path]:
        src, out_wav = Path(src_path), (self.wav_dir / Path(src_path).name).with_suffix(".wav")
        af = [f"silenceremove=start_periods=1:start_threshold={self.A.sil_start_db}dB",
              f"stop_periods=1:stop_threshold={self.A.sil_stop_db}dB"]
        cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src), "-ac", "1", "-ar", str(self.A.sr_target), "-sample_fmt",
               "s16", "-af", ",".join(af), str(out_wav)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out_wav
        except subprocess.CalledProcessError:
            return None

    # ---------- 5. Feature Extraction + Train-only CMVN ----------
    def extract_features_and_cmvn(self, wav_manifest_csv: Optional[Path] = None) -> pd.DataFrame:
        dfm = pd.read_csv(wav_manifest_csv or self.wav_manifest_csv)
        out_paths, shapes = [], []
        for p in tqdm(dfm["audio_wav"], desc="Extract log-mel feats"):
            F = self._logmel_from_wav(Path(p)) if p and isinstance(p, str) else None
            if F is None:
                out_paths.append("");
                shapes.append("");
                continue
            out_p = self.features_dir / (Path(p).stem + ".npy")
            np.save(out_p, F);
            out_paths.append(out_p.as_posix());
            shapes.append(f"{F.shape[0]}x{F.shape[1]}")
        dfm["feat_npy"], dfm["feat_shape"] = out_paths, shapes
        # CMVN
        train_feats = dfm.loc[dfm["split"] == "train", "feat_npy"].dropna()
        sum_feat, sum_sq, count = None, None, 0
        for p in tqdm(train_feats, desc="CMVN accumulate"):
            F = np.load(p)
            sum_feat = F.sum(axis=1) if sum_feat is None else sum_feat + F.sum(axis=1)
            sum_sq = (F ** 2).sum(axis=1) if sum_sq is None else sum_sq + (F ** 2).sum(axis=1)
            count += F.shape[1]
        mean = (sum_feat / count);
        var = (sum_sq / count - mean ** 2);
        std = np.sqrt(np.maximum(var, 1e-8))
        cmvn = {"mean": mean.tolist(), "std": std.tolist()}
        with open(self.cmvn_json, "w") as f:
            json.dump(cmvn, f)
        for p in tqdm(dfm["feat_npy"].dropna(), desc="Apply CMVN"): np.save(p,
                                                                            (np.load(p) - mean[:, None]) / std[:, None])
        dfm.to_csv(self.manifest_features_csv, index=False)
        print("💾 Saved CMVN stats and final feature manifest.\n")
        return dfm

    def _logmel_from_wav(self, path_wav: Path) -> Optional[np.ndarray]:
        try:
            y, sr = sf.read(str(path_wav))
            n_fft, hop = int(sr * self.A.win_ms / 1000), int(sr * self.A.hop_ms / 1000)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.A.n_mels, n_fft=n_fft, hop_length=hop)
            return np.log(np.maximum(S, 1e-10)).astype(np.float32)
        except Exception:
            return None


# =====================
# Text Processor
# =====================
class TextProcessor:
    def __init__(self, paths: Paths, t: TextParams):
        self.P = paths
        self.T = t
        self.artifacts_dir = self.P.out_dir
        self.text_out = self.artifacts_dir / "text_outputs"
        self.vocab_dir = self.text_out / "spm"
        self.cleaned_dir = self.text_out / "cleaned"
        self.encoded_dir = self.text_out / "encoded"
        for d in [self.text_out, self.vocab_dir, self.cleaned_dir, self.encoded_dir]: _ensure(d)
        self.model_prefix = (self.vocab_dir / f"spm_shared_{self.T.spm_model_type}_{self.T.spm_vocab_size}").as_posix()
        self.corpus_txt = self.vocab_dir / "spm_corpus.txt"


    def uclean(self, s: str) -> str:
        import html, unicodedata
        s = html.unescape(str(s if s is not None else ""))
        s = unicodedata.normalize("NFKC", s)
        s = _RE.compile(r"\[(noise|music|laughter)\]", _RE.IGNORECASE).sub(" ", s)
        return _RE.compile(r"\s+").sub(" ", s).strip()

    def clean_en(self, s: str) -> str:
        return self.uclean(s).lower()

    def clean_de(self, s: str) -> str:
        o = self.uclean(s)
        # optional: fix German quotes
        o = o.replace("\u201e", '"').replace("\u201c", '"').replace("\u201f", '"')
        return o

    def load_merge_pairs(self, audio_seed_csv: Optional[Path] = None) -> pd.DataFrame:
        # Use manifest_seed.csv from audio pipeline to ensure identical rows/order
        seed_csv = audio_seed_csv or (self.P.base / "manifest_seed.csv")
        if not seed_csv.exists():
            raise FileNotFoundError("Run AudioProcessor.attach_speaker_and_split() first to create manifest_seed.csv")
        seed = pd.read_csv(seed_csv)
        # Keep only columns we need
        return seed[["audio_path","en_text","de_text","split"]].copy()

    def train_shared_spm(self, merged_seed_csv: Path) -> None:
        if not HAS_SPM: return
        df = pd.read_csv(merged_seed_csv)
        with open(self.corpus_txt, "w", encoding="utf-8") as f:
            for _, r in tqdm(df[df["split"] == "train"].iterrows(), total=len(df[df["split"] == "train"]),
                             desc="Build SPM corpus"):
                f.write("<en> " + self.uclean(r["en_text"]).lower() + "\n")
                f.write("<de> " + self.uclean(r["de_text"]) + "\n")
        spm.SentencePieceTrainer.Train(
            f"--input={self.corpus_txt} --model_prefix={self.model_prefix} "
            f"--vocab_size={self.T.spm_vocab_size} --model_type={self.T.spm_model_type} "
            f"--user_defined_symbols=<en>,<de>"
        )
        print("Trained SPM:", self.model_prefix + ".model\n")

    def tokenize_and_filter(self, merged_seed_csv: Path) -> None:
        df = pd.read_csv(merged_seed_csv)
        sp = spm.SentencePieceProcessor(model_file=self.model_prefix + ".model") if HAS_SPM else None
        for split, sdf in df.groupby("split"):
            rows = []
            for _, r in tqdm(sdf.iterrows(), total=len(sdf), desc=f"Clean/filter {split}"):
                en_c, de_c = self.uclean(r["en_text"]).lower(), self.uclean(r["de_text"])
                if not en_c or not de_c: continue
                en_tok, de_tok = len(sp.encode(en_c)) if sp else len(en_c.split()), len(sp.encode(de_c)) if sp else len(
                    de_c.split())
                ratio = max(en_tok / max(de_tok, 1), de_tok / max(en_tok, 1))
                if (self.T.min_tok <= en_tok <= self.T.max_tok_en and
                        self.T.min_tok <= de_tok <= self.T.max_tok_de and
                        ratio <= self.T.max_len_ratio):
                    rows.append({"audio_path": r["audio_path"], "en_text": r["en_text"], "de_text": r["de_text"]})
            out_p = self.cleaned_dir / f"{split}_clean.tsv"
            pd.DataFrame(rows).to_csv(out_p, sep="\t", index=False)
            print(f"💾 Saved cleaned ({len(rows)} rows) → {out_p.name}\n")

    def _sp(self):
        sp = spm.SentencePieceProcessor()
        sp.load(self.model_prefix + ".model")
        return sp

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


def run_preprocessing(base_dir, covost_tsv, output_dir):
    """Sort of main function to run the local offline preprocessing pipeline.
       No need to run when you already have preprocessed data.
       Run in part by commenting out steps if needed.
    """
    paths = Paths.make(base=base_dir, covost=covost_tsv, out=output_dir)
    print(f"Using base data dir: {paths}")
    audio_params = AudioParams()
    text_params = TextParams()

    print("========== AUDIO PREPROCESSING ==========")
    ap = AudioProcessor(paths, audio_params)
    ap.load_and_link()
    ap.compute_metrics_duration_snr()
    ap.filter_by_duration_snr()
    ap.attach_speaker_and_split()
    ap.standardize_all_to_wav()
    ap.extract_features_and_cmvn()
    print("--- Audio Preprocessing Complete ---")

    print("\n========== TEXT PREPROCESSING ==========")
    tp = TextProcessor(paths, text_params)
    seed_manifest = tp.load_merge_pairs()  # ap.seed_manifest_csv
    # print(f"Using seed manifest: {seed_manifest}")
    tp.train_shared_spm(ap.seed_manifest_csv)
    tp.tokenize_and_filter(ap.seed_manifest_csv)
    tp.encode_splits(seed_manifest)
    print("--- Text Preprocessing Complete ---")
