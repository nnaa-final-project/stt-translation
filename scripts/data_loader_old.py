import torch
import pandas as pd
import numpy as np
import config
import gc

from torch.utils.data import Dataset
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass


class PreprocessedDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset that loads features in chunks to handle large datasets.
    """


    def __init__(self, split: str, base_output_dir: Path, subset_params=None, chunk_size_gb=5):
        """
        Args:
            split (str): The dataset split to load ('train', 'dev', or 'test').
            base_output_dir (Path): The main output directory where artifacts are stored.
            chunk_size_gb (float): Maximum memory usage per chunk in GB.
        """
        self.split = split
        self.base_dir = base_output_dir
        self.features_dir = config.FEATURES_DIR
        self.logmels_dir = self.features_dir / "logmel"
        self.chunk_size_bytes = chunk_size_gb * 1024 ** 3

        # Current chunk info
        self.current_chunk_idx = -1
        self.feature_cache = {}
        self.chunk_ranges = []

        # Load manifests and prepare dataset
        self._load_manifests()
        self._prepare_chunks()

        # Use subset of train data if specified
        if subset_params and subset_params.use_subset and split == "train":
            self._apply_subset(subset_params)
            self._prepare_chunks()  # Recalculate chunks after subsetting


    def _load_manifests(self):
        """Load and merge the feature and encoded text manifests."""
        features_manifest_path = self.base_dir / "manifest_features.csv"
        encoded_text_path = self.base_dir / "text_outputs" / "encoded" / f"{self.split}.csv"

        if not features_manifest_path.exists() or not encoded_text_path.exists():
            raise FileNotFoundError(
                f"Manifests not found. Please run the preprocessing step first. "
                f"Checked for: {features_manifest_path} and {encoded_text_path}"
            )

        # Load and merge manifests
        df_features = pd.read_csv(features_manifest_path)
        df_encoded = pd.read_csv(encoded_text_path)
        df_features_split = df_features[df_features['split'] == self.split].copy()

        self.manifest = pd.merge(
            df_features_split, df_encoded, on="audio_path", how="inner"
        ).dropna(subset=['feat_npy', 'tgt_ids'])

        # Filter for existing files
        self._filter_existing_files()

    def _filter_existing_files(self):
        """Filter manifest to only include samples with existing feature files."""
        print("Scanning available feature files...")
        available_npy_files = set(f.name for f in self.logmels_dir.glob('*.npy'))
        print(f"Found {len(available_npy_files)} feature files in {self.logmels_dir}")

        if not available_npy_files:
            raise FileNotFoundError(
                f"No .npy feature files found in {self.logmels_dir}. "
                f"Please run the preprocessing step first."
            )

        # Create mapping for O(1) lookup
        filename_to_idx = {}
        for idx, row in self.manifest.iterrows():
            feature_file = row["feat_npy"].split('/')[-1]
            filename_to_idx[feature_file] = idx

        # Filter manifest
        valid_indices = [filename_to_idx[npy_file] for npy_file in available_npy_files
                         if npy_file in filename_to_idx]

        if not valid_indices:
            raise FileNotFoundError("No matching feature files found between manifest and directory.")

        merged_manifest = self.manifest
        self.manifest = merged_manifest.iloc[valid_indices].reset_index(drop=True)

        print(f"Original samples: {len(merged_manifest)}")
        print(f"Valid samples with existing features: {len(self.manifest)}")

    def _prepare_chunks(self):
        """Prepare chunks based on file sizes to stay within memory limits."""
        print("Preparing memory-efficient chunks...")

        # Get file sizes and sort by size for better packing
        file_info = []
        total_size = 0

        for idx, row in self.manifest.iterrows():
            feature_file = row["feat_npy"].split('/')[-1]
            feature_path = self.logmels_dir / feature_file
            try:
                file_size = feature_path.stat().st_size
                file_info.append((idx, file_size))
                total_size += file_size
            except FileNotFoundError:
                continue

        print(f"Total dataset size: {total_size / 1024 ** 3:.2f} GB")

        # Create chunks based on cumulative size
        self.chunk_ranges = []
        current_chunk = []
        current_chunk_size = 0

        for idx, file_size in file_info:
            if current_chunk_size + file_size > self.chunk_size_bytes and current_chunk:
                # Save current chunk and start new one
                self.chunk_ranges.append(current_chunk)
                current_chunk = [idx]
                current_chunk_size = file_size
            else:
                current_chunk.append(idx)
                current_chunk_size += file_size

        # Add final chunk
        if current_chunk:
            self.chunk_ranges.append(current_chunk)

        # Fix: Use self.chunk_size_bytes instead of undefined chunk_size_gb
        chunk_size_gb = self.chunk_size_bytes / 1024 ** 3
        print(f"Created {len(self.chunk_ranges)} chunks (target: {chunk_size_gb:.1f}GB each)")


    def _load_chunk(self, chunk_idx: int):
        """Load a specific chunk into memory."""
        if chunk_idx == self.current_chunk_idx:
            return

        # Clear previous chunk
        self.feature_cache.clear()
        gc.collect()

        print(f"Loading chunk {chunk_idx + 1}/{len(self.chunk_ranges)}...")

        chunk_indices = self.chunk_ranges[chunk_idx]
        loaded_count = 0

        for idx in chunk_indices:
            sample = self.manifest.iloc[idx]
            feature_file = sample["feat_npy"].split('/')[-1]
            feature_path = self.logmels_dir / feature_file

            try:
                features = np.load(feature_path).T
                self.feature_cache[idx] = features
                loaded_count += 1
            except Exception as e:
                print(f"Failed to load {feature_file}: {e}")

        self.current_chunk_idx = chunk_idx
        print(f"Loaded {loaded_count} features into memory")


    def _get_chunk_for_idx(self, idx: int) -> int:
        """Get which chunk contains the given index."""
        for chunk_idx, chunk_indices in enumerate(self.chunk_ranges):
            if idx in chunk_indices:
                return chunk_idx
        raise IndexError(f"Index {idx} not found in any chunk")


    def __len__(self) -> int:
        return len(self.manifest)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single data sample, loading chunks as needed."""
        # Determine which chunk contains this index
        chunk_idx = self._get_chunk_for_idx(idx)

        # Load chunk if not already loaded
        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)

        # Get features from cache or load individually
        if idx in self.feature_cache:
            features = self.feature_cache[idx]
        else:
            # Fallback: load individual file
            sample = self.manifest.iloc[idx]
            feature_file = sample["feat_npy"].split('/')[-1]
            feature_path = self.logmels_dir / feature_file
            features = np.load(feature_path).T

        # Get target IDs
        sample = self.manifest.iloc[idx]
        target_ids = list(map(int, sample["tgt_ids"].split()))

        return {
            "input_features": torch.from_numpy(features.copy()).float(),
            "labels": torch.tensor(target_ids, dtype=torch.long),
        }

    # Before code
    def _apply_subset(self, params):
        """Apply subsetting to the dataset."""
        original_size = len(self.manifest)
        print(f"Applying subset: original size {original_size}")

        if params.subset_size:
            print(f"Using fixed subset size: {params.subset_size}")
            n_samples = min(params.subset_size, original_size)
            print(f"Adjusted subset size to {n_samples} based on available data.")
        else:
            n_samples = int(original_size * params.subset_fraction)
            print(f"Using subset fraction: {params.subset_fraction}, resulting in {n_samples} samples.")

        if params.split_method == "random":
            self.manifest = self.manifest.sample(n=n_samples, random_state=params.random_seed)
        elif params.split_method == "first_n":
            self.manifest = self.manifest.head(n_samples)

        # Reset index
        self.manifest = self.manifest.reset_index(drop=True)

        print(f"Using subset of training data: {len(self.manifest)} samples out of {original_size}")


@dataclass
class PaddingDataCollator:
    """
    Pads features and labels to the same length in batches
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad  features - log-mels
        batch = {}
        inputs_padded = torch.nn.utils.rnn.pad_sequence(
            [f["input_features"] for f in input_features], batch_first=True, padding_value=0.0
        )
        batch["input_features"] = inputs_padded

        # Pad token IDs
        tokens_padded = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in label_features], batch_first=True, padding_value=1
        )
        batch["labels"] = tokens_padded

        return batch
