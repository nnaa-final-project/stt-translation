import torch
import pandas as pd
import numpy as np
import config
import gc

from torch.utils.data import Dataset, IterableDataset
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass


class PreprocessedDataset(IterableDataset):
    """
    Dataset that loads features in chunks to handle large datasets.
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
        self.chunk_size_gb = chunk_size_gb

        self.current_chunk_idx = -1
        self.feature_cache: Dict[int, np.ndarray] = {}
        self.chunk_ranges: List[List[int]] = []
        self.chunk_map: List[int] = []

        self._load_manifests()


        if subset_params and subset_params.use_subset and split == "train":
            self._apply_subset(subset_params)

        self._prepare_chunks()

    def _load_manifests(self):
        """Load and merge the feature and encoded text manifests."""
        features_manifest_path = self.base_dir / "manifest_features.csv"
        encoded_text_path = self.base_dir / "text_outputs" / "encoded" / f"{self.split}.csv"

        if not features_manifest_path.exists() or not encoded_text_path.exists():
            raise FileNotFoundError(
                f"Manifests not found. Please run the preprocessing step first. "
                f"Checked for: {features_manifest_path} and {encoded_text_path}"
            )

        df_features = pd.read_csv(features_manifest_path)
        df_encoded = pd.read_csv(encoded_text_path)
        df_features_split = df_features[df_features['split'] == self.split].copy()

        self.manifest = pd.merge(
            df_features_split, df_encoded, on="audio_path", how="inner"
        ).dropna(subset=['feat_npy', 'tgt_ids'])

        self._filter_existing_files()

    def _filter_existing_files(self):
        """Filter manifest to only include samples with existing feature files."""
        print(f"[{self.split}] Scanning available feature files...")
        available_npy_files = set(f.name for f in self.logmels_dir.glob('*.npy'))

        filename_to_global_idx = {row["feat_npy"].split('/')[-1]: idx for idx, row in self.manifest.iterrows()}

        valid_indices = [filename_to_global_idx[npy_file]
                         for npy_file in available_npy_files
                         if npy_file in filename_to_global_idx]

        if not valid_indices:
            raise FileNotFoundError("No matching feature files found between manifest and directory.")

        merged_manifest = self.manifest
        self.manifest = merged_manifest.loc[valid_indices].reset_index(drop=True)

        print(f"[{self.split}] Valid samples with existing features: {len(self.manifest)}/{len(merged_manifest)}")

    def _prepare_chunks(self):
        """Prepare chunks based on file sizes to stay within memory limits."""
        print(f"[{self.split}] Preparing memory-efficient chunks (target: {self.chunk_size_gb:.1f}GB)...")
        file_info = []
        total_size = 0

        for local_idx, row in self.manifest.iterrows():
            feature_file = row["feat_npy"].split('/')[-1]
            feature_path = self.logmels_dir / feature_file

            try:
                file_size = feature_path.stat().st_size
                file_info.append((local_idx, file_size))
                total_size += file_size
            except FileNotFoundError:
                continue

        print(f"[{self.split}] Total dataset size: {total_size / 1024 ** 3:.2f} GB")

        file_info.sort(key=lambda x: x[1], reverse=True)

        self.chunk_ranges.clear()
        self.chunk_map = [-1] * len(self.manifest)
        current_chunk_indices = []
        current_chunk_size = 0
        chunk_idx_counter = 0

        for local_idx, file_size in file_info:
            if current_chunk_size + file_size > self.chunk_size_bytes and current_chunk_indices:
                # Save current chunk and start new one
                self.chunk_ranges.append(current_chunk_indices)

                # Update chunk map for all indices in the finished chunk
                for idx in current_chunk_indices:
                    self.chunk_map[idx] = chunk_idx_counter

                chunk_idx_counter += 1
                current_chunk_indices = [local_idx]
                current_chunk_size = file_size
            else:
                current_chunk_indices.append(local_idx)
                current_chunk_size += file_size

        if current_chunk_indices:
            self.chunk_ranges.append(current_chunk_indices)
            for idx in current_chunk_indices:
                self.chunk_map[idx] = chunk_idx_counter

        print(f"[{self.split}] Created {len(self.chunk_ranges)} chunks.")

    def _load_chunk(self, chunk_idx: int):
        """Load a specific chunk into memory."""
        if chunk_idx == self.current_chunk_idx:
            return

        # Clear previous chunk and free memory
        self.feature_cache.clear()
        gc.collect()

        print(f"[{self.split}] Reloading chunk {chunk_idx + 1}/{len(self.chunk_ranges)} into memory...")

        chunk_indices = self.chunk_ranges[chunk_idx]
        loaded_count = 0

        for local_idx in chunk_indices:
            sample = self.manifest.iloc[local_idx]
            feature_file = sample["feat_npy"].split('/')[-1]
            feature_path = self.logmels_dir / feature_file

            try:
                features = np.load(feature_path).T
                self.feature_cache[local_idx] = features
                loaded_count += 1
            except Exception as e:
                print(f"[{self.split}] Failed to load {feature_file} at index {local_idx}: {e}")

        self.current_chunk_idx = chunk_idx
        print(f"[{self.split}] Loaded {loaded_count} features into memory")

    def _get_chunk_for_idx(self, local_idx: int) -> int:
        if 0 <= local_idx < len(self.chunk_map):
            return self.chunk_map[local_idx]
        raise IndexError(f"Local index {local_idx} out of range of chunk map")

    def __iter__(self):
        """
        Iterator that loops through chunks sequentially, loads the whole chunk
        into memory, and yields individual samples from that chunk.
        """

        self.current_chunk_idx = -1
        self.feature_cache.clear()
        gc.collect()

        for chunk_idx in range(len(self.chunk_ranges)):
            self._load_chunk(chunk_idx)
            chunk_indices = self.chunk_ranges[chunk_idx]

            for local_idx in chunk_indices:
                if local_idx not in self.feature_cache:
                    print(f"[{self.split}] ERROR: Sample {local_idx} missing from loaded chunk {chunk_idx}.")
                    continue

                features = self.feature_cache[local_idx]
                sample = self.manifest.iloc[local_idx]
                target_ids = list(map(int, sample["tgt_ids"].split()))

                yield {
                    "input_features": torch.from_numpy(features.copy()).float(),
                    "labels": torch.tensor(target_ids, dtype=torch.long),
                }

        self.feature_cache.clear()
        gc.collect()
        self.current_chunk_idx = -1  # Reset state for next epoch


    def __len__(self) -> int:
        return len(self.manifest)


    def _apply_subset(self, params):
        """Apply subsetting to the dataset."""
        original_size = len(self.manifest)
        print(f"[{self.split}] Applying subset: original size {original_size}")

        if params.subset_size:
            n_samples = min(params.subset_size, original_size)
            print(f"[{self.split}] Using fixed subset size: {n_samples}")
        else:
            n_samples = int(original_size * params.subset_fraction)
            print(f"[{self.split}] Using subset fraction: {params.subset_fraction}, resulting in {n_samples} samples.")

        if params.split_method == "random":
            self.manifest = self.manifest.sample(n=n_samples, random_state=params.random_seed)
        elif params.split_method == "first_n":
            self.manifest = self.manifest.head(n_samples)

        # Reset index
        self.manifest = self.manifest.reset_index(drop=True)

        print(f"[{self.split}] Final subset size: {len(self.manifest)}")


@dataclass
class PaddingDataCollator:
    """
    Pads features and labels to the same length in batches
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        inputs_padded = torch.nn.utils.rnn.pad_sequence(
            [f["input_features"] for f in input_features], batch_first=True, padding_value=0.0
        )
        attention_mask = (inputs_padded.sum(dim=-1) != 0).long()

        tokens_padded = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in label_features], batch_first=True, padding_value=1
        )
        labels = tokens_padded.clone()
        labels[labels == 1] = -100

        return {
            "input_features": inputs_padded,
            "attention_mask": attention_mask,
            "labels": labels,
        }