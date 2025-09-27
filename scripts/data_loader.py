import torch
import pandas as pd
import numpy as np
import config

from torch.utils.data import Dataset
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass


class PreprocessedDataset(Dataset):
    """
    PyTorch-like Dataloader to load the local preprocessed data from
    data_processor.py | data_manger.py.
    """

    def __init__(self, split: str, base_output_dir: Path, subset_params=None):
        """
        Args:
            split (str): The dataset split to load ('train', 'dev', or 'test').
            base_output_dir (Path): The main output directory where artifacts are stored.
        """
        self.split = split
        self.base_dir = base_output_dir
        self.features_dir = config.FEATURES_DIR
        self.logmels_dir = self.features_dir / "logmel"

        # Load the feature manifest to get paths to .npy files
        features_manifest_path = self.base_dir / "manifest_features.csv"
        # Load the encoded text manifest to get token ID sequences
        encoded_text_path = self.base_dir / "text_outputs" / "encoded" / f"{split}.csv"

        if not features_manifest_path.exists() or not encoded_text_path.exists():
            raise FileNotFoundError(
                f"Manifests not found. Please run the preprocessing step first. "
                f"Checked for: {features_manifest_path} and {encoded_text_path}"
            )

        # Merge the two manifests on 'audio_path' to align features and labels
        df_features = pd.read_csv(features_manifest_path)
        df_encoded = pd.read_csv(encoded_text_path)

        # Filter features manifest for the correct split
        df_features_split = df_features[df_features['split'] == self.split].copy()

        self.manifest = pd.merge(
            df_features_split, df_encoded, on="audio_path", how="inner"
        ).dropna(subset=['feat_npy', 'tgt_ids'])

        # Use subset of train data if specified (to save time during development)
        if subset_params and subset_params.use_subset and split == "train":
            self._apply_subset(subset_params)


    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads a single data sample from disk.
        """
        sample = self.manifest.iloc[idx]

        # Load the log-mel feature matrix from the .npy file and get file name
        sample_feature_file = sample["feat_npy"].split('/')[-1]
        print(f"Loading features from: {sample_feature_file}")

        feature_path = self.logmels_dir / sample_feature_file

        features = np.load(feature_path).T  # Transpose to (Time, Freq)

        # The target IDs are stored as a space-separated string
        target_ids = list(map(int, sample["tgt_ids"].split()))

        return {
            "input_features": torch.from_numpy(features).float(),
            "labels": torch.tensor(target_ids, dtype=torch.long),
        }

    def _apply_subset(self, params):
        """Apply subsetting to the dataset."""
        original_size = len(self.manifest)

        if params.subset_size:
            n_samples = min(params.subset_size, original_size)
        else:
            n_samples = int(original_size * params.subset_fraction)

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