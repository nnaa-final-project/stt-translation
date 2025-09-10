import argparse
from pathlib import Path

import config
from scripts.data_manager import run_preprocessing
from scripts.trainer import TrainingManager
from playground.inference_engine import InferenceEngine


def find_best_checkpoint(output_dir: Path) -> Path:
    """Finds the best checkpoint directory saved by the Trainer."""
    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")
    return max(checkpoints, key=lambda d: int(d.name.split('-')[1]))


def main():
    """Main function to run preprocessing, training, or inference based on user input."""

    # Command line argument parser setup
    parser = argparse.ArgumentParser(description="Speech-to-Text Translation Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "train", "infer"],
        help="Select mode: 'preprocess' data, 'train' model, or 'infer' with trained model."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to audio file to be translated (required for 'infer' mode)."
    )
    args = parser.parse_args()

    if args.mode == "preprocess":
        print("========= In Preprocessing Mode =========")
        run_preprocessing(
            base_dir=config.COMMON_VOICE_BASE_DATA_DIR,
            covost_tsv=config.COVOST_TSV_PATH,
            output_dir=config.OUTPUT_DIR
        )
        print("========= Preprocessing Mode Done! =========")

    elif args.mode == "train":
        print("========= In Training Mode ========")
        training_manager = TrainingManager()
        training_manager.train()
        print("========= Training Mode Done! =========")

    elif args.mode == "infer":
        # Inference Mode should be added here
        pass


if __name__ == "__main__":
    main()
