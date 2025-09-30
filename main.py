import argparse
import os
from pathlib import Path

import psutil

import config
from scripts.data_manager import run_preprocessing
from scripts.trainer import TrainingManager
from scripts.inference import InferenceEngine  # <- keep Anthony's inference engine


# ---------- tiny helper so --ckpt can be a folder or file ----------
def _pick_checkpoint_file(p: Path) -> Path:
    """
    Accepts a file OR a checkpoint folder. Returns the actual model file.
    Looks for model.safetensors or pytorch_model.bin inside a folder.
    """
    p = p.expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            q = p / name
            if q.exists():
                return q
    raise FileNotFoundError(
        f"No checkpoint file found at {p} (looked for model.safetensors / pytorch_model.bin)"
    )
# -------------------------------------------------------------------


def find_best_checkpoint(output_dir: Path) -> Path:
    """Finds the best/latest checkpoint directory saved by the Trainer."""
    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")
    return max(checkpoints, key=lambda d: int(d.name.split("-")[1]))


def print_memory_usage():
    """Memory usage monitoring."""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024**3
    print(f"Current memory usage: {memory_gb:.2f} GB")


def main():
    """Run preprocessing, training, or inference based on user input."""

    parser = argparse.ArgumentParser(description="Speech-to-Text Translation Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "train", "infer"],
        help="Select mode: 'preprocess' data, 'train' model, or 'infer' with trained model.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to audio file to be translated (required for 'infer' mode).",
    )

    # ---- minimal additions: two optional flags for inference paths ----
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to checkpoint FILE or checkpoint FOLDER (we auto-pick model.safetensors/pytorch_model.bin).",
    )
    parser.add_argument(
        "--spm",
        type=str,
        help="Path to SentencePiece .model file.",
    )
    # ------------------------------------------------------------------

    args = parser.parse_args()

    if args.mode == "preprocess":
        print("========= In Preprocessing Mode =========")
        run_preprocessing(
            base_dir=config.COMMON_VOICE_BASE_DATA_DIR,
            covost_tsv=config.COVOST_TSV_PATH,
            output_dir=config.OUTPUT_DIR,
        )
        print("========= Preprocessing Mode Done! =========")

    elif args.mode == "train":
        print("========= In Training Mode =========")
        print("Starting chunked dataset training...")
        print_memory_usage()

        training_manager = TrainingManager()
        print_memory_usage()

        training_manager.train()
        print_memory_usage()
        print("========= Training Mode Done! =========")

    elif args.mode == "infer":
        if not args.audio_path:
            raise ValueError("--audio_path is required for inference mode.")

        print("========= Starting Inference Mode =========")

        # resolve checkpoint path
        if args.ckpt:
            ckpt_path = _pick_checkpoint_file(Path(args.ckpt))
        else:
            # fallback to latest/best checkpoint in your training output dir
            best_dir = find_best_checkpoint(config.TRAINING_OUTPUT_DIR)
            ckpt_path = _pick_checkpoint_file(best_dir)

        # resolve sentencepiece model path
        if args.spm:
            spm_path = Path(args.spm).expanduser()
        else:
            # if you keep a default in config, you could use it here.
            # otherwise, set your local default path below:
            spm_path = Path(r"C:\data\text_outputs\spm\spm_shared_unigram_16000.model").expanduser()

        if not spm_path.exists():
            raise FileNotFoundError(f"SPM model not found: {spm_path}")

        print(f"Checkpoint : {ckpt_path}")
        print(f"SPM model  : {spm_path}")

        engine = InferenceEngine(ckpt_path, spm_path)
        translation = engine.translate_audio(args.audio_path)

        print("\n" + "=" * 50)
        print("TRANSLATION RESULT")
        print("=" * 50)
        print(f"Input Audio: {args.audio_path}")
        print(f"German Translation: {translation}")
        print("=" * 50)


if __name__ == "__main__":
    main()
