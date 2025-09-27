import evaluate
import json
import torch
import numpy as np
import os
import time
import sentencepiece as spm
from datasets import Dataset

import config


from transformers import Trainer, TrainingArguments, AutoConfig
from scripts.encoder_decoder_transformer import SpeechToTextTranslationModel
from scripts.data_loader import PreprocessedDataset, PaddingDataCollator
#from scripts.noshuffle_trainer import NoShuffleTrainer
#from torch.utils.data import SequentialSampler, DataLoader
from typing import Optional


class TrainingManager:
    def __init__(self):
        print("Initializing Training Manager with memory-efficient datasets...")

        # Create chunked datasets with memory management
        # Ensure config.CHUNK_SIZE_GB is defined in your config file (e.g., 5.0)
        self.train_dataset = PreprocessedDataset(
            split="train",
            base_output_dir=config.OUTPUT_DIR,
            subset_params=config.DATASET_PARAMS,
            chunk_size_gb=config.CHUNK_SIZE_GB  # Use configured chunk size (e.g., 5GB)
        )

        self.eval_dataset = PreprocessedDataset(
            split="dev",
            base_output_dir=config.OUTPUT_DIR,
            chunk_size_gb=1.0  # Smaller chunks for eval dataset, e.g., 1GB
        )


        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Eval dataset: {len(self.eval_dataset)} samples")

        # Load the preprocessed SentencePiece model
        spm_path = config.OUTPUT_DIR / "text_outputs" / "spm" / f"spm_shared_{config.TextParams.spm_model_type}_{config.TextParams.spm_vocab_size}.model"
        self.sentence_piece = spm.SentencePieceProcessor(model_file=str(spm_path))

        self.model = SpeechToTextTranslationModel(
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            embed_dim=config.EMBED_DIM,
            num_attn_heads=config.NUM_HEADS,
            tgt_vocab_size=self.sentence_piece.get_piece_size(),
            d_ff=config.D_FF,
            dropout=config.DROPOUT,
            input_feat_dim=config.AudioParams.n_mels
        )



    def train(self):
        print("Setting up training arguments...")

        training_args = TrainingArguments(
            output_dir=str(config.TRAINING_OUTPUT_DIR),
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            eval_strategy=config.EVAL_STRATEGY,
            num_train_epochs=config.NUM_TRAIN_EPOCHS,
            fp16=config.FP16,  # Use float16 for memory efficiency (requires MPS/GPU)
            #bf16=True,  # Use bfloat16 if supported (MPS on Apple Silicon)
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            warmup_steps=config.WARMUP_STEPS,
            save_steps=config.SAVE_STEPS,
            eval_steps=config.EVAL_STEPS,
            logging_steps=config.LOGGING_STEPS,
            save_total_limit=config.SAVE_TOTAL_LIMIT,
            load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
            greater_is_better=config.GREATER_IS_BETTER,
            remove_unused_columns=False,
            label_names=["labels"],
            # Memory optimization settings
            # Setting workers to 0 is generally best practice for Apple Silicon MPS
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            # Use MPS device if available (PyTorch handles this via the default 'cuda' argument if MPS is active)
            # You might need to explicitly set device in Trainer constructor for older PyTorch versions.
            # torch_dtype=torch.float16 # You may set this if needed, but fp16=True usually handles it.
            #shuffle=False,  # Disable shuffling to reduce memory overhead (especially for large datasets
        )

        data_collator = PaddingDataCollator()


        print("Creating trainer with chunked datasets...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self._metrics,
            data_collator=data_collator,
        )

        # Manually save the custom model config
        config.TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config_filename = f"config_{int(time.time())}.json"

        # The model config should be saved using the model's built-in save or manually using dict
        # Assuming self.model.config is a dictionary or has a 'to_dict' method
        try:
            model_config_dict = self.model.config.to_dict()
        except AttributeError:
            model_config_dict = self.model.config  # Assume it's already a dict

        with open(config.TRAINING_OUTPUT_DIR / config_filename, "w") as f:
            json.dump(model_config_dict, f)

        print(f"Saved model config: {config_filename}")

        print("======= Starting Memory-Efficient Training =======")
        print(f"Training with {config.CHUNK_SIZE_GB}GB memory chunks")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Number of epochs: {config.NUM_TRAIN_EPOCHS}")
        print(f"FP16 enabled: {config.FP16}")

        trainer.train()

        print("======= Training completed, saving model =======")
        trainer.save_model()
        print(f"Model saved in: {config.TRAINING_OUTPUT_DIR}")

    def _metrics(self, pred):
        # Decode strategy: ignore tokens marked as -100 by the data collator
        labels_ids = pred.label_ids
        pred_ids = pred.predictions.argmax(-1)

        if isinstance(labels_ids, np.ndarray):
            # This handles the case where it comes as a NumPy array (the common MPS behavior)
            labels_ids = torch.from_numpy(labels_ids)
        if isinstance(pred_ids, np.ndarray):
            pred_ids = torch.from_numpy(pred_ids)


        # Replace -100 in labels with the SentencePiece PAD ID for decoding
        # labels_ids[labels_ids == -100] = self.sentence_piece.pad_id()

        labels_cpu_tensor = labels_ids.detach().cpu()
        preds_cpu_tensor = pred_ids.detach().cpu()

        # Replace -100 loss tokens with the actual PAD_ID
        pad_id = self.sentence_piece.pad_id()
        labels_cpu_tensor[labels_cpu_tensor == -100] = pad_id

        labels_np = labels_cpu_tensor.numpy().astype(np.int32)
        preds_np = preds_cpu_tensor.numpy().astype(np.int32)

        labels_np = np.clip(labels_np, 0, config.TextParams.spm_vocab_size - 1)
        preds_np = np.clip(preds_np, 0, config.TextParams.spm_vocab_size - 1)


        # Decode and compute BLEU score
        pred_str = self.sentence_piece.decode(preds_np.tolist())
        label_str = self.sentence_piece.decode(labels_np.tolist())

        bleu_metric = evaluate.load("sacrebleu")
        # sacrebleu expects a list of reference lists
        result = bleu_metric.compute(predictions=pred_str, references=[[l] for l in label_str])

        return {"bleu": result["score"]}

