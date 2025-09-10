from datetime import time, datetime

from transformers import Trainer, TrainingArguments, AutoConfig
import evaluate
import json
import os
import sentencepiece as spm

import config
from scripts.encoder_decoder_transformer import SpeechToTextTranslationModel
from scripts.data_loader import PreprocessedDataset, PaddingDataCollator


class TrainingManager:
    def __init__(self):
        self.train_dataset = PreprocessedDataset(
            split="train",
            base_output_dir=config.OUTPUT_DIR,
            subset_params=config.DATASET_PARAMS
        )
        self.eval_dataset = PreprocessedDataset(split="dev", base_output_dir=config.OUTPUT_DIR)

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
        training_args = TrainingArguments(
            output_dir=str(config.TRAINING_OUTPUT_DIR),
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            eval_strategy=config.EVAL_STRATEGY,
            num_train_epochs=config.NUM_TRAIN_EPOCHS,
            fp16=config.FP16,
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
            label_names=["labels"]
        )

        data_collator = PaddingDataCollator()

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
        with open(config.TRAINING_OUTPUT_DIR / f"config_{datetime.now()}.json", "w") as f:
            json.dump(self.model.config, f)

        print("======= Training =======")
        trainer.train()
        print("======= Training done, saving model =======")
        trainer.save_model()
        print(f"Model saved in: {config.TRAINING_OUTPUT_DIR}")

    def _metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions.argmax(-1)
        labels_ids[labels_ids == -100] = self.sentence_piece.pad_id()
        pred_str = self.sentence_piece.decode(pred_ids.tolist())
        label_str = self.sentence_piece.decode(labels_ids.tolist())
        bleu_metric = evaluate.load("sacrebleu")
        result = bleu_metric.compute(predictions=pred_str, references=[[l] for l in label_str])
        return {"bleu": result["score"]}
