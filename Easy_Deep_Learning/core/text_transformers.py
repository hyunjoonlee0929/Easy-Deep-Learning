"""Transformer-based text classification (Hugging Face)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd


@dataclass
class TextRunResult:
    run_id: str
    run_path: Path
    metrics: dict[str, float]


def _load_text_dataset(path: Path, text_col: str, label_col: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError("text/label columns not found in CSV.")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    return texts, labels


def train_text_transformer(
    data_path: Path,
    text_column: str,
    label_column: str,
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    run_type: str = "text_transformer",
    reuse_if_exists: bool = True,
) -> TextRunResult:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from sklearn.metrics import accuracy_score, f1_score

    from Easy_Deep_Learning.core.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(base_dir=Path("runs"))
    data_hash = tracker.file_hash(data_path)
    metadata = {
        "data_hash": data_hash,
        "text_column": text_column,
        "label_column": label_column,
        "model_name": model_name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
    }
    if reuse_if_exists:
        existing = tracker.find_matching_run(run_type, metadata)
        if existing:
            run_path = Path("runs") / existing
            metrics_path = run_path / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            return TextRunResult(run_id=existing, run_path=run_path, metrics={"accuracy": float(metrics.get("eval_accuracy", 0.0)), "f1_weighted": float(metrics.get("eval_f1_weighted", 0.0))})

    texts, labels = _load_text_dataset(data_path, text_column, label_column)
    label_set = sorted(list(set(labels)))
    label_to_id = {l: i for i, l in enumerate(label_set)}
    y = [label_to_id[l] for l in labels]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_set))

    dataset = Dataset.from_dict({"text": texts, "label": y})
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        }

    args = TrainingArguments(
        output_dir=str(Path("runs") / "hf_text"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    run_id, run_path = tracker.create_run(model_type=run_type)
    tracker.save_json(run_path / "metrics.json", {k: float(v) for k, v in eval_metrics.items()})
    tracker.save_json(run_path / "model_info.json", {"model_type": run_type, "model_name": model_name})
    tracker.save_json(
        run_path / "run_metadata.json",
        {
            "model_type": run_type,
            **metadata,
        },
    )

    model.save_pretrained(run_path / "model")
    tokenizer.save_pretrained(run_path / "tokenizer")
    tracker.save_json(run_path / "labels.json", {"labels": label_set})

    return TextRunResult(run_id=run_id, run_path=run_path, metrics={"accuracy": float(eval_metrics.get("eval_accuracy", 0.0)), "f1_weighted": float(eval_metrics.get("eval_f1_weighted", 0.0))})
