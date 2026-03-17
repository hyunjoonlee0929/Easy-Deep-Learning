"""Lightweight LLM fine-tuning with LoRA (Hugging Face Transformers + PEFT)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import pandas as pd

from Easy_Deep_Learning.core.experiment_tracker import ExperimentTracker


@dataclass
class LLMFineTuneResult:
    run_id: str
    run_path: Path
    metrics: dict[str, float]


def _load_prompt_dataset(path: Path, prompt_col: str, completion_col: str) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        prompts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(f"{row.get(prompt_col, '')}{row.get(completion_col, '')}")
        return prompts

    df = pd.read_csv(path)
    if prompt_col not in df.columns or completion_col not in df.columns:
        raise ValueError("Prompt/completion columns not found.")
    return (df[prompt_col].astype(str) + df[completion_col].astype(str)).tolist()


def finetune_llm_lora(
    data_path: Path,
    model_name: str,
    prompt_column: str,
    completion_column: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    max_length: int = 512,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    reuse_if_exists: bool = True,
) -> LLMFineTuneResult:
    """Fine-tune a causal LM with LoRA on prompt+completion text."""
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as exc:
        raise RuntimeError("transformers, datasets, torch, and peft are required for LLM fine-tuning.") from exc

    tracker = ExperimentTracker(base_dir=Path("runs"))
    data_hash = tracker.file_hash(data_path)
    metadata = {
        "data_hash": data_hash,
        "model_name": model_name,
        "prompt_column": prompt_column,
        "completion_column": completion_column,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "max_length": max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }
    if reuse_if_exists:
        existing = tracker.find_matching_run("llm_finetune", metadata)
        if existing:
            run_path = Path("runs") / existing
            metrics_path = run_path / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            return LLMFineTuneResult(run_id=existing, run_path=run_path, metrics=metrics)

    texts = _load_prompt_dataset(data_path, prompt_column, completion_column)
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.train_test_split(test_size=0.1, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def data_collator(features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=str(Path("runs") / "hf_llm_finetune"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=seed,
        report_to=[],
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    run_id, run_path = tracker.create_run(model_type="llm_finetune")
    tracker.save_json(run_path / "metrics.json", {k: float(v) for k, v in eval_metrics.items()})
    tracker.save_json(run_path / "model_info.json", {"model_type": "llm_finetune", "model_name": model_name})
    tracker.save_json(run_path / "run_metadata.json", {"model_type": "llm_finetune", **metadata})

    model.save_pretrained(run_path / "adapter")
    tokenizer.save_pretrained(run_path / "tokenizer")

    return LLMFineTuneResult(
        run_id=run_id,
        run_path=run_path,
        metrics={k: float(v) for k, v in eval_metrics.items()},
    )
