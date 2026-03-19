"""Easy fine-tuning helpers for images and text."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from Easy_Deep_Learning.core.experiment_tracker import ExperimentTracker
from Easy_Deep_Learning.core.mlops import finalize_run_tracking
from Easy_Deep_Learning.core.security import ensure_model_download_allowed
from Easy_Deep_Learning.core.torch_workflows import _build_image_model, _save_torch_model, _require_torch
from Easy_Deep_Learning.core.text_transformers import train_text_transformer, TextRunResult


@dataclass
class ImageFineTuneResult:
    run_id: str
    run_path: Path
    metrics: dict[str, float]


def _hash_dir(path: Path) -> str:
    hasher = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            stat = p.stat()
            hasher.update(str(p.relative_to(path)).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
    return hasher.hexdigest()


def _set_finetune_params(model: Any, freeze_backbone: bool) -> list[Any]:
    if not freeze_backbone:
        return list(model.parameters())

    for param in model.parameters():
        param.requires_grad = False

    trainable = []
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
        trainable.extend(list(model.fc.parameters()))
    if hasattr(model, "classifier"):
        if hasattr(model.classifier, "parameters"):
            for param in model.classifier.parameters():
                param.requires_grad = True
            trainable.extend(list(model.classifier.parameters()))
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        for param in model.heads.head.parameters():
            param.requires_grad = True
        trainable.extend(list(model.heads.head.parameters()))

    return trainable or list(model.parameters())


def finetune_image_folder(
    data_dir: Path,
    model_arch: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    use_pretrained: bool = True,
    freeze_backbone: bool = True,
    val_split: float = 0.2,
    reuse_if_exists: bool = True,
) -> ImageFineTuneResult:
    """Fine-tune a pretrained image model on a folder dataset."""
    torch = _require_torch()
    import torchvision
    from torchvision import transforms

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    tracker = ExperimentTracker(base_dir=Path("runs"))
    data_hash = _hash_dir(data_dir)
    metadata = {
        "data_hash": data_hash,
        "model_arch": model_arch,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "use_pretrained": use_pretrained,
        "freeze_backbone": freeze_backbone,
        "val_split": val_split,
    }
    if reuse_if_exists:
        existing = tracker.find_matching_run("finetune_image", metadata)
        if existing:
            run_path = Path("runs") / existing
            metrics_path = run_path / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            return ImageFineTuneResult(run_id=existing, run_path=run_path, metrics=metrics)

    model_arch = model_arch.lower()
    if use_pretrained:
        ensure_model_download_allowed(model_arch)
    resize = 224 if model_arch in ["resnet18", "resnet50", "resnet101", "resnet152", "convnext_tiny", "convnext_base", "vit_b_16", "vit_l_16"] else 32
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(root=str(data_dir), transform=transform)
    num_classes = len(dataset.classes)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes in dataset folder.")

    generator = torch.Generator().manual_seed(seed)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = max(1, len(dataset) - val_size)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = _build_image_model(model_arch, num_classes, channels=3, resize=resize, use_pretrained=use_pretrained)
    params = _set_finetune_params(model, freeze_backbone=freeze_backbone)
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    torch.manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.shape[0])
    acc = correct / max(total, 1)

    run_id, run_path = tracker.create_run(model_type="finetune_image")
    tracker.save_json(run_path / "metrics.json", {"val_accuracy": acc})
    tracker.save_json(
        run_path / "dataset.json",
        {
            "dataset": "image_folder",
            "data_dir": str(data_dir),
            "classes": dataset.classes,
            "model_arch": model_arch,
            "input_size": resize,
            "use_pretrained": use_pretrained,
            "freeze_backbone": freeze_backbone,
            "val_split": val_split,
        },
    )
    tracker.save_json(run_path / "model_info.json", {"model_type": "finetune_image", "model_arch": model_arch})
    tracker.save_json(run_path / "run_metadata.json", {"model_type": "finetune_image", **metadata})
    model_path = _save_torch_model(model, run_path)
    finalize_run_tracking(
        run_path=run_path,
        run_type="finetune",
        task_type="classification",
        model_type="finetune_image",
        dataset_hash=data_hash,
        metrics={"val_accuracy": acc},
        model_params=metadata,
        model_artifact=model_path.name,
        config_hash=None,
        seed=seed,
        extra={"data_dir": str(data_dir), "model_arch": model_arch},
    )

    return ImageFineTuneResult(run_id=run_id, run_path=run_path, metrics={"val_accuracy": acc})


def finetune_text_transformer(
    data_path: Path,
    text_column: str,
    label_column: str,
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    reuse_if_exists: bool = True,
) -> TextRunResult:
    """Fine-tune a transformer on custom text data."""
    return train_text_transformer(
        data_path=data_path,
        text_column=text_column,
        label_column=label_column,
        model_name=model_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        run_type="text_finetune",
        reuse_if_exists=reuse_if_exists,
    )
