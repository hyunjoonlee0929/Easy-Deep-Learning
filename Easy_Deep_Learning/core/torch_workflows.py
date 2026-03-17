"""Optional Torch-based workflows for image/text models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import re
import inspect
from io import BytesIO

import numpy as np
from sklearn.preprocessing import LabelEncoder

from Easy_Deep_Learning.core.experiment_tracker import ExperimentTracker

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")


@dataclass
class TorchRunResult:
    run_id: str
    run_path: Path
    metrics: dict[str, float]


def _require_torch() -> Any:
    import torch

    torch.set_num_threads(1)
    return torch


def _save_torch_model(model: Any, run_path: Path) -> Path:
    torch = _require_torch()
    model_path = run_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    return model_path


def _load_torch_model(model: Any, run_path: Path) -> Any:
    torch = _require_torch()
    model_path = run_path / "model.pt"
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _build_image_model(
    model_arch: str,
    num_classes: int,
    channels: int,
    resize: int,
    use_pretrained: bool = False,
) -> Any:
    torch = _require_torch()
    import torchvision

    model_arch = model_arch.lower()

    if model_arch == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if use_pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        if channels != 3:
            model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_arch == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT if use_pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        if channels != 3:
            model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_arch == "resnet101":
        weights = torchvision.models.ResNet101_Weights.DEFAULT if use_pretrained else None
        model = torchvision.models.resnet101(weights=weights)
        if channels != 3:
            model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_arch == "resnet152":
        weights = torchvision.models.ResNet152_Weights.DEFAULT if use_pretrained else None
        model = torchvision.models.resnet152(weights=weights)
        if channels != 3:
            model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_arch == "convnext_tiny":
        if not hasattr(torchvision.models, "convnext_tiny"):
            raise RuntimeError("torchvision does not include convnext_tiny. Update torchvision.")
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if use_pretrained else None
        model = torchvision.models.convnext_tiny(weights=weights)
        if channels != 3:
            first = model.features[0][0]
            model.features[0][0] = torch.nn.Conv2d(
                channels,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=first.bias is not None,
            )
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if model_arch == "convnext_base":
        if not hasattr(torchvision.models, "convnext_base"):
            raise RuntimeError("torchvision does not include convnext_base. Update torchvision.")
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT if use_pretrained else None
        model = torchvision.models.convnext_base(weights=weights)
        if channels != 3:
            first = model.features[0][0]
            model.features[0][0] = torch.nn.Conv2d(
                channels,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=first.bias is not None,
            )
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if model_arch == "vit_b_16":
        if not hasattr(torchvision.models, "vit_b_16"):
            raise RuntimeError("torchvision does not include vit_b_16. Update torchvision.")
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
        model = torchvision.models.vit_b_16(weights=weights)
        if channels != 3:
            conv = model.conv_proj
            model.conv_proj = torch.nn.Conv2d(
                channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None,
            )
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        return model

    if model_arch == "vit_l_16":
        if not hasattr(torchvision.models, "vit_l_16"):
            raise RuntimeError("torchvision does not include vit_l_16. Update torchvision.")
        weights = torchvision.models.ViT_L_16_Weights.DEFAULT if use_pretrained else None
        model = torchvision.models.vit_l_16(weights=weights)
        if channels != 3:
            conv = model.conv_proj
            model.conv_proj = torch.nn.Conv2d(
                channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None,
            )
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        return model

    if hasattr(torchvision.models, model_arch):
        model_fn = getattr(torchvision.models, model_arch)
        weights = None
        if use_pretrained:
            try:
                if hasattr(torchvision.models, "get_model_weights"):
                    weights_enum = torchvision.models.get_model_weights(model_fn)
                    weights = weights_enum.DEFAULT
            except Exception:
                weights = None
        try:
            if weights is not None:
                model = model_fn(weights=weights)
            else:
                sig = inspect.signature(model_fn)
                if "pretrained" in sig.parameters:
                    model = model_fn(pretrained=use_pretrained)
                else:
                    model = model_fn()
        except Exception:
            model = model_fn()

        if channels != 3:
            if hasattr(model, "conv1"):
                model.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            else:
                raise RuntimeError("Custom model requires 3-channel inputs.")

        if hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            return model
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, torch.nn.Linear):
                model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
                return model
            if isinstance(model.classifier, torch.nn.Sequential):
                for i in reversed(range(len(model.classifier))):
                    if isinstance(model.classifier[i], torch.nn.Linear):
                        model.classifier[i] = torch.nn.Linear(model.classifier[i].in_features, num_classes)
                        return model
        if hasattr(model, "heads") and hasattr(model.heads, "head"):
            model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
            return model

        return model

    class SimpleCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            conv_out = (resize // 4) * (resize // 4) * 64
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(conv_out, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return SimpleCNN()


def train_cnn_image(
    dataset_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    data_dir: Path,
    model_arch: str = "cnn",
    use_pretrained: bool = False,
    reuse_if_exists: bool = True,
) -> TorchRunResult:
    """Train a simple CNN on a built-in image dataset."""
    torch = _require_torch()
    import torchvision
    from torchvision import transforms

    torch.manual_seed(seed)

    model_arch = model_arch.lower()
    tracker = ExperimentTracker(base_dir=Path("runs"))
    metadata = {
        "dataset_name": dataset_name,
        "model_arch": model_arch,
        "use_pretrained": use_pretrained,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
    }
    if reuse_if_exists:
        existing = tracker.find_matching_run("cnn", metadata)
        if existing:
            run_path = Path("runs") / existing
            metrics_path = run_path / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            return TorchRunResult(run_id=existing, run_path=run_path, metrics=metrics)

    if dataset_name == "MNIST":
        channels = 1
        input_size = 28
        ds_cls = torchvision.datasets.MNIST
    elif dataset_name == "FashionMNIST":
        channels = 1
        input_size = 28
        ds_cls = torchvision.datasets.FashionMNIST
    elif dataset_name == "SVHN":
        channels = 3
        input_size = 32
        ds_cls = torchvision.datasets.SVHN
    elif dataset_name == "EMNIST":
        channels = 1
        input_size = 28
        ds_cls = torchvision.datasets.EMNIST
    elif dataset_name == "CIFAR10":
        channels = 3
        input_size = 32
        ds_cls = torchvision.datasets.CIFAR10
    else:
        raise ValueError("Unsupported dataset")

    resize = 224 if model_arch in ["resnet18", "resnet50", "resnet101", "resnet152", "convnext_tiny", "convnext_base", "vit_b_16", "vit_l_16"] else input_size
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    if dataset_name == "SVHN":
        train_ds = ds_cls(root=str(data_dir), split="train", download=True, transform=transform)
        test_ds = ds_cls(root=str(data_dir), split="test", download=True, transform=transform)
    elif dataset_name == "EMNIST":
        train_ds = ds_cls(root=str(data_dir), split="balanced", train=True, download=True, transform=transform)
        test_ds = ds_cls(root=str(data_dir), split="balanced", train=False, download=True, transform=transform)
    else:
        train_ds = ds_cls(root=str(data_dir), train=True, download=True, transform=transform)
        test_ds = ds_cls(root=str(data_dir), train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(getattr(train_ds, "classes", [])) or int(np.max(train_ds.labels) + 1) if hasattr(train_ds, "labels") else 10

    model = _build_image_model(model_arch, num_classes, channels, resize, use_pretrained=use_pretrained)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

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
        for xb, yb in test_loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.shape[0])

    acc = correct / max(total, 1)

    run_id, run_path = tracker.create_run(model_type="cnn")
    class_names = getattr(train_ds, "classes", None)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    tracker.save_json(run_path / "metrics.json", {"accuracy": acc})
    tracker.save_json(
        run_path / "dataset.json",
        {
            "dataset": dataset_name,
            "data_dir": str(data_dir),
            "classes": class_names,
            "model_arch": model_arch,
            "input_size": resize,
            "channels": channels,
            "use_pretrained": use_pretrained,
        },
    )
    tracker.save_json(
        run_path / "run_metadata.json",
        {
            "model_type": "cnn",
            **metadata,
        },
    )
    tracker.save_json(run_path / "model_info.json", {"model_type": "cnn", "dataset": dataset_name, "model_arch": model_arch})
    _save_torch_model(model, run_path)

    return TorchRunResult(run_id=run_id, run_path=run_path, metrics={"accuracy": acc})


def test_cnn_image(run_id: str) -> dict[str, Any]:
    """Evaluate saved CNN run on its dataset test split."""
    torch = _require_torch()
    import torchvision
    from torchvision import transforms

    run_path = Path("runs") / run_id
    dataset_info = json.loads((run_path / "dataset.json").read_text(encoding="utf-8"))
    dataset_name = dataset_info["dataset"]
    data_dir = Path(dataset_info.get("data_dir", "/tmp/easy_dl"))
    model_arch = dataset_info.get("model_arch", "cnn")
    resize = int(dataset_info.get("input_size", 28))

    if dataset_name == "MNIST":
        channels = 1
        input_size = 28
        ds_cls = torchvision.datasets.MNIST
    elif dataset_name == "FashionMNIST":
        channels = 1
        input_size = 28
        ds_cls = torchvision.datasets.FashionMNIST
    elif dataset_name == "SVHN":
        channels = 3
        input_size = 32
        ds_cls = torchvision.datasets.SVHN
    elif dataset_name == "EMNIST":
        channels = 1
        input_size = 28
        ds_cls = torchvision.datasets.EMNIST
    elif dataset_name == "CIFAR10":
        channels = 3
        input_size = 32
        ds_cls = torchvision.datasets.CIFAR10
    else:
        raise ValueError("Unsupported dataset")

    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
    if dataset_name == "SVHN":
        test_ds = ds_cls(root=str(data_dir), split="test", download=True, transform=transform)
    elif dataset_name == "EMNIST":
        test_ds = ds_cls(root=str(data_dir), split="balanced", train=False, download=True, transform=transform)
    else:
        test_ds = ds_cls(root=str(data_dir), train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    num_classes = len(getattr(test_ds, "classes", [])) or int(np.max(test_ds.labels) + 1) if hasattr(test_ds, "labels") else 10

    model = _build_image_model(model_arch, num_classes, channels, resize, use_pretrained=False)
    model = _load_torch_model(model, run_path)

    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.shape[0])
    acc = correct / max(total, 1)

    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "accuracy": acc,
    }


def predict_cnn_images(run_id: str, images: list[bytes]) -> list[dict[str, Any]]:
    """Run inference on uploaded images for a saved CNN run."""
    torch = _require_torch()
    import torchvision
    from torchvision import transforms
    from PIL import Image

    run_path = Path("runs") / run_id
    dataset_info = json.loads((run_path / "dataset.json").read_text(encoding="utf-8"))
    dataset_name = dataset_info.get("dataset")
    model_arch = dataset_info.get("model_arch", "cnn")
    resize = int(dataset_info.get("input_size", 28))
    channels = int(dataset_info.get("channels", 3))
    class_names = dataset_info.get("classes", [])

    if dataset_name == "MNIST":
        ds_cls = torchvision.datasets.MNIST
        channels = 1
    elif dataset_name == "FashionMNIST":
        ds_cls = torchvision.datasets.FashionMNIST
        channels = 1
    elif dataset_name == "SVHN":
        ds_cls = torchvision.datasets.SVHN
        channels = 3
    elif dataset_name == "EMNIST":
        ds_cls = torchvision.datasets.EMNIST
        channels = 1
    else:
        ds_cls = torchvision.datasets.CIFAR10
        channels = 3

    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

    num_classes = len(class_names) if class_names else 10
    model = _build_image_model(model_arch, num_classes, channels, resize, use_pretrained=False)
    model = _load_torch_model(model, run_path)
    model.eval()

    tensors = []
    for img_bytes in images:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        if channels == 1:
            img = img.convert("L")
        tensors.append(transform(img))
    batch = torch.stack(tensors, dim=0)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    results = []
    for idx, pred in enumerate(preds):
        label = class_names[pred] if class_names and pred < len(class_names) else str(int(pred))
        results.append({"index": idx, "label": label, "prob": float(probs[idx][pred])})
    return results


def predict_cnn_images_with_cam(run_id: str, images: list[bytes]) -> list[dict[str, Any]]:
    """Run inference and Grad-CAM for uploaded images."""
    torch = _require_torch()
    import torchvision
    from torchvision import transforms
    from PIL import Image
    import torch.nn.functional as F

    run_path = Path("runs") / run_id
    dataset_info = json.loads((run_path / "dataset.json").read_text(encoding="utf-8"))
    dataset_name = dataset_info.get("dataset")
    model_arch = dataset_info.get("model_arch", "cnn")
    resize = int(dataset_info.get("input_size", 28))
    channels = int(dataset_info.get("channels", 3))
    class_names = dataset_info.get("classes", [])

    if dataset_name == "MNIST":
        channels = 1
    elif dataset_name == "FashionMNIST":
        channels = 1
    elif dataset_name == "SVHN":
        channels = 3
    elif dataset_name == "EMNIST":
        channels = 1
    else:
        channels = 3

    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

    num_classes = len(class_names) if class_names else 10
    model = _build_image_model(model_arch, num_classes, channels, resize, use_pretrained=False)
    model = _load_torch_model(model, run_path)
    model.eval()

    tensors = []
    for img_bytes in images:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        if channels == 1:
            img = img.convert("L")
        tensors.append(transform(img))
    batch = torch.stack(tensors, dim=0)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _save_activation(module, inp, out):
        activations.clear()
        activations.append(out)

    def _save_gradient(module, grad_in, grad_out):
        gradients.clear()
        gradients.append(grad_out[0])

    if model_arch == "resnet18":
        target_layer = model.layer4
    else:
        if hasattr(model, "conv"):
            target_layer = model.conv[3]
        else:
            raise RuntimeError("Grad-CAM is only supported for cnn/resnet18.")

    target_layer.register_forward_hook(_save_activation)
    target_layer.register_full_backward_hook(_save_gradient)

    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    cams = []
    for idx in range(batch.size(0)):
        model.zero_grad()
        score = logits[idx, preds[idx]]
        score.backward(retain_graph=True)
        act = activations[0][idx]
        grad = gradients[0][idx]
        weights = grad.mean(dim=(1, 2))
        cam = (weights[:, None, None] * act).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(resize, resize), mode="bilinear", align_corners=False)
        cams.append(cam.squeeze().cpu().numpy())

    results = []
    for i, pred in enumerate(preds):
        label = class_names[pred] if class_names and pred < len(class_names) else str(int(pred))
        results.append(
            {
                "index": i,
                "label": label,
                "prob": float(probs[i][pred]),
                "cam": cams[i],
            }
        )
    return results


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "as", "by", "at", "from", "this", "that", "it",
    "i", "you", "he", "she", "they", "we", "me", "my", "your", "his", "her", "their", "our",
}


def _basic_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _make_ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 1:
        return tokens
    grams = tokens[:]
    for k in range(2, n + 1):
        grams.extend(["_".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)])
    return grams


def _get_bpe_vocab(texts: list[str]) -> dict[tuple[str, ...], int]:
    vocab: dict[tuple[str, ...], int] = {}
    for text in texts:
        for word in _basic_tokenize(text):
            tokens = tuple(list(word) + ["</w>"])
            vocab[tokens] = vocab.get(tokens, 0) + 1
    return vocab


def _get_stats(vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    pairs: dict[tuple[str, str], int] = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def _merge_vocab(pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    merged: dict[tuple[str, ...], int] = {}
    bigram = " ".join(pair)
    pattern = re.compile(rf"(?<!\\S){re.escape(bigram)}(?!\\S)")
    for word, freq in vocab.items():
        word_str = " ".join(word)
        new_word = tuple(pattern.sub("".join(pair), word_str).split(" "))
        merged[new_word] = merged.get(new_word, 0) + freq
    return merged


def train_bpe(texts: list[str], vocab_size: int) -> list[tuple[str, str]]:
    vocab = _get_bpe_vocab(texts)
    merges: list[tuple[str, str]] = []
    while len(merges) < max(0, vocab_size - 100) and vocab:
        pairs = _get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = _merge_vocab(best, vocab)
        merges.append(best)
    return merges


def bpe_encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    if not merges:
        return [word]
    ranks = {pair: i for i, pair in enumerate(merges)}
    tokens = list(word) + ["</w>"]

    while True:
        pairs = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
        if not pairs:
            break
        best = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if best not in ranks:
            break
        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    if tokens and tokens[-1] == "</w>":
        tokens = tokens[:-1]
    return tokens


def _tokenize_text(
    text: str,
    stopwords: bool,
    ngram: int,
    bpe_merges: list[tuple[str, str]] | None,
) -> list[str]:
    tokens = _basic_tokenize(text)
    if stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if bpe_merges is not None:
        sub_tokens: list[str] = []
        for word in tokens:
            sub_tokens.extend(bpe_encode_word(word, bpe_merges))
        tokens = sub_tokens
    tokens = _make_ngrams(tokens, ngram)
    return tokens


def _build_vocab(
    texts: list[str],
    max_vocab: int,
    stopwords: bool,
    ngram: int,
    bpe_merges: list[tuple[str, str]] | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for tok in _tokenize_text(text, stopwords, ngram, bpe_merges):
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[: max_vocab - 2]:
        vocab[token] = len(vocab)
    return vocab


def _encode_text(
    text: str,
    vocab: dict[str, int],
    max_len: int,
    stopwords: bool,
    ngram: int,
    bpe_merges: list[tuple[str, str]] | None,
) -> list[int]:
    tokens = _tokenize_text(text, stopwords, ngram, bpe_merges)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


def _load_text_dataset(path: Path, text_col: str, label_col: str) -> tuple[list[str], list[str]]:
    import pandas as pd

    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError("Text or label column not found in dataset.")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    return texts, labels


def train_rnn_text(
    dataset_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    data_dir: Path,
    data_path: Path | None = None,
    text_column: str = "text",
    label_column: str = "label",
    max_vocab: int = 5000,
    max_len: int = 100,
    stopwords: bool = False,
    ngram: int = 1,
    bpe: bool = False,
    bpe_vocab_size: int = 200,
    model_arch: str = "gru",
    reuse_if_exists: bool = True,
) -> TorchRunResult:
    """Train a simple RNN classifier on a CSV dataset."""
    torch = _require_torch()
    torch.manual_seed(seed)

    if data_path is None:
        if dataset_name != "AG_NEWS_SAMPLE":
            raise ValueError("Provide --data for custom text dataset or use AG_NEWS_SAMPLE.")
        data_path = Path("Easy_Deep_Learning/data/text_sample.csv")

    tracker = ExperimentTracker(base_dir=Path("runs"))
    data_hash = tracker.file_hash(Path(data_path))
    metadata = {
        "dataset_name": dataset_name,
        "data_hash": data_hash,
        "text_column": text_column,
        "label_column": label_column,
        "max_vocab": max_vocab,
        "max_len": max_len,
        "stopwords": stopwords,
        "ngram": ngram,
        "bpe": bpe,
        "bpe_vocab_size": bpe_vocab_size,
        "model_arch": model_arch,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
    }
    if reuse_if_exists:
        existing = tracker.find_matching_run("rnn", metadata)
        if existing:
            run_path = Path("runs") / existing
            metrics_path = run_path / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            return TorchRunResult(run_id=existing, run_path=run_path, metrics=metrics)

    texts, labels = _load_text_dataset(data_path, text_column, label_column)
    bpe_merges = train_bpe(texts, bpe_vocab_size) if bpe else None
    vocab = _build_vocab(texts, max_vocab=max_vocab, stopwords=stopwords, ngram=ngram, bpe_merges=bpe_merges)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(np.array(labels))

    X = np.array([_encode_text(t, vocab, max_len, stopwords, ngram, bpe_merges) for t in texts], dtype=np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = max(1, int(len(X) * 0.8))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    if len(X_test) == 0:
        X_test, y_test = X_train, y_train

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class RNNClassifier(torch.nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 64,
            hidden_dim: int = 64,
            num_class: int = 4,
            cell: str = "gru",
        ) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            if cell == "lstm":
                self.rnn = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            else:
                self.rnn = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, num_class)

        def forward(self, x):
            x = self.embedding(x)
            _, h = self.rnn(x)
            if isinstance(h, tuple):
                h = h[0]
            return self.fc(h[-1])

    class TransformerLite(torch.nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int = 64, num_class: int = 4, num_heads: int = 2, num_layers: int = 2) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = torch.nn.Linear(embed_dim, num_class)

        def forward(self, x):
            x = self.embedding(x)
            enc = self.encoder(x)
            pooled = enc.mean(dim=1)
            return self.fc(pooled)

    class TextCNN(torch.nn.Module):
        def __init__(self, vocab_size: int, num_class: int, embed_dim: int = 64, kernels: list[int] | None = None) -> None:
            super().__init__()
            if kernels is None:
                kernels = [3, 4, 5]
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = torch.nn.ModuleList(
                [torch.nn.Conv1d(embed_dim, 64, kernel_size=k) for k in kernels]
            )
            self.fc = torch.nn.Linear(64 * len(kernels), num_class)

        def forward(self, x):
            x = self.embedding(x).transpose(1, 2)
            pooled = []
            for conv in self.convs:
                y = torch.relu(conv(x))
                y = torch.max(y, dim=2).values
                pooled.append(y)
            feats = torch.cat(pooled, dim=1)
            return self.fc(feats)

    num_classes = int(np.max(y) + 1)
    model_arch = model_arch.lower()
    if model_arch == "transformer":
        model = TransformerLite(len(vocab), num_class=num_classes)
    elif model_arch == "textcnn":
        model = TextCNN(len(vocab), num_class=num_classes)
    else:
        cell = "lstm" if model_arch == "lstm" else "gru"
        model = RNNClassifier(len(vocab), num_class=num_classes, cell=cell)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

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
        for xb, yb in test_loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.shape[0])
    acc = correct / max(total, 1)

    run_id, run_path = tracker.create_run(model_type="rnn")
    tracker.save_json(run_path / "metrics.json", {"test_accuracy": acc})
    tracker.save_json(
        run_path / "dataset.json",
        {
            "dataset": dataset_name,
            "data_path": str(data_path),
            "text_column": text_column,
            "label_column": label_column,
            "max_vocab": max_vocab,
            "max_len": max_len,
            "data_dir": str(data_dir),
            "stopwords": stopwords,
            "ngram": ngram,
            "bpe": bpe,
            "bpe_vocab_size": bpe_vocab_size,
            "model_arch": model_arch,
            "label_classes": label_encoder.classes_.tolist(),
        },
    )
    tracker.save_json(
        run_path / "run_metadata.json",
        {
            "model_type": "rnn",
            **metadata,
        },
    )
    tracker.save_json(run_path / "model_info.json", {"model_type": "rnn", "dataset": dataset_name, "model_arch": model_arch})
    _save_torch_model(model, run_path)
    torch.save(vocab, run_path / "vocab.pt")
    if bpe_merges is not None:
        tracker.save_json(run_path / "bpe_merges.json", [list(p) for p in bpe_merges])

    return TorchRunResult(run_id=run_id, run_path=run_path, metrics={"test_accuracy": acc})


def test_rnn_text(run_id: str, data_path: Path | None = None) -> dict[str, Any]:
    """Evaluate saved RNN run on a CSV dataset (default: saved dataset)."""
    torch = _require_torch()

    run_path = Path("runs") / run_id
    dataset_info = json.loads((run_path / "dataset.json").read_text(encoding="utf-8"))
    data_path = Path(data_path) if data_path else Path(dataset_info["data_path"])

    text_col = dataset_info["text_column"]
    label_col = dataset_info["label_column"]
    max_len = int(dataset_info["max_len"])
    stopwords = bool(dataset_info.get("stopwords", False))
    ngram = int(dataset_info.get("ngram", 1))
    bpe = bool(dataset_info.get("bpe", False))
    bpe_merges = None
    if bpe and (run_path / "bpe_merges.json").exists():
        bpe_merges = [tuple(x) for x in json.loads((run_path / "bpe_merges.json").read_text(encoding="utf-8"))]

    vocab = torch.load(run_path / "vocab.pt", map_location="cpu")
    texts, labels = _load_text_dataset(data_path, text_col, label_col)
    label_classes = dataset_info.get("label_classes", [])
    label_to_id = {label: idx for idx, label in enumerate(label_classes)}
    y = np.array([label_to_id.get(lbl, -1) for lbl in labels], dtype=np.int64)
    if (y < 0).any():
        raise ValueError("Found labels not present in training label set.")

    X = np.array([_encode_text(t, vocab, max_len, stopwords, ngram, bpe_merges) for t in texts], dtype=np.int64)

    ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

    model_arch = dataset_info.get("model_arch", "gru")

    class RNNClassifier(torch.nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 64,
            hidden_dim: int = 64,
            num_class: int = 4,
            cell: str = "gru",
        ) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            if cell == "lstm":
                self.rnn = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            else:
                self.rnn = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, num_class)

        def forward(self, x):
            x = self.embedding(x)
            _, h = self.rnn(x)
            if isinstance(h, tuple):
                h = h[0]
            return self.fc(h[-1])

    class TransformerLite(torch.nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int = 64, num_class: int = 4, num_heads: int = 2, num_layers: int = 2) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = torch.nn.Linear(embed_dim, num_class)

        def forward(self, x):
            x = self.embedding(x)
            enc = self.encoder(x)
            pooled = enc.mean(dim=1)
            return self.fc(pooled)

    class TextCNN(torch.nn.Module):
        def __init__(self, vocab_size: int, num_class: int, embed_dim: int = 64, kernels: list[int] | None = None) -> None:
            super().__init__()
            if kernels is None:
                kernels = [3, 4, 5]
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = torch.nn.ModuleList(
                [torch.nn.Conv1d(embed_dim, 64, kernel_size=k) for k in kernels]
            )
            self.fc = torch.nn.Linear(64 * len(kernels), num_class)

        def forward(self, x):
            x = self.embedding(x).transpose(1, 2)
            pooled = []
            for conv in self.convs:
                y = torch.relu(conv(x))
                y = torch.max(y, dim=2).values
                pooled.append(y)
            feats = torch.cat(pooled, dim=1)
            return self.fc(feats)

    num_classes = len(label_classes) if label_classes else int(np.max(y) + 1) if len(y) else 2
    if model_arch == "transformer":
        model = TransformerLite(len(vocab), num_class=num_classes)
    elif model_arch == "textcnn":
        model = TextCNN(len(vocab), num_class=num_classes)
    else:
        cell = "lstm" if model_arch == "lstm" else "gru"
        model = RNNClassifier(len(vocab), num_class=num_classes, cell=cell)
    model = _load_torch_model(model, run_path)

    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.shape[0])

    acc = correct / max(total, 1)

    return {
        "run_id": run_id,
        "dataset": dataset_info.get("dataset"),
        "test_accuracy": acc,
    }
