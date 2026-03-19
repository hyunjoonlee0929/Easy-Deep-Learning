"""Simple multimodal embedding + retrieval (image/text)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MMItem:
    id: str
    text: str
    image: Any  # PIL.Image


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / norm


def _image_embed(image) -> np.ndarray:
    import torchvision
    from torchvision import transforms
    torch = __import__("torch")

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        feat = model(x).cpu().numpy().astype(np.float32)
    return feat.ravel()


def _text_embed(text: str) -> np.ndarray:
    # Fixed-size embedding avoids varying dimensions across items.
    import sklearn.feature_extraction.text as sk_text
    vectorizer = sk_text.HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=512,
        alternate_sign=False,
        norm=None,
    )
    vec = vectorizer.transform([text]).toarray().astype(np.float32)[0]
    return vec


def build_index(items: list[MMItem]) -> dict[str, Any]:
    image_vectors = []
    text_vectors = []
    for item in items:
        image_vectors.append(_image_embed(item.image))
        text_vectors.append(_text_embed(item.text))

    image_matrix = _normalize(np.vstack(image_vectors))
    text_matrix = _normalize(np.vstack(text_vectors))
    return {
        "items": items,
        "image_matrix": image_matrix,
        "text_matrix": text_matrix,
    }


def search_by_text(index: dict[str, Any], query: str, top_k: int = 3) -> list[dict[str, Any]]:
    q = _normalize(_text_embed(query)[None, :])
    sims = (index["text_matrix"] @ q.T).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{"id": index["items"][i].id, "score": float(sims[i]), "text": index["items"][i].text} for i in top_idx]


def search_by_image(index: dict[str, Any], image, top_k: int = 3) -> list[dict[str, Any]]:
    q = _normalize(_image_embed(image)[None, :])
    sims = (index["image_matrix"] @ q.T).ravel()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{"id": index["items"][i].id, "score": float(sims[i]), "text": index["items"][i].text} for i in top_idx]
