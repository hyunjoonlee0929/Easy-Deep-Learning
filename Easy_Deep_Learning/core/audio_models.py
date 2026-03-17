"""Audio model utilities (pretrained inference)."""

from __future__ import annotations

from io import BytesIO
from typing import Any


def _load_audio_bytes(audio_bytes: bytes) -> tuple[Any, int]:
    try:
        import torchaudio
    except Exception as exc:
        raise RuntimeError("torchaudio is required for audio models.") from exc

    waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0), int(sample_rate)


def classify_audio_bytes(
    audio_bytes: bytes,
    model_name: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run audio classification on WAV bytes using a Hugging Face pipeline."""
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers is required for audio models.") from exc

    waveform, sample_rate = _load_audio_bytes(audio_bytes)
    clf = pipeline("audio-classification", model=model_name)
    results = clf({"array": waveform.numpy(), "sampling_rate": sample_rate}, top_k=top_k)
    return results

