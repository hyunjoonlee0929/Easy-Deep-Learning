"""Audio transcription and evaluation utilities."""

from __future__ import annotations

import io
import os
import re
from typing import Any

from Easy_Deep_Learning.core.security import ensure_external_request_allowed, validate_openai_key_format


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_text(reference).split()
    hyp = _normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return float(dp[-1][-1] / max(1, len(ref)))


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = _normalize_text(reference).replace(" ", "")
    hyp = _normalize_text(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return float(dp[-1][-1] / max(1, len(ref)))


def transcribe_openai(
    audio_bytes: bytes,
    model: str = "gpt-4o-mini-transcribe",
    language: str | None = None,
    prompt: str | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    if not validate_openai_key_format(api_key):
        raise RuntimeError("OPENAI_API_KEY format looks invalid.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package not installed.") from exc

    ensure_external_request_allowed("https://api.openai.com/v1")
    client = OpenAI()
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"

    params: dict[str, Any] = {"model": model, "file": audio_file}
    if language:
        params["language"] = language
    if prompt:
        params["prompt"] = prompt

    result = client.audio.transcriptions.create(**params)
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)
