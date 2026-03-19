"""Security and cost guardrails."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class SecurityPolicy:
    allow_external_requests: bool
    allowed_domains: list[str]
    allow_dataset_download: bool
    allow_large_model_download: bool
    max_model_name_len: int


def get_security_policy() -> SecurityPolicy:
    domains = [d.strip().lower() for d in os.getenv("EASY_DL_ALLOWED_DOMAINS", "github.com,raw.githubusercontent.com,api.github.com").split(",") if d.strip()]
    return SecurityPolicy(
        allow_external_requests=os.getenv("EASY_DL_ALLOW_EXTERNAL_REQUESTS", "1") == "1",
        allowed_domains=domains,
        allow_dataset_download=os.getenv("EASY_DL_ALLOW_DATASET_DOWNLOAD", "1") == "1",
        allow_large_model_download=os.getenv("EASY_DL_ALLOW_LARGE_MODEL_DOWNLOAD", "0") == "1",
        max_model_name_len=int(os.getenv("EASY_DL_MAX_MODEL_NAME_LEN", "80")),
    )


def mask_api_key(key: str | None) -> str:
    if not key:
        return ""
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def validate_openai_key_format(key: str | None) -> bool:
    if not key:
        return False
    return key.startswith("sk-") and len(key) >= 20


def ensure_external_request_allowed(url: str) -> None:
    policy = get_security_policy()
    if not policy.allow_external_requests:
        raise PermissionError("External requests are disabled by policy (EASY_DL_ALLOW_EXTERNAL_REQUESTS=0).")
    host = (urlparse(url).hostname or "").lower()
    if policy.allowed_domains and not any(host == d or host.endswith(f".{d}") for d in policy.allowed_domains):
        raise PermissionError(f"Blocked external domain by policy: {host}")


def ensure_dataset_download_allowed(dataset_name: str) -> None:
    policy = get_security_policy()
    if not policy.allow_dataset_download:
        raise PermissionError(f"Dataset download blocked by policy for '{dataset_name}'.")


def ensure_model_download_allowed(model_name: str) -> None:
    policy = get_security_policy()
    if len(model_name) > policy.max_model_name_len:
        raise PermissionError("Model name is too long and blocked by policy.")
    large_tokens = ["large", "xl", "xxl", "70b", "13b", "34b", "65b", "llama-3.1-70b"]
    is_large = any(token in model_name.lower() for token in large_tokens)
    if is_large and not policy.allow_large_model_download:
        raise PermissionError(
            "Large model download blocked by policy (set EASY_DL_ALLOW_LARGE_MODEL_DOWNLOAD=1 to allow)."
        )


def ensure_openai_key_not_persisted(path: Path | str) -> None:
    """Best-effort check to avoid accidental key persistence in files."""
    p = Path(path)
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8", errors="ignore")
    if "sk-" in text and "OPENAI_API_KEY" not in text:
        raise RuntimeError(f"Potential API key leakage detected in {p}")
