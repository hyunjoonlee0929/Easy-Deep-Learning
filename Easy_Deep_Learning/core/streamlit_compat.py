"""Compatibility helpers for Streamlit version differences."""

from __future__ import annotations

import inspect
from typing import Sequence


def supports_audio_input(st_module) -> bool:
    """Return True if ``st.audio_input`` exists in this Streamlit version."""
    return hasattr(st_module, "audio_input")


def supports_tabs_index(st_module) -> bool:
    """Return True if ``st.tabs`` supports an ``index`` argument."""
    try:
        sig = inspect.signature(st_module.tabs)
        return "index" in sig.parameters
    except Exception:
        return False


def render_navigation(st_module, labels: Sequence[str], default_label: str, key: str = "active_tab") -> str:
    """Render top-level navigation with best available Streamlit API.

    - Uses ``st.tabs(..., index=...)`` when supported.
    - Falls back to horizontal ``st.radio`` for older versions.
    """
    labels = list(labels)
    default = default_label if default_label in labels else labels[0]
    idx = labels.index(default)

    if supports_tabs_index(st_module):
        st_module.tabs(labels, index=idx)  # type: ignore[arg-type]
        return default

    return st_module.radio("Navigation", labels, index=idx, horizontal=True, key=key)

