from __future__ import annotations

import streamlit as st

from Easy_Deep_Learning.core.streamlit_compat import supports_audio_input, supports_tabs_index


def test_streamlit_compat_functions() -> None:
    assert isinstance(supports_audio_input(st), bool)
    assert isinstance(supports_tabs_index(st), bool)

