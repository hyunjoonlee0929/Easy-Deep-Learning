from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from Easy_Deep_Learning.agents.tool_agent import AgentInput, ToolUsingAgent, make_default_tools
import Easy_Deep_Learning.core.multimodal as mm
from Easy_Deep_Learning.core.rag import run_rag


def test_rag_smoke() -> None:
    result = run_rag(
        query="What is this project?",
        docs=[
            "Easy Deep Learning is an AI platform.",
            "It supports tabular and multimodal workflows.",
        ],
        top_k=2,
        chunk_size=120,
        overlap=20,
    )
    assert result.answer
    assert result.contexts


def test_agent_smoke(project_root: Path) -> None:
    agent = ToolUsingAgent()
    result = agent.run(
        AgentInput(
            dataset_path=project_root / "data" / "example_dataset.csv",
            target_column="target",
            task_type="classification",
        ),
        tools=make_default_tools(),
    )
    assert result.final_summary
    assert len(result.tool_calls) > 0


def test_multimodal_smoke(monkeypatch) -> None:
    monkeypatch.setattr(mm, "_image_embed", lambda _img: np.ones(512, dtype=np.float32))
    img_a = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    img_b = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    items = [
        mm.MMItem(id="a", text="cat image", image=img_a),
        mm.MMItem(id="b", text="car image with road", image=img_b),
    ]
    index = mm.build_index(items)

    by_text = mm.search_by_text(index, "cat", top_k=1)
    by_image = mm.search_by_image(index, img_a, top_k=1)

    assert len(by_text) == 1
    assert len(by_image) == 1
