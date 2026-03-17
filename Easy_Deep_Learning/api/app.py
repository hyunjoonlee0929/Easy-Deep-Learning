"""FastAPI prediction server for Easy Deep Learning."""

from __future__ import annotations

from typing import Any, List

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Easy_Deep_Learning.core.inference import predict_from_dataframe
from Easy_Deep_Learning.core.llm_finetune import generate_with_lora, generate_chat_with_lora

app = FastAPI(title="Easy Deep Learning API", version="0.1.0")


class PredictRequest(BaseModel):
    run_id: str = Field(..., description="Run ID under runs/")
    records: List[dict[str, Any]] = Field(..., description="List of row objects")
    target_column: str | None = Field(None, description="Optional target column to drop")


class LLMGenerateRequest(BaseModel):
    run_id: str = Field(..., description="LLM fine-tune run ID under runs/")
    prompt: str = Field(..., description="Prompt text")
    max_new_tokens: int = Field(128, description="Max new tokens")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p nucleus sampling")


class LLMChatMessage(BaseModel):
    role: str
    content: str


class LLMChatRequest(BaseModel):
    run_id: str = Field(..., description="LLM fine-tune run ID under runs/")
    messages: List[LLMChatMessage] = Field(..., description="Chat history messages")
    max_new_tokens: int = Field(128, description="Max new tokens")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p nucleus sampling")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    if not payload.records:
        raise HTTPException(status_code=400, detail="records must not be empty")
    try:
        import pandas as pd

        df = pd.DataFrame(payload.records)
        result = predict_from_dataframe(
            run_id=payload.run_id,
            df=df,
            target_column=payload.target_column,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/llm/generate")
def llm_generate(payload: LLMGenerateRequest) -> dict[str, Any]:
    try:
        output = generate_with_lora(
            run_path=Path("runs") / payload.run_id,
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
        return {"run_id": payload.run_id, "output": output}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/llm/chat")
def llm_chat(payload: LLMChatRequest) -> dict[str, Any]:
    try:
        output = generate_chat_with_lora(
            run_path=Path("runs") / payload.run_id,
            messages=[m.dict() for m in payload.messages],
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
        return {"run_id": payload.run_id, "output": output}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
