"""FastAPI prediction server for Easy Deep Learning."""

from __future__ import annotations

from typing import Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Easy_Deep_Learning.core.inference import predict_from_dataframe

app = FastAPI(title="Easy Deep Learning API", version="0.1.0")


class PredictRequest(BaseModel):
    run_id: str = Field(..., description="Run ID under runs/")
    records: List[dict[str, Any]] = Field(..., description="List of row objects")
    target_column: str | None = Field(None, description="Optional target column to drop")


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
