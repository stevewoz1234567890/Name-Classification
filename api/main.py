"""FastAPI service: top-N language-origin labels with log-probability scores."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from name_classification.inference import NameClassifier

_REPO_ROOT = Path(__file__).resolve().parents[1]

_classifier: NameClassifier | None = None


def _default_weights_path() -> Path:
    env = os.environ.get("WEIGHTS_PATH")
    if env:
        return Path(env)
    for candidate in (
        _REPO_ROOT / "weights.pt",
        Path("/app/weights.pt"),
        Path("./weights.pt"),
    ):
        if candidate.is_file():
            return candidate
    return _REPO_ROOT / "weights.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier
    path = _default_weights_path()
    if not path.is_file():
        raise RuntimeError(
            f"Checkpoint not found at {path}. Train with "
            "`python -m name_classification.train -o path/to/weights.pt` "
            "or set WEIGHTS_PATH."
        )
    _classifier = NameClassifier(path)
    yield
    _classifier = None


app = FastAPI(
    title="Name classification",
    description=(
        "Predicts likely language or cultural origin of a personal name using a "
        "character-level RNN. Scores are log-probabilities (natural log) from the model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256, examples=["Quang"])
    top_n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top labels to return, ranked by log-probability",
    )


class LabelScore(BaseModel):
    label: str
    log_probability: float


class ClassifyResponse(BaseModel):
    name: str
    normalized_name: str
    predictions: list[LabelScore]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _classifier is not None}


@app.post("/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest):
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        normalized, preds = _classifier.predict_topk(body.name, body.top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ClassifyResponse(
        name=body.name.strip(),
        normalized_name=normalized,
        predictions=[
            LabelScore(label=p.label, log_probability=p.log_probability) for p in preds
        ],
    )
