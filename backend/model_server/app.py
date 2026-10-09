"""Small authenticated HTTP wrapper for the MIRA sentence embedding model."""

from __future__ import annotations

import hmac
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("mira.model_server")

EXPECTED_DIMENSION = 1024
DEFAULT_MODEL_ID = "AshIndian/Mira.ai"
DEFAULT_BATCH_SIZE = 32
MAX_REQUEST_ITEMS = 64
MAX_TEXT_CHARS = 16_000

_model: SentenceTransformer | None = None
_device = "unloaded"
_model_id = ""
_model_lock = threading.Lock()


class EmbedRequest(BaseModel):
    text: str | list[str]


def _load_model(device: str) -> SentenceTransformer:
    model_path = os.getenv("MIRA_MODEL_PATH", "").strip()
    model_id = model_path or os.getenv("MIRA_MODEL_ID", DEFAULT_MODEL_ID).strip()
    cache_folder = os.getenv("HF_HOME", "").strip() or None
    return SentenceTransformer(model_id, device=device, cache_folder=cache_folder)


def _load_best_available_model() -> tuple[SentenceTransformer, str]:
    try:
        import torch

        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        preferred = "cpu"

    try:
        model = _load_model(preferred)
        return model, preferred
    except Exception:
        if preferred != "cuda":
            raise
        logger.exception("Could not load MIRA embeddings on CUDA; retrying on CPU")
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return _load_model("cpu"), "cpu"


def _verify_model(model: SentenceTransformer) -> None:
    dimension = model.get_sentence_embedding_dimension()
    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Expected a {EXPECTED_DIMENSION}-dimension model; got {dimension}."
        )
    warmup = model.encode(
        ["MIRA model server warmup"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if np.asarray(warmup).shape != (1, EXPECTED_DIMENSION):
        raise RuntimeError("MIRA model warmup returned an unexpected vector shape.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _device, _model_id
    if not os.getenv("MIRA_API_KEY", "").strip():
        raise RuntimeError("MIRA_API_KEY must be set before starting the model server.")

    _model_id = os.getenv("MIRA_MODEL_PATH", "").strip() or os.getenv(
        "MIRA_MODEL_ID", DEFAULT_MODEL_ID
    ).strip()
    _model, _device = _load_best_available_model()
    _verify_model(_model)
    logger.info("Loaded %s on %s (%d dimensions)", _model_id, _device, EXPECTED_DIMENSION)
    try:
        yield
    finally:
        _model = None


app = FastAPI(
    title="MIRA Embedding Service",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


def _require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("MIRA_API_KEY", "").strip()
    if not expected or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded")
    return {
        "status": "ok",
        "model": _model_id,
        "device": _device,
        "dimensions": EXPECTED_DIMENSION,
    }


@app.post("/embed")
def embed(payload: EmbedRequest, x_api_key: str = Header(default="")) -> dict[str, Any]:
    global _model, _device
    _require_api_key(x_api_key)
    if _model is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded")

    texts = [payload.text] if isinstance(payload.text, str) else payload.text
    if not texts or len(texts) > MAX_REQUEST_ITEMS:
        raise HTTPException(
            status_code=413,
            detail=f"Send between 1 and {MAX_REQUEST_ITEMS} texts per request.",
        )
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise HTTPException(status_code=422, detail="Each text must be a non-empty string.")
    if any(len(text) > MAX_TEXT_CHARS for text in texts):
        raise HTTPException(
            status_code=413,
            detail=f"Each text must be at most {MAX_TEXT_CHARS} characters.",
        )

    try:
        batch_size = int(os.getenv("MIRA_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH_SIZE
    batch_size = max(1, min(batch_size, MAX_REQUEST_ITEMS))
    try:
        with _model_lock:
            vectors = _model.encode(
                [" ".join(text.split()) for text in texts],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
    except RuntimeError as exc:
        if _device != "cuda" or "out of memory" not in str(exc).lower():
            logger.exception("MIRA embedding request failed")
            raise HTTPException(status_code=500, detail="Embedding failed") from exc

        logger.exception("CUDA out of memory; switching embedding service to CPU")
        try:
            import torch

            torch.cuda.empty_cache()
            cpu_model = _load_model("cpu")
            _verify_model(cpu_model)
            with _model_lock:
                _model = cpu_model
                _device = "cpu"
                vectors = _model.encode(
                    [" ".join(text.split()) for text in texts],
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        except Exception as cpu_exc:
            logger.exception("CPU retry also failed")
            raise HTTPException(status_code=500, detail="Embedding failed on GPU and CPU") from cpu_exc

    vectors_array = np.asarray(vectors, dtype=np.float32)
    if vectors_array.shape != (len(texts), EXPECTED_DIMENSION):
        raise HTTPException(status_code=500, detail="Embedding model returned an invalid shape")
    return {"embeddings": vectors_array.tolist()}
