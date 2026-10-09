from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import httpx
import numpy as np


EXPECTED_EMBEDDING_DIM = 1024
REMOTE_MODEL_NAME = "AshIndian/Mira.ai"
REMOTE_SERVER_ENV = "MIRA_MODEL_SERVER_URL"
API_KEY_ENV = "MIRA_API_KEY"
REMOTE_MODEL_TOKEN = "__MIRA_REMOTE_AWS__"

DEFAULT_TIMEOUT = 180.0
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_CONCURRENCY = 2
DEFAULT_RETRIES = 2
MIN_ADAPTIVE_BATCH_SIZE = 4
MAX_BATCH_SIZE = 128


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


EMBED_BATCH_SIZE = _env_int("MIRA_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE, 1, MAX_BATCH_SIZE)
EMBED_MAX_CONCURRENCY = _env_int("MIRA_EMBED_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY, 1, 8)
EMBED_RETRIES = _env_int("MIRA_EMBED_RETRIES", DEFAULT_RETRIES, 0, 5)
EMBED_TIMEOUT = _env_float("MIRA_EMBED_TIMEOUT", DEFAULT_TIMEOUT, 15.0, 600.0)


class MiraRemoteEmbeddingError(RuntimeError):
    pass


def _normalize_embedding_text(text: str) -> str:
    return " ".join(text.split()).strip()


class RemoteMiraEmbeddingModel:
    def __init__(self, base_url: str | None = None, *, timeout: float = EMBED_TIMEOUT) -> None:
        configured_url = (
            base_url if base_url is not None else os.getenv(REMOTE_SERVER_ENV, "")
        ).strip()
        self.base_url = configured_url.rstrip("/")
        if not self.base_url:
            raise RuntimeError(f"{REMOTE_SERVER_ENV} is not configured.")
        if not self.base_url.startswith(("http://", "https://")):
            raise RuntimeError(f"{REMOTE_SERVER_ENV} must start with http:// or https://.")

        self.timeout = timeout
        headers: dict[str, str] = {}
        api_key = os.getenv(API_KEY_ENV, "").strip()
        if api_key:
            headers["X-API-Key"] = api_key

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=min(timeout, 20.0),
                read=timeout,
                write=min(timeout, 60.0),
                pool=min(timeout, 30.0),
            ),
            headers=headers,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def get_embedding_dimension(self) -> int:
        return EXPECTED_EMBEDDING_DIM

    def get_sentence_embedding_dimension(self) -> int:
        return EXPECTED_EMBEDDING_DIM

    def health(self) -> dict[str, Any]:
        try:
            response = self._client.get(
                "/health",
                timeout=httpx.Timeout(connect=10.0, read=min(self.timeout, 30.0), write=10.0, pool=10.0),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                raise MiraRemoteEmbeddingError(
                    f"Invalid MIRA model server health response: {payload}"
                )
            return payload
        except MiraRemoteEmbeddingError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise MiraRemoteEmbeddingError(
                f"Unable to reach MIRA model server at {self.base_url}: {exc}"
            ) from exc

    @staticmethod
    def _clean_texts(texts: Iterable[str]) -> list[str]:
        cleaned: list[str] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"Expected string text, got {type(text).__name__}.")
            value = " ".join(text.split()).strip()
            if not value:
                raise ValueError("Embedding text cannot be empty.")
            cleaned.append(value)
        return cleaned

    @staticmethod
    def _validate_payload(payload: Any, expected_count: int) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise MiraRemoteEmbeddingError("MIRA model server returned a non-object JSON response.")

        embeddings = payload.get("embeddings")
        if embeddings is None:
            single = payload.get("embedding")
            if single is None:
                raise MiraRemoteEmbeddingError(
                    "MIRA model server returned neither embeddings nor embedding."
                )
            embeddings = [single]

        if not isinstance(embeddings, list) or len(embeddings) != expected_count:
            received = len(embeddings) if isinstance(embeddings, list) else "invalid"
            raise MiraRemoteEmbeddingError(
                f"Embedding count mismatch: sent {expected_count}, received {received}."
            )

        validated: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or len(embedding) != EXPECTED_EMBEDDING_DIM:
                raise MiraRemoteEmbeddingError(
                    f"Invalid embedding dimension: expected {EXPECTED_EMBEDDING_DIM}."
                )
            vector: list[float] = []
            for value in embedding:
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise MiraRemoteEmbeddingError(
                        "MIRA model server returned a non-numeric embedding value."
                    ) from exc
                if not math.isfinite(number):
                    raise MiraRemoteEmbeddingError(
                        "MIRA model server returned a non-finite embedding value."
                    )
                vector.append(number)
            validated.append(vector)
        return validated

    def _post_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post("/embed", json={"text": texts})
        response.raise_for_status()
        return self._validate_payload(response.json(), len(texts))

    def _request_embeddings(self, texts: list[str], *, allow_split: bool = True) -> list[list[float]]:
        if not texts:
            return []

        cleaned = self._clean_texts(texts)
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(EMBED_RETRIES + 1):
            try:
                return self._post_embeddings(cleaned)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 413 and allow_split and len(cleaned) > MIN_ADAPTIVE_BATCH_SIZE:
                    middle = len(cleaned) // 2
                    return self._request_embeddings(cleaned[:middle]) + self._request_embeddings(cleaned[middle:])
                retryable = status in {408, 409, 425, 429, 500, 502, 503, 504}
                if not retryable or attempt >= EMBED_RETRIES:
                    raise MiraRemoteEmbeddingError(
                        f"MIRA embedding request failed with HTTP {status}: {exc}"
                    ) from exc
                last_error = exc
            except httpx.ReadTimeout as exc:
                if allow_split and len(cleaned) > MIN_ADAPTIVE_BATCH_SIZE:
                    middle = len(cleaned) // 2
                    return self._request_embeddings(cleaned[:middle]) + self._request_embeddings(cleaned[middle:])
                if attempt >= EMBED_RETRIES:
                    raise MiraRemoteEmbeddingError(
                        f"MIRA embedding request timed out after {EMBED_RETRIES + 1} attempts."
                    ) from exc
                last_error = exc
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                if attempt >= EMBED_RETRIES:
                    raise MiraRemoteEmbeddingError(
                        f"MIRA embedding request failed after {EMBED_RETRIES + 1} attempts: {exc}"
                    ) from exc
                last_error = exc
            except httpx.HTTPError as exc:
                if attempt >= EMBED_RETRIES:
                    raise MiraRemoteEmbeddingError(f"MIRA embedding request failed: {exc}") from exc
                last_error = exc
            except ValueError as exc:
                raise MiraRemoteEmbeddingError(
                    f"MIRA embedding response could not be decoded: {exc}"
                ) from exc

            if attempt < EMBED_RETRIES:
                time.sleep(delay)
                delay = min(delay * 2.0, 8.0)

        raise MiraRemoteEmbeddingError(f"MIRA embedding request failed: {last_error}")

    def encode(
        self,
        sentences: str | list[str] | tuple[str, ...],
        *,
        normalize_embeddings: bool = True,
        batch_size: int | None = None,
        **_: Any,
    ) -> np.ndarray:
        del normalize_embeddings

        if isinstance(sentences, str):
            result = self._request_embeddings([sentences])
            return np.asarray(result[0], dtype=np.float32)

        if not isinstance(sentences, (list, tuple)):
            raise TypeError("sentences must be str, list[str], or tuple[str, ...].")

        if not sentences:
            return np.empty((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)

        requested = EMBED_BATCH_SIZE if batch_size is None else int(batch_size)
        effective_batch_size = max(1, min(requested, EMBED_BATCH_SIZE, MAX_BATCH_SIZE))
        batches = [
            list(sentences[start:start + effective_batch_size])
            for start in range(0, len(sentences), effective_batch_size)
        ]

        if len(batches) == 1:
            batch_results = [self._request_embeddings(batches[0])]
        elif EMBED_MAX_CONCURRENCY == 1:
            batch_results = [self._request_embeddings(batch) for batch in batches]
        else:
            worker_count = min(EMBED_MAX_CONCURRENCY, len(batches))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                batch_results = list(executor.map(self._request_embeddings, batches))

        flattened = [embedding for batch in batch_results for embedding in batch]
        if len(flattened) != len(sentences):
            raise MiraRemoteEmbeddingError(
                f"Embedding count mismatch after batching: sent {len(sentences)}, received {len(flattened)}."
            )

        matrix = np.asarray(flattened, dtype=np.float32)
        expected_shape = (len(sentences), EXPECTED_EMBEDDING_DIM)
        if matrix.shape != expected_shape:
            raise MiraRemoteEmbeddingError(
                f"Invalid embedding matrix shape: {matrix.shape}; expected {expected_shape}."
            )
        return matrix


@lru_cache(maxsize=1)
def _load_model(target: str) -> RemoteMiraEmbeddingModel:
    if target != REMOTE_MODEL_TOKEN:
        raise RuntimeError(f"Unsupported production model target: {target}")
    return RemoteMiraEmbeddingModel()


def resolve_model_name(name_or_alias: str | None = None) -> str:
    target = (
        name_or_alias
        if name_or_alias is not None
        else os.getenv("MIRA_EMBEDDING_MODEL", "")
    )
    target = str(target).strip()

    if not target or target == REMOTE_MODEL_TOKEN:
        return REMOTE_MODEL_TOKEN

    production_aliases = {
        "default",
        "production",
        "qwen",
        "mira",
        "mira.ai",
        "ashindian/mira.ai",
    }
    if target.lower() in production_aliases:
        return REMOTE_MODEL_TOKEN

    if target.lower() in {"minilm", "base-minilm", "base_minilm"}:
        raise RuntimeError(
            "Legacy MiniLM models are not enabled in the Render production deployment."
        )

    raise RuntimeError(f"Unsupported embedding model '{target}'.")


def get_embedding_model_name() -> str:
    return resolve_model_name()


def get_embedding_model(model_name: str | None = None) -> RemoteMiraEmbeddingModel:
    return _load_model(resolve_model_name(model_name))


def get_embedding_dimension(model_name: str | None = None) -> int:
    return get_embedding_model(model_name).get_embedding_dimension()


class EmbeddingCache:
    def __init__(
        self,
        initial_embeddings: dict[str, np.ndarray] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model_name = resolve_model_name(model_name)
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        if initial_embeddings:
            for text, embedding in initial_embeddings.items():
                self._cache[(self.model_name, _normalize_embedding_text(text))] = embedding

    def _resolve_model(self, model_name: str | None = None) -> str:
        return resolve_model_name(model_name) if model_name is not None else self.model_name

    def get(self, text: str, model_name: str | None = None) -> np.ndarray | None:
        return self._cache.get((self._resolve_model(model_name), _normalize_embedding_text(text)))

    def set(self, text: str, embedding: np.ndarray, model_name: str | None = None) -> None:
        self._cache[(self._resolve_model(model_name), _normalize_embedding_text(text))] = embedding

    def precompute(
        self,
        texts: Iterable[str],
        batch_size: int = EMBED_BATCH_SIZE,
        model_name: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        model = self._resolve_model(model_name)
        unique_texts = list(
            dict.fromkeys(
                _normalize_embedding_text(text) for text in texts
                if isinstance(text, str) and text.strip()
            )
        )
        missing = [
            text for text in unique_texts
            if (model, text) not in self._cache
        ]
        if not missing:
            if progress_callback is not None:
                progress_callback(0, 0)
            return

        embedding_model = get_embedding_model(model)
        effective_batch_size = max(1, min(int(batch_size), EMBED_BATCH_SIZE))
        # Smaller HTTP chunks make long matching runs report visible progress
        # while keeping each model request comfortably below its API limit.
        chunk_size = min(effective_batch_size, 16)
        if progress_callback is not None:
            progress_callback(0, len(missing))

        done = 0
        for start in range(0, len(missing), chunk_size):
            chunk = missing[start : start + chunk_size]
            embeddings = embedding_model.encode(
                chunk,
                batch_size=effective_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            expected_shape = (len(chunk), EXPECTED_EMBEDDING_DIM)
            if embeddings.shape != expected_shape:
                raise MiraRemoteEmbeddingError(
                    f"Invalid embedding matrix shape: {embeddings.shape}; expected {expected_shape}."
                )

            for text, embedding in zip(chunk, embeddings):
                self._cache[(model, text)] = embedding
            done += len(chunk)
            if progress_callback is not None:
                progress_callback(done, len(missing))

    def get_or_encode(self, text: str, model_name: str | None = None) -> np.ndarray | None:
        text = _normalize_embedding_text(text)
        if not text:
            return None

        model = self._resolve_model(model_name)
        key = (model, text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        embedding = get_embedding_model(model).encode(
            text,
            normalize_embeddings=True,
        )
        if embedding.shape != (EXPECTED_EMBEDDING_DIM,):
            raise MiraRemoteEmbeddingError(
                f"Invalid single embedding shape: {embedding.shape}"
            )
        self._cache[key] = embedding
        return embedding

    def similarity(
        self,
        left: str,
        right: str,
        model_name: str | None = None,
    ) -> float:
        if not left or not right:
            return 0.0

        vec_a = self.get_or_encode(left, model_name=model_name)
        vec_b = self.get_or_encode(right, model_name=model_name)
        if vec_a is None or vec_b is None:
            return 0.0

        similarity = float(np.dot(vec_a, vec_b))
        if not math.isfinite(similarity):
            return 0.0
        return max(0.0, min(1.0, similarity))

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, text: str) -> bool:
        return (self.model_name, _normalize_embedding_text(text)) in self._cache


def precompute_embeddings(
    texts: Iterable[str],
    batch_size: int = EMBED_BATCH_SIZE,
    model_name: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EmbeddingCache:
    cache = EmbeddingCache(model_name=model_name)
    cache.precompute(
        texts,
        batch_size=batch_size,
        model_name=model_name,
        progress_callback=progress_callback,
    )
    return cache


def generate_embedding(text: str, model_name: str | None = None) -> list[float]:
    if not text:
        return []
    embedding = get_embedding_model(model_name).encode(
        text,
        normalize_embeddings=True,
    )
    return embedding.astype(np.float32, copy=False).tolist()


def semantic_similarity(
    left: str,
    right: str,
    embedding_cache: EmbeddingCache | None = None,
    model_name: str | None = None,
) -> float:
    if not left or not right:
        return 0.0

    if embedding_cache is not None:
        return embedding_cache.similarity(left, right, model_name=model_name)

    embeddings = get_embedding_model(model_name).encode(
        [left, right],
        normalize_embeddings=True,
        batch_size=2,
    )
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    if not math.isfinite(similarity):
        return 0.0
    return max(0.0, min(1.0, similarity))


def health_check() -> dict[str, Any]:
    return get_embedding_model().health()
