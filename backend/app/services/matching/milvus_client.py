"""
Shared Milvus connection + write helpers.

Used by the material-upload route (writes embeddings in) and by
vector_search.py (reads them back out).

Design notes:
  * Everything is driven by app.core.config.settings, so host/port/collection
    /top_k are overridable via .env without code edits.
  * Every public function is a no-op (or returns empty) when Milvus is
    unavailable or settings.milvus_enabled is False. Ingestion and matching
    must never fail because the vector store is down -- vector search is an
    additive candidate source, not a dependency.
  * Writes use upsert(), not insert(). The collection's primary key mirrors
    Postgres materials.id; re-uploading a file that reuses ids would create
    duplicate rows under insert(), which would then surface as duplicate
    neighbours during search.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Iterable, Sequence

from app.core.config import settings
from app.services.matching.batch_embeddings import generate_embeddings
from app.services.matching.embeddings import generate_embedding

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.milvus_collection
MILVUS_HOST = settings.milvus_host
MILVUS_PORT = settings.milvus_port
EMBEDDING_BATCH_SIZE = 64

# Vectors written to Milvus must come from the same 1024D Qwen model the
# collection was created with. Resolving through the module-level default
# would let an unrelated change to that default silently switch the embedding
# model: a 384D vector against a 1024D collection fails the upsert, and the
# except below would swallow it, so matching would quietly lose every vector
# candidate. Pin it explicitly. An explicit MIRA_EMBEDDING_MODEL still wins,
# so evaluation harnesses keep their override.
MILVUS_EMBEDDING_MODEL = os.getenv("MIRA_EMBEDDING_MODEL", "").strip() or "qwen"

_connected = False
_collection: Any = None
_loaded = False


class MilvusUnavailable(RuntimeError):
    """Raised internally when Milvus cannot be reached; callers degrade gracefully."""


_PROBE_TIMEOUT_S = 2.0
_PROBE_TTL_UP_S = 60.0
_PROBE_TTL_DOWN_S = 15.0
_probe: dict[str, Any] = {"ok": False, "until": 0.0}


def _mark_down() -> None:
    _probe.update(ok=False, until=time.monotonic() + _PROBE_TTL_DOWN_S)


def milvus_available(force: bool = False) -> bool:
    """True if Milvus is enabled and its port accepts connections. Never raises."""
    if not settings.milvus_enabled:
        return False

    now = time.monotonic()
    if not force and now < _probe["until"]:
        return bool(_probe["ok"])

    try:
        with socket.create_connection(
            (MILVUS_HOST, int(MILVUS_PORT)), timeout=_PROBE_TIMEOUT_S
        ):
            ok = True
    except (OSError, ValueError):
        ok = False

    _probe.update(ok=ok, until=now + (_PROBE_TTL_UP_S if ok else _PROBE_TTL_DOWN_S))
    if not ok:
        logger.warning(
            "Milvus not reachable at %s:%s -- vector search skipped.",
            MILVUS_HOST,
            MILVUS_PORT,
        )
    return ok


def _ensure_connected() -> None:
    global _connected
    if _connected:
        return
    if not settings.milvus_enabled:
        raise MilvusUnavailable("Milvus disabled via settings.milvus_enabled")
    try:
        from pymilvus import connections
    except ImportError as exc:
        raise MilvusUnavailable("pymilvus is not installed") from exc

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    _connected = True


def get_collection() -> Any:
    """Returns the cached Collection handle, connecting on first use."""
    global _collection
    if _collection is not None:
        return _collection

    if not milvus_available():
        raise MilvusUnavailable(
            f"Milvus not reachable at {MILVUS_HOST}:{MILVUS_PORT} "
            f"(or disabled). Start it with: docker compose -f docker-compose.milvus.yml up -d"
        )

    try:
        _ensure_connected()
        from pymilvus import Collection, utility

        if not utility.has_collection(COLLECTION_NAME):
            raise MilvusUnavailable(
                f"Collection '{COLLECTION_NAME}' does not exist. "
                f"Run: python create_milvus_collection.py"
            )

        _collection = Collection(COLLECTION_NAME)
    except MilvusUnavailable:
        raise
    except Exception as exc:
        _mark_down()
        raise MilvusUnavailable(f"Milvus call failed: {exc}") from exc

    return _collection


def ensure_loaded() -> Any:
    """Loads the collection into memory once per process."""
    global _loaded
    collection = get_collection()
    if not _loaded:
        collection.load()
        _loaded = True
    return collection


def reset_client_state() -> None:
    """Drops cached handles. Useful in tests and after recreating the collection."""
    global _connected, _collection, _loaded
    _connected = False
    _collection = None
    _loaded = False
    _probe.update(ok=False, until=0.0)


def insert_material_embedding(
    material_id: int,
    description: str,
    cpse: str,
    category: str,
) -> bool:
    """Embeds one material description and upserts it into Milvus."""
    return insert_material_embeddings(
        [
            {
                "id": material_id,
                "description": description,
                "cpse": cpse,
                "category": category,
            }
        ]
    ) > 0


def insert_material_embeddings(
    records: Iterable[dict[str, Any]],
    *,
    vectors_by_id: dict[int, Sequence[float]] | None = None,
) -> int:
    """Batch variant: embeds in batch and upserts many materials into Milvus in one round trip."""
    rows = [r for r in records if (r.get("description") or "").strip()]
    if not rows:
        return 0

    if not settings.milvus_enabled:
        return 0

    try:
        collection = get_collection()
    except MilvusUnavailable as exc:
        logger.warning("Skipping Milvus write: %s", exc)
        return 0
    except Exception as exc:
        logger.warning("Skipping Milvus write, Milvus unreachable: %s", exc)
        return 0

    embedding_map: dict[str, Sequence[float]] = {}
    if vectors_by_id is None:
        descriptions = [str(r["description"]).strip() for r in rows]
        try:
            embedding_map = generate_embeddings(
                descriptions,
                batch_size=EMBEDDING_BATCH_SIZE,
                model_name=MILVUS_EMBEDDING_MODEL,
            )
        except Exception as exc:
            logger.warning("Batch embedding generation failed during Milvus insert: %s", exc)
            return 0

    ids: list[int] = []
    vectors: list[Sequence[float]] = []
    cpses: list[str] = []
    categories: list[str] = []

    for r in rows:
        desc = str(r["description"]).strip()
        vector = (
            vectors_by_id.get(int(r["id"]))
            if vectors_by_id is not None
            else embedding_map.get(desc)
        )
        if not vector:
            continue
        ids.append(int(r["id"]))
        vectors.append(vector)
        cpses.append((r.get("cpse") or "")[:100])
        categories.append((r.get("category") or "")[:100])

    if not ids:
        return 0

    try:
        collection.upsert([ids, vectors, cpses, categories])
        collection.flush()
    except Exception as exc:
        logger.warning("Milvus upsert failed for %d rows: %s", len(ids), exc)
        _mark_down()
        return 0

    logger.info("Milvus batch upsert complete: %d vectors.", len(ids))
    return len(ids)


def delete_material_embeddings(material_ids: Sequence[int]) -> int:
    """Removes rows by primary key. Returns count attempted; never raises."""
    if not material_ids or not settings.milvus_enabled:
        return 0
    try:
        collection = get_collection()
        id_list = ", ".join(str(int(i)) for i in material_ids)
        collection.delete(f"id in [{id_list}]")
        collection.flush()
    except Exception as exc:
        logger.warning("Milvus delete failed: %s", exc)
        return 0
    return len(material_ids)
