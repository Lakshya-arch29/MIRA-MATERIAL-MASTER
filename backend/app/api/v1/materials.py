import csv
import hashlib
import io
import logging
from typing import Any
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.exc import IntegrityError

from app.core.rbac import require_permission
from app.core.security import get_current_active_user
from app.db_adapter import materials_table, next_id
from app.models.user import User
from app.services.ingestion.provenance import resolve_provenance
from app.services.ingestion.service import parse_legacy_file
from app.services.normalization.service import normalize_material_description
from app.services.parsing.service import parse_specifications
from app import store
from app.services.matching.batch_embeddings import generate_embeddings
from app.services.matching.embeddings import get_embedding_model_name
from app.services.matching.milvus_client import insert_material_embeddings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/materials", tags=["Materials"])


def _preview(pairs: list[tuple[str, str]], limit: int = 5) -> str:
    shown = ", ".join(f"{cpse}/{code}" for cpse, code in pairs[:limit])
    extra = len(pairs) - limit
    return shown + (f" (+{extra} more)" if extra > 0 else "")


@router.post("/upload")
async def upload_materials_csv(
    file: UploadFile = File(...),
    current_user: User = Depends(require_permission("upload_data")),
):
    """Upload material master records in legacy formats (CSV, TXT, XML, JSON, XLS, XLSX)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename must be provided")

    content = await file.read()

    # Every format parser already wraps its own errors into ValueError, with one
    # exception: csv.Error, which is not a ValueError subclass. A file whose
    # extension does not match its content (an Excel workbook saved as .csv)
    # reaches parse_csv and raises it, so it is caught separately. The final
    # clause exists because an upload endpoint should never answer 500 for
    # something a user handed it -- the traceback is logged, not swallowed.
    try:
        raw_rows = parse_legacy_file(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except csv.Error as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{file.filename}' could not be read as delimited text: {exc}. "
                "This usually means the extension does not match the real format "
                "- an Excel workbook saved with a .csv name, for example. Save it "
                "in its true format and upload again."
            ),
        )
    except Exception as exc:
        logger.exception("Unexpected error while parsing uploaded file %r", file.filename)
        raise HTTPException(
            status_code=400,
            detail=f"'{file.filename}' could not be parsed: {exc}",
        )

    start_id = next_id(materials_table)
    new_records: list[dict[str, Any]] = []
    for row in raw_rows:
        sheet_name = row.get("_sheet_name")
        prov = resolve_provenance(row, sheet_name=sheet_name, filename=file.filename)
        cpse = prov.cpse if (prov.cpse and prov.cpse != "UNKNOWN") else "CPSE_GENERIC"
        code = (
            row.get("material_code")
            or row.get("source_material_code")
            or row.get("item_code")
            or f"MAT-{uuid.uuid4().hex[:8].upper()}"
        )
        raw_desc = (row.get("description") or row.get("material_description") or "").strip()

        if not raw_desc:
            continue

        norm_desc = normalize_material_description(raw_desc)
        parsed = parse_specifications(raw_desc)

        record: dict[str, Any] = {
            "id": start_id + len(new_records),
            "cpse": cpse,
            "provenance_level": prov.level.value,
            "provenance_confidence": prov.confidence,
            "provenance_source": prov.source,
            "provenance_conflict": prov.conflict_detected,
            "requires_review": prov.requires_review,
            "provenance_details": {
                "all_evidence": prov.all_evidence,
                "conflicts": prov.conflicting_evidence,
            },
            "material_code": code,
            "description": raw_desc,
            "normalized_description": norm_desc,
            # parse_specifications returns {material_grade, pressure_rating, dimensions, voltage_class}
            "category": row.get("category") or "General",
            "unit": row.get("unit"),
            "manufacturer": row.get("manufacturer"),
            "manufacturer_part_number": row.get("manufacturer_part_number"),
            "material_grade": row.get("material_grade") or parsed.get("material_grade"),
            "parsed_specifications": {
                k: v for k, v in parsed.items() if v is not None
            },
            "other_attributes": row.get("other_attributes") or {},
        }
        new_records.append(record)

    # (cpse, material_code) is UNIQUE in the schema. Check for collisions before
    # inserting so the caller gets an actionable 409 instead of an IntegrityError
    # that surfaces as a 500. Two cases: rows that already exist in the database,
    # and rows duplicated inside the uploaded file itself.
    existing_keys = {(m["cpse"], m["material_code"]) for m in store.MATERIALS}
    already_stored = [
        (r["cpse"], r["material_code"])
        for r in new_records
        if (r["cpse"], r["material_code"]) in existing_keys
    ]

    seen_in_file: set[tuple[str, str]] = set()
    duplicated_in_file: list[tuple[str, str]] = []
    for r in new_records:
        key = (r["cpse"], r["material_code"])
        if key in seen_in_file:
            duplicated_in_file.append(key)
        seen_in_file.add(key)

    if already_stored:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{len(already_stored)} record(s) in '{file.filename}' are already "
                f"in the database: {_preview(already_stored)}. A material code may "
                "appear only once per CPSE. Clear the existing materials first, or "
                "upload a file with different codes."
            ),
        )

    if duplicated_in_file:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{file.filename}' contains {len(duplicated_in_file)} repeated "
                f"cpse/material_code pair(s): {_preview(duplicated_in_file)}. "
                "Each row needs a unique code within its CPSE."
            ),
        )

    embedding_texts = {
        int(record["id"]): (
            " ".join((record.get("normalized_description") or record["description"]).split())
        )
        for record in new_records
    }
    try:
        model_name = get_embedding_model_name()
        vectors_by_text = generate_embeddings(embedding_texts.values())
    except Exception as exc:
        logger.exception("Embedding service failed during material ingestion")
        raise HTTPException(
            status_code=503,
            detail="The embedding service is unavailable; no material rows were stored. Start the model service and retry the upload.",
        ) from exc

    embedding_rows = [
        {
            "material_id": material_id,
            "model_name": model_name,
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "embedding": vectors_by_text[text],
        }
        for material_id, text in embedding_texts.items()
    ]
    try:
        # The row and its vector commit together so matching never sees a new
        # material that was only partially embedded.
        store.MATERIALS.extend_with_embeddings(new_records, embedding_rows)
    except IntegrityError as exc:
        # Backstop for a concurrent upload that landed between the check above
        # and this insert.
        logger.warning("Upload rejected by a database constraint: %s", exc)
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{file.filename}' could not be stored: one or more records "
                "conflict with data already in the database. Another upload may "
                "have landed at the same time - reload the materials list and try "
                "again."
            ),
        )

    vectors_by_id = {
        material_id: vectors_by_text[text]
        for material_id, text in embedding_texts.items()
    }
    milvus_vectors_stored = insert_material_embeddings(
        [
            {
                "id": record["id"],
                "description": embedding_texts[int(record["id"])],
                "cpse": record["cpse"],
                "category": record.get("category"),
            }
            for record in new_records
        ],
        vectors_by_id=vectors_by_id,
    )

    return {
        "status": "success",
        "records_ingested": len(new_records),
        "embeddings_stored": len(embedding_rows),
        "milvus_vectors_stored": milvus_vectors_stored,
        "total_materials": len(store.MATERIALS),
        "sample": new_records[:10],
    }


@router.get("")
def list_materials(
    query: str | None = Query(None, description="Search description or code"),
    cpse: str | None = Query(None, description="Filter by CPSE"),
    category: str | None = Query(None, description="Filter by category"),
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
):
    """List ingested material records with optional filtering and pagination."""
    filtered = store.MATERIALS

    if cpse:
        filtered = [m for m in filtered if m["cpse"].lower() == cpse.lower()]

    if category:
        filtered = [
            m for m in filtered if (m.get("category") or "").lower() == category.lower()
        ]

    if query:
        q = query.lower()
        filtered = [
            m for m in filtered
            if q in m["description"].lower() or q in m["material_code"].lower()
        ]

    total = len(filtered)
    paginated = filtered[skip: skip + limit]

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "materials": paginated,
    }


@router.get("/stats")
def materials_stats(current_user: User = Depends(get_current_active_user)):
    """Summary counts for the dashboard / analytics panels."""
    cpse_set = {m["cpse"] for m in store.MATERIALS}
    category_counts: dict[str, int] = {}
    for m in store.MATERIALS:
        cat = m.get("category") or "Uncategorised"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total_materials": len(store.MATERIALS),
        "cpse_count": len(cpse_set),
        "cpse_list": sorted(cpse_set),
        "category_distribution": category_counts,
    }


@router.get("/{material_id}")
def get_material(
    material_id: int,
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve a single material record by ID."""
    for item in store.MATERIALS:
        if item["id"] == material_id:
            return item
    raise HTTPException(status_code=404, detail=f"Material {material_id} not found")
