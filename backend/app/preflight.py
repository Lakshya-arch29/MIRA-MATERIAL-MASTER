"""Fail early with actionable database setup errors before importing route tables."""

from __future__ import annotations

import os
import sys

from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.database import engine

REQUIRED_TABLES = {
    "audit_logs",
    "cnmc",
    "cpses",
    "feedback",
    "mappings",
    "match_suggestions",
    "materials",
    "material_embedding_cache",
    "upload_batches",
    "users",
}


def main() -> int:
    missing_config = []
    model_server_url = os.getenv("MIRA_MODEL_SERVER_URL", "").strip()
    if not model_server_url:
        missing_config.append("MIRA_MODEL_SERVER_URL")
    elif not model_server_url.startswith("https://"):
        missing_config.append("MIRA_MODEL_SERVER_URL (use HTTPS to protect the API key)")
    if not os.getenv("MIRA_API_KEY", "").strip():
        missing_config.append("MIRA_API_KEY")
    if settings.secret_key == "mira-development-secret-key-change-in-production-min32chars":
        missing_config.append("SECRET_KEY (replace the development default)")
    if settings.seed_admin_password == "Admin@123":
        missing_config.append("SEED_ADMIN_PASSWORD (replace the development default)")
    configured_cors = os.getenv("CORS_ORIGINS", "").strip()
    cors_origins = [origin.strip() for origin in configured_cors.split(",") if origin.strip()]
    if not cors_origins or "*" in cors_origins:
        missing_config.append("CORS_ORIGINS (set the exact frontend origin)")

    if missing_config:
        print(
            "Deployment preflight failed: configure " + ", ".join(missing_config) + ".",
            file=sys.stderr,
        )
        return 1

    try:
        existing = set(inspect(engine).get_table_names())
    except SQLAlchemyError as exc:
        print(f"Database preflight failed: could not connect to PostgreSQL ({exc}).", file=sys.stderr)
        return 1

    missing = sorted(REQUIRED_TABLES - existing)
    if missing:
        print(
            "Database preflight failed: missing tables: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "For a fresh database, apply mira_full_schema.sql; for an existing "
            "database, apply 002_create_material_embedding_cache.sql, then redeploy.",
            file=sys.stderr,
        )
        return 1

    print("Database preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
