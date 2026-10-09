"""
Generic Postgres-backed replacements for the plain Python lists that
api/routes/*.py currently read and write.

WHY THIS FILE EXISTS
---------------------
store.py, mappings.py and audit.py all declare things like:
    MATERIALS: list[dict] = []
and every route touches them only as a plain list: .append(), .extend(),
.clear(), iteration, len(), and (in review.py) in-place dict mutation
like `candidate["review_status"] = "APPROVED"`.

PersistentList below supports every one of those operations, but backs
them with real Postgres reads/writes -- so no route file needs to change
how it uses MATERIALS / CANDIDATES / MAPPINGS / AUDIT_EVENTS.

DBRow is a dict subclass returned by iteration: it looks and behaves like
a plain dict everywhere (JSON-serializes fine, supports .get(), etc.) but
writing to a key (`row["review_status"] = "APPROVED"`) also issues a
real UPDATE against Postgres, so review.py's in-place mutation persists
correctly.
"""

from datetime import datetime
from typing import Any, Iterator

from sqlalchemy import MetaData, Table, select, insert, update, delete, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.types import DateTime

from app.core.database import engine


metadata = MetaData()

# ---------------------------------------------------------------------------
# Table definitions - column names match the dict keys used in the route
# code exactly (see mira_full_schema.sql for the matching CREATE TABLEs).
# ---------------------------------------------------------------------------

materials_table = Table("materials", metadata, autoload_with=engine)
match_suggestions_table = Table("match_suggestions", metadata, autoload_with=engine)
mappings_table = Table("mappings", metadata, autoload_with=engine)
audit_logs_table = Table("audit_logs", metadata, autoload_with=engine)
cnmc_table = Table("cnmc", metadata, autoload_with=engine)
material_embedding_cache_table = Table("material_embedding_cache", metadata, autoload_with=engine)


def _coerce_for_column(table: Table, key: str, value: Any) -> Any:
    """
    matching.py and review.py write timestamps as ISO strings
    (`.isoformat()`), not datetime objects, into TIMESTAMP columns.
    Coerce here so no route file needs to change how it builds values.
    """
    if isinstance(value, str) and key in table.c:
        col_type = table.c[key].type
        if isinstance(col_type, DateTime):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return value
    return value


def _coerce_item(table: Table, item: dict[str, Any]) -> dict[str, Any]:
    return {k: _coerce_for_column(table, k, v) for k, v in item.items()}


class DBRow(dict):
    """A dict that writes each key assignment straight through to Postgres."""

    def __init__(self, table: Table, id_value: Any, data: dict[str, Any]):
        super().__init__(data)
        # Use object.__setattr__ so these don't get treated as dict items.
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_id_value", id_value)

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, value)
        db_value = _coerce_for_column(self._table, key, value)
        with engine.begin() as conn:
            conn.execute(
                update(self._table)
                .where(self._table.c.id == self._id_value)
                .values(**{key: db_value})
            )


class PersistentList:
    """Drop-in replacement for `list[dict]`, backed by one Postgres table."""

    def __init__(self, table: Table):
        self.table = table

    def append(self, item: dict[str, Any]) -> None:
        with engine.begin() as conn:
            conn.execute(insert(self.table).values(**_coerce_item(self.table, item)))

    def extend(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        coerced = [_coerce_item(self.table, item) for item in items]
        with engine.begin() as conn:
            conn.execute(insert(self.table), coerced)

    def extend_with_embeddings(
        self,
        items: list[dict[str, Any]],
        embedding_rows: list[dict[str, Any]],
    ) -> None:
        """Insert materials and their durable vectors in one transaction."""
        if not items:
            return
        coerced = [_coerce_item(self.table, item) for item in items]
        with engine.begin() as conn:
            conn.execute(insert(self.table), coerced)
            if embedding_rows:
                conn.execute(insert(material_embedding_cache_table), embedding_rows)

    def clear(self) -> None:
        with engine.begin() as conn:
            conn.execute(delete(self.table))

    def __iter__(self) -> Iterator[DBRow]:
        with engine.begin() as conn:
            rows = conn.execute(select(self.table)).mappings().all()
        for row in rows:
            yield DBRow(self.table, row["id"], dict(row))

    def __len__(self) -> int:
        with engine.begin() as conn:
            result = conn.execute(
                select(func.count()).select_from(self.table)
            ).scalar()
        return result or 0

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, index: int) -> DBRow:
        # Only used by list slicing (e.g. filtered[skip:skip+limit]) after
        # a route has already built a plain Python list via a comprehension
        # -- PersistentList itself is never sliced directly in the routes,
        # this is here defensively so `for c in store.CANDIDATES` composed
        # with later slicing on the resulting list still works.
        return list(self)[index]

    def __reversed__(self) -> Iterator[DBRow]:
        # audit.py calls reversed(AUDIT_EVENTS). Without this, Python would
        # fall back to repeated __getitem__ calls -- one DB round-trip per
        # row. This does it in a single query instead.
        return reversed(list(self))


def next_id(table: Table) -> int:
    """Next integer id for tables where the app assigns ids itself
    (materials, match_suggestions, mappings all do this in the route
    code, rather than relying on SERIAL) -- based on current DB max,
    so ids stay correct across server restarts.
    """
    with engine.begin() as conn:
        current_max = conn.execute(
            select(func.coalesce(func.max(table.c.id), 0))
        ).scalar()
    return (current_max or 0) + 1


def load_material_embeddings() -> list[dict[str, Any]]:
    """Load durable embeddings without exposing them in material API rows."""
    with engine.begin() as conn:
        rows = conn.execute(select(material_embedding_cache_table)).mappings().all()
    return [dict(row) for row in rows]


def upsert_material_embeddings(rows: list[dict[str, Any]]) -> None:
    """Persist newly generated or refreshed vectors by material ID."""
    if not rows:
        return
    statement = pg_insert(material_embedding_cache_table).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=[material_embedding_cache_table.c.material_id],
        set_={
            "model_name": statement.excluded.model_name,
            "text_hash": statement.excluded.text_hash,
            "embedding": statement.excluded.embedding,
            "updated_at": func.now(),
        },
    )
    with engine.begin() as conn:
        conn.execute(statement)


def next_global_id() -> int:
    """Monotonically increasing global sequence for CNMC across all categories."""
    from sqlalchemy import text
    with engine.begin() as conn:
        try:
            return conn.execute(text("SELECT nextval('cnmc_global_id_seq')")).scalar()
        except Exception:
            current_max = conn.execute(
                select(func.coalesce(func.max(cnmc_table.c.global_id), 0))
            ).scalar()
            return (current_max or 0) + 1
