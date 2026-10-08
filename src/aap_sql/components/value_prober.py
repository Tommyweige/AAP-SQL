"""Format real probed values while keeping backend failures explicit."""

from typing import Any

from aap_sql.backends import get_backend
from aap_sql.config import DatabaseConfig


def probe_column(
    db_id: str,
    table: str,
    column: str,
    limit: int = 3,
    database: DatabaseConfig | None = None,
) -> str:
    values = get_backend(database).probe_column(db_id, table, column, limit)
    return f"{table}.{column} examples: {', '.join(values)}" if values else ""


def probe_columns(
    db_id: str,
    columns: list[dict[str, Any]],
    n: int = 10,
    values_per_column: int = 3,
    database: DatabaseConfig | None = None,
) -> list[str]:
    results = []
    for info in columns[:n]:
        full_name = info.get("full_name", "")
        if "." in full_name:
            table, column = full_name.split(".", 1)
            value = probe_column(db_id, table, column, values_per_column, database)
            if value:
                results.append(value)
    return results
