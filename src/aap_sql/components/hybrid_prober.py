"""
Hybrid Prober - Targeted Verification of Literal Candidates.
Executes SQL probes to verify that retrieved literal candidates exist in the database.
"""

from typing import Any

from aap_sql.backends import get_backend
from aap_sql.config import DatabaseConfig


def probe_literal_candidates(
    db_id: str,
    candidates: list[dict[str, Any]],
    database: DatabaseConfig | None = None,
) -> list[dict[str, Any]]:
    """Verify candidates; backend errors never make unverified literals trusted."""
    if not candidates:
        return []
    backend = get_backend(database)
    validated = []
    for candidate in candidates:
        table, column = candidate["table"], candidate["column"]
        original = candidate["original_literal"]
        matched = candidate["matched_value"]
        if backend.literal_exists(db_id, table, column, original):
            candidate["use_value"] = original
            validated.append(candidate)
        elif backend.literal_exists(db_id, table, column, matched):
            candidate["use_value"] = matched
            validated.append(candidate)
    return validated
