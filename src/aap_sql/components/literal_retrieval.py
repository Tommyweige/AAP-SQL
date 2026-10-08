"""Exact, case-insensitive alignment of quoted question values (thesis 3.4.2)."""

import re
from typing import Any

from aap_sql.backends import get_backend
from aap_sql.config import DatabaseConfig, get_config


def extract_quoted_strings(question: str) -> list[str]:
    """Extract strings enclosed in quotes from question."""
    # Read a complete double-quoted span before considering its apostrophes.
    return [
        double or single
        for double, single in re.findall(
            r""""([^"]+)"|(?<!\w)'(.+?)'(?!\w)""", question
        )
    ]


def get_literal_candidates(
    db_id: str,
    question: str,
    database: DatabaseConfig | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Get raw candidate matches for literals in question.

    Args:
        db_id: Database identifier
        question: User query

    Returns:
        List of candidate dicts:
        {
            "original_literal": str,
            "matched_value": str,
            "table": str,
            "column": str,
            "count": int
        }
    """
    literals = extract_quoted_strings(question)
    if not literals:
        return []

    backend = get_backend(database)
    candidates = []
    limit = get_config().aap.literal_retrieval_limit if limit is None else limit

    seen = set()
    for literal in literals:
        key = literal.lower()
        if key in seen:
            continue
        seen.add(key)
        matches = backend.find_literal_candidates(db_id, literal, min(limit, 5))
        for m in matches:
            candidates.append(
                {
                    "original_literal": literal,
                    "matched_value": m["matched_value"],
                    "table": m["table"],
                    "column": m["column"],
                    "count": m["count"],
                }
            )

    return candidates


def format_literal_hints(
    candidates: list[dict[str, Any]], column_mapping: dict[str, str] | None = None
) -> list[str]:
    """
    Format candidates into hints for the Prompt.
    """
    hints = []
    for c in candidates:
        # Preserve stored case for exact SQL predicates.
        display_val = c.get("use_value", c["matched_value"])
        hint = f'"{display_val}" is in column {c["table"]}.{c["column"]}'

        # Schema-Aware: Append description if mapping provided
        if column_mapping:
            key = f"{c['table']}.{c['column']}".lower()
            desc = column_mapping.get(key)
            if desc:
                hint += f" (Description: {desc})"

        hints.append(hint)
    return hints
