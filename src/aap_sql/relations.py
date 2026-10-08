"""Declared and optional undeclared database relation cues."""

import json
from pathlib import Path
from typing import Any, cast

from .backends import get_backend
from .config import DatabaseConfig


def format_fk_line(
    from_ref: str, to_ref: str, fk_type: str = "explicit", score: float | None = None
) -> str:
    """Format FK lines using the thesis prompt convention."""
    if fk_type == "explicit":
        score = 1.0 if score is None else score
        score_str = f"{score:.4f}".rstrip("0").rstrip(".")
        return f"{from_ref} -> {to_ref} (explicit, score={score_str})"
    if score is not None:
        score_str = f"{score:.4f}".rstrip("0").rstrip(".")
        return f"{from_ref} -> {to_ref} ({fk_type}, score={score_str})"
    return f"{from_ref} -> {to_ref} ({fk_type})"


def load_relation_graph(path: Path | None) -> dict[str, list[list[Any]]]:
    """No graph means no mined relations; an explicitly configured graph must load."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as stream:
        return cast(dict[str, list[list[Any]]], json.load(stream))


def declared_relations(db_id: str, database: DatabaseConfig | None = None) -> list[str]:
    return get_backend(database).declared_relations(db_id)


def undeclared_relations(
    graph: dict[str, list[list[Any]]], db_id: str, columns: list[dict[str, Any]]
) -> list[str]:
    """Use implicit cues only when both tables occur in the Top-10 columns.

    This is the user-selected interpretation of question relevance in §3.5.4.
    """
    tables = {column["full_name"].split(".", 1)[0].casefold() for column in columns}
    return [
        format_fk_line(f"{fk[0]}.{fk[1]}", f"{fk[2]}.{fk[3]}", "implicit", fk[4])
        for fk in graph.get(db_id, [])
        if fk[0].casefold() in tables and fk[2].casefold() in tables
    ]
