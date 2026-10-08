"""Question-level training data for thesis 3.3.2.

SQL scope resolution is reused from thesis-writing 2397ab5c,
AAP-SQL-Original/scripts/training/train_paper_repro.py, lines 75-211.
"""
import random
from typing import Dict, Optional, Set
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, traverse_scope


def _normalize_identifier(value: str) -> str:
    if not value:
        return ""
    return value.strip('`"[]').lower()


def _source_table_name(table_expr: exp.Table) -> str:
    return _normalize_identifier(table_expr.name)


def _get_source(scope: Scope, source_alias: str):
    if source_alias in scope.sources:
        return scope.sources[source_alias]
    source_key = _normalize_identifier(source_alias)
    for key, value in scope.sources.items():
        if _normalize_identifier(key) == source_key:
            return value
    return None


def _resolve_source_column(
    scope: Scope,
    source_alias: str,
    column_name: str,
    cache: dict,
    db_id: Optional[str],
    db_table_columns: Optional[Dict[str, Dict[str, Set[str]]]],
) -> set[str]:
    source = _get_source(scope, source_alias)
    if source is None:
        return set()

    if isinstance(source, exp.Table):
        table_name = _source_table_name(source)
        if not table_name:
            return set()
        return {f"{table_name}.{column_name}"}

    if isinstance(source, Scope):
        projection_map = _build_scope_projection_map(source, cache, db_id, db_table_columns)
        return projection_map.get(column_name, set())

    return set()


def _resolve_column(
    scope: Scope,
    column_expr: exp.Column,
    cache: dict,
    db_id: Optional[str],
    db_table_columns: Optional[Dict[str, Dict[str, Set[str]]]],
) -> set[str]:
    column_name = _normalize_identifier(column_expr.name)
    table_alias = _normalize_identifier(column_expr.table)
    if not column_name:
        return set()

    if table_alias:
        return _resolve_source_column(scope, table_alias, column_name, cache, db_id, db_table_columns)

    selected_aliases = list(scope.selected_sources.keys())
    if len(selected_aliases) == 1:
        return _resolve_source_column(scope, selected_aliases[0], column_name, cache, db_id, db_table_columns)

    candidate_columns: Set[str] = set()
    if db_id and db_table_columns and db_id in db_table_columns:
        for alias in selected_aliases:
            source = _get_source(scope, alias)
            if source is None:
                continue
            if isinstance(source, exp.Table):
                table_name = _source_table_name(source)
                if table_name and column_name in db_table_columns[db_id].get(table_name, set()):
                    candidate_columns.add(f"{table_name}.{column_name}")
            elif isinstance(source, Scope):
                projection_map = _build_scope_projection_map(source, cache, db_id, db_table_columns)
                candidate_columns.update(projection_map.get(column_name, set()))

    if len(candidate_columns) == 1:
        return candidate_columns

    return set()


def _build_scope_projection_map(
    scope: Scope,
    cache: dict,
    db_id: Optional[str],
    db_table_columns: Optional[Dict[str, Dict[str, Set[str]]]],
) -> dict[str, set[str]]:
    cache_key = id(scope)
    if cache_key in cache:
        return cache[cache_key]

    projection_map: dict[str, set[str]] = {}
    if not isinstance(scope.expression, exp.Select):
        cache[cache_key] = projection_map
        return projection_map

    for projection in scope.expression.expressions:
        if isinstance(projection, exp.Star):
            continue

        output_name = _normalize_identifier(projection.alias_or_name)
        if not output_name:
            continue

        projection_expr = projection.this if isinstance(projection, exp.Alias) else projection
        resolved_sources: set[str] = set()
        for col in projection_expr.find_all(exp.Column):
            resolved_sources.update(_resolve_column(scope, col, cache, db_id, db_table_columns))

        if resolved_sources:
            projection_map.setdefault(output_name, set()).update(resolved_sources)

    cache[cache_key] = projection_map
    return projection_map


def extract_columns_fixed(
    sql: str,
    db_id: Optional[str] = None,
    db_table_columns: Optional[Dict[str, Dict[str, Set[str]]]] = None,
) -> set[str]:
    try:
        parsed = parse_one(sql, read="sqlite")
    except ParseError:
        return set()

    resolved_columns: set[str] = set()
    projection_cache: dict[int, dict[str, set[str]]] = {}

    for scope in traverse_scope(parsed) or []:
        for col in scope.columns:
            resolved_columns.update(_resolve_column(scope, col, projection_cache, db_id, db_table_columns))

    return resolved_columns


def question_splits(questions, seed=42):
    """Split IDs before expanding column pairs, so no question leaks across splits."""
    ids = list(range(len(questions)))
    random.Random(seed).shuffle(ids)
    train_end, val_end = int(len(ids) * .8), int(len(ids) * .9)
    return {"train": ids[:train_end], "validation": ids[train_end:val_end], "test": ids[val_end:]}


def build_pairs(questions, descriptions, ids, seed=42):
    lookup = {(d["db_id"], d["full_name"].lower()): d["description"] for d in descriptions}
    columns = {}
    for db_id, full_name in lookup:
        table, col = full_name.split(".", 1)
        columns.setdefault(db_id, {}).setdefault(table, set()).add(col)
    rng = random.Random(seed)
    retriever, reranker = [], []
    for idx in ids:
        q = questions[idx]
        used = extract_columns_fixed(q["SQL"], q["db_id"], columns)
        available = {f"{table}.{col}" for table, cols in columns.get(q["db_id"], {}).items() for col in cols}
        positives = sorted(available & used)
        negatives = sorted(available - used)
        for col in positives:
            description = lookup[(q["db_id"], col)]
            retriever.append((idx, q["question"], description))
            reranker.append((idx, q["question"], description, 1.0))
        for col in rng.sample(negatives, min(len(positives), len(negatives))):
            reranker.append((idx, q["question"], lookup[(q["db_id"], col)], 0.0))
    return retriever, reranker


def question_batches(pairs, batch_size, seed=42):
    """Use each question at most once per retriever batch, without discarding pairs."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    groups = {}
    for pair in pairs:
        groups.setdefault(pair[0], []).append(pair)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    while groups:
        ids = list(groups)
        rng.shuffle(ids)
        for start in range(0, len(ids), batch_size):
            batch = []
            for idx in ids[start:start + batch_size]:
                batch.append(groups[idx].pop())
                if not groups[idx]:
                    del groups[idx]
            yield batch
