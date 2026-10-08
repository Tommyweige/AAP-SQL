"""Mine directed relation cues using thesis 3.5.2; never reads Gold SQL to choose edges."""
import argparse
from collections import Counter
from contextlib import closing
from itertools import combinations
import json
from pathlib import Path
import re
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aap_sql.utils.database import foreign_key_rows

ROW_LIMIT = 200_000
VALUE_LIMIT = 50_000
BLACKLIST = set("score count status type rank ranking rating level size weight height age total amount value date time year month day hour flag tag label category priority order quantity index position sequence version commentcount answercount favoritecount viewcount posthistorytypeid".split())
KEY_SUFFIXES = ("number", "code", "key", "ref", "num", "id", "no", "nr")


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def words(name):
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.findall(r"[a-z0-9]+", name.lower())


def entity(name):
    """Normalize separators, CamelCase, key suffixes and common plurals."""
    parts = words(name)
    if parts and parts[-1] in KEY_SUFFIXES:
        parts.pop()
    elif parts:
        for suffix in KEY_SUFFIXES:
            if parts[-1].endswith(suffix) and len(parts[-1]) > len(suffix):
                parts[-1] = parts[-1][:-len(suffix)]
                break
    if parts:
        last = parts[-1]
        parts[-1] = (last[:-3] + "y" if last.endswith("ies") else
                     last[:-1] if last.endswith("s") and not last.endswith("ss")
                     and last not in {"gas", "status", "series", "species", "news"} else last)
    return "".join(parts)


def numeric(col_type):
    return any(t in col_type.upper() for t in ("INT", "REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL", "NUMBER"))


def text_type(col_type):
    return "INT" not in col_type.upper() and any(t in col_type.upper() for t in ("CHAR", "CLOB", "TEXT"))


def columns_and_keys(conn):
    columns = []
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for table in tables:
        rows = list(conn.execute(f"PRAGMA table_info({quote(table)})"))
        primary = [r[1] for r in rows if r[5]]
        keys = set(primary) if len(primary) == 1 else set()
        for idx in conn.execute(f"PRAGMA index_list({quote(table)})"):
            if idx[2] and not idx[4]:  # A partial UNIQUE index is not a global key.
                names = [r[2] for r in conn.execute(f"PRAGMA index_info({quote(idx[1])})")]
                if len(names) == 1 and names[0] is not None:
                    keys.add(names[0])
        for row in rows:
            if numeric(row[2]) or text_type(row[2]):
                columns.append({"table": table, "column": row[1], "numeric": numeric(row[2]), "key": row[1] in keys})
    return columns, tables


def sample(conn, column, row_limit, value_limit):
    table, col = quote(column["table"]), quote(column["column"])
    # Bound rows before counting distinct values; keep original scalar values.
    counts, total = Counter(), 0
    # Keep the same table-row prefix for each column; a covering index must not
    # turn the sample into a different, value-sorted subset for each column.
    for (value,) in conn.execute(f"SELECT {col} FROM {table} NOT INDEXED LIMIT ?", (row_limit,)):
        if value is not None:
            total += 1
            if value in counts or len(counts) < value_limit:
                counts[value] += 1
    return counts, total


def name_evidence(source, target):
    source_name = entity(source["column"])
    target_name = entity(target["column"])
    if not source_name or source_name in BLACKLIST:
        return 0
    # A table-name match cannot override a contradictory destination entity:
    # customer_id -> customers.vendor_id is not customer identity evidence.
    if target_name and source_name != target_name:
        return 0
    # Prefer the explicit source-entity -> target-table correspondence on ties.
    return 2 * (source_name == entity(target["table"])) + (source_name == entity(target["column"]))


def numeric_name_scope(column):
    # ponytail: require a key suffix; bare numeric names need explicit identity
    # evidence before widening this reconstructed miner's candidate scope.
    return any(column["column"].lower().endswith(suffix) for suffix in KEY_SUFFIXES)


def numeric_direction(a, b):
    """Choose from schema/name evidence before allowing data to validate a cue.

    Reconstructed from the thesis's conservative direction decision and omission
    counts; this ordering is inferred, not recovered historical source code.
    """
    directions = []
    for source, target in ((a, b), (b, a)):
        evidence = name_evidence(source, target)
        if evidence and numeric_name_scope(source):
            directions.append((source, target, (target["key"], evidence)))
    if not directions:
        return None
    if len(directions) == 2 and directions[0][2] == directions[1][2]:
        return None
    return max(directions, key=lambda d: d[2])[:2]


def numeric_metrics(source_counts, source_total, target_counts, target_total):
    intersection = source_counts.keys() & target_counts.keys()
    return {
        "distinct_coverage": len(intersection) / len(source_counts) if source_counts else 0,
        "row_coverage": sum(source_counts[v] for v in intersection) / source_total if source_total else 0,
        "target_uniqueness": len(target_counts) / target_total if target_total else 0,
    }


def passes(metrics, target):
    threshold = .90 if target["key"] else .98
    return (metrics["distinct_coverage"] >= threshold and metrics["row_coverage"] >= threshold
            and (target["key"] or metrics["target_uniqueness"] >= .98))


def full_metrics(conn, source, target):
    a, b = quote(source["table"]), quote(target["table"])
    ac, bc = quote(source["column"]), quote(target["column"])
    distinct, total, matched_distinct, matched_rows = conn.execute(f"""
        SELECT COUNT(DISTINCT a.{ac}), COUNT(*),
               COUNT(DISTINCT CASE WHEN b.value IS NOT NULL THEN a.{ac} END),
               COALESCE(SUM(b.value IS NOT NULL), 0)
        FROM {a} AS a LEFT JOIN (SELECT DISTINCT {bc} AS value FROM {b}
                                WHERE {bc} IS NOT NULL) AS b ON a.{ac} = b.value
        WHERE a.{ac} IS NOT NULL
    """).fetchone()
    target_distinct, target_total = conn.execute(f"SELECT COUNT(DISTINCT {bc}), COUNT({bc}) FROM {b}").fetchone()
    return {"distinct_coverage": matched_distinct / distinct if distinct else 0,
            "row_coverage": matched_rows / total if total else 0,
            "target_uniqueness": target_distinct / target_total if target_total else 0,
            "source_distinct": distinct, "source_non_null": total,
            "target_distinct": target_distinct, "target_non_null": target_total}


def mine_database(path, row_limit=ROW_LIMIT, value_limit=VALUE_LIMIT):
    if row_limit <= 0 or value_limit <= 0:
        raise ValueError("Sampling limits must be positive")
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        columns, tables = columns_and_keys(conn)
        declared = set()
        for table in tables:
            for fk in foreign_key_rows(conn, table):
                declared.add(tuple(v.casefold() for v in (table, fk[3], fk[2], fk[4])))
        samples = [sample(conn, c, row_limit, value_limit) for c in columns]
        edges, audit = [], []
        for i, j in combinations(range(len(columns)), 2):
            a, b = columns[i], columns[j]
            if a["table"] == b["table"] or a["numeric"] != b["numeric"]:
                continue
            direction = numeric_direction(a, b) if a["numeric"] else None
            candidates = []
            for si, ti in ((i, j), (j, i)):
                source, target = columns[si], columns[ti]
                sc, sn = samples[si]
                tc, tn = samples[ti]
                if not sc or not tc:
                    continue
                ref = (source["table"], source["column"], target["table"], target["column"])
                is_declared = tuple(v.casefold() for v in ref) in declared
                if source["numeric"]:
                    if direction != (source, target):
                        continue
                    evidence = name_evidence(source, target)
                    metrics = numeric_metrics(sc, sn, tc, tn)
                    if not evidence or not passes(metrics, target):
                        continue
                    # Validate every new numeric candidate against the entire DB.
                    if not is_declared:
                        metrics = full_metrics(conn, source, target)
                        if not passes(metrics, target):
                            continue
                    candidates.append({"edge": ref, "kind": "numeric", "metrics": metrics,
                                       "priority": (target["key"], evidence), "declared": is_declared})
                else:
                    if any(c["column"].lower() in BLACKLIST or BLACKLIST.intersection(words(c["column"]))
                           for c in (source, target)):
                        continue
                    overlap = len(sc.keys() & tc.keys())
                    fc, br = overlap / len(sc), overlap / len(tc)
                    if fc >= .90 and br >= .10:
                        candidates.append({"edge": ref, "kind": "text", "metrics": {"distinct_coverage": fc, "backward_overlap": br}, "declared": is_declared})
            for candidate in candidates:
                audit.append(candidate)
                if not candidate["declared"]:
                    edges.append([*candidate["edge"], round(candidate["metrics"]["distinct_coverage"], 4)])
        return edges, audit


def gold_join_usage(questions, audit, database_root):
    """Post-hoc query-usage check; this result never participates in mining."""
    from sqlglot import exp, parse_one
    from sqlglot.optimizer.scope import traverse_scope, find_all_in_scope
    from scripts.research.training_data import _resolve_column
    counts = {(db_id, *c["edge"]): 0 for db_id, candidates in audit.items()
              for c in candidates if c["kind"] == "numeric" and not c["declared"]}
    schemas = {}
    for db_id in audit:
        path = database_root / db_id / f"{db_id}.sqlite"
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            columns, _ = columns_and_keys(conn)
        schemas[db_id] = {}
        for col in columns:
            schemas[db_id].setdefault(col["table"].lower(), set()).add(col["column"].lower())
    for question in questions:
        db_id = question["db_id"]
        if db_id not in audit:
            continue
        pairs = set()
        cache = {}
        for scope in traverse_scope(parse_one(question["SQL"], read="sqlite")):
            for equal in find_all_in_scope(scope.expression, exp.EQ):
                if not isinstance(equal.left, exp.Column) or not isinstance(equal.right, exp.Column):
                    continue
                left = _resolve_column(scope, equal.left, cache, db_id, schemas)
                right = _resolve_column(scope, equal.right, cache, db_id, schemas)
                pairs.update(frozenset((a, b)) for a in left for b in right)
        for candidate in audit[db_id]:
            if candidate["kind"] != "numeric" or candidate["declared"]:
                continue
            a, ac, b, bc = candidate["edge"]
            pair = frozenset((f"{a}.{ac}".lower(), f"{b}.{bc}".lower()))
            key = (db_id, *candidate["edge"])
            counts[key] = counts.get(key, 0) + (pair in pairs)
    return [{"db_id": key[0], "edge": list(key[1:]), "question_count": count} for key, count in sorted(counts.items())]


def relation_summary(database_root, audit):
    declared, declared_numeric = set(), set()
    for db_id in audit:
        path = database_root / db_id / f"{db_id}.sqlite"
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            types = {(t.casefold(), c[1].casefold()): c[2] for t in tables
                     for c in conn.execute(f"PRAGMA table_info({quote(t)})")}
            for table in tables:
                for fk in foreign_key_rows(conn, table):
                    edge = (db_id, *[v.casefold() for v in (table, fk[3], fk[2], fk[4])])
                    declared.add(edge)
                    if numeric(types[(edge[1], edge[2])]) and numeric(types[(edge[3], edge[4])]):
                        declared_numeric.add(edge)
    candidates = {(db, *[v.casefold() for v in c["edge"]]) for db, cs in audit.items()
                  for c in cs if c["kind"] == "numeric"}
    reverse = {e for e in candidates if (e[0], e[3], e[4], e[1], e[2]) in declared_numeric
               and e not in declared_numeric}
    return {"declared_total": len(declared), "declared_numeric": len(declared_numeric),
            "numeric_candidates": len(candidates), "recovered_numeric": len(candidates & declared_numeric),
            "reverse_predictions": len(reverse), "new_numeric": len(candidates - declared_numeric - reverse)}


def declared_numeric_rejections(database_root, audit):
    """Post-hoc omissions ledger; FK labels never change candidate selection."""
    omissions = []
    for db_id, candidates in audit.items():
        found = {tuple(v.casefold() for v in c["edge"]) for c in candidates if c["kind"] == "numeric"}
        path = database_root / db_id / f"{db_id}.sqlite"
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            columns, tables = columns_and_keys(conn)
            lookup = {(c["table"].casefold(), c["column"].casefold()): c for c in columns}
            seen, samples = set(), {}
            for table in tables:
                for fk in foreign_key_rows(conn, table):
                    edge = tuple(v.casefold() for v in (table, fk[3], fk[2], fk[4]))
                    source, target = lookup.get(edge[:2]), lookup.get(edge[2:])
                    if edge in found or edge in seen or not source or not target or not (source["numeric"] and target["numeric"]):
                        continue
                    seen.add(edge)
                    detail = None
                    metrics = None
                    if source["table"].casefold() == target["table"].casefold():
                        reason, detail = "outside_scope", "same_table"
                    elif not name_evidence(source, target):
                        reason = "name_evidence"
                    elif not numeric_name_scope(source):
                        reason, detail = "outside_scope", "unmarked_numeric_name"
                    elif numeric_direction(source, target) is None:
                        reason = "ambiguous_direction"
                    elif numeric_direction(source, target) != (source, target):
                        reason, detail = "direction_conflict", "preferred_opposite_direction"
                    else:
                        for column in (source, target):
                            key = (column["table"], column["column"])
                            if key not in samples:
                                samples[key] = sample(conn, column, ROW_LIMIT, VALUE_LIMIT)
                        sc, sn = samples[(source["table"], source["column"])]
                        tc, tn = samples[(target["table"], target["column"])]
                        metrics = numeric_metrics(sc, sn, tc, tn)
                        bounded = len(sc) >= VALUE_LIMIT or len(tc) >= VALUE_LIMIT or sn >= ROW_LIMIT or tn >= ROW_LIMIT
                        reason, detail = ("outside_scope", "sampling_limits") if bounded and not passes(metrics, target) else ("data_coverage", None)
                    omissions.append({"db_id": db_id, "edge": list(edge), "reason": reason,
                                      "detail": detail, "sample_metrics": metrics})
    return omissions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/research/thesis/undeclared_relations.json"))
    parser.add_argument("--gold-sql", type=Path, help="Optional post-hoc usage check; requires the training extra")
    args = parser.parse_args()
    paths = sorted(args.database_root.glob("*/*.sqlite"))
    if not paths:
        parser.error("No databases found at <root>/<db_id>/<db_id>.sqlite")
    graph, audit = {}, {}
    for path in paths:
        if path.stem != path.parent.name:
            parser.error(f"Unexpected database path: {path}")
        print(f"Mining {path.stem}...", flush=True)
        graph[path.stem], audit[path.stem] = mine_database(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = relation_summary(args.database_root, audit)
    omissions = declared_numeric_rejections(args.database_root, audit)
    summary["omission_reasons"] = dict(Counter(c["reason"] for c in omissions))
    args.output.with_suffix(".omissions.json").write_text(json.dumps(omissions, indent=2), encoding="utf-8")
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)
    if args.gold_sql:
        questions = json.loads(args.gold_sql.read_text(encoding="utf-8"))
        usage = gold_join_usage(questions, audit, args.database_root)
        args.output.with_suffix(".gold-usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    print(f"Wrote {sum(map(len, graph.values()))} undeclared cues to {args.output}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
