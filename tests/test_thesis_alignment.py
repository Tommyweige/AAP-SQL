from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest
from aap_sql import AAPSQLPipeline, DatabaseConfig
from aap_sql.components.literal_retrieval import extract_quoted_strings, get_literal_candidates
from aap_sql.config import PROJECT_ROOT
from aap_sql.prompt import build_prompt
from aap_sql.relations import undeclared_relations, declared_relations
from aap_sql.components.schema_loader import load_schema
from aap_sql.settings import final_configurations, thesis_configurations
from scripts.research import reproduce_bird_dev as runner
from scripts.research.build_relation_graph import mine_database, columns_and_keys


def create_database(path, ddl, inserts=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(ddl)
        for sql, rows in inserts:
            conn.executemany(sql, rows)
        conn.commit()


def test_literals_are_unique_quoted_values_with_five_candidates_each(tmp_path):
    path = tmp_path / "demo/demo.sqlite"
    ddl = ";".join(f"CREATE TABLE t{i}(name VARCHAR(30), number NUMERIC, raw BLOB)" for i in range(6))
    values = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
    create_database(path, ddl, [(f"INSERT INTO t{i} VALUES (?, ?, ?)", [(v, v, v) for v in values]) for i in range(6)])
    question = 'Find "alice", "ALICE", "Bob", "Carol", "Dave", "Eve", "Frank".'
    candidates = get_literal_candidates("demo", question, DatabaseConfig(tmp_path))
    assert len(candidates) == 30
    assert {c["matched_value"] for c in candidates} == set(values)
    assert all(c["column"] == "name" for c in candidates)
    assert extract_quoted_strings('Alice\'s title is "O\'Connor" or \'O\'Neil\'.') == ["O'Connor", "O'Neil"]
    assert get_literal_candidates("demo", "Find Alice without quotes", DatabaseConfig(tmp_path)) == []


def test_pipeline_does_not_rerank_literal_matches(tmp_path):
    path = tmp_path / "demo/demo.sqlite"
    create_database(path, ";".join(f"CREATE TABLE t{i}(name TEXT); INSERT INTO t{i} VALUES ('Alice')" for i in range(5)))
    calls = []
    ranker = SimpleNamespace(predict=lambda pairs, **kwargs: calls.append(pairs) or [1.] * len(pairs))
    pipeline = AAPSQLPipeline(
        DatabaseConfig(tmp_path), SimpleNamespace(encode=lambda *a, **k: [[1.]]),
        SimpleNamespace(search=lambda *a, **k: ([[1.]], [[0]])),
        [{"db_id": "demo", "full_name": "t0.name", "description": "Name"}], ranker, lambda v: None)
    prepared = pipeline.prepare('Find "alice"', "demo")
    assert len(calls) == 1 and len(prepared.literals) == 5
    assert all('"Alice" is in column' in hint for hint in prepared.literals)


def test_relation_requires_both_core_tables():
    graph = {"demo": [["people", "team_id", "teams", "id", 1.], ["teams", "id", "other", "team_id", 1.]]}
    assert undeclared_relations(graph, "demo", [{"full_name": "people.name"}]) == []
    cues = undeclared_relations(graph, "demo", [{"full_name": "PEOPLE.name"}, {"full_name": "teams.name"}])
    assert cues == ["people.team_id -> teams.id (implicit, score=1)"]
    assert undeclared_relations(graph, "demo", []) == []


def test_runner_and_core_use_the_same_probe_format_and_fill_partial_cache(tmp_path, monkeypatch):
    path = tmp_path / "demo/demo.sqlite"
    create_database(path, "CREATE TABLE people(name TEXT, city TEXT); INSERT INTO people VALUES ('Alice', 'Taipei')")
    from aap_sql.config import Config
    import aap_sql.config as config_module
    monkeypatch.setattr(config_module, "_default_config", Config(database=DatabaseConfig(tmp_path)))
    monkeypatch.setattr(runner, "PROBING_CACHE", {("demo", "people.name", 3): ["people.name examples: Alice"]})
    values = runner.cached_probe_columns("demo", [{"full_name": "people.name"}, {"full_name": "people.city"}])
    assert values == ["people.name examples: Alice", "people.city examples: Taipei"]
    assert runner.execute_sql("demo", "DELETE FROM people")[0] is False
    assert runner.execute_sql("demo", "SELECT name FROM people") == (True, {("Alice",)})


def test_ablations_preserve_ddl_and_remove_the_specified_sections():
    configs = {c.name: c for c in thesis_configurations()}
    assert len(configs) == 16
    ddl = "CREATE TABLE people(id INT REFERENCES teams(id));"
    common = dict(schema=ddl, foreign_keys=["extra fk"], probed_values=["value"], literals=["literal"], evidence="hint")
    no_fk = build_prompt("q", [{"full_name": "people.id", "description": "id"}], configs["ddl-fk-only"], **common)
    assert ddl in no_fk and "### Foreign Keys:" not in no_fk
    baseline = build_prompt("q", [], configs["baseline"], **common)
    assert all(s not in baseline for s in ["### Columns:", "### Evidence:", "### Probed Values", "### Found Values", "### Foreign Keys:"])
    task = build_prompt("q", [], configs["task-instructions"], **common)
    cot = build_prompt("q", [], configs["explicit-cot"], **common)
    assert "Do not output explanations" in task and "reason step by step" in cot and "Final SQL:" in cot


def test_numeric_direction_uses_target_key_and_full_data(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE customers(customer_id INTEGER PRIMARY KEY); CREATE TABLE transactions(customer_id INTEGER)",
                    [("INSERT INTO customers VALUES (?)", [(i,) for i in range(1, 6)]),
                     ("INSERT INTO transactions VALUES (?)", [(1,), (2,), (3,), (1,), (2,)])])
    edges, audit = mine_database(path)
    assert edges == [["transactions", "customer_id", "customers", "customer_id", 1.]]
    assert audit[0]["metrics"]["target_uniqueness"] == 1
    with closing(sqlite3.connect(path)) as conn:
        conn.executemany("INSERT INTO transactions VALUES (?)", [(999,)] * 10)
        conn.commit()
    # A valid sampled prefix cannot bypass validation over all source rows.
    assert mine_database(path, row_limit=3)[0] == []


def test_numeric_direction_abstains_on_equal_evidence(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE a(customer_id INTEGER PRIMARY KEY); CREATE TABLE b(customer_id INTEGER PRIMARY KEY)",
                    [("INSERT INTO a VALUES (?)", [(1,), (2,)]), ("INSERT INTO b VALUES (?)", [(1,), (2,)])])
    assert mine_database(path)[0] == []


def test_numeric_names_do_not_override_a_conflicting_target_entity(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE orders(customer_id INTEGER); CREATE TABLE customers(vendor_id INTEGER PRIMARY KEY)",
                    [("INSERT INTO orders VALUES (?)", [(1,), (2,)]), ("INSERT INTO customers VALUES (?)", [(1,), (2,)])])
    assert mine_database(path)[0] == []


def test_numeric_direction_does_not_use_accidental_uniqueness_to_break_a_name_tie(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE attributes(entity_api_id INTEGER); CREATE TABLE entities(entity_api_id INTEGER)",
                    [("INSERT INTO attributes VALUES (?)", [(1,), (1,), (2,)]),
                     ("INSERT INTO entities VALUES (?)", [(1,), (2,)])])
    assert mine_database(path)[0] == []


def test_generic_numeric_entities_and_bare_numeric_names_are_conservative():
    from scripts.research.build_relation_graph import name_evidence, numeric_name_scope
    def column(table, name):
        return {"table": table, "column": name, "key": False}
    assert name_evidence(column("facts", "status_id"), column("status", "status_id")) == 0
    assert name_evidence(column("facts", "year"), column("years", "year")) == 0
    assert not numeric_name_scope(column("addresses", "postal"))
    assert numeric_name_scope(column("addresses", "postal_code"))


def test_omissions_ledger_explains_general_name_scope_and_direction_cases(tmp_path):
    from scripts.research.build_relation_graph import declared_numeric_rejections
    path = tmp_path / "demo/demo.sqlite"
    create_database(path, """
        CREATE TABLE calendar(year INTEGER PRIMARY KEY);
        CREATE TABLE facts(year INTEGER REFERENCES calendar(year));
        CREATE TABLE postal_codes(postal_code INTEGER PRIMARY KEY);
        CREATE TABLE addresses(postal INTEGER REFERENCES postal_codes(postal_code));
        CREATE TABLE entities(entity_api_id INTEGER);
        CREATE TABLE attributes(entity_api_id INTEGER REFERENCES entities(entity_api_id));
    """, [("INSERT INTO calendar VALUES (?)", [(1,), (2,)]),
           ("INSERT INTO facts VALUES (?)", [(1,), (1,), (2,)]),
           ("INSERT INTO postal_codes VALUES (?)", [(1,), (2,)]),
           ("INSERT INTO addresses VALUES (?)", [(1,), (1,), (2,)]),
           ("INSERT INTO entities VALUES (?)", [(1,), (2,)]),
           ("INSERT INTO attributes VALUES (?)", [(1,), (1,), (2,)])])
    _, audit = mine_database(path)
    omissions = declared_numeric_rejections(tmp_path, {"demo": audit})
    reasons = {row["edge"][0]: (row["reason"], row["detail"]) for row in omissions}
    assert reasons == {"facts": ("name_evidence", None),
                       "addresses": ("outside_scope", "unmarked_numeric_name"),
                       "attributes": ("ambiguous_direction", None)}


def test_numeric_near_unique_target_and_complete_uniqueness_validation(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE customers(customer_id INTEGER); CREATE TABLE transactions(customer_id INTEGER)",
                    [("INSERT INTO customers VALUES (?)", [(1,), (2,), (3,)]),
                     ("INSERT INTO transactions VALUES (?)", [(1,), (2,), (1,)])])
    assert mine_database(path)[0] == [["transactions", "customer_id", "customers", "customer_id", 1.]]
    with closing(sqlite3.connect(path)) as conn:
        conn.executemany("INSERT INTO customers VALUES (?)", [(1,)] * 10)
        conn.commit()
    assert mine_database(path, row_limit=3)[0] == []


def test_text_relations_use_directional_coverage_and_blacklist(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE source(code TEXT, status TEXT); CREATE TABLE target(code TEXT, status TEXT)",
                    [("INSERT INTO source VALUES (?, ?)", [("0", "active"), ("1", "active")]),
                     ("INSERT INTO target VALUES (?, ?)", [(str(i), "active") for i in range(20)])])
    edges, _ = mine_database(path)
    assert edges == [["source", "code", "target", "code", 1.]]


def test_blacklist_matches_name_words_without_confusing_count_and_county(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE a(positionText TEXT, CountyName TEXT); CREATE TABLE b(positionText TEXT, CountyName TEXT)",
                    [("INSERT INTO a VALUES (?, ?)", [("1", "A"), ("2", "B")]),
                     ("INSERT INTO b VALUES (?, ?)", [("1", "A"), ("2", "B")])])
    edges, _ = mine_database(path)
    assert len(edges) == 2 and all(e[1] == "CountyName" and e[3] == "CountyName" for e in edges)


def test_partial_and_composite_unique_indexes_are_not_single_column_keys(tmp_path):
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE a(id INTEGER, x INTEGER, PRIMARY KEY(id,x)); CREATE UNIQUE INDEX partial ON a(id) WHERE x > 0")
    with closing(sqlite3.connect(path)) as conn:
        columns, _ = columns_and_keys(conn)
    assert not any(c["key"] for c in columns)


def test_shorthand_foreign_key_is_resolved_in_schema_cues_and_mining(tmp_path):
    path = tmp_path / "demo/demo.sqlite"
    create_database(path, "CREATE TABLE Customers(customer_id INTEGER PRIMARY KEY, code TEXT UNIQUE); CREATE TABLE transactions(customer_id INTEGER REFERENCES customers)",
                    [("INSERT INTO Customers VALUES (?, ?)", [(1, "A"), (2, "B")]),
                     ("INSERT INTO transactions VALUES (?)", [(1,), (1,)])])
    database = DatabaseConfig(tmp_path)
    assert "customers.customer_id" in declared_relations("demo", database)[0]
    assert "`None`" not in load_schema("demo", database)
    edges, audit = mine_database(path)
    assert edges == []
    assert len(audit) == 1 and audit[0]["declared"]


def test_full_schema_preserves_composite_key_order_and_grouping(tmp_path):
    path = tmp_path / "demo/demo.sqlite"
    create_database(path, "CREATE TABLE parent(a INTEGER, b INTEGER, PRIMARY KEY(b,a)); CREATE TABLE child(x INTEGER, y INTEGER, FOREIGN KEY(x,y) REFERENCES parent)")
    ddl = load_schema("demo", DatabaseConfig(tmp_path))
    assert "PRIMARY KEY (`b`, `a`)" in ddl
    assert "FOREIGN KEY (`x`, `y`) REFERENCES `parent` (`b`, `a`)" in ddl
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(ddl)
        conn.execute("INSERT INTO parent VALUES(1,2)")
        conn.execute("INSERT INTO child VALUES(2,1)")


def test_sampling_obeys_row_and_distinct_caps(tmp_path):
    from scripts.research.build_relation_graph import sample
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE a(id INTEGER)", [("INSERT INTO a VALUES (?)", [(1,), (2,), (3,), (1,), (None,), (4,)])])
    with closing(sqlite3.connect(path)) as conn:
        counts, total = sample(conn, {"table": "a", "column": "id"}, row_limit=5, value_limit=2)
    assert counts == {1: 2, 2: 1} and total == 4


def test_sampling_uses_the_table_prefix_even_with_a_covering_index(tmp_path):
    from scripts.research.build_relation_graph import sample
    path = tmp_path / "demo.sqlite"
    create_database(path, "CREATE TABLE a(id INTEGER, other TEXT); CREATE INDEX by_id ON a(id)",
                    [("INSERT INTO a VALUES (?, ?)", [(3, "first"), (1, "second"), (2, "third")])])
    with closing(sqlite3.connect(path)) as conn:
        counts, total = sample(conn, {"table": "a", "column": "id"}, row_limit=2, value_limit=5)
    assert counts == {3: 1, 1: 1} and total == 2


def test_question_split_and_batches_do_not_leak_or_drop_pairs():
    pytest.importorskip("sqlglot")
    from scripts.research.training_data import question_splits, build_pairs, question_batches
    questions = [{"question": f"q{i}", "db_id": "demo", "SQL": "SELECT name FROM people"} for i in range(20)]
    desc = [{"db_id": "demo", "full_name": "people.name", "description": "Name"},
            {"db_id": "demo", "full_name": "people.age", "description": "Age"}]
    splits = question_splits(questions)
    ids = [set(v) for v in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(range(20))
    retriever, reranker = build_pairs(questions, desc, splits["train"])
    assert len(retriever) == 16 and len(reranker) == 32
    repeated = retriever + [(0, "q0", "Another positive")]
    batches = list(question_batches(repeated, 64))
    assert sum(map(len, batches)) == len(repeated)
    assert all(len({p[0] for p in b}) == len(b) for b in batches)


def test_sql_parser_resolves_aliases_ctes_and_ambiguous_columns():
    pytest.importorskip("sqlglot")
    from scripts.research.training_data import extract_columns_fixed
    schema = {"demo": {"movies": {"movie_id", "year"}, "ratings": {"movie_id", "score"}}}
    sql = "WITH m AS (SELECT movie_id, year FROM movies) SELECT m.year FROM m JOIN ratings AS r ON m.movie_id=r.movie_id"
    assert extract_columns_fixed(sql, "demo", schema) == {"movies.movie_id", "movies.year", "ratings.movie_id"}
    # Do not invent a positive for an unqualified column shared by two tables.
    assert extract_columns_fixed("SELECT movie_id FROM movies JOIN ratings", "demo", schema) == set()


def test_training_and_listing_help_need_no_models():
    for script, arg in [("train_models.py", "--help"), ("build_relation_graph.py", "--help"), ("reproduce_bird_dev.py", "--list-configs")]:
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/research" / script), arg], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_training_refuses_to_overwrite_existing_weights(tmp_path):
    from scripts.research.train_models import train
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"existing model")
    with pytest.raises(FileExistsError):
        train("retriever", {}, tmp_path, 42)
    assert checkpoint.read_bytes() == b"existing model"


def test_prepare_only_records_counts_and_preserves_a_different_manifest(tmp_path):
    pytest.importorskip("sqlglot")
    questions = tmp_path / "questions.json"
    descriptions = tmp_path / "descriptions.json"
    out = tmp_path / "prepared"
    questions.write_text(json.dumps([{"question": f"q{i}", "db_id": "demo", "SQL": "SELECT name FROM people"} for i in range(20)]))
    descriptions.write_text(json.dumps([{"db_id": "demo", "full_name": f"people.{name}", "description": name} for name in ("name", "age")]))
    command = [sys.executable, str(PROJECT_ROOT / "scripts/research/train_models.py"), "--questions", str(questions),
               "--descriptions", str(descriptions), "--output-root", str(out), "--prepare-only"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest_path = out / "training_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["pair_counts"]["train"] == {"retriever": 16, "reranker": 32}
    assert not list(out.glob("*/training_report.json"))
    before = manifest_path.read_bytes()
    descriptions.write_text(descriptions.read_text().replace('"age"', '"city"'))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "Training inputs differ" in result.stderr
    assert manifest_path.read_bytes() == before


def test_gold_usage_is_post_hoc_and_does_not_change_the_graph(tmp_path):
    pytest.importorskip("sqlglot")
    from scripts.research.build_relation_graph import gold_join_usage
    path = tmp_path / "demo/demo.sqlite"
    create_database(path, "CREATE TABLE customers(customer_id INTEGER PRIMARY KEY); CREATE TABLE transactions(customer_id INTEGER)",
                    [("INSERT INTO customers VALUES (?)", [(1,), (2,), (3,)]),
                     ("INSERT INTO transactions VALUES (?)", [(1,), (1,)])])
    edges, audit = mine_database(path)
    questions = [{"db_id": "demo", "SQL": "SELECT t.customer_id FROM transactions t JOIN customers c ON t.customer_id=c.customer_id"}]
    usage = gold_join_usage(questions, {"demo": audit}, tmp_path)
    assert usage[0]["question_count"] == 1
    assert mine_database(path)[0] == edges


def test_runner_counts_failed_generation_in_denominator_and_separates_thought_tokens(tmp_path, monkeypatch):
    class Progress:
        n = 0
        def update(self, n): self.n += n
        def set_postfix(self, *args, **kwargs): pass
        def close(self): pass
    captured = []
    failures = iter([False, True])
    def generate(**kwargs):
        should_fail = next(failures)
        captured.append(kwargs)
        if should_fail:
            raise RuntimeError("400 invalid input")
        return SimpleNamespace(text="SELECT 1", usage_metadata=SimpleNamespace(
            prompt_token_count=17, candidates_token_count=5, thoughts_token_count=80))
    monkeypatch.setattr(runner, "BASE_DIR", tmp_path)
    monkeypatch.setattr(runner, "MODEL_NAME_API", "gemini-2.5-flash-lite")
    monkeypatch.setattr(runner, "llm_model", SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(runner, "genai_types", SimpleNamespace(GenerateContentConfig=lambda **kw: kw, ThinkingConfig=lambda **kw: kw), raising=False)
    monkeypatch.setattr(runner, "tqdm", lambda **kw: Progress())
    monkeypatch.setattr(runner, "precompute_retrieval", lambda questions, config: ([[]] * 2, [{"mode": "empty", "candidate_count": 0}] * 2))
    monkeypatch.setattr(runner, "precompute_literals", lambda questions, config: [[]] * 2)
    monkeypatch.setattr(runner, "build_prompt", lambda *args: "schema-only prompt")
    monkeypatch.setattr(runner, "execute_sql", lambda db, sql: (True, {(1,)}) if sql else (False, None))
    result = runner.run_experiment(final_configurations()[0], [
        {"question": "one", "db_id": "demo", "SQL": "SELECT 1"},
        {"question": "two", "db_id": "demo", "SQL": "SELECT 1"}])
    assert (result["correct"], result["total"], result["accuracy"]) == (1, 2, "50.00%")
    assert result["total_tokens"] == 22 and result["thought_tokens"] == 80
    assert captured[0]["config"] == {"temperature": 0., "max_output_tokens": 2048, "thinking_config": {"thinking_budget": 0}}
