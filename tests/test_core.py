from contextlib import closing
import json
import os
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from aap_sql import AAPSQLPipeline, DatabaseConfig
from aap_sql.config import PROJECT_ROOT, resolve_resource_path
from aap_sql.components.schema_loader import load_schema
from aap_sql.components.description_loader import get_description_loader
from aap_sql.prompt import build_prompt
from aap_sql.relations import declared_relations, load_relation_graph, undeclared_relations
from aap_sql.retrieval import retrieve_batch
from aap_sql.settings import final_configurations
from aap_sql.smoke import run_smoke
from aap_sql.sql import clean_sql


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not make network requests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def database(tmp_path):
    (tmp_path / "demo").mkdir()
    with closing(sqlite3.connect(tmp_path / "demo/demo.sqlite")) as conn:
        conn.executescript("CREATE TABLE teams(id INTEGER PRIMARY KEY); CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT, team_id INTEGER REFERENCES teams(id)); INSERT INTO people VALUES(1, 'Alice', NULL);")
        conn.commit()
    return DatabaseConfig(dev_db_path=tmp_path)


class Encoder:
    def encode(self, texts, **kwargs):
        self.texts = texts
        return [[1] for _ in texts]


class Index:
    def __init__(self, results):
        self.results = iter(results)
        self.ks = []
    def search(self, vectors, k):
        self.ks.append(k)
        indices = next(self.results)
        return [[1.0] * len(indices)], [indices]


class Reranker:
    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        return list(range(len(pairs)))


def test_resource_resolution_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_resource_path("resources/bird_dev_20240627/column_mapping.json").exists()
    assert resolve_resource_path(tmp_path / "external.json") == tmp_path / "external.json"
    assert "BIRD" not in str(DatabaseConfig().dev_db_path)


def test_schema_and_declared_relations(database):
    schema = load_schema("demo", database)
    assert "CREATE TABLE `people`" in schema and "CREATE TABLE `teams`" in schema
    assert "people.team_id -> teams.id (explicit, score=1)" in declared_relations("demo", database)


def test_metadata_cache_switches_paths(tmp_path):
    def metadata(path, description):
        path.write_text(json.dumps([{"db_id": "demo", "table_names_original": ["people"], "column_names_original": [[0, "name"]], "column_names": [[0, description]]}]))
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    metadata(first, "Name one")
    metadata(second, "Name two")
    assert get_description_loader(first).get_descriptions("demo")["people.name"] == "Name one"
    assert get_description_loader(second).get_descriptions("demo")["people.name"] == "Name two"


def test_relation_resource_errors_are_explicit(tmp_path):
    assert load_relation_graph(None) == {}
    with pytest.raises(FileNotFoundError):
        load_relation_graph(tmp_path / "missing.json")
    path = tmp_path / "relations.json"
    expected = {"demo": [["people", "team_id", "teams", "id", .75]]}
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert load_relation_graph(path) == expected
    assert undeclared_relations({"demo": [["people", "team_id", "teams", "id", .75]]}, "demo", [{"full_name": "people.name"}, {"full_name": "teams.id"}]) == ["people.team_id -> teams.id (implicit, score=0.75)"]


def test_retrieval_filters_database_reranks_and_limits():
    mapping = [{"db_id": "other" if i == 0 else "demo", "full_name": f"t.c{i}", "description": f"Column {i}"} for i in range(14)]
    model, ranker = Encoder(), Reranker()
    index = Index([list(range(14)) + [-1, 100]])
    normalizations = []
    columns, meta = retrieve_batch([{"question": "q", "db_id": "demo"}], final_configurations()[0], model=model, index=index, mapping=mapping, reranker=ranker, normalize=normalizations.append)
    assert index.ks == [50] and model.texts == ["q"] and len(normalizations) == 1
    assert len(columns[0]) == 10 and columns[0][0]["full_name"] == "t.c13"
    assert all(pair[0] == "q" and pair[1] != "Column 0" for pair in ranker.pairs)
    assert meta == [{"mode": "direct", "candidate_count": 13}]


def test_retrieval_does_not_widen_global_top_k():
    mapping = [{"db_id": "other", "full_name": "t.other", "description": "Other"},
               {"db_id": "demo", "full_name": "t.local", "description": "Local"}]
    index = Index([[0]])
    columns, meta = retrieve_batch([{"question": "q", "db_id": "demo"}], final_configurations()[0], model=Encoder(), index=index, mapping=mapping, reranker=Reranker(), normalize=lambda v: None)
    assert index.ks == [50]
    assert columns == [[]] and meta == [{"mode": "empty", "candidate_count": 0}]


def test_prompt_section_order_and_content():
    prompt = build_prompt("question", [{"full_name": "t.c", "description": "description"}], final_configurations()[0], schema="schema", foreign_keys=["foreign"], probed_values=["probed"], literals=["literal"], evidence="evidence")
    sections = ["### Schema:", "### Foreign Keys:", "### Columns:", "### Probed Values", "### Found Values", "### Evidence:", "### Question:", "### SQL:"]
    positions = [prompt.index(section) for section in sections]
    assert positions == sorted(positions)
    for content in ["schema", "foreign", "t.c: description", "probed", "literal", "evidence", "question"]:
        assert content in prompt


def test_pipeline_uses_explicit_database_and_generator(database):
    mapping = [{"db_id": "demo", "full_name": "people.name", "description": "Person name"}]
    pipeline = AAPSQLPipeline(database, Encoder(), Index([[0], [0]]), mapping, Reranker(), lambda v: None)
    prepared = pipeline.prepare('Find "alice"', "demo", "Use person names")
    assert '"Alice" is in column people.name' in prepared.prompt
    assert "people.name examples: Alice" in prepared.prompt
    assert "Foreign Keys" in prepared.prompt and "Use person names" in prepared.prompt
    prompts = []
    def generator(prompt):
        prompts.append(prompt)
        return "```sql\nSELECT name FROM people;\n```"
    assert pipeline.generate('Find "alice"', "demo", generator) == "SELECT name FROM people;"
    assert len(prompts) == 1


def test_offline_smoke():
    assert run_smoke().retrieval["candidate_count"] == 1


def test_import_has_no_provider_model_or_process_side_effects(tmp_path):
    code = "import os,sys; before=dict(os.environ); cwd=os.getcwd(); import aap_sql; import scripts.research.reproduce_bird_dev; assert before==dict(os.environ); assert cwd==os.getcwd(); assert not any(m in sys.modules for m in ['google.genai','sentence_transformers','faiss'])"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]), GOOGLE_VERTEX_PROJECT="offline-test", GOOGLE_API_KEY="keep-this-value")
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)


@pytest.mark.parametrize("value, expected", [("", ""), ("SQL: SELECT 1", "SELECT 1"), ("```sql\nWITH x AS (SELECT 1) SELECT * FROM x;\n```", "WITH x AS (SELECT 1) SELECT * FROM x;")])
def test_sql_cleaning(value, expected):
    assert clean_sql(value) == expected
