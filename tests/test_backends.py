import sqlite3
from pathlib import Path
from typing import Any

import pytest

from aap_sql import AAPSQLPipeline, DatabaseConfig, PreparedQuery
from aap_sql.backends import SQLiteBackend, UnknownIdentifierError
from aap_sql.smoke import DemoEncoder, DemoIndex, DemoReranker


def test_sqlite_backend_preserves_schema_probes_literals_and_relations(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "demo"
    directory.mkdir()
    with sqlite3.connect(directory / "demo.sqlite") as connection:
        connection.executescript(
            "CREATE TABLE groups(id INTEGER PRIMARY KEY);"
            "CREATE TABLE people(id INTEGER PRIMARY KEY,name TEXT,"
            "group_id INTEGER REFERENCES groups);"
            "INSERT INTO people VALUES(1,'Alice',NULL),(2,'alice',NULL),(3,NULL,NULL);"
        )
    backend = SQLiteBackend(tmp_path)
    assert backend.table_names("demo") == ["groups", "people"]
    assert [column.name for column in backend.columns("demo", "people")] == [
        "id",
        "name",
        "group_id",
    ]
    assert "CREATE TABLE `people`" in backend.schema_ddl("demo")
    assert backend.declared_relations("demo") == [
        "people.group_id -> groups.id (explicit, score=1)"
    ]
    assert backend.probe_column("demo", "people", "name", 3) == ["Alice"]
    matches = backend.find_literal_candidates("demo", "ALICE", 5)
    assert matches == [
        {"table": "people", "column": "name", "matched_value": "Alice", "count": 1}
    ]
    assert backend.literal_exists("demo", "people", "name", "Alice")
    assert not backend.literal_exists("demo", "people", "name", "ALICE")
    assert not backend.literal_exists("demo", "people", "name", "' OR 1=1 --")
    with pytest.raises(UnknownIdentifierError):
        backend.probe_column("demo", "people; DROP TABLE people", "name", 3)
    with pytest.raises(UnknownIdentifierError):
        backend.literal_exists("demo", "people", 'name" OR 1=1 --', "Alice")
    with pytest.raises(ValueError):
        backend.probe_column("demo", "people", "name", 0)
    pipeline = AAPSQLPipeline(
        DatabaseConfig(dev_db_path=None, backend=backend),
        DemoEncoder(),
        DemoIndex(),
        [{"db_id": "demo", "full_name": "people.name", "description": "Person name"}],
        DemoReranker(),
        lambda vectors: None,
    )
    prepared = pipeline.prepare('Find "alice"', "demo")
    assert '"Alice" is in column people.name' in prepared.prompt
    assert "people.name examples: Alice" in prepared.prompt
    assert "people.group_id -> groups.id" in prepared.prompt

    def question_aware_probe(
        db_id: str, question: str, columns: list[dict[str, Any]]
    ) -> list[str]:
        assert db_id == "demo" and question == "Find Al"
        assert columns[0]["full_name"] == "people.name"
        values = backend.probe_column(db_id, "people", "name", 1)
        return ["people.name examples: " + ", ".join(values)]

    augmented = pipeline.prepare(
        "Find Al",
        "demo",
        "Base context",
        external_literals=[
            {
                "original_literal": "Al",
                "matched_value": "Alice",
                "table": "people",
                "column": "name",
                "count": 1,
            }
        ],
        semantic_evidence="Person semantics",
        temporal_evidence="Date scope",
        value_prober=question_aware_probe,
    )
    assert augmented.probed_values == ["people.name examples: Alice"]
    assert augmented.declared_relations == [
        "people.group_id -> groups.id (explicit, score=1)"
    ]
    assert augmented.undeclared_relations == []
    assert augmented.literal_candidates[0]["use_value"] == "Alice"
    assert augmented.retrieval["candidate_columns"] == ["people.name"]
    assert '"Alice" is in column people.name' in augmented.prompt
    assert "Base context\nPerson semantics\nDate scope" in augmented.prompt
    assert PreparedQuery("p", [], [], {}).probed_values == []
