import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from aap_sql import AAPSQLPipeline, DatabaseConfig
from aap_sql.backends import (
    DatabaseBackendError,
    PostgreSQLBackend,
    SQLiteBackend,
    UnknownIdentifierError,
)
from aap_sql.smoke import DemoEncoder, DemoIndex, DemoReranker

psycopg = pytest.importorskip("psycopg")
sql = pytest.importorskip("psycopg.sql")
conninfo = pytest.importorskip("psycopg.conninfo")


@pytest.fixture
def postgres_fixture() -> Iterator[tuple[str, str]]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL for PostgreSQL integration")
    if not conninfo.conninfo_to_dict(url).get("dbname", "").endswith("_test"):
        pytest.fail("PostgreSQL fixtures require a database name ending in _test")
    schema = "aap_fixture_" + uuid4().hex[:12]
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        connection.execute(
            sql.SQL("SET search_path TO {}").format(sql.Identifier(schema))
        )
        connection.execute("CREATE TABLE groups(id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE people(id INTEGER PRIMARY KEY,name TEXT,"
            "group_id INTEGER REFERENCES groups(id))"
        )
        connection.execute("INSERT INTO groups VALUES(1)")
        connection.execute("INSERT INTO people VALUES(1,'Alice',1),(2,NULL,1)")
        connection.execute('CREATE TABLE "People"("Name" TEXT,name TEXT)')
        connection.execute("INSERT INTO \"People\" VALUES('Upper','Lower')")
        connection.execute("CREATE TABLE pairs(a INTEGER,b INTEGER,PRIMARY KEY(b,a))")
        connection.execute(
            "CREATE TABLE pair_refs(a INTEGER,b INTEGER,"
            "FOREIGN KEY(b,a) REFERENCES pairs(b,a))"
        )
        connection.execute(
            sql.SQL("CREATE TABLE {}({} TEXT)").format(
                sql.Identifier('odd"table'),
                sql.Identifier('odd"column'),
            )
        )
        connection.execute(
            sql.SQL("INSERT INTO {} VALUES (%s)").format(
                sql.Identifier('odd"table'),
            ),
            ("Safe",),
        )
    try:
        yield url, schema
    finally:
        with psycopg.connect(url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def test_postgres_pipeline_has_equivalent_probes_literals_schema_and_relations(
    postgres_fixture: tuple[str, str],
    tmp_path: Path,
) -> None:
    url, schema = postgres_fixture
    backend = PostgreSQLBackend({"demo": url}, schema=schema)
    assert backend.probe_column("demo", "People", "Name", 3) == ["Upper"]
    assert backend.probe_column("demo", "People", "name", 3) == ["Lower"]
    assert [column.name for column in backend.columns("demo", "people")] == [
        "id",
        "name",
        "group_id",
    ]
    assert backend.probe_column("demo", "people", "name", 3) == ["Alice"]
    assert backend.literal_exists("demo", "people", "name", "Alice")
    assert not backend.literal_exists("demo", "people", "name", "alice")
    assert not backend.literal_exists("demo", "people", "name", "' OR 1=1 --")
    assert backend.find_literal_candidates("demo", "alice", 5) == [
        {"table": "people", "column": "name", "matched_value": "Alice", "count": 1},
    ]
    assert backend.probe_column("demo", 'odd"table', 'odd"column', 3) == ["Safe"]
    ddl = backend.schema_ddl("demo")
    assert 'CREATE TABLE "odd""table"' in ddl
    assert 'PRIMARY KEY ("b", "a")' in ddl
    assert 'FOREIGN KEY ("b", "a") REFERENCES "pairs" ("b", "a")' in ddl
    restricted = PostgreSQLBackend(
        {"demo": url}, schema=schema, tables=frozenset({"people"})
    )
    assert restricted.table_names("demo") == ["people"]
    assert restricted.declared_relations("demo") == []
    with pytest.raises(UnknownIdentifierError):
        restricted.probe_column("demo", "groups", "id", 3)
    with pytest.raises(UnknownIdentifierError):
        backend.probe_column("demo", "people; DROP TABLE people", "name", 3)
    pipeline = AAPSQLPipeline(
        DatabaseConfig(
            dev_db_path=None,
            backend=PostgreSQLBackend(
                {"demo": url},
                schema=schema,
                tables=frozenset({"groups", "people"}),
            ),
        ),
        DemoEncoder(),
        DemoIndex(),
        [{"db_id": "demo", "full_name": "people.name", "description": "Person name"}],
        DemoReranker(),
        lambda vectors: None,
    )
    prepared = pipeline.prepare('Find "alice"', "demo", "Use person names")
    assert 'CREATE TABLE "people"' in prepared.prompt
    assert '"Alice" is in column people.name' in prepared.prompt
    assert "people.name examples: Alice" in prepared.prompt
    assert "people.group_id -> groups.id (explicit, score=1)" in prepared.prompt
    assert "Use person names" in prepared.prompt
    sqlite_path = tmp_path / "equivalent.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(
            "CREATE TABLE groups(id INTEGER PRIMARY KEY);"
            "CREATE TABLE people(id INTEGER PRIMARY KEY,name TEXT,"
            "group_id INTEGER REFERENCES groups(id));"
            "INSERT INTO groups VALUES(1);"
            "INSERT INTO people VALUES(1,'Alice',1),(2,NULL,1);"
        )
    sqlite_pipeline = AAPSQLPipeline(
        DatabaseConfig(
            dev_db_path=None, backend=SQLiteBackend(paths={"demo": sqlite_path})
        ),
        DemoEncoder(),
        DemoIndex(),
        pipeline.mapping,
        DemoReranker(),
        lambda vectors: None,
    )
    sqlite_prepared = sqlite_pipeline.prepare(
        'Find "alice"', "demo", "Use person names"
    )
    assert prepared.columns == sqlite_prepared.columns
    assert prepared.literals == sqlite_prepared.literals
    assert prepared.probed_values == sqlite_prepared.probed_values
    assert prepared.declared_relations == sqlite_prepared.declared_relations
    assert prepared.retrieval == sqlite_prepared.retrieval


def test_postgres_failures_are_predictable_without_exposing_credentials() -> None:
    backend = PostgreSQLBackend(
        {"demo": "postgresql://user:private-secret@127.0.0.1:1/unavailable"}
    )
    with pytest.raises(DatabaseBackendError) as failure:
        backend.schema_ddl("demo")
    assert "private-secret" not in str(failure.value)
