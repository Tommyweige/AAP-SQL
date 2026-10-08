"""Offline integration demo using synthetic model adapters and a temporary SQLite DB."""
from contextlib import closing
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from . import AAPSQLPipeline, DatabaseConfig


class DemoEncoder:
    def encode(self, texts, **kwargs):
        return [[1.0] for _ in texts]


class DemoIndex:
    def search(self, vectors, k):
        return [[1.0] for _ in vectors], [[0] for _ in vectors]


class DemoReranker:
    def predict(self, pairs, **kwargs):
        return [1.0 for _ in pairs]


def run_smoke():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        db_dir = root / "demo"
        db_dir.mkdir()
        with closing(sqlite3.connect(db_dir / "demo.sqlite")) as conn:
            conn.executescript("CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT); INSERT INTO people VALUES(1, 'Alice');")
            conn.commit()
        pipeline = AAPSQLPipeline(
            database=DatabaseConfig(dev_db_path=root), model=DemoEncoder(),
            index=DemoIndex(), mapping=[{"db_id": "demo", "full_name": "people.name", "description": "Person name"}],
            reranker=DemoReranker(), normalize=lambda vectors: None)
        prepared = pipeline.prepare('Find the person named "alice".', "demo")
        assert "Alice" in prepared.prompt and "CREATE TABLE" in prepared.prompt
        sql = pipeline.generate('Find the person named "alice".', "demo",
                                generator=lambda prompt: "```sql\nSELECT name FROM people WHERE name = 'Alice';\n```")
        with closing(sqlite3.connect(db_dir / "demo.sqlite")) as conn:
            assert conn.execute(sql).fetchall() == [("Alice",)]
        return prepared


if __name__ == "__main__":
    print(run_smoke().prompt)
    print("Offline smoke passed (synthetic model/generator adapters; no benchmark inference).")
