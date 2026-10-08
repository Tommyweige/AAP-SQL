"""Safe relational operations used by AAP prompt preparation."""

from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from .utils.schema import ColumnSchema, SchemaInfo, TableSchema

if TYPE_CHECKING:
    from .config import DatabaseConfig


def get_backend(database: DatabaseConfig | None = None) -> AAPDatabaseBackend:
    from .config import get_config

    configured = database or get_config().database
    if configured.backend is None:
        configured.backend = SQLiteBackend(
            configured.dev_db_path,
            timeout_seconds=configured.query_timeout,
        )
    return configured.backend


class DatabaseBackendError(Exception):
    """A backend operation failed; messages contain no credentials or raw SQL."""


class UnknownIdentifierError(ValueError):
    """A requested identifier was not found in the introspected schema."""


class LiteralCandidate(TypedDict):
    table: str
    column: str
    matched_value: str
    count: int


class AAPDatabaseBackend(Protocol):
    def schema(self, db_id: str) -> SchemaInfo: ...
    def schema_ddl(self, db_id: str) -> str: ...
    def table_names(self, db_id: str) -> list[str]: ...
    def columns(self, db_id: str, table: str) -> list[ColumnSchema]: ...
    def declared_relations(self, db_id: str) -> list[str]: ...
    def probe_column(
        self, db_id: str, table: str, column: str, limit: int
    ) -> list[str]: ...
    def find_literal_candidates(
        self,
        db_id: str,
        literal: str,
        limit: int,
    ) -> list[LiteralCandidate]: ...
    def literal_exists(
        self, db_id: str, table: str, column: str, value: str
    ) -> bool: ...


class _SQLBackend(ABC):
    placeholder = "?"

    @abstractmethod
    def _fetch(
        self,
        db_id: str,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    def _load_schema(self, db_id: str) -> SchemaInfo: ...

    @abstractmethod
    def _quote(self, identifier: str) -> str: ...

    def _table_ref(self, table: str) -> str:
        return self._quote(table)

    @lru_cache(maxsize=32)
    def schema(self, db_id: str) -> SchemaInfo:
        """Cache introspected identifiers; recreate the backend after a migration."""
        return self._load_schema(db_id)

    def schema_ddl(self, db_id: str) -> str:
        return self.schema(db_id).to_ddl()

    def table_names(self, db_id: str) -> list[str]:
        return [table.name for table in self.schema(db_id).tables]

    def columns(self, db_id: str, table: str) -> list[ColumnSchema]:
        known = self.schema(db_id).get_table(table)
        if known is None:
            raise UnknownIdentifierError("Unknown table")
        return known.columns

    def _identifiers(self, db_id: str, table: str, column: str) -> tuple[str, str]:
        known = self.schema(db_id).get_table(table)
        field = known.get_column(column) if known else None
        if known is None or field is None:
            raise UnknownIdentifierError("Unknown table or column")
        return self._table_ref(known.name), self._quote(field.name)

    def declared_relations(self, db_id: str) -> list[str]:
        return [
            f"{table.name}.{fk['from']} -> {fk['table']}.{fk['to']} (explicit, score=1)"
            for table in self.schema(db_id).tables
            for fk in table.foreign_keys
        ]

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if not 1 <= limit <= 1000:
            raise ValueError("Backend limits must be between 1 and 1000")

    def probe_column(
        self, db_id: str, table: str, column: str, limit: int
    ) -> list[str]:
        self._validate_limit(limit)
        table_ref, column_ref = self._identifiers(db_id, table, column)
        rows = self._fetch(
            db_id,
            f"SELECT DISTINCT {column_ref} FROM {table_ref} LIMIT {self.placeholder}",
            (limit * 2,),
        )
        values: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if row[0] is None:
                continue
            value = str(row[0]).strip()
            if value and len(value) <= 50 and value.lower() not in seen:
                seen.add(value.lower())
                values.append(value)
        return values[:limit]

    def find_literal_candidates(
        self,
        db_id: str,
        literal: str,
        limit: int,
    ) -> list[LiteralCandidate]:
        self._validate_limit(limit)
        matches: list[LiteralCandidate] = []
        for table in self.schema(db_id).tables:
            for column in table.columns:
                if not self._is_text_column(column.data_type):
                    continue
                table_ref, column_ref = self._identifiers(
                    db_id, table.name, column.name
                )
                rows = self._fetch(
                    db_id,
                    f"SELECT {column_ref},COUNT(*) FROM {table_ref} "
                    f"WHERE LOWER(CAST({column_ref} AS TEXT))="
                    f"LOWER({self.placeholder}) "
                    f"GROUP BY {column_ref} ORDER BY {column_ref} LIMIT 1",
                    (literal,),
                )
                if rows and str(rows[0][0]).lower() == literal.lower():
                    matches.append(
                        {
                            "table": table.name,
                            "column": column.name,
                            "matched_value": str(rows[0][0]),
                            "count": int(rows[0][1]),
                        }
                    )
                    if len(matches) >= limit:
                        return matches
        return matches

    def _is_text_column(self, data_type: str) -> bool:
        return data_type.upper() not in ("INTEGER", "REAL", "NUMERIC", "FLOAT")

    def literal_exists(self, db_id: str, table: str, column: str, value: str) -> bool:
        table_ref, column_ref = self._identifiers(db_id, table, column)
        return bool(
            self._fetch(
                db_id,
                f"SELECT 1 FROM {table_ref} WHERE {column_ref}="
                f"{self.placeholder} LIMIT 1",
                (value,),
            )
        )


class SQLiteBackend(_SQLBackend):
    def _is_text_column(self, data_type: str) -> bool:
        # SQLite TEXT affinity: INT takes precedence over CHAR/CLOB/TEXT.
        declared = data_type.upper()
        return "INT" not in declared and any(
            marker in declared for marker in ("CHAR", "CLOB", "TEXT")
        )

    def __init__(
        self,
        root: Path | None = None,
        *,
        paths: Mapping[str, Path] | None = None,
        timeout_seconds: float = 3,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Backend timeout must be positive")
        self.root = root
        self.paths = dict(paths or {})
        self.timeout_seconds = timeout_seconds

    def _quote(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _path(self, db_id: str) -> Path:
        if db_id in self.paths:
            return self.paths[db_id]
        if self.root is None or Path(db_id).name != db_id or db_id in ("", ".", ".."):
            raise DatabaseBackendError("Unknown SQLite database")
        paths = list((self.root / db_id).glob("*.sqlite"))
        if not paths:
            raise DatabaseBackendError("SQLite database not found")
        return paths[0]

    def _fetch(
        self,
        db_id: str,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> list[tuple[Any, ...]]:
        deadline = time.monotonic() + self.timeout_seconds
        try:
            connection = sqlite3.connect(
                self._path(db_id).resolve().as_uri() + "?mode=ro",
                uri=True,
                timeout=self.timeout_seconds,
            )
            try:
                connection.set_progress_handler(
                    lambda: int(time.monotonic() >= deadline), 1000
                )
                return connection.execute(query, params).fetchall()
            finally:
                connection.close()
        except sqlite3.Error:
            raise DatabaseBackendError("SQLite operation failed") from None

    def _load_schema(self, db_id: str) -> SchemaInfo:
        names = self._fetch(
            db_id,
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'",
        )
        schema = SchemaInfo(db_id=db_id)
        for (name,) in names:
            info = self._fetch(db_id, f"PRAGMA table_info({self._quote(name)})")
            keys = self._fetch(db_id, f"PRAGMA foreign_key_list({self._quote(name)})")
            foreign_keys = [
                {
                    "id": row[0],
                    "seq": row[1],
                    "table": row[2],
                    "from": row[3],
                    "to": row[4],
                }
                for row in keys
            ]
            lookup = {row[3]: f"{row[2]}.{row[4]}" for row in keys}
            schema.tables.append(
                TableSchema(
                    name=name,
                    columns=[
                        ColumnSchema(
                            name=row[1],
                            data_type=row[2] or "TEXT",
                            is_primary_key=bool(row[5]),
                            nullable=not bool(row[3]),
                            is_foreign_key=row[1] in lookup,
                            foreign_key_target=lookup.get(row[1]),
                        )
                        for row in info
                    ],
                    primary_keys=[
                        row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]
                    ],
                    foreign_keys=foreign_keys,
                )
            )
        for table in schema.tables:
            for key in table.foreign_keys:
                if key["to"] is not None:
                    continue
                parent = schema.get_table(key["table"])
                if parent is None or key["seq"] >= len(parent.primary_keys):
                    raise DatabaseBackendError("SQLite foreign key target is invalid")
                key["to"] = parent.primary_keys[key["seq"]]
                column = table.get_column(key["from"])
                if column is not None:
                    column.foreign_key_target = f"{parent.name}.{key['to']}"
        return schema


class PostgreSQLBackend(_SQLBackend):
    placeholder = "%s"

    def __init__(
        self,
        databases: Mapping[str, str],
        *,
        schema: str = "public",
        tables: frozenset[str] | None = None,
        timeout_seconds: float = 3,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Backend timeout must be positive")
        try:
            import psycopg
            from psycopg import sql
        except ImportError:
            raise DatabaseBackendError(
                "Install aap-sql[postgres] for PostgreSQL support"
            ) from None
        self._driver = psycopg
        self._sql = sql
        self._databases = dict(databases)
        self.schema_name = schema
        self.allowed_tables = tables
        self.timeout_seconds = timeout_seconds

    def _quote(self, identifier: str) -> str:
        return self._sql.Identifier(identifier).as_string()

    def _table_ref(self, table: str) -> str:
        return self._sql.Identifier(self.schema_name, table).as_string()

    def _fetch(
        self,
        db_id: str,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> list[tuple[Any, ...]]:
        if db_id not in self._databases:
            raise DatabaseBackendError("Unknown PostgreSQL database")
        try:
            with self._driver.connect(
                self._databases[db_id],
                connect_timeout=max(1, int(self.timeout_seconds)),
            ) as connection:
                connection.execute("SET TRANSACTION READ ONLY")
                connection.execute("SET LOCAL search_path TO pg_catalog")
                connection.execute(
                    "SELECT set_config('statement_timeout',%s,true)",
                    (str(max(1, int(self.timeout_seconds * 1000))),),
                )
                return [
                    tuple(row) for row in connection.execute(query, params).fetchall()
                ]
        except self._driver.Error:
            raise DatabaseBackendError("PostgreSQL operation failed") from None

    def _is_text_column(self, data_type: str) -> bool:
        return data_type.lower().split("(")[0] not in (
            "smallint",
            "integer",
            "bigint",
            "real",
            "double precision",
            "numeric",
            "decimal",
            "boolean",
        )

    def literal_exists(self, db_id: str, table: str, column: str, value: str) -> bool:
        table_ref, column_ref = self._identifiers(db_id, table, column)
        return bool(
            self._fetch(
                db_id,
                f"SELECT 1 FROM {table_ref} "
                f"WHERE CAST({column_ref} AS TEXT)=%s LIMIT 1",
                (value,),
            )
        )

    def _load_schema(self, db_id: str) -> SchemaInfo:
        rows = self._fetch(
            db_id,
            """
            SELECT c.relname,a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod),
                   a.attnotnull,COALESCE(array_position(pk.indkey::smallint[],a.attnum)+1,0)
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
            JOIN pg_catalog.pg_attribute a ON a.attrelid=c.oid
            LEFT JOIN pg_catalog.pg_index pk ON pk.indrelid=c.oid AND pk.indisprimary
            WHERE n.nspname=%s AND c.relkind IN ('r','p')
                  AND a.attnum>0 AND NOT a.attisdropped
                  AND has_table_privilege(c.oid,'SELECT')
            ORDER BY c.relname,a.attnum
        """,
            (self.schema_name,),
        )
        tables: dict[str, TableSchema] = {}
        primary_keys: dict[str, list[tuple[int, str]]] = {}
        for name, column, data_type, not_null, primary_position in rows:
            if self.allowed_tables is not None and name not in self.allowed_tables:
                continue
            table = tables.setdefault(
                name,
                TableSchema(
                    name=name, identifier_quote='"', case_sensitive_identifiers=True
                ),
            )
            table.columns.append(
                ColumnSchema(
                    name=column,
                    data_type=data_type,
                    nullable=not not_null,
                    is_primary_key=bool(primary_position),
                )
            )
            if primary_position:
                primary_keys.setdefault(name, []).append((primary_position, column))
        if self.allowed_tables is not None and set(tables) != set(self.allowed_tables):
            raise DatabaseBackendError("Requested PostgreSQL tables are unavailable")
        for name, positions in primary_keys.items():
            tables[name].primary_keys = [column for _, column in sorted(positions)]
        keys = self._fetch(
            db_id,
            """
            SELECT child.relname,parent.relname,a.attname,target.attname,
                   con.oid::text,key.ordinality
            FROM pg_catalog.pg_constraint con
            JOIN pg_catalog.pg_class child ON child.oid=con.conrelid
            JOIN pg_catalog.pg_namespace cn ON cn.oid=child.relnamespace
            JOIN pg_catalog.pg_class parent ON parent.oid=con.confrelid
            JOIN pg_catalog.pg_namespace pn ON pn.oid=parent.relnamespace
            CROSS JOIN LATERAL unnest(con.conkey,con.confkey)
                WITH ORDINALITY AS key(attnum,target_attnum,ordinality)
            JOIN pg_catalog.pg_attribute a
                ON a.attrelid=child.oid AND a.attnum=key.attnum
            JOIN pg_catalog.pg_attribute target
                ON target.attrelid=parent.oid AND target.attnum=key.target_attnum
            WHERE con.contype='f' AND cn.nspname=%s AND pn.nspname=%s
            ORDER BY child.relname,con.oid,key.ordinality
        """,
            (self.schema_name, self.schema_name),
        )
        for child, parent, column, target_column, identity, position in keys:
            if child not in tables or parent not in tables:
                continue
            tables[child].foreign_keys.append(
                {
                    "from": column,
                    "table": parent,
                    "to": target_column,
                    "id": identity,
                    "seq": str(position),
                }
            )
            field = tables[child].get_column(column)
            if field:
                field.is_foreign_key = True
                field.foreign_key_target = f"{parent}.{target_column}"
        return SchemaInfo(
            db_id=db_id, tables=list(tables.values()), case_sensitive_identifiers=True
        )
