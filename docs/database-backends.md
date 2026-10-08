# Database backends and application integration

The reusable core accepts an `AAPDatabaseBackend`. SQLite remains the default;
PostgreSQL requires the optional driver:

```bash
python -m pip install -e '.[postgres]'
```

Supply models, a column mapping, and an index as in the README. Replace only the
database configuration for PostgreSQL:

```python
import os
from aap_sql import DatabaseConfig, PostgreSQLBackend

backend = PostgreSQLBackend(
    {"analytics": os.environ["DATABASE_QUERY_URL"]},
    schema="public",
    tables=frozenset({"people", "groups"}),
    timeout_seconds=3,
)
database = DatabaseConfig(dev_db_path=None, backend=backend)
# AAPSQLPipeline(database=database, ...).prepare(question, "analytics")
```

Use a dedicated SELECT-only role. Operations use read-only transactions, bound
values, schema-qualified quoted identifiers, connection timeouts, and a local
statement timeout. Table/column identifiers must exist in introspected schema.
The optional table allowlist limits introspection, probing, literal lookup, and
relationship cues. Every requested table must be accessible. Database grants and the application's
SQL validation/execution guard control generated-query execution.

SQLite accepts a database root or explicit `paths={db_id: Path(...)}` and opens
files read-only. Introspection preserves table creation order, SQLite DDL quoting,
composite keys, and implicit foreign-key references to a parent's primary key.
PostgreSQL DDL uses double quotes and catalog types. Retrieval limits (50 recalled,
10 reranked by default), value limits, literal hints, relationship cues, and prompt
section order remain shared across backends.

The default method follows the thesis: filter the global retrieval Top-50 by
database, then rerank to Top-10 in a single recall pass. Quoted literals are deduplicated case-insensitively, matched
across text columns, and capped at five candidates per value; hints preserve
stored case and become prompt hints directly. Implicit relationship cues
require both endpoint tables in the selected columns. PostgreSQL and optional
application hooks extend the reusable core beyond the SQLite research setting.

Schema introspection is cached per backend and database. Create a new backend
after a schema-changing migration. A connection is opened per operation; use
application-level serialization according to the shared model's concurrency requirements.
Applications manage model loading, cloud clients, and generated-SQL execution.

`DatabaseBackendError` reports sanitized database/driver failures;
`UnknownIdentifierError` rejects unknown identifiers. Backend failures propagate to the application.
Applications should catch these errors and return their own safe response.

## Context hooks and traces

`PreparedQuery` retains its original four positional fields and adds defaulted
`probed_values`, `declared_relations`, `undeclared_relations`, `literal_candidates`,
and `evidence`. Pipeline retrieval metadata also contains `candidate_columns`
before reranking. These expose the context used for generation. Applications decide which context is safe to return to their users.

`prepare()` accepts optional keyword-only integration hooks:

- `external_literals`: candidate dictionaries with `original_literal`,
  `matched_value`, `table`, `column`, and `count`. Candidates are still verified
  against stored values before becoming hints.
- `semantic_evidence` and `temporal_evidence`: strings appended to base evidence.
- `value_prober(db_id, question, selected_columns) -> list[str]`: a question-aware
  replacement for standard probing. The application must preserve its own limits
  and use backend operations when querying data.

Domain aliases, date resolution, resource descriptions, and model/index lifecycle
belong to the integrating application.

## Validation

```bash
make install
make check
```

Formatting, lint, and strict typing cover the reusable integration modules and
new backend tests. Research utilities have runtime regression coverage;
the complete test suite and offline smoke command also run.

The PostgreSQL integration test uses an actual database, compares equivalent
SQLite/PostgreSQL pipeline results, and exercises quoted identifiers, composite
keys, restricted tables, and sanitized failures. Set `TEST_DATABASE_URL` to a
disposable database **whose name ends in `_test`** to enable it. Each run creates
and drops its own unique schema. Without that variable or the optional driver,
PostgreSQL integration coverage is skipped. CI provides PostgreSQL 17 and runs all
checks; the fixture uses synthetic model adapters.
