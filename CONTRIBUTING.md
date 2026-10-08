# Contributing

Use Python 3.12+ and install the development tools:

```bash
python -m pip install -e '.[test,dev,postgres]'
make check
```

`make check` runs core formatting, lint, strict typing, tests, offline smoke,
and compilation. PostgreSQL integration uses `TEST_DATABASE_URL` pointing to
a disposable database whose name ends in `_test`; CI supplies PostgreSQL 17.
Install `.[training]` to exercise SQL training-data preparation and the local
Transformer training tests.

Keep changes focused on the reusable AAP pipeline or research utilities. Include
a regression check for behavior changes, and update the relevant method or
reproduction documentation when changing defaults. Preserve input provenance,
read-only database access, bounded probes, and provider-independent core imports.

For bug reports, provide the command, Python/package versions, expected result,
and a small synthetic database or fixture. Keep credentials and private database
rows local. Performance reports should identify the model, dataset release,
configuration, question count, and input hashes.
