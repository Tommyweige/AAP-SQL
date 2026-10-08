PYTHON ?= python
CORE_MODULES = aap_sql.backends aap_sql.config aap_sql.pipeline aap_sql.retrieval \
 aap_sql.relations aap_sql.prompt aap_sql.sql aap_sql.settings \
 aap_sql.components.schema_loader aap_sql.components.value_prober \
 aap_sql.components.literal_retrieval aap_sql.components.hybrid_prober
CORE_FILES = $(addprefix src/,$(addsuffix .py,$(subst .,/,$(CORE_MODULES)))) \
 src/aap_sql/__init__.py tests/test_backends.py tests/test_postgres_backend.py

.PHONY: install format lint typecheck test smoke check

install:
	$(PYTHON) -m pip install -e '.[test,postgres,dev]'

format:
	$(PYTHON) -m ruff format $(CORE_FILES)

lint:
	$(PYTHON) -m ruff format --check $(CORE_FILES)
	$(PYTHON) -m ruff check --select E,F,I,UP $(CORE_FILES)

typecheck:
	MYPYPATH=src $(PYTHON) -m mypy --strict --follow-imports=silent $(addprefix -m ,$(CORE_MODULES))
	MYPYPATH=src $(PYTHON) -m mypy --strict --follow-imports=silent tests/test_backends.py tests/test_postgres_backend.py

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m aap_sql.smoke
	$(PYTHON) -m compileall -q src scripts

check: lint typecheck test smoke
