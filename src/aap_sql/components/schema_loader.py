"""Load and format introspected schema through the configured backend."""

from copy import deepcopy

from aap_sql.backends import get_backend
from aap_sql.config import DatabaseConfig, get_config
from aap_sql.utils.schema import enrich_schema_with_descriptions

from .description_loader import get_description_loader


def load_schema(db_id: str, database: DatabaseConfig | None = None) -> str:
    database = database or get_config().database
    schema = get_backend(database).schema(db_id)
    if database.dev_tables_path is not None:
        descriptions = get_description_loader(
            database.dev_tables_path
        ).get_descriptions(db_id)
        schema = enrich_schema_with_descriptions(deepcopy(schema), descriptions)
    return schema.to_ddl()
