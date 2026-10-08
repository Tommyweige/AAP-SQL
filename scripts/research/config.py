"""Paths for the published BIRD dev_20240627 reproduction only."""
from dataclasses import dataclass, field
from pathlib import Path
from aap_sql.config import Config, DatabaseConfig, PROJECT_ROOT

@dataclass
class BirdDatabaseConfig(DatabaseConfig):
    dev_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "BIRD/dev/dev_20240627/dev_databases/dev_databases")
    dev_tables_path: Path = field(default_factory=lambda: PROJECT_ROOT / "BIRD/dev/dev_20240627/dev_tables.json")
    dev_json_path: Path = field(default_factory=lambda: PROJECT_ROOT / "BIRD/dev/dev_20240627/dev.json")

def bird_config():
    return Config(database=BirdDatabaseConfig())
