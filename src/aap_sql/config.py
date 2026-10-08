"""Runtime paths and limits for the final AAP-SQL publication runner."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends import AAPDatabaseBackend


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DatabaseConfig:
    """Optional SQLite root, semantic metadata, or an injected relational backend."""

    dev_db_path: Path | None = field(default_factory=lambda: Path.cwd() / "databases")
    dev_tables_path: Path | None = None
    query_timeout: int = 3
    backend: "AAPDatabaseBackend | None" = None


@dataclass
class AAPConfig:
    """Small runtime limits shared by retrieval components."""

    literal_retrieval_limit: int = 5


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    aap: AAPConfig = field(default_factory=AAPConfig)


_default_config: Config | None = None


def get_config() -> Config:
    """Return the process-wide runtime configuration."""

    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Replace the process-wide runtime configuration."""

    global _default_config
    _default_config = config


def resolve_resource_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve project resources independently of the working directory."""
    path = Path(value)
    return path if path.is_absolute() else root / path
