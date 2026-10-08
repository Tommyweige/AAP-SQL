"""SQLite metadata and benchmark paths; live core operations use backends."""

from pathlib import Path

from ..config import get_config


def foreign_key_rows(conn, table_name):
    """Resolve SQLite REFERENCES table shorthand to its ordered primary key."""
    quoted = table_name.replace('"', '""')
    rows = []
    for fk in conn.execute(f'PRAGMA foreign_key_list("{quoted}")'):
        row = list(fk)
        if row[4] is None:
            target = row[2].replace('"', '""')
            primary = sorted(
                (c for c in conn.execute(f'PRAGMA table_info("{target}")') if c[5]),
                key=lambda c: c[5],
            )
            if row[1] >= len(primary):
                raise ValueError(
                    f"Unresolved primary-key reference: {table_name} -> {row[2]}"
                )
            row[4] = primary[row[1]][1]
        rows.append(tuple(row))
    return rows


def get_db_path(db_id: str, database=None) -> Path:
    """
    Get the database file path for a given database ID.

    Args:
        db_id: Database identifier (e.g., 'california_schools')

    Returns:
        Path to the SQLite database file
    """
    database = database or get_config().database
    if database.dev_db_path is None:
        raise FileNotFoundError("No SQLite database root configured")
    db_dir = database.dev_db_path / db_id

    # Find .sqlite file in the directory
    sqlite_files = list(db_dir.glob("*.sqlite"))
    if sqlite_files:
        return sqlite_files[0]

    # Try direct path
    direct_path = db_dir / f"{db_id}.sqlite"
    if direct_path.exists():
        return direct_path

    raise FileNotFoundError(f"Database not found for db_id: {db_id}")
