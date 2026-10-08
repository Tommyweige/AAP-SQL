"""
Schema loading and processing utilities for AAP-SQL.
"""

import re
from dataclasses import dataclass, field

# ============ Selective Augmentation: Column Classification ============
# Regex patterns for opaque columns (cryptic names that NEED descriptions)
OPAQUE_PATTERNS = [
    r"^[A-Za-z]\d+$",  # A1, A11, c2, f14
    r"^col_?\d+$",  # col_1, col1
    r"^field\d*$",  # field1, field
    r"^[A-Z]{1,3}\d{1,3}$",  # DIM01, VAR002
    r"^\w_\d+$",  # x_1, f_2
]

# Common clear column name suffixes/tokens (do NOT need descriptions)
CLEAR_SUFFIXES = [
    "_id",
    "_date",
    "_at",
    "_name",
    "_count",
    "_total",
    "_amount",
    "_flag",
    "_code",
]
CLEAR_WORDS = {
    "name",
    "city",
    "country",
    "email",
    "phone",
    "address",
    "price",
    "age",
    "gender",
    "status",
    "type",
    "id",
    "date",
    "time",
    "year",
    "month",
    "day",
    "title",
    "description",
    "school",
    "county",
    "street",
    "state",
    "zip",
    "first",
    "last",
    "full",
}


def classify_column(col_name: str) -> str:
    """Classify column into 'clear', 'opaque', or 'ambiguous'.

    Returns:
        'opaque': Column needs description (e.g., A1, A11, DIM01)
        'clear': Column is self-descriptive (e.g., name, city, email)
        'ambiguous': Column may or may not need description
    """
    lower = col_name.lower()

    # Check Opaque patterns first (highest priority)
    for pattern in OPAQUE_PATTERNS:
        if re.match(pattern, col_name, re.IGNORECASE):
            return "opaque"

    # Check Clear suffixes
    for suffix in CLEAR_SUFFIXES:
        if lower.endswith(suffix):
            return "clear"

    # Check Clear dictionary words
    tokens = re.split(r"[_\s]+", lower)
    if all(t in CLEAR_WORDS or len(t) <= 2 for t in tokens):
        return "clear"

    # Default to ambiguous (inject description to be safe)
    return "ambiguous"


@dataclass
class ColumnSchema:
    """Schema information for a database column."""

    name: str
    data_type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_target: str | None = None
    nullable: bool = True
    short_description: str | None = None


@dataclass
class TableSchema:
    """Schema information for a database table."""

    name: str
    columns: list[ColumnSchema] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[dict[str, str]] = field(default_factory=list)
    identifier_quote: str = "`"
    case_sensitive_identifiers: bool = False

    def quote_identifier(self, value: str) -> str:
        quote = self.identifier_quote
        return quote + value.replace(quote, quote + quote) + quote

    def get_column(self, column_name: str) -> ColumnSchema | None:
        """Get column by name."""
        for col in self.columns:
            if col.name == column_name or (
                not self.case_sensitive_identifiers
                and col.name.lower() == column_name.lower()
            ):
                return col
        return None

    def to_ddl(self) -> str:
        """Generate CREATE TABLE DDL statement with comments."""
        lines = [f"CREATE TABLE {self.quote_identifier(self.name)} ("]

        # Prepare column definitions
        col_lines = []
        for i, col in enumerate(self.columns):
            col_def = f"  {self.quote_identifier(col.name)} {col.data_type}"
            if not col.nullable:
                col_def += " NOT NULL"
            if col.is_primary_key and len(self.primary_keys) == 1:
                col_def += " PRIMARY KEY"

            if (
                i < len(self.columns) - 1
                or len(self.primary_keys) > 1
                or self.foreign_keys
            ):
                col_def += ","

            # Selective Augmentation: Only inject descriptions for opaque/ambiguous columns
            if col.short_description:
                col_category = classify_column(col.name)
                if col_category in ("opaque", "ambiguous"):
                    col_def += f" -- {col.short_description}"
                # 'clear' columns: skip description to avoid hallucination

            col_lines.append(col_def)

        lines.extend(col_lines)

        # Add composite primary key if needed
        if len(self.primary_keys) > 1:
            pk_cols = ", ".join(self.quote_identifier(pk) for pk in self.primary_keys)
            pk_line = f"  PRIMARY KEY ({pk_cols})"
            if self.foreign_keys:  # If FKs follow, need comma
                pk_line += ","
            lines.append(pk_line)

        # Preserve compound foreign-key identity and ordinal column order.
        groups: dict[str, list[dict[str, str]]] = {}
        for i, fk in enumerate(self.foreign_keys):
            groups.setdefault(str(fk.get("id", str(i))), []).append(fk)
        for i, group in enumerate(groups.values()):
            ordered = sorted(group, key=lambda fk: int(fk.get("seq", "0")))
            sources = ", ".join(self.quote_identifier(fk["from"]) for fk in ordered)
            targets = ", ".join(self.quote_identifier(fk["to"]) for fk in ordered)
            fk_def = (
                f"  FOREIGN KEY ({sources}) "
                f"REFERENCES {self.quote_identifier(ordered[0]['table'])} ({targets})"
            )
            if i < len(groups) - 1:
                fk_def += ","
            lines.append(fk_def)

        lines.append(");")

        return "\n".join(lines)


@dataclass
class SchemaInfo:
    """Complete schema information for a database."""

    db_id: str
    tables: list[TableSchema] = field(default_factory=list)
    case_sensitive_identifiers: bool = False

    def get_table(self, table_name: str) -> TableSchema | None:
        """Get table by name."""
        for table in self.tables:
            if table.name == table_name or (
                not self.case_sensitive_identifiers
                and table.name.lower() == table_name.lower()
            ):
                return table
        return None

    def to_ddl(self) -> str:
        """Generate DDL for all tables."""
        return "\n\n".join(table.to_ddl() for table in self.tables)


def enrich_schema_with_descriptions(
    schema: SchemaInfo, descriptions: dict[str, str]
) -> SchemaInfo:
    """
    Add short descriptions to schema columns.

    Args:
        schema: SchemaInfo object
        descriptions: Dictionary mapping "table.column" to description

    Returns:
        Updated SchemaInfo object
    """
    for table in schema.tables:
        for col in table.columns:
            key = f"{table.name}.{col.name}".lower()
            if key in descriptions:
                col.short_description = descriptions[key]

    return schema
