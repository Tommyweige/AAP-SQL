"""Normalize generated SQL without executing it."""

import re


def clean_sql(sql_text: str) -> str:
    if not sql_text:
        return ""
    sql = sql_text.strip()

    # Handle "SQL:" prefix from new prompt format
    if "SQL:" in sql:
        parts = sql.split("SQL:", 1)
        if len(parts) > 1 and parts[1].strip():
            sql = parts[1].strip()

    # Handle Markdown blocks
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        parts = sql.split("```")
        if len(parts) >= 2:
            sql = parts[1].strip()

    # Final cleanup of common prefixes
    match = re.search(r"\b(SELECT|WITH)\b", sql, re.IGNORECASE)
    if match:
        sql = sql[match.start() :]

    return sql.strip()
