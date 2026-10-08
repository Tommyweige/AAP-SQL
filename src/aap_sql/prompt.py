"""Prompt formatting for the final AAP-SQL pipeline."""

from collections.abc import Iterable
from typing import Any

from .settings import PipelineConfig


def build_prompt(
    question: str,
    columns: list[dict[str, Any]],
    config: PipelineConfig,
    *,
    schema: str,
    foreign_keys: Iterable[str] = (),
    probed_values: Iterable[str] = (),
    evidence: str = "",
    literals: list[str] | None = None,
) -> str:
    """Build prompt based on config. Accepts pre-computed literals."""
    col_lines = [f"- {c['full_name']}: {c['description']}" for c in columns]

    # Thesis A-2 removes the entire extra FK block, retaining DDL constraints.
    fk_text = ""
    if config.description == "D2" and config.fk != "FK0":
        fk_info = list(foreign_keys)
        if fk_info:
            fk_text = "\n### Foreign Keys:\n" + "\n".join(fk_info)

    # Evidence section
    evidence_text = ""
    if config.evidence == "V1" and evidence:
        evidence_text = f"\n### Evidence:\n{evidence}"

    # Value Probing section (P0/P1)
    probing_text = ""
    if config.probing == "P1":
        probed = list(probed_values)
        if probed:
            probing_text = (
                "\n### Probed Values (ACTUAL data in database):\n" + "\n".join(probed)
            )

    # Literal Retrieval section (L0/L1)
    literal_text = ""
    if config.literal == "L1" and literals:
        literal_text = "\n### Found Values (from question):\n" + "\n".join(
            f"- {lit}" for lit in literals
        )

    instructions = "### SQL:"
    task = ""
    if config.prompt_style in ("task", "cot"):
        task = (
            "Task:\nYou are an expert Text-to-SQL generator. Given the database "
            "schema, foreign keys, column information, probed values, found "
            "values, evidence, and the user question, "
        )
        if config.prompt_style == "task":
            task += (
                "generate one correct and executable SQLite SQL query.\n"
                "Output Requirements:\n- Use the provided schema, foreign keys, "
                "evidence, found values, and probed values when relevant.\n"
                "- Output exactly one valid SQLite SQL query.\n"
                "- Do not output explanations, analysis, markdown, "
                "or intermediate reasoning.\n"
                "- Start directly with SELECT or WITH and end with a semicolon.\n"
            )
        else:
            task += (
                "reason step by step and then generate one correct and "
                "executable SQLite SQL query.\nReasoning Requirements:\n"
                "Before writing the final SQL, briefly identify:\n"
                "- the required tables;\n- the required join paths;\n"
                "- the required filters;\n- the required aggregation or grouping;\n"
                "- the selected output columns.\nOutput Requirements:\n"
                "- After the reasoning, output the final SQL under the marker:\n"
                "Final SQL:\n"
                "- The final SQL must be exactly one valid SQLite SQL query.\n"
                "- The final SQL must start with SELECT or WITH "
                "and end with a semicolon.\n"
                "- Do not put markdown code fences around the SQL.\n"
            )
    columns_text = ""
    if config.embedding != "E0":
        columns_text = "\n### Columns:\n" + "\n".join(col_lines)

    return f"""{task}### Schema:
{schema}
{fk_text}
{columns_text}
{probing_text}
{literal_text}
{evidence_text}
### Question:
{question}

{instructions}"""
