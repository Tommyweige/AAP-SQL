from dataclasses import dataclass, replace


@dataclass
class PipelineConfig:
    name: str
    embedding: str
    reranking: str
    evidence: str
    probing: str
    literal: str
    fk: str = "FK2"
    description: str = "D2"
    prompt_style: str = "plain"


def final_configurations() -> list[PipelineConfig]:
    """Return the single configuration reported by this publication branch."""
    return [
        PipelineConfig(
            "E2-R1-V1-P1-L1-FK2",
            "E2",
            "R1",
            "V1",
            "P1",
            "L1",
            fk="FK2",
        )
    ]


def thesis_configurations() -> list[PipelineConfig]:
    """Chapter 4 ablations; selecting them never changes the core default."""
    full = final_configurations()[0]
    variants = [
        full,
        replace(full, name="no-reranking", reranking="R0"),
        replace(full, name="no-probing", probing="P0"),
        replace(full, name="no-literals", literal="L0"),
        replace(full, name="generic-retriever", embedding="E1"),
        replace(full, name="ddl-fk-only", fk="FK0"),
    ]
    variants += [
        replace(c, name=c.name + "-no-evidence", evidence="V0") for c in variants
    ]
    baseline = replace(
        full,
        name="baseline-with-evidence",
        embedding="E0",
        reranking="R0",
        probing="P0",
        literal="L0",
        fk="FK0",
        description="D0",
    )
    return variants + [
        baseline,
        replace(baseline, name="baseline", evidence="V0"),
        replace(full, name="task-instructions", prompt_style="task"),
        replace(full, name="explicit-cot", prompt_style="cot"),
    ]
