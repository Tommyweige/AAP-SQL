"""Reusable AAP-SQL orchestration; models and SQL generation are caller-owned."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .components.hybrid_prober import probe_literal_candidates
from .components.literal_retrieval import format_literal_hints, get_literal_candidates
from .components.schema_loader import load_schema
from .components.value_prober import probe_columns
from .config import DatabaseConfig
from .prompt import build_prompt
from .relations import declared_relations, load_relation_graph, undeclared_relations
from .retrieval import Encoder, Reranker, SearchIndex, retrieve_batch
from .settings import PipelineConfig, final_configurations
from .sql import clean_sql


@dataclass
class PreparedQuery:
    prompt: str
    columns: list[dict[str, Any]]
    literals: list[str]
    retrieval: dict[str, Any]
    probed_values: list[str] = field(default_factory=list)
    declared_relations: list[str] = field(default_factory=list)
    undeclared_relations: list[str] = field(default_factory=list)
    literal_candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence: str = ""


@dataclass
class AAPSQLPipeline:
    """Inject a bi-encoder, index, mapping, cross-encoder and vector normalizer.

    No credentials, downloads or network requests occur during construction.
    prepare() creates the final prompt; generate() calls the supplied generator.
    Model adapters should be externally serialized if shared between threads.
    """

    database: DatabaseConfig
    model: Encoder
    index: SearchIndex
    mapping: list[dict[str, Any]]
    reranker: Reranker
    normalize: Callable[[Any], None]
    relation_path: Path | None = None
    config: PipelineConfig = field(
        default_factory=lambda: final_configurations()[0], init=False
    )

    def prepare(
        self,
        question: str,
        db_id: str,
        evidence: str = "",
        *,
        external_literals: list[dict[str, Any]] | None = None,
        semantic_evidence: str = "",
        temporal_evidence: str = "",
        value_prober: Callable[[str, str, list[dict[str, Any]]], list[str]]
        | None = None,
    ) -> PreparedQuery:
        columns, meta = retrieve_batch(
            [{"question": question, "db_id": db_id}],
            self.config,
            model=self.model,
            index=self.index,
            mapping=self.mapping,
            reranker=self.reranker,
            normalize=self.normalize,
            include_candidates=True,
        )
        descriptions = {
            c["full_name"].lower().strip(): c["description"]
            for c in self.mapping
            if c["db_id"] == db_id
        }
        candidates = get_literal_candidates(
            db_id, question, database=self.database, limit=5
        )
        # Backend lookup already verifies quoted values. Only caller-supplied
        # aliases need separate verification; neither path is semantically reranked.
        candidates += probe_literal_candidates(
            db_id,
            [dict(candidate) for candidate in (external_literals or [])],
            database=self.database,
        )
        literals = format_literal_hints(candidates, descriptions)
        declared = declared_relations(db_id, self.database)
        undeclared = undeclared_relations(
            load_relation_graph(self.relation_path), db_id, columns[0]
        )
        probed = (
            value_prober(db_id, question, columns[0])
            if value_prober
            else probe_columns(db_id, columns[0], database=self.database)
        )
        combined_evidence = "\n".join(
            part for part in (evidence, semantic_evidence, temporal_evidence) if part
        )
        prompt = build_prompt(
            question,
            columns[0],
            self.config,
            schema=load_schema(db_id, self.database),
            foreign_keys=declared + undeclared,
            probed_values=probed,
            evidence=combined_evidence,
            literals=literals,
        )
        return PreparedQuery(
            prompt,
            columns[0],
            literals,
            meta[0],
            probed_values=probed,
            declared_relations=declared,
            undeclared_relations=undeclared,
            literal_candidates=candidates,
            evidence=combined_evidence,
        )

    def generate(
        self,
        question: str,
        db_id: str,
        generator: Callable[[str], str],
        evidence: str = "",
    ) -> str:
        """Generate SQL through a caller-supplied provider; does not execute SQL."""
        return clean_sql(generator(self.prepare(question, db_id, evidence).prompt))
