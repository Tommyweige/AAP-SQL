"""Batch retrieval/reranking with injected model and search-index adapters."""

from collections.abc import Callable
from typing import Any, Protocol

from .settings import PipelineConfig


class Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


class SearchIndex(Protocol):
    def search(self, vectors: Any, k: int) -> tuple[Any, Any]: ...


class Reranker(Protocol):
    def predict(self, pairs: list[list[str]], **kwargs: Any) -> Any: ...


def retrieve_batch(
    questions: list[dict[str, Any]],
    config: PipelineConfig,
    *,
    model: Encoder,
    index: SearchIndex,
    mapping: list[dict[str, Any]],
    reranker: Reranker,
    normalize: Callable[[Any], None],
    candidate_limit: int = 50,
    final_limit: int = 10,
    include_candidates: bool = False,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Pre-compute retrieval for all questions in batch."""
    emb_model = model
    # 1. Batch Encode
    input_texts = [q["question"] for q in questions]
    q_embs = emb_model.encode(
        input_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False
    )
    normalize(q_embs)

    # 2. Batch Search
    k = candidate_limit if config.reranking == "R1" else final_limit
    distances, indices_arr = index.search(q_embs, k=k)

    all_candidates = []
    retrieval_meta: list[dict[str, Any]] = []

    # 3. Process Initial Results
    for i, (dists, idxs) in enumerate(zip(distances, indices_arr)):
        db_id = questions[i]["db_id"]
        current_candidates = []
        retrieval_mode = "direct"
        for dist, idx in zip(dists, idxs):
            if idx < 0 or idx >= len(mapping):
                continue
            info = mapping[idx]
            if info["db_id"] == db_id:  # Filter by matching DB
                current_candidates.append(
                    {
                        "full_name": info["full_name"],
                        "description": info["description"],
                        "score": float(dist),
                    }
                )

        # The thesis filters the global Top-k once; it does not widen recall.
        if not current_candidates:
            retrieval_mode = "empty"

        all_candidates.append(current_candidates)
        retrieval_meta.append(
            {"mode": retrieval_mode, "candidate_count": len(current_candidates)}
        )
        if include_candidates:
            retrieval_meta[-1]["candidate_columns"] = [
                candidate["full_name"] for candidate in current_candidates
            ]

    # 4. Batch Rerank (if enabled)
    if config.reranking == "R1":
        pairs = []
        pair_indices = []  # (question_idx, candidate_idx)

        for i, cands in enumerate(all_candidates):
            q_text = questions[i]["question"]
            for j, c in enumerate(cands):
                pairs.append([q_text, c["description"]])
                pair_indices.append((i, j))

        if pairs:
            scores = reranker.predict(pairs, batch_size=128, show_progress_bar=False)
            for (q_idx, c_idx), score in zip(pair_indices, scores):
                all_candidates[q_idx][c_idx]["score"] = float(score)

            # Re-sort and slice
            for i in range(len(all_candidates)):
                all_candidates[i] = sorted(
                    all_candidates[i], key=lambda x: x["score"], reverse=True
                )[:final_limit]
    else:
        # Just slice if no reranking
        for i in range(len(all_candidates)):
            all_candidates[i] = all_candidates[i][:final_limit]

    return all_candidates, retrieval_meta
