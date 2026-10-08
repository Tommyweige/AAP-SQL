#!/usr/bin/env python3
"""Run the thesis-aligned full method or explicitly selected Chapter 4 ablations."""
import argparse
import json
import hashlib
import time
import random
from collections import Counter
import threading
import sqlite3
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import asdict
import os

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aap_sql.utils.database import get_db_path

from aap_sql.components.schema_loader import load_schema
from aap_sql.components.value_prober import probe_columns
from aap_sql.config import get_config

from aap_sql.components.literal_retrieval import (
    get_literal_candidates,
    format_literal_hints
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
E2_MODEL_PATH = BASE_DIR / "models/column_retriever_v3_paper_earlystop"
R1_MODEL_PATH = BASE_DIR / "models/cross_encoder_schema_paper_repro"
MAPPING_BASIC_PATH = BASE_DIR / "resources/bird_dev_20240627/column_mapping.json"
MAPPING_CSV_PATH = BASE_DIR / "resources/bird_dev_20240627/column_mapping_csv_enriched.json"
E4_GRAPH_PATH = BASE_DIR / "outputs/research/thesis/undeclared_relations.json"

DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_INFERENCE_WORKERS = 24
DEFAULT_VALIDATION_WORKERS = 4
DEFAULT_SQL_EXEC_TIMEOUT_SEC = 60

# Only command-line research runs read local environment settings.
if __name__ == "__main__" and not any(arg in sys.argv for arg in ("--help", "-h", "--list-configs")):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


def tqdm(*args, **kwargs):
    from tqdm import tqdm as progress
    return progress(*args, **kwargs)

LLM_PROVIDER = os.getenv("AAP_LLM_PROVIDER", "vertex").strip().lower()
VERTEX_PROJECT = os.getenv("GOOGLE_VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.getenv("GOOGLE_VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION)
genai_client = None

# ============ Configuration ============
from aap_sql.settings import PipelineConfig as ExperimentConfig, final_configurations, thesis_configurations

EXPERIMENTS = thesis_configurations()

MODEL_NAME = os.getenv("AAP_GEMINI_MODEL_NAME", "models/gemini-2.5-flash-lite")
MODEL_DISPLAY_NAME = os.getenv("AAP_GEMINI_MODEL_DISPLAY_NAME", "Gemini-2.5-Flash-Lite")
MODEL_FILE_TAG = os.getenv("AAP_GEMINI_MODEL_FILE_TAG", "gemini_2.5_flash_lite")
MODEL_LOG_DIR = os.getenv("AAP_GEMINI_MODEL_LOG_DIR", "thesis-aligned-gemini-2.5")
TEMPERATURE = 0.0
MAX_TOKENS = 2048
INFERENCE_WORKERS = int(os.getenv("AAP_INFERENCE_WORKERS", str(DEFAULT_INFERENCE_WORKERS)))
VALIDATION_WORKERS = int(
    os.getenv("AAP_VALIDATION_WORKERS", str(DEFAULT_VALIDATION_WORKERS))
)
PRECOMPUTE_WORKERS = 1 # Serial execution for literals to avoid Fortran/MKL crash
VALIDATION_TIMEOUT_SEC = int(os.getenv("AAP_VALIDATION_WATCHDOG_SEC", "3600"))
SQL_EXEC_TIMEOUT_SEC = int(
    os.getenv("AAP_SQL_EXEC_TIMEOUT_SEC", str(DEFAULT_SQL_EXEC_TIMEOUT_SEC))
)
GEMINI_MAX_RETRIES = int(os.getenv("AAP_GEMINI_MAX_RETRIES", "8"))
TOP_K_BIENC = 50
TOP_K_FINAL = 10
PROBE_VALUES_PER_COLUMN = 3

if os.getenv("AAP_PRECOMPUTE_WORKERS"):
    try:
        PRECOMPUTE_WORKERS = max(1, int(os.getenv("AAP_PRECOMPUTE_WORKERS", "1")))
    except Exception:
        pass

# Global lock for model inference (SentenceTransformer/CrossEncoder are not thread-safe)
# Per-db locks: 同一個 SQLite 檔案同時只允許一個執行緒執行 SQL
# 避免 SQLite「database is locked」→ future watchdog 超時 → 正確 SQL 被記為 false
_DB_LOCKS: dict = {}
_DB_LOCKS_LOCK = threading.Lock()

def _get_db_lock(db_id: str) -> threading.Lock:
    with _DB_LOCKS_LOCK:
        if db_id not in _DB_LOCKS:
            _DB_LOCKS[db_id] = threading.Lock()
        return _DB_LOCKS[db_id]

# ============ Model Loading ============
EMBEDDING_MODELS = {}
CROSS_ENCODER = None
INDICES = {}
llm_model = None
MODEL_NAME_API = None
MAPPING_BASIC = None
MAPPING_CSV = None
PROBING_CACHE = {}


def cached_probe_columns(db_id, columns, values_per_column=PROBE_VALUES_PER_COLUMN):
    """Memoize freshly queried values within this process, using the core format."""
    result = []
    for col in columns:
        key = (db_id, col["full_name"], values_per_column)
        if key not in PROBING_CACHE:
            PROBING_CACHE[key] = probe_columns(db_id, [col], values_per_column=values_per_column)
        result.extend(PROBING_CACHE[key])
    return result


def load_models(configs):
    """Load the E2 retriever, R1 reranker, mappings, and Vertex client."""
    global EMBEDDING_MODELS, CROSS_ENCODER, INDICES, llm_model, MODEL_NAME_API, MAPPING_BASIC, MAPPING_CSV

    global faiss
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder

    required_embeddings = {c.embedding for c in configs}
    if "E1" in required_embeddings:
        EMBEDDING_MODELS["E1"] = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("Loading E2 embedding model...")
    if "E2" in required_embeddings:
        e2_path = E2_MODEL_PATH
        if not e2_path.exists():
            raise FileNotFoundError(
                "Column retriever model not found. Download TommyPanLab/AAP-SQL-Column-Retriever to "
                "models/column_retriever_v3_paper_earlystop"
            )
        EMBEDDING_MODELS["E2"] = SentenceTransformer(str(e2_path))

    print("Loading R1 cross-encoder...")
    if any(c.reranking == "R1" for c in configs):
        ce_path = R1_MODEL_PATH
        if not ce_path.exists():
            raise FileNotFoundError(
                "Candidate reranker model not found. Download TommyPanLab/AAP-SQL-Candidate-Reranker to "
                "models/cross_encoder_schema_paper_repro"
            )
        CROSS_ENCODER = CrossEncoder(str(ce_path))

    print("Loading column mappings...")
    with open(MAPPING_BASIC_PATH, "r", encoding='utf-8') as f:
        MAPPING_BASIC = json.load(f)
    with open(MAPPING_CSV_PATH, "r", encoding='utf-8') as f:
        MAPPING_CSV = json.load(f)

    print("Building FAISS indices...")
    mappings = {
        "basic": MAPPING_BASIC,
        "csv": MAPPING_CSV
    }

    for emb_id, model in EMBEDDING_MODELS.items():
        for desc_id, mapping in mappings.items():
            key = f"{emb_id}_{desc_id}"
            descriptions = [col['description'] for col in mapping]
            embeddings = model.encode(descriptions, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            INDICES[key] = (index, mapping)
            print(f"  Built index: {key}")

    llm_model = genai_client
    MODEL_NAME_API = MODEL_NAME[7:] if MODEL_NAME.startswith("models/") else MODEL_NAME

# FK cache
FK_CACHE = {}
from aap_sql.relations import format_fk_line, declared_relations, load_relation_graph, undeclared_relations

def get_foreign_keys(db_id):
    if db_id not in FK_CACHE:
        FK_CACHE[db_id] = declared_relations(db_id)
    return FK_CACHE[db_id]

# Augmented FK from offline mining
AUGMENTED_FK_GRAPH = None
AUGMENTED_FK_GRAPH_MTIME = None


def resolve_augmented_fk_path(value: str | None = None) -> Path:
    """Resolve an FK graph path relative to the AAP-SQL project root."""
    configured = value if value is not None else os.getenv("AAP_AUGMENTED_FK_PATH", "")
    path = (
        Path(configured)
        if configured
        else E4_GRAPH_PATH
    )
    return path if path.is_absolute() else BASE_DIR / path


AUGMENTED_FK_PATH = resolve_augmented_fk_path()

def load_augmented_fk_graph():
    global AUGMENTED_FK_GRAPH, AUGMENTED_FK_GRAPH_MTIME
    aug_path = AUGMENTED_FK_PATH
    current_mtime = aug_path.stat().st_mtime
    if AUGMENTED_FK_GRAPH is None or AUGMENTED_FK_GRAPH_MTIME != current_mtime:
        AUGMENTED_FK_GRAPH = load_relation_graph(aug_path)
        AUGMENTED_FK_GRAPH_MTIME = current_mtime
    return AUGMENTED_FK_GRAPH

def get_augmented_fks(db_id, columns):
    """Get implicit FKs from offline mining."""
    graph = load_augmented_fk_graph()
    # Format: [fk_table, fk_col, pk_table, pk_col, similarity_score]
    return undeclared_relations(graph, db_id, columns)

# ============ Core Functions ============
def precompute_retrieval(questions, config):
    from aap_sql.retrieval import retrieve_batch
    if config.embedding == "E0":
        return [[] for _ in questions], [{"mode": "none", "candidate_count": 0} for _ in questions]
    index, mapping = INDICES[f"{config.embedding}_{'basic' if config.description == 'D0' else 'csv'}"]
    return retrieve_batch(questions, config, model=EMBEDDING_MODELS[config.embedding],
                          index=index, mapping=mapping, reranker=CROSS_ENCODER,
                          normalize=faiss.normalize_L2)

def _process_literal_wrapper(args):
    db_id, question, col_mapping = args
    candidates = get_literal_candidates(db_id, question)
    return format_literal_hints(candidates, col_mapping)


def precompute_literals(questions, config: ExperimentConfig):
    """Pre-compute literal retrieval for all questions in parallel (Process-based)."""
    if config.literal != "L1":
        return [[] for _ in questions]

    print("  [INFO] Pre-computing exact quoted-value matches...")

    # Build flat mapping for schema-awareness
    # Try CSV mapping first (richer), fallback to basic
    flat_mapping = {} # {db_id: {'table.col': desc}}

    # Merge mappings: Basic first, then CSV overwrites (if available)
    source_mappings = [MAPPING_BASIC]
    if MAPPING_CSV:
        source_mappings.append(MAPPING_CSV)

    for mapping_list in source_mappings:
        for item in mapping_list:
            db = item['db_id']
            if db not in flat_mapping:
                flat_mapping[db] = {}
            # 'full_name' format is 'table.column'
            if 'full_name' in item:
                col_key = item['full_name'].lower().strip()
                flat_mapping[db][col_key] = item['description']
            elif 'original_column' in item: # Fallback just in case
                col_key = item['original_column'].lower().strip()
                flat_mapping[db][col_key] = item['description']

    args_list = []

    for q in questions:
        db_id = q['db_id']
        question = q['question']
        db_map = flat_mapping.get(db_id, {})
        args_list.append((db_id, question, db_map))

    with ThreadPoolExecutor(max_workers=PRECOMPUTE_WORKERS) as executor:
        literals_batch = list(
            tqdm(executor.map(_process_literal_wrapper, args_list), total=len(args_list), desc="Literals")
        )

    return literals_batch

def build_prompt(question, db_id, columns, config, evidence="", literals=None):
    from aap_sql.prompt import build_prompt as format_prompt
    fks = list(get_foreign_keys(db_id)) if config.fk != "FK0" else []
    if config.fk == "FK2":
        fks += get_augmented_fks(db_id, columns)
    probed = cached_probe_columns(db_id, columns) if config.probing == "P1" else []
    return format_prompt(question, columns, config, schema=load_schema(db_id),
                         foreign_keys=fks, probed_values=probed, evidence=evidence,
                         literals=literals)

from aap_sql.sql import clean_sql

def execute_sql(db_id, sql):
    """執行 SQL 並回傳 (ok, result_set)。
    使用 per-db lock 序列化同一資料庫的所有讀取操作，
    避免 SQLite WAL 以外模式下的「database is locked」導致
    future watchdog 超時，正確的 SQL 被誤記為 False。
    """
    conn = None
    db_lock = _get_db_lock(db_id)
    with db_lock:  # 同一 db 串行化，不同 db 並行
        try:
            db_path = get_db_path(db_id)
            # timeout=60：讓 SQLite 自行等待鎖，遠長於 watchdog VALIDATION_TIMEOUT_SEC 前段
            conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro",
                                   uri=True, timeout=60, check_same_thread=False)
            start_time = time.time()

            def _progress_handler():
                if time.time() - start_time > SQL_EXEC_TIMEOUT_SEC:
                    return 1
                return 0

            conn.set_progress_handler(_progress_handler, 10000)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            conn.close()
            return True, set(tuple(row) for row in rows) if rows else set()
        except Exception:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            return False, None

def config_metadata(config):
    return {**asdict(config), "model": MODEL_NAME.removeprefix("models/"),
            "temperature": TEMPERATURE, "max_output_tokens": MAX_TOKENS,
            "thinking_budget": 0 if MODEL_NAME.removeprefix("models/") == "gemini-2.5-flash-lite" else None,
            "provider": LLM_PROVIDER, "region": VERTEX_LOCATION, "sql_timeout_sec": SQL_EXEC_TIMEOUT_SEC}


def input_provenance(configs, questions):
    paths = {"mapping_basic": MAPPING_BASIC_PATH, "mapping_csv": MAPPING_CSV_PATH}
    if any(c.fk == "FK2" for c in configs):
        load_augmented_fk_graph()  # A missing graph fails before any generation call.
        paths["relations"] = AUGMENTED_FK_PATH
    for kind, path in (("E2", E2_MODEL_PATH), ("R1", R1_MODEL_PATH)):
        needed = any(c.embedding == kind if kind == "E2" else c.reranking == kind for c in configs)
        if needed:
            if not path.is_dir():
                raise FileNotFoundError(f"Required model directory missing: {path}")
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    paths[f"{kind}/{file.relative_to(path).as_posix()}"] = file
    hashes = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    hashes["questions"] = hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"implementation": "thesis-reconstruction-v2", "sha256": hashes, "question_count": len(questions)}


# ============ Experiment Runner ============
def run_experiment(config: ExperimentConfig, questions: list, provenance=None) -> dict:
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"  Embedding: {config.embedding}, Reranking: {config.reranking}")
    print(f"  Description: {config.description}, Evidence: {config.evidence}")
    print(f"  Probing: {config.probing}, Literal: {config.literal}")
    print(f"{'='*60}")

    correct_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_thought_tokens = 0
    token_lock = threading.Lock()
    all_predictions = []

    def process_single(args):
        nonlocal total_prompt_tokens, total_completion_tokens, total_thought_tokens
        idx, q, columns, literals, retrieval_info = args
        question = q.get("question", "")
        db_id = q.get("db_id", "")
        gold_sql = q.get("SQL", "")
        evidence = q.get("evidence", "") if config.evidence == "V1" else ""
        retrieval_mode = (retrieval_info or {}).get("mode", "unknown")
        retrieval_candidate_count = int((retrieval_info or {}).get("candidate_count", len(columns) if columns else 0))

        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                if idx % 50 == 0:
                    print(f"Processing {idx}...", end='\r')

                prompt = build_prompt(question, db_id, columns, config, evidence, literals)

                thinking_config = (genai_types.ThinkingConfig(thinking_budget=0)
                                   if MODEL_NAME_API == "gemini-2.5-flash-lite" else None)

                response = llm_model.models.generate_content(
                    model=MODEL_NAME_API,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_TOKENS,
                        thinking_config=thinking_config,
                    ),
                )
                response_text = response.text if hasattr(response, "text") else ""
                if config.prompt_style == "cot":
                    if "Final SQL:" not in response_text:
                        raise RuntimeError("Missing Final SQL marker")
                    response_text = response_text.rsplit("Final SQL:", 1)[1]
                predicted_sql = clean_sql(response_text)

                if not predicted_sql:
                    raise RuntimeError("Empty response from Gemini")

                prompt_tokens = 0
                completion_tokens = 0
                thought_tokens = 0

                if hasattr(response, 'usage_metadata'):
                    prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
                    thought_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0
                    with token_lock:
                        total_prompt_tokens += prompt_tokens
                        total_completion_tokens += completion_tokens
                        total_thought_tokens += thought_tokens

                return {
                    "idx": idx,
                    "db_id": db_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "evidence": evidence if config.evidence == "V1" else "",
                    "prompt": prompt,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "thought_tokens": thought_tokens,
                    "retrieval_mode": retrieval_mode,
                    "retrieval_candidate_count": retrieval_candidate_count,
                    "error": None,
                    "is_correct": False
                }
            except Exception as e:
                err_text = str(e).lower()
                retryable = (
                    "429" in err_text or
                    "resource_exhausted" in err_text or
                    "rate limit" in err_text or
                    "quota" in err_text or
                    "503" in err_text or
                    "unavailable" in err_text or
                    "deadline exceeded" in err_text or
                    "timed out" in err_text
                )
                if retryable and attempt < GEMINI_MAX_RETRIES - 1:
                    backoff_sec = min(60.0, (2 ** attempt) * 1.5 + random.random())
                    time.sleep(backoff_sec)
                    continue
                return {
                    "idx": idx,
                    "db_id": db_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "predicted_sql": "",
                    "evidence": evidence if config.evidence == "V1" else "",
                    "prompt": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "thought_tokens": 0,
                    "retrieval_mode": retrieval_mode,
                    "retrieval_candidate_count": retrieval_candidate_count,
                    "error": str(e),
                    "is_correct": False
                }
        return {
            "idx": idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "predicted_sql": "",
            "evidence": evidence if config.evidence == "V1" else "",
            "prompt": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thought_tokens": 0,
            "retrieval_mode": retrieval_mode,
            "retrieval_candidate_count": retrieval_candidate_count,
            "error": f"Max retries exceeded ({GEMINI_MAX_RETRIES})",
            "is_correct": False
        }

    def validate_single(item):
        idx = item.get("idx", -1)
        db_id = item.get("db_id", "")
        gold_sql = item.get("gold_sql", "")
        predicted_sql = item.get("predicted_sql", "")

        pred_ok, pred_res = execute_sql(db_id, predicted_sql)
        gold_ok, gold_res = execute_sql(db_id, gold_sql)
        is_correct = pred_ok and gold_ok and pred_res == gold_res
        return idx, is_correct, None

    # Pre-compute retrieval for ALL questions
    all_columns, retrieval_meta = precompute_retrieval(questions, config)

    # Pre-compute literals for ALL questions
    all_literals = precompute_literals(questions, config)

    # Bundle args with precomputed columns and literals
    args_list = [(i, q, all_columns[i], all_literals[i], retrieval_meta[i]) for i, q in enumerate(questions)]

    print(f"  [INFO] Phase 1: Generating SQL ({INFERENCE_WORKERS} threads)...")
    with ThreadPoolExecutor(max_workers=INFERENCE_WORKERS) as executor:
        future_to_idx = {executor.submit(process_single, args): args[0] for args in args_list}
        pbar = tqdm(total=len(questions), desc=config.name, unit="q")
        for future in as_completed(future_to_idx):
            try:
                result = future.result(timeout=90)
                all_predictions.append(result)
            except Exception as e:
                idx = future_to_idx[future]
                q = questions[idx]
                all_predictions.append({
                    "idx": idx,
                    "db_id": q.get("db_id", ""),
                    "question": q.get("question", ""),
                    "gold_sql": q.get("SQL", ""),
                    "predicted_sql": "",
                    "evidence": q.get("evidence", "") if config.evidence == "V1" else "",
                    "prompt": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "thought_tokens": 0,
                    "retrieval_mode": (retrieval_meta[idx] or {}).get("mode", "unknown"),
                    "retrieval_candidate_count": int((retrieval_meta[idx] or {}).get("candidate_count", 0)),
                    "error": str(e),
                    "is_correct": False
                })
            pbar.update(1)
            pbar.set_postfix({"generated": f"{pbar.n}/{len(questions)}"})
        pbar.close()

    generated_tokens = total_prompt_tokens + total_completion_tokens
    print(
        f"  [OK] Generated {len(all_predictions)} predictions | "
        f"Tokens: {generated_tokens:,} (thinking: {total_thought_tokens:,})"
    )

    print(f"  [INFO] Phase 2: Validating ({VALIDATION_WORKERS} threads, timeout={VALIDATION_TIMEOUT_SEC}s)...")
    idx_to_pos = {p.get("idx", -1): i for i, p in enumerate(all_predictions)}
    to_validate = [p for p in all_predictions if not p.get("error") and p.get("predicted_sql")]
    with ThreadPoolExecutor(max_workers=VALIDATION_WORKERS) as executor:
        future_to_idx = {executor.submit(validate_single, p): p.get("idx", -1) for p in to_validate}
        future_start = {future: time.time() for future in future_to_idx}
        pending = set(future_to_idx.keys())
        val_pbar = tqdm(total=len(to_validate), desc=f"{config.name}-Val", unit="q")
        validated = 0

        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

            for future in done:
                idx = future_to_idx[future]
                try:
                    result_idx, is_correct, validation_error = future.result()
                    pos = idx_to_pos.get(result_idx)
                    if pos is not None:
                        all_predictions[pos]["is_correct"] = is_correct
                        if validation_error:
                            all_predictions[pos]["error"] = validation_error
                    if is_correct:
                        correct_count += 1
                except Exception as e:
                    pos = idx_to_pos.get(idx)
                    if pos is not None:
                        all_predictions[pos]["is_correct"] = False
                        all_predictions[pos]["error"] = f"Validation Exception: {e}"
                validated += 1
                acc = correct_count / validated if validated else 0
                val_pbar.update(1)
                val_pbar.set_postfix({"正確": correct_count, "準確率": f"{acc:.1%}"})

            now = time.time()
            timed_out = [
                f for f in list(pending)
                if f.running() and now - future_start[f] > VALIDATION_TIMEOUT_SEC
            ]
            for future in timed_out:
                idx = future_to_idx[future]
                pos = idx_to_pos.get(idx)
                if pos is not None:
                    all_predictions[pos]["is_correct"] = False
                    all_predictions[pos]["error"] = f"Validation Timeout ({VALIDATION_TIMEOUT_SEC}s)"
                pending.remove(future)
                future.cancel()
                validated += 1
                acc = correct_count / validated if validated else 0
                val_pbar.update(1)
                val_pbar.set_postfix({"正確": correct_count, "準確率": f"{acc:.1%}"})

        val_pbar.close()
        executor.shutdown(wait=False, cancel_futures=True)

    # Sort predictions by idx
    finalize_start = time.time()
    print("  [INFO] Finalizing: sorting predictions...", flush=True)
    all_predictions.sort(key=lambda x: x.get("idx", -1))
    print(f"  [OK] Sorting done in {time.time() - finalize_start:.2f}s", flush=True)
    retrieval_mode_counts = Counter(p.get("retrieval_mode", "unknown") for p in all_predictions)

    # Save detailed predictions to JSON
    # Save detailed predictions to JSON
    log_dir = BASE_DIR / "outputs/research" / MODEL_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{MODEL_FILE_TAG}_{config.name}.json"
    json_write_start = time.time()
    print(f"  [INFO] Writing JSON: {log_file}", flush=True)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": config_metadata(config),
            "provenance": provenance or {},
            "summary": {
                "correct": correct_count,
                "total": len(questions),
                "accuracy": correct_count / len(questions) if questions else 0,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "thought_tokens": total_thought_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "retrieval_mode_counts": dict(retrieval_mode_counts)
            },
            "predictions": all_predictions
        }, f, indent=2, ensure_ascii=False)
    print(f"  [OK] JSON write done in {time.time() - json_write_start:.2f}s", flush=True)
    print(f"  [OK] Saved detailed log to: {log_file}")

    accuracy = correct_count / len(questions) if questions else 0

    return {
        "name": config.name,
        "embedding": config.embedding,
        "reranking": config.reranking,
        "description": config.description,
        "evidence": config.evidence,
        "probing": config.probing,
        "literal": config.literal,
        "fk": config.fk,
        "accuracy": f"{accuracy*100:.2f}%",
        "correct": correct_count,
        "total": len(questions),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "thought_tokens": total_thought_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
    }

# ============ Main ============
def main():
    global genai_client, genai_types
    parser = argparse.ArgumentParser(description="Final AAP-SQL Gemini 2.5 runner")
    parser.add_argument("--limit", type=int, help="Positive question limit for a smoke run")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--all-configs", action="store_true", help="Run every thesis ablation (billable)")
    parser.add_argument(
        "--config",
        type=str,
        choices=[c.name for c in EXPERIMENTS],
        default=final_configurations()[0].name,
        help="Thesis configuration; defaults to the full method",
    )
    parser.add_argument("--force-all", action="store_true", help="Ignore local output and rerun the final configuration")
    args = parser.parse_args()
    if args.list_configs:
        print("\n".join(c.name for c in EXPERIMENTS))
        return
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    print(f"Provider: {LLM_PROVIDER}")
    print(f"Using Gemini model: {MODEL_NAME} ({MODEL_DISPLAY_NAME})")
    print(f"Log tag/dir: {MODEL_FILE_TAG} / {MODEL_LOG_DIR}")
    print("Decoding: temperature=0, max_output_tokens=2048; Gemini 2.5 Flash-Lite thinking disabled")
    print("FK style: unified score (explicit / implicit)")
    print(f"Augmented FK graph: {AUGMENTED_FK_PATH}")
    if LLM_PROVIDER == "vertex":
        print(f"Vertex project: {'configured' if VERTEX_PROJECT else '(missing)'}")
        print(f"Vertex location: {VERTEX_LOCATION}")

    if LLM_PROVIDER != "vertex":
        parser.error("Only AAP_LLM_PROVIDER=vertex is supported")
    if not VERTEX_PROJECT:
        parser.error("Set GOOGLE_VERTEX_PROJECT and configure Vertex ADC")
    from google import genai
    from google.genai import types as genai_types
    genai_client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    from aap_sql.config import set_config
    from scripts.research.config import bird_config
    set_config(bird_config())

    # Load Models (Only in Main Process)
    # Load questions (Use config for flexibility)
    config_obj = get_config()
    dev_path = Path(config_obj.database.dev_json_path)
    if not dev_path.exists():
        script_base = Path(__file__).resolve().parent.parent.parent
        dev_path = script_base / dev_path
    print(f"Loading questions from: {dev_path}")

    with open(dev_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)


    if len(questions) != 1534 or len({q["db_id"] for q in questions}) != 11:
        parser.error("Expected BIRD dev_20240627: 1,534 questions in 11 databases")
    if args.limit:
        questions = questions[:args.limit]

    results = []
    csv_path = Path(os.getenv("AAP_REPRODUCTION_RESULTS_CSV", f"outputs/research/{MODEL_LOG_DIR}/results.csv"))
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    log_dir = BASE_DIR / "outputs/research" / MODEL_LOG_DIR
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    selected = EXPERIMENTS if args.all_configs else [c for c in EXPERIMENTS if c.name == args.config]
    provenance = input_provenance(selected, questions)
    remaining = []
    for config in selected:
        log_file = log_dir / f"{MODEL_FILE_TAG}_{config.name}.json"
        if log_file.exists():
            saved = json.loads(log_file.read_text(encoding="utf-8"))
            matches = (saved.get("provenance") == provenance and saved.get("config") == config_metadata(config)
                       and len(saved.get("predictions", [])) == len(questions))
            if not args.force_all:
                if not matches:
                    parser.error(f"Existing log has different inputs/settings: {log_file}. Select a new output directory or --force-all.")
                summary = saved["summary"]
                results.append({**{k: v for k, v in asdict(config).items() if k != "prompt_style"},
                                **{k: summary[k] for k in ("correct", "total", "prompt_tokens", "completion_tokens", "thought_tokens", "total_tokens")},
                                "accuracy": f"{summary['accuracy'] * 100:.2f}%"})
                continue
            from shutil import copy2
            copy2(log_file, log_file.with_suffix(f".{datetime.now():%Y%m%dT%H%M%S%f}.bak"))
        remaining.append(config)
    if csv_path.exists():
        from shutil import copy2
        copy2(csv_path, csv_path.with_suffix(f".{datetime.now():%Y%m%dT%H%M%S%f}.bak"))
    completed_names = {r["name"] for r in results}

    print(f"\nTotal questions: {len(questions)}")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Already completed: {len(completed_names)}")
    print(f"Remaining to run: {len(remaining)}")
    print(f"Start: {datetime.now().isoformat()}")

    if not remaining:
        print("\nAll experiments already completed!")
        return

    load_models(remaining)

    grand_total_tokens = sum(int(r.get('total_tokens', 0)) for r in results)

    for config in remaining:
        result = run_experiment(config, questions, provenance)
        results.append(result)
        grand_total_tokens += result['total_tokens']

        # Save intermediate results (append mode - rewrite all)
        csv_write_start = time.time()
        print(f"  [INFO] Writing CSV: {csv_path}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["name", "embedding", "reranking", "description", "evidence",
                         "probing", "literal", "fk", "accuracy", "correct", "total",
                         "prompt_tokens", "completion_tokens", "thought_tokens", "total_tokens"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  [OK] CSV write done in {time.time() - csv_write_start:.2f}s")

        print(f"\n[OK] {config.name}: {result['accuracy']} ({result['correct']}/{result['total']}) | Tokens: {result['total_tokens']:,}")

    # Save token summary
    token_summary = {
        "model": MODEL_NAME,
        "total_experiments": len(results),
        "total_questions": len(questions),
        "grand_total_tokens": grand_total_tokens,
        "experiments": [
            {
                "name": r["name"],
                "prompt_tokens": r.get("prompt_tokens", 0),
                "completion_tokens": r.get("completion_tokens", 0),
                "thought_tokens": r.get("thought_tokens", 0),
                "total_tokens": r.get("total_tokens", 0),
            }
            for r in results
        ]
    }
    with open(BASE_DIR / "outputs/research/token_summary.json", 'w', encoding='utf-8') as f:
        json.dump(token_summary, f, indent=2)


    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("Results saved to: outputs/research/results.csv")
    print("Token summary: outputs/research/token_summary.json")
    print(f"Grand total tokens: {grand_total_tokens:,}")
    print(f"End: {datetime.now().isoformat()}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
