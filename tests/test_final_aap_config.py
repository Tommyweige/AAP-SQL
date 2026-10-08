import json

from scripts.research import reproduce_bird_dev as reproduction
from scripts.research.config import bird_config


FINAL_CONFIG_NAME = "E2-R1-V1-P1-L1-FK2"
E4_RELATIVE_PATH = "outputs/research/thesis/undeclared_relations.json"


def test_full_aap_config_is_available():
    assert len(reproduction.EXPERIMENTS) == 16
    config = next(
        item for item in reproduction.EXPERIMENTS if item.name == FINAL_CONFIG_NAME
    )

    assert (
        config.embedding,
        config.reranking,
        config.evidence,
        config.probing,
        config.literal,
        config.fk,
    ) == ("E2", "R1", "V1", "P1", "L1", "FK2")


def test_final_vertex_defaults_and_project_rooted_inputs():
    assert reproduction.LLM_PROVIDER == "vertex"
    assert reproduction.DEFAULT_VERTEX_LOCATION == "global"
    assert reproduction.MODEL_NAME == "models/gemini-2.5-flash-lite"
    assert reproduction.TEMPERATURE == 0.0
    assert reproduction.MAX_TOKENS == 2048
    assert reproduction.DEFAULT_INFERENCE_WORKERS == 24
    assert reproduction.DEFAULT_VALIDATION_WORKERS == 4
    assert reproduction.DEFAULT_SQL_EXEC_TIMEOUT_SEC == 60

    assert reproduction.E2_MODEL_PATH == (
        reproduction.BASE_DIR / "models/column_retriever_v3_paper_earlystop"
    )
    assert reproduction.R1_MODEL_PATH == (
        reproduction.BASE_DIR / "models/cross_encoder_schema_paper_repro"
    )
    assert reproduction.MAPPING_BASIC_PATH == (
        reproduction.BASE_DIR / "resources/bird_dev_20240627/column_mapping.json"
    )
    assert reproduction.MAPPING_CSV_PATH == (
        reproduction.BASE_DIR
        / "resources/bird_dev_20240627/column_mapping_csv_enriched.json"
    )

    database = bird_config().database
    assert database.dev_json_path == (
        reproduction.BASE_DIR / "BIRD/dev/dev_20240627/dev.json"
    )
    assert database.dev_db_path == (
        reproduction.BASE_DIR
        / "BIRD/dev/dev_20240627/dev_databases/dev_databases"
    )


def test_e4_graph_loads_and_relative_path_is_project_rooted(monkeypatch, tmp_path):
    expected_path = reproduction.BASE_DIR / E4_RELATIVE_PATH

    assert reproduction.resolve_augmented_fk_path("") == reproduction.BASE_DIR / "outputs/research/thesis/undeclared_relations.json"
    assert reproduction.resolve_augmented_fk_path(E4_RELATIVE_PATH) == expected_path
    graph_path = tmp_path / "relations.json"
    expected_graph = {"demo": [["people", "team_id", "teams", "id", 1.]]}
    graph_path.write_text(json.dumps(expected_graph), encoding="utf-8")

    monkeypatch.setattr(reproduction, "AUGMENTED_FK_PATH", graph_path)
    monkeypatch.setattr(reproduction, "AUGMENTED_FK_GRAPH", None)
    monkeypatch.setattr(reproduction, "AUGMENTED_FK_GRAPH_MTIME", None)
    loaded_graph = reproduction.load_augmented_fk_graph()

    assert loaded_graph == expected_graph
    assert sum(len(edges) for edges in loaded_graph.values()) == 1
