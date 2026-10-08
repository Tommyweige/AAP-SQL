# Reproducing the research

Run the stages from the repository root. The workflow is **prepare assets → train or load models → mine relations → generate SQL → evaluate**. See the [research overview](research.md) for method and experiment tables, and [implementation notes](thesis-alignment.md) for the choices used by this version.

## Install

Python 3.12+:

```bash
python -m pip install -e '.[research,training,test]'
python -m aap_sql.smoke
python -m pytest -q
```

The research extra installs generation/retrieval tools; the training extra installs SQL parsing and model-training tools. The offline example uses temporary SQLite data and synthetic adapters. Training-loop tests initialize tiny local Transformers.

## Prepare data and models

Download data from [BIRD](https://bird-bench.github.io/) and select **Dev `dev_20240627`**, 1,534 questions in 11 databases. Use the original 9,428-question Train release for training. Place files as follows:

```text
BIRD/
  train/
    train.json
    train_column_descriptions.json
  dev/dev_20240627/
    dev.json
    dev_tables.json
    dev_databases/dev_databases/<db_id>/<db_id>.sqlite
models/
  column_retriever_v3_paper_earlystop/
  cross_encoder_schema_paper_repro/
```

`train_column_descriptions.json` is an input you prepare: a list of full natural-language column descriptions in this format:

```json
[
  {"db_id": "demo", "full_name": "people.name", "description": "The name of the person."}
]
```

Include the columns of the training databases. The two bundled 798-column mappings cover **Dev** and are used by generation. [Asset descriptions and provenance](../resources/bird_dev_20240627/README.md).

To use the published model weights instead of training:

```bash
python -m pip install 'huggingface-hub>=0.35'
hf download TommyPanLab/AAP-SQL-Column-Retriever --local-dir models/column_retriever_v3_paper_earlystop
hf download TommyPanLab/AAP-SQL-Candidate-Reranker --local-dir models/cross_encoder_schema_paper_repro
```

Model cards: [retriever](https://huggingface.co/TommyPanLab/AAP-SQL-Column-Retriever), [reranker](https://huggingface.co/TommyPanLab/AAP-SQL-Candidate-Reranker). The runner records the local weight hashes for each generation run.

## Train the column models

Prepare the question splits and pair manifest first:

```bash
python scripts/research/train_models.py --questions BIRD/train/train.json --descriptions BIRD/train/train_column_descriptions.json --prepare-only
```

Train both models:

```bash
python scripts/research/train_models.py --questions BIRD/train/train.json --descriptions BIRD/train/train_column_descriptions.json
```

Use `--mode retriever` or `--mode reranker` for one model, `--seed` for another split seed, and `--output-root` for a fresh output directory. The manifest saves question IDs, pair counts, seed, and input hashes. The default split is 80/10/10 by question with seed 42; pair expansion happens after splitting.

| Model | Backbone | Batch | Warmup | Loss |
|---|---|---:|---:|---|
| Retriever | all-mpnet-base-v2 | 64 | 200 | MultipleNegativesRankingLoss |
| Reranker | ms-marco-MiniLM-L-6-v2 | 32 | 100 | BCEWithLogitsLoss |

Both use learning rate 2e-5, at most 30 epochs, patience 4, and validation-loss checkpoint selection. Test evaluation loads the selected checkpoint. Existing weight directories are protected; choose a fresh output root for a new experiment. The thesis's reranker data contain 38,030 positives and 37,821 negatives; compare these with your manifest's observed counts.

## Mine the relation graph

```bash
python scripts/research/build_relation_graph.py --database-root BIRD/dev/dev_20240627/dev_databases/dev_databases --gold-sql BIRD/dev/dev_20240627/dev.json
```

The miner reads SQLite databases in read-only mode. `--gold-sql` adds a post-hoc query-usage audit. Outputs under `outputs/research/thesis/`:

| File | Contents |
|---|---|
| `undeclared_relations.json` | Online relation cues |
| `undeclared_relations.audit.json` | Candidate metrics and classification |
| `undeclared_relations.summary.json` | Recovery / new / reverse counts |
| `undeclared_relations.omissions.json` | Reasons for omitted declared numeric FKs |
| `undeclared_relations.gold-usage.json` | Post-hoc Gold SQL join usage |

Numeric mining uses bounded samples, then fully verifies new candidates. See [the numeric relation rules](relation-reconstruction.md) for thresholds, direction selection, and the 42 + 3 result. Online injection requires both endpoint tables in the core-column selection.

## Generate and evaluate SQL

Authenticate Vertex AI locally with Application Default Credentials:

```bash
gcloud auth application-default login
```

Set the project and model using your shell, or copy `.env.example` to `.env`. PowerShell:

```powershell
$env:GOOGLE_VERTEX_PROJECT = "<your-gcp-project-id>"
$env:GOOGLE_VERTEX_LOCATION = "global"
$env:AAP_GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite"
$env:AAP_AUGMENTED_FK_PATH = "outputs/research/thesis/undeclared_relations.json"
```

A small run exercises the data/model/provider path; a full run evaluates the 1,534 questions:

```bash
python scripts/research/reproduce_bird_dev.py --config E2-R1-V1-P1-L1-FK2 --limit 3
python scripts/research/reproduce_bird_dev.py --config E2-R1-V1-P1-L1-FK2
```

Vertex AI bills generation usage. The default uses temperature 0, maximum output 2,048, and Gemini 2.5 Flash-Lite thinking disabled. The runner records operational settings, model/input hashes, question count, generated SQL, execution outcome, and token usage.

## Ablations

```bash
python scripts/research/reproduce_bird_dev.py --list-configs
python scripts/research/reproduce_bird_dev.py --config no-probing
python scripts/research/reproduce_bird_dev.py --config ddl-fk-only-no-evidence
python scripts/research/reproduce_bird_dev.py --all-configs
```

Six full/ablation configurations each run with and without Evidence, plus two baselines, task instructions, and explicit CoT: 16 configurations in total. The default runs one configuration; `--all-configs` runs all 16. Explicit-CoT evaluation extracts the query after `Final SQL:`. DIN-SQL/CHESS figures in the [research tables](research.md) come from the thesis's comparison experiments; this runner covers the AAP configurations.

## Outputs and resume

Generation logs and `results.csv` live under `outputs/research/thesis-aligned-gemini-2.5/`. `AAP_REPRODUCTION_RESULTS_CSV` overrides the CSV destination. Each configuration log contains summary counts, per-question predictions, configuration, and provenance.

EX is correct execution-result sets divided by the loaded question count, including failures. Total Tokens are prompt + candidate output; thought usage is recorded separately. Probe/literal results are queried during the run and cached in memory.

Resume checks input hashes, model files, configuration, and question count. Select a fresh output directory for changed conditions, or use `--force-all` to rerun with backups of prior outputs. Additional local settings are in `.env.example` and the runner's constants.
