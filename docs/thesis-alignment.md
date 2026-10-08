# Implementation notes and validation

This file maps the [thesis](https://hdl.handle.net/11296/rjag8t) method to the reusable implementation. Page references use printed thesis pages. The final source PDF checksum is `fc3d8fc50532d5eb5a434f2786bbb477a7861a763393305e503665b7c9ecf782`.

## Method-to-code map

| Thesis | Implementation | Check |
|---|---|---|
| §3.3.2, pp.51–54: SQL-positive columns, question split, same-DB negatives | [training_data.py](../scripts/research/training_data.py) | Aliases, CTEs, ambiguous columns, disjoint question IDs |
| §3.3.3, pp.55–57: retriever / reranker training | [train_models.py](../scripts/research/train_models.py) | Native Torch/ST gradients, early stopping, checkpoint save/reload |
| §3.3.4, p.57: global 50 → DB filter → reranked 10 | [retrieval.py](../src/aap_sql/retrieval.py) | Single recall pass, DB filtering, ranking and limits |
| §3.3.5, p.58: complete schema, selective augmentation | [pipeline.py](../src/aap_sql/pipeline.py) | Schema remains available with zero retrieved columns |
| §3.4.1, p.59: up to three core-column values | [value_prober.py](../src/aap_sql/components/value_prober.py) | Shared backend formatting and fresh query cache |
| §3.4.2, pp.60–63: exact quoted literals, five candidates per value | [literal_retrieval.py](../src/aap_sql/components/literal_retrieval.py) | TEXT affinity, apostrophes, deduplication, stored case |
| §3.5.2–3, pp.64–70: numeric/text relation discovery | [build_relation_graph.py](../scripts/research/build_relation_graph.py) | Direction, uniqueness, coverage, full-data verification, attribute blacklist |
| §3.5.4, p.71: question-relevant explicit/implicit cues | [relations.py](../src/aap_sql/relations.py) | Both-table relevance gate and relation labels |
| §3.6, pp.71–73: prompt and decoding | [prompt.py](../src/aap_sql/prompt.py) | Block order and module toggles; temperature 0 / output 2,048 in runner |
| §3.7 / Appendix A: ablations and supplementary prompts | [settings.py](../src/aap_sql/settings.py) | 16 configurations; DDL-only FK and Final SQL extraction |
| §4.1, pp.79–80: execution accuracy and token accounting | [reproduce_bird_dev.py](../scripts/research/reproduce_bird_dev.py) | Failed questions in denominator; input/model hashes and resume checks |

## Explicit implementation choices

- Question split: 80/10/10, seed 42. The thesis specifies question-level separation; this implementation records these proportions and seed in the manifest.
- Relation relevance: both endpoint tables must occur among the selected Top-10 columns. Declared FKs remain available for the whole database.
- Numeric names/direction: normalized source entity, supported key suffixes, compatible target entity, formal-key priority, and abstention on ties. These reconstruction rules are detailed in [the relation audit](relation-reconstruction.md).
- Generation service: Vertex AI, region `global`, 24 inference workers / four validation workers, and SQL timeout 60 seconds. These are runner defaults.
- Application integration: SQLite is the research engine; PostgreSQL, external literal/evidence hooks, and structured context traces support reuse in other applications.

## Prompt presentation

The block order and ablation contents follow Appendix A/B. The current strings use Markdown headings and list markers:

| Appendix presentation | Current implementation |
|---|---|
| `Schema:` | `### Schema:` |
| `disp.type: ...` | `- disp.type: ...` in Columns |
| `disp.type: OWNER, DISPONENT` | `disp.type examples: OWNER, DISPONENT` in Probed Values |
| `"OWNER" is in column ...` | A list item in Found Values |

Descriptions preserve the supplied mapping text, including backticks and embedded newlines. PDF line wrapping and Markdown layout therefore differ. A component-level format check used Appendix B's eight fixed columns and the local financial database. The eight probe-value lists and eight declared FK cues matched the appendix.

## Observed engineering validation

| Area | Evidence |
|---|---|
| Core and research utilities | 46 local tests passed; PostgreSQL integration needs its test URL |
| PostgreSQL 17 | CI runs the equivalent SQLite/PostgreSQL pipeline and identifier/key tests |
| Models and training loops | Tiny randomly initialized local Transformers exercise native SDKs, gradients and checkpoint selection |
| Relation mining | All 11 BIRD Dev databases: 42 recovered numeric FKs, three new, zero reverse; omissions 38/5/1 |
| Packaging | Wheel installation and offline smoke; packaged `py.typed` |
| Code checks | Core format/lint, strict typing, compileall and whitespace checks |

The numeric count/omission observations are saved in [thesis_relation_validation.json](../resources/bird_dev_20240627/thesis_relation_validation.json). They agree with Tables 4-7/4-8 under the documented reconstruction rules. The reconstruction uses the thesis's aggregate statistics as constraints; validation compares those counts and the three documented new edges.

The [research overview](research.md) and [thesis_results.json](../resources/bird_dev_20240627/thesis_results.json) contain the thesis experiments. Full retraining, new 1,534-question LLM generation, and new DIN-SQL/CHESS runs are separate measurements from the engineering checks above.
