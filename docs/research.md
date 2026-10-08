# Research overview

AAP-SQL prepares database context for Text-to-SQL generation through **complete schema structure and selective column augmentation**. The research asks how to provide useful schema semantics, stored values, and join cues while keeping the generation prompt compact.

Source: Tung-Wei Pan, *An Efficient and Low-Cost Adaptive Augmented Prompting Framework for Text-to-SQL Generation*, National Dong Hwa University, June 2026. [Thesis record](https://hdl.handle.net/11296/rjag8t). All experiment tables below are transcribed from the final thesis; the structured source is [thesis_results.json](../resources/bird_dev_20240627/thesis_results.json).

## Method

The offline stage trains column models and mines candidate relationships. The online stage selects useful context for each question:

1. Encode the question with the fine-tuned all-mpnet-base-v2 retriever. Recall 50 candidates from the global column index, then retain candidates from the question's database.
2. Score question/column-description pairs with the MiniLM cross-encoder and retain at most 10 core columns.
3. Read up to three valid, distinct values for each core column. Match unique quoted strings from the question against text columns using case-insensitive exact lookup, with at most five candidate columns per value.
4. Supply all declared foreign keys and the mined relations whose two endpoint tables occur in the selected columns.
5. Assemble the complete schema, foreign keys, column descriptions, probed values, aligned literals, evidence, and question; generate SQL and compare execution results.

The complete table/column structure remains available when a column misses Top-10. Retrieval controls extra descriptions and values; the schema includes all tables and columns. The reusable implementation is in [src/aap_sql](../src/aap_sql/); operational details and method choices are in [implementation notes](thesis-alignment.md).

### Training

Gold SQL is resolved into positive column pairs. Questions are split before pair expansion; reranker negatives are unused columns from the same database, capped at the number of positive columns. Retriever batches keep one pair per question to avoid treating another positive from that question as a negative.

| Model | Backbone | Loss | Batch | Warmup |
|---|---|---|---:|---:|
| Retriever | all-mpnet-base-v2 | Multiple Negatives Ranking Loss | 64 | 200 |
| Reranker | ms-marco-MiniLM-L-6-v2 | Binary cross entropy | 32 | 100 |

Both use learning rate 2e-5, a maximum of 30 epochs, validation-loss checkpoint selection, and early stopping patience 4. The implementation uses question splits 80/10/10 and seed 42. [Training commands and input format](research-reproduction.md#train-the-column-models).

### Relationship discovery

Numeric candidates use source-to-target distinct-value coverage, source-row coverage, target uniqueness, and name evidence. Formal single-column PK/UNIQUE targets use coverage thresholds of 0.90; near-unique targets use 0.98 coverage and uniqueness thresholds. Schema/name evidence selects the direction; equal evidence leads to abstention. Text candidates use forward coverage of at least 0.90 and backward overlap of at least 0.10, with an attribute blacklist.

The miner samples at most 200,000 rows per table and retains at most 50,000 distinct values per column. Newly discovered numeric candidates receive a full-data verification. Gold SQL is used afterward to count query usage. [Numeric rules and validation](relation-reconstruction.md).

## Experimental setting

| Setting | Value |
|---|---|
| Dataset | BIRD Dev `dev_20240627` |
| Questions / databases | 1,534 / 11 |
| Database engine | SQLite |
| Primary generator | Gemini 2.5 Flash-Lite |
| Temperature / maximum output | 0 / 2,048 tokens |
| Primary evidence condition | BIRD-provided Evidence |
| EX | Equality of generated-SQL and Gold-SQL execution result sets |
| Total Token accounting | Prompt + completion |

The denominator includes generation and execution failures. The thesis compares AAP-SQL with DIN-SQL and CHESS reproduced using the same generator and evaluation setting.

## Main comparison — Table 4-3

| Method | EX | Prompt tokens | Completion tokens | Total tokens |
| --- | --- | --- | --- | --- |
| AAP-SQL | 62.58% | 3,204,990 | 115,045 | 3,320,035 |
| DIN-SQL (thesis reproduction) | 61.08% | 45,849,661 | 1,471,252 | 47,320,913 |
| CHESS (thesis reproduction) | 61.21% | 38,053,070 | 4,375,655 | 42,428,725 |
| Baseline + Evidence | 57.76% | 2,077,284 | 111,024 | 2,188,308 |

AAP-SQL uses 92.98% fewer tokens than the DIN-SQL reproduction and 92.18% fewer than the CHESS reproduction. Relative to Baseline + Evidence, it adds 74 correct answers, improving EX by 4.82 percentage points.

## Training and retrieval — Tables 4-1 and 4-2

| Module | Best validation loss | Held-out metric | Best / completed epochs |
|---|---:|---|---|
| Retriever | 0.460746 | Test loss 0.486277 | 21 / 25 |
| Reranker | 0.241477 | Test accuracy 0.90501 | 4 / 8 |

| Stage | Gold column recall | Complete coverage | Average missing columns |
| --- | --- | --- | --- |
| fine-tuned-top50 | 86.01% | 56.26% | 0.752 |
| fine-tuned-top10-no-reranking | 65.86% | 28.94% | 1.774 |
| fine-tuned-reranked-top10 | 75.47% | 38.20% | 1.285 |
| generic-reranked-top10 | 68.70% | 31.49% | 1.604 |

At the same Top-10 budget, reranking improves Gold column recall from 65.86% to 75.47%. Fine-tuning also improves the reranked Top-10 result relative to the generic retriever.

## Generator comparison — Table 4-4

| Generator | DIN-SQL EX | CHESS EX | AAP-SQL EX | Baseline + Evidence EX |
| --- | --- | --- | --- | --- |
| Gemini 2.5 Flash-Lite | 61.08% | 61.21% | 62.58% | 57.76% |
| Gemini 3.1 Flash-Lite | 66.56% | 66.20% | 68.97% | 65.84% |
| Mistralai_devstral_2_123b | — | — | 58.38% | 54.49% |

## Ablations — Tables 4-5 and 4-6

The CLI names below map directly to [the 16 configurations](../src/aap_sql/settings.py). Each Evidence condition has its own full-method reference. Without-Evidence CLI names append `-no-evidence`, except the baseline uses `baseline`.

| Configuration | With Evidence EX | Tokens | Without Evidence EX | Tokens |
| --- | --- | --- | --- | --- |
| E2-R1-V1-P1-L1-FK2 | 62.58% | 3,320,035 | 52.10% | 3,182,639 |
| no-reranking | 61.34% | 3,166,373 | 51.04% | 3,125,062 |
| no-probing | 60.63% | 2,848,901 | 51.83% | 2,808,024 |
| no-literals | 60.89% | 3,214,700 | 50.52% | 3,175,919 |
| generic-retriever | 60.95% | 3,217,487 | 50.72% | 3,171,008 |
| ddl-fk-only | 61.93% | 3,218,735 | 51.24% | 3,085,902 |
| baseline-with-evidence / baseline | 57.76% | 2,188,308 | 39.05% | 2,067,645 |

`ddl-fk-only` removes the extra Foreign Keys block and retains FK declarations inside the schema DDL. With Evidence, probing has the largest single-module EX difference (1.95 points); without Evidence, literal alignment has a larger difference (1.58 points) than probing (0.27 points).

### Supplemental prompts — Section 4.3.4

| Configuration | EX | Total tokens |
| --- | --- | --- |
| task-instructions | 61.41% | 3,893,904 |
| explicit-cot | 58.87% | 4,582,593 |

The task and explicit-CoT variants add role/output instructions and intermediate reasoning requirements, respectively. Their prompt layouts appear in thesis Appendix A-8/A-9.

## Numeric relation validation — Tables 4-7 and 4-8

The implemented miner reproduces 42 recovered directions out of 86 declared numeric foreign keys (48.84%), three new candidates, and zero reversed predictions. The 44 omissions split into 38 name-evidence, five scope, and one direction-ambiguity cases. The three new numeric edges are `transactions_1k.CustomerID → customers.CustomerID`, `GasStationID → gasstations.GasStationID`, and `ProductID → products.ProductID`; full-data coverage and target uniqueness are 100%, and Gold SQL usage counts are 9, 18, and 4.

Rules were reconstructed using the thesis text and these published constraints. The [relation audit](relation-reconstruction.md) records the assumptions and [machine-readable observed evidence](../resources/bird_dev_20240627/thesis_relation_validation.json) records the counts and omissions.

## Error analysis — Tables 4-16 to 4-20

| Error category | Baseline + Evidence | AAP-SQL |
| --- | --- | --- |
| schema-only | 23 | 12 |
| literal-only | 91 | 72 |
| join-only | 106 | 94 |
| compound | 242 | 247 |
| filter-logic | 102 | 81 |
| others | 84 | 68 |

Total errors decrease from 648 to 574. Compound errors remain the largest category; schema-plus-join cases account for 146 of the 247 compound errors. Their overlap shows why complete schema and a good Top-10 list still need correct join, filter, and aggregation reasoning.

| Core-column coverage / outcome | Questions | Incorrect SQL |
| --- | --- | --- |
| initial-recall-incomplete | 671 | 275 |
| recall-complete-top10-incomplete | 277 | 116 |
| top10-complete-sql-incorrect | 183 | 183 |
| top10-complete-sql-correct | 403 | 0 |

Among the 574 failures, 391 have incomplete core-column coverage and 183 have complete Top-10 coverage. This is an association between coverage and execution outcome; the remaining schema is available to the generator in both cases.

| Baseline + Evidence | AAP-SQL | Questions |
| --- | --- | --- |
| Correct | Correct | 817 |
| Correct | Incorrect | 69 |
| Incorrect | Correct | 143 |
| Incorrect | Incorrect | 505 |

The per-question comparison contains 143 improvements and 69 regressions, producing the net gain of 74 correct answers. The thesis's qualitative examples (Tables 4-10 to 4-15 and 4-21 to 4-24) illustrate schema, value, join, compound, filter, and other errors.

## Reproduce and extend

Start with [the offline example](../README.md#quick-start), follow [the reproduction guide](research-reproduction.md) for training and generation, or integrate the [database backends](database-backends.md) into an application.

Current engineering validation covers core and backend behavior, training-data construction, tiny local Transformer training/checkpoint cycles, and numeric relation mining. Full retraining and generation results can be recorded alongside the thesis tables using the runner's input hashes and configuration metadata.

## Citation

GitHub citation metadata is in [CITATION.cff](../CITATION.cff). For the thesis:

```bibtex
@mastersthesis{pan2026aapsql,
  author = {Pan, Tung-Wei},
  title = {An Efficient and Low-Cost Adaptive Augmented Prompting Framework for Text-to-SQL Generation},
  school = {National Dong Hwa University},
  year = {2026},
  month = {June},
  url = {https://hdl.handle.net/11296/rjag8t}
}
```
