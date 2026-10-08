# Numeric relation rules and audit

The numeric miner reconstructs the thesis's directed-relation method using its text, Table 4-7/4-8 results, and omission categories. On the 11 BIRD Dev databases, it produces **45 directed candidates: 42 recovered declared FKs + three new candidates**, zero reversed predictions, and omissions of **38 name-evidence / five scope / one direction-ambiguity** cases.

## Direction and coverage rules

1. Normalize CamelCase/separators, supported key suffixes, and common plural endings. After normalization, generic attributes such as `year` and `status` provide insufficient entity evidence.
2. Require compatible source and target entities. The target-column entity must match the source entity; `customer_id → customers.vendor_id` fails this rule.
3. Numeric source names use one of `id`, `code`, `key`, `ref`, `number`, `num`, `no`, or `nr`. Bare numeric names are outside this reconstructed candidate scope.
4. Select a direction by formal single-column PK/UNIQUE priority and name evidence before data verification. Equal evidence leads to abstention.
5. Apply the thesis coverage/uniqueness thresholds to the selected direction; fully verify new numeric candidates using the complete database.

The name normalization, explicit source-suffix scope and stage ordering are implementation choices inferred from the published constraints. The reconstruction evidence consists of the method text, published statistics, and candidate-level observations recorded here.

## Reconstruction sequence

Thresholds remain 0.90 / 0.98; sample limits remain 200,000 rows and 50,000 distinct values. The sequence below diagnoses the initial 51 candidates, followed by a fresh full-database mining run for the final row.

| Cumulative rule | Recovered | New |
|---|---:|---:|
| Literal implementation of the method text | 46 | 5 |
| Generic attributes excluded as entities | 44 | 5 |
| Nonempty target entity must be compatible | 44 | 4 |
| Numeric source requires a supported key suffix | 43 | 4 |
| Direction chosen before verification; ties abstain | 42 | 3 |

A historical `_NON_FK_GENERIC` blacklist supplied additional evidence for excluding generic attributes. This implementation applies the blacklist to normalized entities.

## Six changed candidates

| Initial candidate | Final treatment | General rule |
|---|---|---|
| `races.year → seasons.year` | Name evidence insufficient | Generic attribute entity |
| `results.statusId → status.statusId` | Name evidence insufficient | Generic attribute entity |
| `member.zip → zip_code.zip_code` | Outside scope | Bare numeric source name |
| `Team_Attributes.team_fifa_api_id → Team.team_fifa_api_id` | Ambiguous direction | No formal single-column key; equal names |
| `Match.country_id → League.country_id` | Omitted | Equal schema/name evidence |
| `Match.league_id → League.country_id` | Omitted | Target entity conflicts with source |

Gold SQL usage is computed after mining.

## Observed results

| Metric | Thesis | Implemented miner |
|---|---:|---:|
| Declared FKs | 104 | 104 |
| Declared numeric FKs | 86 | 86 |
| Recovered numeric directions | 42 | 42 |
| Directed numeric candidates | 45 | 45 |
| Reversed predictions | 0 | 0 |
| New numeric candidates | 3 | 3 |

All three new numeric edges occur in `debit_card_specializing`:

| Source | Target | Distinct / row coverage | Target uniqueness | Gold usage |
|---|---|---|---|---:|
| `transactions_1k.CustomerID` | `customers.CustomerID` | 100% / 100% | 100% | 9 |
| `transactions_1k.GasStationID` | `gasstations.GasStationID` | 100% / 100% | 100% | 18 |
| `transactions_1k.ProductID` | `products.ProductID` | 100% / 100% | 100% | 4 |

## Omission ledger

The miner writes `.omissions.json` after generating candidates:

- 38 `name_evidence` cases.
- Five `outside_scope` cases: one self-reference, one bare numeric source, and three coverage failures caused by bounded sampling/distinct-value retention.
- One `ambiguous_direction` case.

Declared FK labels support post-hoc classification, duplicate-cue suppression, and identification of new candidates requiring full-data verification. Candidate direction comes from keys/names. The ledger includes observed coverage for the three high-cardinality cases.

```bash
python scripts/research/build_relation_graph.py --database-root BIRD/dev/dev_20240627/dev_databases/dev_databases --gold-sql BIRD/dev/dev_20240627/dev.json
```

See the [machine-readable audit](../resources/bird_dev_20240627/thesis_relation_validation.json), [miner](../scripts/research/build_relation_graph.py), and [regression tests](../tests/test_thesis_alignment.py).
