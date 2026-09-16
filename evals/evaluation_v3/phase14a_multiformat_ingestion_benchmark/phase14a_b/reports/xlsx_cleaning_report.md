# Phase 14A-B XLSX Cleaning Report

- **XLSX-B0**: `PASS`  |  **XLSX-B1**: `PASS`
- **Real corpus**: PENDING（受控基准 ≠ 真实语料验证）

## 逐 case 硬门（X-G1..G10）
| Case | XLSX-B0 | XLSX-B1 |
|---|---|---|
| X1 | PASS | PASS |
| X2 | PASS | PASS |
| X3 | PASS | PASS |
| X4 | PASS | PASS |
| X5 | PASS | PASS |
| X6 | PASS | PASS |
| X7 | PASS | PASS |
| X8 | PASS | PASS |
| X9 | PASS | PASS |
| X10 | PASS | PASS |
| X11 | PASS | PASS |
| X12 | PASS | PASS |
| X13 | PASS | PASS |
| X14 | PASS | PASS |
| X15 | PASS | PASS |

## 指标均值（case 均值；None = 该 arm 不产生该指标，未计入；B0 无区域推断，inferred_table_* 对 B0 语义上不适用）
| Metric | XLSX-B0 | XLSX-B1 |
|---|---|---|
| non_empty_cell_recall | 1.0 | 1.0 |
| cell_coordinate_accuracy | 1.0 | 1.0 |
| cell_type_accuracy | 0.981 | 0.981 |
| formula_recall | 1.0 | 1.0 |
| cached_value_preservation | 1.0 | 1.0 |
| formula_silent_overwrite | 0.0 | 0.0 |
| merged_range_accuracy | 1.0 | 1.0 |
| merge_anchor_coverage | 1.0 | 1.0 |
| named_table_accuracy | 1.0 | 1.0 |
| inferred_table_precision | None | 1.0 |
| inferred_table_recall | 0.0 | 0.9286 |
| header_hierarchy_accuracy | 0.0 | 1.0 |
| hidden_state_accuracy | 1.0 | 1.0 |
| hidden_role_policy_ok | True | True |
| sheet_order_accuracy | 1.0 | 1.0 |
| number_format_preserved | None | None |
| display_overwrite_count | 0.0 | 0.0 |
| auto_filter_preserved | True | True |
| source_ref_coverage | 1.0 | 1.0 |
| duplicate_cell_rate | 0.0 | 0.0 |
| fake_provenance_count | 0.0 | 0.0 |
| cell_elements | 11.7333 | 11.7333 |
| table_elements | 0.0667 | 1.3333 |
| cleaning_log_count | 0.0 | 2.1333 |
| unresolved_count | 0.0 | 0.0 |