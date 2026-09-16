# Phase 13C-B — Fusion 分析（B2）

- **生成时间**：2026-09-14T12:40:40.009540+00:00
- **Fusion 版本**：`phase13c-b-fusion-1.0.0`
- **对齐键优先级**（方案 §6）：same page -> bbox overlap/containment -> normalized text overlap -> section proximity。

## 1. 对齐结果

| 文档 | scope | 已对齐 | P0-only | P1-only | 对齐率 |
|---|---|---|---|---|---|
| `T01_single_column_plain` | controlled | 52 | 12 | 3 | 0.7761 |
| `T02_ieee_two_column` | controlled | 52 | 13 | 3 | 0.7647 |
| `T03_cvpr_two_column` | controlled | 56 | 11 | 3 | 0.8000 |
| `T04_acm_two_column` | controlled | 50 | 13 | 3 | 0.7576 |
| `T05_springer_lncs` | controlled | 53 | 11 | 3 | 0.7910 |
| `T06_elsevier_journal` | controlled | 50 | 15 | 3 | 0.7353 |
| `T07_arxiv_preprint` | controlled | 52 | 14 | 1 | 0.7761 |
| `T08_table_heavy` | controlled | 54 | 20 | 3 | 0.7013 |
| `T09_formula_heavy` | controlled | 60 | 12 | 2 | 0.8108 |
| `T10_figure_heavy` | controlled | 54 | 12 | 6 | 0.7500 |
| `T11_references_heavy` | controlled | 115 | 11 | 3 | 0.8915 |
| `T12_scanned_image_only` | controlled | 0 | 0 | 21 | 0.0000 |
| `adapointr` | real | 290 | 323 | 13 | 0.4633 |
| `hyper_pcn` | real | 187 | 145 | 16 | 0.5374 |
| `pcn_point_completion_network` | real | 204 | 44 | 33 | 0.7260 |
| `pfnet_point_fractal_network` | real | 129 | 53 | 9 | 0.6754 |
| `pointattn` | real | 151 | 45 | 6 | 0.7475 |
| `pointr_geometry_aware_transformers` | real | 171 | 137 | 9 | 0.5394 |
| `proxyformer` | real | 158 | 212 | 7 | 0.4191 |
| `seedformer` | real | 134 | 109 | 18 | 0.5134 |
| `snowflakenet` | real | 143 | 164 | 11 | 0.4497 |
| `svdformer` | real | 155 | 102 | 11 | 0.5784 |

## 2. 冲突分布

| 文档 | 冲突总数 | 类型冲突 | 章节冲突 | 文本分歧 | 图注取舍 | 未决 | 抑制的 P1 元素 |
|---|---|---|---|---|---|---|---|
| `T01_single_column_plain` | 22 | 6 | 9 | 6 | 1 | 0 | 1 |
| `T02_ieee_two_column` | 20 | 5 | 5 | 9 | 1 | 0 | 3 |
| `T03_cvpr_two_column` | 20 | 6 | 7 | 7 | 0 | 0 | 1 |
| `T04_acm_two_column` | 21 | 6 | 6 | 8 | 1 | 0 | 3 |
| `T05_springer_lncs` | 25 | 7 | 10 | 7 | 1 | 0 | 2 |
| `T06_elsevier_journal` | 13 | 4 | 3 | 6 | 0 | 0 | 1 |
| `T07_arxiv_preprint` | 27 | 10 | 9 | 7 | 1 | 0 | 2 |
| `T08_table_heavy` | 25 | 6 | 9 | 8 | 2 | 0 | 5 |
| `T09_formula_heavy` | 45 | 13 | 17 | 14 | 1 | 0 | 1 |
| `T10_figure_heavy` | 23 | 5 | 9 | 8 | 1 | 0 | 2 |
| `T11_references_heavy` | 24 | 6 | 10 | 7 | 1 | 0 | 3 |
| `T12_scanned_image_only` | 1 | 0 | 0 | 0 | 0 | 1 | 23 |
| `adapointr` | 612 | 110 | 290 | 182 | 29 | 1 | 14 |
| `hyper_pcn` | 285 | 29 | 135 | 109 | 11 | 1 | 3 |
| `pcn_point_completion_network` | 301 | 22 | 150 | 115 | 13 | 1 | 19 |
| `pfnet_point_fractal_network` | 211 | 20 | 91 | 86 | 13 | 1 | 5 |
| `pointattn` | 227 | 26 | 103 | 89 | 8 | 1 | 0 |
| `pointr_geometry_aware_transformers` | 266 | 25 | 124 | 102 | 14 | 1 | 1 |
| `proxyformer` | 219 | 22 | 107 | 86 | 3 | 1 | 17 |
| `seedformer` | 303 | 72 | 130 | 88 | 12 | 1 | 40 |
| `snowflakenet` | 231 | 23 | 81 | 114 | 12 | 1 | 12 |
| `svdformer` | 219 | 30 | 95 | 84 | 9 | 1 | 7 |

## 3. 冲突处理规则（全部显式记录，无静默覆盖）

| 情形 | 规则 | 严重度 |
|---|---|---|
| 正文段落文本不同 | P0 为权威文本源（§7.1）；P1 原文留在 `source_refs` | INFO / RESOLVED |
| P0 该处无文本、P1 有 | 按 §7.1 例外由 P1 补齐 | RESOLVED |
| 类型不一致 | 结构型（table/figure/equation/caption/section）采信 P1；正文型保留 P0 | RESOLVED / INFO |
| section path 不一致 | 采信 P1（§7.2），两侧路径都记下来 | RESOLVED |
| 表格 | 采信 P1 结构，同时保留 P0 原文作 fallback 与校验（§7.3） | RESOLVED |
| 图注 | 取「隔离度更高」的一侧（单句 vs 被正文稀释） | RESOLVED |
| 公式 | 双表示并存（`raw_text` + `normalized_latex`），禁止互相覆盖（§7.5） | RESOLVED |
| 参考文献无法对齐 | 不合并，记 `REFERENCE_ALIGNMENT_UNCERTAIN`（§7.6） | **UNRESOLVED** |

## 4. 内容不重复的保障

1. P1-only 的**正文型**元素一律不加入（`suppressed_secondary_elements`），正文由 P0 独家提供；
2. P0 与 P1 的参考文献条目切分方式不同 -> 一律不合并，只进 `reference_sidecar`；
3. 加入前先查重：P1-only 元素若文本已被同页已入列元素覆盖 ≥ 0.90 -> `DUPLICATE_SUPPRESSED`。

- 融合引入的重复（至少一侧 P1_ONLY）：受控均值 0.0000，真实均值 0.0000；**实测为 0**，即 §35 的「fusion 出现 duplicate body 内容且无法去重」未触发。
