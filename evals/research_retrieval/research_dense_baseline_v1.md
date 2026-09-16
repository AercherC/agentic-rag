# Academic Research Current Dense Baseline v1

- Dataset hash: `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`
- Strategy: current Original/Translation Dense Top-10 per branch + Equal RRF；evaluation union depth 20。
- 未运行 BM25、Hybrid、Reranker、Query Rewrite 或参数搜索。

## Overall

| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 40 | 0.175 | 0.350 | 0.475 | 0.650 | 0.650 | 0.302 | 0.404 |

## language_group

| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chinese_to_english | 18 | 0.278 | 0.444 | 0.667 | 0.833 | 0.833 | 0.421 | 0.518 |
| english_to_english | 18 | 0.111 | 0.333 | 0.389 | 0.611 | 0.611 | 0.250 | 0.335 |
| mixed_multiple_sources | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.198 |

## question_type

| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation | 4 | 0.250 | 0.500 | 0.500 | 0.500 | 0.500 | 0.375 | 0.408 |
| conclusion_limitation | 2 | 0.500 | 0.500 | 0.500 | 1.000 | 1.000 | 0.571 | 0.667 |
| experiment_setup | 5 | 0.000 | 0.200 | 0.200 | 0.400 | 0.400 | 0.133 | 0.197 |
| implementation | 5 | 0.400 | 0.400 | 0.600 | 0.800 | 0.800 | 0.475 | 0.549 |
| method | 8 | 0.250 | 0.750 | 0.875 | 0.875 | 0.875 | 0.469 | 0.570 |
| motivation_concept | 6 | 0.000 | 0.000 | 0.333 | 0.833 | 0.833 | 0.150 | 0.307 |
| multi_document | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.198 |
| quantitative_results | 6 | 0.167 | 0.333 | 0.500 | 0.667 | 0.667 | 0.288 | 0.377 |

## evidence_type

| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| body_text | 22 | 0.182 | 0.364 | 0.500 | 0.773 | 0.773 | 0.330 | 0.434 |
| figure_caption | 4 | 0.250 | 0.500 | 0.750 | 0.750 | 0.750 | 0.396 | 0.483 |
| multi_document | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.198 |
| supplementary | 4 | 0.250 | 0.500 | 0.500 | 0.500 | 0.500 | 0.375 | 0.408 |
| table | 6 | 0.167 | 0.333 | 0.500 | 0.667 | 0.667 | 0.288 | 0.377 |

## paper

| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adapointr.pdf | 4 | 0.000 | 0.250 | 0.250 | 0.250 | 0.250 | 0.125 | 0.202 |
| hyper_pcn.pdf | 5 | 0.200 | 0.200 | 0.600 | 0.800 | 0.800 | 0.333 | 0.566 |
| pcn_point_completion_network.pdf | 5 | 0.000 | 0.000 | 0.000 | 0.200 | 0.200 | 0.033 | 0.071 |
| pfnet_point_fractal_network.pdf | 2 | 0.000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.167 | 0.250 |
| pointattn.pdf | 5 | 0.400 | 0.400 | 0.400 | 0.600 | 0.600 | 0.429 | 0.467 |
| pointr_geometry_aware_transformers.pdf | 5 | 0.000 | 0.400 | 0.600 | 0.600 | 0.600 | 0.173 | 0.313 |
| proxyformer.pdf | 4 | 0.250 | 0.250 | 0.500 | 0.750 | 0.750 | 0.348 | 0.441 |
| seedformer.pdf | 5 | 0.000 | 0.400 | 0.600 | 0.800 | 0.800 | 0.242 | 0.375 |
| snowflakenet.pdf | 4 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| svdformer.pdf | 5 | 0.200 | 0.400 | 0.400 | 0.800 | 0.800 | 0.357 | 0.582 |

## Failure diagnosis

- candidate_retrieval_failure: 12
- chunk_failure: 2
- extraction_failure: 0
- ranking_failure: 7
- success: 19

## Academic-specific diagnostic labels

- figure_semantics_missing: 1
- formula_representation: 0
- reference_pollution: 11
- section_boundary: 2
- supplement_boundary: 2
- table_structure_loss: 3

## Retrieval artifacts

- Top-5 中出现 References Chunk 的问题：22/40。
- 40×Top-5 中的 References Chunk 总数：32。

诊断标签是相关性线索，不把版面问题自动宣称为唯一因果；逐题候选、页码和分支排名保存在 JSON。
