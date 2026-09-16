# Phase 13C-A — Parser Benchmark 报告（Sandbox-Only）

- **生成时间**：2026-09-14T10:36:35.287463+00:00
- **沙盒根**：`C:\Users\chchxh\AppData\Local\Temp\evidenceflow_phase13c`
- **产物根**：`D:\Agent_learn\evidenceFlow\evals\evaluation_v3\phase13c_scientific_ingestion_benchmark`
- **状态**：`PARTIAL`
- **运行模式**：`REUSE_NATIVE`（映射复用 112 次 / 实际解析 22 次）

## 0. 一句话结论

5 个 arm 完成运行（1 个被环境阻塞）。受控语料上段落文本召回最高的是 `P2_docling`（0.9178），表格结构可得性、图注隔离与装饰处理是各 arm 的分水岭。进入 13C-B 的候选：`P0_current`、`P1_mineru`。

## 1. arm 可运行性与证据

| arm | 工具版本 | 可运行 | 说明 |
|---|---|---|---|
| `P0_current` | pymupdf==1.28.2 | YES | runs inside the project process |
| `P1_mineru` | mineru==3.4.5 | YES | venv=C:\Users\chchxh\AppData\Local\Temp\evidenceflow_phase13c\envs\mineru\Scripts\python.exe |
| `P2_docling` | docling==2.126.0 | YES | venv=C:\Users\chchxh\AppData\Local\Temp\evidenceflow_phase13c\envs\docling\Scripts\python.exe |
| `P3_marker` | marker-pdf==2.0.0 | YES | venv=C:\Users\chchxh\AppData\Local\Temp\evidenceflow_phase13c\envs\marker\Scripts\python.exe |
| `P4_grobid` | - | NO | 本机无 Java 运行时且无 Docker：GROBID 是 Java 服务（jar/容器），需要 JVM 与可运行的服务进程；sandbox 内无法满足。 |
| `O2_ocr` | rapidocr-onnxruntime==1.4.4 | YES | venv=C:\Users\chchxh\AppData\Local\Temp\evidenceflow_phase13c\envs\ocr\Scripts\python.exe |

**本机不可实现的 OCR 档位**（环境限制，不构成工具能力结论）：

- `P3_marker` / `auto`：surya-ocr-2 的 OCR 档需要 VLM 推理服务：surya 经 nvidia-smi 回退判定本机有 GPU 而选择 vllm 后端，该后端通过 Docker 拉起 vLLM 容器，但 `docker` 不在 PATH（SpawnError: docker binary not found）；CPU 后备的 llama.cpp 后端又需要 PATH 中的 `llama-server`，同样不存在。
- `P3_marker` / `force`：surya-ocr-2 的 OCR 档需要 VLM 推理服务：surya 经 nvidia-smi 回退判定本机有 GPU 而选择 vllm 后端，该后端通过 Docker 拉起 vLLM 容器，但 `docker` 不在 PATH（SpawnError: docker binary not found）；CPU 后备的 llama.cpp 后端又需要 PATH 中的 `llama-server`，同样不存在。

## 2. 受控语料有效性自证

- 判定方法：解析器无关的 PDF 文本层自证：①GT 文本是否出现在 GT 声称的页（page token recall）；②是否落在 GT 声称的 bbox 内（bbox token recall）；③该页是否存在 GT 未声明的幽灵文字（unclaimed ratio）。三项全部通过才认为语料可用于评测 parser。
- 通过模板数：11/11（阈值 0.95）

| 模板 | page 召回(最小) | bbox 召回(最小) | bbox 召回(均值) | 幽灵文字比例(最大) | 判定 |
|---|---|---|---|---|---|
| T01_single_column_plain | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T02_ieee_two_column | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T03_cvpr_two_column | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T04_acm_two_column | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T05_springer_lncs | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T06_elsevier_journal | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T07_arxiv_preprint | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T08_table_heavy | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T09_formula_heavy | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T10_figure_heavy | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T11_references_heavy | 1.0000 | 1.0000 | 1.0000 | 0.0000 | PASS |
| T12_scanned_image_only | - | - | - | - | 排除（无文本层） |

> 自证与解析器无关（直接校验 PDF 文本层），原因：首版用 P0 当探针会把 parser 的结构性局限（无表格结构、页码归入 furniture）误判成语料缺陷，因此改为直接校验 PDF 文本层。

## 3. 受控语料上的结构指标（各 arm 均值）

| arm | 段落文本召回 | 段落文本精度 | 精确匹配 | CER | 块对顺序 | Kendall tau | 双栏交叉率 | 标题 F1 | section_path | 段落碎片率 | 段落误合并率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `P0_current` | 0.8754 | 0.9298 | 0.9644 | 0.0201 | 0.9352 | 0.8705 | 0.0000 | 1.0000 | 0.9000 | 0.0000 | 0.0118 |
| `P1_mineru` | 0.6917 | 0.9488 | 0.9471 | 0.0235 | 0.9291 | 0.8583 | 0.0014 | 0.9683 | 0.9907 | 0.0238 | 0.0182 |
| `P2_docling` | 0.9178 | 0.9791 | 0.6054 | 0.0236 | 0.9036 | 0.8072 | 0.0226 | 0.9524 | 0.9478 | 0.0149 | 0.0138 |
| `P3_marker` | 0.6530 | 0.7155 | 0.9342 | 0.0267 | 0.9991 | 0.9981 | 0.0009 | 0.8543 | 0.0000 | 0.0060 | 0.0822 |
| `P4_grobid` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `O2_ocr` | 0.4327 | 0.4297 | 0.2826 | 0.3662 | 0.8894 | 0.7787 | 0.0583 | n/a | 0.0000 | 0.2083 | 0.1670 |

## 4. 结构元素与装饰（各 arm 均值）

> **图 / 表检出判据**：受控语料里的图是矢量绘制的（GT 文本为空），表格内容在结构化的 `table_structure` 里（扁平文本为空或只有片段）。对这类元素做 token 文本匹配**在构造上**恒为 0——它测不出「有没有检出」，只能测出「有没有一段可比的文字」。因此检出改用以**同页 bbox IoU ≥ 0.5** 为主判据（实测：真正对应的框 IoU 普遍在 0.80–0.98，不对应的多为 0），文本对齐作为补充证据（parser 把表格降级成纯文本时仍算检出），两者取并集。「几何」列的分母只算**有合法 bbox 的 GT 元素**——没有坐标的 GT 本来就无法做几何判定，把它计入分母等于编造一次失败。

> **几何精确率**（`表检出精确率` / `图检出精确率`）是召回的必要补充：分母是 parser 输出的、带合法 bbox 的该类元素数，分子是其中真的对应上一个 GT 元素的数量。缺了它，「把图统统标成表」这种错误只会在召回上拿满分——实测 Marker 在 `T10_figure_heavy`（GT 有 2 表 5 图）输出了 6 个 table 块（其中 4 个实际是图），表召回 1.0 而精确率仅 0.33。

> **公式为什么要拆成三列**（`公式检出` / `公式检出(几何)` / `公式检出(文本)`）：各 parser 对公式的**文本约定**不同，而 GT 用 asciiMath 纯文本。实测 `T09_formula_heavy`（GT 11 个公式）：Docling 输出同约定文本，文本召回 0.958；**MinerU 输出原始 LaTeX**（`$$ \mathrm { L } \operatorname { e n c } = ...`），与 GT 逐 token 不可比，文本召回仅 0.182——但它**确实检出了全部 11 个公式**（几何召回 1.000）。因此文本召回测的是「选了哪种输出约定」，几何召回才是「有没有检出」。公式的几何判据用**重叠系数**（`inter / min(面积)`，阈值 0.6）而不是 IoU：GT 的公式框是行级框（占满栏宽、含右侧编号），parser 的框紧贴字形，IoU 会被并集分母压低到 0.24 左右而低于 0.5 阈值。改用重叠系数后实测对应块最低 0.840、误配块最高 0.412，两簇相距 2 倍。每次运行的 `<arm>/latex_convention.comparable` 字段会显式标出该 arm 的公式文本是否与 GT 可比；为 `False` 时不得用文本召回给该 arm 的公式能力下结论。

| arm | 表检出 | 表检出(几何 IoU) | 表检出(文本) | 表检出精确率 | 表格行列关系 | 表头归属 | 跨列单元格 | 图检出 | 图检出(几何 IoU) | 图检出(文本) | 图检出精确率 | 图注检出 | 图注文本召回 | 图注隔离（ARR-007 类） | 公式检出 | 公式检出(几何) | 公式检出(文本) | 公式检出精确率 | 公式编号保留 | 装饰精确率 | 装饰召回 | 正文误删率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P0_current` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | n/a | 0.9167 | 0.9167 | 0.9167 | 0.2854 | 0.2854 | 0.2854 | 0.9091 | 1.0000 | 1.0000 | 0.5597 | 0.0938 |
| `P1_mineru` | 0.9583 | 0.8333 | 0.9583 | 0.7500 | 0.8750 | 1.0000 | 1.0000 | 0.5556 | 0.5556 | 0.0000 | 1.0000 | 0.8167 | 0.8397 | 0.3333 | 0.9444 | 0.9444 | 0.0152 | 1.0000 | 0.0000 | n/a | 0.0000 | 0.3590 |
| `P2_docling` | 1.0000 | 1.0000 | 0.0000 | 0.6361 | n/a | n/a | n/a | 0.2806 | 0.2806 | 0.0000 | 1.0000 | 0.3125 | 0.3213 | 0.0833 | 0.7816 | 0.7816 | 0.7399 | 1.0000 | 1.0000 | 0.9876 | 0.9479 | 0.0521 |
| `P3_marker` | 1.0000 | 1.0000 | 0.9167 | 0.5417 | 0.5202 | 0.5227 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | 0.9167 | 0.9167 | 0.9167 | 0.8333 | 0.8333 | 0.6667 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4017 |
| `P4_grobid` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `O2_ocr` | 0.6250 | 0.0000 | 0.6250 | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | 0.5833 | 0.5936 | 0.0833 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a | 0.0000 | 0.6512 |

### 4.1 「正文误删」的成因分解（决定门 5 该怎么读）

`furniture_false_removal` 的定义是「GT 正文元素**没有对齐到任何 parser 元素**的比例」。它**不是**「正文被当成装饰删掉的比例」——只要文本还能对上，正文即使被标成 header/footer 也**不算**丢失（见下方 `正文被标成装饰(均值)` 列，其底层字段是 `body_matched_to_furniture`）。因此这个数值背后可能是三件完全不同的事，必须分开读：

1. **分块/合并差异**（`高覆盖占比` 列）：token 全部能在文档别处找到，只是没对齐到单个元素；
2. **内容真的缺失**（`低覆盖占比` 列）：文档级 token 覆盖率 < 0.5，没抽出来；
3. 扫描页无文本层（看 `低覆盖集中处` 列落在哪些模板上）。

> **计数口径（重要）**：占比由**同一行的计数**现算（`高 / 未对齐数`），因此每一行内部自洽——同一行不会出现「计数算出 0.11、占比却写 0.83」这种矛盾。两列计数的单位是**受控 run**，不是文档：跑多个 OCR 档的 arm（MinerU、Docling 同时跑 `none` 与 `auto`）会在每个模板上贡献两次计数，因此计数的**跨 arm 绝对值不可直接比较**，必须先除以该 arm 的 `受控 run 数`。注意指标 JSON 里的 arm 级 `body_loss_high_coverage_ratio` / `body_loss_low_coverage_ratio` 是**逐文档均值**，与这里不同：一个文档可以装下几十个低覆盖元素（扫描页模板），逐文档均值会把它与只有 1 个元素的模板等同看待，而元素级占比会如实反映它的支配地位。归因「门 5 为何失败」要读元素级（本表），跨 arm 比较严重程度时才需要同时看两者。

| arm | 正文误删率 | 未对齐数(受控 run 合计) | 受控 run 数 | 高覆盖占比(分块差异) | 低覆盖占比(真缺失) | 正文被标成装饰(均值) | 低覆盖集中处 |
|---|---|---|---|---|---|---|---|
| `P0_current` | 0.0938 | 45 | 12 | 0.1111 | 0.8889 | 0.0000 | T12_scanned_image_only ×40 |
| `P1_mineru` | 0.3590 | 431 | 24 | 0.0441 | 0.5290 | 0.0000 | T11_references_heavy ×122、T01_single_column_plain ×10、T03_cvpr_two_column ×10 |
| `P2_docling` | 0.0521 | 50 | 24 | 0.2000 | 0.8000 | 0.0000 | T12_scanned_image_only ×40 |
| `P3_marker` | 0.4017 | 236 | 12 | 0.8178 | 0.1695 | 0.0000 | T12_scanned_image_only ×40 |
| `P4_grobid` | — | — | 0 | — | — | — | — |
| `O2_ocr` | 0.6512 | 366 | 12 | 0.8798 | 0.0000 | 0.0000 | — |

> **读法**：`正文被标成装饰` 一列在本轮语料上**全部为 0** —— 没有任何 arm 把正文误标为装饰。因此硬门 5 在这套语料上对这一维度**没有区分力**；它失败的原因来自前三类，与「装饰能否可靠过滤」无关。修订该门前应先决定：是要度量「装饰元素能否被单独标记」，还是要度量「正文内容是否完整」——这是两个不同的量，不应由同一个数承担。 `低覆盖集中处` 一列标出了每 arm 低覆盖（真缺失）数量最多的前三个模板：若多个 arm 集中同一模板，应连同该模板的语料特性（例如是否为扫描页）一起归因，而不是当成 parser 的通用缺陷。

## 5. 真实语料 canonical anchor 召回

| arm | 平均 anchor 召回 | 最低 anchor 召回 | 平均 page provenance |
|---|---|---|---|
| `P0_current` | 1.0000 | 1.0000 | 1.0000 |
| `P1_mineru` | 0.9800 | 0.8000 | 1.0000 |
| `P2_docling` | 0.9800 | 0.8000 | 1.0000 |
| `P3_marker` | 1.0000 | 1.0000 | 1.0000 |
| `P4_grobid` | n/a | n/a | n/a |
| `O2_ocr` | 0.9750 | 0.7500 | 1.0000 |

## 6. 历史失败 case 单列（方案 §28）

| 题号 | 锚点 | 文档 | 页 | anchor 文本 | `P0_current` | `P1_mineru` | `P2_docling` | `P3_marker` | `P4_grobid` | `O2_ocr` |
|---|---|---|---|---|---|---|---|---|---|---|
| ARR-007 | EVID-ARR-007-G1-A1 | pcn_point_completion_network.pdf | 3 | Figure 2: PCN architecture | HIT (1.00) | HIT (1.00) | HIT (1.00) | HIT (1.00) | MISS (0.00) | HIT (1.00) |
| ARR-037 | EVID-ARR-037-G1-A1 | pointr_geometry_aware_transformers.pdf | 2 | dynamic queries | HIT (1.00) | HIT (1.00) | HIT (1.00) | HIT (1.00) | MISS (0.00) | HIT (1.00) |
| ARR-037 | EVID-ARR-037-G2-A1 | adapointr.pdf | 4 | dynamic query | HIT (1.00) | HIT (1.00) | HIT (1.00) | HIT (1.00) | MISS (0.00) | HIT (1.00) |
| ARR-040 | EVID-ARR-040-G1-A1 | svdformer.pdf | 1 | self-structure | HIT (1.00) | HIT (1.00) | HIT (1.00) | HIT (1.00) | MISS (0.00) | HIT (1.00) |
| ARR-040 | EVID-ARR-040-G2-A1 | hyper_pcn.pdf | 1 | hypergraph | HIT (1.00) | HIT (1.00) | HIT (1.00) | HIT (1.00) | MISS (0.00) | HIT (1.00) |

## 7. 成本与工程指标（方案 §30）

| arm | 受控语料均耗时/页(s) | 真实语料均耗时/页(s) | 峰值内存(MB) | 需要 GPU | 解析页数 |
|---|---|---|---|---|---|
| `P0_current` | 0.368 | 0.027 | n/a | False | 189 |
| `P1_mineru` | 17.715 | 12.017 | n/a | False | 233 |
| `P2_docling` | 6.207 | 5.721 | n/a | False | 233 |
| `P3_marker` | 7.741 | 2.576 | n/a | False | 189 |
| `P4_grobid` | n/a | n/a | n/a | None | None |
| `O2_ocr` | 13.873 | 22.127 | n/a | False | 189 |

## 8. Parser Selection Gate（方案 §14）

> 门 6（可复现性）在模板 `T02_ieee_two_column` 上重复解析一次后比较规范化摘要：5/5 个 arm 可复现。

### `P0_current` — FAIL

通过 8/10 条可判定硬门。

| 门 | 结果 | 证据 |
|---|---|---|
| canonical_anchor_no_new_loss | PASS | {"baseline_hit_count": 44, "arm_hit_count": 44, "new_loss_count": 0, "new_loss_anchor_ids": [], "recovered_anchor_ids": []} |
| no_systematic_column_interleave | PASS | {"two_column_templates": ["none::T02_ieee_two_column", "none::T03_cvpr_two_column", "none::T04_acm_two_column", "none::T06_elsevier_journal", "none::T10_figure_heavy"], "interleaved_templates": []} |
| page_provenance_on_retained_elements | PASS | {"min_page_provenance_coverage": 1.0} |
| structured_elements_not_fully_downgraded | FAIL | {"table_detected_mean": 0.0, "figure_detection_mean": 0.0, "caption_detection_mean": 0.916667} |
| furniture_isolated_or_filtered | FAIL | {"max_parsed_furniture_count": 6, "mean_furniture_false_removal": 0.09375} |
| reproducible_output | PASS | {"template_id": "T02_ieee_two_column", "ocr_mode": "none", "checked": true, "reason": null, "digest_sha256_prefix": "e5cb55c3de269079"} |
| no_undocumented_external_llm | PASS | {"declaration": "无外部模型：纯 PyMuPDF 文本/块抽取 + 本地正则判定。"} |
| failures_locatable_to_pdf_page | PASS | {"failure_record_count": 0, "records_missing_attribution_keys": 0} |
| source_pdfs_untouched | PASS | {"verified_by": "input_hash_recheck"} |
| maps_to_unified_schema | PASS | {"validation_valid": true, "problem_count": 0} |


### `P1_mineru` — FAIL

通过 8/10 条可判定硬门。

| 门 | 结果 | 证据 |
|---|---|---|
| canonical_anchor_no_new_loss | FAIL | {"baseline_hit_count": 44, "arm_hit_count": 43, "new_loss_count": 1, "new_loss_anchor_ids": ["EVID-ARR-017-G1-A1"], "recovered_anchor_ids": []} |
| no_systematic_column_interleave | PASS | {"two_column_templates": ["none::T02_ieee_two_column", "none::T03_cvpr_two_column", "none::T04_acm_two_column", "none::T06_elsevier_journal", "none::T10_figure_heavy", "auto::T02_ieee_two_column", "auto::T03_cvpr_two_col |
| page_provenance_on_retained_elements | PASS | {"min_page_provenance_coverage": 1.0} |
| structured_elements_not_fully_downgraded | PASS | {"table_detected_mean": 0.958333, "figure_detection_mean": 0.555556, "caption_detection_mean": 0.816667} |
| furniture_isolated_or_filtered | FAIL | {"max_parsed_furniture_count": 0, "mean_furniture_false_removal": 0.358958} |
| reproducible_output | PASS | {"template_id": "T02_ieee_two_column", "ocr_mode": "none", "checked": true, "reason": null, "digest_sha256_prefix": "5ec1248e4f723215"} |
| no_undocumented_external_llm | PASS | {"declaration": "pipeline 后端，本地模型（layout/OCR/公式/表格），不调用任何 LLM API。"} |
| failures_locatable_to_pdf_page | PASS | {"failure_record_count": 0, "records_missing_attribution_keys": 0} |
| source_pdfs_untouched | PASS | {"verified_by": "input_hash_recheck"} |
| maps_to_unified_schema | PASS | {"validation_valid": true, "problem_count": 0} |


### `P2_docling` — FAIL

通过 7/10 条可判定硬门。

| 门 | 结果 | 证据 |
|---|---|---|
| canonical_anchor_no_new_loss | FAIL | {"baseline_hit_count": 44, "arm_hit_count": 43, "new_loss_count": 1, "new_loss_anchor_ids": ["EVID-ARR-017-G1-A1"], "recovered_anchor_ids": []} |
| no_systematic_column_interleave | FAIL | {"two_column_templates": ["none::T02_ieee_two_column", "none::T03_cvpr_two_column", "none::T04_acm_two_column", "none::T06_elsevier_journal", "none::T10_figure_heavy", "auto::T02_ieee_two_column", "auto::T03_cvpr_two_col |
| page_provenance_on_retained_elements | PASS | {"min_page_provenance_coverage": 1.0} |
| structured_elements_not_fully_downgraded | PASS | {"table_detected_mean": 1.0, "figure_detection_mean": 0.280556, "caption_detection_mean": 0.3125} |
| furniture_isolated_or_filtered | FAIL | {"max_parsed_furniture_count": 7, "mean_furniture_false_removal": 0.052083} |
| reproducible_output | PASS | {"template_id": "T02_ieee_two_column", "ocr_mode": "none", "checked": true, "reason": null, "digest_sha256_prefix": "52c214af3084e78c"} |
| no_undocumented_external_llm | PASS | {"declaration": "standard pipeline，本地模型（layout heron + TableFormerV2）；未启用 VLM pipeline。"} |
| failures_locatable_to_pdf_page | PASS | {"failure_record_count": 0, "records_missing_attribution_keys": 0} |
| source_pdfs_untouched | PASS | {"verified_by": "input_hash_recheck"} |
| maps_to_unified_schema | PASS | {"validation_valid": true, "problem_count": 0} |


### `P3_marker` — FAIL

通过 8/10 条可判定硬门。

| 门 | 结果 | 证据 |
|---|---|---|
| canonical_anchor_no_new_loss | PASS | {"baseline_hit_count": 44, "arm_hit_count": 44, "new_loss_count": 0, "new_loss_anchor_ids": [], "recovered_anchor_ids": []} |
| no_systematic_column_interleave | PASS | {"two_column_templates": ["none::T02_ieee_two_column", "none::T03_cvpr_two_column", "none::T04_acm_two_column", "none::T06_elsevier_journal", "none::T10_figure_heavy"], "interleaved_templates": []} |
| page_provenance_on_retained_elements | PASS | {"min_page_provenance_coverage": 1.0} |
| structured_elements_not_fully_downgraded | FAIL | {"table_detected_mean": 1.0, "figure_detection_mean": 0.0, "caption_detection_mean": 0.916667} |
| furniture_isolated_or_filtered | FAIL | {"max_parsed_furniture_count": 7, "mean_furniture_false_removal": 0.401667} |
| reproducible_output | PASS | {"template_id": "T02_ieee_two_column", "ocr_mode": "none", "checked": true, "reason": null, "digest_sha256_prefix": "197052bd26c02e12"} |
| no_undocumented_external_llm | PASS | {"declaration": "fast 模式：本地轻量检测器 + 按块 OCR；未使用 LLM 服务（--llm_service 未设置）。"} |
| failures_locatable_to_pdf_page | PASS | {"failure_record_count": 0, "records_missing_attribution_keys": 0} |
| source_pdfs_untouched | PASS | {"verified_by": "input_hash_recheck"} |
| maps_to_unified_schema | PASS | {"validation_valid": true, "problem_count": 0} |


### `P4_grobid` — BLOCKED（未评估，非工具能力结论）

| 门 | 结果 | 证据 |
|---|---|---|
| availability | BLOCKED | {"available": false, "reason": "本机无 Java 运行时且无 Docker：GROBID 是 Java 服务（jar/容器），需要 JVM 与可运行的服务进程；sandbox 内无法满足。"} |


### `O2_ocr` — FAIL

通过 6/10 条可判定硬门。

| 门 | 结果 | 证据 |
|---|---|---|
| canonical_anchor_no_new_loss | FAIL | {"baseline_hit_count": 44, "arm_hit_count": 43, "new_loss_count": 1, "new_loss_anchor_ids": ["EVID-ARR-003-G1-A1"], "recovered_anchor_ids": []} |
| no_systematic_column_interleave | FAIL | {"two_column_templates": ["force::T02_ieee_two_column", "force::T03_cvpr_two_column", "force::T04_acm_two_column", "force::T06_elsevier_journal", "force::T10_figure_heavy"], "interleaved_templates": ["force::T02_ieee_two |
| page_provenance_on_retained_elements | PASS | {"min_page_provenance_coverage": 1.0} |
| structured_elements_not_fully_downgraded | FAIL | {"table_detected_mean": 0.625, "figure_detection_mean": 0.0, "caption_detection_mean": 0.583333} |
| furniture_isolated_or_filtered | FAIL | {"max_parsed_furniture_count": 0, "mean_furniture_false_removal": 0.65125} |
| reproducible_output | PASS | {"template_id": "T02_ieee_two_column", "ocr_mode": "force", "checked": true, "reason": null, "digest_sha256_prefix": "2f9d59409c77a6db"} |
| no_undocumented_external_llm | PASS | {"declaration": "RapidOCR（本地 ONNX 模型），不调用 LLM。"} |
| failures_locatable_to_pdf_page | PASS | {"failure_record_count": 0, "records_missing_attribution_keys": 0} |
| source_pdfs_untouched | PASS | {"verified_by": "input_hash_recheck"} |
| maps_to_unified_schema | PASS | {"validation_valid": true, "problem_count": 0} |


## 9. 入选下一轮（方案 §33 要求 STOP 于此）

1. `P0_current`（**有条件**，未过门外） — 通过 8/10 条硬门，未过门；仅作为有条件候选列出，需在 13C-B 前先解决未过门项。
2. `P1_mineru`（**有条件**，未过门外） — 通过 8/10 条硬门，未过门；仅作为有条件候选列出，需在 13C-B 前先解决未过门项。

**未入选 / 未过门的 arm**：

- `P2_docling`：未过门 ['canonical_anchor_no_new_loss', 'no_systematic_column_interleave', 'furniture_isolated_or_filtered']
- `P3_marker`：未过门 ['structured_elements_not_fully_downgraded', 'furniture_isolated_or_filtered']
- `P4_grobid`：BLOCKED — 本机无 Java 运行时且无 Docker：GROBID 是 Java 服务（jar/容器），需要 JVM 与可运行的服务进程；sandbox 内无法满足。
- `O2_ocr`：未过门 ['canonical_anchor_no_new_loss', 'no_systematic_column_interleave', 'structured_elements_not_fully_downgraded', 'furniture_isolated_or_filtered']

> **STOP**：本阶段为 Phase 13C-A（PDF → parser → structured representation）。不进入 13C-B（Cleaning / Chunking），不修改任何生产路径。

## 10. 未决问题与限制

- 受控语料为程序化渲染（PyMuPDF base-14 字体 + 手工排版），不是真实 LaTeX 编译产物；因此公式指标只比较文本/符号层面的还原，不代表真实 LaTeX 语义还原能力。
- 受控语料中表格由网格线绘制、单元格为独立文本，真实论文表格常以矢量线 + 跨列合并呈现，本阶段的 row/column 指标偏向乐观。
- 扫描件模板（T12）由矢量页栅格化而来（200 dpi），与真实扫描件的噪声/倾斜/装订阴影不同，因此 OCR 相关指标只体现「有无文本层」这一主要变量。
- 本机装有 GPU（`nvidia-smi -L` 报告 NVIDIA GeForce RTX 5060 Laptop GPU），但环境中的 torch 是 CPU-only wheel（2.13.0+cpu），`torch.cuda.is_available()` 为 False，因此所有 arm 均实际运行在 CPU 上，耗时指标不能外推到 GPU 部署。
- 各 arm 的成本按「每个文档一个独立进程」计量，因此包含每进程一次的模型加载开销（MinerU 首次冷启动约 139s，其中大部分是模型加载/初始化），不代表稳定状态下的吞吐。
- GROBID 需要 Java 运行时或 Docker 服务，本机两者都不具备，故 P4 未获得任何结构化输出。
- Marker 的 OCR 档（auto/force）需要 surya-ocr-2 的 VLM 推理服务：surya 判定本机有 GPU 而选择 vllm 后端，该后端用 Docker 拉起容器，但本机无 docker；CPU 后备的 llama.cpp 需要 PATH 中的 llama-server，同样不存在。故 P3 只提供原生文本层档（--disable_ocr）。
- Marker 的 balanced 模式需要 VLM 布局模型 + 全页 OCR，本机 CPU 不可行，仅运行 fast 模式。
- MinerU 默认后端（hybrid-engine / vlm-engine）需要额外 VLM，本阶段只运行 pipeline 后端。
- reading order 的 Kendall tau 与块对准确率只基于「成功匹配」的元素对，因此不惩罚匹配失败所隐含的顺序错误；匹配失败由 missing / fragment 指标单独承担。
- 受控语料每个模板仅 3–5 页（一份完整短论文的内容量），页数较少使「页级装饰重复检测」这类依赖跨页重复的启发式难以充分展开。
- 图 / 表的检出用「同页 bbox IoU ≥ 0.5」判定。阈值 0.5 依据实测分布选取（对应块 IoU 0.80–0.98、不对应块多为 0），但 IoU 只比较外接矩形，对「框对而内容错」不敏感——该风险由 caption_text_recall 与表格单元格指标承担，不在此处重复度量。
- **公式的几何检出用「同页重叠系数 ≥ 0.6」而不是 IoU**，因为 GT 的公式框是行级框（占满栏宽、含右侧编号），parser 的框紧贴字形，两者面积差 2–3 倍，IoU 被并集分母系统性压低（实测对应块中位仅 0.24，低于 0.5 阈值，会把真实检出判成未检出）。重叠系数（inter / min(面积)）对框大小约定不敏感：实测对应块最低 0.840、同页误配块最高 0.412、跨页误配最高 0.253，两簇相距 2 倍，0.6 落在空隙正中。**两种判据的分数不可跨类型直接比较**（公式 0.6 / 图·表 0.5），每条命中记录的 criterion 字段标明了所用判据。
- 公式的文本级指标（normalized_latex_similarity、equation_detection_by_text）**受输出约定影响，不可用于跨 arm 比较**：GT 为 asciiMath 纯文本，而 MinerU 输出原始 LaTeX。每次运行的 latex_convention.comparable 字段标明该 arm 的约定是否与 GT 可比；为 False 时结论须以 equation_detection_by_geometry 为准。本阶段无法渲染真实 LaTeX，LaTeX 语义还原能力未被度量。
- 若某个 parser 输出了 bbox 却没有页码，几何匹配照常进行并计入 `page_unverified_match_count`，而不是被直接判为未检出：页码缺失是 provenance 缺陷，应由 page_provenance_coverage 承担，两类缺陷不应互相掩盖。
- `furniture_false_removal` 的语义是「**GT 正文元素未被任何 parser 元素对齐**的比例」，包含两种成因：被 parser 当作装饰丢弃，以及被彻底漏掉（抽取失败）。因此它的值**不能**单独读作「家具误删率」——实测 MinerU 的 `parsed_furniture_count` 为 0（一个装饰元素都没标），却因 36% 正文元素无法对齐而有 `furniture_false_removal = 0.359`。门 5 是**两个条件的并列**（能单独标记 且 误删 ≤ 2%），此时它失败在第一条件上；判定正确，但归因必须读 gate evidence 里的 `max_parsed_furniture_count` 与 `mean_furniture_false_removal` 两项，不能只看后者。
- 门 5 的第二个条件（误删 ≤ 2%）在**本语料上对全部已评估 arm 都不可满足**，但它失败的主因不是「装饰过滤不可靠」：按文档级 token 覆盖率分解后，P0/P2/P3 的低覆盖元素全部集中在同一处（T12 扫描页无文本层，各 40），MinerU 另有一处集中来源（T11 参考文献段，每 OCR 档 61），OCR arm 的低覆盖为 0；而「正文文本对得上、却被标成 header/footer」在全部 arm 上均为 0 次。即该门在本语料上对被评估对象**缺少区分力**，是给每个候选加同一个失败项。修订该门前需先决定要度量的是「装饰元素能否被单独标记」还是「正文内容是否完整」。
- 受控指标的**计数**（`未对齐数` / `低覆盖集中处`）粒度是 run 而非文档：同时跑 `none` 与 `auto` 两个 OCR 档的 arm，会在每个模板上贡献两次计数。跨 arm 比较计数前必须先除以各 arm 的受控 run 数。报告 §4.1 的两个占比列由**同一行的计数现算**（元素级空间，行内自洽），与指标 JSON 里的 arm 级 `body_loss_*_coverage_ratio`（**逐文档均值**）不是同一个量：某文档独占几十个元素时两者可相差数倍，归因要读元素级，跨 arm 比较严重程度时需同时看两者。
