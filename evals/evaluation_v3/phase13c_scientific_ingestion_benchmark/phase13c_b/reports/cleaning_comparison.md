# Phase 13C-B — Cleaning Benchmark 对照报告（Sandbox-Only）

- **生成时间**：2026-09-14T12:40:39.988997+00:00
- **Cleaner 版本**：`phase13c-b-cleaner-1.1.0`
- **Fusion 版本**：`phase13c-b-fusion-1.0.0`
- **Gate 定义版本**：`phase13c-b-gate-definitions-v2.2.0`
- **沙盒根**：`C:\Users\chchxh\AppData\Local\Temp\evidenceflow_phase13c`
- **产物根**：`D:\Agent_learn\evidenceFlow\evals\evaluation_v3\phase13c_scientific_ingestion_benchmark\phase13c_b`

## 0. 一句话结论

三个候选 arm 在真实语料上的 canonical anchor 命中为 `B0_p0_cleaned` 44/44, `B1_mineru_cleaned` 44/44, `B2_fused` 44/44；通过全部 10 条硬门的 arm：`B0_p0_cleaned`, `B2_fused`。B1 的正文保持提升来自两处**可证的映射/规范化修复**，而不是解析器换用。

## 1. canonical evidence anchor（真实语料 10 篇）

| arm | 清洗前命中 | 清洗后命中 | 新增丢失 | 恢复 | 新增丢失锚点 |
|---|---|---|---|---|---|
| `B0_p0_cleaned` | 44 | 44 | 0 | 0 | — |
| `B1_mineru_cleaned` | 44 | 44 | 0 | 0 | — |
| `B2_fused` | 44 | 44 | 0 | 0 | — |

> **ARR-017 的归因（方案 §11 / Q3）**：13C-A 记录 P1_mineru 在 `seedformer.pdf` 上丢失 `EVID-ARR-017-G1-A1`（`Np = 128`）。逐元素追查后确认三类原因中的两类同时存在：①**映射缺口**——MinerU 原生 `content_list` 里有 `page_number` / `header` / `footer` / `aside_text` / `list(sub_type=ref_text)` / `chart` 六种类型，13C-A 的映射层全部折叠成文本为空的 `unknown`；②**表示约定差异**——该 anchor 的内容确实在输出里，形态是行内公式 `$N _ { p } = 1 2 8$`，tokenizer 切出 `n/p/1/2/8`，取不到 `np`/`128`。Cleaner 分别用「按原生 content_list 再标注」与「数学跨度词法规范化」修复了这两类，因此 **B1 的 anchor 从 43/44 回到 44/44**。这不是解析器能力不足，而是本工程映射层的缺口。

## 2. 正文保持（受控 12 模板 + 真实 10 篇）

| arm | 正文 token 覆盖 | 未对齐正文元素 | 真误删 | 无文本层页正文 | 低覆盖元素 |
|---|---|---|---|---|---|
| `B0_p0_cleaned` | 0.9167 | 0.8747 | 0.0000 | 40.0000 | 0.0000 |
| `B1_mineru_cleaned` | 0.9962 | 0.9362 | 1.0000 | 1.0000 | 0.0000 |
| `B2_fused` | 0.9405 | 0.9116 | 0.0000 | 30.0000 | 0.0000 |

> `真误删`（`unintended_body_loss_count`）**只统计「该页解析器有产出、但正文未被覆盖」的元素**；无文本层页（扫描件）的正文单独记在 `无文本层页正文` 列，不参与误删判定。这是方案 §3 的修订：13C-A 的门 5 把这两类混在同一个 `false_removal` 里，于是 T12 扫描模板的 40 个元素掩盖了真实结论。

## 3. 装饰可分性（Gate 5A，与正文完整度完全解耦）

| arm | 装饰召回 | 装饰精确率 | 正文被标成装饰 | 删除记录数 | 记录可追溯率 | 可追溯性 |
|---|---|---|---|---|---|---|
| `B0_p0_cleaned` | n/a | n/a | 0.0000 | 4.2 | 1.0000 | n/a |
| `B1_mineru_cleaned` | n/a | n/a | 0.0021 | 5.0 | 1.0000 | n/a |
| `B2_fused` | n/a | n/a | 0.0000 | 4.2 | 1.0000 | n/a |

> 新口径下装饰通过条件不再要求 `parsed_furniture_count > 0`：「显式检测」**或**「可追溯删除记录」二者之一即可。B1 的显式检测来自 MinerU 原生的 `header`/`footer`/`page_number`/`aside_text` 类型（13C-A 的映射层把它们丢掉了，因此当时 P1 的 `parsed_furniture_count` 为 0）。

## 4. 章节结构

| arm | 标题 F1 | section_path | 章节冲突 | 跨章节误合并 | 双栏交叉率 |
|---|---|---|---|---|---|
| `B0_p0_cleaned` | 1.0000 | 0.9000 | 0.92 | 16.83 | 0.0000 |
| `B1_mineru_cleaned` | 0.9643 | 0.9815 | 0.17 | 17.42 | 0.0019 |
| `B2_fused` | 0.9881 | 0.9815 | 0.17 | 17.17 | 0.0000 |

## 5. 结构元素

> 表 / 图 / 公式的检出用几何判据（表与图 IoU ≥ 0.5、公式重叠系数 ≥ 0.6），因为受控语料里图是矢量绘制（GT 文本为空）、表格内容在结构化字段里、公式的文本约定各 arm 不同——文本匹配在这三类上测的是「选了哪种输出约定」。
| arm | 表检出 | 表检出精确率 | 行列关系 | 表头归属 | 图检出 | 图注文本召回 | 公式检出(几何) | 公式编号保留 | 公式双表示 |
|---|---|---|---|---|---|---|---|---|---|
| `B0_p0_cleaned` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 13 |
| `B1_mineru_cleaned` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 42 |
| `B2_fused` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 46 |

- 受控语料结构元素计数（B2）：{"type_counts": {"document_title": 11, "equation": 46, "figure": 21, "figure_caption": 29, "paragraph": 279, "reference_entry": 197, "section_header": 110, "table": 78, "table_caption": 25}, "table_count": 78, "structured_table_count": 15, "figure_count": 21, "figure_with_caption_count": 20, "equation_count": 46, "equation_with_dual_representation_count": 46, "structured_types_present": ["equation", "figure", "figure_caption", "table", "table_caption"]}

## 6. 参考文献（Reference Preservation）

| arm | 章节检测 | 条目数 | 条目召回 | 条目精确率 | 正文污染 | 误合并率 | 碎片率 |
|---|---|---|---|---|---|---|---|
| `B0_p0_cleaned` | n/a | n/a | 1.0000 | 0.9660 | 0.0000 | 0.0000 | 0.0000 |
| `B1_mineru_cleaned` | n/a | n/a | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `B2_fused` | n/a | n/a | 1.0000 | 0.9660 | 0.0000 | 0.0000 | 0.0000 |

> T11_references_heavy 的 GT 有 72 条参考文献。13C-A 的 P1 输出把它们折叠成一个文本为空的 `unknown`，因此 `T11` 贡献了 122 个「低覆盖正文误删」。Cleaner 按 MinerU 原生 `list_items`（解析器已给出的边界，**不是文本猜测**）逐条展开为 `reference_entry`。

## 7. 来源可追溯

| arm | page 覆盖 | bbox 覆盖 | source_parser | 融合可追溯 |
|---|---|---|---|---|
| `B0_p0_cleaned` | 1.0000 | 1.0000 | 1.0000 | n/a |
| `B1_mineru_cleaned` | 1.0000 | 1.0000 | 1.0000 | n/a |
| `B2_fused` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 8. 融合（B2）

| 指标 | 受控语料均值 | 真实语料均值 |
|---|---|---|
| `aligned_element_count` | 54.000 | 172.200 |
| `primary_only_count` | 12.000 | 133.400 |
| `secondary_only_count` | 4.500 | 13.300 |
| `conflict_count` | 22.167 | 287.400 |
| `unresolved_conflict_count` | 0.083 | 1.000 |
| `duplicate_content_rate` | 0.000 | 0.000 |
| `pre_existing_duplicate_pair_count` | 0.167 | 19.900 |

> `duplicate_content_rate` **只统计融合引入的重复**（至少一侧 `fusion_status=P1_ONLY`）。`pre_existing_duplicate_pair_count` 是 P0 自身输出的重复文本块对数（实测 `adapointr` 有 73 对：P0 把表格单元格与表头都当作独立段落输出）。两者口径不同，不可相加——把解析器层的既存重复记成融合缺陷会是错误归因。

## 9. Conditional OCR

- 判定：`PASS`
- 触发页数：3；born-digital 误触发：0
- 触发规则的三种原因与验证方式见 `metrics/ocr_trigger.json`。
- 本阶段**未对 born-digital 页运行 OCR**（方案 §10 禁止 force OCR）；触发页的 OCR 效果用 13C-A 已缓存的 O2 输出验证，未新增任何 OCR 调用。

## 10. 硬门（B1–B10）

| arm | 通过 | 可判定 | 未过门 | 待定 |
|---|---|---|---|---|
| `B0_p0_cleaned` | 10/10 | 10 | — | — |
| `B1_mineru_cleaned` | 8/10 | 10 | B2, B4 | — |
| `B2_fused` | 10/10 | 10 | — | — |

### 逐门明细

| 门 | `B0_p0_cleaned` | `B1_mineru_cleaned` | `B2_fused` |
|---|---|---|---|
| **B1** | PASS | PASS | PASS |
| **B2** | PASS | FAIL | PASS |
| **B3** | PASS | PASS | PASS |
| **B4** | PASS | FAIL | PASS |
| **B5** | PASS | PASS | PASS |
| **B6** | PASS | PASS | PASS |
| **B7** | PASS | PASS | PASS |
| **B8** | PASS | PASS | PASS |
| **B9** | PASS | PASS | PASS |
| **B10** | PASS | PASS | PASS |

## 11. 成本与复杂度

| arm | 清洗耗时(s) | 元素数(受控均值) | 沙盒产物(MB) |
|---|---|---|---|
| `B0_p0_cleaned` | 0.116 | 174.9 | 5.824 |
| `B1_mineru_cleaned` | 0.147 | 131.3 | 6.248 |
| `B2_fused` | 2.563 | 183.4 | 10.105 |

## 12. 环境阻塞（不作为能力结论）

| arm | 状态 | 原因 |
|---|---|---|
| `P4_grobid` | **BLOCKED** | 本机无 Java 运行时且无 Docker：GROBID 是 Java 服务，sandbox 内无法满足。 |

## 13. 未决问题与限制

- 受控语料为程序化渲染（PyMuPDF base-14 字体 + 手工排版），不是真实 LaTeX 编译产物；公式指标只比较文本/符号层面与几何检出，不代表真实 LaTeX 语义还原能力。
- 真实语料没有元素级 GT，因此 section / table / figure / equation 族在真实语料上只报告「结构元素是否存在」，其余字段为 `n/a`（未采集 ≠ 为 0）。
- 数学跨度规范化是本 Cleaner 唯一改写公式文本的地方，只做词法层（分组花括号、下标连接符、排版空白），`raw_text` 永远保留；**未做**任何语义重写。
- P1 的 `-m txt` 档在**无文本层的 T12** 上仍返回了文本，说明该档在本机实际启用了内置 OCR 通道。这不影响 born-digital 页上的结论（未观察到 OCR 特征），但 13C-C 之前需要澄清该档位的语义，否则「未启用 OCR」这句话在本机上不成立。
- 未实现段落视觉换行合并（方案 §9.1 允许）：13C-A 实测 P0 的段落碎片率为 0.0000、P1 为 0.0238，而真实损失集中在「类型再标注」与「数学跨度」两处。在没有评测证据支持前不引入这一改写。
