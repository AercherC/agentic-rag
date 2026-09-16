# Phase 13 — Knowledge Base Structure & Chunk Quality Audit（第一轮审计报告）

- **阶段**：Phase 13A（Audit only）
- **范围**：Step 0–8（**已按计划止于 Step 8，未进入 Step 9–19 的受控重建**）
- **产物目录**：`evals/evaluation_v3/phase13_kb_chunk_audit_v1/`
- **执行入口**：`experiments/run_phase13_kb_chunk_audit.py`
- **日期**：2026-09-10

> 本报告按计划 §31 的返回模板组织。所有数字均可在同目录的 JSON 产物中逐项复算。

---

## 0. 一句话结论

**三题的证据丢失层 100% 落在 Dense 候选召回（dense rank 25 / 77 / 43，均未进 Top20），parser 抽取了全部必需事实且与 qrels 一致 —— 因此「PDF 解析 / chunk 边界」不是可证的根因。**

同时，知识库确实存在**大规模、可复现的结构性表示缺陷**（表格线性化 10/10 篇、0/361 chunk 保留行结构、章节 metadata 覆盖率 0、28.25% 检索单元吃进页面装饰），但**这些缺陷与本次三题丢失的因果关系未被证明**。

按计划 §13 的 OR 判据：**`PROCEED_TO_CONTROLLED_REBUILD = YES`（经由判据 2）**，候选方案 `SECTION_AWARE_MINIMAL`。

---

## 1. 冻结基线与 §4 停止条件

| 项 | 值 |
|---|---|
| Collection | `ef_research_bge_m3_c512_o64_v1` |
| Chunk 数 | **361**（复算值与线上 Chroma 逐一对账一致） |
| `sorted_chunk_id_identity_hash` | `d81f9bc529b008e8d0cad65a19f33330c81291458341d69362bbc5c7bf31b0e0` |
| 审计后身份 hash（finalize 复验） | 同上，**未变** |
| Chunk 配置 | C512 / O64 / `BAAI/bge-m3` |
| 标准 IR 指标 | **未重算，但已回读校验**：Step 0 读取 `evals/evaluation_v3/standard_ir_v1/resume_metrics.json` 的 `current_runtime`，逐项比对冻结值（容差 5e-7）→ 6/6 `matches = true`；并断言 `Recall@20` 仍为 `null`（N/A）。冻结值：Precision@5=0.115, Recall@5=0.5583, Recall@10=0.7333, MRR@5=0.432083, MAP@5=0.427917, nDCG@5=0.461851 |

**§4 停止条件触发情况：0 次。**

- PDF↔抽取文本对应性：`extraction_8gram_coverage` 在全部 35 张抽查页上 **最小值 = 1.0**（阈值 0.95）。
- `verify_identity()` 在 Step 0 与 finalize 各执行一次，均通过；361 chunk 身份、10 篇 PDF hash、5 个 ingestion 源文件 hash 全部记录在 `baseline.json`。
- Runtime / 基线 Collection / canonical Gold / qrels / PROJECT_SPEC **全部未修改**。
- finalize 时重算 11 个产物的 sha256 写入 `run_manifest.json`，并逐项核对与磁盘内容一致。

---

## 2. Chunks audited（Step 1–2）

| 指标 | 值 |
|---|---|
| 审计 chunk 数 | **361**（10 篇 PDF / 145 页） |
| token 分布 | min 129 / median 512 / max 512 / mean **506.16** |
| near-max chunk 率（≥0.9×512） | **0.975069**（352/361） |
| very-short chunk 率（<64） | 0 |
| 跨页 chunk 率 | **0.390582**（141/361） |
| `starts_mid_sentence` | **0.479224** |
| `ends_mid_sentence` | **0.919668** |
| `starts_mid_word` | 0.174515 |
| **保留行结构的 chunk** | **0 / 361** |
| 含 figure caption 且与正文混排 | 0.315789（114） |
| 含 table caption 且与正文混排 | 0.180055（65） |
| 只承载 figure caption 的 chunk | **0** |
| 只承载 table caption 的 chunk | **0** |
| 跨越章节标题的 chunk | 114（0.315789） |
| 跨 ≥2 个标题的 chunk | 0.163435 |
| references 边界 chunk | 4 |
| 高置信页面装饰污染 chunk | **102（0.282548）** |

### 2.1 结论性事实：**行结构在检索单元里彻底消失**

`chunker` 用 `"\n\n"` 作为人工分隔符，但 BGE-M3 的 SentencePiece `decode()` **会把换行折叠成空格**，且 `encode("\n\n") == []`（Chunker 实际上从不产生边界 token）。审计脚本用断言守护这一事实，并实测：**361/361 个 chunk 的存储文本不含任何换行**。

直接后果：章节标题、caption、表格行、参考文献条目的**行级边界在检索单元中全部丢失**。抽取文本本身是有换行的（35/35 抽查页 `extracted_text_has_newlines = True`）——**丢行发生在 decode，不在 PDF 解析**。

### 2.2 两处原有指标的测量缺陷已被更正（本次审计的附带发现）

| 指标 | 旧值 | 更正后 | 缺陷原因 |
|---|---|---|---|
| header/footer 污染率 | 0.0 | **0.282548（高置信口径）** | `furniture_hit()` 按**行**匹配，而 chunk 文本已无换行 → 该判定**结构性恒为 False**。旧值不是"知识库干净"的证据。 |
| 表格结构丢失篇数 | 5/10（纯数值行启发式） | **10/10（逐页看图确认）** | 启发式要求"≥4 token 且 ≥3 个纯数值"，在 `Table 6: … (Pred → GT / GT → Pred)` 这类双数值单元格上会漏检。 |

更正后的口径（`chunk_quality_metrics.json → document_furniture.definition_exact_span`）：

- 判定方式：**chunk 字符区间 ∩ 页级装饰字符区间 ≠ ∅**（不再依赖 chunk 文本的行结构）。
- 污染类别：`PAGE_NUMBER` 96 chunk、`ARXIV_SIDEBAR` 8、`COPYRIGHT` 2、`OPEN_ACCESS_WATERMARK` 3（去重后 102 chunk）。
- **页码识别自证**：对每篇 PDF 校验页码候选序列的单调性 —— **10/10 篇 `inversions = 0`，偏移量恒定**（如 `adapointr` 偏移 0、`pfnet` 偏移 7661、`proxyformer` 偏移 9465、`svdformer` 偏移 14507）。这证明识别出的是真页码而非表格数值。
- 跨页重复串口径另计（82 chunk），但**精度低**：实测同时命中跨页重复的参考文献 venue 串与表格行标签，已在产物中显式标注 `HEURISTIC`，不作为结论使用。

---

## 3. Layout metrics（Step 3）

**方法**：每篇 PDF 抽取代表页（首页 / 方法页 / 表格最密页）**并强制并入三题 Gold 证据所在页**，共 **35 页**；渲染成 PNG（`tmp/phase13_pages/`），逐页对照「渲染图 / 抽取文本 / 落在该页的 chunk」，并用 PyMuPDF block 级坐标（y0 + 栏分类）复核抽取顺序。

| 指标 | 值 |
|---|---|
| 抽查页数 / 覆盖文档 | **35 / 10** |
| 抽查页渲染 + 人工判定 | 100% |
| `reading_order_error_rate` | **0.0** |
| `COLUMN_MERGE_ERROR` | **0** |
| `HEADER_FOOTER_ERROR` | **27 页** |
| `TABLE_FLATTEN_ERROR` | **10 页** |
| `CAPTION_PLACEMENT_ERROR` | 0 |
| `PAGE_BREAK_ERROR` | 0 |

### 3.1 关键发现：**阅读顺序不是问题**

机器侧的 `reading_order_upward_jumps` / `column_back_jumps` 在 29/35 页上报警，但**人工看图全部证伪**：

- "向上跳行"来自「左栏到底 → 右栏顶部」的正常换栏；
- "RIGHT→LEFT 回跳"来自**旋转的 arXiv 侧栏**（被排到页末）以及**并排表格的逐行读取**（如 `seedformer` p12 的 Table 5/6/7/8 两两并排）；
- 唯一真实的顺序错位出现在**复杂示意图内部的子标签**（`hyper_pcn` 图 2、`proxyformer` 图 2），属图内标注本身不构成线性文本。

block 级复核示例（`pointr` p1）：左栏读完（摘要 → Introduction 起始 → 脚注）→ 右栏（图标签 → Figure 1 caption → 正文续写），与人类阅读顺序一致。

> 这是本次审计**推翻假设**的地方：最容易先入为主认定的"双栏被压平导致乱序"并不成立。真正的缺陷在别处。

### 3.2 页面装饰确实进入检索单元

27/35 页的抽取文本里含页面装饰，且经字符区间相交确认**真的落进 chunk**。典型证据：

- `adapointr` 第 0 号 chunk 的文本**以页号 `1` 开头**；
- `adapointr` 第 3 号 chunk 含旋转侧栏 `arXiv:2301.04545v1 [cs.CV] 11 Jan 2023`；
- `svdformer` / `pfnet` / `proxyformer` 第 1 页含 CVF 开放获取水印三行；
- 印刷页号（`7662` / `9466` / `14508` 等）随 chunk 进入知识库。

---

## 4. Tables（Step 4）

| 指标 | 值 |
|---|---|
| 抽查页确认的表格线性化页数 | **10 页 / 10 篇（每篇 1 页）** |
| 承接表格的 caption 总数 | 87 |
| ├ `STRUCTURE_LOST` | **38** |
| └ `MIXED_WITH_BODY` | 49 |
| 纯数值行启发式检出文档 | 5/10（**下界**，见 §2.2） |
| `table_isolation_rate` | **0.0** |

**肉眼确认的硬证据**（可直接回看 PNG）：

- `pcn` p17：**整页是 Table 4/5/6/7**，10 列 × 7 行纯数值；
- `pfnet` p6：每个单元格是 `Pred→GT / GT→Pred` **两个数值**，线性化后两个数值的归属无法区分；
- `pointr` p6 / `svdformer` p7：**两级分组表头**（`34 seen categories` / `21 unseen categories` 跨列分组），线性化后分组语义不可恢复；
- `proxyformer` p8：Table 7 含**跨行合并单元格**。

---

## 5. Figures（Step 4）

| 指标 | 值 |
|---|---|
| figure caption 总数 | 133 |
| ├ `MIXED_WITH_BODY` | **133（100%）** |
| └ 与正文隔离 | **0** |
| `figure_caption_isolation_rate` | **0.0** |
| caption 原本独占一行的比例 | 0.706767 |
| figure caption 保留率（能被 chunk 承载） | **1.0** |

caption **文本**全部被抽取且能被检索（`figure_caption_preservation_rate = 1.0`），但**没有任何一个 caption 被隔离成独立检索单元**，必然与正文同块。图内像素信息（曲线、结构示意）不进文本流。

---

## 6. Formula（Step 4 追加发现）

| 指标 | 值 |
|---|---|
| LaTeX 命令泄漏片段总数 | **219** |
| 受影响文档 | **4/10**（`hyper_pcn` 96 / `proxyformer` 57 / `seedformer` 50 / `svdformer` 16） |
| 受影响 chunk | 25（0.069252） |

部分 PDF 的公式文本层**直接保存 LaTeX 源**，PyMuPDF 原样取出，Cleaner 不做识别，于是检索单元里出现如 `\ma th cal {G} = (V, E, W), (1)`、`\beg i n { ca ses} 1, & \text {if vertex } v \in e` 这样的源码片段。

**重要约束**：泄漏集中在第 3–8 页的公式区，经核对**不落在** ARR-007 / ARR-037 / ARR-040 的 Gold chunk 上（三题 Gold 证据都在第 1–5 页的 caption 或正文区）。**因此不能用来解释这三题的丢失。**

---

## 7. Section metadata（Step 5）

| 指标 | 值 |
|---|---|
| `section_attribution_rate`（审计期离线推断） | **0.969529** |
| `subsection_attribution_rate` | 0.296399 |
| 有标题上下文的 chunk | 350 |
| 跨越章节边界的 chunk | 114 |
| **知识库中携带 section metadata 的 chunk** | **0 / 361** |
| `structure_metadata_present` | **False** |

Chroma metadata 的字段集为 `document_id / content_hash / file_name / chunk_index / token_count / page_start / page_end`（+ `domain_metadata_json`），**完全没有** `section` / `subsection` / `chunk_type` / `source_element_type` / `parent_section`。

需要强调：`section_attribution_rate = 0.9695` 衡量的是**审计脚本在离线旁路重建时能否推断出章节**，**不代表知识库里存了这个信息**。两者必须分开陈述，否则会得出"知识库有章节结构"的错误印象。

---

## 8. 三个 Retrieval Failure 的逐题根因（Step 6）

| 题号 | 类型 | 证据类型 | Query | dense rank（各 Gold） | 可证丢失层 | root_cause |
|---|---|---|---|---|---|---|
| **ARR-007** | method | figure_caption | What two outputs are shown in the PCN architecture? | **25**（未进 Top20） | `DENSE_CANDIDATE_RECALL` | `DENSE_RETRIEVAL_MISS` |
| **ARR-037** | multi_document | multi_document | Compare how PoinTr and AdaPoinTr construct or adapt transformer queries. | **17**（进）/ **77**（未进） | `DENSE_CANDIDATE_RECALL` | `DENSE_RETRIEVAL_MISS` + `MULTI_EVIDENCE_FRAGMENTATION` |
| **ARR-040** | multi_document | multi_document | 比较 SVDFormer 与 Hyper-PCN 如何建模点云结构关系。 | **6**（进）/ **43**（未进） | `DENSE_CANDIDATE_RECALL` | `DENSE_RETRIEVAL_MISS` + `MULTI_EVIDENCE_FRAGMENTATION` |

**`evidence_loss_layer_counts = {'DENSE_CANDIDATE_RECALL': 3}`** —— 三题**全部**、且**只有**这一层。

三题共同的已证事实（`all_facts_present_in_pdf = all_anchors_parsed = all_anchors_chunked = gold_chunks_match_qrels = True`）：

1. 必需事实**全部**存在于 PDF；
2. **全部** Gold anchor 被 parser 抽取，并按 `(page, anchor, occurrence)` 精确定位；
3. **全部** anchor 落在至少一个 chunk 内（ARR-007: chunk[4]；ARR-037: chunk[3], chunk[12]；ARR-040: chunk[0,1], chunk[0]）；
4. 承载 chunk 与 qrels 完全一致。

**逐题细节**

- **ARR-007**：Gold anchor `Figure 2: PCN architecture`（`pcn` p3）。看图确认：该 caption 独占整行、被完整抽出、顺序正确、落在 chunk[4]（pages 2–3，512 token 满块）。**dense rank = 25 → 未进 Top20 → Reranker 根本没机会看到它。**
- **ARR-037**：G1（`pointr`）dense rank **17**（进 Top20），G2（`adapointr`）dense rank **77**。两篇论文的候选**未能同时进入 Top20**。
- **ARR-040**：中文 query。G1（`svdformer`，chunk[0,1] 重叠归属）dense rank **6**；G2（`hyper_pcn`，chunk[0]）dense rank **43**。同样无法同时命中。

**结构条件（仅标注，不等于因果）**

三题的 `structure_causality` 全部为 **`PLAUSIBLE_NOT_PROVEN`**。落在 Gold chunk 上的结构条件：

- ARR-007：`BAD_CHUNK_BOUNDARY`（跨 `3 Problem Statement` / `4 Point Completion Network`）、`FIGURE_INFORMATION_LOSS`；
- ARR-037：`BAD_CHUNK_BOUNDARY`（跨 `2 Related Work`）、`FIGURE_INFORMATION_LOSS`；
- ARR-040：`BAD_CHUNK_BOUNDARY`（跨 `1 Introduction`、`Abstract / 1 Introduction`）、`SECTION_METADATA_LOSS`。

**纪律声明**：这些结构缺陷确实落在 Gold chunk 上，但**本次丢失发生在它们之前**（Dense 召回阶段）。审计只能证明"缺陷存在"，**不能**证明"缺陷导致了本次丢失"。要判定因果，必须在第二轮的受控 A/B 中检验。

---

## 9. Gold chunk quality（Step 7）

| 指标 | 值 |
|---|---|
| Gold chunk 数（含重复计入） | 45 |
| 去重后 distinct Gold chunk | 38 |
| `GOOD` | 5 |
| `PROBLEMATIC` | **40** |
| `problems_per_gold_chunk` | **0.888889** |

§12 六个问题的逐项答案（`answer_counts`）：

| 问题 | true | false |
|---|---|---|
| self_contained（能独立回答） | **0** | **45** |
| has_heading_context | 35 | 10 |
| too_much_irrelevant_content | 17 | 28 |
| crosses_section | **29** | 16 |
| table_or_caption_polluted | **20** | 25 |
| reference_polluted | 4 | 41 |
| required_claim_fragmented | 9 | 36 |

`flag_counts`：`CROSSES_SECTION_BOUNDARY` 29、`CAPTION_POLLUTION` 20、`NO_HEADING_CONTEXT` 10、`MULTI_GROUP_QUESTION` 9、`REFERENCE_POLLUTION` 4。

**关键观察**：**0/45 个 Gold chunk 是自包含的**。即便检索命中，交给生成层的仍是"被切开的半句 + 混排的 caption + 无章节标识"的文本。这解释了 Phase 12 中"漏 > 错"的保守画像。

---

## 10. PRIMARY DIAGNOSIS

> **本题集的三个 Retrieval Failure，其可证证据丢失层 100% 位于 Dense 候选召回，而不是 PDF 解析、阅读顺序、章节边界、表格/图处理、chunk 边界或 metadata 丢失中的任何一项。**
>
> PDF 解析侧的表现实际上**很好**：35 张抽查页的 8-gram 覆盖率全部为 1.0，全部必需事实被抽取并按页精确定位，caption 保留率 1.0。
>
> **但是**，知识库作为"检索单元"的**结构性表示**是差的，且有四条互相独立的可复现证据：表格行列关系线性化（10/10 篇）、行结构归零（0/361）、章节 metadata 覆盖率 0（0/361）、页面装饰进入检索单元（102/361）。**约 89% 的 Gold chunk 被判为 PROBLEMATIC**，且 0/45 自包含。
>
> 因此正确的表述是：**结构缺陷是"系统性质量负债"（已证），但它是不是这三题的具体失因（未证）。** 两者必须分开陈述 —— 把它们合并成"所以按章节重建就能修好"正是计划 §32 明令禁止的推理。

---

## 11. PROCEED_TO_CONTROLLED_REBUILD

### **YES**（经由判据 2）

| 判据 | 结果 |
|---|---|
| **判据 1**：3 题中 ≥2 题有**可证**的 PDF/chunk/structure 根因 | **不成立**（0 题；三题可证根因全部是 `DENSE_RETRIEVAL_MISS`） |
| **判据 2**：结构问题达到明显且可复现的规模 | **成立**（6 条独立理由） |

判据 2 的具体理由：

1. 跨越章节标题的 chunk 占比 **0.3158 ≥ 0.15**；
2. **0 个** chunk 只承载 figure caption —— caption 从不被隔离；
3. **0/361** chunk 保留行边界；
4. 逐页看图确认 **10/10** 篇 PDF 存在多列数值表被线性化（`adapointr:p11, hyper_pcn:p11, pcn:p17, pfnet:p6, pointattn:p7, pointr:p6, proxyformer:p8, seedformer:p12, snowflakenet:p8, svdformer:p7`）；
5. 知识库 metadata **完全不含** section / subsection / chunk_type / parent_section；
6. **102/361** chunk 吃进页级装饰（率 0.2825）。

> 判据 1 与判据 2 是 **OR** 关系，判据 1 不成立也可进入第二轮。

### recommended candidate：`SECTION_AWARE_MINIMAL`

选择理由：缺陷的核心是"**检索单元里没有任何结构标识**"（无章节、无行、无 chunk_type、无 caption 隔离）。`SECTION_AWARE_MINIMAL` 用最小变量同时检验多个假设：章节感知切分 + caption 隔离 + 页面装饰剔除。它是唯一能一次性覆盖判据 2 第 1/2/5/6 条的最小改动。

### ⚠️ 第二轮必须同时检验两个方向相反的假设

因为三题的可证丢失层全部是 Dense 候选召回，第二轮必须**预设两种可能结局**：

- **若 section-aware 候选能恢复这 3 题** → 结构确实是主因（计划 §27-A）；
- **若结构更干净但 3 题仍不恢复** → 按计划 §27-C 转向 sparse / lexical candidate augmentation（例：ARR-037/040 是跨论文多证据题，G2 的 dense rank 77/43 更像"词汇不匹配"而非"结构损失"）。

**特别提示**：ARR-037 与 ARR-040 的失败模式是"一篇进 Top20、另一篇不进"。这更像是**跨文献证据召回**问题，section-aware 切分对它的帮助可能有限。

---

## 12. 边界声明

| 项 | 状态 |
|---|---|
| Runtime 代码 | **未修改**（`src/evidenceflow/**` 无改动） |
| 基线 Collection `ef_research_bge_m3_c512_o64_v1` | **未修改 / 未重建**（361 chunk 身份 hash 复验一致） |
| canonical Gold / qrels / gold_mapping_v3 / 冻结 40Q 排名 | **未修改** |
| PROJECT_SPEC.md | **未修改**（属 `runtime_source_fingerprint` 采集范围，避免破坏 Phase 12 冻结指纹；与 Phase 12 同一处理） |
| 新增第三方依赖 | **无** |
| Chroma 重建 | **未发生** |
| Step 9+ | **未进入**（`candidate_*.json` 全部标记 `NOT_RUN`） |

---

## 13. 产物清单

| 文件 | 内容 |
|---|---|
| `baseline.json` | Step 0 冻结基线：361 身份 hash、10 PDF hash、5 源文件 hash、冻结指标 |
| `chunk_inventory.json` | Step 1：361 chunk 逐条 inventory（section / 边界 / 元素标记 / covered_headings） |
| `chunk_quality_metrics.json` | Step 2：分布 / 边界 / 结构污染 / 页面装饰（含更正后的高置信口径与页码单调性自证） |
| `layout_review_scaffold.json` | Step 3 工作台：35 页渲染路径 + 机器版面信号 + 待填判定 |
| `layout_review_annotations.json` | Step 3 **人工看图结论**（35 页，逐页可追溯，含 `reviewer` 与判定语义） |
| `pdf_layout_audit.json` | Step 3 合并产物：多层 verdict 计数 + 机器证据汇总 + §4 对应性校验 |
| `visual_structural_elements_audit.json` | Step 4：954 个结构元素逐项判定 + 表格线性化证据 + LaTeX 泄漏报告 |
| `section_hierarchy_audit.json` | Step 5：章节归属率 vs store metadata 覆盖率（两者严格区分） |
| `retrieval_failure_root_cause.json` | Step 6：三题逐条证据链 + 丢失层 + 结构条件 + 因果标记 |
| `gold_chunk_quality.json` | Step 7：45 个 Gold chunk 的 §12 六问答案与三级判定 |
| `audit_decision.json` | Step 8：判据 1/2 结果、规模证据、PROCEED 结论、guardrails |
| `run_manifest.json` | 执行步骤、11 个产物 sha256、冻结复验、`candidate_*` = NOT_RUN 理由 |
| `execution_result.json` | 结论摘要 + headline metrics + 三题发现 + 决策指针 |
| `progress_checkpoint.json` | 阶段检查点 |
| `anomaly_report.md` | §4 异常轨迹（含一次**误报**的原文与解除记录，见 §1 说明） |
| `layout_review_annotations.json` / `layout_review_scaffold.json` | Step 3 人工判定与渲染工作台 |

**关于 `anomaly_report.md`**：审计过程中 `finalize` 首版因脚本自身取值错误（读取了不存在的键
`matches`，实际键为 `local_recompute_matches_store`）抛出过一次 §4 异常。异常 `detail` 本身
即显示 `chunk_count = 361`、`local_recompute_matches_store = true` 且 hash 与冻结值一致 ——
**误报，非真实异常**。已修正并加严校验，原文与解除记录一并保留以形成完整审计轨迹。
**本阶段 §4 停止条款真实触发次数 = 0。**

渲染图：`tmp/phase13_pages/*.png`（35 张，dpi=130）

---

## 14. Tests

| 项 | 值 |
|---|---|
| 新增测试文件 | `tests/test_phase13_kb_chunk_audit.py` |
| 新增测试数 | **53**（全部通过） |
| 全量套件 | **471 passed / 17 warnings / 0 failed**（Phase 11C 后为 418；17 warnings 为既有 chromadb 弃用告警，与基线一致） |
| 运行命令 | `python -m pytest -q` |

测试覆盖三类内容：

1. **纯函数**：章节标题识别（含 4 条排除规则：一级编号去重、具名标题唯一性、引用特征、数学符号、尾标点、主编号 >20）、章节归属（`resolve_section` 的 subsection 语义与首屏无归属）、结构元素抽取（caption 形态 + 行首属性 + 精确字符区间 + 公式编号行）、LaTeX 泄漏识别、句切分、分布分位点、切词/切句/句末中断启发式、文本块栏分类。
2. **页面装饰**：首/末非空行的页码识别、正文中间孤立数字**不得**被当成页码、页码序列单调性自证、纯数字重复串排除、跨页重复页眉识别、`span_intersects_furniture` 的 **document_id 防误匹配**、作者型页眉被引用过滤漏检（显式记录为已知取舍）。
3. **产物一致性（只读）**：13 个必需产物存在、基线身份冻结与 finalize 复验一致、guardrails 声明无改动、`candidate_*` 全部 NOT_RUN 且无 candidate 文件、§4 对应性未越界、35 页人工 verdict 全覆盖且都在允许集合内、阅读顺序判定全 OK、表格线性化覆盖 10/10、装饰指标更正的旧口径恒 0 与新口径 >0、0/361 保留行结构、caption 从不隔离、store 章节 metadata 为 0 且离线归属率 >0.9、三题丢失层全部是 DENSE_CANDIDATE_RECALL 且事实全在 PDF 内、结构条件永不声称因果、判据 1 不成立/判据 2 成立 → PROCEED=YES、Gold 六问全部作答、LaTeX 泄漏被量化但不落在三题 Gold 上、manifest 记录 11 个产物 sha256。

**未解决的限制**

1. Step 3 的"看图"由 agent 在渲染 PNG 上完成（非人类专家）。已通过 `layout_review_annotations.json` 的 `reviewer` 字段与逐页 `note` 显式声明，并附机器信号供交叉复核。
2. `REPEATED_FURNITURE` 口径双向偏差（过召回 venue 串、欠召回作者页眉），已在产物中标注为启发式，未用于结论。
3. 结构缺陷与三题丢失的因果关系**未证明**，只能由第二轮受控 A/B 判定。
4. `section_attribution_rate` 是审计期旁路重建结果，**不等于**知识库拥有该 metadata。
5. 表格线性化的篇数口径已明确区分：**看图确认 10/10**（结论来源）、纯数值行启发式 5/10（下界）。
