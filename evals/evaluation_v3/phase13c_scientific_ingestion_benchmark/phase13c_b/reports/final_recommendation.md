# Phase 13C-B — 最终建议

- **生成时间**：2026-09-14T12:40:40.031079+00:00
- **PHASE_13C_B_STATUS**：`PASS`
- **SELECTED_FOR_13C_C**：`B2_fused`
- **STOPPED_AT**：Phase 13C-B

## 1. 结论

B2_fused 在硬门全过的前提下同时取得最强正文（0.940461）与最强结构（0.885714）。

## 2. 逐 arm 判定

| arm | 硬门 | anchor | 正文覆盖 | 装饰可分 | 引用隔离 | 结构可用 | 融合冲突 |
|---|---|---|---|---|---|---|---|
| `B0_p0_cleaned` | 10/10 | 44/44 | 0.9167 | 0.0000 | 1.0000 | n/a | n/a |
| `B1_mineru_cleaned` | 8/10 | 44/44 | 0.9962 | 0.0021 | 1.0000 | n/a | n/a |
| `B2_fused` | 10/10 | 44/44 | 0.9405 | 0.0000 | 1.0000 | n/a | 0.1 |

## 3. 回答方案 §1 的六个问题

**Q1（能否用 P0 保正文、MinerU 补结构而不产生内容冲突）**
- 可以，但有边界。B2 的对齐率受控均值 n/a、真实均值 n/a；未对齐的部分**不是**靠猜测补齐，而是分为「P0-only」（正文保留）与「P1-only」（只补结构）；
- 内容冲突全部显式记录在 `conflicts[]`，唯一一类未决冲突是参考文献条目对齐（`REFERENCE_ALIGNMENT_UNCERTAIN`），处理方式是**不合并**。

**Q2（正文保持与装饰可分性是否已彻底拆开）**
- 是。5A 只看装饰（识别 / 隔离 / 可追溯），5B 只看正文（锚点无新增丢失）；
- 量化证据：全部 arm 的 `body_matched_to_furniture_rate` = 0（没有任何 arm 把正文当装饰），而 13C-A 的旧门 5 在这同一批数据上仍然 fail —— 这就证明了旧口径测的不是装饰能力。

**Q3（MinerU 失去的 anchor 属于哪一类）**
- 结论：**结构重组导致的 mapping 失败 + OCR/normalization 的表示约定差异**，**不是真丢失**（见 `cleaning_comparison.md` §1 与 `metrics/anchor_attribution.json`）；
- 证据：`seedformer.pdf` 的第 8 页在 P1 输出里含 `$N _ { p } = 1 2 8$`，内容在、形态不同；同时 MinerU 原生 `content_list` 里被 13C-A 映射层丢弃的六种类型（`page_number`/`header`/`footer`/`aside_text`/`list(ref_text)`/`chart`）是本工程映射层的缺口，不是解析器缺陷。

**Q4（References 如何独立）**
- 用「最后一个参考文献标题 + 条目标记兜底」划区，区内正文型元素转为 `reference_entry`，含 ≥2 个条目标记的段落按标记**切分**（这是修复「条目粘连」的关键）；
- 参考文献**永不删除**，只把 `retrieval_role` 标成 `reference`，因此检索正文不会再被reference-heavy 模板污染，同时引用完整性有据可查。

**Q5（表格 / 公式 / 图注如何进入统一 schema 并保留全部 provenance）**
- 每类都有规范表示：表 `{caption, headers, rows, raw_structure, serialized_markdown, page, bbox, section_path}`；图 `{figure_id, caption, page, bbox, section_path, nearby_paragraph_ids}`（**不做 VLM 理解**）；公式双表示 `{raw_text, normalized_latex, equation_number}` 且都不允许覆盖对方；
- 所有元素保留 `page` / `bbox` / `source_parser` / `source_ref`；融合后的元素另有`source_parsers` / `source_refs` / `fusion_status` / `alignment`。

**Q6（哪个 arm 最适合做 13C-C 的输入）**
- 见 `selected_cleaning_pipeline.json`：B2_fused。

## 4. 沙盒与边界

| 项 | 状态 |
|---|---|
| source PDFs modified | NO |
| src/evidenceflow modified | NO |
| 13B/13A 冻结产物 modified | NO |
| baseline Chroma modified | NO |
| 项目 venv 新增依赖 | NO（本阶段只使用 PyMuPDF 与标准库） |
| Conditional OCR 误触发 | 0 页 |
| 可复现（T02 digest） | {"B0_p0_cleaned": true, "B1_mineru_cleaned": true, "B2_fused": true} |

## 5. 下一步（不在本阶段执行）

- 13C-C — Chunk Benchmark：输入冻结为选定的 ScientificDocument，比较 C0 C512/O64、C1 Section-aware Hybrid、C2 Contextual Element-aware、C3 Parent/Child optional；
- `P4_GROBID = BLOCKED_ENVIRONMENT` 保留在 manifest；若未来 Java/Docker 可用，可作为 scholarly metadata / reference sidecar 追加独立补测；
- 生成层指标（Claim / Citation / Groundedness）留待切分定案后再评。
