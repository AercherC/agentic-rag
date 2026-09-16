# Phase 14A-B Final Recommendation

- **PHASE_14A_B_STATUS**: `PASS`
- **STOPPED_AT**: Phase 14A-B
- **Sandbox boundary**: PASS（src 树 identical、冻结工程文件未变、14A-A 冻结 YES、语料哈希未变）
- **Corpus self-check**: docx 12/12, xlsx 15/15（解析器无关 OOXML 自证）
- **Real corpus**: DOCX REAL_CORPUS_PENDING / XLSX REAL_CORPUS_PENDING —— 本阶段结论仅基于受控语料

## 结论
- DOCX: B0 PASS / B1 PASS；SELECTED_FOR_14A_C = DOCX-B1
- XLSX: B0 PASS / B1 PASS；SELECTED_FOR_14A_C = XLSX-B1
- 选型理由：硬门全过前提下，按方案 §22 顺序（Source Fidelity → Semantic Structure → Provenance → Explainability）优先 semantic cleaner（B1）：其保留 B0 全部事实，并额外提供 heading 推断留痕 / list 归组 / 表格序列化 / 区域推断与表头层级。

## 硬门汇总
- 12/12 cases pass D-G1..G8 (DOCX-B1)
- 15/15 cases pass X-G1..G10 (XLSX-B1)
- FAKE_PROVENANCE = 0；FORMULA_SILENT_OVERWRITE = 0；MERGED_RANGE_LOSS = 0；HIDDEN_CONTENT_SILENT_DROP = 0

## 14A-C 前置条件
- 仅当本阶段 PASS 后才开始 Chunk Benchmark（DOCX: fixed-token vs heading-aware vs element-aware；XLSX: whole-sheet vs table-level vs row-group+repeated header）。
- 遗留：真实语料待用户提供（导入后须走 §18 人工 review 清单）；
- 本阶段未安装任何新依赖，未触碰 Docling/Unstructured/Mammoth（作为架构参考，方案 §23）。