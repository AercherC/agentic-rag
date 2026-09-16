# ARCHIVED — Phase 14A Multiformat (DOCX/XLSX) Ingestion Benchmark

> 状态：**ARCHIVED_CONTROLLED_EXPERIMENT**（2026-09-15 收口）
> 本目录为逻辑归档，源码未物理移动；`src/evidenceflow` 对本目录的依赖为 0（见
> `artifacts/cleanup/runtime_reachability.json`）。

## Purpose

Validate DOCX/XLSX ingestion, cleaning and chunking concepts on a controlled
corpus, using the frozen PDF pipeline (13C-B B2_fused) as the reference adapter.

## Result

- **Phase 14A-A: PASS** — DOCX W-A0 8/8、XLSX X-A0 10/10；Docling/Unstructured/Mammoth
  未安装 → BLOCKED ≠ FAIL。
- **Phase 14A-B: PASS** — SELECTED_FOR_14A_C = DOCX-B1 / XLSX-B1；
  27 case × 4 arm 全过 D-G1..8 / X-G1..10。
- **Phase 14A-C: PASS** — 8 arm（DC0..3 / XC0..3）全过 C-G1..8 / D-CG1..4 / X-CG1..5；
  CONTROLLED_WINNER = DC0 / XC0（最粗粒度，小语料下粗 chunk 占优，**禁止外推真实语料**）。

## Why archived

EvidenceFlow runtime is focused on scientific PDF literature.
No real Office corpus exists in the target use case.

## Production status

**NOT PROMOTED.** Office adapters 不进入 `src/evidenceflow`，不参与默认 Runtime 契约
（Dense → Chroma Top20 → Learned Reranker → Top5）。

## Real corpus

INTENTIONALLY_DEFERRED_OUT_OF_SCOPE — Real-world Office validation was
intentionally deferred because Office ingestion is outside the current
scientific-PDF runtime scope.（详见 `phase14a_b/metrics/real_corpus_status.json`）

## Minimal retention set（长期保留）

- 本 README（ARCHIVED.md）
- `phase14a_a_status.json` / `phase14a_b/phase14a_b_status.json` / `phase14a_c/phase14a_c_status.json`
- `phase14a_c/reports/final_recommendation.md` 与各 final report
- `ground_truth/controlled_corpus_gold.json`、`controlled_corpus/`（可重建语料的冻结输入）
- `manifests/`、`metrics/`、`gates/`
- `experiments/phase14a/**`（corpus generator、benchmark code、mutation tests）

## Removed as reproducible（已删除，可重建）

- `phase14a_c/chroma/` — 8 个 benchmark 向量索引（ef_14ac_docx_dc0..3、ef_14ac_xlsx_xc0..3），
  2026-09-15 删除；collection 名称/记录数/config 已导出至
  `artifacts/cleanup/chroma_collection_inventory.json`；可由 `chunks/*.jsonl` +
  `embeddings/*.npz` 经 `experiments/phase14a/runners/embed14c.py` 重建。
- embedding cache 保留未删：896KB 且被 `manifests/manifest.json` hash 锚定（frozen）。

## Reusable findings（工程价值已固化）

- OOXML 原生结构优先：python-docx/openpyxl 直接读结构，优于转文本。
- formula + cached value 双轨：公式单元格禁止用 cached value 覆盖公式语义。
- provenance 契约：来源证据不可伪造（FAKE_PROVENANCE = 0 硬门）。
- Source Evidence Anchor → per-arm qrels projection → 标准 IR 的评测框架。
- 跨文档稳定 key 是 `(document_id, element_id)` 二元组，不是 element_id 单值。
- 评测语义陷阱：None ≠ 0 ≠ False；BLOCKED ≠ FAIL；count metric ≠ probability metric；
  比例指标分子分母必须同计数空间。

## Where to look

| 内容 | 路径 |
|---|---|
| 归档分类与依赖证据 | `artifacts/cleanup/cleanup_inventory.json` |
| 清理计划（dry-run） | `artifacts/cleanup/cleanup_plan.json` |
| Chroma collection 清单 | `artifacts/cleanup/chroma_collection_inventory.json` |
| 语义目录映射 | `artifacts/cleanup/phase_directory_mapping.json` |
