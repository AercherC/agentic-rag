# Phase 14 多格式（DOCX/XLSX）Ingestion 历史

> 本文档是 Office 分支的历史归档入口。EvidenceFlow 当前 runtime 规范
> （PROJECT_SPEC.md）不包含 Office 内容——Office ingestion 从未进入 production runtime。

## 结论速览

| 阶段 | 日期 | 结论 |
|---|---|---|
| 14A-A adapter 基准 | 2026-09-14 | PASS：DOCX W-A0 8/8、XLSX X-A0 10/10；Docling/Unstructured/Mammoth 未安装 → BLOCKED ≠ FAIL |
| 14A-B cleaning 基准 | 2026-09-14 | PASS：SELECTED_FOR_14A_C = DOCX-B1 / XLSX-B1；27 case × 4 arm 全过硬门 |
| 14A-C chunk 基准 | 2026-09-15 | PASS：8 arm 全过门；CONTROLLED_WINNER = DC0/XC0（最粗粒度，小语料下粗 chunk 占优，禁止外推真实语料） |

## 归档状态（2026-09-15 收口）

- `STATUS = COMPLETE_CONTROLLED_EXPERIMENT`
- `REAL_CORPUS = INTENTIONALLY_DEFERRED_OUT_OF_SCOPE`
- `RUNTIME_PROMOTED = FALSE`

Real-world Office validation was intentionally deferred because Office ingestion
is outside the current scientific-PDF runtime scope.

## 固化的工程经验

1. 跨文档稳定 key 是 `(document_id, element_id)` 二元组，不是 element_id 单值。
2. None ≠ 0 ≠ False（未采集 / 为 0 / 非 0 三态必须区分）。
3. BLOCKED ≠ FAIL：环境阻塞不是能力缺陷。
4. count metric ≠ probability metric，比例指标分子分母必须同计数空间。
5. formula + cached value 双轨：公式单元格禁止用 cached value 覆盖公式语义。
6. provenance 不可伪造（FAKE_PROVENANCE = 0 硬门）。
7. Source Evidence Anchor → per-arm qrels projection → 标准 IR 的评测框架可跨格式复用。

## 产物位置

- 基准产物：`evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/`
- 基准源码：`experiments/phase14a/`（含 [ARCHIVED.md](../../experiments/phase14a/ARCHIVED.md)）
- 归档分类证据：`artifacts/cleanup/cleanup_inventory.json`
- 已删除的可重建索引清单：`artifacts/cleanup/chroma_collection_inventory.json`
