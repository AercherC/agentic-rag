# Phase 14A-C Final Recommendation

> **重建说明**：14A-C 方案原文在上下文压缩中丢失，本阶段门定义与 arm 细节按压缩摘要 + 14A-B 惯例重建执行（用户确认）。

## 硬门汇总

- DC0: PASS
- DC1: PASS
- DC2: PASS
- DC3: PASS
- XC0: PASS
- XC1: PASS
- XC2: PASS
- XC3: PASS

## SELECTED_FOR_REAL_CORPUS (docx)

- CONTROLLED_WINNER: `DC0`（clear=True）
- 排序准则：Hard Gates → EvidenceAnchorRecall@20 → AllRequired@20 → nDCG@5 → MRR@5 → Recall@10 → 结构保真 → 膨胀/延迟

## SELECTED_FOR_REAL_CORPUS (xlsx)

- CONTROLLED_WINNER: `XC0`（clear=True）
- 排序准则：Hard Gates → EvidenceAnchorRecall@20 → AllRequired@20 → nDCG@5 → MRR@5 → Recall@10 → 结构保真 → 膨胀/延迟

## 遗留

- 真实语料 PENDING（导入须人工 review）；本阶段结论仅基于 27 case 受控语料。
- STOPPED AT Phase 14A-C（未启动后续阶段）。

## 解读与限制（受控语料，必须与结论同读）

- 受控语料为小文档（每 arm 12~47 个 chunk），Top-20 检索接近全语料：Recall@20 与 EvidenceAnchorRecall@20 饱和（≈1.0），区分度主要来自 nDCG@5 / MRR@5 / Recall@10。
- CONTROLLED_WINNER 为最粗粒度 arm（DC0 fixed_token / XC0 whole_sheet）：小语料下粗粒度 chunk 语义完整性占优；该结论在真实语料（长文档、大表、跨节检索）上未必保持，禁止直接外推。
- 受控语料所有表格 ≤8 数据行 → XC1（table_level）与 XC2（row_group_header8）在本语料上 chunk 集合完全一致、指标相同；二者区分度需真实语料大表验证。

