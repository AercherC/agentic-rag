# EvidenceFlow Cleanup Inventory（Step 3 分类清单）

生成时间：2026-09-15T05:44:52.512549+00:00  
文件总数：1221

| 分类 | 文件数 | 字节 | 动作分布 |
|---|---:|---:|---|
| REVIEW | 73 | 113,901,723 | REVIEW=73 |
| SOURCE_OF_TRUTH | 70 | 29,556,475 | KEEP=70 |
| CORE_EVAL | 179 | 25,178,703 | KEEP=179 |
| GENERATED_REPRODUCIBLE | 310 | 21,429,746 | DELETE_CANDIDATE=310 |
| PDF_ACTIVE_RESEARCH | 115 | 18,285,591 | KEEP=115 |
| TEMP_REVIEW | 54 | 14,423,328 | REVIEW=54 |
| MULTIFORMAT_ARCHIVE | 293 | 4,667,278 | ARCHIVE=293 |
| CORE_TEST | 51 | 394,584 | KEEP=51 |
| CORE_RUNTIME | 71 | 352,356 | KEEP=71 |
| DOCS | 5 | 101,576 | KEEP=5 |

## 分类判据摘要

- CORE_RUNTIME：src/evidenceflow/**、pyproject、.env*、根规范文档 —— KEEP，本轮零改动
- CORE_TEST：tests/** —— KEEP
- CORE_EVAL：standard_ir_v1、evaluation_v2/v3 历史正式评测、research_* —— KEEP + FREEZE HASH
- PDF_ACTIVE_RESEARCH：evals phase13*、experiments/phase13c —— KEEP
- MULTIFORMAT_ARCHIVE：evals phase14a_*、experiments/phase14a —— 逻辑归档（本轮不物理移动）
- GENERATED_REPRODUCIBLE：__pycache__/.pytest_cache/*.pyc/Phase14 chroma —— DELETE_CANDIDATE
- SOURCE_OF_TRUTH：tmp/pdfs（不可重建真实语料）、data/source_docs、data/chroma baseline —— KEEP
- TEMP_REVIEW：tmp/ 下 Phase12/13 中间文件 —— 本轮 REVIEW 不删
