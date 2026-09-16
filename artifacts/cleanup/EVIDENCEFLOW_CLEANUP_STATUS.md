# EVIDENCEFLOW_CLEANUP_STATUS

```
EVIDENCEFLOW_CLEANUP_STATUS:
PASS

MODE:
NON_DESTRUCTIVE_SCOPE_CONSOLIDATION

SOURCE_RUNTIME_MODIFIED:
NO            （Gate1：src/evidenceflow 全部 64 个 .py sha256 与基线逐一相同）

PROJECT_SCOPE:
SCIENTIFIC_PDF_RAG

FILES_BEFORE:
1221 files / 228,291,360 bytes（pre_cleanup_manifest.json 全库 sha256 冻结）

FILES_AFTER:
1018 files / 209,946,040 bytes（不含 artifacts/cleanup 自身；
其中 ~104 个为本次验证运行的 pytest 临时文件 tmp/pytest-root/*，
扣除后净文件数 914 ≈ 1221 - 307 删除 + 7 新增归档/文档/脚本）

BYTES_RECLAIMED:
20,302,708 bytes 直接删除收益
= cache 1,438,164 + phase14 chroma 18,864,544
净账面差异 18,345,320 bytes（含新增 pytest 临时与日志）

PHASE14_STATUS:
ARCHIVED_CONTROLLED_EXPERIMENT
STATUS = COMPLETE_CONTROLLED_EXPERIMENT
REAL_CORPUS = INTENTIONALLY_DEFERRED_OUT_OF_SCOPE
RUNTIME_PROMOTED = FALSE
（三个 status JSON 增量写入 ARCHIVE_CLOSEOUT 块，历史字段未改写；
 real_corpus_status.json 由 PENDING 更名，不再用 PENDING 表述未完成产品功能）

PHASE14_RUNTIME_REFERENCES:
0（src import = 0 / production tests import = 0 / CLI import = 0（无 CLI entrypoint）
  / config reference = 0 / docs execution reference = 0）
证据：artifacts/cleanup/runtime_reachability.json（AST 扫描）

PHASE14_CHROMA_REMOVED:
8 collections（ef_14ac_docx_dc0..3: 12/32/38/41 chunks；
ef_14ac_xlsx_xc0..3: 17/24/24/47 chunks），161 files / 18,864,544 bytes。
删除前已导出 collection inventory（名称/记录数/config 元数据/重建入口）：
artifacts/cleanup/chroma_collection_inventory.json
可由 chunks/*.jsonl + embeddings/*.npz 经 runners/embed14c.py 重建。

PHASE14_EMBEDDING_CACHE_REMOVED:
0（保留：896KB 且被 phase14a_c/manifests/manifest.json hash 锚定，属冻结产物；
方案 §16 的长期保留条件已满足——metrics/query rankings/manifest 均在）

DUPLICATE_REPORTS_REMOVED:
2（tmp/phase12_regression_out/baseline.json 12,958 B +
tmp/phase12_regression_out/chroma_identity.json 561 B，
均与 evals/evaluation_v3/phase11c_technical_debt_closure_v1/ 内 canonical 逐字节相同，
且不在 phase11c tmp_retention.json 保留清单）
REVIEW 组：phase14a _a/_b 对（invariance 双跑证据）、13C-B __B2_fused 对（Gate5 保护）、
phase9/10 rollback_validation（目录自包含）、chroma 内部 bin —— 全部 KEEP。

OBSOLETE_CODE_CANDIDATES:
0 deleted / 3 REVIEW（见 obsolete_code_candidates.md）：
1) experiments/debug/*.py 6 个调试脚本（无引用，可第二轮归档）
2) v1 metrics 兼容层（production test + 冻结 manifest 引用 ≠ 0，不可删）
3) RRF（runtime + 7 个测试文件引用 ≠ 0，保持 retired-from-default 分层）
runner 血统扫描：无 *_v1/_v2/_fixed/_final lineage；指标 canonical 实现唯一。

ROOT_TESTS:
591 passed / 0 failed（17 warnings；exit=0）
注：首轮未设 HF_HUB_OFFLINE=1 等环境变量导致 1F+5E（bge-m3 加载走网络被 502 拦截），
按项目环境约定设置后复跑全绿；沙箱删除守卫吞汇总行时按
pytest-in-workbuddy-sandbox skill 用 PYTEST_DEBUG_TEMPROOT 指向工作区取得真实汇总。

STANDARD_IR_SMOKE:
PASS（ir_metric_conformance_fixture.json 存在且由全量回归数值校验；
standard_ir_v1/execution_result.json 黄金结果可读：
P@5=0.1150 R@5=0.5583 R@10=0.7333 MRR@5=0.4321 MAP@5=0.4279 nDCG@5=0.4619）

FROZEN_ARTIFACT_HASH:
PASS（583 文件逐字节一致，0 意外差异；
保护集：standard_ir_v1=16、phase12_rag_quality_v1=24、phase13c_b=57 全部 intact；
预期变更仅 4 个 Phase14 状态收口文件 + 2 个 tmp 去重副本）

DEFAULT_RUNTIME:
Dense -> Chroma Top20 -> Learned Reranker -> Top5（未动，src hash 证明）

NEXT_ACTIVE_WORK:
PDF ingestion/chunk benchmark
（B2_fused ScientificDocument -> section-aware / element-aware chunking 对比，
固定 BGE-M3、Dense Top20、标准 IR；复核 ARR-007 改善与 ARR-037/ARR-040 Top20 miss）

STOPPED_AT:
PROJECT_CLEANUP
```

## 本轮产出物清单

| 文件 | 职责 |
|---|---|
| `scripts/build_cleanup_inventory.py` | Step 0-4 盘点/分类/dry-run（只读） |
| `scripts/cleanup_safe_cache.py` | Step 5 缓存清理 + chroma inventory 导出 |
| `scripts/verify_cleanup_gates.py` | Step 9 Gate 1/3/5 验证 |
| `experiments/phase14a/ARCHIVED.md` | Phase14 归档 README（新增） |
| `docs/history/phase14_multiformat.md` | Office 历史收口（新增） |
| `docs/evaluation/history.md` | Phase 历史移入（新增） |
| `README.md` | 重写为能力层叙事（§18） |
| `artifacts/cleanup/*` | 全部清理证据（11 个文件） |

## 未做 / 后续（第二轮，需单独指示）

- 物理移动 Phase14/Phase13 目录（映射已备好：phase_directory_mapping.json）
- experiments/debug/ 归档或删除
- v1 metrics 兼容层与 RRF 的移除评审
- tmp/ 中 Phase12/13 中间文件（TEMP_REVIEW 类）的去留决策
- 项目无 .git，方案 §31 的 commit 拆分不适用；如需版本管理请先 git init
