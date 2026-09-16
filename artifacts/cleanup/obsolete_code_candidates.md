# Obsolete Code Candidates（Step 8 — 仅记录，本轮不删除任何源码）

依据清理方案 §7-H / §21 / §22 / §23 生成。删除候选必须同时满足
runtime import = 0、production test import = 0、CLI reference = 0、
config reference = 0、documentation execution reference = 0。

## 结论总览

| 检查项 | 结果 | 判定 |
|---|---|---|
| `src/evidenceflow` 内同名不同语义指标实现并存 | 否。canonical 唯一：`src/evidenceflow/eval/ir_metrics.py`（标准 TREC 语义） | 无冲突 |
| 旧 v1 `metrics.calculate_question_metrics`（recall 实为 HitRate） | 被 `eval/runner.py`、`eval/__init__.py`、`tests/test_eval_metrics.py` 及 Phase11A/11B 冻结 manifest 引用 | **REVIEW（非候选）**：production test import ≠ 0；已由 Phase 11A 命名澄清 docstring 标注语义 |
| `experiments/phase14a/metrics14c.py::hit_rate_at_k` | 命名与语义一致（HitRate 就叫 HitRate），属归档实验代码 | 保留于归档，不构成双实现 |
| runner 血统（`runner_v1/_v2/_fixed/_final` 模式） | 全仓库 0 个匹配；`audit_gold_v2.py`/`stabilize_gold_v3.py` 为正式 Phase（gold audit/stabilization）复现入口，非 bugfix lineage | 无候选 |
| RRF 实现（§22 retired from default） | `src/evidenceflow/retrieval/fusion.py`、`hybrid_fusion.py`、`multilingual.py`、`runtime.py`、`config.py` 及 7 个测试文件引用 | **REVIEW（非候选）**：runtime + test 依赖 ≠ 0；保持 default/fallback/retired 分层，不删 |
| `experiments/debug/*.py`（6 个调试脚本） | 无 src/tests/CLI 引用 | **REVIEW 候选**（第一轮不删；其中 gen_eval/inspect_retrieval 等仍有面试/排查演示价值） |
| `experiments/phase11b_*.py`、`run_phase6..9*.py` 等历史 runner | 每个对应 evals/ 下正式 Phase 产物，属复现入口 | KEEP（归档复现需要） |

## 分层状态确认（§22 Retrieval 路径收口）

- **default**：Dense → Chroma Top20 → Learned Reranker → Top5
- **fallback**：Translation（multilingual.py，默认关闭）、BM25/hybrid（hybrid_fusion.py，conditional）
- **retired from default**：RRF（代码与测试保留，不再位于默认路径）

## 后续动作建议（第二轮，需单独评审）

1. `experiments/debug/` 6 个脚本 → 可移入 `experiments/archived/debug/` 或删除（逐个确认后）。
2. 若未来移除 v1 metrics 兼容层，须先解冻 `eval/runner.py` 契约并同步 Phase11 冻结 manifest 说明。
3. RRF 删除仅在“runtime+test+复现引用全部清零”后可行，当前不满足。
