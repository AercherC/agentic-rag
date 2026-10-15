# A4 报告：清理

## 结论：Gate A4 通过

| Gate | 结果 |
|---|---|
| 1. `python -m pytest -q` 全部通过 | **500 passed**（`A4_pytest.log`），算式见 §5 |
| 2. CPU 完整回放 vs B0' | 通过：`replay_runtime_retrieval.py`（不加 `--candidates-only`，约 4 分 49 秒）→ `A4_replay_runtime.json`；`a3_compare_replay.py` → `A4_replay_vs_b0prime.json`：基线 sha256 校验通过，40 题 Top5 **0 差异**，Top20 候选 ID / 分数 **0 差异**，精排分数 **0 差异**，5 项指标相等，parity = true；collection `ef_research_bge_m3_hybrid_c512_v2`，device cpu，backend numpy.exact |
| 3. `verify_active_index.py` 检查 v2 | 通过：7 项检查全 true，count 817，`lifecycle: ACTIVE`（`A4_verify_active_index.txt`）。另跑了 `verify_phase15a.py --output artifacts/stage_a/A4_verify_phase15a.json` → PASS |
| 4. `python -m evidenceflow ask` | 通过：问题 “AdaPoinTr 相比 PoinTr 做了哪些改进？”，`Status: answered`，退出码 0；runtime 实际打开 `ef_research_bge_m3_hybrid_c512_v2`；**LLM HTTP 请求 1 次**（在 `openai` 的 `Completions.create` 处计数；`A4_ask_smoke_stdout.txt` / `_stderr.txt`） |
| 5. 不变量 | `index_state.json` sha256 = `5fbad4f0…8809`，与开工前一致；`git diff -- evals artifacts/phase15a` 为空；Chroma 剩 3 个 collection |

没有触发计划 §7 的 STOP 条件。有一处与主会话复核结果不同的引用（见 §8 第 1 条），我判断不构成 STOP，请确认。

## 1. 开工前复核的事实
- tag `stage-a-start` 存在 ✔；`src/evidenceflow.zip` **不存在**，跳过 ✔；`1.26` 为 0 字节 ✔
- `tmp/`、`data/chroma/`、`*.egg-info/` 在 `.gitignore` 中 ✔；data/chroma 78M、tmp 43M ✔
- Phase 12 的 4 个标注产物都在（移动 tmp 前由备份脚本再次检查）✔
- `index_state.json`：active = v2，staging / previous = null ✔
- 复制前没有任何 python 进程在运行（无进程占用 Chroma）✔

## 2. A4.1 备份
- 备份目录：`D:\Agent_learn\evidenceFlow_backup\stage_a_20261015`
  - `chroma\`：data/chroma 的完整复制（源保留）
  - `tmp\`：仓库 tmp/ 整体移入
  - `1.26`：移入（0 字节）
- 校验（`artifacts/stage_a/A4_backup_manifest.json`，每个文件记录相对路径、大小、sha256）：
  - chroma：源与备份 **33 个文件、81,646,880 字节，逐文件 sha256 完全一致**；复制后再对源生成一次清单，与复制前一致（复制期间源未变化）
  - tmp：移动前后 **122 个文件、43,966,284 字节**，逐文件 sha256 也一致（比要求的文件数 + 总大小更强）
- 删除 `src/evidenceflow.egg-info` 后执行 `pip install -e . --no-deps`（**偏离计划**：多加 `--no-deps`，防止 pip 改动依赖）。前后 `pip freeze` 完全相同；在仓库外目录 `import evidenceflow` 指向 `src/evidenceflow/__init__.py`。pip 重新生成了 `src/evidenceflow.egg-info`（gitignored）。
- 未触碰 `Claude outputs/`。

## 3. A4.2 删除孤儿 collection
- 新增 `scripts/prune_chroma_collections.py`（200 行）：默认 dry-run，`--execute` 才删；`ALLOWED_TO_DELETE`（4 个）与 `FORBIDDEN_TO_DELETE`（3 个）写死，模块级断言二者不相交；**动态保护**：读 `index_state.json` 把 active / previous / staging 指针的 collection 名并入禁止集合，与允许列表有交集或状态文件读不出 → `REFUSED`，退出码 2（在打开 Chroma 之前）；只删实际存在的名字；删除前导出 before，`--execute` 后导出 after；Chroma 出错 → 一行 `FAILED`，退出码 1。
- 删除前 grep（`--include=*.py/*.toml/*.json/*.yaml/*.cfg/*.ini/*.ps1/*.sh`，排除 evals、artifacts、.git）：
  - `…_v2_staging`：仅 `scripts/repair_active_index_metadata.py`
  - `…_refmeta_v1`：仅 `build_research_reference_metadata_collection.py`、`run_research_references_aware_experiment.py`（A4.3 删除）
  - `ef_research_structure_v1` / `_elemaware`：仅 `run_phase13b_controlled_rebuild.py`（A4.3 删除）
  - 另：`scripts/analyze_runtime_reachability.py:410` 按前缀 `ef_research_structure` / 子串 `refmeta` 给 collection 分类，是历史分析工具，不依赖这些 collection 存在，不算引用
  - 非代码：`.workbuddy/memory/2026-09-10.md`（历史笔记）、`docs/agentic_rag/01/02`（审查与计划）
  - 结论：没有需要移出删除列表的名字
- 顺序：先 dry-run（targets = 4 个，would_remain = 3 个，与预期一致）→ `--execute`。

| collection | 删除前 count | 删除后 |
|---|---|---|
| `ef_research_bge_m3_hybrid_c512_v2`（ACTIVE） | 817 | 817 保留 |
| `ef_research_bge_m3_hybrid_c512_v1` | 817 | 817 保留 |
| `ef_research_bge_m3_c512_o64_v1` | 361 | 361 保留 |
| `ef_research_bge_m3_hybrid_c512_v2_staging`（STAGING） | 817 | 已删 |
| `ef_research_bge_m3_c512_o64_refmeta_v1` | 361 | 已删 |
| `ef_research_structure_v1` | 385 | 已删 |
| `ef_research_structure_v1_elemaware` | 935 | 已删 |

（`A4_chroma_before.json` / `A4_chroma_after.json`，含 id 与 metadata。被删 collection 可从备份目录的 `chroma\` 恢复。）
- `scripts/repair_active_index_metadata.py` 第 2–6 行加注释：历史一次性脚本，依赖的 staging collection 已删除，不能再运行。**未删除该文件。**
- 测试 `tests/test_prune_chroma_collections.py`（11 个，全部在临时目录的真实 Chroma 中）：默认 dry-run 不删；`--execute` 只删允许列表（无关 collection 保留）并导出 after；不存在的名字跳过；硬编码禁止名进入允许列表 → 2 且未打开存储；active / previous / staging 指针名进入允许列表 → 2（参数化 3 个）；状态文件缺失 / 损坏 → 2（参数化 2 个）；before JSON 结构；两份硬编码名单不相交。

## 4. A4.3 / A4.4 删除的代码
删除前用 `pytest --collect-only -q` 统计用例数：

| 测试文件 | 用例数 |
|---|---|
| `tests/test_phase13_kb_chunk_audit.py` | 53 |
| `tests/test_phase13b_controlled_rebuild.py` | 30 |
| `tests/test_phase13c_cleaning_benchmark.py` | 17 |
| `tests/test_phase13c_parser_benchmark.py` | 73 |
| `tests/test_eval_runner.py`（A4.4） | 9 |
| 合计 | **182** |

删除（`git rm`，共 102 个被跟踪文件，可从 tag `stage-a-start` 恢复）：`experiments/phase13c/`（含 runners/）、`experiments/phase14a/`、`experiments/archive/`、`experiments/run_phase13_kb_chunk_audit.py`、`run_phase13b_controlled_rebuild.py`、`run_phase13c_cleaning_benchmark.py`、`run_phase13c_parser_benchmark.py`、`build_research_reference_metadata_collection.py`、`run_research_references_aware_experiment.py`、`run_phase15c1_exact_dense_architecture.py`、上表 5 个测试文件、`src/evidenceflow/eval/runner.py`。`experiments/phase13c/` 下残留的未跟踪 `__pycache__` 一并删除（只有 .pyc）。

复核（grep 被删模块名、`evidenceflow.eval.runner`、`EvaluationRunner` 等符号，范围 src/tests/scripts/experiments 中保留的文件）：
- import：除 §8 第 1 条外，没有任何保留文件 import 被删模块。`evidenceflow.eval.runner` 的使用者只有 `eval/__init__.py`、`tests/test_eval_runner.py` 和已删的 `experiments/archive`，与主会话一致；`EvaluationReport`、`MetricSummary`、`PerQuestionResult`、`RetrievedResultSummary` 在其他地方也没有使用。
- `src/evidenceflow/ingestion/structure.py:3-5`：docstring 改为说明来源脚本已在 Stage A A4 删除、可从 tag `stage-a-start` 恢复，代码不变。
- 未修改、只列出的历史清理工具（把 phase13c / phase14a 路径当数据引用，删除后输出可能变化）：`scripts/build_cleanup_inventory.py`、`rebaseline_post_promotion.py`、`verify_cleanup_gates.py`、`audit_project_cleanup.py`、`analyze_runtime_reachability.py`、`cleanup_safe_cache.py`（第 65 行的重建提示指向已删除的 `experiments/phase14a/runners/embed14c.py`）。`tests/test_hygiene_scripts.py:31` 只把 evals 下的 phase14a 路径作为字符串输入，不受影响。
- `experiments/run_phase15c_dense30_promotion.py:353` 提到 `experiments/archive/` 的历史说明文字，保留。
- `src/evidenceflow/eval/__init__.py`：docstring 第 1–4 行；去掉 `from evidenceflow.eval.runner import (...)` 及 `__all__` 中 `EvaluationReport`、`EvaluationRunner`、`MetricSummary`、`PerQuestionResult`、`RetrievedResultSummary`、`save_evaluation_report`。
- COMPATIBILITY_ONLY 说明（模块 docstring 第 3 行）：`eval/metrics.py`、`eval/dataset.py`、`domain/evaluation.py`，措辞“COMPATIBILITY_ONLY：legacy HitRate 语义，新代码用 eval/ir_metrics.py。”

**A3 报告 §7 中不跟随 active 指针的脚本，随本次删除消失的有**：`run_phase13_kb_chunk_audit.py`、`run_phase13b_controlled_rebuild.py`、`build_research_reference_metadata_collection.py`、`run_research_references_aware_experiment.py`、`run_phase15c1_exact_dense_architecture.py`，4 个 runner（`phase13c/runners/run_phase13c_c.py`、`run_phase13e_hybrid.py`、**`run_phase13f_runtime_promotion.py`——会在生产 `data/chroma` 中先删除再重建 v1**、`phase14a/runners/embed14c.py`），以及 `experiments/archive/*`。仍保留、仍读 `Settings().collection_name` 的：`run_phase10_reranker_performance.py`、`run_phase12_rag_quality.py`、`run_phase15b1_dense_candidate_depth.py`、`run_phase15c_dense30_promotion.py`、`run_production_reranked_baseline_replay.py`、`audit_gold_v2.py`、`experiments/stage_a/a1_crosscheck_batch_embedding.py`、`scripts/verify_active_index.py`、`scripts/verify_phase15a.py`；写死历史名的：`prepare_research_corpus.py`、`scripts/repair_active_index_metadata.py`（已标注不可运行）、`scripts/analyze_runtime_reachability.py`；以及 `experiments/debug/*`。

## 5. 测试数算式
`N0 (534) + A1 到 A3.2 新增 (137) − 被删测试文件 (182) + A4 新增 (11) = 500`
- N0 = 534：`A0_report.md`
- A1 到 A3.2 新增 137 = A1 +4（538，`A1_report.md`）+ A2a +53（591，`A2a_report.md`）+ A2a.1/A2b +18（609，`A2_report.md`）+ A3 +34（643）+ A3.1 +23（666）+ A3.2 +5（671）（后三项见 `A3_report.md`）
- 被删 182：§4 表格（删除前 `--collect-only` 实测）
- A4 新增 11：`tests/test_prune_chroma_collections.py`
- 校验：671 + 11 − 182 = 500，与实测一致。

## 6. A4.5 过期注释和字段（不改变行为）
- `src/evidenceflow/storage/base.py:47-48`：`iter_all_chunks` docstring 去掉 BM25Index / §17.3.1，改为当前用途（生命周期校验、Exact 索引构建前的全量读取）。
- `src/evidenceflow/storage/chroma.py:462`：实现类同一方法的 docstring 有相同的 BM25Index 说法，一并改掉（**计划未列，属于同类小改**）。
- `src/evidenceflow/retrieval/reranker.py:1`：模块 docstring “实验可用的…”改为线上链路用途；原第 323 行日志 extra 中的 `"fallback_branch_used": False` 删除（tests 中没有断言这个字段）。
- `PROJECT_SPEC.md:1616-1617`：logging 清单去掉 `fallback_branch_used`，注明已在 Stage A A4 移除。
- `src/evidenceflow/domain/chunk.py:76,80`：`candidate_rank`、`similarity_score` 的 description 去掉 RRF 说法。
- `.workbuddy/memory/MEMORY.md`（不在 git 中）：第 47、106 行“项目无 .git”改为已有 .git；第 64 行 “C512/O64” 改为 hybrid（C512、零重叠），注明 C512/O64 fixed 已退为回滚通道；末尾追加 “Stage A 状态” 一节（环境 agent-gpu、Exact Dense20 + v3 链路、B0' 路径、A4 清理与备份位置、旧测试基线已过期）。只修正和追加，没有删除内容；保持原 LF 换行。

## 7. 修改或创建的文件
| 文件 | 职责 / 改动 |
|---|---|
| `scripts/prune_chroma_collections.py` | 新增：受保护的孤儿 collection 清理 |
| `tests/test_prune_chroma_collections.py` | 新增：11 个用例 |
| `scripts/repair_active_index_metadata.py` | 文件头注释：历史脚本，不能再运行 |
| `src/evidenceflow/eval/__init__.py` | 去掉 runner 导出 |
| `src/evidenceflow/eval/metrics.py`、`eval/dataset.py`、`domain/evaluation.py` | COMPATIBILITY_ONLY 说明 |
| `src/evidenceflow/ingestion/structure.py` | 来源脚本已删除的说明 |
| `src/evidenceflow/storage/base.py`、`storage/chroma.py` | `iter_all_chunks` docstring |
| `src/evidenceflow/retrieval/reranker.py` | 模块 docstring；删除 `fallback_branch_used` |
| `src/evidenceflow/domain/chunk.py` | description 去掉 RRF |
| `PROJECT_SPEC.md` | logging 清单 |
| 删除 102 个文件 | 见 §4 |
| `artifacts/stage_a/A4_*` | 备份清单、Chroma 前后快照、pytest 日志、回放与比较、verify 输出、ask 冒烟输出、本报告 |

## 8. 偏离与需要你确认的地方
1. **主会话复核遗漏了一处动态 import。** `scripts/repair_active_index_metadata.py:263-264`（加注释后的行号）的 `compute_ir_parity` 在函数内执行 `sys.path.insert(0, ROOT/"experiments"/"phase13c")` 后 `import metrics13cc`，而 `experiments/phase13c/` 已在 A4.3 删除。这个脚本本来就因 staging collection 被删而不能再运行，而且主会话已决定保留并标注它；没有任何测试或其他代码 import 它。所以我没有 STOP，继续删除 phase13c，并在该文件头注释中写明这个依赖也已删除、可从 tag `stage-a-start` 恢复。如果你认为应当按“被删对象仍被引用”处理，可以从 tag 恢复 `experiments/phase13c/metrics13cc.py`，或经你同意后删除该脚本。
2. **`verify_active_index.py` 并非完全只读**：它默认把结果写到被跟踪的 `artifacts/cleanup/active_index_smoke.json`。本次运行改动了该文件（`expected_count` 由 null 变 817、metadata 键顺序变化），我已用 `git checkout` 还原，结果另存为 `A4_verify_active_index.txt`。以后运行应传 `--output`。（`verify_phase15a.py` 也默认写 `artifacts/phase15a/phase15a_report.json`，我传了 `--output` 到 stage_a，`artifacts/phase15a/` 无改动。）
3. `pip install -e . --no-deps`：比计划多加 `--no-deps`（按指令）。
4. `storage/chroma.py:462` 的 docstring 也改了（计划只列 base.py）。
5. tmp 中的 `tmp/pdfs`（MEMORY 记为“不可重建须保留”）已随 tmp 移到备份目录，不在仓库内原位置。目前没有代码依赖它（全量测试与回放都通过）。
6. 完整回放耗时约 4 分 49 秒（A3 为 3 分 27 秒），结果逐题一致；差异来自机器负载，与正确性无关。

## 9. 尚未解决的问题或限制
- 上面列出的历史清理工具和仍读 `Settings().collection_name` 的实验脚本未改（按指令），以后 activate 到新 collection 后它们不会跟随 active 指针。
- A3.1 的待决项（多 collection 同标 ACTIVE、`index_state.json` 被 git 跟踪等）仍留给 A5。
- 按指令没有进入 A5。
