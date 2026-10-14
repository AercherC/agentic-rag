# A1 报告：线上检索切换为 Exact Dense（numpy.exact）

## 结论：Gate A1 通过

| Gate | 结果 |
|---|---|
| 10 个独立进程的 Top20 候选（ID 与分数）100% 一致 | 通过：40 题全部逐元素一致，设备 cpu，后端 numpy.exact（`A1_determinism.json`） |
| 两次独立完整回放 Top5 100% 一致 | 通过：40/40 一致，reranker 分数也逐位一致 |
| 与 HNSW B0 相比 R@5、nDCG@5 下降 ≤ 0.01 | 通过：**5 项指标差全部为 0，40 题 Top5 全部一致**（`A1_exact_vs_hnsw.json`，`compare_replay` 的身份校验也为 true） |
| pytest 全部通过 | 通过：**538 passed**（534 + 新增 4） |

**新基线 B0'**（Exact Dense20 → Reranker → Top5，CPU）：`artifacts/stage_a/A1_exact_baseline.json`
sha256 `422fd629b22f26caf9452e3598bf805d01df9b9cf24e6d42b5a6890e40a1411f`（同目录 `A1_exact_baseline.sha256`）

| P@5 | R@5 | MRR@5 | MAP@5 | nDCG@5 |
|---|---|---|---|---|
| 0.2684 | 0.5103 | 0.5175 | 0.3631 | 0.4903 |

## 1. 修改或创建的文件
| 文件 | 职责 |
|---|---|
| `PROJECT_SPEC.md` | 顶部新增 §S.1：后端固定 numpy.exact（修订 §42）、启动校验、B0' 记录 |
| `src/evidenceflow/retrieval/exact_dense.py` | 删除 faiss 导入与 IndexFlatIP 分支；`backend` 恒为 `"numpy.exact"` |
| `src/evidenceflow/runtime.py` | 新增 `build_retriever(settings)`（Exact 索引装配 + 启动校验）；`build_rag_service` 调用它 |
| `tests/test_exact_dense.py` | 新增用例：环境存在 faiss 时也不使用，backend 固定（原文件无 faiss 相关用例） |
| `tests/test_runtime.py` | 更新 Fake；断言 DenseRetriever 收到的是 Exact 索引；新增大小/维度不一致拒绝启动、校验通过 3 个用例 |
| `experiments/stage_a/replay_runtime_retrieval.py` | 走 `build_retriever` 的线上路径回放；`--candidates-only`、`--allow-gpu`；非 cpu 且未传 `--allow-gpu` 时退出码 2 |
| `experiments/stage_a/a1_analyze.py` | 确定性、15C.1 交叉校验、Exact-vs-HNSW 对比 |
| `experiments/stage_a/a1_crosscheck_batch_embedding.py` | 解释 15C.1 差异的验证脚本 |
| `artifacts/stage_a/A1_*` | 候选 10 次、完整回放 2 次、基线、对比、GPU 参考、pytest 日志 |

## 2. 关于 faiss 的依赖排查（改代码前用 grep 完成）
- `src/`：只有 `exact_dense.py` 导入 faiss，已删除。`pyproject.toml` 未声明 faiss。
- `tests/`：没有任何 faiss 相关用例（所以没有需要“更新”的旧用例，仅新增了固定后端的用例）。
- `experiments/run_phase15c1_exact_dense_architecture.py`：
  - 第 243、837、849 行用 `index.backend == "faiss.IndexFlatIP"` 计算 `faiss_numpy_conformance` / Gate G1。**该脚本现在若重跑，这个条件恒为 False，会触发 STOP。**
  - 第 164 行的注释和第 850 行的 stop_reason 提到 FAISS。
  - 该脚本的冻结产物（`evals/evaluation_v3/phase15c1_...`）已存在且未改动；其记录的 `search_backend` 本来就是 `numpy.exact`。我没有修改这个脚本，按计划它属于保留文件；如需让它仍可重跑，需要你另行决定。
- `experiments/stage_a/a0_compare_replay.py`：只用 `import faiss` 探测环境，保留不影响。
- `PROJECT_SPEC.md` §42 第 2765–2766 行、§43 第 2790 行含 FAISS 表述；§S.1 已声明修订，A5 合并时统一改写。
- `docs/`、`artifacts/environment/…/report.md` 中有历史性提及，不改。

## 3. 交叉校验：与 15C.1 的 `exact20.json`（不作为 Gate，有差异）
- **有差异**：ARR-001 的 Top20 中有一个 chunk 不同；其余 39 题 ID 相同，但 439/780 个分数有 1 ULP 级差异（最大绝对差 2.4e-7）。
- **原因已验证**：15C.1 用 `embed_documents` 把 40 个 query 批量编码，而线上用逐条 `embed_query`，批量填充导致 float32 末位不同。我用批量编码在 CPU 上重算 Exact Top20，与 `exact20.json` **0 题不一致**，所以 Exact 实现本身与 15C.1 等价。
- 这些差异没有传导到最终结果：Exact20 的 5 项 @5 指标与 15C.1 的 E20、HNSW B0 完全相同（最大差 0.0）。
- 含义：以后任何离线实验如果要和线上基线逐位对齐，query 必须逐条 `embed_query`，不能批量。

## 4. 计时（CPU）
- 完整 `build_retriever`（含 BGE-M3 与 reranker 加载）约 6.0 s。
- Exact 索引本身（读 Chroma 向量 + 构建矩阵）约 0.35 s。
- 单次 Exact dense 搜索：平均 5.6 ms，最大 39 ms（不含 query embedding 和 rerank）。
- 完整 40 题回放约 207 s（含 rerank，是主要耗时）。

## 5. GPU 参考（可选，不作 Gate）
`A1_gpu_reference.json`（`summary` 见 `A1_gpu_reference.summary.json`）：cuda:0 完整回放用时 31 s（CPU 207 s），5 项指标与 CPU 相同；但 **ARR-001 候选与 ARR-023 的 Top5 与 CPU 不同**。这印证了计划的要求：数值 Gate 只能在 CPU 上跑。

## 6. 产物保护与范围
- 本步没有运行 verify 脚本；`git status` 显示没有已跟踪产物被覆盖；`evals/` 无改动。
- 未新增依赖，未新增 backend 开关，K 保持 20，没有做 D30。

## 7. 限制与待确认
1. 上面第 2 节的 15C.1 脚本在 faiss 条件上已与代码不一致，是否需要更新由你决定。
2. 回放脚本复用了 `compare_replay`，但它要求 pipeline 字符串相同，我在比较时仅覆盖了这一个字段（其余身份、逐题 Top5、指标均原样比较），已在 `a1_analyze.py` 注释中说明。
3. 基线是 CPU 回放；GPU 线上运行可能出现个别题的 Top5 差异（见第 5 节）。
