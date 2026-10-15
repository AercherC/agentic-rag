# Stage A 总结：基础修复（A0–A5）

阶段 A 没有增加任何 Agent 能力。完成了五件事：线上检索改为确定性的 Exact Dense；生成改为结构化输出并加入 `RAGAnswer.status`；线上索引跟随 lifecycle active 指针，ACTIVE 不可直接修改；清理废弃代码和孤儿数据；PROJECT_SPEC 精简为 v0.27。
起点 tag 为 `stage-a-start`，终点 tag 为 `stage-a-done`（本地）。

## 1. Gate 结果

| 步骤 | 结果 | 要点 | pytest |
|---|---|---|---|
| A0 冻结起点 | 通过 | 索引身份通过；HNSW D20 回放与冻结 B0 40 题 Top5 一致；修复了 1 个依赖设备的测试 | 534（N0） |
| A1 Exact Dense | 通过 | 10 个独立进程 Top20 ID 与分数 100% 一致；两次完整回放 Top5 一致；与 HNSW B0 相比 5 项指标差为 0；后端固定 numpy.exact | 538 |
| A2 结构化生成 v3 | 通过 | 探测结果：json_schema 400，选定 json_object；25 题冒烟 GenerationContractError 0、修复 0、Citation 错误 0、LLMRequestError 0 | 591（A2a）→ 609 |
| A3 lifecycle 接入线上 | 通过（含 A3.1、A3.2 修复） | ask 读取 v2；ingest 到 active 返回 2；staging 全量重建 817 chunk，ID 集合与 active 一致后丢弃；回放与 B0' 一致 | 643 → 666 → 671 |
| A4 清理 | 通过 | 删除 4 个孤儿 collection（剩 3 个）、102 个文件、5 个测试文件（182 个用例）；备份校验通过；回放与 B0' 一致；ask 正常 | 500 |
| A5 SPEC 精简 | 通过（AGENTS.md 一项待决） | SPEC v0.27，376 行，核对表全部一致；退休概念 grep 通过；两处脚本小修 | 507 |

## 2. 新基线 B0'

Exact Dense20 → bge-reranker-base → Top5，CPU，40 题（eligible 38）：`artifacts/stage_a/A1_exact_baseline.json`
sha256 `422fd629b22f26caf9452e3598bf805d01df9b9cf24e6d42b5a6890e40a1411f`（`A1_exact_baseline.sha256`）

| P@5 | R@5 | MRR@5 | MAP@5 | nDCG@5 |
|---|---|---|---|---|
| 0.2684 | 0.5103 | 0.5175 | 0.3631 | 0.4903 |

A3、A4 两次走线上 `build_retriever` 的回放与 B0' 逐题一致：Top5、Top20 候选 ID 与分数、精排分数全部相同。GPU 回放有个别题的 Top5 不同（A1_gpu_reference），所以数值 Gate 只认 CPU。

## 3. A2 生成冒烟（deepseek-flash，json_object，thinking disabled）

- 25 题：answered 19，insufficient_context 6。
- 修复 **0 次**（attempted 0 / succeeded 0）。GenerationContractError 0，CitationValidationError 0，LLMRequestError 0，限流耗尽 0，thinking_ignored_calls 0。
- token 用量：prompt **47,376**，completion **2,708**，共 25 次调用，每次都有 usage。总耗时 150.5 s，单题中位数 6.09 s。
- 探测：9 次请求，json_schema 3/3 返回 400，json_object 3/3 成功，prompt_only 3/3 成功。
- 与 Phase 12 v2（Qwen3.5-35B-A3B，3/25 格式错误）**不是同一模型**，只能参考。

## 4. 遗留问题

1. 激活后旧 active 的 metadata 仍为 `lifecycle=ACTIVE`，rollback 依赖这一点，所以可能同时有多个 collection 标为 ACTIVE。
2. `artifacts/phase15a/index_state.json` 被 git 跟踪。真正执行 activate/rollback 后，不要 `git restore` 或 `checkout` 它；有未提交的指针变更时不要切换分支。
3. 被 `mark_collection_lifecycle` 改写过的 collection 不再有 `hnsw:space`。以后改写 v2 时，需要同步 `verify_phase15a.py` 的快照比对。
4. `--discard-staging` 在没有登记 staging 时返回 2。
5. `data/chroma` 下有 5 个不再被引用的 HNSW 段目录（Chroma 1.5.9 的 `delete_collection` 不删除它们），**由用户手动删除**，删除前要备份并确认没有进程在用 Chroma：
   `5d5d025e-d9b1-4199-a0ae-bee85794fd95`、`928e9c57-83df-4191-b11e-36151d59da7e`、`edd92cf6-6f2c-4a51-9594-906705b0b628`、`f32f608b-ad2c-447a-8e7c-3cb87d2e28c7`、`09024bdd-6ae1-4f44-a8b8-065e719eded7`。方法见 SPEC §6.6。
6. `chroma.sqlite3` 不会因删除 collection 而缩小；是否 VACUUM 未决，本阶段不做。
7. `--activate` 只在单元测试中验证过，没有在真实数据上运行；`mark_collection_lifecycle` 与 `activate` 不是原子操作。
8. 部分保留的实验脚本仍读 `Settings().collection_name`，不会跟随 active 指针（A4 报告 §4）。`verify_phase15a.py` 默认写被跟踪的 `artifacts/phase15a/phase15a_report.json`，运行时须传 `--output`。
9. AGENTS.md 中没有“文档优先级”一节（A0 快照前已被删除），计划 A5 第 5 步因此未执行，等待用户决定（A5 报告第三部分 §5）。
10. 你本地 `.env` 中显式写着 `COLLECTION_NAME`（A3 报告 §8）。将来 activate/rollback 到其他 collection 后，`ask` 会报 `Index configuration error`，建议删掉这一行。

## 5. 给阶段 B 的输入

- **基线**：B0' = `artifacts/stage_a/A1_exact_baseline.json`（sha256 见上）。线上回放用 `experiments/stage_a/replay_runtime_retrieval.py`（走 `build_retriever`，CPU），比较用 `experiments/stage_a/a3_compare_replay.py`。离线实验的 query 必须逐条 `embed_query`。
- **生成冒烟的发现**：
  - `ARR-040` 返回 insufficient_context，原因是检索漏召回：3 个可接受证据 chunk 都不在 Top5 中，不是生成问题。
  - “status=insufficient_context 但正文含带引用的部分回答”如何计分，需要阶段 B 决定。
  - `eval/quality.py` 的 `abstention_metrics` 只接受大写状态，与 `AnswerStatus` 的小写不一致。
  - 同一模型下 v2 与 v3 的格式可靠性对比尚未做。
  - 冒烟原始回复保存在 `A2_generation_smoke.jsonl`，可以用 `reparse_smoke_offline.py` 离线重新解析。
- **合同**：以 `PROJECT_SPEC.md` v0.27 为准；阶段 B 的新合同先写进 SPEC，再改代码。

## 6. §6 交付物核对

| 交付物 | 状态 |
|---|---|
| `A0_report.md` … `A5_report.md` | 都在。A2 另有 `A2a_report.md`；A3.1、A3.2 写在 `A3_report.md` 中 |
| `A1_exact_baseline.json` 及 sha256 | 在（`A1_exact_baseline.sha256`） |
| `A1_determinism.json`、`A1_exact_vs_hnsw.json` | 在 |
| `A2_probe.json`、`A2_generation_smoke.json` | 在 |
| `A4_chroma_before.json` | 在（另有 `A4_chroma_after.json`） |
| `STAGE_A_SUMMARY.md` | 本文件 |

没有缺失项。
