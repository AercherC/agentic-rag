# Stage B · B2 报告：RunTrace（只观测，不改行为）

起点 `a91fd82`。先补 SPEC §12.5，再改代码。

## 1. 修改/创建的文件
| 文件 | 作用 |
|---|---|
| `PROJECT_SPEC.md` | §12.5 补充：组件 recorder 默认 None、`RunTraceRecorder` 语义（start_run / finish / 复用 / 未 start 时丢弃）、计时规则、各事件记录者、`run.failed.stage` 取值、`build_retriever` / `build_rag_service` 签名与“调用方注入组件不自动挂 recorder”、CLI `--trace` 细节；dense / rerank 列表字段名定为 `candidates` / `results`（每项带 `rank`）；§3 表增 `RunTrace`/`TraceEvent` 行；§7 `build_retriever` 签名；版本行注明 B2 |
| `src/evidenceflow/domain/trace.py`（新） | `RunTrace`、`TraceEvent`（Pydantic，`extra="forbid"`），字段按 §12.5 |
| `src/evidenceflow/domain/__init__.py` | 导出 `RunTrace`、`TraceEvent` |
| `src/evidenceflow/tracing.py`（新） | `TraceRecorder` 协议、`NullTraceRecorder`、`RunTraceRecorder`（可注入单调时钟与 UTC 时钟）、`safe_record` |
| `src/evidenceflow/retrieval/reranker.py` | `DenseRerankerRetriever(..., trace_recorder=None)`：记 `retrieval.dense.started/completed`、`rerank.started/completed`；公开属性 `trace_recorder` |
| `src/evidenceflow/generation/generator.py` | `Generator(llm_client, *, trace_recorder=None)`：记 `generation.started/completed`（确定性拒答也记，`llm_called=false`）；usage 只保留 `prompt_tokens`、`completion_tokens` |
| `src/evidenceflow/rag/service.py` | `RAGService(..., *, trace_recorder=None)`：记 `run.*`、`context.*`、`citation_validation.completed`；阶段 try/except → `run.failed` 后原样 `raise` |
| `src/evidenceflow/runtime.py` | `build_retriever(settings, *, trace_recorder=None)`；`build_rag_service(..., trace_recorder=None)` 传给自己装配的 retriever / generator / RAGService |
| `src/evidenceflow/__main__.py` | `ask --trace PATH`；`finally` 中 `finish()` 并写 JSON；失败只在 stderr 警告一行 |
| `tests/test_tracing.py`（新） | 25 个测试（见 §4） |
| `tests/test_architecture_boundaries.py` | 新增：eval 之外的正式源码不得 import `evidenceflow.eval`（显式确认 `tracing.py`、`domain/trace.py` 在扫描范围内） |
| `tests/test_runtime.py` | 适配签名：FakeGenerator 接受 `trace_recorder`，装配断言含 `trace_recorder: None` |
| `experiments/stage_b/b2_trace_parity.py`（新） | 40 题开/关 trace 检索对照 + 与 A1 基线逐位比对 |
| `artifacts/stage_b/B2_replay_runtime.json`、`B2_replay_vs_b0prime.json`、`B2_trace_parity.json`、`B2_trace_sample.json` | Gate 产物 |

## 2. 实现要点与理由
- **recorder 为 None 时不构造 payload**：所有带 payload 的事件都在 `if recorder is not None` 内构造（列表推导、sha256、
  `parse_citations` 重算、`last_diagnostics` 拷贝），无 payload 的事件经 `safe_record` 在 None 时立即返回。默认路径只多了
  几次 `None` 判断，排序、分数、日志字段不变。
- **异常原样传播**：`RAGService` 捕获 `Exception` 只为写 `run.failed`（`error_type` 为类名、`stage` 为当前阶段，不含消息），
  随后裸 `raise`，异常对象与 `__cause__` 链不变；CLI 错误前缀映射因此不受影响。空白 query 校验仍在最前，不记事件。
- **recorder 自身故障隔离**：`safe_record` 吞掉 `record` 的任何异常，只 `logger.warning` 记异常类名。
- **`citation_validation.completed`**：在 `assemble_rag_answer` 成功后用 `parse_citations(answer.answer, built_context)` 重算
  （输入与 assemble 内部相同，确定性，必然成功），`source_count = len(answer.sources)`。
- **CLI**：仅传 `--trace` 时创建 recorder 并走 `build_rag_service(settings, trace_recorder=recorder)`；不传时调用与原来逐字相同。
  装配阶段失败（如 `ActiveCollectionError`）时没有 run.* 终止事件，`finish()` 抛 `ValueError`，按“trace 写入失败”只警告一行
  （已写进 SPEC §12.5）。`KeyboardInterrupt` 经过 `finally` 后照常传播。

## 3. Gate B2 逐项结果
| 项 | 结果 |
|---|---|
| 全量 pytest | **N_B2 = 536 passed**（基线 510 + 26：test_tracing 25、架构边界 1） |
| 回放 parity | true |
| reranker_score_mismatches / top5_mismatches / top20_candidate_id_mismatches / top20_candidate_score_mismatches | 全为 [] |
| replay_device / replay_collection_name | cpu / `ef_research_bge_m3_hybrid_c512_v2`（`summary_equal`、`baseline_sha256_ok` 亦为 true） |
| trace 对照（40 题） | parity = true，mismatches = []：(a) 开/关 trace Top5 四项逐位相等；(b) dense Top20 ID 与分数与 A1 `candidate_run` / `candidate_scores` 逐位相等；(c) `rerank.completed` 与 Top5 一致 |
| 开关等价（Fake LLM，3 题 + Null recorder + 抛错 recorder） | `RAGAnswer.model_dump()` 逐字段相等 |
| 真实冒烟 | 退出码 0，`RunTrace.model_validate` 通过，status = completed |
| 冒烟泄漏检查 | 文件不含 query 原文、回答全文（以 stdout 为准）及其前 20 字、`sk-`、`Bearer`、System prompt 片段、`Content:`；`query_sha256` 与 query 的 UTF-8 sha256 一致 |
| `index_state.json` sha256 | `5fbad4f02215d120128ffedf83b2708901ba881bbc258f8c86957ebcc63d8809`（一致） |
| `evals/`、`artifacts/phase15a/`、`artifacts/stage_a/` | `git diff` 无改动；`git status` 无被跟踪文件被回放/冒烟改写 |
| 自查 grep（eval import） | 空 |
| 函数内动态 import | 新增代码无；`__main__.py` 中 `from evidenceflow.runtime import build_rag_service` 为原有的延迟导入，仅因外包 try/finally 重新缩进 |

## 4. 新增测试覆盖
成功路径 11 个事件的顺序与 seq；每个事件 payload 键集合与白名单完全相等（含列表项的键）；序列化 JSON 不含 chunk 正文、
context 结构、query、回答、假 API key、System prompt、白名单外的 usage 字段；context sha256；检索失败（异常对象 `is` 原对象）、
`GenerationContractError`、`CitationValidationError`（类型、消息、`__cause__` 类型与未接 trace 时一致）、`LLMRequestError`
（`is` 原对象）均以 `run.failed` 结束且 stage 正确、`finish().status == "failed"`；空白 query 不记事件；空上下文确定性拒答有
`generation.completed` 且 `llm_called=false`；抛错 recorder 不改变 `RAGAnswer`；3 题开/关等价；`NullTraceRecorder`；
`RunTraceRecorder` 的未 start 丢弃、复用重置、`finish` 的 `ValueError`、假时钟下 `elapsed_ms` / `duration_ms`；
CLI 成功/失败都写 trace 且 stdout/stderr/退出码与不传 `--trace` 时一致、写入失败只警告一行且退出码不变、不传 `--trace` 不写文件、
`KeyboardInterrupt` 不被转换；架构边界。

## 5. 冒烟（ARR-020，中文题）
命令：`python -m evidenceflow ask "<ARR-020 题干>" --trace artifacts/stage_b/B2_trace_sample.json`（CPU、离线模型）。

| seq | name | duration_ms |
|---|---|---|
| 0 | run.started | null |
| 1 | retrieval.dense.started | null |
| 2 | retrieval.dense.completed | 913.6 |
| 3 | rerank.started | null |
| 4 | rerank.completed | 4385.2 |
| 5 | context.started | null |
| 6 | context.completed | 35.6 |
| 7 | generation.started | null |
| 8 | generation.completed | 2108.1 |
| 9 | citation_validation.completed | null |
| 10 | run.completed | 7444.3 |

`generation.completed`：status=answered，llm_called=true，repair_attempted=false，
`usages=[{"prompt_tokens": 2456, "completion_tokens": 85}]`，`raw_response_lengths=[174]`，prompt_version=rag-generator-v3。
`citation_validation.completed`：valid_labels=["[S5]"]，invalid_labels=[]，source_count=1。

**真实 LLM 请求数：1**（无重试）。

## 6. 偏离与遗留
- 任务描述写“完整 13 个事件”，但 SPEC 事件名列表与代码实际成功路径都是 **11 个**（不含 `run.failed`）；测试按 SPEC 名称与代码顺序断言 11 个。
- §12.5 原文只写“按 rank 的 chunk_id、分数”，本步把列表字段名定为 `candidates` / `results`，每项含 `rank`，已写进 SPEC。
- 装配阶段失败时不写 trace 文件（无 run.* 终止事件，`finish()` 无法成立），只在 stderr 警告一行；这会让该场景下 stderr 比不传
  `--trace` 时多一行警告（仅在传了 `--trace` 时），已写进 SPEC。如主会话希望装配失败也产出 trace（例如增加 `stage=setup`），需改合同。
- `RAGService`、`Generator` 新增公开属性 `trace_recorder`（与 `DenseRerankerRetriever` 一致），b2_trace_parity 只用了 retriever 的该属性，未新增实验专用方法。
- 冒烟的 key 泄漏检查只能按模式（`sk-`、`Bearer`）进行，因为不读 `.env`、不知道 key 值；payload 结构上本就不含 key。
