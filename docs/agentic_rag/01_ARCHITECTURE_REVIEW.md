# EvidenceFlow Agentic RAG 架构审查

- 审查日期：2026-10-14
- 审查方式：只读。审查过程中没有修改项目文件
- 审查对象：`D:\Agent_learn\evidenceFlow`（git HEAD `521a059`，另有 35 处未提交改动和若干未跟踪的 Phase 15 产物）
- 配套文档：`02_STAGE_A_EXECUTION_PLAN.md`（阶段 A 执行计划）

> 本文档是决策依据，不是实现规范。按照 `AGENTS.md`，任何实现都要先写进 `PROJECT_SPEC.md`。阶段 A 的具体执行顺序、Gate 和停止条件以 `02_STAGE_A_EXECUTION_PLAN.md` 为准。

---

## 0. 结论摘要

1. **当前瓶颈有评测证据支持。** 正式链路（Dense20 → Reranker → Top5）的 Recall@5 为 0.510。失败归因中 RERANKER_MISS 7 题、CANDIDATE_MISS 6 题。旧基线下多文档题的完整覆盖为 0/4。生成侧有 12% 的输出不符合控制头格式。这几类问题分别对应 Query Planning、Iterative Retrieval 和结构化生成。
2. **在加 Agent 之前，先修三处基础，否则 Agent 无法被可靠评测：**
   - (a) HNSW 检索在不同进程间结果不一致，Phase 15C 因此 STOP。语料只有 817 个 chunk，完全可以改用已有的 `ExactDenseIndex`。
   - (b) 生成的控制头解析很脆弱，而且 `RAGAnswer` 丢掉了 `status` 字段。
   - (c) 当前 817-chunk 的正式链路没有端到端生成评测，Phase 12 测的是旧的 361-chunk 索引。
3. **推荐受控单 Agent，不做自由 ReAct。** 流程固定为：规划 → 按子问题检索（可限定论文）→ 一次批量的证据充分性判断 → 至多补检一轮 → 结构化生成 → 确定性引用校验。每题最多 4 次 LLM 调用、4 次检索。
4. **本轮不引入 LangGraph。** 节点统一写成 `node(state) -> update` 的纯函数，满足触发条件后再机械迁移（见 §6）。
5. **评测先于功能。** 40 题里只有 4 道多文档题，需要扩充多文档、对比和不可答题，并用 Phase 12 的 25 题人工标注来校准 LLM Judge。

---

## 1. 审查范围与局限

**完整阅读**
- `src/evidenceflow/` 全部生产模块（约 7.8k 行）
- README、AGENTS.md、PROJECT_VISION.md
- PROJECT_SPEC.md（2792 行）中的当前合同部分：顶部摘要、§16–17、§20.8、§24–27、§39–43
- Phase 12、13F、15A、15B v2、15B.1、15C、15C.1 的报告
- `evals/research_retrieval/*.md`
- 关键测试

**静态分析**
- 对 `src/tests/scripts/experiments` 全量做 AST 解析，构建 import 依赖图
- 只读查询 `data/chroma/chroma.sqlite3` 的 collection 清单

**按结构浏览**
- `experiments/`（约 39k 行，是生产代码的 5 倍）和 `scripts/`（约 3.2k 行）

**没有做到**
- 没能运行 pytest。审查使用的远程环境碰不到 Windows 上的 `agent-gpu` Conda 环境，云端也装不了依赖。
- 下文的指标全部引自项目内的冻结产物，不是重新跑出来的。
- **阶段 A 的第一步就是在 `agent-gpu` 环境里重新建立可验证的基线。**

**参考项目（直接读源码）**
- SuperMew，commit `e2ebabe`
- LangGraph 官方 `examples/rag`，commit `ccb851e`，以及 “Build a custom RAG agent” 文档页
- PaperQA2，commit `57e89f7`。它是面向科研论文的问答 Agent，与本项目场景最接近

---

## 2. 现有架构与实际数据流

```text
ingest（CLI）
  PDF/MD/TXT → loader(PyMuPDF)
    ├─ hybrid 且是 PDF → structure.py（标题、装饰、表格识别）→ chunk_document_structured（standalone + section 聚合）
    └─ 其他 → cleaner → 固定窗口切分
  → BGE-M3 embedding → Chroma（直接写入 ACTIVE collection）

ask（CLI）
  query → DenseRetriever（Chroma HNSW Top20）→ LearnedReranker（bge-reranker-base Top5）
        → ContextBuilder（4096 token 预算，标签 S1..S5）→ Generator（STATUS/ANSWER 控制头）
        → citation parser（只校验标签是否合法）→ RAGAnswer

不在线上链路
  knowledge/（manifest、lifecycle、BuildJob）只被 scripts/verify_phase15a.py 和测试使用
  retrieval/exact_dense.py 只被实验使用
```

| 模块 | 职责 | 是否在线上 | 评价 |
|---|---|---|---|
| `runtime.py` | 唯一 Composition Root | 是 | 结构干净；Agent 控制器应作为第二个 builder 加在这里 |
| `rag/service.py` | Retrieve → Context → Generate → Citation | 是 | 保留为“简单路径”和对照组 |
| `retrieval/` | Dense、Reranker、Exact primitive | Dense 与 Reranker 在；Exact 不在 | 不支持按文档过滤；reranker 在 CPU 上约 7 s/次 |
| `context/builder.py` | 严格按 rank 纳入完整 chunk | 是 | 只支持单路结果，多子问题合并时需要配额选择 |
| `generation/generator.py` | Prompt v2 + 严格控制头解析 | 是 | 格式失败率 12% |
| `citation/parser.py` | 只校验标签是否存在 | 是 | 不校验引用是否支持论断（与 SPEC §27 一致） |
| `llm/` | `generate(messages) -> str` | 是 | 不支持结构化输出和工具调用，也不记录 usage |
| `knowledge/` | Manifest、签名、staging/activate/rollback | **否** | 写的是 `index_state.json`，线上读的却是 `Settings.collection_name` |
| `eval/` | v1 runner、v2、v3、ir_metrics、quality、mapping | 离线 | 四代并存；`metrics.recall_at_k` 实际计算的是 HitRate |

---

## 3. 决定方向的评测证据

标注“旧链路”的实验是在 Phase 13F 之前的 361-chunk 索引上做的，只能作为方向参考。

| 证据 | 数值 | 来源 | 含义 |
|---|---|---|---|
| 正式 B0 | R@5 0.510 · nDCG@5 0.490 · MRR@5 0.518 | phase15b_hybrid_candidate_v2 | 基线 |
| B0 失败归因（40 题） | UNINDEXABLE 2 · CANDIDATE_MISS 6 · RERANKER_MISS 7 · SUCCESS 25 | 同上 | 排序失误多于召回失误 |
| D30 对比 D20 | 候选 AllRequired 0.80 → 0.925；R@5 +0.069（CI 跨 0） | phase15b1 | 扩大候选池有效 |
| D30 晋升 | STOP：ARR-032/033 在不同新进程中候选结果不一致 | phase15c | 不确定的检索会让 A/B 实验无法复现 |
| BM25 混合 v2 | DO_NOT_PROMOTE，只额外救回 1 个可索引 miss | phase15b v2 | 不要重新引入 BM25 |
| LLM 查询改写（旧链路） | Hit@5 0.475 → 0.600 | query_rewrite_experiment_v1 | 被定向召回 Gate 否决；Gate 的设计与收益来源不匹配，值得重测 |
| 多文档题（旧链路） | 完整覆盖 0/4 | research_dense_baseline_v1 | Query Decomposition 的直接动机 |
| Phase 12（旧链路，25 题，人工标注） | 格式失败 3/25 · Claim Recall 0.77 · Groundedness 1.0 · 引用 P 1.0 / Completeness 0.91 · 错误作答率 0 | phase12_rag_quality_v1 | 风险在于漏答、漏引，不是幻觉 |
| Reranker 延迟（CPU） | D20 约 7.0 s/查询 | phase15b1 | 每多一次检索多约 7 s |

---

## 4. 设计缺陷与技术债

| 级别 | 问题 | 代码证据 | 建议 |
|---|---|---|---|
| P0 | 线上检索结果不确定 | `runtime.py:52` 使用 Chroma HNSW；Phase 15C 报告 | 线上改用 `ExactDenseIndex` |
| P0 | 生成格式失败 12%；`RAGAnswer` 没有 status | `generator.py:55-61`；`citation/parser.py:79` | 改为 JSON 结构化输出 + 一次修复重试；`RAGAnswer.status` |
| P0 | 当前链路没有端到端生成评测 | Phase 12 的 fingerprint 是 361 chunks | 阶段 B 自动化复跑 |
| P1 | lifecycle 与线上读取脱节 | `knowledge/lifecycle.py` 中 activate/rollback 只写 `index_state.json`；`runtime.py:39,52` 读 Settings | Runtime 以 active pointer 为准 |
| P1 | ingest 直接修改 ACTIVE，且先删后写 | `ingestion/service.py:143-145` | ACTIVE 上禁止 ingest，更新走 staging |
| P1 | 配置默认值与实际不一致 | `config.py:138` 默认 `..._c512_v1`；`.env.example` 写的是 gpt-4o-mini | 与 active manifest 对齐 |
| P1 | LLM Client 缺少结构化输出和 usage | `llm/base.py` | 新增 `generate_json` 和 `last_usage` |
| P1 | 检索不支持文档过滤 | `DenseSearch.search(query_embedding, top_k)` | 阶段 C 加 `document_ids` |
| P2 | SPEC 膨胀且自相矛盾 | 2792 行；§17.3.1 标“已实施 ✅”实际已退休；§24 第 1744 行写“Citation Parser 属于下一阶段” | 拆成当前合同 + `docs/history/` |
| P2 | 过期注释和字段 | `storage/base.py:47`（BM25Index）、`reranker.py:323`（`fallback_branch_used`）、`domain/chunk.py:76,80`（RRF） | 清理 |
| P2 | 实验与测试耦合 | experiments 39k 行；7 个测试文件 import experiments | 删除已结束阶段的实验及其测试 |
| P2 | 杂项与孤儿数据 | `1.26`、`src/evidenceflow.zip`、`egg-info`、`tmp/` 43 MB；Chroma 中有 7 个 collection，只有 1 个在用 | 清理（见 §9） |
| P2 | eval 模块四代并存 | `eval/runner.py`、`metrics.py`、`v2.py`、`v3.py`、`ir_metrics.py`、`quality.py` | 确定 canonical 实现，legacy 打标或删除 |

---

## 5. 开源方案源码对比

| 维度 | SuperMew | LangGraph 官方 Agentic RAG | PaperQA2 | EvidenceFlow 的取舍 |
|---|---|---|---|---|
| 控制方式 | 外层 Agent 只拿到一个知识库工具，且**每轮最多调用 1 次**（`tools/knowledge.py` 的 `acquire_knowledge_tool_slot`），迭代在固定 StateGraph 内完成 | Agent → 二值判断 → 改写，**循环没有上限** | 工具 Agent（search / gather_evidence / gen_answer / complete），另有确定性的 `run_fake_agent` | 固定流程 + 预算；保留 RAGService 作为基线 |
| Query Planning | 规则快速路径 + Fast 模型一次输出 2–4 个子问题（`ComplexityResult`），用 `Send` 并行 | 无 | Agent 自己生成 query | 借鉴，并加上“目标论文”字段 |
| Evidence Grading | 对整批证据做一次结构化判断 `EvidenceGrade`（relevance / answerability / ambiguity / route），再由 `_resolve_route` 用规则修正 | 二值判断 | 逐 chunk 做 LLM 摘要并打 0–10 分（RCS） | 借鉴“批量判断 + 规则修正”；不逐 chunk 打分，reranker 已经在做 |
| 迭代 | 改写一次，Step-back 和 HyDE 二选一；子问题不再改写 | 无上限 | `max_timesteps` | 至多补检一轮 |
| 结果语义 | ANSWERABLE / NO_KNOWLEDGE / INSUFFICIENT / PARTIAL + coverage gap（`rag/outcomes.py`） | 无 | `has_successful_answer` | AnswerStatus 扩展出 PARTIAL |
| HITL / Memory | clarify / scope_select + checkpoint；Thread 持久化 | 无 | 无 | 暂缓 |
| 评测 | `RagExpectedBehavior` + 7 项 Judge 指标 + CI Gate（`evaluation/rag.py`、`.github/workflows/rag-eval.yml`） | 无 | LitQA | 评测集加“期望状态”；对齐 `eval/quality.py` |

SuperMew 的平台层（PostgreSQL Run/Event、Redis、Milvus、RBAC、Sandbox、SQL 助手）有 5 万行以上，面向多租户的企业场景，不需要照搬。值得借鉴的是 `backend/rag/` 里的路由语义、预算规则和结果分类。LangGraph 教程适合理解流程形状，但没有循环上限，也没有证据状态，不能直接用于生产。

---

## 6. 是否引入 LangGraph

**结论：本轮不引入。** 理由如下：
- 需要的流程只有 6 个节点、1 个分支、1 个有界循环，用纯 Python 写大约 200 行。
- LangGraph 的核心价值是 checkpoint、`interrupt` 式 HITL、流式事件和 `Send` 并行，当前都用不上。
- 子问题并行的收益有限。瓶颈在 reranker 的 CPU 计算，更合适的做法是把多个子问题合成一次批量前向。

**满足下列任意两条时再迁移：**
1. Phase 16 的 API/UI 需要按 `docs/EVIDENCEFLOW_UI_RUNTIME_CONTRACT_v0.md` 的事件名推送流式 RunEvent；
2. 需要在同一个 Run 中暂停和恢复的 HITL；
3. 接入第二类外部工具（例如 Web），需要模型自主选择；
4. Run 需要持久化，并能从中断处恢复。

**为了降低迁移成本，现在就这样写：** 节点是 `def plan(state: ResearchState) -> dict` 形式的纯函数，state 用 Pydantic 定义。

---

## 7. 八项能力逐项建议

### 7.0 基础前置（阶段 A）
- **当前实现：** HNSW Top20；控制头文本解析。
- **实际问题：** Phase 15C STOP；12% 格式失败。
- **修改依据：** Phase 15C.1 已经实现 `ExactDenseIndex`，NumPy 参照 40/40 一致；结构化输出是 Planner 和 Grader 的前提。
- **参考来源：** SuperMew `_invoke_structured_model`。
- **预期收益：** A/B 实验可以逐题复现；格式失败归零。
- **复杂度：** 低。
- **风险：** ModelScope 上的 Qwen3.5 对 `json_schema` 的支持还没验证，需要兜底方案；切换 Exact 后要重新冻结基线。

### 7.1 Query Planning（阶段 C）
- **当前实现：** 没有；Translation 分支已退休。
- **实际问题：** 多文档完整覆盖 0/4；中文问题依赖已退休的翻译分支（旧链路下 zh→en Hit@5 0.667，en→en 0.389）。
- **修改依据：** LLM 改写在旧链路上让 Hit@5 提升 0.125；语料只有 10 篇，文档目录可以整段放进 Prompt。输出结构为 `{kind: single|multi_doc|comparison, sub_queries:[{text_en, aspect, target_document_ids}], keep_original}`，最多 3 个子问题。
- **参考来源：** SuperMew `classify_complexity`。
- **预期收益：** 多文档、对比题的覆盖率上升；用一个机制同时替代翻译分支和查询改写。
- **复杂度：** 中，约 150 行。
- **风险：** 简单题被拆分后可能变差，所以保留原始 query 一路，并加规则快速路径；目标论文判断错时，要靠一路不过滤的检索兜底。

### 7.2 Tool-based Retrieval（阶段 C）
- **当前实现：** 只有 `DenseRerankerRetriever.retrieve(query)` 一个入口。
- **修改依据：** 工具在这里是内部接口，由控制器按计划确定性地调用，不暴露给 LLM 自由选择：
  - `search(query, document_ids=None, k=5)`
  - `expand_neighbors(chunk_id, window=1)`
  - `list_documents()`
- **参考来源：** PaperQA2 的 `paper_search` / `gather_evidence` 分工；SuperMew 把检索做成受预算限制的工具。
- **预期收益：** 计划可以落地；邻近扩展能补上被 chunk 边界切断的证据。
- **复杂度：** 低到中。
- **风险：** rerank 调用次数会翻倍，需要多 query 批量接口。

### 7.3 Evidence Grading（阶段 D）
- **修改依据：** 每轮对整个证据池做一次批量判断，输出：
  - 每个子问题是 covered、partial 还是 missing
  - `missing_aspects[]`
  - route：`answer | retrieve_more | insufficient`

  再用确定性规则修正这个结果。reranker 分数先只写入 trace，等评测校准出阈值后再用于路由。
- **参考来源：** SuperMew `EvidenceGrade` + `_resolve_route`。不采用 PaperQA2 的 RCS（每题多 5–20 次 LLM 调用）。
- **风险：** 判断过于保守会导致误拒答，Phase 12 已出现 1 例。

### 7.4 Iterative Retrieval（阶段 D）
- **修改依据：** 至多补检一轮，查询由 `missing_aspects` 生成；证据池按 `chunk_id` 去重。合并后先按子问题配额选择，再交给 ContextBuilder。
- **参考来源：** SuperMew 只改写一次。LangGraph 教程的无上限循环是反例。
- **风险：** CPU 上每多一轮约多 7 s，需要记录 p95 并设上限。

### 7.5 Citation Verification（阶段 E）
- **修改依据：** 先做两项确定性检查：
  - (a) 逐句检查引用覆盖；
  - (b) 数值一致性：句子里的数字必须出现在它引用的 chunk 原文中。

  用 LLM 判断“引用是否支持论断”目前只放在离线评测里。
- **风险：** 数字格式和单位不一致会造成误报，所以第一版只报告、不拦截。

### 7.6 Memory（暂缓）
- 只做单次运行内的工作记忆，放在 `ResearchState` 里。多轮对话等 API 阶段再做，届时把追问改写成独立完整的问题。

### 7.7 Trace（阶段 B）
- 新增 `RunTrace` Domain Model，记录 plan、每次检索、grade、生成和校验，以及 token 用量。事件名复用 UI 合同 v0；CLI 加 `--trace` 参数。
- 不引入 LangSmith、Langfuse 或 OpenTelemetry。

### 7.8 Evaluation（阶段 B）
- 用 `eval/quality.py` 自动化复跑 Phase 12。
- LLM Judge 用 25 题人工标注校准，与人工的一致率 ≥ 0.85 才能作为主指标。
- 按现有 anchor 协议扩充约 15 道多文档/对比题和约 10 道不可答题，并标注期望 status。
- 新增指标：Context AllRequired、Claim Recall、拒答混淆矩阵、引用覆盖、数值一致、步数、LLM 调用次数、p50/p95 延迟、token。
- 每个阶段都和 RAGService 做逐题配对比较，使用 `eval/comparison.py` 的 bootstrap 和 Holm 校正。
- 不把 RAGAS 作为主依赖。

---

## 8. 明确不做

| 方案 | 理由 |
|---|---|
| 自由 ReAct 工具调用 | 只有一个知识源；SuperMew 生产中也限制为每轮调用一次 |
| Multi-Agent | Vision §5 的前提条件没有满足 |
| 逐 chunk 的 LLM 相关性打分 | reranker 已经在做；每题多 5–20 次调用 |
| 重新引入 BM25 | Phase 15B v2 结论为 DO_NOT_PROMOTE |
| HITL 澄清 | CLI 场景需求弱 |
| 长期记忆 / 向量记忆 | 没有多轮需求 |
| 现在接 Web Search | Vision Phase 3，需要单独设计来源合同 |
| 现在引入 LangGraph | 见 §6 |

---

## 9. 文件处置分类（基于 import 依赖图与 Chroma 元数据）

### 9.1 参照开源设计重构
| 文件 | 参照 | 阶段 |
|---|---|---|
| `generation/generator.py` | SuperMew 结构化调用 | A |
| `llm/base.py`、`llm/openai_compatible.py` | 同上 | A |
| `domain/answer.py`、`domain/generation.py` | SuperMew `outcomes.py`（A 只加 status，PARTIAL 放到 D） | A / D |
| `eval/` 端到端部分 | SuperMew `RagExpectedBehavior` 和 Gate | B |
| 新增 `research/` | SuperMew pipeline、PaperQA2 fake agent | C / D |

### 9.2 只需修改
- `runtime.py`：接入 Exact；以 active pointer 为准
- `ingestion/service.py` 与 `__main__.py`：ACTIVE 保护
- `config.py`：默认值、index state 路径
- `retrieval/reranker.py`、`storage/base.py`、`domain/chunk.py`：过期注释和字段
- `citation/parser.py`：透传 status

### 9.3 保持不动
ingestion 的 loader / structure / chunker / cleaner / tokenizer、embedding、`knowledge/*`、`domain/ids.py`、`ContextBuilder`、citation 的解析逻辑、`model_device.py`、`inspection.py`。

### 9.4 删除或移出（阶段 A4 执行，执行前先备份）
| 对象 | 依据 |
|---|---|
| `1.26`、`src/evidenceflow.zip` | 未被 git 跟踪，没有任何引用 |
| `src/evidenceflow.egg-info` | 构建产物，已在 .gitignore 里 |
| `tmp/`（43 MB） | 没有任何代码引用；`tmp/phase12_ann_batch*` 已有正式产物 `evals/.../phase12_rag_quality_v1/claim_annotations.json` 等。**先移到仓库外备份**，因为 tmp 不在 git 里 |
| Chroma `ef_research_bge_m3_hybrid_c512_v2_staging`（817 条） | 与 active 重复；`index_state.json` 中 staging 为空；只有 `scripts/repair_active_index_metadata.py` 引用 |
| Chroma `..._c512_o64_refmeta_v1`、`ef_research_structure_v1`、`ef_research_structure_v1_elemaware` | 只有已结束的实验脚本引用 |
| `experiments/phase13c/`、`experiments/phase14a/`、`experiments/run_phase13*.py`、`experiments/run_phase13c_*.py`，以及 `tests/test_phase13_kb_chunk_audit.py`、`test_phase13b_controlled_rebuild.py`、`test_phase13c_cleaning_benchmark.py`、`test_phase13c_parser_benchmark.py` | 阶段已结束，结果冻结在 `evals/`；文件已在 git 中（phase13c 22 个、phase14a 34 个），可以恢复 |
| `experiments/archive/`（33 个文件，已在 git 中） | 15B.5 保留它的理由是“没有 git”，这个理由已经不成立 |

**必须保留：**
- Chroma `ef_research_bge_m3_hybrid_c512_v2`（active）
- `ef_research_bge_m3_hybrid_c512_v1`（SPEC §39.1 禁止删除）
- `ef_research_bge_m3_c512_o64_v1`（canonical qrels 和 Phase 12 基线的来源；被 config 和多个测试引用）
- `experiments/run_phase15b1_dense_candidate_depth.py`、`phase15b_active_relevance.py`、`run_phase15c*`、`stabilize_gold_v3.py`（被当前测试或实验 import）

---

## 10. 分阶段路线与 Gate

| 阶段 | 内容 | Gate |
|---|---|---|
| A 基础修复 | 冻结起点；线上切换 Exact；结构化生成 + `RAGAnswer.status`；lifecycle 接到线上；清理；精简 SPEC | 见 `02_STAGE_A_EXECUTION_PLAN.md` |
| B 评测先行 | 自动化端到端评测；Judge 校准；扩充约 25 题；RunTrace | Judge 一致率 ≥ 0.85；生成新的 RAGService 端到端基线 |
| C 规划与检索工具 | Planner、文档过滤、邻近扩展、配额选择 | 多文档/对比子集 Context AllRequired 配对提升，CI 不跨 0；简单题不退化 |
| D 判断与补检 | Grader、一轮补检、PARTIAL | Claim Recall 提升；错误作答率为 0；误拒答不上升 |
| E 引用校验 | 逐句覆盖、数值一致 | 误报率可接受后，再决定是否触发重新生成 |
| F 按需 | LangGraph、Web、多轮对话 | §6 的触发条件 |

**与 SuperMew 的距离：**
- RAG 核心能力对齐需要 A–E，共 5 个阶段。
- 完整平台对齐还要再加约 4 个阶段：API + Run/Event + SSE；HITL + checkpoint；多工具 / Skill；多用户认证、后台 worker、前端。
- 建议做到 E，加上 API 和 Trace UI 就停。

---

## 11. 执行模型建议

- **Sonnet：** 负责阶段 A 的实现、测试和清理。这些任务边界清楚、Gate 明确。
- **Opus：** 负责各阶段的 SPEC 合同、Planner/Grader Prompt、Judge 校准方法、不确定性排查，以及每个阶段结束时的 diff 验收。
- **不用 Haiku：** 项目有冻结 hash Gate 和中文注释规范，出错代价高。
- **节奏：** Opus 写任务书 → Sonnet 实现并跑测试 → Opus 验收。

---

## 12. 风险

- ModelScope 上的 Qwen/Qwen3.5-35B-A3B 是否支持结构化输出、输出里是否会混入思考内容，都还没验证。
- reranker 在 CPU 上约 7 s/次。GPU 路径（§43）还没经过质量 Gate；冻结基线都是 CPU 跑出来的。
- 评测集扩充后约 65 题，统计功效有限，结论要表述为“方向一致 / 置信区间”，不能说“显著提升”。

## 参考来源
- SuperMew：https://github.com/icey1287/SuperMew（`backend/rag/pipeline.py`、`backend/rag/outcomes.py`、`backend/tools/knowledge.py`、`backend/agent/factory.py`、`backend/agent/middleware.py`、`backend/evaluation/rag.py`、`.github/workflows/rag-eval.yml`）
- LangGraph examples/rag：https://github.com/langchain-ai/langgraph/tree/main/examples/rag
- LangGraph 文档：https://docs.langchain.com/oss/python/langgraph/agentic-rag
- PaperQA2：https://github.com/Future-House/paper-qa（`src/paperqa/agents/tools.py`、`src/paperqa/prompts.py`、`src/paperqa/agents/main.py`）
