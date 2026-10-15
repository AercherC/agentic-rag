# EvidenceFlow — Current Specification

**Version:** 0.28
**Last Updated:** 2026-10-15（Stage B B2）
**Scope:** 本地科研 PDF 知识库 RAG：ingest / ask 两条正式链路及其评测基线。

本文件只写**当前合同**。数值与规则以代码为准；与代码不一致时视为本文件的缺陷，按 §15 修正。
v0.26 及更早的完整历史（Phase 1–15C、Stage A §S 原文）见 `docs/history/PROJECT_SPEC_v0.26_full.md`，
索引见 §16。

---

## 1. 当前架构

```text
ingest:  Source docs → Loader → Cleaner → (PDF+hybrid: 结构解析 → 混合切分 | 其他: 固定窗口)
         → BGE-M3 Dense Embedding（L2 归一化）→ Chroma（唯一持久化来源）
ask:     Query → BGE-M3 embed_query → ExactDenseIndex（numpy.exact）Top20
         → BAAI/bge-reranker-base Top5 → ContextBuilder → Generator rag-generator-v3（JSON 输出）
         → Citation 校验 → RAGAnswer（含 status）
```

- 唯一 Composition Root：`evidenceflow.runtime`（`build_retriever`、`build_rag_service`、`build_ingestion_service`、
  `resolve_active_collection`、`guard_ingest_target`）。CLI 只触发装配，问答编排只在 `RAGService`。
- 线上 collection 由 lifecycle active 指针决定（§6），当前为 `ef_research_bge_m3_hybrid_c512_v2`：10 篇文档、817 个 chunk。
- `evidenceflow.eval.*` 不得被 `runtime` / `retrieval` 反向依赖（`tests/test_architecture_boundaries.py` 守护）。
- Runtime 不维护第二套检索算法，也没有 backend 开关；回滚依赖 lifecycle 的 previous 指针、冻结产物与 git tag
  （`stage-a-start` 为 Stage A 之前的代码）。

## 2. 开发环境与依赖

- Conda 环境：`agent-gpu`（Python 3.12；torch 2.13 + CUDA；chromadb 1.5.9；未安装 faiss）。不另建环境。
- 依赖以 `pyproject.toml` 为准：chromadb、httpx、numpy、openai、pydantic、pydantic-settings、pymupdf、python-dotenv、
  sentence-transformers、transformers；测试依赖 pytest。新增依赖须先说明理由（AGENTS.md §5）。
- 离线运行：`HF_HOME=.cache/huggingface`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`，
  `RERANKER_MODEL` 指向本地 `.cache/models/bge-reranker-base`。
- 测试：`python -m pytest -q`；只用 Fake/Stub 与临时目录，不调用真实 LLM、不打开 `data/chroma`。
- 数值 Gate 一律 `MODEL_DEVICE=cpu`；GPU 浮点差异可改变个别题排序，GPU 结果只作参考。

## 3. Domain Models 摘要（Pydantic，`src/evidenceflow/domain/`）

| 模型 | 关键字段与约束 |
|---|---|
| `DocumentSegment` | `content`、`page_number`（PDF 1-based；MD/TXT 为 None）、`metadata` |
| `Document` | `document_id`、`source_path`（root 内 NFC/POSIX 相对路径）、`file_name`、`file_type ∈ {pdf, md, txt}`、`content_hash`、`segments`、`metadata` |
| `DocumentChunk` | `chunk_id`、`document_id`、`content`、`token_count`、`chunk_index`（0 起连续）、`page_start/page_end`（PDF 1-based，MD/TXT 两端 None）、`file_name`、`content_hash`（原文件 bytes）、`metadata` |
| `RetrievedChunk` | `chunk`、`rank`（1 起）、`similarity_score`（Exact 下为 float32 内积 = cosine） |
| `RerankedRetrievedChunk` | `chunk`、`rank`（重排后）、`candidate_rank`（Dense 名次）、`similarity_score`、`reranker_score`；各分数互不覆盖 |
| `ContextEntry` / `BuiltContext` | 见 §8 |
| `SourceReference` | `source_label`、`document_id`、`file_name`、`page_start/page_end`（同为 None 或 start ≤ end）、`chunk_id` |
| `AnswerStatus` | `StrEnum`：`answered`、`insufficient_context` |
| `GeneratedAnswer` | `status: AnswerStatus`、`answer_text`（非空白） |
| `GeneratorOutput` | 模型 JSON 合同，`extra="forbid"`：`status: AnswerStatus`、`answer`（非空白） |
| `RAGAnswer` | `query`、`status: AnswerStatus`（必填）、`answer`、`sources: list[SourceReference]`、`retrieved_chunks: list[RetrievedChunk | RerankedRetrievedChunk]` |
| `LLMMessage` | `role ∈ {system, user, assistant}`、`content`（非空白） |
| `CitationParseResult` | `valid_labels`、`invalid_labels`（按首次出现去重） |
| `RunTrace` / `TraceEvent` | 见 §12.5（`extra="forbid"`；不进入 `RAGAnswer`） |

业务层只传递项目 Domain Model，不向外传播第三方对象（Chroma 结果、Tensor 等）。

## 4. ID 与 hash 规则（`domain/ids.py`）

- `document_id = sha256(UTF-8(ingestion root 内相对路径经 NFC、`\`→`/`、normpath))`；绝对路径或 `..` 逃逸报错。
  内容不参与，同一路径更新后身份不变。
- `content_hash = sha256(原始文件 bytes)`；Cleaner 不得重算。
- `chunk_id = sha256(紧凑、键排序 JSON {document_id, chunk_size, chunk_overlap, chunk_index, content_hash=sha256(content)})`。
  hybrid 策略中 `chunk_overlap` 载荷固定记录 0。相同文档、配置、文本 → 相同 ID。
- 结构元素 ID：`el-` + sha256(`document_id|page|block_index`) 前 24 位。
- `corpus_signature` 只由按 `(document_id, relative_path, content_hash)` 排序的 source 记录生成；
  `index_signature` 由 corpus_signature、parser/cleaner/chunker 版本与配置 hash、embedding model/version 生成，
  不含 reranker、TopK、Context、Generator、Prompt、Citation 等配置。

## 5. Ingestion 合同

- Loader `load_document(path, ingestion_root)`：`.pdf/.md/.txt`（小写匹配）。PDF 用 PyMuPDF 逐页提取，空页不生成
  Segment 但保留原页码；整份无文本报错 `PDF contains no extractable text`。MD/TXT 按 UTF-8 作为一个无页码 Segment。
  root 外文件报错。metadata 至少 `file_size`，PDF 另有 `page_count`。
- Cleaner `clean_document(document)`：逐 Segment NFC、换行统一为 `\n`、仅删开头 BOM、去首尾空白；PDF/TXT 另删行尾
  空格/Tab 并把 ≥3 个换行压为 2 个；Markdown 不压缩。空 Segment 删除（页码不重排），全部为空报错
  `Document contains no usable text after cleaning`。返回新对象，身份字段不变。禁止全局空白压缩。
- 切分：
  - `chunk_strategy="hybrid"`（默认）且 PDF：在原始页 block 视图上确定性结构解析（标题/装饰/元素分类），standalone
    元素（table、table_caption、figure_caption、equation）独立成块，正文按连续相同 section_path 分组贪心打包；
    元素分隔符 `"\n\n"`；任何 chunk ≤ `chunk_size` token（拼接后文本、`add_special_tokens=False`）；窗口重叠为 0；
    metadata 记录 `chunk_strategy="hybrid"`、`element_ids`、可选 `heading_path`。
  - MD/TXT 或 `chunk_strategy="fixed"`：固定窗口 `step = chunk_size - chunk_overlap`，Segment 间插入 `"\n\n"`，
    页码按 token 来源取 min/max，content 由 `tokenizer.decode()` 生成。
- `IngestionService` 是唯一摄取入口：按 root 内相对路径字典序串行；每文档状态 `inserted/skipped/updated/failed`；
  同 `document_id` 同 `content_hash` 跳过；内容变化时先完整生成新 chunk 与向量，再删旧写新（“先删后写”）。
  单文件失败不终止批处理，任何失败使 CLI 非零退出。一个文档一次 `embed_documents()`。
- Collection 身份隔离：hybrid 不得写入 fixed 身份 collection（`ef_research_bge_m3_c512_o64_v1`、`ef_bge_m3_c512_o64`，
  Settings 校验拒绝）。Chroma 记录 metadata 由 `storage.serialize_chroma_record_metadata` 唯一序列化
  （`chunk_id`、`document_id`、`file_name`、`content_hash`、`chunk_index`、`token_count`、页码、`domain_metadata_json`）。
- ACTIVE collection 不允许直接 ingest（§6.3）；重建只走 staging（§6.4），因此“先删后写”不会作用于线上索引。

## 6. 索引生命周期

### 6.1 状态与 manifest
- 状态文件 `Settings.index_state_path`（默认 `artifacts/phase15a/index_state.json`，相对当前工作目录）：
  `LifecycleState{schema_version="index-lifecycle-v1", knowledge_base_id, active, previous?, staging?}`，
  每个指针为 `IndexPointer{collection_name, index_signature, manifest_path, document_count, chunk_count, status}`。
  写入一律 `_write_json_atomic`（键排序）。
- `IndexLifecycle` 方法：`initialize_active`、`prepare_staging`、`activate`（旧 active → previous）、`rollback`
  （只切指针，不重建）、`discard_staging`（只把 staging 置 null；本来为 null 时不重写文件；**不删 collection**）。
- `KnowledgeBaseManifest` 确定性序列化；source diff 只有 `UNCHANGED/ADDED/UPDATED/DELETED/RENAMED`，同 hash 多个
  rename 候选为 `BLOCKED`。`BuildJob` 只记录同步执行状态（含 `FAILED`、`BLOCKED`），无队列/worker。

### 6.2 active 指针解析（`runtime.resolve_active_collection`，线上选 collection 的唯一入口）
1. 状态文件存在 → 返回 `state.active.collection_name`；文件无法解析 → `ActiveCollectionError`（不回落）。
   `COLLECTION_NAME` 被**显式设置**（`"collection_name" in settings.model_fields_set`，含环境变量、`.env`、构造参数）
   且与 active 不一致 → `ActiveCollectionError`；一致则放行。
2. `INDEX_STATE_PATH` 被显式设置但文件不存在 → `ActiveCollectionError`（不回落，避免绕过指针与保护）。
3. 未显式设置且默认路径无文件（测试、全新环境）→ 返回 `Settings.collection_name`。
- 规则 1 下 `build_retriever` 在加载任何模型之前用只读 `storage.read_collection_metadata` 校验 collection 存在且
  `metadata["lifecycle"] == "ACTIVE"`，再以 `ChromaVectorStore(..., create_if_missing=False)` 打开；不满足抛
  `ActiveCollectionError`，绝不创建空 collection。规则 3 不校验 lifecycle，沿用 `get_or_create`。

### 6.3 ACTIVE 保护（`runtime.guard_ingest_target`，在构造任何模型前执行）
- 状态文件存在且 ingest 目标等于 active → `ActiveIndexProtectedError`。
- 纵深防御（与状态文件无关）：目标 collection metadata 为 `lifecycle=ACTIVE` → `ActiveIndexProtectedError`；
  collection 不存在或无该标记时放行（全新环境首次 ingest 可用），检查不创建目录或 collection。
- CLI 退出码见 §12。

### 6.4 staging 构建脚本 `scripts/build_staging_index.py`
- 复用 `StagingBuildCoordinator`（SCANNING→…→READY 与校验）、`IngestionService`（builder 回调）、`validate_collection`；
  不复制 ingestion 逻辑。
- 参数：`--expected-chunk-ids-from-active`（chunk ID 集合须与 active 完全一致）、`--activate`（默认不激活）、
  `--discard-after-validate`（校验**通过**后删 staging collection 再 `discard_staging()`；失败时保留；与 `--activate`
  互斥）、`--discard-staging`（不构建，只删已登记 staging 并清指针；不能与其他选项组合）、`--job-dir`。
- 已有 staging 指针或同名 collection 存在时拒绝开始。manifest 规模取 active 的数量（只支持同一 corpus 全量重建）。
- 命名 `build_staging_collection_name(base, signature, *, reserved)` = 清洗后 `base[:24]` + `-` + `sha256(完整 base)[:8]`
  + `__staging__` + `signature[:16]`（≤60 字符）；`reserved` 中的名字按 `#1…#8` 盐确定性避让，全占用抛 `ValueError`。
  脚本传 `reserved={active, previous}`，生成后再查一次，等于 active/previous 时构建前拒绝。
- `discard_staging_collection`：先确认目标不是 active/previous，先删 collection 后清指针；collection 已不存在时只清指针
  （幂等）；staging 指针为 null 时拒绝。
- `--activate`：切指针**之前**经 `mark_collection_lifecycle` 把新 collection metadata 改为 `lifecycle=ACTIVE`。合并
  metadata 时去掉旧键 `hnsw:space`（Chroma 只拒绝含该键的 `modify`；cosine 由 configuration 保证，不变）。新建 staging
  不写 `hnsw:space`，因为源码不读它。
- `validate_collection` 只校验数量、文档集合、chunk ID 重复/缺失/多余、必需 metadata（`chunk_strategy`、`element_ids`）、
  向量维度、一次非空 Dense smoke；不比较向量数值或排序，构建设备不影响结果。

### 6.5 孤儿 collection 清理 `scripts/prune_chroma_collections.py`
- 默认 dry-run，`--execute` 才删除；允许删除名单与禁止名单（v2、hybrid v1、c512_o64_v1）写死且不相交；状态文件中
  active/previous/staging 引用的名字并入禁止集合；有交集或状态文件读不出时拒绝。
- 删除前导出全部 collection 的 `name/id/count/metadata` 到 before JSON，`--execute` 后导出 after JSON。本次要写的
  输出文件已存在时拒绝（需显式 `--overwrite`），保护 A4 审计记录。所有拒绝都在连接 Chroma 之前。

### 6.6 运维说明：Chroma 磁盘残留
- Chroma 1.5.9 的 `delete_collection` **不会**删除磁盘上的 HNSW 段目录（`data/chroma/<segment-uuid>/`）。
- 找出不再被引用的目录：先确认没有进程在使用 Chroma，再以只读方式打开 `chroma.sqlite3`
  （如 `sqlite3.connect("file:data/chroma/chroma.sqlite3?mode=ro", uri=True)`），读取
  `SELECT id, scope, collection FROM segments`；`data/chroma` 下名字为 UUID、但不在 `segments.id` 中的目录即为残留。
- 删除残留目录前必须先完整备份 `data/chroma`，并确认没有任何进程打开 Chroma；只删核对过的目录。
- `chroma.sqlite3` 的体积不会因删除 collection 而缩小；是否执行 `VACUUM` 未决，Stage A 不做（§14.1）。

### 6.7 已知限制（文档化，行为未改；汇总于 §14.1）
- 激活后旧 active 的 metadata 仍是 `lifecycle=ACTIVE`，rollback 依赖这一点（只切指针、不改 metadata），因此可能同时有
  多个 collection 标为 ACTIVE，纵深防御会把它们都当作 ACTIVE 保护。
- `artifacts/phase15a/index_state.json` 被 git 跟踪。真正执行 activate/rollback 之后，不要对它执行 `git restore` /
  `git checkout`，也不要在有未提交的指针变更时切换分支，否则线上 active 指针会被改回旧值。
- 被 `mark_collection_lifecycle` 改写过的 collection 不再有 `hnsw:space`；以后若改写 v2，需同步
  `scripts/verify_phase15a.py` 的 metadata 快照比对。
- `--discard-staging` 在没有登记 staging 时返回 2（拒绝、不改动），不是 0。
- `mark_collection_lifecycle` 与 `lifecycle.activate` 不是原子操作；`--activate` 尚未在真实数据上运行过。

## 7. Retrieval 合同

- `build_retriever(settings, *, trace_recorder=None)`（recorder 见 §12.5）：启动时经 `ChromaVectorStore.iter_all_chunks_with_embeddings()`（按 `chunk_id` 升序）在进程内
  构建 `ExactDenseIndex`，作为 `DenseRetriever` 的 `DenseSearch`。后端固定 `numpy.exact`（float32 全量内积），代码不导入
  faiss，不写任何第二份持久化向量文件；Exact 索引不进入 `index_signature`。
- 启动校验：索引 `size == store.count` 且 `dimension == 1024`，否则抛 `ExactDenseIndexError`，拒绝启动。
- 向量须有限、同维、float32、L2 归一化。Dense 候选按 `similarity_score DESC, chunk_id ASC` 排序，保留原始 float32 分数。
- 深度：Dense Top `reranker_input_k=20` → `LearnedReranker`（`BAAI/bge-reranker-base`，batch 4，max_length 512，float32）→
  Top `reranker_final_k=5`。重排按 reranker score 降序，同分按候选原始顺序、再按 `chunk_id`；Reranker 只改顺序，
  不改候选集合；失败抛错，不静默返回 Dense 顺序。
- 设备：`Settings.model_device ∈ {auto, cpu, cuda}` 是 Embedding 与 Reranker 共用的唯一设备配置；`auto` 在 CUDA 可用时为
  `cuda:0`，否则 cpu；显式 `cuda` 不可用时报错。Tokenizer 始终在 CPU；GPU Tensor 不进入 Domain Model/JSON。
- 日志（`retrieval/reranker.py`）：`retrieval_strategy`、`candidate_count`、`reranker_enabled`、`reranker_input_count`、
  `reranker_output_count`、`reranker_latency`、`reranker_truncated_pair_count`；不记录 secret、向量或大段正文。
- 离线实验规则：query 必须逐条 `embed_query`，不得用 `embed_documents` 批量编码（批量填充会造成 float32 末位差异，
  无法与线上基线逐位对齐）。回放脚本在设备不是 cpu 且未传 `--allow-gpu` 时退出。

## 8. Context Builder 合同

- `ContextBuilder.build(retrieved_chunks, max_tokens=None) -> BuiltContext`；默认预算 `Settings.max_context_tokens`，
  tokenizer 为 BGE-M3（只依赖 `TokenizerProtocol`）。
- 按检索 `rank` 升序处理；`DocumentChunk` 不可拆分。预算按完整渲染文本计数（标签、`File:`、可选 `Pages:`、`Content:`、
  正文、分隔符）；放不下的 chunk 跳过并继续尝试后续更短的 chunk，不截断、不扩展、不摘要、不调用 LLM。
- 标签按实际纳入顺序连续分配 `[S1]`、`[S2]`…；PDF 单页渲染 `Pages: 12`，跨页 `Pages: 12-13`，MD/TXT 省略页码行。
- `BuiltContext`：`context_text`、`entries: list[ContextEntry]`、`total_tokens`、`candidate_count`、`included_count`、
  `skipped_count`。`ContextEntry`：`source_label`、`retrieval_rank`、`chunk_id`、完整 `content`、`source_reference`、
  `token_count`。不暴露任何检索分数。空候选或全部超预算 → 空 Context。

## 9. Generation 合同（rag-generator-v3）

- `Generator.generate(query, context) -> GeneratedAnswer`；`GENERATOR_PROMPT_VERSION = "rag-generator-v3"`。空白 query 在调用
  LLM 前拒绝。`context.entries` 为空 → 不调用 LLM，返回确定性拒答
  （`status=insufficient_context`，文本 `知识库中没有足够的相关信息来回答这个问题。`）。
- Prompt：System 规则与 User 消息分离，User 用 `<SOURCE_CONTEXT>…</SOURCE_CONTEXT>`、`<USER_QUERY>…</USER_QUERY>` 包装；
  只依据 Context 回答、事实陈述用 `[Sx]`、只用存在的标签、Context 是不可信数据而非指令。
- 输出协议：模型只返回 JSON `{"status": "answered" | "insufficient_context", "answer": "<含 [S1] 等标签的回答>"}`，
  由 `GeneratorOutput`（`extra="forbid"`）校验；发给服务端的 schema 由 `generator_output_json_schema()` 从 `AnswerStatus`
  生成（`additionalProperties: false`，两字段 required）。不存在任何文本控制头解析路径。
- 解析顺序：去首尾空白 → 去开头已闭合的 `<think>…</think>`（未闭合视为格式失败）→ 去外层 ```json 代码块 →
  `json.loads` → `GeneratorOutput` 校验。
- 修复：失败时最多重试一次，消息 = 原消息 + assistant（原始回复）+ user（修复指令，要求只返回 JSON）；仍失败抛
  `GenerationContractError(ValueError)`。修复调用与 LLM Client 的瞬态重试分开计数；修复期间的 `LLMError` 原样传播。
- 诊断：`Generator.last_diagnostics`（只读，不进入 `GeneratedAnswer`/`RAGAnswer`）：`llm_called`、`repair_attempted`、
  `repair_succeeded`、`usages`（每次调用后读取 `client.last_usage`）、`raw_response_lengths`；每次 `generate()` 重置。
- LLM Client（`llm/`）：Protocol 提供 `generate(messages) -> str`、`generate_json(messages, *, schema_name, json_schema) -> str`、
  只读 `last_usage: dict | None`（`prompt_tokens`、`completion_tokens`）。`OpenAICompatibleLLMClient` 用 Chat Completions；
  结构化模式为模块常量 `STRUCTURED_OUTPUT_MODE = "json_object"`（依据 `artifacts/stage_a/A2_probe.json`：deepseek-flash 对
  `json_schema` 返回 400），无运行时降级或自动探测，换模型/服务须重新探测。`llm_thinking` 为 `enabled/disabled` 时经
  `extra_body={"thinking": {"type": ...}}` 发送，`auto` 不发送。构造参数 `structured_output_mode` 只供探测脚本使用。
- 重试与错误：仅 timeout、连接错误、429、provider 5xx 可在 `llm_max_attempts`（≤2）内再试一次；401/403、其他 4xx、配置错误
  不重试。API Key 缺失/空 Base URL/空模型 → `LLMConfigurationError`；请求失败或空文本 → `LLMRequestError`（保留异常链）。
  不得打印或记录 API Key、Authorization Header 或完整请求体。

## 10. Citation 合同

- 语法：只识别完整方括号 `[S<正整数>]`；`[S0]` 识别为候选但必然无效；`[S-1]`、`[Sx]`、`[S 1]` 不视为 Citation。
- `parse_citations(answer_text, built_context) -> CitationParseResult`；允许标签只来自 `BuiltContext.entries[*].source_label`，
  未知标签不得丢弃、猜测或替换。
- `assemble_rag_answer(query, generated_answer, built_context, retrieved_chunks) -> RAGAnswer`：`status` 从
  `GeneratedAnswer.status` 透传；`answer` 与 `answer_text` 完全一致；`sources` 按首次引用顺序映射已有 `SourceReference`，
  只含实际引用的来源。
- 规则：`answered` 必须至少一个合法 Citation；`insufficient_context` 允许无 Citation（`sources=[]`）；任何状态出现未知标签
  都抛 `CitationValidationError`。只校验标签存在，不判断语义支持。
- `RAGService.answer(query)`：Retrieve → Context → Generate → Assemble；不重试、不吞异常、不维护第二套拒答文案。

## 11. Settings 一览（`src/evidenceflow/config.py`）

环境变量大小写不敏感，读取 `.env`（UTF-8），未知变量忽略。

| 字段 | 默认值 | 含义 / 约束 |
|---|---|---|
| `llm_api_key` | `None` | LLM API Key，只从环境读取 |
| `llm_base_url` | `https://api.deepseek.com` | OpenAI-compatible 基础地址；非空白 |
| `llm_model` | `deepseek-flash` | 模型名；非空白 |
| `llm_temperature` | `0.0` | 0–2 |
| `llm_timeout_seconds` | `60.0` | 单次请求超时，>0 |
| `llm_max_attempts` | `2` | 1–2；仅瞬态错误可第二次尝试 |
| `llm_thinking` | `disabled` | `auto` / `enabled` / `disabled`；非 DeepSeek 服务须设 `auto` |
| `embedding_model` | `BAAI/bge-m3` | 文档与 query 共用 |
| `source_docs_path` | `data/source_docs` | 稳定 ingestion root |
| `chunk_size` | `512` | >0，按 Embedding tokenizer 计 |
| `chunk_overlap` | `64` | ≥0 且 < chunk_size；仅 fixed 策略使用 |
| `chunk_strategy` | `hybrid` | `hybrid` / `fixed`（回滚通道） |
| `max_context_tokens` | `4096` | Context 预算，>0 |
| `reranker_model` | `BAAI/bge-reranker-base` | 模型名或本地路径；非空白 |
| `reranker_input_k` | `20` | Dense 候选数，>0 |
| `reranker_final_k` | `5` | 最终 Top-K，>0 且 ≤ reranker_input_k |
| `reranker_batch_size` | `4` | >0 |
| `reranker_max_length` | `512` | >0 |
| `model_device` | `auto` | `auto` / `cpu` / `cuda` |
| `reranker_dtype` | `float32` | 只允许 float32 |
| `vector_db_path` | `data/chroma` | Chroma 持久化目录 |
| `collection_name` | `ef_research_bge_m3_hybrid_c512_v2` | 仅无状态文件时的回落值；hybrid 下不得为 fixed 身份 collection |
| `index_state_path` | `artifacts/phase15a/index_state.json` | lifecycle 状态文件 |

- **LLM 默认值是用户本人的决定（2026-10-14 确认）**：默认 DeepSeek（`https://api.deepseek.com` / `deepseek-flash`），
  `llm_thinking` 默认 `disabled`。使用其他服务时必须同时覆盖这三项（尤其 `LLM_THINKING=auto`，否则可能返回 400）。
  `.env.example` 与之一致；其中 `COLLECTION_NAME` 已注释，通常不应设置。
- 实验参数不得进入全局 Settings。

## 12. CLI 与错误处理

### 12.1 `python -m evidenceflow`（`src/evidenceflow/__main__.py`）
- `ingest [path]`：省略 path 用 `source_docs_path`；输出逐文件状态与汇总。退出码 0 = 全部成功；1 = 有文件失败，或装配/执行
  异常（stderr `Ingestion failed: <异常类型>: <消息>`）；2 = ACTIVE 保护拒绝（stderr `Ingestion refused: <消息>`，未改动索引）。
- `ask "<query>"`：输出 `Question:`、`Status: <answered|insufficient_context>`、`Answer:`、`Sources:`（无引用时
  `No cited sources.`；不显示 document_id、chunk_id 或分数）。退出码 0 = 成功；1 = 任何异常，stderr 一行
  `<前缀>: <消息>`，不输出未经校验的回答。前缀：`ActiveCollectionError` → `Index configuration error`；
  `LLMConfigurationError` → `LLM configuration error`；`LLMRequestError` → `LLM request failed`；
  `GenerationContractError` → `Generation format error`；`CitationValidationError` → `Citation validation failed`；
  其他 → `Retrieval or RAG processing failed`。`KeyboardInterrupt` 不被转换。
- 参数错误由 argparse 处理（退出码 2）。

### 12.2 `scripts/build_staging_index.py`
- 构建模式：0 = 校验通过且所选动作完成；1 = 构建或校验失败（staging 与指针保留）；2 = 拒绝开始，未构建、未删除、未改状态
  文件（已有 staging 指针、生成名与 active/previous 或已有 collection 重名、参数冲突），stderr `REFUSED: <消息>`；
  3 = 校验已通过但构建后的收尾动作被保护拒绝（`ProtectedCollectionError`，只可能因状态文件在构建期间被外部改动；staging
  与指针保留）。
- `--discard-staging`：0 = 删除并清指针完成；2 = 拒绝且未改动——状态文件不存在/读不了/校验失败（stderr
  `REFUSED: cannot read lifecycle state <path>: <异常类型>: <消息>`）、没有登记的 staging、目标是 active/previous、与构建选项
  组合（`REFUSED: …`）；1 = 连接 Chroma、列出或删除时出错（stderr `FAILED: <异常类型>: <消息>`，删除失败时指针不变）。
  错误都只输出一行，不抛 traceback。

### 12.3 `scripts/prune_chroma_collections.py`
- 0 = 成功（dry-run 或删除完成）；1 = Chroma 操作失败（stderr `FAILED: <异常类型>: <消息>`）；2 = 拒绝执行，未连接 Chroma、
  未改动任何东西（输出文件已存在且未传 `--overwrite`、允许名单与禁止集合有交集、状态文件不可读；stderr `REFUSED: <消息>`）。

### 12.4 其他
- `scripts/verify_active_index.py`：只读完整性 smoke（存在性、计数、metadata 覆盖率、`iter_all_chunks` 恢复、样本查询、
  cosine 与 chunk_strategy 身份）。不创建 Collection（以 `create_if_missing=False` 打开）；持久化目录不存在时直接判失败、
  不创建目录。默认只打印 JSON 到 stdout、不写文件；`--output PATH` 时另写文件。0 = 通过，2 = 有检查失败。
- `scripts/verify_phase15a.py` 默认写 `artifacts/phase15a/phase15a_report.json`（被跟踪），运行时应传 `--output`。

### 12.5 RunTrace（Stage B 新增；B2 实现：`domain/trace.py`、`tracing.py`）
- Domain Model `domain/trace.py`（Pydantic，`extra="forbid"`）：`RunTrace{schema_version="run-trace-v1", run_id,
  query_sha256（UTF-8 query 的 sha256）, started_at_utc, completed_at_utc（ISO 8601 UTC）, status: completed|failed,
  events: list[TraceEvent]}`；`TraceEvent{seq（0 起连续）, name, elapsed_ms（相对 run 开始）, duration_ms?, payload: dict}`。
  `run_id` 由调用方传入，否则 uuid4。trace 不进入 `RAGAnswer` 或任何现有 Domain Model。
- 事件名（UI 合同 v0 taxonomy 的子集，不用 lexical / pool / blocked）：`run.started`、`retrieval.dense.started/completed`、
  `rerank.started/completed`、`context.started/completed`、`generation.started/completed`、`citation_validation.completed`、
  `run.completed`、`run.failed`。
- payload 白名单（未列出的事件 payload 为 `{}`）：
  - `retrieval.dense.completed`：`candidate_count`；`candidates: [{rank, chunk_id, similarity_score}]`（按 rank）。
  - `rerank.completed`：`input_count`、`output_count`、`truncated_pair_count`（整批一个值）；
    `results: [{rank, chunk_id, candidate_rank, reranker_score}]`（按重排后 rank）。
  - `context.completed`：`included_count`、`skipped_count`、`total_tokens`、`entries: [{source_label, chunk_id, token_count}]`、
    `context_text_sha256`（UTF-8 `context_text` 的 sha256）。
  - `generation.completed`：`status`、`llm_called`、`repair_attempted`、`repair_succeeded`、`usages`（`prompt_tokens`、
    `completion_tokens`）、`raw_response_lengths`、`prompt_version`（取自 `Generator.last_diagnostics` 与 §9 常量）。
  - `citation_validation.completed`：`valid_labels`、`invalid_labels`、`source_count`。`run.failed`：`error_type`（异常类名）、`stage`（失败事件所属阶段名）。
  - 禁止出现：上下文全文、chunk 正文、Prompt 全文、回答全文、API Key、模型思考内容、异常消息全文。
- `TraceRecorder` 协议（`evidenceflow.tracing`，不 import `evidenceflow.eval`）：`record(name, **payload)`。
  `DenseRerankerRetriever`、`Generator`、`RAGService` 的 `trace_recorder` 参数默认 `None`；为 `None` 时不构造任何 payload，
  开销与当前一致。`NullTraceRecorder`（不做任何事）保留，供调用方显式使用。组件统一经 `safe_record(recorder, name, **payload)`
  记录：recorder 为 `None` 直接返回；`record` 抛出的任何异常只以 `logger.warning` 记录异常类名后吞掉。
- `RunTraceRecorder`（实现 `record`）：`start_run(query, *, run_id=None)` 重置事件，记录 `query_sha256`、`started_at_utc`
  与单调时钟起点；`finish() -> RunTrace`：最后一个事件是 `run.completed` → `completed`，是 `run.failed` → `failed`，否则
  `ValueError`。同一 recorder 可跨 run 复用，每次 run 前调用 `start_run`；`start_run` 之前的 `record` 只记日志并丢弃事件。
- 计时：`elapsed_ms` 用单调时钟，从 `start_run` 起算；`X.completed` 的 `duration_ms` = 与最近一个 `X.started` 的间隔，其余事件
  为 null。时钟可注入（测试用）。
- 记录者：`run.*`、`context.*`、`citation_validation.completed` 由 `RAGService`；`retrieval.dense.*`、`rerank.*` 由
  `DenseRerankerRetriever`；`generation.*` 由 `Generator`（确定性拒答也记 `generation.started/completed`，`llm_called=false`）。
  `citation_validation.completed` 的标签由 `RAGService` 在 `assemble_rag_answer` 成功后以 `parse_citations` 重算（仅 recorder
  非 None 时）。`run.failed.stage ∈ {retrieval, context, generation, citation_validation}`；异常记录后原样 `raise`（不包装、
  不改异常链）。空白 query 校验仍在最前，不记任何事件。
- 装配：`runtime.build_retriever(settings, *, trace_recorder=None)`；`runtime.build_rag_service(settings, *, retriever=None,
  generator=None, trace_recorder=None)` 把 recorder 传给它自己装配的 retriever 与 generator 以及 `RAGService`；调用方自行注入的
  retriever / generator **不会**被自动挂上 recorder。
- 不变式：开/关 trace 时 `RAGAnswer` 逐字段相等；recorder 内部异常只记日志，不得改变回答或异常传播。
- CLI `ask --trace PATH`：创建 `RunTraceRecorder` 并 `start_run(query)`，在 `finally` 中 `finish()` 后写 UTF-8、`sort_keys=True`、
  `ensure_ascii=False`、`indent=2` 的 JSON；失败时也写出含 `run.failed` 的 trace（装配阶段失败时没有 run.* 终止事件，
  `finish()` 无法成立，此时不写文件，只在 stderr 警告一行）；trace 写入失败只在 stderr 警告一行；trace 不打印到 stdout；
  退出码语义与 stdout/stderr 原有内容不变（§12.1），`KeyboardInterrupt` 不被转换。

## 13. Evaluation

- Canonical 实现：`eval/ir_metrics.py`（标准 IR）、`eval/quality.py`（基于显式人工/校准标注的答案、引用、拒答、Context 指标）、
  `eval/comparison.py`（配对比较、Holm 校正）、`eval/stable_mapping.py`（Gold anchor → chunk 映射）。各 runner 不得复制公式。
  `eval/metrics.py`、`eval/dataset.py`、`domain/evaluation.py` 为 COMPATIBILITY_ONLY（legacy HitRate 语义）。
- 标准指标（evaluation_version 3.0，二元相关、TREC/BEIR 语义）：Precision@K、Recall@K（分母为相关总数）、MRR@K、MAP@K、
  nDCG@K（增益 `1/log2(rank+1)`，IDCG 截断到 `min(|relevant|, K)`）。排名单位 `chunk_id`，查询单位 `question_id`。
  语义由 `tests/test_ir_metrics.py` 的确定性 fixture 固定；不得创造新的 IR 指标名（端到端指标名只由 §13.1 定义）。
- 相关性：canonical qrels 来自冻结 Gold mapping（仅 FULLY_MAPPED / FRAGMENTED_ACROSS_CHUNKS）。当前 hybrid 索引使用冻结
  v2 Gold anchor 对 active chunk 文本的只读投影（`active_relevance_projection_hash`），比较时必须一致。无法映射的题计入
  `question_count` 但不进入均值（40 题，eligible 38）。
- Gate 规则：Candidate Gate 主 Recall@20、辅 Recall@10；Ranking Gate 主 nDCG@5、MRR@5、MAP@5，辅 Precision@5、Recall@5；
  Multi-evidence 诊断（EvidenceGroupRecall@5、AllRequiredEvidence@5）只作回归保护，标准指标提升但它明显退化时不得晋升；
  性能报告 p50/p95/throughput。比较按 question_id 配对；实验 runner 不得自动修改 Runtime。
- **冻结基线 B0'**（Exact Dense20 → Reranker → Top5，CPU 回放）：`artifacts/stage_a/A1_exact_baseline.json`，
  sha256 `422fd629b22f26caf9452e3598bf805d01df9b9cf24e6d42b5a6890e40a1411f`（`A1_exact_baseline.sha256`）。

  | P@5 | R@5 | MRR@5 | MAP@5 | nDCG@5 |
  |---|---|---|---|---|
  | 0.2684 | 0.5103 | 0.5175 | 0.3631 | 0.4903 |

  与此前 HNSW B0（`evals/evaluation_v3/phase15b_hybrid_candidate_v2/production_reranked_baseline.json`）40 题 Top5 逐题一致。
  线上回放用 `experiments/stage_a/replay_runtime_retrieval.py`（走 `build_retriever`），与 B0' 比较用
  `experiments/stage_a/a3_compare_replay.py`。
- `evals/` 下全部为冻结产物，不得覆盖或修改；Gold、qrels 只能经独立版本化流程变更。

### 13.1 端到端评测 rag_e2e_v1（Stage B 新增；已定合同，B3/B4/B5b 实现，当前代码尚无）
**(a) 题集 schema（B3 实现）**：`EvalCase`（`extra="forbid"`）：`case_id`（新题 `E2E-M*` / `E2E-U*`）、`family_id`、`query`、
`query_language`、`question_type`、`tags`、`critical: bool`、`expected_status ∈ AnswerStatus`、`acceptable_abstention: bool`、
`required_evidence_groups: list[EvidenceGroup]`、`required_claims`、`forbidden_claims`（错误前提题必填；每条
`claim_id`、`text`、`evidence_anchors`＝与之矛盾的原文锚点）、`unanswerable_kind ∈ {out_of_corpus, near_miss, false_premise} | null`、
`provenance{drafted_by, approved_by, source}`。`required_claims[*]`：`claim_id`、`text`、`required: bool`、
`evidence_anchors: list[EvidenceAlternative]`（复用 `domain/evaluation_v2.py`，与 40 题同一锚点协议）、`answer_keys: list[list[str]]`
（外层组间“与”，组内别名间“或”；空列表 ⇒ `key_matchable=false`）。`acceptable_abstention` 只对 `expected_status=answered`
的题有意义，表示这道可答题允许拒答；不可答题必须为 `false`（B3 校验）（D7）。**状态正确** ⇔ 实际状态 = 期望状态；或期望为
answered、实际为 insufficient_context 且 `acceptable_abstention=true`。`fingerprint` = 除 `fingerprint` 字段外整个数据集的规范化 JSON
（`sort_keys`、`ensure_ascii=False`、`separators=(",", ":")`、UTF-8）的 sha256；`dataset.sha256` = 文件字节 sha256，两者都记录。

**(b) 证据映射与题集分离（B3 实现）**：题集只存锚点；对某索引的映射是独立文件 `evals/rag_e2e_v1/mapping_<collection>.json`，
记录 `index_signature`、`collection_name`、`dataset_fingerprint`，以及每个证据组和每条 claim 锚点的映射状态与 chunk 集合。状态取
`stable_mapping` 现有 `MappingStatus`；`FULLY_MAPPED` / `FRAGMENTED_ACROSS_CHUNKS` 为可用，其余状态统称 UNMAPPABLE。
`score` 发现 mapping 与 observations 的三项身份任一不一致即拒绝计分。换索引只重新生成 mapping，不改题集。

**(c) answer_keys 校验（B3，确定性；任一不通过该题不得冻结）**：① 每个要素组至少一个别名经 (d) 规范化后出现在该 claim
某个锚点的 locator 页源文本中（MD/TXT 为整篇；中文别名可只作补充）；② 任何别名按 (d) 匹配规则不得命中题干 `query`；
③ 别名去首尾空白非空、规范化后长度 ≥ 2，组内去重。可答题中 key_matchable required claim 的比例写进 B3 报告，不设硬下限，
低于 60% 须说明原因。

**(d) 关键要素匹配 `key_match`（纯函数，B4 实现）**：规范化 = NFKC → casefold → 连续空白压成一个空格 → 去掉 `[S\d+]` 引用
标签；别名与回答同样规范化后做子串匹配；别名首/尾字符为 ASCII 字母数字时，命中位置两侧不得是 ASCII 字母数字（防 `1.5`
命中 `11.5`、`ViT` 命中 `ViTs`），CJK 字符不要求边界。不做数字解析、单位换算或模糊匹配，不同写法（`0.5` / `50%`）只能列为
显式别名。claim 命中 ⇔ 所有要素组都命中。

**(e) 指标（B4 实现）**：每个指标给出分子、分母，空分母返回 null，带 `status ∈ {primary, diagnostic_uncalibrated, unavailable}`。
observation 带 `GenerationContractError` / `CitationValidationError` / `LLMRequestError` 的题从所有回答类与状态类指标的分母中
排除并单独计数；上下文层指标对所有记录了 `context.completed` 的题计算（含这些错误题）。“可答题”指
`expected_status=answered`，“不可答题”指 `insufficient_context`。
- 上下文层（primary）：`context_all_required` = 所有必需证据组都至少有一个 chunk 集合完整进入上下文的题数 / 可答且无
  UNMAPPABLE 组的题数；`context_claim_recall` = 至少一个锚点的 chunk 集合完整进入上下文的 required claim 数 / 有可用锚点的
  required claim 数（UNMAPPABLE 锚点不计入）。
- 状态层（primary）：期望 × 实际状态混淆矩阵（按原始期望状态与实际状态统计，不受 `acceptable_abstention` 影响）；
  `unanswerable_false_answer_rate` = 实际 answered / 不可答题；`answerable_false_abstention_rate` = 实际 insufficient_context 且
  `acceptable_abstention=false` 的可答题数 / 可答题数；另报 `answerable_abstention_rate_raw` = 实际 insufficient_context 的可答题数 /
  可答题数（不扣除可接受拒答）；`status_body_inconsistent_count` = status 为
  insufficient_context 但 `sources` 非空（至少一个合法引用）的题数（D4）。
- 关键要素（primary，只算 `expected_status=answered` 且至少有一条 key_matchable required claim 的题）：`key_fact_recall`（macro）
  = 各题“命中的 key_matchable required claim 数 / key_matchable required claim 数”的平均，配对比较用它；另报 micro =
  总命中数 / 总数。D4：实际 status 为 insufficient_context 时正文照常匹配。
- 准确率与精确率（primary，PaperQA2 口径）：答对 ⇔ 实际 answered 且全部 key_matchable required claim 命中；
  `answer_accuracy` = 答对数 / 可答题数；`answer_precision` = 答对数 / 实际 answered 的可答题数；两数一起报告；无
  key_matchable claim 的题不进入任何分母；`acceptable_abstention` 不改变这两个指标的分子分母（它们衡量的是回答内容）。
- 引用格式（primary）：合法引用数；`cited_source_in_evidence_rate` = 可答题 `sources` 中 chunk 属于该题证据映射 chunk 集合并集
  的条数 / 可答题 `sources` 总条数。
- 评委（`diagnostic_uncalibrated`；对应任务 B6 未通过时为 `unavailable`）：评委 claim recall = COVERED / 已判 required claim；
  groundedness = SUPPORTED / 已判陈述，无依据论断率 = UNSUPPORTED / 已判陈述；引用支持精确率 = SUPPORTS / 已判引用对；
  另报评委 COVERED 与 key_match 命中在 key_matchable required claim 上的逐条一致率。
- 成本（primary，只记录、不设合格线）：LLM 调用次数、prompt/completion token、端到端与各阶段 p50/p95 延迟（来自 RunTrace）。
- 失败层归因（逐题按下列顺序取第一条成立的规则，都不成立记 `PASS`）：
  1. 错误题：`LLMRequestError` → `ERROR`（不归因）；`GenerationContractError` → `GENERATION`；`CitationValidationError` → `CITATION`。
  2. 不可答题：实际 answered → `STATUS`，否则 → `PASS`。以下规则只用于可答题。
  3. 含 UNMAPPABLE 证据组的可答题 → `UNATTRIBUTABLE`。
  4. `CANDIDATE_MISS`：某必需证据组没有任何 chunk 集合完整落在 Dense Top20 候选中（候选取自 RunTrace 的
     `retrieval.dense.completed`）。
  5. `RERANK_MISS`：各组都在 Top20 中，但某组没有任何 chunk 集合完整落在最终 Top-K（`retrieved_chunks`）中。
  6. `CONTEXT`：各组都在 Top-K 中，但 `context_all_required` 不成立。
  7. `STATUS`：`context_all_required` 成立，但状态不正确（按 (a) 定义）。实际 insufficient_context 且
     `acceptable_abstention=true` 时状态算正确，本条不成立，直接记 `PASS`，不再往下判断。
  8. `UNVERIFIED`：实际 answered，但该题没有任何 key_matchable required claim，生成层无法确定性判定。
  9. `GENERATION`：实际 answered，但有 key_matchable required claim 未命中。
  10. `CITATION`：已答对（按本节定义），但 `sources` 中没有任何 chunk 属于证据映射 chunk 集合。
  - `CANDIDATE_MISS` / `RERANK_MISS` 分别对应 UI 合同 v0（`docs/EVIDENCEFLOW_UI_RUNTIME_CONTRACT_v0.md`）Failure Attribution
    的 `CANDIDATE_MISS` / `RERANKER_MISS`。报告中 `UNVERIFIED` 单独计数，不并入 `PASS`。
- 合同变更：`eval/quality.py::abstention_metrics` 改为接受 `AnswerStatus` 小写取值与字面量 `"error"`，大写取值报错；B4 实现并
  同步修改 `tests/test_evaluation_v2.py` 中的调用。

**(f) 受控上下文检验（B5b，生成器层，不是端到端）**：
- 输入：B5 observations 中 `expected_status=answered`、原上下文至少含一个证据映射 chunk 的题。原上下文按 chunk ID 从 active
  索引只读复原（`create_if_missing=False`），按原 `source_label` 顺序作为 rank 用 `ContextBuilder` 重建 `BuiltContext`，并用
  `context_text_sha256` 校验；再调用 `Generator.generate` + `assemble_rag_answer`。不经过检索，不改线上任何组件。
- `evidence_ablation`：删除该题所有证据组与 claim 锚点映射到的 chunk，剩余 chunk 按原 rank 用 `ContextBuilder` 重新编号。
  有效性（任一不满足 → `INVALID`，不调用 LLM，只列出）：剩余上下文经 `stable_mapping` 规范化后不含任何锚点原文；任何
  key_matchable required claim 都不在剩余上下文中命中（按 (d)）。剩余为空 → `trivially_empty`（§9 确定性拒答），单独计数，
  不进比率。预期判定 `insufficient_context`。`ablation_false_answer_rate` = 实际 answered 的有效非空变体 / 有效非空变体；
  另报 D4 不一致数。
- `counterfactual_number`：选题条件——某 required claim 的某要素组只有 1 个别名、该别名匹配 `^\d+(\.\d+)?%?$`，且按 (d)
  出现在上下文中；每题按 claim、组顺序取第一个符合者，至多一个变体。替换规则：有小数则最后一位小数 +1（9 变 0 并进位，
  位数不变）；整数 ≥ 10 则 +1；< 10 的整数不选；新值若出现在原上下文中则再应用一次，最多 3 次，仍冲突 → `INVALID`。
  在 chunk 正文中按 (d) 的 ASCII 边界替换该别名的全部出现，再用 `ContextBuilder` 重建；纳入的 chunk 集合变化或替换后仍
  出现原值 → `INVALID`。判定（对回答做 `key_match`，按序）：status 为 insufficient_context → `abstained`；只含新值 →
  `context_faithful`；只含原值 → `parametric_override`；两者都含 → `mixed`；都不含 → `not_mentioned`。
  `counterfactual_faithfulness_rate` = faithful / (faithful + parametric_override + mixed)，其余各类单独报告。
- 两个变体指标都是 primary（标签由构造决定）。已知局限：消融有效性检查只排除原文和要素的重复出现，不排除同义改写，
  `ablation_false_answer_rate` 偏高方向可能含标签噪声；逐题明细必须保留供抽查。
- 产物 `artifacts/stage_b/B5b_controlled.jsonl`（每变体一行）：`case_id`、`variant`、`validity`、`expected`、构造记录（删除的
  chunk、替换的值）、新上下文 sha256、生成结果、诊断、错误类型。`score` 读取它，输出到 E0 报告独立一节“受控上下文（生成器层）”。

**(g) 评委协议（B4 实现）**：评委只做逐条二元/三元判定，不打总分；每次输出 `{"judgments": [...]}`，条目 ID 集合须与输入
完全一致，每条带 `verdict` 与简短 `reason`。
- `claim_coverage`（`judge-claim-v1`）：输入 query、回答全文、`claims[{claim_id, text}]`；输出 `{claim_id, verdict ∈ {COVERED,
  MISSING, CONTRADICTED}, reason}`。
- `groundedness`（`judge-ground-v1`）：输入 query、`BuiltContext.context_text`、回答全文；先把回答拆成不含代词、可独立理解的
  陈述，再逐条输出 `{statement_id, statement, verdict ∈ {SUPPORTED, UNSUPPORTED}, reason}`。
- `citation_support`（`judge-cite-v1`）：输入 `pairs[{pair_id, sentence, source_label, source_content}]`（回答句与其中合法标签
  对应来源，确定性拆分）；输出 `{pair_id, verdict ∈ {SUPPORTS, NOT_SUPPORTS}, reason}`。
- 结构化模式固定 `json_object`（依据 `artifacts/stage_b/B0_judge_probe.json`）；该模式不把 schema 发给服务端，prompt 必须写全字段名
  与枚举取值。解析顺序同 §9：去首尾空白 → 去开头 `<think>` → 去外层 ```json 代码块 → `json.loads` → Pydantic `extra="forbid"`；
  失败修复一次，仍失败记 `judge_parse_error`，该条不进分子也不进分母，单独计数。
- 配置：`EVAL_JUDGE_BASE_URL` / `EVAL_JUDGE_MODEL` / `EVAL_JUDGE_API_KEY` 只在评测脚本中读取，不进全局 Settings；temperature 0；
  评委与生成器 base URL 注册域相同时拒绝运行。落盘：prompt 版本、模型、temperature、原始回复、解析结果、usage、耗时。
- 评委看到的上下文是由 observation 复原并经 sha256 校验的 `BuiltContext`；prompt 不含被评系统的名字或版本。

**(h) Gate 策略文件 `evals/rag_e2e_v1/gates_v1.json`（B7 写，B4 实现加载与判定）**：每个指标声明 `metric`、
`direction ∈ {higher_better, lower_better}`、`min?`、`max?`、`max_regression?`、`required: bool`；`critical_cases: list[case_id]`
中各题的逐题确定性结果（状态正确，按 (a) 定义；可答题另加 `context_all_required`、关键要素全部命中）不得由通过变为不通过。只有
`status=primary` 的指标可写进 Gate，写入 `diagnostic_uncalibrated` 指标时加载阶段直接报错。

**(i) 评委检验合格线与指标地位（B6）**：每类判定任务分别判断：构造负例检出率 ≥ 0.9；未破坏样本误报率 ≤ 0.1；复测一致率
≥ 0.95；与 Phase 12 agent 标注的一致率与 κ 只报告不设线。**评委指标未经人工校准，只作诊断，不作为任何阶段的晋升 Gate。**
阶段 D 的“Claim Recall 提升”以 `key_fact_recall` 为 primary 依据，评委 claim recall 只作诊断旁证（最终以阶段 D 计划为准）。

## 14. 不在当前范围内

- 见 `docs/agentic_rag/01_ARCHITECTURE_REVIEW.md` §8：自由 ReAct 工具调用、Multi-Agent、逐 chunk LLM 相关性打分、重新引入
  BM25、HITL 澄清、长期/向量记忆、现在接 Web Search、现在引入 LangGraph。
- 已退休、不得作为当前能力描述或恢复的：查询翻译、多语双路检索、BM25 生产分支、RRF 融合、以 Chroma HNSW 作为线上候选检索、
  v2 文本控制头协议、`reranker_enabled` / `top_k` / `multilingual_*` 配置。历史源码可从 tag `stage-a-start` 恢复，证据在 `evals/`。
- 阶段 B 已由用户确认进入（2026-10-15）：RunTrace（§12.5）与端到端评测 `rag_e2e_v1`（§13.1）属于阶段 B 范围。
  阶段 C–E（Planner、Grader、补检、PARTIAL、文档过滤、逐句引用校验）仍不在范围内，只有在用户明确进入对应阶段并更新
  本文件后才实现。

### 14.1 已知限制与待决（Stage A 结束时）
1. 激活后旧 active 的 metadata 仍为 `lifecycle=ACTIVE`，rollback 依赖这一点，可能同时有多个 collection 标为 ACTIVE（§6.7）。
2. `artifacts/phase15a/index_state.json` 被 git 跟踪；activate/rollback 之后不要 `git restore`/`checkout` 它，有未提交指针
   变更时不要切换分支（§6.7）。
3. 被 `mark_collection_lifecycle` 改写过的 collection 不再有 `hnsw:space`；以后改写 v2 时须同步 `verify_phase15a.py` 的快照比对。
4. `--discard-staging` 在没有登记 staging 时返回 2。
5. Chroma `delete_collection` 留下的 HNSW 段目录需按 §6.6 手动清理；`chroma.sqlite3` 是否 `VACUUM` 未决。
6. `abstention_metrics` 的状态取值与部分回答（D4）计分：由 §13.1(e) 定义，B4 实现（当前代码仍只接受大写状态）。
7. 部分保留的实验脚本仍读 `Settings().collection_name`，不跟随 active 指针（见 `artifacts/stage_a/A4_report.md` §4）。

## 15. 变更流程

- 与 AGENTS.md 一致：每次只做用户指定的当前步骤；涉及数据模型、核心接口、ID 规则、配置定义、Evaluation 规则或范围的改动，
  先更新本文件再改代码，并同步测试。公共接口变更须先说明原因与影响。
- 本文件只保留当前合同；被取代的内容移入 `docs/history/`，不在此处累积历史。
- 阶段计划与审查（`docs/agentic_rag/`）是阶段性文档；其中被采纳的合同以本文件为准。

## 16. 历史索引

v0.26 完整原文：`docs/history/PROJECT_SPEC_v0.26_full.md`（下表为其中的旧章节号）。

| 旧章节 | 主题 | 当前对应 |
|---|---|---|
| §S.1–§S.3 | Stage A 合同变更（A1–A3.2） | 已并入 §6、§7、§9、§11、§12、§13 |
| Current Architecture Summary、§3、§37.1 | 架构 | §1 |
| §1 | 开发环境（旧 `agent` 环境） | §2 |
| §5、§32 | Document / Domain Models | §3、§5 |
| §6、§10 | document_id、content_hash、chunk_id | §4 |
| §7–§9 | 重复导入、清洗、切分、身份隔离 | §5 |
| §11、§14 | Chunk Size 实验、Collection 命名 | 历史 |
| §12、§13、§15 | Embedding、Vector Store、Chroma 距离 | §5、§7 |
| §16–§17 | 多语检索、Translation、RRF、干预历史 | 已退休（§14） |
| §18–§20.7 | Gold v1/v2、Phase 3.5–10 评测 | 历史；当前规则见 §13 |
| §20.8 | Evaluation v3 标准 IR 与 Gate | §13 |
| §21–§22 | Context Builder、Source Mapping | §8 |
| §23–§24 | LLM Client、Generation v2 | §9（v3） |
| §25–§27 | Citation、SourceReference、RAGAnswer | §3、§10 |
| §28 | Settings | §11 |
| §29–§31 | CLI、RAG Service、Logging、Error Handling | §7、§10、§12 |
| §33–§36 | 目录建议、开发顺序、Step 1、Phase 1 完成标准 | 历史 |
| §37–§38 | Phase 11B/11C 清理、Hygiene Contract | 历史；索引身份见 §5、§6 |
| §39 | Phase 15A Manifest / Lifecycle | §4、§6 |
| §40–§41 | Phase 15B v2、15B.5 收敛 | §13、§14 |
| §42 | Exact Dense 架构 | §7 |
| §43 | 共享推理设备 | §7 |
| —（Stage B 新增，v0.28） | RunTrace；端到端评测 rag_e2e_v1、answer_keys / key_match、受控上下文检验、评委协议、Gate 策略文件 | §12.5、§13.1 |
