# A3 报告：索引生命周期接到线上

## 结论：Gate A3 通过

| Gate | 结果 |
|---|---|
| `python -m pytest -q` 全部通过 | **643 passed**（A2 为 609，本步新增 34；`A3_pytest.log`） |
| 真实验证 1：`ask` 冒烟 | 通过：`Status: answered`，runtime 实际打开的 collection 为 `ef_research_bge_m3_hybrid_c512_v2` |
| 真实验证 2：`ingest` 保护 | 通过：保护函数对 active 给出拒绝；`ingest` 退出码 **2**；之后 active 身份与运行前**逐项一致** |
| 真实验证 3：staging 全量重建（不激活） | 通过：817 个 chunk，chunk ID 集合与 active 完全一致；staging collection 已删除，`staging` 指针为 null |
| 真实验证 4：收尾检查 | 通过：collection 仍是那 7 个；active 身份不变；`index_state.json` **字节级一致**（无需 `git restore`） |
| 线上回放 vs B0' | 通过：CPU、`build_retriever`（collection 由 active 指针决定），40 题 Top5 / Top20 候选 ID 与分数 / 精排分数 / 5 项指标全部与 `A1_exact_baseline.json` 一致 |
| `evals/` 无改动 | `git diff -- evals` 为空；`artifacts/phase15a/` 无改动 |

没有触发计划 §7 的任何 STOP 条件。启动前复核计划中的事实（见 §0），全部与实际相符。

## 0. A2 报告修正（与 A3 一并提交）
`artifacts/stage_a/A2_report.md` 已按要求修改：
1. 第 3 节补充：`ARR-040` 的 3 个可接受证据 chunk（`0b3ae8…`、`2f9922…`、`d9b0f2…`）都不在 Top5 中，是检索漏召回而非生成问题。**写入前已逐个核对**：数据集 `acceptable_source_ids` 与冒烟记录的 `top5_chunk_ids`（`b514a8…`、`4ee5db…`、`084937…`、`b8d95a…`、`496f53…`）无交集。
2. 第 5 节新增遗留问题 8、9：“status 为 insufficient_context 但正文给出了部分有引用的回答”怎么计分留给阶段 B 决定；`experiments/archive/phase12_fallback_metrics.py` 也使用大写状态值（A4 会删除）。
3. 第 5 节第 7 条改为：`src/evidenceflow.zip` 是用户自己删除的，不需要再追查。

## 1. 开工前复核的事实
- `artifacts/phase15a/index_state.json`：`active.collection_name = ef_research_bge_m3_hybrid_c512_v2`，`status = ACTIVE`，`staging`、`previous` 均为 null ✔
- active collection metadata 含 `lifecycle: "ACTIVE"`（`A0_verify_active_index.txt`，并由 `A3_active_before.json` 再次确认）✔
- `Settings.collection_name` 默认值为 `ef_bge_m3_hybrid_c512_v1`，已改为 v2 ✔
- `IndexLifecycle` 只有 `initialize_active / prepare_staging / activate / rollback`，没有清空 staging 指针的方法 ✔
- Chroma 中 7 个 collection、chunk_id_hash `e87a613e…`、index_signature `b4e9432c…` 与计划一致 ✔

## 2. 修改或创建的文件
| 文件 | 职责 |
|---|---|
| `PROJECT_SPEC.md` | 新增 §S.3（A3 合同）；§S.2 补充 LLM 默认值修订。**先于代码写入** |
| `src/evidenceflow/config.py` | 新增 `index_state_path`；`collection_name` 默认改为 v2；LLM 默认 `https://api.deepseek.com` / `deepseek-flash` / `disabled`（**用户本人决定，2026-10-14 确认**；与计划 A3 表格中“与 ModelScope/Qwen 一致”的字面要求不同，该偏离已获用户认可，见 PROJECT_SPEC §S.2） |
| `src/evidenceflow/knowledge/lifecycle.py`、`knowledge/__init__.py` | 新增 `ActiveCollectionError`、`ActiveIndexProtectedError`；新增 `IndexLifecycle.discard_staging()` |
| `src/evidenceflow/runtime.py` | 新增 `resolve_active_collection`、`guard_ingest_target`；`build_retriever` / `build_ingestion_service` 都经 `resolve_active_collection` 决定 collection；`build_retriever` 校验 lifecycle metadata |
| `src/evidenceflow/__main__.py` | `ingest` 被保护拒绝时输出提示并返回退出码 2；`ask` 对 `ActiveCollectionError` 渲染 `Index configuration error: …` |
| `scripts/build_staging_index.py` | 新增 staging 构建脚本（见 §5） |
| `.env.example` | LLM 三项与代码默认值一致；`COLLECTION_NAME` 改为注释并说明“通常不需要设置”；新增 `INDEX_STATE_PATH` |
| `experiments/stage_a/replay_runtime_retrieval.py` | 回放脚本的身份校验改用 `resolve_active_collection`，不再读 `Settings.collection_name` |
| `experiments/stage_a/a3_active_identity.py` | 只读身份快照（collection 列表、计数、metadata、chunk_id_hash、内容哈希含向量字节、state / manifest sha256） |
| `experiments/stage_a/a3_compare_replay.py` | 回放与 `A1_exact_baseline.json` 逐题比较（含基线 sha256 校验） |
| `tests/test_runtime_lifecycle.py`、`test_build_staging_index.py`（新）；`test_runtime.py`、`test_config.py`、`test_ingest_cli.py`、`test_phase15a_lifecycle.py`（改） | 新增 / 更新用例（见 §6） |
| `artifacts/stage_a/A2_report.md`、`A3_*` | A2 报告修正；A3 产物 |

## 3. 实现内容与原因
- **active 指针解析**：状态文件存在时以 `state.active.collection_name` 为准；`COLLECTION_NAME` 只有在**显式设置**（`"collection_name" in settings.model_fields_set`，含环境变量、`.env`、构造参数）且与 active 不一致时才报错；一致则放行。状态文件损坏时拒绝启动，不回落。无状态文件时回落到 `Settings.collection_name`。
- **ingest 保护**：保护检查在构造任何模型 / Chroma 之前触发（有测试证明构造器一个都没被调用）。
- **`discard_staging()` 的设计理由**：计划允许“使用现有 API，或手动处理”。现有四个方法都做不到清空 staging 指针，手工编辑 `index_state.json` 容易出错，所以加这个最小方法：只把 `staging` 置 null，不动 active / previous；沿用 `_write_json_atomic`；staging 本来为 null 时直接返回、**不重写文件**；**不删除 Chroma collection**——删除由调用方通过 Chroma API 完成，调用方必须先确认它不是 active 或 previous（脚本里的 `discard_staging_collection` 就是这样做的，且先删 collection、后清指针）。
- **IngestionService 的“先删后写”未改动**：ACTIVE 不允许 ingest，staging 是全新的空 collection，该分支不会被触发。

## 4. 真实验证（顺序与计划一致，pytest 全部通过后才开始）
### 第 0 步：记录 active 身份 → `A3_active_before.json`
count 817；chunk_id_hash `e87a613ebe65…f153d`；index_signature `b4e9432c7170…739e5`；内容哈希（id+文本+metadata+float32 向量）`615d09ff407b…`；metadata `{chunk_strategy: hybrid, hnsw:space: cosine, lifecycle: ACTIVE, storage_contract: domain_chunk_metadata_v2}`；`index_state.json` sha256 `5fbad4f0…`；`active_manifest.json` sha256 `04c9729a…`。

### 第 1 步：`ask` 冒烟（1 次 LLM 请求）
命令：`python -m evidenceflow ask "PoinTr 如何表示用于 Transformer 处理的点云？"`（`MODEL_DEVICE=cpu`、离线 HF、本地 reranker 目录）。
- 为了证明“用的是 v2”，我用 `runpy.run_module("evidenceflow", run_name="__main__")` 执行了与上面命令**相同的 CLI 入口**，只在外面套了一个观察者，把 runtime 打开的 collection 名打到 stderr（脚本在会话临时目录中，不入库）。观察到：`runtime opens collection: ef_research_bge_m3_hybrid_c512_v2`。
- 输出：`Status: answered`，回答引用 `[S2]`、`[S3]`（均来自 `adapointr.pdf`）。退出码 0。`A3_ask_smoke_stdout.txt` / `_stderr.txt`。
- 注意：观察者同时显示 `explicit_collection_name=True`——你本地 `.env` 里显式设置了 `COLLECTION_NAME=ef_research_bge_m3_hybrid_c512_v2`，与 active 一致，所以放行。

### 第 2 步：ingest 保护
1. 先在 Python 里直接调用保护函数（`A3_guard_direct.txt`）：`guard_ingest_target(settings, <active>)` 抛 `ActiveIndexProtectedError`；对照：非 active 名称放行。
2. 然后运行 `python -m evidenceflow ingest data/source_docs`：**退出码 2**，stderr 为 `Ingestion refused: Refusing to ingest into the ACTIVE collection 'ef_research_bge_m3_hybrid_c512_v2'; build a staging index with scripts/build_staging_index.py instead`，stdout 为空。
3. 运行后立刻重新取身份快照并与第 0 步比较：**整份快照完全一致**（7 个 collection 的计数 / metadata、active 的内容哈希、state / manifest sha256）。

### 第 3 步：staging 全量重建，不激活
- **`validate_collection` 校验的内容**（先读代码确认）：只读 staging 的 Chunk——chunk 数量与文档数量是否等于 manifest、文档集合是否一致、chunk ID 是否重复 / 缺失 / 多余（给定 `expected_chunk_ids` 时）、必需 metadata（`chunk_strategy`、`element_ids`）、向量维度（1024）、一次非空 Dense smoke。**它不比较 staging 的向量数值或检索结果与 active 是否相同。**
- **因此设备选择**：不需要 `MODEL_DEVICE=cpu`，使用 `auto`（本机解析为 `cuda:0`）以加快构建。chunk ID 由内容哈希决定，与设备无关；staging 的向量在校验后即删除，不进入任何基线。
- 命令：`python scripts/build_staging_index.py --expected-chunk-ids-from-active --discard-after-validate`，退出码 0。
- 结果（`A3_staging_jobs/`、`A3_staging_stdout.txt`）：`outcome = VALIDATED_AND_DISCARDED`；staging collection `ef_research_bge_m3_hybrid_c512_v__staging__b4e9432c7170615a`；BuildJob 走完 SCANNING→…→READY（约 31 秒）；`validation.passed = true`，817 个 chunk，10 个文档，`missing_chunk_ids = 0`，`unexpected_chunk_ids = 0`，`errors = []`。staging 的 `index_signature` 与 active 相同（`b4e9432c…`）——这正是同一 corpus + 同一 pipeline 的确定性结果。
- 之后 staging collection 经 Chroma API 删除，`discard_staging()` 清空了指针。

### 第 4 步：收尾检查 → `A3_active_after.json`
- Chroma collection 列表与运行前相同：仍是那 7 个，计数与 metadata 一致；
- active 的身份（计数、chunk_id_hash、内容哈希、metadata、index_signature）不变；
- `index_state.json` 的 sha256 与运行前**相同**（`prepare_staging` 写入、`discard_staging` 清除后字节完全一致，因为写入时键已排序），因此无需 `git restore`；`active_manifest.json` sha256 不变；
- 全部流程结束后（pytest、回放之后）又取了一次快照，仍与第 0 步完全一致。
- 没有触碰 A4 要删除的 4 个孤儿 collection（现存的 `ef_research_bge_m3_hybrid_c512_v2_staging` 是 Phase 15A 遗留的旧名 collection，与本次 staging 名不同，未动）。

### Gate 回放：线上路径 vs B0'
`MODEL_DEVICE=cpu`，`experiments/stage_a/replay_runtime_retrieval.py` 走线上 `build_retriever`（collection 由 active 指针解析），耗时约 3 分 27 秒。`experiments/stage_a/a3_compare_replay.py` 比较结果（`A3_replay_vs_b0prime.json`）：基线 sha256 与 A1 冻结值一致；40 题 Top5 **0 差异**；Top20 候选 ID **0 差异**、分数 **0 差异**；精排分数 **0 差异**；P@5 0.2684 · R@5 0.5103 · MRR@5 0.5175 · MAP@5 0.3631 · nDCG@5 0.4903，与 B0' 差为 0；`compare_replay` 的身份与 projection hash 校验 parity = true。

## 5. `scripts/build_staging_index.py`
- 复用 `StagingBuildCoordinator`、`IngestionService`（经 builder 回调）、`validate_collection`，不复制 ingestion 逻辑。参数：`--expected-chunk-ids-from-active`、`--activate`（默认不激活）、`--discard-after-validate`（二者互斥）、`--job-dir`（默认 `artifacts/stage_a/A3_staging_jobs/`）。
- 开始前拒绝（退出码 2）：状态中已有 staging 指针；同名 collection 已存在；参数冲突。校验失败时**保留** staging collection 与指针（退出码 1），`--discard-after-validate` 不会删除失败的证据。
- 阶段上报：`IngestionService` 没有阶段钩子，且逐文件循环，所以用透明代理（tokenizer / embedding / store）在**首次出现**某阶段时单向上报 PARSING→CHUNKING→EMBEDDING→INDEXING，不改 `IngestionService`。
- `--activate`：先把新 collection 的 metadata `lifecycle` 改为 `ACTIVE`，再切换指针（否则新 active 会被 runtime 的 lifecycle 校验拒绝）。**这一路径只在单元测试里验证（Fake + 临时目录的真实 Chroma 做 metadata 改写），没有对真实数据运行。**

## 6. 测试
- 全量：`python -m pytest -q` → **643 passed**（`A3_pytest.log`）。
- 计划要求的用例：有 state 且未显式设置 → 用 active；显式设置且不一致 → 报错；`activate` 再 `rollback` → 解析随指针改变；无 state 文件 → 回落 Settings；metadata 的 lifecycle 不是 ACTIVE → 报错；ingest 指向 active → 退出码 2；staging 脚本用 Fake builder 完成 SCANNING…READY 且默认不激活。
- 补充的用例：`discard_staging` 不改动 active / previous；staging 为 null 时不重写文件（内容与 mtime 都不变）；写入失败时原文件不变且不残留临时文件；`--discard-after-validate` 遇到 active 或 previous 名称时拒绝删除（且什么都没删）；校验失败时不删除；构建异常时不删除；环境变量里的 `COLLECTION_NAME` 也算“显式设置”；损坏的状态文件拒绝启动；保护先于模型加载；`IndexLifecycle`、`StageReporter`、`Observed` 的单元测试。全部使用临时目录和 Fake。
- 新用例放在 `tests/test_runtime_lifecycle.py` 与 `tests/test_build_staging_index.py` 两个新文件里（计划写的是补进 `test_runtime.py`）；`test_runtime.py` 里的既有用例只加了 `index_state_path=<不存在的临时路径>`，避免读到仓库里真实的状态文件，并把 `llm_thinking` 默认值的断言改为 `"disabled"`。

## 7. 仍在直接使用 `collection_name` 的脚本（未改动，按要求）
`grep` 结果（`settings.collection_name` / `COLLECTION_NAME` 常量 / `collection_name=`）：
- **读 `Settings().collection_name`，在 A3 之后仍可用**（`.env` 与默认值都是 v2，等于 active）：`experiments/run_phase10_reranker_performance.py`、`run_phase12_rag_quality.py`、`run_phase15b1_dense_candidate_depth.py`、`run_phase15c_dense30_promotion.py`、`run_phase15c1_exact_dense_architecture.py`（已被 A1 取代）、`run_production_reranked_baseline_replay.py`、`audit_gold_v2.py`、`experiments/stage_a/a1_crosscheck_batch_embedding.py`、`scripts/verify_active_index.py`、`scripts/verify_phase15a.py`。其中 `run_production_reranked_baseline_replay.py` 的 `validate_identity` 仍被 A1 回放脚本复用（我传入解析后的 Settings 副本）。
- **写死历史 collection 名**（与 v2 无关，不受影响）：`prepare_research_corpus.py`、`run_phase13_kb_chunk_audit.py`（`c512_o64_v1`，并在第 3309 行要求 `settings.collection_name` 等于它——本来就不适用于当前配置）、`run_phase13b_controlled_rebuild.py`、`build_research_reference_metadata_collection.py`、`run_research_references_aware_experiment.py`、`scripts/repair_active_index_metadata.py`、`scripts/analyze_runtime_reachability.py`（只读 `.env` 文本并报告默认值）。
- **A3 初版漏列的 4 个 runner（A3.1 补）**：均写死 collection 名，不读 `Settings.collection_name`，也不跟随 active 指针：
  - `experiments/phase13c/runners/run_phase13c_c.py`：`ef_13cc_<arm>`，位于实验工作目录 `WORKDIR/chroma`；
  - `experiments/phase13c/runners/run_phase13e_hybrid.py`：`ef_13e_<arm>`，位于 `WORKDIR/chroma`；
  - `experiments/phase13c/runners/run_phase13f_runtime_promotion.py`：`ef_research_bge_m3_hybrid_c512_v1`，**位于生产 `data/chroma`，并会先 `delete_collection` 再重建它**（v1 在 A4 的禁止删除列表中；该脚本在 A4.3 的删除清单里）；
  - `experiments/phase14a/runners/embed14c.py`：`ARM_SPECS` 中的各 arm collection，位于实验工作目录 `workdir/chroma`。
  这 4 个文件都在 A4.3 要删除的 `experiments/phase13c/`、`experiments/phase14a/` 下。
- `experiments/debug/*`（4 个）与 `experiments/archive/*`（15 个）：历史 / 调试脚本。
- 一个需要注意的行为：以上读 `Settings().collection_name` 的脚本**不会**跟随 active 指针。将来 `activate` 到新 collection 后，它们仍会读 v2，需在 A4 / 阶段 B 决定是删除还是改用 `resolve_active_collection`。

## 8. 与计划 / 指令的偏差与需要你知悉的地方
1. **lifecycle metadata 校验只在 collection 由 active 指针解析而来时执行。** 全新环境（无状态文件）下 collection 由首次 ingest 创建，没有 `lifecycle` 标记，若也校验就会让“ingest 之后马上 ask”失败。已写入 §S.3。
2. **staging collection 创建时不写 `hnsw:space`。**（A3.1 更正）Chroma 的 `Collection.modify(metadata=…)` 整体替换 metadata，并且只是拒绝**含** `hnsw:space` 键的 metadata——合并时去掉这个键即可修改带该键创建的 collection（如 v2），cosine 由 configuration 决定、不变（Chroma 1.5.9 实测，A3.1 已据此修改 `mark_collection_lifecycle` 并补测试）。A3 初版写的“Chroma 拒绝，所以只能不写”不准确。新 staging 不写这个键的真正原因是：源码里没有任何地方读它，cosine 由 collection `configuration` 保证。所以新建的 staging 只写 `{chunk_strategy, storage_contract, lifecycle}`。这意味着：将来经 `--activate` 激活的 collection，metadata 里没有这个旧键，与当前 v2 的 metadata 键集合不同（功能不受影响；`verify_phase15a.py` 把 v2 的 metadata 与冻结快照比对，它本来就是针对 v2 的）。已写入 §S.3。
3. **`.env.example`**：除了 LLM 三项和注释，我把 `COLLECTION_NAME` 这一行整个注释掉了（而不是保留一个带值的行），因为照抄 `.env.example` 就会变成“显式设置”，激活新索引后会被拒绝启动；另外新增了一行 `INDEX_STATE_PATH=…`。
4. **你本地的 `.env` 没有改（它不入库）。** 里面显式写着 `COLLECTION_NAME=ef_research_bge_m3_hybrid_c512_v2`，目前与 active 一致所以放行；但将来 `activate` / `rollback` 到另一个 collection 后，这一行会让 `ask` 报 `Index configuration error`。建议你之后把它删掉。
5. **manifest 的预期规模取 active 的数量**（817 chunk、len(sources) 个文档）——本脚本目前只支持“同一 corpus 的全量重建”；corpus 变化时校验会如实报告规模不符。已写入 §S.3。
6. `A3_active_before/after.json` 里的“内容哈希”是我加的比单纯计数更强的检查（含向量字节），不是计划要求。

## 9. 尚未解决的问题或限制
- `--activate` 没有在真实数据上运行；`mark_collection_lifecycle` 与 `lifecycle.activate` 不是原子的——前者成功、后者失败时，collection 会被标记为 ACTIVE 但指针不变（不影响线上，因为线上以指针为准；需要时可手工改回）。
- 脚本中已存在 staging 指针时拒绝开始，没有“自动清理上次遗留”的功能（有意为之，由用户决定）。
- 其余历史脚本仍读 `Settings().collection_name`（见 §7）。
- 按指令没有进入 A4。

## A3.1 审查修复（2026-10-14）

基线：A3 提交 `9a848d3`，643 个用例。A3.1 后 `python -m pytest -q` → **666 passed**（新增 23；`A3.1_pytest.log`）。
没有发起 LLM 请求，没有对真实数据运行 ingest / build_staging_index / `--activate` / rollback，没有改动主 Chroma 中任何
collection 或其 metadata；所有涉及真实 Chroma 的测试都在 pytest 临时目录中。

| # | 问题 | 修复 | 测试 |
|---|---|---|---|
| 1 | staging 名只取 base 前 32 字符，`…c512_v2` → `…c512_v`，对结果再生成仍是它本身：激活后重建的 staging 名等于 active；rollback 后重建（同 signature）等于 previous。拒绝提示还建议“先用 Chroma API 删掉它”，可能诱导删除线上索引 | `build_staging_collection_name` 改为 `base[:24]-sha256(完整 base)[:8]__staging__sig[:16]`（≤60 字符），再生成必然不同；新增 `reserved` 参数，确定性避让 active/previous（rollback 场景下首选名就是 previous，单靠哈希无法避免）。脚本生成后再检查，等于 active/previous 时构建前拒绝（退出码 2），提示明确“不要删除或修改”；“同名已存在”提示改为“先检查，本脚本不复用也不删除” | `test_phase15a_lifecycle.py`：再生成不同、截断前缀相同的 base 不同名、reserved 避让、候选耗尽报错；`test_build_staging_index.py`：激活后重建、rollback 后重建都不撞名，撞名时拒绝（active/previous 两种）且状态文件不变 |
| 2a | `INDEX_STATE_PATH` 显式设置但文件缺失时静默回落到默认 v2，绕过 active 指针与 ingest 保护 | `_lifecycle_governed` 用 `model_fields_set` 判断，显式设置但缺失时抛 `ActiveCollectionError` | 构造参数与环境变量两种显式设置；`test_runtime.py` 中原先“显式传入不存在路径”的用例改为切到临时目录、使用默认相对路径 |
| 2b | 没有状态文件时 ingest 不受保护 | `guard_ingest_target` 增加纵深防御：用新增的只读 `storage.read_collection_metadata`（`get_collection`，存储目录不存在时直接返回 None）检查目标 metadata，`lifecycle == ACTIVE` 拒绝；collection 不存在放行。不加载模型、不创建 collection | 临时目录真实 Chroma：无状态文件 + 目标标为 ACTIVE → CLI 退出码 2；全新环境（无存储目录 / 有目录无 collection / collection 无标记）首次 ingest 放行，且检查不创建目录或 collection |
| 3 | `mark_collection_lifecycle` 无法改写带 `hnsw:space` 的旧 collection（如 v2） | 合并 metadata 时去掉 `hnsw:space` | 临时目录真实 Chroma：带 `hnsw:space` 的 cosine collection 改写后 lifecycle 更新、其他键保留、`configuration["hnsw"]["space"] == "cosine"`；原用例改为显式断言 cosine |
| 4 | 指针指向不存在的 collection 时经 `get_or_create_collection` 留下空 collection；lifecycle 校验在加载 Embedding 模型之后 | 新增 `runtime._open_active_store`：先 `read_collection_metadata` 校验存在且 ACTIVE，再 `ChromaVectorStore(..., create_if_missing=False)`（`get_collection`）打开；校验在 Embedding 模型之前。无状态文件路径不变（`get_or_create`，构造顺序不变） | 真实 Chroma：缺失 collection → `ActiveCollectionError` 且未创建任何 collection、未加载模型；真实 ACTIVE collection 正常打开；Fake：调用顺序 read_metadata → store → embedding，lifecycle 不符时只执行了 read_metadata |
| 5 | discard 先删 collection 后清指针，清指针失败后无法恢复（再次构建被“已有 staging 指针”拒绝，脚本又没有单独 discard 的入口）；`ProtectedCollectionError` 在构建后抛却返回 2 | `discard_staging_collection` 增加 `list_collections` 参数，collection 已不存在时只清指针（幂等）；新增 `--discard-staging` 独立模式（不加载模型）；撞名保护前置到构建前；构建后仍出现的保护拒绝返回新退出码 3 | 幂等 discard、模拟清指针失败后重试恢复且不重复删除、构建后保护拒绝 → 3 且 staging 保留、CLI 构建前拒绝 → 2、`--discard-staging` 在临时目录真实 Chroma 上删除并清指针、重复执行 → 2、与构建选项组合 → 2 |
| 6 | 缺少 ask 遇 `ActiveCollectionError` 的 CLI 测试 | — | `test_ask_cli.py`：真实 `build_rag_service` 路径，显式 `INDEX_STATE_PATH` 缺失 → stderr 以 `Index configuration error` 开头、退出码 1 |
| 7 | 文档 | §7 补 4 个 runner；§8 第 2 条更正 `hnsw:space` 说法；`PROJECT_SPEC.md` §S.2 写明 LLM 默认值是用户决定（2026-10-14）及对计划字面要求的偏离已获认可；§S.3 补充 A3.1 合同、退出码、两项待决 | — |

**`build_staging_index.py` 退出码约定的变化**：0 成功；1 构建或校验失败（staging 与指针保留）；2 拒绝开始——没有构建、没有删除、没有改动状态文件（新增情形：生成名等于 active/previous、`--discard-staging` 无登记或目标受保护、`--discard-staging` 与构建选项组合）；**3（新增）** 校验已通过但收尾动作被保护拒绝（A3 中此情形返回 2）。

**其他变化**：`ChromaVectorStore` 新增关键字参数 `create_if_missing: bool = True`（默认行为不变）；`storage` 新增并导出 `read_collection_metadata`；`build_staging_collection_name` 新增可选关键字参数 `reserved`，同一 base 的首选名称与 A3 不同（A3 时生成的 staging 已删除，不受影响）。

**待决项（留给 A5，未改）**
1. 激活后旧 active 的 metadata 仍是 `lifecycle=ACTIVE`，rollback 依赖这一点；因此可能同时有多个 collection 标为 ACTIVE（纵深防御会把它们都保护起来）。是否在 activate/rollback 时同步改写 metadata 属于 rollback 语义变更。
2. `artifacts/phase15a/index_state.json` 被 git 跟踪：以后 activate/rollback 会改动被跟踪文件，`git restore` / `checkout` / 切换分支可能改变线上指针。
3. 经 `mark_collection_lifecycle` 改写过的旧 collection，其 metadata 中不再有 `hnsw:space`；`scripts/verify_phase15a.py` 把 v2 的 metadata 与冻结快照比对，若将来改写 v2，需同步该校验。
4. `--discard-staging` 在没有登记 staging 时返回 2（拒绝、不改动），而不是 0；如希望“重复执行也返回 0”，需另定。
