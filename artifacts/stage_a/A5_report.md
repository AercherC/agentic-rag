# A5 报告：精简 PROJECT_SPEC（含三处小修）

## 结论：Gate A5 通过，有一处偏离待用户决定（AGENTS.md，见第三部分 §5）

| Gate | 结果 |
|---|---|
| SPEC 中与代码直接对应的内容逐项核对 | 通过：Settings 23 个字段及默认值、三处 CLI 退出码与错误前缀、Generator 输出 schema、RAGAnswer 字段与 status 取值全部一致（核对表见第二部分 §3） |
| 已退休概念不被描述为“当前” | 通过：grep 剩余命中均为“已退休/历史/运维”语境，逐条解释见第二部分 §4 |
| 新 SPEC 行数 300–450 | 376 行 |
| 全量 pytest | **507 passed**（基线 500 + 小修新增 7；`A5_pytest.log`） |
| `index_state.json` sha256 | `5fbad4f02215d120128ffedf83b2708901ba881bbc258f8c86957ebcc63d8809`，未变 |
| `evals/`、`artifacts/phase15a/` | `git diff -- evals artifacts/phase15a` 为空 |

本步没有打开 `data/chroma`，没有运行 ask / 回放 / verify 脚本，没有发 LLM 请求，没有读或改 `.env`，没有碰 `Claude outputs/`。

---

## 第一部分：小修

### 1. `scripts/verify_active_index.py` 默认写入被跟踪文件
- 修复：`--output` 默认改为 `None`；不传时只打印 JSON 到 stdout，传入时才写文件（`scripts/verify_active_index.py:7-9` docstring，`:150-163` `main(argv)` 与 help，`:172-175` 条件写盘）。`main` 增加 `argv` 参数以便测试。
- 依赖检查：`grep active_index_smoke / verify_active_index` 覆盖 src、tests、scripts、experiments，除脚本自身外没有代码依赖原默认路径；只有计划文档和历史 SPEC 提到脚本名。
- 新增 `tests/test_verify_active_index_script.py`（3 个用例，monkeypatch 替换 `inspect_active_index` 与 `Settings`，不打开 Chroma、不读 `.env`）：不传 `--output` 时当前目录无新文件且 `artifacts/cleanup/active_index_smoke.json` 字节不变；传 `--output` 时写入；检查失败返回 2。

### 2. `scripts/prune_chroma_collections.py` 覆盖 A4 审计记录
- 修复：新增 `check_outputs_writable`（`:78-85`），在 `prune()` 开头、读取状态文件和连接 Chroma 之前调用（`:147-148`）；输出已存在且未传 `--overwrite` → `PruneRefused` → `REFUSED: output file(s) already exist: [...]; pass --overwrite to replace them`，退出码 2。新增 `--overwrite`（`:192-196`），传入 `prune(..., overwrite=)`（`:142`、`:209`）。docstring 安全约束与退出码说明同步（`:10-12`、`:19-20`）。
- 检查范围：只检查**本次会写**的文件——dry-run 只查 before，`--execute` 查 before 和 after（dry-run 不写 after，所以 after 已存在不阻止 dry-run）。用默认路径再跑一次 dry-run 也会因 `A4_chroma_before.json` 已存在而被拒绝。
- 新增 4 个用例（`tests/test_prune_chroma_collections.py:180-224`）：dry-run 时 before 已存在、`--execute` 时 after 已存在 → 2、文件内容不变、collection 不变（参数化 2 个）；monkeypatch `chromadb.PersistentClient` 证明拒绝发生在打开 Chroma 之前；`--execute --overwrite` 可以覆盖两份文件。

### 3. Chroma 删除 collection 后残留的段目录（只写文档，未打开、未删除）
主会话已只读查询 `chroma.sqlite3` 的 segments 表。A4 删除 4 个 collection 后，`data/chroma` 下有 5 个不再被引用的 HNSW 段目录，**由用户手动删除**（先备份、确认没有进程使用 Chroma）：

| 段目录 | 原 collection |
|---|---|
| `5d5d025e-d9b1-4199-a0ae-bee85794fd95` | `ef_research_structure_v1` |
| `928e9c57-83df-4191-b11e-36151d59da7e` | `ef_research_bge_m3_hybrid_c512_v2_staging` |
| `edd92cf6-6f2c-4a51-9594-906705b0b628` | `ef_research_structure_v1_elemaware` |
| `f32f608b-ad2c-447a-8e7c-3cb87d2e28c7` | `ef_research_bge_m3_c512_o64_refmeta_v1` |
| `09024bdd-6ae1-4f44-a8b8-065e719eded7` | A4 之前就已不被引用，来自更早删除的某个 collection |

仍被引用（不能删）：`7429ba3c…` → v2，`ccb26c71…` → hybrid v1，`beb2c4ac…` → c512_o64_v1。
文档：新 SPEC §6.6（`PROJECT_SPEC.md:147-153`）写明 `delete_collection` 不删段目录、如何以只读方式查 segments 表找出残留、删除前必须备份并确认无进程占用、`chroma.sqlite3` 不会缩小且 VACUUM 未决（Stage A 不做）。

---

## 第二部分：A5

### 1. 改动清单
| 文件 | 内容 |
|---|---|
| `docs/history/PROJECT_SPEC_v0.26_full.md` | 新增：第 1 行为“历史版本，不再是当前规范”的说明，第 3 行起与重写前的 `PROJECT_SPEC.md`（含 §S 的 A3.1/A3.2 条目）逐字节一致（`cmp` 验证） |
| `PROJECT_SPEC.md` | 重写为 v0.27 `EvidenceFlow — Current Specification`，16 章，376 行；§S 已并入各章并删除 |
| `README.md` | 架构改为 Exact Dense Top20，并说明 active 指针与 ACTIVE 保护（`:21-28`、`:34`）；新增结构化生成、status、DeepSeek 默认值说明（`:49-57`）；指标表换为 B0'，并给出路径与 sha256（`:62-75`）；退休路径说明改为从 tag `stage-a-start` 恢复（`:80-83`）；环境改为 `agent-gpu`，去掉会被保护拒绝的 `ingest` 示例并加说明（`:90-97`）；新增 “Design Docs” 一节，链接 `docs/agentic_rag/` 与 SPEC（`:99-105`） |
| `artifacts/stage_a/STAGE_A_SUMMARY.md` | 新增：阶段 A 总结 |
| `artifacts/stage_a/A5_report.md`、`A5_pytest.log` | 本报告与测试日志 |
| `AGENTS.md` | **未改**，见第三部分 §5 |

### 2. §S 合并去向
| §S 条目 | 新 SPEC 位置 |
|---|---|
| §S.1 Exact Dense、numpy.exact、启动校验、K、离线逐条 embed_query、CPU Gate | §7、§2、§13 |
| §S.1 B0' | §13 |
| §S.2 v3 协议、解析顺序、修复、诊断、LLM Client、json_object、thinking、CLI 前缀 | §9、§12.1 |
| §S.2 LLM 默认值（用户 2026-10-14 决定） | §11 |
| §S.2 在线脚本策略（6 秒间隔、429、JSONL 续跑、退出码 4） | 未并入：只属于 `experiments/stage_a/` 实验脚本，不是产品合同；原文保留在 history |
| §S.3 index_state_path、active 指针三条规则、按指针读取不创建 collection | §6.2、§11 |
| §S.3 ingest 保护与纵深防御 | §6.3、§12.1 |
| §S.3 discard_staging、staging 脚本、命名、discard 幂等、`--activate` 与 `hnsw:space` | §6.1、§6.4 |
| §S.3 退出码（A3.1 修订的 0/1/2/3）与 `--discard-staging` 退出码（A3.2） | §12.2 |
| §S.3 validate_collection 范围、IngestionService 先删后写不变 | §6.4、§5 |
| §S.3 两项待决 + A3.1 另两项待决 | §6.7、§14.1 |

采用的是 A3.1/A3.2 修订后的说法：新 staging 不写 `hnsw:space` 的原因是“源码不读它、cosine 由 configuration 保证”（不是“Chroma 拒绝所以只能不写”）；构建后保护拒绝返回 **3**（不是 2）。

### 3. 核对表（SPEC ↔ 代码）

**Settings**（SPEC §11 表，`PROJECT_SPEC.md:236-258`）。另用脚本读取 `Settings.model_fields` 的默认值与 SPEC 表逐项比对：23/23 一致，SPEC 无多余字段。

| 字段 | SPEC 默认值 | 代码位置 | 一致 |
|---|---|---|---|
| llm_api_key | None | `config.py:33` | ✔ |
| llm_base_url | https://api.deepseek.com | `config.py:37` | ✔ |
| llm_model | deepseek-flash | `config.py:41` | ✔ |
| llm_temperature | 0.0（0–2） | `config.py:45` | ✔ |
| llm_timeout_seconds | 60.0（>0） | `config.py:52` | ✔ |
| llm_max_attempts | 2（1–2） | `config.py:58` | ✔ |
| llm_thinking | disabled | `config.py:64` | ✔ |
| embedding_model | BAAI/bge-m3 | `config.py:73` | ✔ |
| source_docs_path | data/source_docs | `config.py:77` | ✔ |
| chunk_size | 512 | `config.py:82` | ✔ |
| chunk_overlap | 64（< chunk_size，`config.py:176`） | `config.py:87` | ✔ |
| chunk_strategy | hybrid | `config.py:92` | ✔ |
| max_context_tokens | 4096 | `config.py:100` | ✔ |
| reranker_model | BAAI/bge-reranker-base | `config.py:9,106` | ✔ |
| reranker_input_k | 20 | `config.py:111` | ✔ |
| reranker_final_k | 5（≤ input_k，`config.py:189`） | `config.py:116` | ✔ |
| reranker_batch_size | 4 | `config.py:121` | ✔ |
| reranker_max_length | 512 | `config.py:126` | ✔ |
| model_device | auto | `config.py:131` | ✔ |
| reranker_dtype | float32 | `config.py:135` | ✔ |
| vector_db_path | data/chroma | `config.py:141` | ✔ |
| collection_name | ef_research_bge_m3_hybrid_c512_v2（hybrid 下不得为 fixed 身份，`config.py:180-187`） | `config.py:145` | ✔ |
| index_state_path | artifacts/phase15a/index_state.json | `config.py:154` | ✔ |

**CLI 退出码与错误前缀**（SPEC §12）
| 项 | SPEC | 代码位置 | 一致 |
|---|---|---|---|
| ingest 保护拒绝 → 2，`Ingestion refused:` | §12.1 | `__main__.py:124-127` | ✔ |
| ingest 异常 → 1，`Ingestion failed: <类型>: <消息>` | §12.1 | `__main__.py:128-130` | ✔ |
| ingest 有失败文件 → 1，否则 0 | §12.1 | `__main__.py:132` | ✔ |
| ask 前缀 6 种 | §12.1 | `__main__.py:82-94` | ✔ |
| ask 异常 → 1，成功 → 0；`Status:` 行 | §12.1 | `__main__.py:54,141-145` | ✔ |
| staging 构建：1 校验失败 / 3 构建后保护拒绝 / 0 成功 | §12.2 | `build_staging_index.py:274-276,293-296,303` | ✔ |
| staging 构建前拒绝 → 2，`REFUSED:` | §12.2 | `build_staging_index.py:204-234`（各拒绝原因）、`:455-457` | ✔ |
| `--discard-staging` 组合拒绝 → 2 | §12.2 | `build_staging_index.py:441-443` | ✔ |
| `--discard-staging` 状态文件不可读 → 2，`REFUSED: cannot read lifecycle state …` | §12.2 | `build_staging_index.py:388-396` | ✔ |
| `--discard-staging` 无 staging / 受保护 → 2 | §12.2 | `build_staging_index.py:150,153,406-409` | ✔ |
| `--discard-staging` Chroma 出错 → 1，`FAILED:`；成功 → 0 | §12.2 | `build_staging_index.py:410-415` | ✔ |
| prune 拒绝 → 2，`REFUSED:`（含输出已存在） | §12.3 | `prune_chroma_collections.py:85,147-148,211-213` | ✔ |
| prune Chroma 失败 → 1，`FAILED:`；成功 → 0 | §12.3 | `prune_chroma_collections.py:214-218` | ✔ |
| verify_active_index：0 通过 / 2 失败；默认不写文件 | §12.4 | `verify_active_index.py:160,172-177` | ✔ |

**Generator 输出 schema**（SPEC §9）
| 项 | 代码位置 | 一致 |
|---|---|---|
| `rag-generator-v3` | `generation/generator.py:21` | ✔ |
| 空 Context 拒答文本 | `generation/generator.py:22,175-176` | ✔ |
| `GeneratorOutput` extra=forbid，字段 status/answer | `domain/generation.py:51-57` | ✔ |
| schema：enum 取自 AnswerStatus，required 两字段，additionalProperties false | `domain/generation.py:69-83` | ✔ |
| 解析顺序（strip → think → 代码块 → json.loads → 校验） | `generation/generator.py:53-90` | ✔ |
| 修复一次后抛 `GenerationContractError(ValueError)` | `generation/generator.py:49,192,198` | ✔ |
| `STRUCTURED_OUTPUT_MODE = "json_object"` | `llm/openai_compatible.py:21` | ✔ |
| 瞬态重试类别 timeout/连接/429/5xx | `llm/openai_compatible.py:184-192` | ✔ |

**RAGAnswer 与 status**（SPEC §3、§10）
| 项 | 代码位置 | 一致 |
|---|---|---|
| 字段 query / status / answer / sources / retrieved_chunks | `domain/answer.py:46-59` | ✔ |
| `AnswerStatus` = answered / insufficient_context | `domain/generation.py:27-31` | ✔ |
| `GeneratedAnswer` status / answer_text | `domain/generation.py:34-39` | ✔ |
| status 由 assemble 透传 | `citation/parser.py:96` | ✔ |
| 未知标签 / answered 无引用 → CitationValidationError | `citation/parser.py:63-70` | ✔ |

### 4. 退休概念 grep
命令（仓库根目录，Git Bash）：
```
grep -n -E "Translation|Multilingual|BM25|RRF|HNSW|hnsw|STATUS:|翻译|多语" PROJECT_SPEC.md README.md
```
结果：
```
PROJECT_SPEC.md:136:  metadata 时去掉旧键 `hnsw:space`（Chroma 只拒绝含该键的 `modify`；cosine 由 configuration 保证，不变）。新建 staging
PROJECT_SPEC.md:137:  不写 `hnsw:space`，因为源码不读它。
PROJECT_SPEC.md:148:- Chroma 1.5.9 的 `delete_collection` **不会**删除磁盘上的 HNSW 段目录（`data/chroma/<segment-uuid>/`）。
PROJECT_SPEC.md:160:- 被 `mark_collection_lifecycle` 改写过的 collection 不再有 `hnsw:space`；以后若改写 v2，需同步
PROJECT_SPEC.md:318:  与此前 HNSW B0（`evals/evaluation_v3/phase15b_hybrid_candidate_v2/production_reranked_baseline.json`）40 题 Top5 逐题一致。
PROJECT_SPEC.md:326:  BM25、HITL 澄清、长期/向量记忆、现在接 Web Search、现在引入 LangGraph。
PROJECT_SPEC.md:327:- 已退休、不得作为当前能力描述或恢复的：查询翻译、多语双路检索、BM25 生产分支、RRF 融合、以 Chroma HNSW 作为线上候选检索、
PROJECT_SPEC.md:335:3. 被 `mark_collection_lifecycle` 改写过的 collection 不再有 `hnsw:space`；以后改写 v2 时须同步 `verify_phase15a.py` 的快照比对。
PROJECT_SPEC.md:337:5. Chroma `delete_collection` 留下的 HNSW 段目录需按 §6.6 手动清理；`chroma.sqlite3` 是否 `VACUUM` 未决。
PROJECT_SPEC.md:363:| §16–§17 | 多语检索、Translation、RRF、干预历史 | 已退休（§14） |
README.md:75:（sha256 `422fd629…`；与此前 HNSW 基线逐题 Top5 一致）。
README.md:80:Phase 15B v2 已将 Hybrid 判定为 `DO_NOT_PROMOTE`。Translation、Multilingual、BM25
README.md:81:Production branch、RRF 以及以 Chroma HNSW 作为线上候选检索均已退休；历史源码可从 git tag
```
逐条说明：
- 136/137/160/335：`hnsw:space` 是 Chroma collection 的 metadata 键名，讲的是 lifecycle 改写时如何处理旧键，与线上检索算法无关。
- 148/337：讲 Chroma 删除 collection 后磁盘上残留的 HNSW 段目录（运维说明），不是线上检索。
- 318、README 75：说明 B0' 与“此前”的 HNSW 基线一致，HNSW 作为历史对照出现。
- 326：引用审查 §8 的“明确不做”清单（重新引入 BM25）。
- 327：“已退休、不得作为当前能力描述”清单本身。
- 363：历史索引表，去向标为“已退休”。
- README 80/81：“Retired Retrieval Paths” 一节，明确写为已退休。
- `STATUS:`（大写，v2 控制头）0 命中。若不区分大小写，另会命中 SPEC 第 52–54 行的字段名 `status: AnswerStatus`、SPEC 第 270 行和 README 第 54 行的 CLI 输出行 `Status:`；这些是 v3 的 `RAGAnswer.status` 与 CLI 显示，不是控制头。`MultilingualRetriever` 0 命中。

### 5. 行数
`wc -l PROJECT_SPEC.md` = 376，在 300–450 之间。

### 6. 测试
`python -m pytest -q`（`agent-gpu` 环境的 python）→ **507 passed, 28 warnings**（`A5_pytest.log`）。500（A4 基线）+ 3（`test_verify_active_index_script.py`）+ 4（prune 新增）= 507。

---

## 第三部分：偏离与待决
1. **在线脚本策略（§S.2 最后两条）没有并入新 SPEC**：它们只约束 `experiments/stage_a/` 的探测与冒烟脚本（请求间隔、429 重试、JSONL 续跑、退出码 4），不是产品合同。原文保留在 `docs/history/PROJECT_SPEC_v0.26_full.md`。
2. **prune 的覆盖检查只针对本次会写的文件**（dry-run 只查 before）。如果你希望 dry-run 时 after 已存在也拒绝，需要另定。
3. **SPEC §6.7 多写了一条限制**：`mark_collection_lifecycle` 与 `activate` 不是原子操作，且 `--activate` 尚未在真实数据上运行过（来自 A3 报告 §9）。这一条只写文档，没有改变行为。
4. 新 SPEC §12.4 记录了 `verify_phase15a.py` 默认写被跟踪的 `artifacts/phase15a/phase15a_report.json`。这个问题与小修 1 同类，本次**没有修**，只写进文档。
5. **AGENTS.md 没有改（需要你决定）**：计划 A5 第 5 步要求在 AGENTS.md 的“文档优先级”中加入阶段计划。但当前 AGENTS.md **没有这一节**：初始提交 `521a059` 里的 “## 1. 文档优先级” 在 A0 快照 `87fa8ad` 中已经被删除（A0 快照提交的是你当时工作区里未提交的改动），现在文件从 “## 2.” 开始。重新加回这一节，等于恢复一段你可能是有意删除的规则，我判断这属于“计划与实际不符、无法判断”，所以没有改。可选方案：（a）恢复原 §1 并加一行阶段计划；（b）只新建一个最小的 §1，内容只有这一项；（c）不改，因为 CLAUDE.md 已经指向这份计划。
6. README 末尾有一行孤立的 `# RAG`，A5 之前就有，没有动。`.env.example` 里的 `RERANKER_DEVICE=cpu` 已不是 Settings 字段（会被忽略），本次没有改。
7. 5 个残留段目录等待你手动删除（见第一部分 §3）。
