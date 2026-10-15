# Stage B · B1 报告：合同先行（只改文档）

## 1. 改动范围
只改 `PROJECT_SPEC.md`（v0.27 → v0.28）与 `docs/agentic_rag/03_STAGE_B_EXECUTION_PLAN.md`（v1 → v2），新增本报告。
未改代码、测试、`evals/`、其他 `artifacts/`、`01_ARCHITECTURE_REVIEW.md`；未发任何 LLM 请求。

### PROJECT_SPEC.md
| 章节 | 改动 |
|---|---|
| 头部 | Version 0.28；Last Updated `2026-10-15（Stage B B1）` |
| §12.5（新） | RunTrace：Domain Model、事件名、payload 白名单与禁止项、TraceRecorder 协议、不变式、`ask --trace`、`build_rag_service(..., trace_recorder=None)`；标注 B2 实现 |
| §13 | 一处措辞：“不得创造新的 IR 指标名（端到端指标名只由 §13.1 定义）”，避免与 §13.1 新指标名冲突 |
| §13.1（新） | rag_e2e_v1：(a) 题集 schema 与 fingerprint；(b) mapping 文件分离；(c) answer_keys 三条校验；(d) key_match；(e) 指标、失败层归因、`abstention_metrics` 合同变更；(f) B5b 受控上下文；(g) 评委协议；(h) Gate 策略文件；(i) 评委合格线与指标地位 |
| §14 第 3 条 | 阶段 B 已进入；阶段 C–E 仍不在范围 |
| §14.1 第 6 条 | 改为“由 §13.1(e) 定义，B4 实现” |
| §16 | 新增一行“Stage B 新增，v0.28 → §12.5、§13.1” |

SPEC 行数：377 → 508；`git diff --numstat` 为 +137 / −6（净增 131 行），比“约 120 行”多约 10%。
超出部分主要来自失败层归因每条一行、两处 TODO/保守说明；没有再压缩，以免合同点含糊。

### 计划 v2
头部版本与 v2 说明；§0.3 执行顺序与允许真实调用的步骤加入 B5b；§1.3 勘误（架构审查 §10 的 Judge 一致率 ≥ 0.85 由 D2 取代）；
§3.1 删除重复的 4 列表头和分隔行，新增 `key_fact_recall`、准确率与精确率（“答对”由 key_match 判定）、受控上下文三行；
B1 第 7–9 条；Gate B1 措辞（允许新增本报告）；B3 字段、起草、校验、冻结、Gate；B4 代码位置、`controlled` 子命令与要点、测试要求；
新增 B5b 一节；B6 第 3 项；B7 E0 报告、gates 候选指标、STAGE_B_SUMMARY 遗留问题、Gate B7；§6、§7、§8（合计改为约 410–530）、§9 D6 与 D2 后续影响。

## 2. 歧义 / 与代码可能冲突之处及处理
1. **映射状态 UNMAPPABLE**：代码 `MappingStatus` 有 7 个取值，没有 `UNMAPPABLE`。SPEC 写成“沿用现有 `MappingStatus`；
   FULLY_MAPPED / FRAGMENTED_ACROSS_CHUNKS 为可用，其余统称 UNMAPPABLE”，不新增枚举值。
2. **`truncated_pair_count`**：代码中是整批一个计数（`LearnedReranker.last_truncated_pair_count`），不是按 rank。SPEC 写为
   `rerank.completed` 的事件级字段。
3. **未列出事件的 payload**：说明只列了部分事件的白名单。SPEC 写成“未列出的事件 payload 为 `{}`”（最保守）。
   `run.failed.stage` 写为“失败事件所属阶段名”，未枚举具体取值。
4. **`build_rag_service` 签名**：代码已有 `retriever=None, generator=None` 两个关键字参数，SPEC 写成三者并存。
5. **错误题的分母**：说明要求从“回答类指标”分母排除；SPEC 扩展为“回答类与状态类”（与现有 `abstention_metrics` 单独计 error
   的做法一致），并规定上下文层指标对所有记录了 `context.completed` 的题计算（含生成错误题）。
6. **`acceptable_abstention` 对指标的影响**：说明未定义。SPEC 写成“目前只用于分片报告，不改变任何分子分母”，留 TODO。
7. **`forbidden_claims` 结构**：说明未定义。SPEC 写成每条 `claim_id`、`text`、`evidence_anchors`（与之矛盾的原文锚点）。
8. **`context_all_required` 分母**：写为“可答且无 UNMAPPABLE 组的题数”（与 §13 现有 eligible 38 的做法一致）。
9. **`cited_source_in_evidence_rate` 分母**：写为可答题 `sources` 总条数（按 `RAGAnswer.sources`，已按首次引用去重）。
10. **失败层归因规则**：说明只给了层名，规则由我按确定性结果组合写出（错误题 → 不可答 STATUS → RETRIEVAL → CONTEXT →
    STATUS → GENERATION → CITATION → PASS；含 UNMAPPABLE 组记 UNATTRIBUTABLE）。RETRIEVAL 以最终 Top-K
    `retrieved_chunks` 为准，没有区分 Dense Top20 漏召与重排漏选。
11. **answer_keys 校验 ① 的规范化**：用 (d) 的 key_match 规范化，不含 `stable_mapping` 的断词连字符修复；PDF 换行断词处的
    别名可能校验失败，届时应换别名。校验 ② 用 (d) 的完整匹配规则（含 ASCII 边界）。
12. **消融有效性中的锚点检查**：写为“经 `stable_mapping` 规范化”，未指定 `hyphen_rules` 的来源；B4 实现时需定。
    要素残留检查只针对 key_matchable 的 required claim（空 `answer_keys` 按“所有组都命中”会恒为真）。
13. **数字反事实**：补了三条确定性细节：每题按 claim、组顺序取第一个符合者、至多一个变体；替换在 chunk 正文上按 ASCII
    边界替换别名全文（含 `%`）；替换后用 `ContextBuilder` 重建，若纳入的 chunk 集合变化则 `INVALID`。判定时 `abstained` 优先。
14. **B5b 重建上下文的 rank**：observation 的 `context.completed.entries` 没有 `retrieval_rank`，SPEC 写为“按原 `source_label`
    顺序作为 rank 重建”，再用 `context_text_sha256` 校验。
15. **评委 schema**：统一外层 `{"judgments": [...]}`，每条带 `reason`；`groundedness` 写成一次调用内“先拆陈述再判定”，
    不是 RAGAS 的两次调用；`citation_support` 的“句子 × 合法标签”拆分写为确定性，具体分句规则留给 B4。
16. **同厂商判定**：写为 base URL 注册域相同即拒绝（沿用 B0 探测脚本的做法）。

## 3. 待主会话决定
1. `acceptable_abstention` 是否影响 `answerable_false_abstention_rate`（当前：不影响，只作分片）。
2. 第 10 条失败层归因的顺序与规则是否认可；RETRIEVAL 是否要拆成候选漏召（Top20）与重排漏选（Top5）。
3. 上下文层指标是否包含生成错误题（当前：包含）。
4. 数字反事实每题至多一个变体的限制是否会使有效变体难以达到 ≥ 8（可放宽为每题多个，但会增加请求数）。
5. 消融锚点检查所用 `hyphen_rules` 的来源（建议 B4 时取该锚点源页的规则）。
6. SPEC 净增 131 行，超出约 120 行的预算约 10%，是否需要压缩。

## 4. 验证
- `python -m pytest -q -p no:cacheprovider`：510 passed。
- `artifacts/phase15a/index_state.json` sha256：`5fbad4f02215d120128ffedf83b2708901ba881bbc258f8c86957ebcc63d8809`（不变）。
- `git diff --stat` 只有两个文档文件，另新增本报告。
