# Stage B · B3a-1 报告：rag_e2e_v1 工具层与修复轮

- 日期：2026-10-20
- 范围：B3a-1（工具层）+ B3a-1 修复轮；**未起草任何评测题**
- 合同：`PROJECT_SPEC.md` v0.30（§13.1(a)–(d)）
- 冻结索引：`ef_research_bge_m3_hybrid_c512_v2`，817 chunks / 10 documents，
  `index_signature=b4e9432c…`，`chunk_id_hash=e87a613ebe65b10262996e83974b55839e0b6c27fbe22053ee7b8826dd1f153d`

## 1. 本步范围与产物

B3a-1 只实现“工具层”，不写题：

- `src/evidenceflow/eval/e2e/dataset.py`：`EvalCase` / `E2EDataset` schema、fingerprint、确定性校验；
- `src/evidenceflow/eval/e2e/key_match.py`：§13.1(d) 关键要素匹配（纯函数）；
- `src/evidenceflow/eval/e2e/mapping.py`：题集锚点 → 索引 chunk 的映射与身份绑定；
- `scripts/e2e_dataset.py`：`validate` / `map` 两个子命令；
- `experiments/stage_b/b3_pilot_mapping.py`：canonical 40 题试点映射与投影对照（一次性脚本）；
- 产物 `artifacts/stage_b/B3a1_pilot_{dataset,mapping,mapping_check}.json`。

试点题集 40 题（canonical 40 题的题干与证据组原样保留），**required_claims 全为 0**，共 44 个证据组。
因此该试点集不用于《§13.1(a) 可答题至少 1 条 required claim》的冻结校验：`validate_dataset` 会按设计报 40 条
`answerable_required_claim`，这是 B3a-2 起草 claims 后才会消除的已知状态，不视为工具缺陷。

## 2. 两个合同决策及理由

### 决策 A：answer_keys 规则① 用「规范化后字面子串」

- 规则① 改为：`normalize_text(alias) in normalize_text(page_text)`，**不套用** (d) 的 ASCII 边界。
- 理由：规则①的职责是证明别名来自原文（provenance）。若同时要求边界，原文写作 `1.5mm` 时别名 `1.5` 会被迫写成
  带单位形式，而自然回答常写裸数字，会让 `key_fact_recall` 系统性漏计。边界规则留在规则②（别名不得命中题干）与
  (e) 计分、B5b span 定位侧，仍能防 `1.5` 误命中 `11.5`。
- 落地位置：`PROJECT_SPEC.md:394`（§13.1(c)①）；`src/evidenceflow/eval/e2e/dataset.py:227-230`。

### 决策 B：`normalize_text` 增加「行尾连字符折叠」

- 规范化顺序固定为：`NFKC → 行尾连字符折叠（正则 -\s*\n\s*）→ 去掉 [S\d+] → casefold → 连续空白压成一个空格并去首尾`。
  该顺序与代码一致，并修正了 v0.29 文档中「先 casefold 再去标签」的矛盾表述（casefold 后 `[S1]` 变 `[s1]`，旧顺序删不掉）。
- 理由：PDF 抽取保留印刷断词（`view-\npoints`），不折叠时自然回答的 `viewpoints` 永远无法与原文对齐；`view- points`
  虽然能过校验但不是自然写法，构成 ARR-003 死结。
- 已知局限：合法连字符复合词若恰在行尾断开（`self-\nattention`）会被折叠成 `selfattention`；B3a-2 起草别名时应使用
  折叠后的形式或改用其他别名。测试用 `pytest.mark.xfail(strict=True)` 固定了该行为。
- 落地位置：`PROJECT_SPEC.md:402-408`（§13.1(d)）；`src/evidenceflow/eval/e2e/key_match.py:23,27-45`。

## 3. 修复清单与验证方式

| 项 | 修复 | 验证证据 |
|---|---|---|
| B1/B2 | `normalize_text` 按决策 B 改顺序并加 `_LINE_END_HYPHEN` | `normalize_text("view-\npoints") == "viewpoints"`；`alias_hits("viewpoints", "source has view-\npoints") == True`；`tests/test_e2e_key_match.py` 新增用例 |
| B1/B2 | 规则① 改字面子串，规则②保持 `alias_hits` | `1.5` 在页文本 `1.5mm` 中通过规则①；`1.5` 不命中题干 `11.5`（规则②仍报 `answer_keys_query`）；`tests/test_e2e_dataset.py` 正反例 |
| B3 | `map` 统计 usable/unusable/source_failures；`usable==0` 或存在 `SOURCE_EXTRACTION_FAILURE` → exit 2（文件照常写出） | `--source-root` 指向空目录：stdout `groups=44 claim_anchors=0 usable=0 unusable=44 source_failures=44`，stderr `REFUSED: mapping has no usable record (usable=0)`，exit 2，映射文件已写出；正常路径 stdout `usable=40 unusable=4 source_failures=0`，exit 0 |
| B4 | `E2EMapping` 增加可选 `chunk_count` / `chunk_id_hash`；`map_dataset` 增加期望值参数并写入实测值 | 真实索引映射文件含 `chunk_count=817`、`chunk_id_hash=e87a613e…`（与 `A1_exact_baseline.json` 一致）；`expected_chunk_count` / `expected_chunk_id_hash` 不符抛 `ValueError`；旧 mapping 文件无这两个字段仍能 `model_validate` |
| B4 | `map` 与 active 指针交叉核对 chunk_count / document_count | 指针 `chunk_count=2` 而 store 只有 1 条 → exit 1、stderr `FAILED: ValueError: chunk_count mismatch`、不写映射文件；`document_count` 同理 |
| S5 | `write_json` 与 pilot 脚本固定 LF 换行 | 三个 pilot 产物字节中 `\r` 计数均为 0；`tests/test_e2e_dataset_script.py::test_write_json_fixes_lf_newlines` |
| S6 | 校验器新增 `answerable_evidence_group`：可答题至少 1 个 `required_evidence_groups` | 空证据组的可答题报 `answerable_evidence_group`；普通可答题与错误前提题正例不报；SPEC §13.1(a) 与 `validate_dataset` docstring 已同步 |
| S10 | `check_output` 拒绝 `evals/` 下任何解析后路径（无论是否存在） | 不存在的 `evals/_b3a1_test_never_created/out.json` 也被拒绝（exit 2，未创建文件）；已有 evals 文件仍拒绝 |
| S9 | 查清 `corpus_version` 口径 | 见 §6；本轮不改数据 |
| 文档同步 | 计划文档 B4 代码位置去掉 `key_match.py`，B3a-1 小节补修复轮一行；两份计划逐字节相同 | `docs/agentic_rag/…` 与 `Claude outputs/…` 字节比较为 True |
| 可选 | `B0_report.md` 末尾加 2 行勘误，不改原文其它内容 | `artifacts/stage_b/B0_report.md` §勘误 |

## 4. pilot 映射结果（40/44 可用，4 组 UNMAPPED）

运行：

```powershell
& 'D:\miniconda\envs\agent-gpu\python.exe' experiments\stage_b\b3_pilot_mapping.py
```

脚本退出码 2（`gate_pass=false`，投影与新映射的口径差异，**预期**）；产物已更新且
`B3a1_pilot_dataset.json` 的 `fingerprint` 与 `B3a1_pilot_mapping.json` 的 `dataset_fingerprint` 相等
（`6d48bad6e917aed3475f0a8dc2c8d8ebcd8b4b8d77b32e40db5627859c2c6ebc`）。映射文件记录
`chunk_count=817`、`chunk_id_hash=e87a613e…`。

| 指标 | 数值 |
|---|---|
| 证据组总数 | 44 |
| FULLY_MAPPED | 40 |
| UNMAPPED | 4（ARR-016 / ARR-019 / ARR-033 / ARR-034，各 1 组） |
| 全部组可用的题数 | 36 / 40 |
| A1 projection eligible | 38 / 40（ARR-003、ARR-033 不可用） |
| `new_tool_matches_map_stable_dataset` | true |
| `projection_eligible_matches_a1` | true |

### 4.1 四个 UNMAPPED 的逐条根因

- **ARR-019：right_context 含页码 `6`。** 锚点 `16384 points` 所在页 6 末尾是页码 `6`；locator 的
  `right_context` 以 `… sampled from 6` 结尾。唯一可能覆盖的 chunk `d6451377…`（pages 6–9）在合并跨页时
  含下一页页眉 `J. Wang et al.`、不含页 6 页码，`whole_chunk_alignments_in_source=0`，唯一上下文 proof 不成立
  → UNMAPPED。投影侧只看子串，`projection_substring_hit=true`。
- **ARR-034：left_context 含页眉 `18 H. Zhou et al.`。** 锚点 `supplementary material` 的 locator
  `left_context` 为 `18 h. zhou et al. in this`；chunk `197c3164…`（pages 15–18）正文从 `references in this
  supplementary material` 开始，页眉被 chunker 排除，`whole_chunk_alignments_in_source=0`，proof 不成立
  → UNMAPPED；投影侧同样是子串命中。
- **ARR-016：锚点所在 chunk 不是 Loader 全文的精确子串。** 锚点 `Chamfer distance` 位于页 5；覆盖它的
  chunk `a40441e1…` 省略了公式 `L = L_completion + λ L_preservation. (6)`（32 个字符被 hybrid chunker
  移出正文流），`whole_chunk_alignments_in_source=0`。页 6/7 另有两个 chunk 可整块对齐，但不覆盖该锚点范围，
  无法碎片拼接 → UNMAPPED。`anchor_occurrences_in_whole_document=4`（同词多处），也是无法用短锚点做保守
  回退的原因。
- **ARR-033：断词问题（`additional`）。** 页 12 与 chunk `3c4e580c…` 都保留 `addi-\ntional`；投影侧
  `phase15b_active_relevance.normalize_text` 不折叠行尾连字符，故投影 ineligible。新映射器用 source-proven
  hyphen rule 把该词规范化为 `additional`，但 chunk `3c4e580c…` 从 `in this document` 开始，缺 left_context
  中的小节标题 `A. Overview`，`whole_chunk_alignments_in_source=0` 且唯一上下文 proof 不成立 → 仍 UNMAPPED。
  实测把该锚点的 `left_context` 去掉后可 FULLY_MAPPED，说明断词本身的规范化已生效，剩余障碍是 chunk 起点边界。

### 4.2 chunk 与源文本的总体量化

817 个 chunk 中，**136 个** 规范化后不是所属文档 Loader 全文的精确子串：

```text
chunks 817 not exact substring (rules=[]): 136
```

计数口径：对每个 chunk 取 `_compact(normalize_mapping(chunk.content, []))`，判断是否为
`_compact(normalize_mapping("".join(文档全部 Segment 原文), []))` 的子串。这是 stable_mapping 的保守对齐前提，
也是 §4.1 中多组 UNMAPPED 的总体背景（16.6% 的 chunk 无法整块对齐）。

## 5. 交叉核对结论

- `new_tool_matches_map_stable_dataset=true`：新工具 44 组的 `(status, chunk_sets)` 与直接调用
  `map_stable_dataset` 逐组完全一致（同为 44 组）。
- `projection_eligible_matches_a1=true` 且 `projection_hash_matches_a1=true`：新工具重算的投影 hash
  `6ccf0ffe…` 与 `A1_exact_baseline.json` 一致，eligible 集合与 A1 逐题一致（38/40）。
- 与旧投影的差异是**规则差异**，不是回归：`diff_type_counts` 为
  `NEW_SUBSET_OF_PROJECTION=20`、`NEW_UNUSABLE_PROJECTION_NONEMPTY=3`、`NEW_USABLE_PROJECTION_EMPTY=1`；
  共 24 个组 / 20 道题不一致。旧投影只做“anchor 是否为 chunk 子串”，新映射要求整块 chunk 唯一对齐或唯一
  context proof 且按源文本位置覆盖，因此新结果更严格（20 组是投影超集的子集）。
- **36 vs 38 的分母差异**：40 题中有 4 题各含 1 个 UNMAPPED 组（ARR-016/019/033/034），所以新工具下
  “全部证据组可用”的题数是 36；A1 投影 eligible 是 38（ARR-003、ARR-033 不可用）。逐题看：ARR-003 投影
  ineligible 但新映射 FULLY_MAPPED（+1），ARR-016/019/034 投影 eligible 但新映射 UNMAPPED（−3），
  ARR-033 两边都不可用（0），故 38 + 1 − 3 = 36。B3b 的映射率与上下文指标分母须以新映射的 36 为准，
  不能沿用 A1 的 38；ARR-003 在冻结题集中可以改为使用 `viewpoints` 作为 key（决策 B 后规则①可通过）。

## 6. `corpus_version` 口径（S9）

- Gold 的 `corpus_version = 1e48b401aa48fc59d94bba627deaa3005a7627860e4fe5815761d45c6ba68675` 来自
  `experiments/prepare_research_corpus.py`（Phase 0/1 的 `academic_corpus_manifest_v1.json`）：对按
  `file_name` 排序的 `[{file_name, document_id, sha256}]` 做 `json.dumps(ensure_ascii=False, sort_keys=True,
  separators=(",", ":"))` 后取 UTF-8 sha256。用当前 active manifest 的 `relative_path/document_id/content_hash`
  按该字段名与排序复算，结果正是 `1e48b401…`（已实测一致）。
- `artifacts/phase15a/active_manifest.json` 的 `corpus_signature = bb184726…` 来自
  `knowledge.manifest.build_corpus_signature`：对 `SourceRecord{document_id, relative_path, content_hash}`
  按 `(document_id, relative_path, content_hash)` 排序后的规范化 JSON 取 sha256，再与 pipeline identity
  组合成 `index_signature`。
- 语义差别：两者覆盖同一 10 篇 PDF、同一 sha256 值，但**身份字段名、排序键与序列化口径不同**——Gold 冻结的是
  “Phase 0/1 选中的论文集合（文件名 + document_id + 原始 PDF 字节 hash）”，active manifest 冻结的是
  “知识库 source 身份（相对路径 + 内容 hash）+ pipeline”，因此值不同且不可互换。Gold 的 `corpus_version`
  是历史沿用的数据集身份，当前代码没有为 `E2EDataset` 生成 `corpus_version` 的生产脚本。
- 结论：**未找到 e2e 题集专用的 corpus_version 生成脚本；B3b 冻结时需由主会话确认口径**（沿用 Gold 的
  `1e48b401…` 以保持与 qrels/Gold 连续，还是改用 active manifest 的 `bb184720…`）。映射文件已独立携带
  `collection_name` / `index_signature` / `chunk_count` / `chunk_id_hash`，不依赖该字段。

## 7. 已知限制

1. **断词折叠的副作用**：合法连字符复合词在行尾断开时（`self-\nattention`）会变成 `selfattention`；B3a-2 起草
   `answer_keys` 时应使用折叠后的形式或改用其他别名。
2. **`corpus_version` 口径未定**（§6）；B3b 冻结前必须确认。
3. **`map` 现在对 `SOURCE_EXTRACTION_FAILURE` 返回 2**：映射文件仍写出便于诊断，但调用方（B3b 冻结脚本）必须
   把退出码 2 当作“不可用于计分”处理，不能只看文件存在。
4. **4 组 UNMAPPED 是真实覆盖缺口**：ARR-016/019/034 是 chunk 与页眉/页码/公式的对齐问题，ARR-033 还叠加
   chunk 起点边界；B3b 应按 UNMAPPABLE 处理并保留题集，不改 `stable_mapping.py`（Gold v3 冻结口径）。
5. **136/817 chunk 非精确子串**：新映射器因此偏保守，部分组只能依赖唯一 context proof；该量化是后续
   mapping 覆盖率讨论的基线。
6. pilot 题集没有 claims，`validate_dataset` 会报 40 条 `answerable_required_claim`；这是 B3a-2 的输入状态，
   不代表工具失败。

## 8. B3a-2 / B3b 需要主会话或用户确认的事项

1. **corpus_version**：冻结题集沿用 Gold 的 `1e48b401…`，还是改用 active manifest 的 `bb184720…`（§6）。
2. **ARR-033 等 4 个 UNMAPPED 组**：B3b 是否接受“保留题集 + UNMAPPABLE 不计入上下文分母”，还是先在 B3a-2
   调整锚点（例如去掉 ARR-033 的 left_context 标题、把 ARR-016 的锚点改到可整块对齐的页 6/7 chunk）。
3. **answer_keys 写法**：确认 B3a-2 按决策 A 使用裸数值别名（如 `1.5`），按决策 B 对断词使用折叠后形式
   （如 `viewpoints`）；合法复合词避免依赖行尾折叠。
4. **冻结写入**：B3b 的 `freeze` 子命令尚未实现；普通 `validate` / `map` 已禁止写 `evals/`，冻结必须由新子命令
   完成并复用本轮的 fingerprint / LF / 身份校验。

## 9. 测试与冻结核对

- 相关 4 个测试文件：**51 passed, 1 xfailed**（xfail 为断词折叠已知局限，strict）。
- 全量：**587 passed, 1 xfailed**（B3a-1 前基线 572 passed；新增 15 个通过用例 + 1 个 xfail）。
- `artifacts/phase15a/index_state.json` sha256 = `5fbad4f0…63d8809`（未改）；`git status -- evals/
  artifacts/phase15a` 为空；`artifacts/stage_a/A1_exact_baseline.json` sha256 = `422fd629…a1411f`（未改）。
- 三个 pilot 产物均无 CR：`B3a1_pilot_dataset.json` sha256 `e91b2cde…`、`B3a1_pilot_mapping.json` sha256
  `972d6c19…`、`B3a1_pilot_mapping_check.json` sha256 `0a288abe…`（LF 固定后的字节 hash）。
