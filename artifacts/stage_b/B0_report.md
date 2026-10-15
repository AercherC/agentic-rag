# Stage B · B0 报告：起点确认与小修

## 起点
- `stage-a-done` = `1516cd9`；计划文档单独提交 `aa0ef91`（docs(stage-b): add stage B execution plan），
  `git log stage-a-done..HEAD` 只有这一条。
- 轻量 tag `stage-b-start` → `aa0ef91`（B0 代码改动之前）。

## sha256 核对
| 文件 | 期望 | 结果 |
|---|---|---|
| `artifacts/phase15a/index_state.json` | `5fbad4f0…63d8809` | 一致（起点与改动后各核对一次） |
| `artifacts/stage_a/A1_exact_baseline.json` | `422fd629…a1411f` | 一致 |

## 1. 修改/创建的文件
| 文件 | 作用 |
|---|---|
| `PROJECT_SPEC.md` §12.4 | 写明 `verify_active_index.py` 不创建 Collection；持久化目录不存在直接判失败、不创建目录 |
| `scripts/verify_active_index.py` | 目录不存在时不构造 `PersistentClient` 直接返回失败；`ChromaVectorStore(..., create_if_missing=False)`；docstring 更新 |
| `tests/test_verify_active_index_script.py` | +3 测试：目录缺失不建目录；空库查不存在的 collection 不会新建；monkeypatch 断言 `create_if_missing=False` |
| `experiments/stage_b/probe_judge.py` | 一次性评委连通性/结构化模式探测（不被 pytest 收集；`experiments/stage_a/` 无 `__init__.py`，此处同样不加） |
| `artifacts/stage_b/B0_replay_runtime.json` | CPU 回放结果 |
| `artifacts/stage_b/B0_replay_vs_b0prime.json` | 回放与 B0' 比对 |
| `artifacts/stage_b/B0_judge_probe.json` | 评委探测结果 |

## 2. 为什么这样实现
`chromadb.PersistentClient` 会在目录不存在时创建目录，`ChromaVectorStore` 默认 `create_if_missing=True` 且会 `mkdir`；
只读 smoke 不应有任何写副作用，因此在入口提前判目录、并沿用 runtime 已有的 `create_if_missing=False` 语义。
SPEC 先改、代码后改。调用方 grep 确认无遗漏（只有本脚本与 runtime 使用）。

## 3. 回放比对（MODEL_DEVICE=cpu，离线）
| 字段 | 结果 |
|---|---|
| parity | true |
| reranker_score_mismatches | [] |
| top5_mismatches | [] |
| top20_candidate_id_mismatches | [] |
| top20_candidate_score_mismatches | [] |
| replay_device | cpu |
| replay_collection_name | ef_research_bge_m3_hybrid_c512_v2 |

两个参数均显式传入；运行后 `git status` 确认无被跟踪文件被改写。结论：通过。

## 4. 评委探测
judge `open.bigmodel.cn` / `glm-4-flash`；generator `api.deepseek.com` / `deepseek-flash`；same_vendor = false。

| 模式 | 请求成功 | json.loads + Pydantic(extra=forbid) | 耗时 | 备注 |
|---|---|---|---|---|
| json_object | 是 | 通过 | 3.7 s | 裸 JSON |
| json_schema (strict) | 是 | 失败 | 1.8 s | 回复被 ```json 代码块包裹；prompt_tokens 71 < json_object 的 123，疑似服务端未真正执行 schema 约束 |

- 推荐模式：**json_object**。
- 真实请求总数：**2**（`max_attempts=1`，无瞬态错误，未发第 3 次）。
- Key 泄漏检查：产物、脚本、脚本 stdout 均不含 key 值（只做比较，未打印）。

## 5. 测试
- 基线：507 passed。
- B0 后：**N_B0 = 510 passed**（+3）。

## 6. 遗留问题
- glm-4-flash 对 `json_schema` 不报错但不按约束输出（带代码块），后续评委应使用 `json_object`；若想用 json_schema，需在解析侧剥离代码块，但不推荐。
- 注册域判断是简化实现（末两段/常见二级后缀三段），对当前 host 足够。
