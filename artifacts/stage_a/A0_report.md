# A0 报告：冻结起点与重建基线

## 结论：Gate A0 通过

| Gate | 结果 |
|---|---|
| pytest 全部通过 | 通过，**N0 = 534**（见下方“偏差 1”） |
| 索引身份检查 | 通过：817 chunk，`pass: true`，lifecycle=ACTIVE，index_signature 与计划一致 |
| HNSW D20 回放与冻结 B0 的 40 题 Top5 逐题一致 | 通过：`parity: true`，`run_mismatches=[]`，5 项指标差异为 0 |

chunk ID hash `e87a613e…` 由 `compare_replay` 的 `identity_matches` 校验（collection_name、corpus_signature、index_signature、chunk_id_hash 全部一致）。HNSW 在 D20 下本次没有出现不一致。

## 1. 修改或创建的文件
| 文件 | 职责 |
|---|---|
| `CLAUDE.md` | 一行指针，指向 AGENTS.md 与阶段 A 计划 |
| `docs/agentic_rag/01_ARCHITECTURE_REVIEW.md`、`02_STAGE_A_EXECUTION_PLAN.md` | 从 `Claude outputs/` 复制而来（见偏差 2） |
| `experiments/stage_a/a0_compare_replay.py` | 复用 `compare_replay`、`load_frozen_inputs`，比较回放与冻结 B0，并记录环境 |
| `tests/test_embedding.py` | 修复一个依赖运行环境的测试（偏差 1） |
| `artifacts/stage_a/A0_*` | pytest 日志、两个 verify 输出、回放 JSON、比较结果、环境信息 |

提交：`87fa8ad stage-a(A0): snapshot before stage A`（已打 tag `stage-a-start`），随后是本步的报告提交。快照未包含 `1.26`、`src/evidenceflow.zip`、`.env`、`Claude outputs/`。

## 2. 实现内容与原因
按计划做了快照、CLAUDE.md、全量测试、索引检查、冻结 B0 回放与逐题比较、环境记录。回放用 `MODEL_DEVICE=cpu`、离线 HF、本地 reranker 目录，耗时约 4 分 34 秒。回放脚本不创建新的冻结文件，输出在 `artifacts/stage_a/`。

## 3. 与计划的偏差（需要用户知悉）
1. **一个测试在 GPU 机器上失败，已修复。** 快照后首次运行为 `1 failed, 533 passed`。失败用例 `tests/test_embedding.py::test_provider_loads_once_batches_documents_and_returns_python_floats` 使用默认 `device="auto"`，本机有 CUDA，解析为 `cuda:0`，而断言写死了 `"cpu"`。这是测试对环境的依赖，不是生产代码缺陷。我只把该用例构造 provider 时改为显式 `device="cpu"`，断言本身没有放宽。这个失败的改动在 A0 快照之后，属于 A0 之后的一次小修。
2. **计划文档路径不一致。** 计划引用 `docs/agentic_rag/`，而文件实际在 `Claude outputs/`。我把两份文档复制到了 `docs/agentic_rag/`，原目录未动、未提交。
3. 新增了 `experiments/stage_a/a0_compare_replay.py`，因为计划要求“写一段小脚本”。

## 4. 测试
- `python -m pytest -q`：**534 passed**，约 28 秒（最终结果）。
- `verify_active_index.py`：pass。`verify_phase15a.py`：status PASS。
- 回放比较：parity true，40/40 题一致。

## 5. 环境（`A0_env.json`）
torch 2.13.0+cu130，CUDA 可用；chromadb 1.5.9；**faiss 不可 import**；Python 3.12.14。

## 6. 限制与待确认
- faiss 不可用。A1 将按计划采用 `numpy.exact`，需要你确认接受它作为正式后端（对 SPEC §42 的修订）；若坚持 FAISS，则需要你同意安装 `faiss-cpu`。
- `Claude outputs/` 目录仍是未跟踪状态，未处理。
