# EvidenceFlow Phase 15A Implementation Plan

## Status

Phase 15A-0 已通过，冻结基线为：

- active Collection：`ef_research_bge_m3_hybrid_c512_v2`
- source documents：10
- production chunks：817
- frozen snapshot：`artifacts/phase15a/baseline_snapshot.json`
- legacy Collection：`ef_research_bge_m3_hybrid_c512_v1`（只读保留）

15A-0 的证据是 `Settings`、`.env`、Chroma active metadata 三方一致，source/content
identity 与 Phase13F chunks 一致，Gold/qrels 与 Phase13F frozen artifact hash 未改变，
以及现有 v2 storage repair report 的 retrieval/IR parity PASS。

## FILES_TO_ADD

- `src/evidenceflow/knowledge/__init__.py`：导出 Phase 15A 公共接口。
- `src/evidenceflow/knowledge/manifest.py`：Domain manifest、canonical serialization、
  corpus/index signature、source diff 和 rebuild decision 纯逻辑。
- `src/evidenceflow/knowledge/job.py`：最小可持久化 BuildJob 状态模型与状态记录器。
- `src/evidenceflow/knowledge/lifecycle.py`：STAGING validation、active pointer、activation
  和 rollback；只更新生命周期状态，不修改 retrieval pipeline。
- `scripts/verify_phase15a.py`：基于 frozen snapshot 的只读 Phase 15A Gate 汇总与报告输出。
- `tests/test_phase15a_manifest.py`：确定性、敏感性、隔离性、source diff 和 rebuild decision。
- `tests/test_phase15a_lifecycle.py`：validation、staging isolation、activation、rollback、
  failed-build safety 和 BuildJob 状态。
- `docs/EVIDENCEFLOW_UI_RUNTIME_CONTRACT_v0.md`：只定义未来 Ask/Knowledge Base/Evaluation
  页面读取的 BuildJob、Run、RunEvent 数据契约，不实现 UI 或 event runtime。

## FILES_TO_MODIFY

- `PROJECT_SPEC.md`：记录 v2 authoritative baseline 和 Phase 15A 当前实现合同。
- `src/evidenceflow/storage/chroma.py`：增加新 staging Collection 所需的可选 metadata 注入，
  以及 validation 所需的只读 count/metadata/embedding dimension 查询；保持现有写入、查询和
  Domain 映射语义不变。

## FILES_NOT_TO_TOUCH

- `.env`、`data/source_docs/`、`data/chroma/` 中的现有 Collection。
- `src/evidenceflow/retrieval/`、`src/evidenceflow/rag/`、`src/evidenceflow/embedding/`、
  `src/evidenceflow/generation/`、`src/evidenceflow/context/`、`src/evidenceflow/citation/`。
- Phase13F、Gold、qrels 和既有 frozen evaluation artifacts。
- 前端、FastAPI、LangChain、LangGraph、Agent、Redis、PostgreSQL、RRF 或 BM25 production path。

## TESTS_TO_ADD

- manifest canonical ordering and repeated-run equality;
- corpus/index signature sensitivity and isolation;
- five source diff states plus ambiguous rename BLOCKED;
- deterministic rebuild decision;
- staging validation with missing/orphan/duplicate records;
- active pointer unchanged on failed validation;
- activation moves old active to previous;
- rollback restores previous pointer without a rebuild;
- BuildJob `FAILED` and `BLOCKED` remain distinct and persistable.

## Gate order

1. 只运行 manifest/source/rebuild 定向单测。
2. 只在临时测试存储中验证 staging、activation、rollback 和 BuildJob。
3. 运行 `python -m pytest -q`。
4. 复核 Phase13F frozen artifacts、baseline snapshot 和 v2 active smoke；不重建 production
   Collection，不进入 Phase 15B。
