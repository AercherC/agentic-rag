# STOP_REASON

- **STOP_REASON**：生产 active Chroma collection 无法满足当前 Domain chunk 读取契约。
- **AFFECTED_FILES**：`data/chroma/chroma.sqlite3` 中的 `ef_research_bge_m3_hybrid_c512_v1`；`src/evidenceflow/storage/chroma.py` 的读取边界；`experiments/phase13c/runners/run_phase13f_runtime_promotion.py` 的写入来源。
- **EXPECTED**：active collection 的 817 个 chunk 均带有 `_restore_chunk` 所需的 Chroma metadata，`iter_all_chunks()` 可成功恢复 Domain chunk。
- **ACTUAL**：直接读取 active collection 返回 `metadata=[None, None]`；`count=817`；`ChromaVectorStore.iter_all_chunks()` 抛出 `VectorStoreQueryError: Stored chunk metadata is invalid`。Phase 13F 写入代码调用 `collection.add(...)` 时未传入 `metadatas`。
- **RESTORE_ACTION**：本轮不删除、不移动、不重建、不改写 active 或 rollback collection。后续应在单独授权的数据迁移/运行时修复任务中，统一修复写入契约或按批准方案重建 active collection，并重新执行 runtime/IR/frozen gates。
- **STATUS**：停止于 `PROJECT_HYGIENE_CLEANUP`；物理动作 0；回收空间 0；删除 Python 文件 0。
