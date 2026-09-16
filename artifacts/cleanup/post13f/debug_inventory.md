# Debug / Temporary Inventory

- `experiments/debug/audit_research_ingestion.py`；最后修改：2026-09-05T11:57:27.761370；引用计数：6；用途："""审计 Academic Research v1 在当前 PDF/C512/O64 Pipeline 下的兼容性。"""；唯一性：待人工确认；正式 runner 覆盖：未证明。
- `experiments/debug/chunk_id_verify.py`；最后修改：2026-09-05T11:57:27.760369；引用计数：6；用途："""临时验证：对 Chroma 中已持久化的 chunk 复算 chunk_id，验证确定性 ID 合同。；唯一性：待人工确认；正式 runner 覆盖：未证明。
- `experiments/debug/gen_eval.py`；最后修改：2026-09-05T11:57:27.762369；引用计数：10；用途："""临时评测：生成层（Generation-level）抽检。；唯一性：待人工确认；正式 runner 覆盖：未证明。
- `experiments/debug/gen_flowchart.py`；最后修改：2026-09-05T11:57:27.763368；引用计数：5；用途：# -*- coding: utf-8 -*-；唯一性：待人工确认；正式 runner 覆盖：未证明。
- `experiments/debug/inspect_retrieval.py`；最后修改：2026-09-09T13:08:45.327402；引用计数：27；用途："""开发者 Retrieval Inspector：定性查看真实 Dense 或正式多语融合结果。；唯一性：待人工确认；正式 runner 覆盖：未证明。
- `experiments/debug/inspect_vector_store.py`；最后修改：2026-09-09T13:08:51.913019；引用计数：10；用途："""开发者 Storage Inspector：只读查看 Chroma 中的 Chunk 文本与 Metadata。"""；唯一性：待人工确认；正式 runner 覆盖：未证明。

结论：debug Python 不进入 DELETE_SAFE；需人工确认后再归档或删除。
