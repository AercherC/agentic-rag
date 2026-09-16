# Config Consistency Report

| Key | .env | .env.example | config.py evidence |
|---|---|---|---|
| `CHUNK_STRATEGY` | `hybrid` | `hybrid` | `True` |
| `COLLECTION_NAME` | `ef_research_bge_m3_hybrid_c512_v2` | `ef_research_bge_m3_hybrid_c512_v2` | `True` |
| `CHUNK_SIZE` | `512` | `512` | `True` |
| `CHUNK_OVERLAP` | `64` | `64` | `True` |
| `TOP_K` | `5` | `5` | `True` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | `BAAI/bge-m3` | `True` |
| `MAX_CONTEXT_TOKENS` | `<REDACTED>` | `<REDACTED>` | `True` |

## Findings

- `.env` 与 `.env.example` 已指向 Phase 13F hybrid collection；`.env` 是当前运行时实际配置来源。
- `src/evidenceflow/config.py` 的 `Settings.collection_name` 硬编码默认仍为 `ef_bge_m3_hybrid_c512_v1`，与方案要求的 `ef_research_bge_m3_hybrid_c512_v1` 不一致；本轮禁止改 production source，标记 REVIEW。
- `README.md` 和 `PROJECT_SPEC.md` 仍含 fixed collection、旧指标或旧阶段描述；历史文档需区分后再修订，本轮不改写。
- `TOP_K` 虽存在 Settings 字段，但默认 Runtime 实际由 `reranker_input_k=20` 与 `reranker_final_k=5` 控制；其余消费者需由 reachability/reference graph 继续约束。
- 过期引用检测：`README.md` contains `ef_research_bge_m3_c512_o64_v1`；`PROJECT_SPEC.md` contains `408 passed`；`PROJECT_SPEC.md` contains `Phase 11C`
