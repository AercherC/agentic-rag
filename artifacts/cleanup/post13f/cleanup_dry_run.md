# Post-13F Cleanup Dry Run

本报告只记录候选，不执行删除、移动或生产源码修改。

- 扫描受控文件：1436 个；输出目录和模型/session cache 按边界排除。
- DELETE_SAFE 候选：245 个（可再生缓存与 pytest 临时产物）。
- DELETE_REVIEW 候选：6 个（debug Python，禁止自动删除）。
- REVIEW：1087 个；包含历史评测、文档和未能证明唯一性的产物。
- runtime fingerprint：`dd2c80423cf5a811c6195cc2cf5dfd15658d29db24271f86b433fb9bf6701c72`；与 Phase 13F 预期一致：False。
- frozen manifest 完整性：True。
- `.cache`（模型缓存）未进入扫描删除范围；未删除任何缓存、Chroma、Python、Gold、qrels 或冻结证据。

## Chroma

- `ef_research_bge_m3_c512_o64_refmeta_v1`：EXPERIMENTAL，documents=10，chunks=361，rebuildable=True。
- `ef_research_bge_m3_c512_o64_v1`：ROLLBACK，documents=10，chunks=361，rebuildable=True。
- `ef_research_bge_m3_hybrid_c512_v1`：ACTIVE，documents=10，chunks=817，rebuildable=True。
- `ef_research_bge_m3_hybrid_c512_v2`：UNKNOWN，documents=10，chunks=817，rebuildable=True。
- `ef_research_bge_m3_hybrid_c512_v2_staging`：UNKNOWN，documents=10，chunks=817，rebuildable=True。
- `ef_research_structure_v1`：EXPERIMENTAL，documents=10，chunks=385，rebuildable=True。
- `ef_research_structure_v1_elemaware`：EXPERIMENTAL，documents=10，chunks=935，rebuildable=True。

## 当前边界

- Phase 14 仅逻辑归档，未物理移动。
- `src/evidenceflow/eval/**`、debug runner、历史 RRF/BM25/Translation 仅生成分类证据，未迁移或删除。
- 下一步若要执行 DELETE_SAFE 或人工归档，需在新指令中明确批准；本轮 STOP AT `PROJECT_HYGIENE_CLEANUP`。
