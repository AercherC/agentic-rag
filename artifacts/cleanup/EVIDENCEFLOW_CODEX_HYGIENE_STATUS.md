# EvidenceFlow + Codex Hygiene Status

```text
EVIDENCEFLOW_STORAGE_REPAIR: PASS
ACTIVE_COLLECTION: ef_research_bge_m3_hybrid_c512_v2
LEGACY_COLLECTION: ef_research_bge_m3_hybrid_c512_v1 (READ_ONLY; BROKEN_METADATA)
STAGING_COLLECTION: ef_research_bge_m3_hybrid_c512_v2_staging (RETAINED)
CHUNK_COUNT: 817
METADATA_COVERAGE: 100%
DOMAIN_RECONSTRUCTION: PASS
RETRIEVAL_PARITY: PASS (40/40 questions)
IR_PARITY: PASS
FROZEN_INPUT_HASHES: PASS

ROOT_TESTS: 615 passed / 0 failed / 17 warnings
ACTIVE_INDEX_SMOKE: PASS

EVIDENCEFLOW_CLEANUP: PASS
PROJECT_DELETE_SAFE_FILES: 507 total across the two executions
PROJECT_BYTES_RECLAIMED: 12,819,288
PROJECT_REVIEW_ITEMS_RETAINED: 1,097
PROJECT_DELETE_SAFE_REMAINING: 0
SOURCE_AND_FROZEN_ARTIFACTS: UNCHANGED BY CLEANUP

CODEX_CLEANUP: PARTIAL (READ_ONLY AUDIT ONLY)
CODEX_HOME: D:\codex\app-data
CODEX_INVENTORY: 27,111 files / 2,151,728,221 bytes
CODEX_PROTECTED_FILES_CHANGED: 0 by this task
SESSIONS_CHANGED: 0 by this task (149 files observed)
ARCHIVED_SESSIONS_CHANGED: 0 by this task (0 files observed)
SAFE_CACHE_REMOVED: 0
PLAINTEXT_LOGS_ROTATED: 0
CODEX_RESTART: DEFERRED (Codex processes were running; no global runtime mutation was performed)
CODEX_RESUME_SMOKE: NOT RUN

STOPPED_AT: PROJECT_AND_CODEX_HYGIENE_CLEANUP
```

Detailed evidence:

- `storage_repair/storage_repair_report.json`: storage repair gates and parity evidence.
- `active_index_smoke.json`: active collection read-only smoke check.
- `project_cleanup_plan.json`: exact project cleanup allowlist and review set.
- `project_cleanup_snapshot.json`: final post-test deletion snapshot.
- `project_cleanup_execution.json`: final cleanup execution result.
- `codex/codex_cleanup_snapshot.json`: Codex runtime protection snapshot.
- `codex/codex_workspace_summary.md`: workspace artifact classification.
