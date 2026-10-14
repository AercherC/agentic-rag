# EvidenceFlow Phase 15C.1 — Paused

## Status

**STOP — user-requested pause.** CPU model inference was interrupted after the Exact index,
NumPy reference conformance, candidate pools, and E20/E30/E40 intermediate reranker artifacts
were saved. Production was not modified.

## Completed before pause

- G0 current Production D20 replay: PASS.
- Active Chroma identity: 10 documents / 817 chunks / metadata coverage 100%.
- In-memory Exact Dense index built from active Chroma embeddings.
- Independent NumPy reference: 40/40 query IDs and scores matched the current
  `numpy.exact` backend.
- E20/E30/E40 candidate artifacts and reranker outputs are present.

## Not completed

- HNSW-vs-Exact report and ARR-032/ARR-033 focus diagnosis.
- 10 fresh-process Candidate Top30 / Top5 / metrics determinism gate.
- Final latency aggregate, cleanup manifest, execution result and final PASS/STOP decision.

FAISS is unavailable in the current Windows `agent` environment, so the current exact backend
is explicitly `numpy.exact`; this cannot satisfy the FAISS-specific G1 until that dependency is
available. No Production promotion was attempted.

Resume from the HNSW-vs-Exact audit in `experiments/run_phase15c1_exact_dense_architecture.py`.
