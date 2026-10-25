"""Dataset, key-requirement matching, and evidence-mapping tooling for the rag_e2e_v1 end-to-end evaluation (PROJECT_SPEC §13.1).

This package belongs to the evaluation layer: it may depend on runtime / storage / domain,
but online modules such as runtime and retrieval must never import it back
(guarded by `tests/test_architecture_boundaries.py`).
"""
