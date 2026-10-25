"""Verify the deterministic rules for the storage contract and the cleanup boundary of this round."""

from scripts.audit_codex_runtime_state import classify_path, retention_action
from scripts.audit_project_cleanup import safe_candidate_reason


def test_codex_deny_paths_override_safe_allowlist() -> None:
    """Sessions, plugins, and SQLite must never become safe candidates, even when the name contains "cache"."""

    assert classify_path("sessions/thread.jsonl") == "C-PROTECT"
    assert classify_path("plugins/cache/package/file.bin") == "C-PROTECT"
    assert classify_path("cache/state.sqlite3") == "C-PROTECT"
    assert classify_path("logs_2.sqlite") == "C-REVIEW"
    assert classify_path("cache/render.png") == "C-SAFE-CANDIDATE"


def test_codex_safe_retention_policy() -> None:
    """Only an exact SAFE_CANDIDATE takes part in the 7/30-day retention policy."""

    assert retention_action(1, "C-SAFE-CANDIDATE") == "KEEP"
    assert retention_action(7, "C-SAFE-CANDIDATE") == "DELETE_CANDIDATE"
    assert retention_action(31, "C-SAFE-CANDIDATE") == "DELETE_SAFE_CANDIDATE"
    assert retention_action(365, "C-PROTECT") == "DO_NOT_AUTO_DELETE"


def test_project_cleanup_allowlist_is_narrow() -> None:
    """Project cleanup accepts only the cache/pytest temp trees, never source code, Gold, or Chroma."""

    assert safe_candidate_reason("src/evidenceflow/__pycache__/module.pyc")
    assert safe_candidate_reason("tmp/pytest-root/pytest-1/result.pdf")
    assert safe_candidate_reason("evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_c/chroma/chroma.sqlite3")
    assert safe_candidate_reason("src/evidenceflow/runtime.py") is None
    assert safe_candidate_reason("data/chroma/chroma.sqlite3") is None
    assert safe_candidate_reason("evals/evaluation_v3/standard_ir_v1/qrels.json") is None
