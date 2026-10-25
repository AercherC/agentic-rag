"""Fake-based tests for ``scripts/e2e_dense.py``: no real model is loaded and no network access is used."""

import json

import pytest

import scripts.e2e_dense as module


class _FakeChunk:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id


class _FakeResult:
    def __init__(self, chunk_id):
        self.chunk = _FakeChunk(chunk_id)


class _FakeDense:
    """Record the received query and top_k, returning chunk_ids by a fixed rule."""

    def __init__(self, *, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def retrieve(self, query, top_k=None):
        self.calls.append((query, top_k))
        if self.fail_on is not None and query == self.fail_on:
            raise RuntimeError("dense boom")
        return [_FakeResult(f"c-{query}-{rank}") for rank in range(1, top_k + 1)]


class _FakeSettings:
    """Need only provide the ``model_device`` field the script reads."""

    model_device = "cpu"


def _patch_runtime(monkeypatch, dense):
    """Replace build_retriever and device resolution to avoid loading models."""

    class _FakeRetriever:
        dense_retriever = dense

    monkeypatch.setattr(module, "build_retriever", lambda settings: _FakeRetriever())
    monkeypatch.setattr(module, "Settings", _FakeSettings)
    monkeypatch.setattr(module, "resolve_model_device", lambda value: "cpu")


def _write_items(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_dense_outputs_top_level_query_to_chunk_ids(tmp_path, monkeypatch):
    dense = _FakeDense()
    _patch_runtime(monkeypatch, dense)
    items = _write_items(tmp_path / "items.jsonl", [
        {"query": "What is the chamfer distance?", "decision": "accepted"},
        {"query": "How many layers are used?", "decision": "accepted"},
    ])
    out = tmp_path / "dense.json"
    assert module.main(["--queries", str(items), "--out", str(out), "--top-k", "3"]) == 0

    ranks = json.loads(out.read_text(encoding="utf-8"))
    # The top level must be exactly {query: [chunk_id...]}, matching how gates --dense reads it.
    assert set(ranks) == {"What is the chamfer distance?", "How many layers are used?"}
    assert ranks["What is the chamfer distance?"] == [
        "c-What is the chamfer distance?-1", "c-What is the chamfer distance?-2",
        "c-What is the chamfer distance?-3",
    ]
    assert dense.calls == [("What is the chamfer distance?", 3), ("How many layers are used?", 3)]
    # Fixed LF: the same bytes across operating systems.
    assert b"\r\n" not in out.read_bytes()


def test_dense_deduplicates_queries_and_skips_dry_run(tmp_path, monkeypatch):
    dense = _FakeDense()
    _patch_runtime(monkeypatch, dense)
    items = _write_items(tmp_path / "items.jsonl", [
        {"query": "Repeated question?", "decision": "accepted"},
        {"query": "Repeated question?", "decision": "accepted"},
        {"query": "<DRY-RUN>", "decision": "accepted"},
        {"query": "", "decision": "discarded"},
    ])
    out = tmp_path / "dense.json"
    assert module.main(["--queries", str(items), "--out", str(out), "--top-k", "2"]) == 0
    # The same question text is retrieved only once; the --dry-run placeholder text takes no part
    # in retrieval.
    assert [call[0] for call in dense.calls] == ["Repeated question?"]


def test_dense_refuses_non_cpu_without_flag(tmp_path, monkeypatch, capsys):
    _patch_runtime(monkeypatch, _FakeDense())
    monkeypatch.setattr(module, "resolve_model_device", lambda value: "cuda")
    items = _write_items(tmp_path / "items.jsonl", [{"query": "Q?", "decision": "accepted"}])
    out = tmp_path / "dense.json"
    assert module.main(["--queries", str(items), "--out", str(out)]) == 1
    assert "allow-gpu" in capsys.readouterr().err
    assert not out.exists()


def test_dense_allow_gpu_overrides_device_check(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch, _FakeDense())
    monkeypatch.setattr(module, "resolve_model_device", lambda value: "cuda")
    items = _write_items(tmp_path / "items.jsonl", [{"query": "Q?", "decision": "accepted"}])
    out = tmp_path / "dense.json"
    assert module.main(["--queries", str(items), "--out", str(out), "--allow-gpu", "--top-k", "1"]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"Q?": ["c-Q?-1"]}


def test_dense_refuses_evals_output(tmp_path, monkeypatch, capsys):
    _patch_runtime(monkeypatch, _FakeDense())
    items = _write_items(tmp_path / "items.jsonl", [{"query": "Q?", "decision": "accepted"}])
    target = module.EVALS_ROOT / "_b3_never_created" / "dense.json"
    assert module.main(["--queries", str(items), "--out", str(target)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not target.exists()


def test_dense_rejects_bad_top_k(tmp_path, monkeypatch, capsys):
    _patch_runtime(monkeypatch, _FakeDense())
    items = _write_items(tmp_path / "items.jsonl", [{"query": "Q?", "decision": "accepted"}])
    out = tmp_path / "dense.json"
    assert module.main(["--queries", str(items), "--out", str(out), "--top-k", "0"]) == 1
    assert not out.exists()


def test_dense_rejects_malformed_jsonl(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch, _FakeDense())
    items = tmp_path / "items.jsonl"
    items.write_text("{not json}\n", encoding="utf-8")
    out = tmp_path / "dense.json"
    assert module.main(["--queries", str(items), "--out", str(out)]) == 1
    assert "is not valid JSON" in _captured_error(monkeypatch, tmp_path, items, out)


def _captured_error(monkeypatch, tmp_path, items, out):
    """Helper: run once more and return the stderr text (for asserting error messages)."""

    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        module.main(["--queries", str(items), "--out", str(out)])
    return buffer.getvalue()
