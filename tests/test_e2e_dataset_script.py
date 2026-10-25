"""Verify the output guards, exit codes and read-only mapping path of scripts/e2e_dataset.py (all inside temporary directories)."""

import json

import pytest

import scripts.e2e_dataset as module
from evidenceflow.config import Settings
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.eval.e2e.mapping import compute_chunk_id_hash
from evidenceflow.ingestion.loader import load_document
from scripts.e2e_dataset import main

REAL_EVALS_FILE = module.ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json"


@pytest.fixture
def no_chroma(monkeypatch):
    """Any call that reads Settings, resolves the active pointer or connects to Chroma makes the test fail."""
    calls = []

    def forbid(*args, **kwargs):
        calls.append(args)
        raise AssertionError("must not be called before output refusal")

    monkeypatch.setattr(module, "ChromaVectorStore", forbid)
    monkeypatch.setattr(module, "resolve_active_collection", forbid)
    monkeypatch.setattr(module, "Settings", forbid)
    monkeypatch.setattr(module, "load_source_documents", forbid)
    return calls


@pytest.mark.parametrize("command", ["validate", "map"])
def test_refuses_existing_output_without_overwrite(tmp_path, capsys, no_chroma, command):
    out = tmp_path / "out.json"
    out.write_text("keep", encoding="utf-8")
    flag = "--report" if command == "validate" else "--out"
    assert main([command, "--dataset", str(tmp_path / "missing.json"), flag, str(out)]) == 2
    assert "REFUSED:" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "keep" and no_chroma == []


@pytest.mark.parametrize("command", ["validate", "map"])
def test_refuses_existing_evals_file_even_with_overwrite(tmp_path, capsys, no_chroma, command):
    assert REAL_EVALS_FILE.is_file()
    before = REAL_EVALS_FILE.read_bytes()
    flag = "--report" if command == "validate" else "--out"
    code = main([command, "--dataset", str(tmp_path / "missing.json"), flag, str(REAL_EVALS_FILE), "--overwrite"])
    assert code == 2
    assert "evals/" in capsys.readouterr().err
    assert REAL_EVALS_FILE.read_bytes() == before and no_chroma == []


def test_refuses_nonexistent_output_under_evals(tmp_path, capsys, no_chroma):
    """Any path under `evals/` is refused, even when the target file does not exist yet (a tool must never write into the frozen directory)."""
    target = module.ROOT / "evals" / "_b3a1_test_never_created" / "out.json"
    assert not target.exists()
    assert main(["validate", "--dataset", str(tmp_path / "missing.json"), "--report", str(target)]) == 2
    assert "evals/" in capsys.readouterr().err
    assert not target.exists() and no_chroma == []


def test_write_json_fixes_lf_newlines(tmp_path):
    """JSON artifacts are pinned to LF so Windows CRLF cannot make dataset.sha256 drift from OS to OS."""
    path = tmp_path / "nested" / "out.json"
    module.write_json(path, {"b": 2, "a": [1]})
    data = path.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert json.loads(data.decode("utf-8")) == {"a": [1], "b": 2}


def _source(tmp_path):
    root = tmp_path / "docs"
    root.mkdir(exist_ok=True)
    (root / "paper.txt").write_text("SnowflakeNet reports a Chamfer distance of 7.21 on PCN.", encoding="utf-8")
    return root, load_document(root / "paper.txt", root)


def _dataset(tmp_path, doc, keys):
    anchor = {"document_id": doc.document_id, "content_version": doc.content_hash,
              "evidence_anchor": "Chamfer distance of 7.21", "locator": {"block_index": 0}}
    case = {"case_id": "E2E-M001", "family_id": "f", "query": "What CD?", "query_language": "en",
            "question_type": "numeric", "tags": [], "critical": False, "expected_status": "answered",
            "acceptable_abstention": False, "required_evidence_groups": [{"group_id": "G1", "alternatives": [anchor]}],
            "required_claims": [{"claim_id": "C1", "text": "CD 7.21", "required": True,
                                 "evidence_anchors": [anchor], "answer_keys": keys}],
            "forbidden_claims": [], "unanswerable_kind": None,
            "provenance": {"drafted_by": "agent", "approved_by": None, "source": "test"}}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"dataset_id": "d", "corpus_version": "c", "cases": [case]}), encoding="utf-8")
    return path


def test_validate_exit_codes(tmp_path):
    root, doc = _source(tmp_path)
    good = _dataset(tmp_path, doc, [["7.21"]])
    report = tmp_path / "r.json"
    assert main(["validate", "--dataset", str(good), "--report", str(report), "--source-root", str(root)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True
    # With --require-approved an unapproved case counts as a problem -> 2;
    # an existing report needs --overwrite.
    args = ["validate", "--dataset", str(good), "--report", str(report), "--source-root", str(root),
            "--require-approved", "--overwrite"]
    assert main(args) == 2
    assert json.loads(report.read_text(encoding="utf-8"))["issues"][0]["rule"] == "not_approved"
    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    assert main(["validate", "--dataset", str(broken), "--report", str(tmp_path / "x.json"),
                 "--source-root", str(root)]) == 1


def test_map_reads_active_pointer_and_opens_readonly(tmp_path, monkeypatch):
    root, doc = _source(tmp_path)
    dataset = _dataset(tmp_path, doc, [["7.21"]])
    state = tmp_path / "index_state.json"
    pointer = {"chunk_count": 1, "collection_name": "active_col", "document_count": 1,
               "index_signature": "sig-123", "knowledge_base_id": "kb", "manifest_path": "m.json", "status": "ACTIVE"}
    state.write_text(json.dumps({"active": pointer, "knowledge_base_id": "kb", "previous": None,
                                 "schema_version": "index-lifecycle-v1", "staging": None}), encoding="utf-8")
    db = tmp_path / "chroma"
    db.mkdir()
    monkeypatch.setattr(module, "Settings", lambda: Settings(_env_file=None, index_state_path=str(state),
                                                             vector_db_path=str(db), source_docs_path=str(root)))
    opened = []

    class FakeStore:
        def __init__(self, path, name, *, create_if_missing=True):
            opened.append((name, create_if_missing))

        def iter_all_chunks(self):
            yield DocumentChunk(chunk_id="k1", document_id=doc.document_id, content_hash=doc.content_hash,
                                file_name="paper.txt", content=doc.segments[0].content, token_count=5,
                                chunk_index=0, page_start=None, page_end=None)

    monkeypatch.setattr(module, "ChromaVectorStore", FakeStore)
    out = tmp_path / "mapping.json"
    assert main(["map", "--dataset", str(dataset), "--out", str(out)]) == 0
    assert opened == [("active_col", False)]
    mapping = json.loads(out.read_text(encoding="utf-8"))
    assert mapping["index_signature"] == "sig-123" and mapping["collection_name"] == "active_col"
    # The mapping file also records the measured chunk identity, so `score` can
    # run its four-item identity check.
    assert mapping["chunk_count"] == 1 and mapping["chunk_id_hash"] == compute_chunk_id_hash(["k1"])
    assert mapping["groups"][0]["status"] == "FULLY_MAPPED" and mapping["groups"][0]["chunk_sets"] == [["k1"]]
    assert mapping["claim_anchors"][0]["kind"] == "required"


def _install_store(monkeypatch, tmp_path, root, doc, content, *, chunk_count=1, document_count=1):
    """Install a read-only FakeStore plus a lifecycle state file; the parameters set the pointer counts."""

    state = tmp_path / "index_state.json"
    pointer = {"chunk_count": chunk_count, "collection_name": "active_col", "document_count": document_count,
               "index_signature": "sig-123", "knowledge_base_id": "kb", "manifest_path": "m.json", "status": "ACTIVE"}
    state.write_text(json.dumps({"active": pointer, "knowledge_base_id": "kb", "previous": None,
                                 "schema_version": "index-lifecycle-v1", "staging": None}), encoding="utf-8")
    db = tmp_path / "chroma"
    db.mkdir()
    monkeypatch.setattr(module, "Settings", lambda: Settings(_env_file=None, index_state_path=str(state),
                                                             vector_db_path=str(db), source_docs_path=str(root)))

    class FakeStore:
        def __init__(self, path, name, *, create_if_missing=True):
            assert create_if_missing is False

        def iter_all_chunks(self):
            yield DocumentChunk(chunk_id="k1", document_id=doc.document_id, content_hash=doc.content_hash,
                                file_name="paper.txt", content=content, token_count=5,
                                chunk_index=0, page_start=None, page_end=None)

    monkeypatch.setattr(module, "ChromaVectorStore", FakeStore)


# ---------- freeze (the only write entry point of B3b) ----------

def _approved_dataset(tmp_path, doc):
    """Build a case set that passes every freeze check: approved by machine, carries an approval_basis, and is grounded."""

    anchor = {"document_id": doc.document_id, "content_version": doc.content_hash,
              "evidence_anchor": "Chamfer distance of 7.21", "locator": {"block_index": 0}}
    case = {"case_id": "E2E-M001", "family_id": "f", "query": "What CD?", "query_language": "en",
            "question_type": "numeric", "tags": [], "critical": False, "expected_status": "answered",
            "acceptable_abstention": False,
            "required_evidence_groups": [{"group_id": "G1", "alternatives": [anchor]}],
            "required_claims": [{"claim_id": "C1", "text": doc.segments[0].content,
                                 "required": True, "evidence_anchors": [anchor],
                                 "answer_keys": [["7.21"]]}],
            "forbidden_claims": [], "unanswerable_kind": None,
            "provenance": {"drafted_by": "agent", "approved_by": "machine",
                           "approval_basis": "B3_gate_report.json#abc", "source": "derived_t2"}}
    path = tmp_path / "approved.json"
    path.write_text(json.dumps({"dataset_id": "d", "corpus_version": "c", "cases": [case]}), encoding="utf-8")
    return path


def test_freeze_writes_dataset_sha256_and_readme(tmp_path):
    import hashlib

    root, doc = _source(tmp_path)
    dataset = _approved_dataset(tmp_path, doc)
    target = tmp_path / "frozen"
    assert main(["freeze", "--dataset", str(dataset), "--target", str(target),
                 "--source-root", str(root)]) == 0
    assert (target / "dataset.json").is_file()
    assert (target / "dataset.sha256").is_file()
    assert (target / "README.md").is_file()
    # dataset.sha256 must match the **bytes actually written to disk** (fixed LF included).
    data = (target / "dataset.json").read_bytes()
    assert b"\r" not in data
    recorded = (target / "dataset.sha256").read_text(encoding="utf-8").split()[0]
    assert recorded == hashlib.sha256(data).hexdigest()
    # The README records the fingerprint and the checksum so the freeze can be re-verified later.
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert recorded in readme and "fingerprint" in readme


def test_freeze_refuses_existing_target(tmp_path, capsys):
    root, doc = _source(tmp_path)
    dataset = _approved_dataset(tmp_path, doc)
    target = tmp_path / "frozen"
    target.mkdir()
    (target / "keep.txt").write_text("keep", encoding="utf-8")
    assert main(["freeze", "--dataset", str(dataset), "--target", str(target),
                 "--source-root", str(root)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    # When the target already exists, no frozen artifact is ever written.
    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (target / "dataset.json").exists()


def test_freeze_refuses_dataset_that_fails_validation(tmp_path, capsys):
    root, doc = _source(tmp_path)
    # A case set that is not approved (approved_by=null) must not be frozen.
    dataset = _dataset(tmp_path, doc, [["7.21"]])
    target = tmp_path / "frozen"
    assert main(["freeze", "--dataset", str(dataset), "--target", str(target),
                 "--source-root", str(root)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not target.exists()


def test_map_with_no_usable_records_exits_2_but_writes_file(tmp_path, monkeypatch, capsys):
    """When every anchor is unusable the mapping file is still written for diagnosis, but the exit code must be non-zero (2)."""

    root, doc = _source(tmp_path)
    dataset = _dataset(tmp_path, doc, [["7.21"]])
    _install_store(monkeypatch, tmp_path, root, doc, "unrelated chunk text")
    out = tmp_path / "mapping.json"
    assert main(["map", "--dataset", str(dataset), "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert "usable=0 unusable=2 source_failures=0" in captured.out
    assert "REFUSED:" in captured.err and "FAILED:" not in captured.err
    assert out.is_file()


@pytest.mark.parametrize("chunk_count,document_count,needle", [
    (2, 1, "chunk_count mismatch"),
    (1, 2, "document_count mismatch"),
])
def test_map_identity_mismatch_exits_1(tmp_path, monkeypatch, capsys, chunk_count, document_count, needle):
    """When the active pointer and the real chunk set disagree the run must fail (exit 1) and write no mapping with a wrong identity."""
    root, doc = _source(tmp_path)
    dataset = _dataset(tmp_path, doc, [["7.21"]])
    _install_store(monkeypatch, tmp_path, root, doc, doc.segments[0].content,
                   chunk_count=chunk_count, document_count=document_count)
    out = tmp_path / "mapping.json"
    assert main(["map", "--dataset", str(dataset), "--out", str(out)]) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and "FAILED:" in captured.err
    assert not out.exists()
