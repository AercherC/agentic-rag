"""验证 scripts/e2e_dataset.py 的输出保护、退出码与只读映射路径（全部在临时目录中）。"""

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
    """任何读取 Settings、解析 active 或连接 Chroma 的调用都会让测试失败。"""
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
    """`evals/` 下任何路径都拒绝，即使目标文件尚不存在（冻结目录不得由工具写入）。"""
    target = module.ROOT / "evals" / "_b3a1_test_never_created" / "out.json"
    assert not target.exists()
    assert main(["validate", "--dataset", str(tmp_path / "missing.json"), "--report", str(target)]) == 2
    assert "evals/" in capsys.readouterr().err
    assert not target.exists() and no_chroma == []


def test_write_json_fixes_lf_newlines(tmp_path):
    """JSON 产物固定 LF，避免 Windows 的 CRLF 让 dataset.sha256 随 OS 漂移。"""
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
    # --require-approved 时未确认的题算问题 → 2；已存在的报告需 --overwrite。
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
    # 映射文件同时记录实测 chunk 身份，供 score 做四项身份校验。
    assert mapping["chunk_count"] == 1 and mapping["chunk_id_hash"] == compute_chunk_id_hash(["k1"])
    assert mapping["groups"][0]["status"] == "FULLY_MAPPED" and mapping["groups"][0]["chunk_sets"] == [["k1"]]
    assert mapping["claim_anchors"][0]["kind"] == "required"


def _install_store(monkeypatch, tmp_path, root, doc, content, *, chunk_count=1, document_count=1):
    """装一个只读 FakeStore 与 lifecycle 状态文件；指针计数由参数控制。"""

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


# ---------- freeze（B3b 唯一写入口） ----------

def _approved_dataset(tmp_path, doc):
    """构造一份可通过 freeze 全部检查的题集：已 machine 确认、有 approval_basis、接地成立。"""

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
    # dataset.sha256 必须与**落盘字节**一致（含固定 LF）。
    data = (target / "dataset.json").read_bytes()
    assert b"\r" not in data
    recorded = (target / "dataset.sha256").read_text(encoding="utf-8").split()[0]
    assert recorded == hashlib.sha256(data).hexdigest()
    # README 记录指纹与校验值，便于冻结后复核。
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
    # 目标已存在时绝不写入任何冻结产物。
    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (target / "dataset.json").exists()


def test_freeze_refuses_dataset_that_fails_validation(tmp_path, capsys):
    root, doc = _source(tmp_path)
    # 未确认（approved_by=null）的题集不得冻结。
    dataset = _dataset(tmp_path, doc, [["7.21"]])
    target = tmp_path / "frozen"
    assert main(["freeze", "--dataset", str(dataset), "--target", str(target),
                 "--source-root", str(root)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not target.exists()


def test_map_with_no_usable_records_exits_2_but_writes_file(tmp_path, monkeypatch, capsys):
    """全部锚点不可用时映射文件照常写出便于诊断，但退出码必须非 0（2）。"""

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
    """active 指针与真实 chunk 集合不一致时必须失败（exit 1），不写出错误身份的映射。"""
    root, doc = _source(tmp_path)
    dataset = _dataset(tmp_path, doc, [["7.21"]])
    _install_store(monkeypatch, tmp_path, root, doc, doc.segments[0].content,
                   chunk_count=chunk_count, document_count=document_count)
    out = tmp_path / "mapping.json"
    assert main(["map", "--dataset", str(dataset), "--out", str(out)]) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and "FAILED:" in captured.err
    assert not out.exists()
