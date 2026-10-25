"""Test scripts/prune_chroma_collections.py (A4.2).

Everything runs against a real Chroma instance in a pytest temporary directory; the repository's
data/chroma and artifacts/phase15a are never touched.
"""

import json

import chromadb
import pytest

from evidenceflow.config import Settings
from evidenceflow.knowledge import IndexLifecycle, SourceRecord, build_manifest, current_pipeline_from_settings
import scripts.prune_chroma_collections as module
from scripts.prune_chroma_collections import ALLOWED_TO_DELETE, FORBIDDEN_TO_DELETE, main

SOURCES = [SourceRecord(document_id="doc-a", relative_path="a.pdf", content_hash="hash-a")]
ACTIVE = "ef_research_bge_m3_hybrid_c512_v2"


def _manifest(collection_name: str, status: str):
    return build_manifest(
        knowledge_base_id="kb-test",
        sources=SOURCES,
        pipeline=current_pipeline_from_settings(Settings(_env_file=None)),
        collection_name=collection_name,
        document_count=1,
        chunk_count=1,
        status=status,
    )


@pytest.fixture
def env(tmp_path):
    """The 7 collections in the temporary directory (named as in production) plus a state file holding only the active pointer."""

    db = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(db))
    for index, name in enumerate(ALLOWED_TO_DELETE + FORBIDDEN_TO_DELETE):
        collection = client.create_collection(name=name, metadata={"tag": name}, embedding_function=None)
        collection.add(ids=[f"id-{index}"], embeddings=[[float(index), 1.0]], documents=["x"])
    state_path = tmp_path / "index_state.json"
    IndexLifecycle(state_path, "kb-test").initialize_active(_manifest(ACTIVE, "ACTIVE"))
    return tmp_path, db, state_path


def _names(db) -> set[str]:
    return {item.name for item in chromadb.PersistentClient(path=str(db)).list_collections()}


def _args(tmp_path, db, state_path, *extra):
    return [
        "--db-path", str(db),
        "--state-path", str(state_path),
        "--before-output", str(tmp_path / "before.json"),
        "--after-output", str(tmp_path / "after.json"),
        *extra,
    ]


def test_default_is_dry_run_and_deletes_nothing(env, capsys) -> None:
    tmp_path, db, state_path = env

    assert main(_args(tmp_path, db, state_path)) == 0

    assert _names(db) == set(ALLOWED_TO_DELETE) | set(FORBIDDEN_TO_DELETE)
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "dry-run" and summary["deleted"] == []
    assert summary["targets"] == list(ALLOWED_TO_DELETE)
    assert summary["would_remain"] == sorted(FORBIDDEN_TO_DELETE)
    assert (tmp_path / "before.json").exists() and not (tmp_path / "after.json").exists()


def test_execute_deletes_only_allowlisted_and_exports_after(env, capsys) -> None:
    tmp_path, db, state_path = env
    client = chromadb.PersistentClient(path=str(db))
    client.create_collection(name="unrelated_collection", embedding_function=None)

    assert main(_args(tmp_path, db, state_path, "--execute")) == 0

    assert _names(db) == set(FORBIDDEN_TO_DELETE) | {"unrelated_collection"}
    summary = json.loads(capsys.readouterr().out)
    assert summary["deleted"] == list(ALLOWED_TO_DELETE)
    after = json.loads((tmp_path / "after.json").read_text(encoding="utf-8"))
    assert after["stage"] == "after" and after["collection_count"] == 4
    assert [row["name"] for row in after["collections"]] == sorted(set(FORBIDDEN_TO_DELETE) | {"unrelated_collection"})


def test_execute_skips_allowlisted_names_that_do_not_exist(env, capsys) -> None:
    tmp_path, db, state_path = env
    chromadb.PersistentClient(path=str(db)).delete_collection(ALLOWED_TO_DELETE[0])

    assert main(_args(tmp_path, db, state_path, "--execute")) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["absent_allowed_names"] == [ALLOWED_TO_DELETE[0]]
    assert summary["deleted"] == list(ALLOWED_TO_DELETE[1:])
    assert _names(db) == set(FORBIDDEN_TO_DELETE)


def test_hardcoded_forbidden_name_in_allowlist_is_refused(env, monkeypatch, capsys) -> None:
    tmp_path, db, state_path = env
    monkeypatch.setattr(module, "ALLOWED_TO_DELETE", ALLOWED_TO_DELETE + ("ef_research_bge_m3_hybrid_c512_v1",))

    assert main(_args(tmp_path, db, state_path, "--execute")) == 2

    assert "REFUSED" in capsys.readouterr().err
    assert _names(db) == set(ALLOWED_TO_DELETE) | set(FORBIDDEN_TO_DELETE)
    assert not (tmp_path / "before.json").exists()  # the refusal happens before the store is opened


@pytest.mark.parametrize("pointer", ["active", "previous", "staging"])
def test_lifecycle_pointer_name_in_allowlist_is_refused(env, monkeypatch, capsys, pointer) -> None:
    """Names referenced by active/previous/staging in the state file must be protected even when they are not on the hard-coded forbidden list."""

    tmp_path, db, state_path = env
    protected = "ef_research_structure_v1"  # a name on the allowlist that must become protected once the state file references it
    lifecycle = IndexLifecycle(state_path, "kb-test")
    if pointer == "active":
        state_path.unlink()
        lifecycle.initialize_active(_manifest(protected, "ACTIVE"))
    elif pointer == "previous":
        state_path.unlink()
        lifecycle.initialize_active(_manifest(protected, "ACTIVE"))
        state = lifecycle.load_state()
        moved = state.model_copy(
            update={
                "previous": state.active.model_copy(update={"status": "PREVIOUS"}),
                "active": state.active.model_copy(update={"collection_name": ACTIVE}),
            }
        )
        state_path.write_text(moved.model_dump_json(indent=2), encoding="utf-8")
    else:
        lifecycle.prepare_staging(_manifest(protected, "STAGING"))
    assert protected in module.lifecycle_protected_names(state_path)

    assert main(_args(tmp_path, db, state_path, "--execute")) == 2

    assert "allowlist contains protected collections" in capsys.readouterr().err
    assert _names(db) == set(ALLOWED_TO_DELETE) | set(FORBIDDEN_TO_DELETE)


@pytest.mark.parametrize("content", [None, "{not json"])
def test_unreadable_state_file_is_refused(env, capsys, content) -> None:
    tmp_path, db, state_path = env
    if content is None:
        state_path.unlink()
    else:
        state_path.write_text(content, encoding="utf-8")

    assert main(_args(tmp_path, db, state_path, "--execute")) == 2

    err = capsys.readouterr().err
    assert err.startswith("REFUSED: cannot read lifecycle state") and "Traceback" not in err
    assert _names(db) == set(ALLOWED_TO_DELETE) | set(FORBIDDEN_TO_DELETE)


def test_before_export_structure(env) -> None:
    tmp_path, db, state_path = env

    assert main(_args(tmp_path, db, state_path)) == 0

    before = json.loads((tmp_path / "before.json").read_text(encoding="utf-8"))
    assert set(before) == {"stage", "db_path", "exported_at", "collection_count", "collections"}
    assert before["stage"] == "before" and before["collection_count"] == 7
    names = [row["name"] for row in before["collections"]]
    assert names == sorted(names)
    for row in before["collections"]:
        assert set(row) == {"name", "id", "count", "metadata"}
        assert row["count"] == 1 and row["metadata"] == {"tag": row["name"]}
        assert isinstance(row["id"], str) and row["id"]


def test_hardcoded_lists_do_not_overlap() -> None:
    assert not set(ALLOWED_TO_DELETE) & set(FORBIDDEN_TO_DELETE)
    assert set(FORBIDDEN_TO_DELETE) == {
        "ef_research_bge_m3_hybrid_c512_v2",
        "ef_research_bge_m3_hybrid_c512_v1",
        "ef_research_bge_m3_c512_o64_v1",
    }


@pytest.mark.parametrize("execute", [False, True])
def test_existing_output_is_refused_and_left_unchanged(env, capsys, execute) -> None:
    """A re-run must not overwrite the A4 audit record: when before (or after under --execute) already exists -> exit code 2, file and store unchanged."""

    tmp_path, db, state_path = env
    existing = tmp_path / ("after.json" if execute else "before.json")
    existing.write_text("original audit record\n", encoding="utf-8")
    extra = ("--execute",) if execute else ()

    assert main(_args(tmp_path, db, state_path, *extra)) == 2

    err = capsys.readouterr().err
    assert err.startswith("REFUSED: output file(s) already exist") and "--overwrite" in err
    assert existing.read_text(encoding="utf-8") == "original audit record\n"
    assert _names(db) == set(ALLOWED_TO_DELETE) | set(FORBIDDEN_TO_DELETE)
    if execute:
        assert not (tmp_path / "before.json").exists()  # the refusal happens before connecting to Chroma and exporting before


def test_existing_output_check_runs_before_opening_chroma(env, monkeypatch) -> None:
    tmp_path, db, state_path = env
    (tmp_path / "before.json").write_text("x", encoding="utf-8")

    def fail_client(*args, **kwargs):  # pragma: no cover - being called would prove the ordering is wrong
        raise AssertionError("Chroma must not be opened when outputs already exist")

    monkeypatch.setattr(chromadb, "PersistentClient", fail_client)

    assert main(_args(tmp_path, db, state_path)) == 2


def test_overwrite_flag_allows_replacing_existing_outputs(env, capsys) -> None:
    tmp_path, db, state_path = env
    (tmp_path / "before.json").write_text("stale\n", encoding="utf-8")
    (tmp_path / "after.json").write_text("stale\n", encoding="utf-8")

    assert main(_args(tmp_path, db, state_path, "--execute", "--overwrite")) == 0

    before = json.loads((tmp_path / "before.json").read_text(encoding="utf-8"))
    after = json.loads((tmp_path / "after.json").read_text(encoding="utf-8"))
    assert before["stage"] == "before" and before["collection_count"] == 7
    assert after["stage"] == "after" and after["collection_count"] == 3
    assert _names(db) == set(FORBIDDEN_TO_DELETE)
