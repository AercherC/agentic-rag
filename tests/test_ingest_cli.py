"""Verify the ingest CLI's service delegation, output and exit codes."""

from pathlib import Path

from evidenceflow.__main__ import main, render_ingestion_summary
from evidenceflow.config import Settings
from evidenceflow.ingestion.service import DocumentIngestionResult, IngestionSummary


class FakeService:
    """Return a fixed Summary and record the scope the CLI passed in."""

    def __init__(self, root: Path, summary: IngestionSummary) -> None:
        self.ingestion_root = root
        self.summary = summary
        self.calls = []

    def ingest(self, path=None):
        self.calls.append(path)
        return self.summary


def _summary(*, failed: int = 0) -> IngestionSummary:
    results = (
        DocumentIngestionResult("a.md", "doc-a", "inserted", 2, 2),
        *(() if not failed else (DocumentIngestionResult("bad.pdf", None, "failed", failed_stage="load", error_message="ValueError: bad"),)),
    )
    return IngestionSummary(len(results), 1, 0, 0, failed, 2, 2, results)


def test_ingest_cli_passes_optional_path_and_renders_summary(tmp_path: Path, capsys) -> None:
    service = FakeService(tmp_path, _summary())
    exit_code = main(["ingest", "nested/a.md"], settings=Settings(_env_file=None), ingestion_service=service)
    output = capsys.readouterr().out
    assert exit_code == 0
    assert service.calls == [Path("nested/a.md")]
    assert "Inserted: 1" in output and "Chunks written: 2" in output


def test_ingest_cli_uses_default_root_scope_and_nonzero_on_failure(tmp_path: Path, capsys) -> None:
    service = FakeService(tmp_path, _summary(failed=1))
    exit_code = main(["ingest"], settings=Settings(_env_file=None), ingestion_service=service)
    output = capsys.readouterr().out
    assert exit_code == 1
    assert service.calls == [None]
    assert "[FAILED]" in output and "stage=load" in output and "Failed: 1" in output


def test_summary_never_prints_embedding_values(tmp_path: Path) -> None:
    output = render_ingestion_summary(_summary(), tmp_path)
    assert "embeddings=2" in output
    assert "[1.0" not in output


def test_ingest_cli_refuses_active_collection_with_exit_code_2(tmp_path: Path, capsys, monkeypatch) -> None:
    """The collection pointed at by lifecycle active must not be ingested directly: exit code 2, loading no model."""

    from evidenceflow.knowledge import IndexLifecycle, PipelineIdentity, SourceRecord, build_manifest

    monkeypatch.delenv("COLLECTION_NAME", raising=False)
    pipeline = PipelineIdentity(
        parser_version="p", cleaner_version="c", chunker_version="k",
        chunker_config={"strategy": "hybrid"}, embedding_model="m", embedding_version="e",
    )
    manifest = build_manifest(
        knowledge_base_id="kb-test",
        sources=[SourceRecord(document_id="d", relative_path="a.pdf", content_hash="h")],
        pipeline=pipeline, collection_name="col-active", document_count=1, chunk_count=1,
        status="ACTIVE",
    )
    IndexLifecycle(tmp_path / "index_state.json", "kb-test").initialize_active(manifest)
    settings = Settings(_env_file=None, index_state_path=tmp_path / "index_state.json")

    # Do not inject ingestion_service: this goes through the real build_ingestion_service, so the
    # protection
    # must trigger before any model is constructed.
    exit_code = main(["ingest", "data/source_docs"], settings=settings)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "col-active" in captured.err and "build_staging_index" in captured.err
    assert captured.out == ""
