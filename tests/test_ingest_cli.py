"""验证 ingest CLI 的服务委托、输出与退出码。"""

from pathlib import Path

from evidenceflow.__main__ import main, render_ingestion_summary
from evidenceflow.config import Settings
from evidenceflow.ingestion.service import DocumentIngestionResult, IngestionSummary


class FakeService:
    """返回固定 Summary 并记录 CLI 传入范围。"""

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
