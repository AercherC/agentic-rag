"""使用 FakeRetriever 验证 Evaluation Runner 的调用次数、汇总与持久化。"""

import json
from pathlib import Path

import pytest

from evidenceflow.domain.chunk import DocumentChunk, RetrievedChunk
from evidenceflow.domain.evaluation import EvaluationQuestion, EvaluationSource
from evidenceflow.eval.runner import EvaluationRunner, save_evaluation_report


def make_result(
    rank: int,
    document_id: str,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> RetrievedChunk:
    """构造 Runner 测试使用的固定检索结果。"""

    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=f"{document_id}-chunk-{rank}",
            document_id=document_id,
            content="A   compact\ncontent preview for failure analysis.",
            token_count=8,
            chunk_index=rank - 1,
            page_start=page_start,
            page_end=page_end,
            file_name="source.pdf" if page_start is not None else "source.md",
            content_hash="content-hash",
            metadata={},
        ),
        rank=rank,
        similarity_score=1.0 - rank / 10,
    )


class FakeRetriever:
    """按 Query 返回预设结果，并记录每次调用的显式 Top-K。"""

    def __init__(self, results_by_query: dict[str, list[RetrievedChunk]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])


def make_question(text: str, document_id: str = "gold-doc") -> EvaluationQuestion:
    """构造文档级 Gold 的最小评测问题。"""

    return EvaluationQuestion(
        question=text,
        relevant_sources=[EvaluationSource(document_id=document_id)],
    )


def test_runner_retrieves_once_per_question_with_top_five() -> None:
    questions = [make_question("q1"), make_question("q2"), make_question("q3")]
    retriever = FakeRetriever(
        {
            "q1": [make_result(1, "gold-doc")],
            "q2": [make_result(3, "gold-doc")],
            "q3": [],
        }
    )

    report = EvaluationRunner(retriever).run(questions)

    assert retriever.calls == [("q1", 5), ("q2", 5), ("q3", 5)]
    assert report.aggregate_metrics.question_count == 3
    assert report.aggregate_metrics.recall_at_1 == pytest.approx(1 / 3)
    assert report.aggregate_metrics.recall_at_3 == pytest.approx(2 / 3)
    assert report.aggregate_metrics.recall_at_5 == pytest.approx(2 / 3)
    assert report.aggregate_metrics.mrr == pytest.approx((1 + 1 / 3) / 3)
    assert [item.question for item in report.failure_cases] == ["q3"]
    assert [item.question for item in report.near_miss_cases] == ["q2"]


def test_runner_keeps_multiple_gold_and_marks_matching_result() -> None:
    question = EvaluationQuestion(
        question="multiple gold",
        relevant_sources=[
            EvaluationSource(document_id="doc-a"),
            EvaluationSource(document_id="doc-b"),
        ],
    )
    retriever = FakeRetriever(
        {"multiple gold": [make_result(1, "doc-b"), make_result(2, "wrong-doc")]}
    )

    report = EvaluationRunner(retriever, preview_length=20).run([question])
    result = report.per_question_results[0]

    assert len(result.relevant_sources) == 2
    assert result.first_relevant_rank == 1
    assert result.retrieved_results[0].relevant is True
    assert result.retrieved_results[1].relevant is False
    assert result.retrieved_results[0].content_preview == "A compact content pr"


def test_runner_creates_pdf_markdown_mixed_and_explicit_breakdowns() -> None:
    questions = [
        EvaluationQuestion(
            question="pdf",
            relevant_sources=[
                EvaluationSource(document_id="pdf-doc", page_start=4, page_end=4)
            ],
        ),
        make_question("markdown", "md-doc"),
        EvaluationQuestion(
            question="mixed",
            relevant_sources=[
                EvaluationSource(document_id="md-doc"),
                EvaluationSource(document_id="pdf-doc", page_start=9, page_end=9),
            ],
        ),
    ]
    retriever = FakeRetriever(
        {
            "pdf": [make_result(1, "pdf-doc", page_start=4, page_end=4)],
            "markdown": [],
            "mixed": [make_result(2, "md-doc")],
        }
    )

    report = EvaluationRunner(retriever).run(
        questions,
        question_groups={"english": {0, 2}, "cross_language": {2}},
    )

    assert report.breakdowns["pdf_gold"].question_count == 1
    assert report.breakdowns["markdown_gold"].question_count == 1
    assert report.breakdowns["mixed_gold"].question_count == 1
    assert report.breakdowns["english"].question_count == 2
    assert report.breakdowns["cross_language"].mrr == 0.5


def test_runner_rejects_empty_dataset_and_invalid_group_indices() -> None:
    runner = EvaluationRunner(FakeRetriever({}))

    with pytest.raises(ValueError, match="questions"):
        runner.run([])
    with pytest.raises(ValueError, match="invalid index"):
        runner.run([make_question("q1")], question_groups={"english": {1}})


@pytest.mark.parametrize("top_k", [0, 3, 6, True])
def test_runner_rejects_non_baseline_top_k(top_k: int) -> None:
    """本阶段只允许一次 Top-5 查询，避免报告配置与固定指标窗口不一致。"""

    with pytest.raises(ValueError, match="must be 5"):
        EvaluationRunner(FakeRetriever({}), top_k=top_k)


def test_report_save_is_utf8_and_refuses_silent_overwrite(tmp_path: Path) -> None:
    report = EvaluationRunner(FakeRetriever({})).run(
        [make_question("中文问题")],
        configuration={"top_k": 5},
    )
    output_path = tmp_path / "results" / "baseline.json"

    save_evaluation_report(report, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["configuration"]["top_k"] == 5
    assert payload["per_question_results"][0]["question"] == "中文问题"
    assert payload["failure_cases"][0]["recall_at_5"] == 0.0
    with pytest.raises(FileExistsError, match="already exists"):
        save_evaluation_report(report, output_path)
    save_evaluation_report(report, output_path, overwrite=True)
