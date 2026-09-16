"""测试 Retrieval Evaluation Dataset 的读取、Schema 与真实 Gold 标注。"""

import json
from pathlib import Path

import pytest

from evidenceflow.eval import EvaluationDatasetError, load_evaluation_questions


def write_dataset(path: Path, data: object) -> None:
    """以 UTF-8 写入临时 JSON，确保测试关注 Loader 行为而非平台编码。"""

    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def make_source(
    document_id: str = "doc-1",
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, object]:
    """构造符合 EvaluationSource 字段合同的最小测试数据。"""

    return {
        "document_id": document_id,
        "page_start": page_start,
        "page_end": page_end,
    }


def test_load_valid_dataset_with_multiple_questions_and_sources(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.json"
    write_dataset(
        dataset_path,
        [
            {
                "question": "What is durable execution?",
                "relevant_sources": [make_source("markdown-doc")],
            },
            {
                "question": "How are tools categorized?",
                "relevant_sources": [
                    make_source("pdf-doc", 9, 9),
                    make_source("markdown-doc"),
                ],
            },
        ],
    )

    questions = load_evaluation_questions(dataset_path)

    assert len(questions) == 2
    assert len(questions[1].relevant_sources) == 2
    assert questions[1].relevant_sources[0].page_start == 9
    assert questions[1].relevant_sources[0].page_end == 9
    assert questions[0].relevant_sources[0].page_start is None
    assert questions[0].relevant_sources[0].page_end is None


def test_missing_dataset_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_evaluation_questions(tmp_path / "missing.json")


def test_invalid_json_is_rejected_with_location(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text('[{"question": }]', encoding="utf-8")

    with pytest.raises(EvaluationDatasetError, match=r"line 1, column"):
        load_evaluation_questions(dataset_path)


def test_non_array_top_level_is_rejected(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.json"
    write_dataset(dataset_path, {"questions": []})

    with pytest.raises(EvaluationDatasetError, match="top level must be a JSON array"):
        load_evaluation_questions(dataset_path)


@pytest.mark.parametrize(
    "invalid_data",
    [
        [{"relevant_sources": [make_source()]}],
        [{"question": "Missing sources"}],
        [
            {
                "question": "Missing document identity",
                "relevant_sources": [{"page_start": None, "page_end": None}],
            }
        ],
    ],
)
def test_missing_required_schema_fields_are_rejected(
    tmp_path: Path, invalid_data: object
) -> None:
    dataset_path = tmp_path / "questions.json"
    write_dataset(dataset_path, invalid_data)

    with pytest.raises(EvaluationDatasetError, match="schema"):
        load_evaluation_questions(dataset_path)


def test_invalid_page_range_is_rejected(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.json"
    write_dataset(
        dataset_path,
        [
            {
                "question": "Invalid PDF range",
                "relevant_sources": [make_source("pdf-doc", 10, 9)],
            }
        ],
    )

    with pytest.raises(EvaluationDatasetError, match="page_start"):
        load_evaluation_questions(dataset_path)


def test_duplicate_question_text_is_rejected(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.json"
    duplicate = {
        "question": "Exactly repeated question",
        "relevant_sources": [make_source()],
    }
    write_dataset(dataset_path, [duplicate, duplicate])

    with pytest.raises(EvaluationDatasetError, match="duplicate question"):
        load_evaluation_questions(dataset_path)

