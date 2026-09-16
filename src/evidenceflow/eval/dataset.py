"""读取并校验人工标注的 Retrieval Evaluation Dataset。"""

import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from evidenceflow.domain.evaluation import EvaluationQuestion
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2


class EvaluationDatasetError(ValueError):
    """表示评测数据的 JSON 结构或业务 Schema 不合法。"""


# 复用 Domain Model 作为唯一数据合同，避免 Loader 维护一套重复字段定义。
_QUESTIONS_ADAPTER = TypeAdapter(list[EvaluationQuestion])


def load_evaluation_questions(path: Path) -> list[EvaluationQuestion]:
    """从 JSON 文件加载问题，并执行 Pydantic 与精确重复校验。

    Args:
        path: 顶层为 JSON 数组的评测数据文件路径。

    Returns:
        按文件顺序排列且已经通过 Domain Model 校验的问题列表。

    Raises:
        FileNotFoundError: 数据文件不存在。
        EvaluationDatasetError: JSON、顶层结构、Schema 或题目唯一性不合法。
    """

    dataset_path = Path(path)
    if not dataset_path.is_file():
        # 保留专门的文件异常类型，让调用方能区分路径问题与数据内容问题。
        raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset_path}")

    try:
        raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        # 行列信息可直接帮助人工维护 questions.json 时定位语法错误。
        raise EvaluationDatasetError(
            f"Invalid JSON in evaluation dataset at line {exc.lineno}, "
            f"column {exc.colno}: {dataset_path}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise EvaluationDatasetError(
            f"Evaluation dataset must be UTF-8 encoded: {dataset_path}"
        ) from exc

    # 显式检查顶层类型，以提供比通用 Pydantic 错误更直接的数据格式提示。
    if not isinstance(raw_data, list):
        raise EvaluationDatasetError("Evaluation dataset top level must be a JSON array")

    try:
        questions = _QUESTIONS_ADAPTER.validate_python(raw_data)
    except ValidationError as exc:
        raise EvaluationDatasetError(f"Invalid evaluation dataset schema: {exc}") from exc

    # 仅检查完全相同的问题文本；语义去重需要人工判断，不属于当前 Loader 职责。
    question_texts = [question.question for question in questions]
    if len(question_texts) != len(set(question_texts)):
        raise EvaluationDatasetError("Evaluation dataset contains duplicate question text")

    return questions


def adapt_research_v1(payload: dict, content_versions: dict[str, str]) -> EvaluationDatasetV2:
    """显式迁移科研 v1 来源组，保留 query ID，不补造 Claim 或 Holdout。"""
    return EvaluationDatasetV2.model_validate({
        "schema_version": "2.0",
        "dataset_id": f"research-v2-from-{payload['dataset_hash']}",
        "split": "legacy_dev_diagnostic",
        "corpus_version": payload["corpus_hash"],
        "questions": [{
            "question_id": q["question_id"], "family_id": q["question_id"],
            "query": q["query"], "query_language": q["language"], "corpus_language": "en",
            "question_type": q["question_type"], "evidence_type": q["evidence_type"],
            "answerability": "answerable", "reference_claims": [],
            "required_evidence_groups": [{
                "group_id": f"G{index}", "alternatives": [{
                    "document_id": source["document_id"],
                    "content_version": content_versions[source["document_id"]],
                    "locator": {"page_start": source["page_start"], "page_end": source["page_end"]},
                    "evidence_anchor": source["evidence_anchor"],
                }],
            } for index, source in enumerate(q["gold_sources"], 1)],
        } for q in payload["questions"]],
    })
