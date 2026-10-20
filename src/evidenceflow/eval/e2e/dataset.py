"""端到端题集 rag_e2e_v1 的 schema、fingerprint 与确定性校验（PROJECT_SPEC §13.1(a)(c)，含 D8）。

- 锚点模型复用 `domain/evaluation_v2.py` 的 `EvidenceGroup` / `EvidenceAlternative`，与 canonical 40 题同一协议；
- 所有模型 `extra="forbid"`，防止拼错的字段被静默丢弃；
- `validate_dataset` 只做确定性检查，不抛错，汇总全部问题供审阅。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field

from evidenceflow.domain.evaluation_v2 import ContractModel, EvidenceAlternative, EvidenceGroup
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.eval.e2e.key_match import alias_hits, normalize_text

SCHEMA_VERSION = "rag-e2e-v1"
# D8：错误前提题的标记 tag；这类题按可答题计分，不属于不可答题。
FALSE_PREMISE_TAG = "false_premise"
MULTI_DOC_TAG = "multi_doc"
# 别名规范化后的最短长度（§13.1(c)③），过短的别名几乎必然误命中。
MIN_ALIAS_LENGTH = 2

UnanswerableKind = Literal["out_of_corpus", "near_miss"]

# page_text_provider(document_id, page_start, page_end) -> 该 locator 页的源文本；
# 页码为 None 表示 MD/TXT 等无页码文档，返回整篇文本。
PageTextProvider = Callable[[str, int | None, int | None], str]


class Provenance(ContractModel):
    """题目来源与确认记录；冻结版要求 `approved_by == "user"`。"""

    drafted_by: str = Field(min_length=1, description="起草者，如 agent / user。")
    approved_by: str | None = Field(default=None, description="确认者；未确认为 None。")
    source: str = Field(min_length=1, description="题目来源，如 canonical_40 / phase12 / new。")


class RequiredClaim(ContractModel):
    """必需论断：文本、证据锚点与关键要素别名。"""

    claim_id: str = Field(min_length=1, description="题内唯一论断 ID。")
    text: str = Field(min_length=1, description="论断原文（人工确认的事实）。")
    required: bool = Field(description="是否为必需论断；指标只统计 required=True 的论断。")
    evidence_anchors: list[EvidenceAlternative] = Field(
        default_factory=list, description="支持该论断的原文锚点；多个锚点之间为“或”。"
    )
    answer_keys: list[list[str]] = Field(
        default_factory=list,
        description="关键要素：外层组间“与”，组内别名“或”；空列表表示 key_matchable=false。",
    )

    @property
    def key_matchable(self) -> bool:
        """有 answer_keys 才能做确定性的关键要素匹配。"""

        return bool(self.answer_keys)


class ForbiddenClaim(ContractModel):
    """错误前提题中不得被肯定的论断，附与之矛盾的原文锚点（D8）。"""

    claim_id: str = Field(min_length=1, description="题内唯一论断 ID（与 required claim 共用 ID 空间）。")
    text: str = Field(min_length=1, description="错误前提的表述。")
    evidence_anchors: list[EvidenceAlternative] = Field(
        default_factory=list, description="与该论断矛盾的原文锚点。"
    )


class EvalCase(ContractModel):
    """一道端到端评测题（§13.1(a)）。"""

    case_id: str = Field(min_length=1, description="数据集内唯一 ID；新题用 E2E-M* / E2E-U*。")
    family_id: str = Field(min_length=1, description="题族；配对比较按题族 bootstrap。")
    query: str = Field(min_length=1, description="提问原文。")
    query_language: str = Field(min_length=1, description="提问语言，如 zh / en。")
    question_type: str = Field(min_length=1, description="题型切片。")
    tags: list[str] = Field(default_factory=list, description="分片标签，如 multi_doc、false_premise。")
    critical: bool = Field(description="关键题：Gate 中逐题结果不得由通过变为不通过。")
    expected_status: AnswerStatus = Field(description="期望回答状态。")
    acceptable_abstention: bool = Field(description="可答题是否允许拒答（D7）；不可答题必须为 false。")
    required_evidence_groups: list[EvidenceGroup] = Field(default_factory=list, description="必需证据组（组间“与”）。")
    required_claims: list[RequiredClaim] = Field(default_factory=list, description="必需论断。")
    forbidden_claims: list[ForbiddenClaim] = Field(default_factory=list, description="只用于 false_premise 题（D8）。")
    unanswerable_kind: UnanswerableKind | None = Field(default=None, description="不可答题类型；可答题为 None。")
    provenance: Provenance


class E2EDataset(ContractModel):
    """冻结题集；`fingerprint` 绑定内容，供 observations / mapping 校验身份。"""

    schema_version: Literal["rag-e2e-v1"] = SCHEMA_VERSION
    dataset_id: str = Field(min_length=1, description="数据集版本身份。")
    corpus_version: str = Field(min_length=1, description="语料版本（manifest / corpus signature）。")
    cases: list[EvalCase] = Field(min_length=1)
    fingerprint: str | None = Field(default=None, description="除本字段外整个数据集规范化 JSON 的 sha256。")


def compute_fingerprint(dataset: E2EDataset) -> str:
    """按 §13.1(a) 计算 fingerprint：排除 `fingerprint` 字段后的规范化 JSON 的 sha256。

    规范化 JSON = `sort_keys=True`、`ensure_ascii=False`、`separators=(",", ":")`、UTF-8；
    `model_dump(mode="json")` 包含全部字段（含默认值），保证同一内容的序列化唯一。
    """

    payload = dataset.model_dump(mode="json", exclude={"fingerprint"})
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def with_fingerprint(dataset: E2EDataset) -> E2EDataset:
    """返回填好 fingerprint 的新对象，原对象不变。"""

    return dataset.model_copy(update={"fingerprint": compute_fingerprint(dataset)})


def load_dataset(path: str | Path, *, verify_fingerprint: bool = True) -> E2EDataset:
    """读取题集；`verify_fingerprint=True` 时缺失或不一致都抛 `ValueError`。"""

    dataset = E2EDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if verify_fingerprint:
        if dataset.fingerprint is None:
            raise ValueError(f"dataset {path} has no fingerprint")
        actual = compute_fingerprint(dataset)
        if actual != dataset.fingerprint:
            raise ValueError(
                f"dataset fingerprint mismatch: recorded {dataset.fingerprint}, computed {actual}"
            )
    return dataset


class ValidationIssue(ContractModel):
    """一条校验问题；`case_id=None` 表示数据集级问题。"""

    case_id: str | None
    rule: str = Field(description="规则名，见 validate_dataset 的 docstring。")
    message: str


class Ratio(ContractModel):
    """报告中的比例：空分母时 value 为 None，不伪造数值。"""

    numerator: int
    denominator: int
    value: float | None


class ValidationReport(ContractModel):
    """确定性校验结果汇总。"""

    ok: bool
    case_count: int
    require_approved: bool
    issue_count: int
    issues: list[ValidationIssue]
    issues_by_case: dict[str, list[ValidationIssue]] = Field(
        description="按 case_id 分组；数据集级问题放在键 \"<dataset>\" 下。"
    )
    key_matchable_required_claim_ratio: Ratio = Field(
        description="可答题中 key_matchable 的 required claim / 全部 required claim。"
    )
    near_miss_ratio: Ratio = Field(description="不可答题（out_of_corpus + near_miss）中 near_miss 的比例。")


def _ratio(numerator: int, denominator: int) -> Ratio:
    """空分母返回 None。"""

    return Ratio(numerator=numerator, denominator=denominator,
                 value=(numerator / denominator) if denominator else None)


def _duplicates(values: list[str]) -> list[str]:
    """按首次出现顺序列出重复值。"""

    seen: set[str] = set()
    duplicated: list[str] = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return duplicated


def _check_answer_keys(case: EvalCase, claim: RequiredClaim, provider: PageTextProvider,
                       add: Callable[[str, str], None]) -> None:
    """§13.1(c) 三条规则；只对 key_matchable 的 claim 调用。"""

    # 预先取出每个锚点 locator 页的源文本；取不到时记录问题，不中断其他检查。
    page_texts: list[str] = []
    for index, anchor in enumerate(claim.evidence_anchors):
        loc = anchor.locator
        try:
            page_texts.append(provider(anchor.document_id, loc.page_start, loc.page_end))
        except Exception as exc:  # noqa: BLE001 - 校验器汇总问题而不是抛错
            add("answer_keys_source",
                f"{claim.claim_id}: cannot read source text of anchor #{index}: {type(exc).__name__}: {exc}")
    if not claim.evidence_anchors:
        add("answer_keys_source", f"{claim.claim_id}: key_matchable claim has no evidence_anchors")
    # 页源文本只需规范化一次，供每个要素组的规则①复用。
    normalized_pages = [normalize_text(text) for text in page_texts]

    for group_index, group in enumerate(claim.answer_keys):
        label = f"{claim.claim_id} group #{group_index}"
        # ③ 组非空、别名去首尾空白后非空、规范化后长度 ≥ 2、组内（按规范化形式）去重。
        if not group:
            add("answer_keys_format", f"{label}: empty alias group")
            continue
        normalized_aliases: list[str] = []
        for alias in group:
            norm = normalize_text(alias)
            if not alias.strip():
                add("answer_keys_format", f"{label}: blank alias")
            elif len(norm) < MIN_ALIAS_LENGTH:
                add("answer_keys_format", f"{label}: alias {alias!r} shorter than {MIN_ALIAS_LENGTH} after normalization")
            normalized_aliases.append(norm)
        for duplicated in _duplicates(normalized_aliases):
            add("answer_keys_format", f"{label}: duplicate alias {duplicated!r} after normalization")
        # ② 任何别名都不得命中题干，否则“回答复述了题目”也会被算作命中。
        for alias in group:
            if alias.strip() and alias_hits(alias, case.query):
                add("answer_keys_query", f"{label}: alias {alias!r} appears in query")
        # ① 至少一个别名经规范化后作为**字面子串**出现在该 claim 某个锚点的页源文本中，证明别名来自原文。
        # 这里刻意不用 alias_hits 的 ASCII 边界：原文写作 `1.5mm` 时别名 `1.5` 必须能过校验，
        # 否则自然回答常写的裸数字会让 key_fact_recall 系统性漏计；边界规则只留在 ② 与计分侧。
        if page_texts and not any(alias in text for alias in normalized_aliases if alias for text in normalized_pages):
            add("answer_keys_source", f"{label}: no alias found in any anchor's locator source text")


def validate_dataset(dataset: E2EDataset, page_text_provider: PageTextProvider, *,
                     require_approved: bool) -> ValidationReport:
    """对题集做全部确定性检查，汇总问题，不抛错。

    规则名（`ValidationIssue.rule`）：
    - `duplicate_case_id` / `duplicate_group_id` / `duplicate_claim_id`：ID 唯一（组、claim 在题内唯一，
      required 与 forbidden claim 共用 ID 空间，因为映射文件按 (case_id, claim_id) 定位）；
    - `fingerprint_mismatch`：数据集带 fingerprint 但与内容不一致；
    - `answerable_required_claim`：可答题至少 1 条 `required=True` 的 claim；
    - `answerable_evidence_group`：可答题至少 1 个 `required_evidence_groups`；
    - `required_claim_anchors`：可答题的 required claim 必须带证据锚点；
    - `multi_doc`：带 `multi_doc` tag 的题至少 2 个证据组，且至少两组分属不同文档；
    - `unanswerable_shape`：不可答题 acceptable_abstention=false、unanswerable_kind 非空、claims/证据组/forbidden 为空；
    - `answerable_unanswerable_kind`：可答题 unanswerable_kind 必须为 null；
    - `false_premise`：D8 各项；`forbidden_claims_non_false_premise`：非 false_premise 题 forbidden_claims 必须为空；
    - `answer_keys_source` / `answer_keys_query` / `answer_keys_format`：§13.1(c) ①②③；
    - `not_approved`：`require_approved=True` 时 `provenance.approved_by` 必须为 "user"。
    """

    issues: list[ValidationIssue] = []

    def adder(case_id: str | None) -> Callable[[str, str], None]:
        """绑定 case_id 的问题收集器。"""

        def add(rule: str, message: str) -> None:
            issues.append(ValidationIssue(case_id=case_id, rule=rule, message=message))

        return add

    dataset_add = adder(None)
    for duplicated in _duplicates([case.case_id for case in dataset.cases]):
        dataset_add("duplicate_case_id", f"case_id {duplicated!r} appears more than once")
    if dataset.fingerprint is not None and dataset.fingerprint != compute_fingerprint(dataset):
        dataset_add("fingerprint_mismatch", "recorded fingerprint does not match dataset content")

    key_matchable_required = 0
    total_required = 0
    unanswerable_total = 0
    near_miss_total = 0
    for case in dataset.cases:
        add = adder(case.case_id)
        answerable = case.expected_status == AnswerStatus.ANSWERED
        is_false_premise = FALSE_PREMISE_TAG in case.tags

        for duplicated in _duplicates([g.group_id for g in case.required_evidence_groups]):
            add("duplicate_group_id", f"group_id {duplicated!r} appears more than once")
        claim_ids = [c.claim_id for c in case.required_claims] + [c.claim_id for c in case.forbidden_claims]
        for duplicated in _duplicates(claim_ids):
            add("duplicate_claim_id", f"claim_id {duplicated!r} appears more than once")

        if answerable:
            required = [c for c in case.required_claims if c.required]
            if not required:
                add("answerable_required_claim", "answerable case has no required claim")
            # 可答题必须至少有一个必需证据组，否则上下文类指标没有可评对象，冻结后无法解释。
            if not case.required_evidence_groups:
                add("answerable_evidence_group", "answerable case has no required_evidence_groups")
            for claim in required:
                if not claim.evidence_anchors:
                    add("required_claim_anchors", f"{claim.claim_id}: required claim has no evidence_anchors")
            # 比例统计：分母为可答题（含 D8 错误前提题）的全部 required claim。
            total_required += len(required)
            key_matchable_required += sum(1 for c in required if c.key_matchable)
            if case.unanswerable_kind is not None:
                add("answerable_unanswerable_kind", "answerable case must have unanswerable_kind=null")
        else:
            # 不可答题只允许拒答，不得携带任何可评分的正例。
            if case.acceptable_abstention:
                add("unanswerable_shape", "unanswerable case must have acceptable_abstention=false (D7)")
            if case.unanswerable_kind is None:
                add("unanswerable_shape", "unanswerable case must set unanswerable_kind")
            if case.required_claims:
                add("unanswerable_shape", "unanswerable case must not have required_claims")
            if case.required_evidence_groups:
                add("unanswerable_shape", "unanswerable case must not have required_evidence_groups")
            if case.forbidden_claims:
                add("unanswerable_shape", "unanswerable case must not have forbidden_claims")
            if case.unanswerable_kind is not None:
                unanswerable_total += 1
                near_miss_total += case.unanswerable_kind == "near_miss"

        if MULTI_DOC_TAG in case.tags:
            doc_sets = [{a.document_id for a in g.alternatives} for g in case.required_evidence_groups]
            # 存在两组 i<j，能各取一篇不同文档，才算“分属不同论文”。
            cross_doc = any(len(doc_sets[i] | doc_sets[j]) >= 2
                            for i in range(len(doc_sets)) for j in range(i + 1, len(doc_sets)))
            if len(doc_sets) < 2 or not cross_doc:
                add("multi_doc", "multi_doc case needs >= 2 evidence groups from >= 2 different documents")

        if is_false_premise:
            # D8：错误前提题按可答题计分，允许拒答，不计入不可答题。
            if not answerable:
                add("false_premise", "false_premise case must have expected_status=answered")
            if not case.acceptable_abstention:
                add("false_premise", "false_premise case must have acceptable_abstention=true")
            if case.unanswerable_kind is not None:
                add("false_premise", "false_premise case must have unanswerable_kind=null")
            if not any(c.required and c.key_matchable for c in case.required_claims):
                add("false_premise", "false_premise case needs a required claim (corrected fact) with answer_keys")
            if not case.forbidden_claims:
                add("false_premise", "false_premise case needs at least one forbidden claim")
            for claim in case.forbidden_claims:
                if not claim.evidence_anchors:
                    add("false_premise", f"{claim.claim_id}: forbidden claim needs contradicting evidence_anchors")
        elif case.forbidden_claims:
            add("forbidden_claims_non_false_premise", "forbidden_claims must be empty unless tagged false_premise")

        for claim in case.required_claims:
            if claim.key_matchable:
                _check_answer_keys(case, claim, page_text_provider, add)

        if require_approved and case.provenance.approved_by != "user":
            add("not_approved", f"provenance.approved_by is {case.provenance.approved_by!r}, expected 'user'")

    by_case: dict[str, list[ValidationIssue]] = {}
    for issue in issues:
        by_case.setdefault(issue.case_id or "<dataset>", []).append(issue)
    return ValidationReport(
        ok=not issues,
        case_count=len(dataset.cases),
        require_approved=require_approved,
        issue_count=len(issues),
        issues=issues,
        issues_by_case=by_case,
        key_matchable_required_claim_ratio=_ratio(key_matchable_required, total_required),
        near_miss_ratio=_ratio(near_miss_total, unanswerable_total),
    )
