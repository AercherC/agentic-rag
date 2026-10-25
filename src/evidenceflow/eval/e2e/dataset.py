"""Schema, fingerprint and deterministic validation for the end-to-end dataset rag_e2e_v1
(PROJECT_SPEC §13.1(a)(c)(c′), including D8).

- Anchor models reuse `EvidenceGroup` / `EvidenceAlternative` from `domain/evaluation_v2.py`, the
  same protocol as the canonical 40 questions;
- Every model sets `extra="forbid"`, so a misspelled field is rejected instead of being silently
  dropped;
- `validate_dataset` runs deterministic checks only and never raises; it aggregates every problem
  so a reviewer can see them all at once;
- The freeze condition (§13.1(a)) keeps human confirmation and machine acceptance apart:
  `approved_by="machine"` must carry `approval_basis` (of the form `<path>#<sha256>`), but the
  model only validates non-emptiness and at least one `#`; whether the report path exists and its
  sha256 matches is the CLI's job (B3a-2′ D2 / B3b).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from evidenceflow.domain.evaluation_v2 import ContractModel, EvidenceAlternative, EvidenceGroup
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.eval.e2e.key_match import alias_hits, normalize_text

SCHEMA_VERSION = "rag-e2e-v1"
# D8: the tag that marks a false-premise case; such cases are scored as answerable and are not
# counted as unanswerable.
FALSE_PREMISE_TAG = "false_premise"
MULTI_DOC_TAG = "multi_doc"
# Minimum length of an alias after normalization (§13.1(c)③); a shorter alias almost always
# produces false hits.
MIN_ALIAS_LENGTH = 2
# §13.1(c′)③: a claim must be at least 10 characters after normalization, which rules out
# yes/no-style phrases.
MIN_CLAIM_LENGTH = 10
# §13.1(c′)④: the longest common run of consecutive words between the question and a claim must
# not exceed 8 words.
MAX_COMMON_WORD_RUN = 8

UnanswerableKind = Literal["out_of_corpus", "near_miss"]

# page_text_provider(document_id, page_start, page_end) -> the source text of the page under that
# locator; a None page number means a document without pages such as MD/TXT, in which case the
# whole text is returned.
PageTextProvider = Callable[[str, int | None, int | None], str]


class Provenance(ContractModel):
    """Provenance and confirmation record of a question (§13.1(a)).

    The frozen dataset accepts `approved_by ∈ {"user", "machine"}`:
    - `"machine"`: `approval_basis` is required, of the form `<path>#<sha256>` (only non-emptiness
      and at least one `#` are validated);
    - `"user"` or `None` (unconfirmed): `approval_basis` must be `None`.
    When `approval_basis` is None it stays out of the fingerprint computation (see
    `compute_fingerprint`), so datasets generated before v0.32 keep the same fingerprint.
    """

    drafted_by: str = Field(min_length=1, description="起草者，如 agent / user。")
    approved_by: Literal["user", "machine"] | None = Field(
        default=None, description="确认者；未确认为 None，冻结时必须为 user 或 machine。"
    )
    approval_basis: str | None = Field(
        default=None,
        description="机器验收证据引用，形如 artifacts/stage_b/B3_gate_report.json#<sha256>。",
    )
    source: str = Field(min_length=1, description="题目来源，如 canonical_40 / derived_t1 / derived_t2。")

    @model_validator(mode="after")
    def validate_approval(self):
        """Machine confirmation requires an acceptance basis; human confirmation and no
        confirmation must not carry one."""

        if self.approved_by == "machine":
            # Shape check only: path existence and sha256 consistency are validated by the CLI;
            # the model layer never reads files.
            if not self.approval_basis or "#" not in self.approval_basis:
                raise ValueError("approved_by='machine' requires approval_basis of the form <path>#<sha256>")
        elif self.approval_basis is not None:
            raise ValueError("approval_basis must be None unless approved_by='machine'")
        return self


class RequiredClaim(ContractModel):
    """A required claim: its text, evidence anchors and key-element aliases."""

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
        """Deterministic key-element matching is only possible when answer_keys are present."""

        return bool(self.answer_keys)


class ForbiddenClaim(ContractModel):
    """A claim that must not be affirmed in a false-premise case, with source anchors
    contradicting it (D8)."""

    claim_id: str = Field(min_length=1, description="题内唯一论断 ID（与 required claim 共用 ID 空间）。")
    text: str = Field(min_length=1, description="错误前提的表述。")
    evidence_anchors: list[EvidenceAlternative] = Field(
        default_factory=list, description="与该论断矛盾的原文锚点。"
    )


class EvalCase(ContractModel):
    """One end-to-end evaluation case (§13.1(a))."""

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
    """The frozen dataset; `fingerprint` binds its content so observations / mapping can verify
    identity."""

    schema_version: Literal["rag-e2e-v1"] = SCHEMA_VERSION
    dataset_id: str = Field(min_length=1, description="数据集版本身份。")
    corpus_version: str = Field(min_length=1, description="语料版本（manifest / corpus signature）。")
    cases: list[EvalCase] = Field(min_length=1)
    fingerprint: str | None = Field(default=None, description="除本字段外整个数据集规范化 JSON 的 sha256。")


def compute_fingerprint(dataset: E2EDataset) -> str:
    """Compute the fingerprint per §13.1(a): the sha256 of the canonical JSON with the
    `fingerprint` field excluded.

    Canonical JSON = `sort_keys=True`, `ensure_ascii=False`, `separators=(",", ":")`, UTF-8;
    `model_dump(mode="json")` emits every field (defaults included), so identical content always
    has exactly one serialization.

    Backward compatibility: when `provenance.approval_basis` is None it is dropped from the hashed
    payload. That field was introduced in v0.32, so datasets frozen earlier (for example
    `artifacts/stage_b/B3a1_pilot_dataset.json`) never had it in their serialization; omitting the
    null is what keeps their fingerprint unchanged. When `approval_basis` does have a value it is
    hashed as usual, so machine-acceptance evidence stays bound by the fingerprint.
    """

    payload = dataset.model_dump(mode="json", exclude={"fingerprint"})
    for case in payload.get("cases", []):
        provenance = case.get("provenance")
        if isinstance(provenance, dict) and provenance.get("approval_basis") is None:
            provenance.pop("approval_basis", None)
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def with_fingerprint(dataset: E2EDataset) -> E2EDataset:
    """Return a copy with the fingerprint filled in; the input object is left untouched."""

    return dataset.model_copy(update={"fingerprint": compute_fingerprint(dataset)})


def load_dataset(path: str | Path, *, verify_fingerprint: bool = True) -> E2EDataset:
    """Load the dataset; with `verify_fingerprint=True` a missing or mismatching fingerprint
    raises `ValueError`."""

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
    """One validation issue; `case_id=None` marks a dataset-level issue."""

    case_id: str | None
    rule: str = Field(description="规则名，见 validate_dataset 的 docstring。")
    message: str


class Ratio(ContractModel):
    """A ratio in the report: with an empty denominator the value is None rather than fabricated."""

    numerator: int
    denominator: int
    value: float | None


class ValidationReport(ContractModel):
    """Aggregated result of the deterministic validation."""

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
    """Return None when the denominator is empty."""

    return Ratio(numerator=numerator, denominator=denominator,
                 value=(numerator / denominator) if denominator else None)


def _duplicates(values: list[str]) -> list[str]:
    """List duplicated values in order of first appearance."""

    seen: set[str] = set()
    duplicated: list[str] = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return duplicated


def _check_answer_keys(case: EvalCase, claim: RequiredClaim, provider: PageTextProvider,
                       add: Callable[[str, str], None]) -> None:
    """The three rules of §13.1(c); called only for claims where key_matchable is set."""

    # Fetch the source text of each anchor's locator page up front; if one cannot be read, record
    # the problem instead of aborting the remaining checks.
    page_texts: list[str] = []
    for index, anchor in enumerate(claim.evidence_anchors):
        loc = anchor.locator
        try:
            page_texts.append(provider(anchor.document_id, loc.page_start, loc.page_end))
        except Exception as exc:  # noqa: BLE001 - validator collects issues, never raises
            add("answer_keys_source",
                f"{claim.claim_id}: cannot read source text of anchor #{index}: {type(exc).__name__}: {exc}")
    if not claim.evidence_anchors:
        add("answer_keys_source", f"{claim.claim_id}: key_matchable claim has no evidence_anchors")
    # The page texts only need normalizing once; rule ① of every element group reuses them.
    normalized_pages = [normalize_text(text) for text in page_texts]

    for group_index, group in enumerate(claim.answer_keys):
        label = f"{claim.claim_id} group #{group_index}"
        # ③ The group is non-empty, each alias is non-empty after stripping, its normalized length
        # is at least 2, and aliases are unique (by normalized form) within the group.
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
        # ② No alias may appear in the question, otherwise an answer that merely restates the
        # question would also be counted as a hit.
        for alias in group:
            if alias.strip() and alias_hits(alias, case.query):
                add("answer_keys_query", f"{label}: alias {alias!r} appears in query")
        # ① At least one alias must appear, after normalization, as a **literal substring** of the
        # source text of one of this claim's anchors, which proves the alias comes from the source.
        # The ASCII word boundary of alias_hits is deliberately not used here: when the source
        # writes `1.5mm`, the alias `1.5` must still pass validation, otherwise the bare numbers
        # that natural answers commonly write would make key_fact_recall systematically
        # under-count; the boundary rule stays only in ② and on the scoring side.
        if page_texts and not any(alias in text for alias in normalized_aliases if alias for text in normalized_pages):
            add("answer_keys_source", f"{label}: no alias found in any anchor's locator source text")


def _longest_common_word_run(left: str, right: str) -> int:
    """Length of the longest common consecutive word run required by §13.1(c′)④ (dynamic
    programming, O(n*m)).

    This is the same algorithm as `eval/e2e/draft.longest_common_word_run`: dataset must not depend
    on draft (draft needs EvalCase to build cases), so a private copy is kept here to avoid a
    circular import.
    """

    left_words = normalize_text(left).split()
    right_words = normalize_text(right).split()
    if not left_words or not right_words:
        return 0
    previous = [0] * (len(right_words) + 1)
    best = 0
    for left_word in left_words:
        current = [0] * (len(right_words) + 1)
        for index, right_word in enumerate(right_words, start=1):
            if left_word == right_word:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def _check_grounding(
    case: EvalCase,
    claim: RequiredClaim | ForbiddenClaim,
    provider: PageTextProvider,
    add: Callable[[str, str], None],
    *,
    forbidden: bool,
) -> None:
    """Apply the §13.1(c′) grounding rules; a forbidden claim only gets ③ length and
    ④ question-copy checks.

    The text of a forbidden claim is a false premise that should not appear in the source in the
    first place, so ① is not applied to it; it also takes no part in answer matching, so ② is not
    applied either.
    """

    claim_norm = normalize_text(claim.text)
    if not forbidden:
        # ① After normalization, claim.text must be a literal substring of the locator page source
        # text of one of its anchors.
        page_texts: list[str] = []
        for index, anchor in enumerate(claim.evidence_anchors):
            locator = anchor.locator
            try:
                page_texts.append(provider(anchor.document_id, locator.page_start, locator.page_end))
            except Exception as exc:  # noqa: BLE001 - validator collects issues, never raises
                add("claim_grounded",
                    f"{claim.claim_id}: cannot read source text of anchor #{index}: "
                    f"{type(exc).__name__}: {exc}")
        if claim.evidence_anchors and page_texts:
            normalized_pages = [normalize_text(text) for text in page_texts]
            if not any(claim_norm in page for page in normalized_pages):
                add("claim_grounded", f"{claim.claim_id}: claim text not found in any anchor's source text")
        # ② When answer_keys are present, at least one alias (after normalization) must be a
        # substring of claim.text, which guarantees "the answer is a fragment of this claim"; the
        # other aliases may come from the derivation table without this constraint.
        if claim.answer_keys:
            normalized_aliases = [
                normalize_text(alias) for group in claim.answer_keys for alias in group
            ]
            if not any(alias and alias in claim_norm for alias in normalized_aliases):
                add("claim_key_in_text", f"{claim.claim_id}: no answer key is a substring of the claim text")
    # ③ A claim must be at least 10 characters after normalization, ruling out yes/no phrasing.
    if len(claim_norm) < MIN_CLAIM_LENGTH:
        add("claim_too_short", f"{claim.claim_id}: claim shorter than {MIN_CLAIM_LENGTH} after normalization")
    # ④ The question and the claim may share at most 8 consecutive words, so a question that
    # restates the source cannot leak the answer.
    if _longest_common_word_run(case.query, claim.text) > MAX_COMMON_WORD_RUN:
        add("query_claim_copy",
            f"{claim.claim_id}: query copies more than {MAX_COMMON_WORD_RUN} consecutive words from the claim")


def validate_dataset(dataset: E2EDataset, page_text_provider: PageTextProvider, *,
                     require_approved: bool, allow_machine: bool = False,
                     enforce_grounding: bool = False) -> ValidationReport:
    """Run every deterministic check on the dataset, aggregate the problems, and never raise.

    Rule names (`ValidationIssue.rule`):
    - `duplicate_case_id` / `duplicate_group_id` / `duplicate_claim_id`: IDs are unique (groups and
      claims are unique within a case; required and forbidden claims share one ID space because the
      mapping file locates them by (case_id, claim_id));
    - `fingerprint_mismatch`: the dataset carries a fingerprint that no longer matches its content;
    - `answerable_required_claim`: an answerable case has at least one claim with `required=True`;
    - `answerable_evidence_group`: an answerable case has at least one `required_evidence_groups`;
    - `required_claim_anchors`: the required claims of an answerable case must carry evidence
      anchors;
    - `multi_doc`: a case tagged `multi_doc` has at least two evidence groups, and at least two of
      them belong to different documents;
    - `unanswerable_shape`: an unanswerable case has acceptable_abstention=false, a non-null
      unanswerable_kind, and empty claims / evidence groups / forbidden claims;
    - `answerable_unanswerable_kind`: an answerable case must have unanswerable_kind null;
    - `false_premise`: the D8 items; `forbidden_claims_non_false_premise`: a case without the
      false_premise tag must have no forbidden_claims;
    - `answer_keys_source` / `answer_keys_query` / `answer_keys_format`: §13.1(c) ①②③;
    - `claim_grounded` / `claim_key_in_text` / `claim_too_short` / `query_claim_copy`: §13.1(c′)
      ①②③④ (only executed when `enforce_grounding=True`; forbidden claims only get ③④);
    - `not_approved`: with `require_approved=True` a usable confirmation must exist. By default only
      `"user"` is accepted; with `allow_machine=True`, `"machine"` carrying a non-empty
      `approval_basis` is accepted as well (the freeze condition of §13.1(a)).
    """

    issues: list[ValidationIssue] = []

    def adder(case_id: str | None) -> Callable[[str, str], None]:
        """Build a problem collector bound to one case_id."""

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
            # An answerable case needs at least one required evidence group; otherwise the
            # context-type metrics have nothing to measure and the case cannot be justified once
            # the dataset is frozen.
            if not case.required_evidence_groups:
                add("answerable_evidence_group", "answerable case has no required_evidence_groups")
            for claim in required:
                if not claim.evidence_anchors:
                    add("required_claim_anchors", f"{claim.claim_id}: required claim has no evidence_anchors")
            # Ratio accounting: the denominator is every required claim of the answerable cases,
            # including the D8 false-premise ones.
            total_required += len(required)
            key_matchable_required += sum(1 for c in required if c.key_matchable)
            if case.unanswerable_kind is not None:
                add("answerable_unanswerable_kind", "answerable case must have unanswerable_kind=null")
        else:
            # An unanswerable case may only be abstained on, so it must not carry any scoreable
            # positive example.
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
            # Two groups i<j count as "covering different papers" only when each of them can pick
            # a different document.
            cross_doc = any(len(doc_sets[i] | doc_sets[j]) >= 2
                            for i in range(len(doc_sets)) for j in range(i + 1, len(doc_sets)))
            if len(doc_sets) < 2 or not cross_doc:
                add("multi_doc", "multi_doc case needs >= 2 evidence groups from >= 2 different documents")

        if is_false_premise:
            # D8: false-premise cases are scored as answerable, may be abstained on, and are not
            # counted as unanswerable.
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

        if enforce_grounding:
            # §13.1(c′): required claims go through ①②③④; forbidden claims only through ③④
            # (their text should not appear in the source at all).
            for claim in case.required_claims:
                _check_grounding(case, claim, page_text_provider, add, forbidden=False)
            for claim in case.forbidden_claims:
                _check_grounding(case, claim, page_text_provider, add, forbidden=True)

        if require_approved:
            # The default policy keeps the strict human confirmation of B3a-1; the machine
            # acceptance of D3 is only honoured once allow_machine is turned on.
            approved = case.provenance.approved_by == "user"
            machine_approved = (
                allow_machine
                and case.provenance.approved_by == "machine"
                and bool(case.provenance.approval_basis)
            )
            if not (approved or machine_approved):
                expected = "'user' or 'machine' with approval_basis" if allow_machine else "'user'"
                add("not_approved",
                    f"provenance.approved_by is {case.provenance.approved_by!r}, expected {expected}")

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
