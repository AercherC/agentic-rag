"""rag_e2e_v1 failure root-cause diagnosis: §13.1(e) layers plus each group's stopping stage.

Read-only and pure: it takes the dataset, the evidence mapping and the per-case observations, and
loads no model and issues no request.

Two levels of output:
- Question-level ``layer``: applies the §13.1(e) "failure layer attribution" rules in order and
  keeps the first layer that holds (ERROR / GENERATION / CITATION / STATUS / UNATTRIBUTABLE /
  CANDIDATE_MISS / RERANK_MISS / CONTEXT / UNVERIFIED / PASS). When generation did not run
  (``generation_ran=False``), a case whose retrieval and context checks both pass is recorded as
  ``RETRIEVAL_PASS``, and an unanswerable case as ``NOT_RUN``.
- Group-level ``stage``: how far each required evidence group's evidence travelled (UNMAPPABLE →
  CANDIDATE_MISS → RERANK_MISS → CONTEXT_DROP → IN_CONTEXT_NOT_CITED / IN_CONTEXT / CITED), plus
  the diagnostics needed to separate root causes: the full Dense rank (would a deeper candidate
  pool recover it), whether a fragmented set is only partly in the candidate pool (a chunking
  artefact), whether the required document never reached the candidates (document-level recall)
  or reached them with the wrong passage (passage-level), where the Reranker placed it, and the
  Top-K document diversity and the space taken by back matter such as references.

"Fully in" follows the §13.1(e) definition: every chunk of one chunk set is present in that stage's
result; across multiple sets inside a group the relation is "or".
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import Field, model_validator

from evidenceflow.domain.evaluation_v2 import ContractModel
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.eval.e2e.dataset import MULTI_DOC_TAG, E2EDataset, EvalCase
from evidenceflow.eval.e2e.key_match import claim_hits
from evidenceflow.eval.e2e.mapping import E2EGroupMapping, E2EMapping, is_usable

OBSERVATION_SCHEMA_VERSION = "rag-e2e-diagnosis-observation-v1"
REPORT_SCHEMA_VERSION = "rag-e2e-diagnosis-report-v1"

QuestionLayer = Literal[
    "ERROR",
    "GENERATION",
    "CITATION",
    "STATUS",
    "UNATTRIBUTABLE",
    "CANDIDATE_MISS",
    "RERANK_MISS",
    "CONTEXT",
    "UNVERIFIED",
    "PASS",
    "RETRIEVAL_PASS",
    "NOT_RUN",
]
GroupStage = Literal[
    "UNMAPPABLE",
    "CANDIDATE_MISS",
    "RERANK_MISS",
    "CONTEXT_DROP",
    "IN_CONTEXT",
    "IN_CONTEXT_NOT_CITED",
    "CITED",
]
# §13.1(e) rule 1: these three errors get a dedicated layer; any other exception is
# recorded as ERROR (left unattributed).
_ERROR_LAYERS = {
    "LLMRequestError": "ERROR",
    "GenerationContractError": "GENERATION",
    "CitationValidationError": "CITATION",
}
# Back matter: references, acknowledgements, appendix, supplementary material. A chunk whose
# heading_path matches is treated as back matter.
_BACK_MATTER = re.compile(r"\b(references?|bibliography|acknowledg\w*|appendix|appendices|supplementar\w*)\b", re.I)


class ObservedChunk(ContractModel):
    """A chunk in the Dense ranking: identity, document and heading path only, no text."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    heading_path: str | None = Field(default=None, description="hybrid 切分记录的章节路径；fixed 切分为 None。")


class E2EObservation(ContractModel):
    """Per-stage observation of one case on one index.

    Produced by ``scripts/e2e_diagnose.py observe``; each field keeps one stage's output so the
    diagnosis can tell where the evidence stopped.
    """

    schema_version: Literal["rag-e2e-diagnosis-observation-v1"] = OBSERVATION_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    chunk_count: int = Field(ge=0, description="索引 chunk 总数，用于核对映射文件身份。")
    dense_ranking: list[ObservedChunk] = Field(
        description="Exact Dense 排名（深度 ≥ 候选池），第 i 项名次为 i+1；超出深度的 chunk 视为未排到。"
    )
    candidate_pool_k: int = Field(gt=0, description="线上送入 Reranker 的 Dense 候选数（Top20）。")
    reranked_pool: list[str] = Field(description="Reranker 对候选池的完整排序（chunk_id）。")
    final_top_k: int = Field(gt=0, description="线上交给 Context 的最终深度（Top5）。")
    context_chunk_ids: list[str] = Field(description="完整纳入 BuiltContext 的 chunk，按标签顺序。")
    generation_ran: bool = Field(description="False = --retrieval-only，只观测检索与上下文层。")
    status: AnswerStatus | None = None
    answer: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list, description="最终回答实际引用的 chunk。")
    error_type: str | None = Field(default=None, description="生成或引用阶段抛出的异常类名。")

    @model_validator(mode="after")
    def validate_consistency(self):
        """Stage results must nest: context ⊆ final top-k ⊆ candidate pool ⊆ Dense ranking."""

        dense_ids = [item.chunk_id for item in self.dense_ranking]
        if len(dense_ids) != len(set(dense_ids)) or len(self.reranked_pool) != len(set(self.reranked_pool)):
            raise ValueError("duplicate chunk ids in a ranking")
        if not set(self.reranked_pool) <= set(dense_ids[: self.candidate_pool_k]):
            raise ValueError("reranked_pool must come from the dense candidate pool")
        if not set(self.context_chunk_ids) <= set(self.reranked_pool[: self.final_top_k]):
            raise ValueError("context chunks must come from the final top-k")
        if not set(self.source_chunk_ids) <= set(self.context_chunk_ids):
            raise ValueError("cited chunks must come from the context")
        if not self.generation_ran and (self.status or self.answer or self.source_chunk_ids or self.error_type):
            raise ValueError("retrieval-only observations must not carry generation results")
        if self.generation_ran and self.error_type is None and self.status is None:
            raise ValueError("a completed generation must record its status")
        return self


class GroupDiagnosis(ContractModel):
    """Stopping stage and root-cause diagnostics for one required evidence group."""

    group_id: str
    mapping_status: str
    stage: GroupStage
    document_ids: list[str] = Field(description="该组各 alternative 所在文档。")
    fragmented: bool = Field(description="映射为 FRAGMENTED_ACROSS_CHUNKS：证据跨多个 chunk。")
    best_dense_rank: int | None = Field(
        default=None, description="各集合“最后一个 chunk 的 Dense 名次”的最小值；超出观测深度为 None。"
    )
    partial_in_candidates: bool = Field(
        default=False, description="有集合部分 chunk 进入候选池，但没有集合完整进入（切分把证据拆散）。"
    )
    document_in_candidates: bool = Field(default=False, description="组内任一文档的任一 chunk 进入候选池。")
    document_in_final: bool = Field(default=False, description="组内任一文档的任一 chunk 进入最终 Top-K。")
    best_rerank_position: int | None = Field(
        default=None, description="完整进入候选池的集合在 Reranker 排序中的最好位置（集合内取最差 chunk）。"
    )


class CaseDiagnosis(ContractModel):
    """Failure layer of one case plus its per-group diagnostics."""

    case_id: str
    tags: list[str]
    expected_status: AnswerStatus
    layer: QuestionLayer
    groups: list[GroupDiagnosis] = Field(default_factory=list)
    final_distinct_documents: int = Field(ge=0, description="最终 Top-K 中不同文档数。")
    final_dominant_share: float = Field(ge=0, le=1, description="最终 Top-K 中占比最高的单一文档份额。")
    candidate_distinct_documents: int = Field(ge=0, description="候选池中不同文档数。")
    final_back_matter_chunks: int = Field(ge=0, description="最终 Top-K 中来自参考文献/附录等后置章节的 chunk 数。")


def is_back_matter(heading_path: str | None) -> bool:
    """Whether the heading path falls in references, acknowledgements, appendix or supplements."""

    return bool(heading_path) and _BACK_MATTER.search(heading_path) is not None


def _fully_in(bundles: Sequence[Sequence[str]], selected: set[str]) -> bool:
    """Every chunk of any set inside the group is in ``selected`` (outer OR, inner AND)."""

    return any(set(bundle) <= selected for bundle in bundles)


def _diagnose_group(
    case: EvalCase, group_mapping: E2EGroupMapping, observation: E2EObservation
) -> GroupDiagnosis:
    """Locate the stage where this group's evidence stopped.

    Walks candidate pool → final top-k → context → citations, in that order.
    """

    group = next(g for g in case.required_evidence_groups if g.group_id == group_mapping.group_id)
    documents = sorted({alternative.document_id for alternative in group.alternatives})
    status = group_mapping.status
    base = {"group_id": group.group_id, "mapping_status": status, "document_ids": documents,
            "fragmented": status == "FRAGMENTED_ACROSS_CHUNKS"}
    if not is_usable(status):
        return GroupDiagnosis(stage="UNMAPPABLE", **base)

    bundles = group_mapping.chunk_sets
    dense_rank = {item.chunk_id: rank for rank, item in enumerate(observation.dense_ranking, start=1)}
    candidates = observation.dense_ranking[: observation.candidate_pool_k]
    candidate_ids = {item.chunk_id for item in candidates}
    final_ids = observation.reranked_pool[: observation.final_top_k]
    final_set = set(final_ids)
    rerank_position = {chunk_id: position for position, chunk_id in enumerate(observation.reranked_pool, start=1)}
    doc_of = {item.chunk_id: item.document_id for item in observation.dense_ranking}

    # A set's rank is the rank of its last chunk, because the set only counts as covered when
    # every chunk is retrieved; if any chunk is beyond the observed depth, the set has no rank.
    set_ranks = [max(dense_rank[c] for c in bundle) for bundle in bundles if all(c in dense_rank for c in bundle)]
    in_candidates = _fully_in(bundles, candidate_ids)
    positions = [
        max(rerank_position[c] for c in bundle)
        for bundle in bundles
        if set(bundle) <= candidate_ids and all(c in rerank_position for c in bundle)
    ]
    diagnosis = dict(
        base,
        best_dense_rank=min(set_ranks) if set_ranks else None,
        partial_in_candidates=not in_candidates and any(set(bundle) & candidate_ids for bundle in bundles),
        document_in_candidates=any(item.document_id in documents for item in candidates),
        document_in_final=any(doc_of.get(chunk_id) in documents for chunk_id in final_ids),
        best_rerank_position=min(positions) if positions else None,
    )
    if not in_candidates:
        stage = "CANDIDATE_MISS"
    elif not _fully_in(bundles, final_set):
        stage = "RERANK_MISS"
    elif not _fully_in(bundles, set(observation.context_chunk_ids)):
        stage = "CONTEXT_DROP"
    elif not observation.generation_ran:
        stage = "IN_CONTEXT"
    elif any(set(bundle) & set(observation.source_chunk_ids) for bundle in bundles):
        stage = "CITED"
    else:
        stage = "IN_CONTEXT_NOT_CITED"
    return GroupDiagnosis(stage=stage, **diagnosis)


def _status_correct(case: EvalCase, actual: AnswerStatus) -> bool:
    """§13.1(a) status is correct: equal, or answered expected while abstaining is allowed."""

    if actual == case.expected_status:
        return True
    return (
        case.expected_status == AnswerStatus.ANSWERED
        and actual == AnswerStatus.INSUFFICIENT_CONTEXT
        and case.acceptable_abstention
    )


def _evidence_chunk_union(case_id: str, mapping: E2EMapping) -> set[str]:
    """All chunks mapped by this case's evidence groups and claim anchors.

    This is the "union of the evidence mapping chunk sets" of §13.1(e).
    """

    records = [g for g in mapping.groups if g.case_id == case_id] + [
        c for c in mapping.claim_anchors if c.case_id == case_id
    ]
    return {chunk_id for record in records for bundle in record.chunk_sets for chunk_id in bundle}


def _question_layer(
    case: EvalCase, groups: list[GroupDiagnosis], observation: E2EObservation, mapping: E2EMapping
) -> str:
    """Return the first failure layer that holds, in §13.1(e) order."""

    if observation.error_type is not None:
        return _ERROR_LAYERS.get(observation.error_type, "ERROR")
    if case.expected_status == AnswerStatus.INSUFFICIENT_CONTEXT:
        if not observation.generation_ran:
            return "NOT_RUN"
        return "STATUS" if observation.status == AnswerStatus.ANSWERED else "PASS"
    stages = {group.stage for group in groups}
    for layer, stage in (
        ("UNATTRIBUTABLE", "UNMAPPABLE"),
        ("CANDIDATE_MISS", "CANDIDATE_MISS"),
        ("RERANK_MISS", "RERANK_MISS"),
        ("CONTEXT", "CONTEXT_DROP"),
    ):
        if stage in stages:
            return layer
    if not observation.generation_ran:
        return "RETRIEVAL_PASS"
    if not _status_correct(case, observation.status):
        return "STATUS"
    if observation.status == AnswerStatus.INSUFFICIENT_CONTEXT:
        # Acceptable abstention: the status is correct, so §13.1(e) rule 7 records PASS directly.
        return "PASS"
    keyed = [claim for claim in case.required_claims if claim.required and claim.key_matchable]
    if not keyed:
        return "UNVERIFIED"
    if not all(claim_hits(claim.answer_keys, observation.answer or "") for claim in keyed):
        return "GENERATION"
    if not set(observation.source_chunk_ids) & _evidence_chunk_union(case.case_id, mapping):
        return "CITATION"
    return "PASS"


def diagnose_case(case: EvalCase, mapping: E2EMapping, observation: E2EObservation) -> CaseDiagnosis:
    """Diagnose one case: failure layer, per-group stages and final Top-K document composition."""

    if observation.case_id != case.case_id:
        raise ValueError(f"observation {observation.case_id} does not belong to case {case.case_id}")
    by_group = {g.group_id: g for g in mapping.groups if g.case_id == case.case_id}
    missing = [g.group_id for g in case.required_evidence_groups if g.group_id not in by_group]
    if missing:
        raise ValueError(f"mapping has no record for {case.case_id} groups {missing}")
    groups = [_diagnose_group(case, by_group[g.group_id], observation) for g in case.required_evidence_groups]

    doc_of = {item.chunk_id: item for item in observation.dense_ranking}
    final = [doc_of[chunk_id] for chunk_id in observation.reranked_pool[: observation.final_top_k]]
    final_docs = Counter(item.document_id for item in final)
    candidate_docs = {item.document_id for item in observation.dense_ranking[: observation.candidate_pool_k]}
    return CaseDiagnosis(
        case_id=case.case_id,
        tags=list(case.tags),
        expected_status=case.expected_status,
        layer=_question_layer(case, groups, observation, mapping),
        groups=groups,
        final_distinct_documents=len(final_docs),
        final_dominant_share=(max(final_docs.values()) / len(final)) if final else 0.0,
        candidate_distinct_documents=len(candidate_docs),
        final_back_matter_chunks=sum(is_back_matter(item.heading_path) for item in final),
    )


def check_identity(mapping: E2EMapping, observations: Sequence[E2EObservation], dataset_fingerprint: str) -> None:
    """Refuse to diagnose unless mapping, observations and dataset share one identity.

    The identity check runs first so a stale mapping or a stale observation cannot silently
    produce meaningless diagnoses.
    """

    if mapping.dataset_fingerprint != dataset_fingerprint:
        raise ValueError("mapping was built for a different dataset fingerprint")
    for observation in observations:
        if observation.collection_name != mapping.collection_name:
            raise ValueError(
                f"observation {observation.case_id} is from collection {observation.collection_name!r}, "
                f"mapping is for {mapping.collection_name!r}"
            )
        if mapping.chunk_count is not None and observation.chunk_count != mapping.chunk_count:
            raise ValueError(
                f"observation {observation.case_id} saw {observation.chunk_count} chunks, "
                f"mapping was built on {mapping.chunk_count}"
            )


def diagnose_dataset(
    dataset: E2EDataset, mapping: E2EMapping, observations: Iterable[E2EObservation], dataset_fingerprint: str
) -> list[CaseDiagnosis]:
    """Diagnose every case that has an observation.

    Cases without an observation are skipped; an observation naming a case outside the dataset is
    an error, because it means the two artifacts come from different dataset versions.
    """

    observations = list(observations)
    check_identity(mapping, observations, dataset_fingerprint)
    cases = {case.case_id: case for case in dataset.cases}
    unknown = sorted({o.case_id for o in observations} - set(cases))
    if unknown:
        raise ValueError(f"observations for unknown cases: {unknown}")
    return [diagnose_case(cases[o.case_id], mapping, o) for o in observations]


def _rank_bucket(rank: int | None, candidate_pool_k: int) -> str:
    """Bucket a Dense rank to show how much deeper the candidate pool must go to recover it."""

    if rank is None:
        return "beyond_observed_depth"
    if rank <= candidate_pool_k:
        return f"within_top{candidate_pool_k}_but_incomplete"
    if rank <= 50:
        return f"rank_{candidate_pool_k + 1}_to_50"
    if rank <= 100:
        return "rank_51_to_100"
    return "rank_over_100"


def summarize(diagnoses: Sequence[CaseDiagnosis], candidate_pool_k: int, final_top_k: int) -> dict:
    """Summarize one slice.

    Covers the question-level failure layers, the group-level stopping stages and the counters that
    separate the root causes.
    """

    groups = [group for diagnosis in diagnoses for group in diagnosis.groups]
    answerable = [d for d in diagnoses if d.expected_status == AnswerStatus.ANSWERED]
    candidate_misses = [g for g in groups if g.stage == "CANDIDATE_MISS"]
    rerank_misses = [g for g in groups if g.stage == "RERANK_MISS"]
    fragmented_usable = [g for g in groups if g.fragmented]

    def share(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "cases": len(diagnoses),
        "answerable_cases": len(answerable),
        "evidence_groups": len(groups),
        "question_layers": dict(sorted(Counter(d.layer for d in diagnoses).items())),
        "group_stages": dict(sorted(Counter(g.stage for g in groups).items())),
        "unmappable_by_status": dict(sorted(Counter(g.mapping_status for g in groups if g.stage == "UNMAPPABLE").items())),
        "candidate_miss": {
            "groups": len(candidate_misses),
            # Document-level vs passage-level: either no chunk of the required document reached
            # the candidate pool (document recall failed), or the document did arrive but without
            # the evidence passage.
            "document_absent_from_candidates": sum(not g.document_in_candidates for g in candidate_misses),
            "document_present_wrong_passage": sum(g.document_in_candidates for g in candidate_misses),
            # Chunking: only part of a fragmented set made it into the candidate pool.
            "fragment_partially_retrieved": sum(g.partial_in_candidates for g in candidate_misses),
            "dense_rank_buckets": dict(sorted(Counter(
                _rank_bucket(g.best_dense_rank, candidate_pool_k) for g in candidate_misses).items())),
        },
        "rerank_miss": {
            "groups": len(rerank_misses),
            "rerank_position_buckets": dict(sorted(Counter(
                "unknown" if g.best_rerank_position is None
                else f"{final_top_k + 1}_to_10" if g.best_rerank_position <= 10
                else f"11_to_{candidate_pool_k}"
                for g in rerank_misses).items())),
            "document_absent_from_final": sum(not g.document_in_final for g in rerank_misses),
        },
        "fragmented_groups": {
            "total": len(fragmented_usable),
            "reached_context": sum(g.stage in {"IN_CONTEXT", "IN_CONTEXT_NOT_CITED", "CITED"} for g in fragmented_usable),
        },
        "final_top_k_composition": {
            "mean_distinct_documents": share([d.final_distinct_documents for d in answerable]),
            "mean_dominant_document_share": share([d.final_dominant_share for d in answerable]),
            "mean_candidate_distinct_documents": share([d.candidate_distinct_documents for d in answerable]),
            "cases_with_back_matter_in_final": sum(d.final_back_matter_chunks > 0 for d in answerable),
            "back_matter_chunks_in_final": sum(d.final_back_matter_chunks for d in answerable),
        },
    }


def build_report(
    diagnoses: Sequence[CaseDiagnosis], *, candidate_pool_k: int, final_top_k: int, generation_ran: bool
) -> dict:
    """Build the three report slices and attach per-case detail for spot checks.

    The slices are multi-doc cases, single-doc answerable cases (the control group) and all cases.
    """

    multi = [d for d in diagnoses if MULTI_DOC_TAG in d.tags]
    single = [d for d in diagnoses if MULTI_DOC_TAG not in d.tags and d.expected_status == AnswerStatus.ANSWERED]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generation_ran": generation_ran,
        "candidate_pool_k": candidate_pool_k,
        "final_top_k": final_top_k,
        "slices": {
            "multi_doc": summarize(multi, candidate_pool_k, final_top_k),
            "single_doc_answerable": summarize(single, candidate_pool_k, final_top_k),
            "all": summarize(diagnoses, candidate_pool_k, final_top_k),
        },
        "cases": [d.model_dump(mode="json") for d in diagnoses],
    }


def render_markdown(report: dict) -> str:
    """Render the diagnosis report as a side-by-side table of multi-doc vs single-doc cases.

    Each cell shows "count (share)".
    """

    multi, single = report["slices"]["multi_doc"], report["slices"]["single_doc_answerable"]

    def cell(count: int, total: int) -> str:
        return f"{count} ({count / total:.0%})" if total else f"{count}"

    def table(title: str, key: str, total_key: str, labels: Sequence[str]) -> list[str]:
        rows = [f"### {title}", "", "| | multi-doc | single-doc |", "| --- | --- | --- |"]
        for label in labels:
            rows.append(
                f"| {label} | {cell(multi[key].get(label, 0), multi[total_key])} "
                f"| {cell(single[key].get(label, 0), single[total_key])} |"
            )
        return rows + [""]

    layers = sorted(set(multi["question_layers"]) | set(single["question_layers"]))
    stages = ["UNMAPPABLE", "CANDIDATE_MISS", "RERANK_MISS", "CONTEXT_DROP", "IN_CONTEXT",
              "IN_CONTEXT_NOT_CITED", "CITED"]
    lines = [
        "# rag_e2e_v1 failure diagnosis",
        "",
        f"Generation {'ran' if report['generation_ran'] else 'not run (retrieval-only)'}; "
        f"candidate pool top-{report['candidate_pool_k']}, final top-{report['final_top_k']}.",
        "",
        *table("Question-level failure layer (§13.1(e))", "question_layers", "cases", layers),
        *table("Where each required evidence group stopped", "group_stages", "evidence_groups", stages),
        "### Root-cause signals",
        "",
        "| signal | multi-doc | single-doc |",
        "| --- | --- | --- |",
    ]
    for label, path in (
        ("candidate miss: required document absent from top-20", ("candidate_miss", "document_absent_from_candidates")),
        ("candidate miss: document present, evidence passage missing", ("candidate_miss", "document_present_wrong_passage")),
        ("candidate miss: fragmented evidence only partly retrieved", ("candidate_miss", "fragment_partially_retrieved")),
        ("rerank miss: document absent from final top-k", ("rerank_miss", "document_absent_from_final")),
        ("fragmented evidence groups (total)", ("fragmented_groups", "total")),
        ("cases with back-matter chunks in final top-k", ("final_top_k_composition", "cases_with_back_matter_in_final")),
        ("mean distinct documents in final top-k", ("final_top_k_composition", "mean_distinct_documents")),
        ("mean share of the dominant document in final top-k", ("final_top_k_composition", "mean_dominant_document_share")),
    ):
        lines.append(f"| {label} | {multi[path[0]][path[1]]} | {single[path[0]][path[1]]} |")
    lines += ["", "### Dense rank of candidate-miss evidence", "", "| bucket | multi-doc | single-doc |", "| --- | --- | --- |"]
    buckets = sorted(set(multi["candidate_miss"]["dense_rank_buckets"]) | set(single["candidate_miss"]["dense_rank_buckets"]))
    for bucket in buckets:
        lines.append(
            f"| {bucket} | {multi['candidate_miss']['dense_rank_buckets'].get(bucket, 0)} "
            f"| {single['candidate_miss']['dense_rank_buckets'].get(bucket, 0)} |"
        )
    return "\n".join(lines) + "\n"
