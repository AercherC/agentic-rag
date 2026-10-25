"""Map the text anchors of the rag_e2e_v1 question set onto the chunks of a given index
(PROJECT_SPEC §13.1(b)).

Locating and normalization fully reuse `validate_source` and `map_source_to_chunks` from
`eval/stable_mapping.py` (the same algorithm as the canonical Gold v3 mapping). This module only:
- walks every anchor in the question set (evidence groups, plus the `evidence_anchors` of required /
  forbidden claims);
- combines the results of several alternatives into one record the same way `map_stable_dataset` does
  (outer OR, inner AND);
- emits a mapping-file model that is separate from the question set and bound to `dataset_fingerprint` /
  `collection_name` / `index_signature` / `chunk_count` / `chunk_id_hash`.

The only difference from `map_stable_dataset`: when one anchor's source text fails validation,
`map_stable_dataset` returns None overall, whereas here only that anchor is recorded as
`SOURCE_EXTRACTION_FAILURE` and the remaining anchors are still mapped, so every problem can be listed
at once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import Field, model_validator

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.document import Document
from evidenceflow.domain.evaluation_v2 import ContractModel, EvidenceAlternative, MappingStatus
from evidenceflow.eval.e2e.dataset import E2EDataset, compute_fingerprint
from evidenceflow.eval.stable_mapping import map_source_to_chunks, validate_source

MAPPING_SCHEMA_VERSION = "rag-e2e-mapping-v1"
# SPEC §13.1(b): only these two statuses may be used for scoring; everything else is
# lumped together as UNMAPPABLE.
USABLE_STATUSES: frozenset[str] = frozenset({"FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS"})


def is_usable(status: str) -> bool:
    """Whether the mapping status can feed context metrics and failure-layer attribution."""

    return status in USABLE_STATUSES


def compute_chunk_id_hash(chunk_ids: Iterable[str]) -> str:
    """Compute chunk_id_hash on the A1 baseline standard: sha256 of the canonical JSON of the
    sorted chunk_id list.

    The standard is fixed as the UTF-8 byte sha256 of
    `json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))` (identical to
    `artifacts/stage_a/A1_exact_baseline.json`). It is deterministic across machines, time zones,
    and processes, and proves that the mapping file corresponds to one specific frozen chunk set.
    """

    serialized = json.dumps(sorted(chunk_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class _MappedRecord(ContractModel):
    """Shared part of a mapping record: the status and the chunk sets must be consistent."""

    status: MappingStatus
    chunk_sets: list[list[str]] = Field(
        default_factory=list, description="外层 OR：任一集合完整进入上下文即覆盖；内层 AND。"
    )

    @model_validator(mode="after")
    def validate_sets(self):
        """A usable status requires chunk sets; an unusable status must not carry fake Gold
        (the same constraint as GroupMapping)."""

        if is_usable(self.status) != bool(self.chunk_sets):
            raise ValueError("mapping status and chunk_sets disagree")
        if any(not bundle or len(bundle) != len(set(bundle)) for bundle in self.chunk_sets):
            raise ValueError("chunk bundles must be non-empty and unique")
        return self


class E2EGroupMapping(_MappedRecord):
    """Mapping of one required evidence group."""

    case_id: str
    group_id: str


class E2EClaimMapping(_MappedRecord):
    """Anchor mapping of one claim; among several anchors, any usable one suffices."""

    case_id: str
    claim_id: str
    kind: Literal["required", "forbidden"]


class E2EMapping(ContractModel):
    """Evidence mapping of the question set onto one index; if any of the four identity items
    disagrees, scoring refuses to score."""

    schema_version: Literal["rag-e2e-mapping-v1"] = MAPPING_SCHEMA_VERSION
    dataset_fingerprint: str = Field(min_length=1, description="题集内容 fingerprint（§13.1(a)）。")
    collection_name: str = Field(min_length=1, description="被映射的 Chroma collection。")
    index_signature: str = Field(min_length=1, description="lifecycle active 指针记录的 index_signature。")
    # Both fields are optional: older pilot artifacts lack them, and reading such files must
    # still pass model_validate.
    chunk_count: int | None = Field(
        default=None, ge=0, description="实测 chunk 总数；必须等于 active 指针的 chunk_count。"
    )
    chunk_id_hash: str | None = Field(
        default=None, min_length=1, description="排序 chunk_id 列表的规范化 JSON sha256（A1 基线口径）。"
    )
    groups: list[E2EGroupMapping]
    claim_anchors: list[E2EClaimMapping]

    @model_validator(mode="after")
    def validate_identity(self):
        """The same (case, group) or (case, claim) must not appear twice."""

        group_keys = [(g.case_id, g.group_id) for g in self.groups]
        claim_keys = [(c.case_id, c.claim_id) for c in self.claim_anchors]
        if len(group_keys) != len(set(group_keys)) or len(claim_keys) != len(set(claim_keys)):
            raise ValueError("duplicate mapping record")
        return self


def map_alternatives(
    alternatives: Sequence[EvidenceAlternative],
    documents_by_id: dict[str, Document],
    chunks: Sequence[DocumentChunk],
) -> tuple[MappingStatus, list[list[str]]]:
    """Map one group of interchangeable anchors and return the combined status and chunk sets.

    Each anchor: first `validate_source` (source text only), then, if it passes,
    `map_source_to_chunks` (chunk alignment). The combination rule matches
    `map_stable_dataset`: the chunk sets of all anchors are listed side by side in order
    (outer OR); a single-chunk set -> FULLY_MAPPED; only multi-chunk sets ->
    FRAGMENTED_ACROSS_CHUNKS; when there are none, PARTIALLY_MAPPED is taken if present,
    otherwise UNMAPPED.
    Additional conventions: no anchors -> UNMAPPED; all anchors fail source-text validation ->
    SOURCE_EXTRACTION_FAILURE.
    """

    outcomes: list[tuple[str, list[list[str]]]] = []
    for alternative in alternatives:
        document = documents_by_id.get(alternative.document_id)
        validation = validate_source(alternative, document)
        if not validation["source_valid"]:
            # Source-text location failed (missing document, hash mismatch, page mismatch, or a
            # non-unique occurrence): skip chunk alignment.
            outcomes.append(("SOURCE_EXTRACTION_FAILURE", []))
            continue
        status, bundles, _trace = map_source_to_chunks(alternative, document, chunks, validation)
        outcomes.append((status, bundles))
    bundles = [bundle for _, sets in outcomes for bundle in sets]
    if any(len(bundle) == 1 for bundle in bundles):
        return "FULLY_MAPPED", bundles
    if bundles:
        return "FRAGMENTED_ACROSS_CHUNKS", bundles
    if any(status == "PARTIALLY_MAPPED" for status, _ in outcomes):
        return "PARTIALLY_MAPPED", []
    if outcomes and all(status == "SOURCE_EXTRACTION_FAILURE" for status, _ in outcomes):
        return "SOURCE_EXTRACTION_FAILURE", []
    return "UNMAPPED", []


def map_dataset(
    dataset: E2EDataset,
    documents: Iterable[Document],
    chunks: Sequence[DocumentChunk],
    *,
    collection_name: str,
    index_signature: str,
    expected_chunk_count: int | None = None,
    expected_chunk_id_hash: str | None = None,
) -> E2EMapping:
    """Map every anchor of the question set onto the given chunk set (read-only; independent
    of retrieval ranking and models).

    `documents` must be the source text of the same version as the anchors' `content_version`
    (the `load_document` result, not cleaned, i.e. the same text basis used by the canonical
    Gold v3 mapping); `chunks` are all chunks of the target index.

    When `expected_chunk_count` / `expected_chunk_id_hash` are supplied, a measured value that
    differs raises `ValueError`: the mapping file must be bound to the real index identity
    recorded by the lifecycle pointer/manifest, not merely repeat the pointer string. The
    measured values are written into the returned model whether or not expectations are
    supplied, so they can be audited later.
    """

    documents = list(documents)
    documents_by_id = {d.document_id: d for d in documents}
    if len(documents_by_id) != len(documents) or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate document/chunk IDs")
    # Verify the index identity first, so the mapping is never written against the wrong chunk set.
    chunk_count = len(chunks)
    chunk_id_hash = compute_chunk_id_hash(chunk.chunk_id for chunk in chunks)
    if expected_chunk_count is not None and chunk_count != expected_chunk_count:
        raise ValueError(f"chunk_count mismatch: expected {expected_chunk_count}, got {chunk_count}")
    if expected_chunk_id_hash is not None and chunk_id_hash != expected_chunk_id_hash:
        raise ValueError(
            f"chunk_id_hash mismatch: expected {expected_chunk_id_hash}, got {chunk_id_hash}"
        )
    groups: list[E2EGroupMapping] = []
    claims: list[E2EClaimMapping] = []
    for case in dataset.cases:
        for group in case.required_evidence_groups:
            status, bundles = map_alternatives(group.alternatives, documents_by_id, chunks)
            groups.append(E2EGroupMapping(case_id=case.case_id, group_id=group.group_id,
                                          status=status, chunk_sets=bundles))
        for kind, items in (("required", case.required_claims), ("forbidden", case.forbidden_claims)):
            for claim in items:
                status, bundles = map_alternatives(claim.evidence_anchors, documents_by_id, chunks)
                claims.append(E2EClaimMapping(case_id=case.case_id, claim_id=claim.claim_id, kind=kind,
                                              status=status, chunk_sets=bundles))
    return E2EMapping(dataset_fingerprint=compute_fingerprint(dataset), collection_name=collection_name,
                      index_signature=index_signature, chunk_count=chunk_count, chunk_id_hash=chunk_id_hash,
                      groups=groups, claim_anchors=claims)
