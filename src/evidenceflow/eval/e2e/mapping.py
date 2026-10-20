"""把 rag_e2e_v1 题集的文本锚点映射到某个索引的 chunk（PROJECT_SPEC §13.1(b)）。

定位与规范化完全复用 `eval/stable_mapping.py` 的 `validate_source` 与 `map_source_to_chunks`
（与 canonical Gold v3 映射同一算法），本模块只负责：
- 遍历题集中的全部锚点（证据组，以及 required / forbidden claim 的 `evidence_anchors`）；
- 按 `map_stable_dataset` 相同的方式把多个 alternative 的结果合成一条记录（外层 OR、内层 AND）；
- 输出与题集分离、绑定 `dataset_fingerprint` / `collection_name` / `index_signature` / `chunk_count` /
  `chunk_id_hash` 的映射文件模型。

与 `map_stable_dataset` 的唯一差别：某个锚点源文本校验失败时，`map_stable_dataset` 整体返回 None，
这里只把该锚点记为 `SOURCE_EXTRACTION_FAILURE`，其余锚点照常映射，便于一次列出全部问题。
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
# SPEC §13.1(b)：只有这两种状态可用于计分，其余统称 UNMAPPABLE。
USABLE_STATUSES: frozenset[str] = frozenset({"FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS"})


def is_usable(status: str) -> bool:
    """映射状态是否可用于上下文类指标与失败层归因。"""

    return status in USABLE_STATUSES


def compute_chunk_id_hash(chunk_ids: Iterable[str]) -> str:
    """按 A1 基线口径计算 chunk_id_hash：排序后 chunk_id 列表的规范化 JSON 的 sha256。

    口径固定为 `json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))` 的 UTF-8
    字节 sha256（与 `artifacts/stage_a/A1_exact_baseline.json` 相同）。跨机器/时区/进程都确定，
    用于证明映射文件对应的正是某个已冻结的 chunk 集合。
    """

    serialized = json.dumps(sorted(chunk_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class _MappedRecord(ContractModel):
    """映射记录公共部分：状态与 chunk 集合必须一致。"""

    status: MappingStatus
    chunk_sets: list[list[str]] = Field(
        default_factory=list, description="外层 OR：任一集合完整进入上下文即覆盖；内层 AND。"
    )

    @model_validator(mode="after")
    def validate_sets(self):
        """可用状态必须有集合，不可用状态不得携带伪 Gold（与 GroupMapping 同一约束）。"""

        if is_usable(self.status) != bool(self.chunk_sets):
            raise ValueError("mapping status and chunk_sets disagree")
        if any(not bundle or len(bundle) != len(set(bundle)) for bundle in self.chunk_sets):
            raise ValueError("chunk bundles must be non-empty and unique")
        return self


class E2EGroupMapping(_MappedRecord):
    """一个必需证据组的映射。"""

    case_id: str
    group_id: str


class E2EClaimMapping(_MappedRecord):
    """一条 claim 的锚点映射；多个锚点之间“任一可用即可”。"""

    case_id: str
    claim_id: str
    kind: Literal["required", "forbidden"]


class E2EMapping(ContractModel):
    """题集对某个索引的证据映射文件；身份四项任一不一致时 score 拒绝计分。"""

    schema_version: Literal["rag-e2e-mapping-v1"] = MAPPING_SCHEMA_VERSION
    dataset_fingerprint: str = Field(min_length=1, description="题集内容 fingerprint（§13.1(a)）。")
    collection_name: str = Field(min_length=1, description="被映射的 Chroma collection。")
    index_signature: str = Field(min_length=1, description="lifecycle active 指针记录的 index_signature。")
    # 两个字段可选：旧 pilot 产物没有它们，读取时必须仍能通过 model_validate。
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
        """同一 (case, group) 或 (case, claim) 不得出现两次。"""

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
    """映射一组可互换锚点，返回合成后的状态与 chunk 集合。

    每个锚点：先 `validate_source`（只看原文），通过后 `map_source_to_chunks`（对齐 chunk）。
    合成规则与 `map_stable_dataset` 一致：所有锚点的 chunk 集合按顺序并列（外层 OR）；
    有单 chunk 集合 → FULLY_MAPPED；只有多 chunk 集合 → FRAGMENTED_ACROSS_CHUNKS；
    都没有时，有 PARTIALLY_MAPPED 取之，否则 UNMAPPED。
    额外约定：没有锚点 → UNMAPPED；全部锚点源文本校验失败 → SOURCE_EXTRACTION_FAILURE。
    """

    outcomes: list[tuple[str, list[list[str]]]] = []
    for alternative in alternatives:
        document = documents_by_id.get(alternative.document_id)
        validation = validate_source(alternative, document)
        if not validation["source_valid"]:
            # 源文本定位失败（文档缺失、hash 不符、页码不符或 occurrence 不唯一）：不进入 chunk 对齐。
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
    """把题集全部锚点映射到给定 chunk 集合（只读，不依赖检索排名或模型）。

    `documents` 必须是与锚点 `content_version` 同一版本的原文（`load_document` 结果，未经 clean，
    与 canonical Gold v3 映射使用同一文本基础）；`chunks` 是目标索引的全部 chunk。

    传入 `expected_chunk_count` / `expected_chunk_id_hash` 时，实测值不等即抛 `ValueError`：
    映射文件必须与 lifecycle 指针/manifest 记录的真实索引身份绑定，不能只抄写指针字符串。
    实测值无论是否传入期望都会写进返回模型，便于事后审计。
    """

    documents = list(documents)
    documents_by_id = {d.document_id: d for d in documents}
    if len(documents_by_id) != len(documents) or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate document/chunk IDs")
    # 先核对索引身份，避免把映射写到错误的 chunk 集合上。
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
