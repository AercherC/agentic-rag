"""以原文定位验证 Gold，再映射到特定版本的 Chunk；不执行模糊相关性判断。"""

import re
import unicodedata

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.document import Document
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, GroupMapping, MappingArtifact


def normalize_anchor(text: str) -> str:
    """保留历史 research-anchor-v1 规范化，明确处理行尾断词。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _loose_hyphen(text: str) -> str:
    """仅用于诊断 decode 丢失换行后的断词；不自动变成 Gold。"""
    return re.sub(r"(?<=\w)-\s+(?=\w)", "", normalize_anchor(text))


def _occurrences(text: str, anchor: str) -> list[int]:
    """返回包括重叠在内的所有位置，不以第一个命中掩盖歧义。"""
    return [match.start() for match in re.finditer(f"(?={re.escape(anchor)})", text)]


def _map_alternative(alternative, document, chunks):
    """验证一条替代证据，并返回状态、足够的 Chunk 集合与诊断。"""
    if document is None or not document.segments:
        return "SOURCE_EXTRACTION_FAILURE", [], "source unavailable or extraction empty"
    if document.content_hash != alternative.content_version:
        raise ValueError(f"source content version mismatch: {alternative.document_id}")
    chunks = [chunk for chunk in chunks if chunk.document_id == document.document_id]
    if any(chunk.content_hash != alternative.content_version for chunk in chunks):
        raise ValueError(f"index content version mismatch: {alternative.document_id}")
    locator = alternative.locator
    if locator.block_index is not None:
        if locator.block_index >= len(document.segments):
            return "UNMAPPED", [], "block locator out of range"
        selected = [document.segments[locator.block_index]]
        text = selected[0].content
        if locator.span_start is not None:
            if locator.span_end > len(text):
                return "UNMAPPED", [], "character span out of range"
            text = text[locator.span_start:locator.span_end]
    else:
        selected = [s for s in document.segments if s.page_number is not None
                    and locator.page_start <= s.page_number <= locator.page_end]
        text = "\n".join(s.content for s in selected)
    if not text.strip():
        return "SOURCE_EXTRACTION_FAILURE", [], "selected source has no extracted text"
    anchor = normalize_anchor(alternative.evidence_anchor)
    positions = _occurrences(normalize_anchor(text), anchor)
    if not positions:
        status = "ANCHOR_NORMALIZATION_FAILURE" if _loose_hyphen(alternative.evidence_anchor) in _loose_hyphen(text) else "UNMAPPED"
        return status, [], "anchor not found in declared source locator"
    if len(positions) > 1:
        return "AMBIGUOUS_MAPPING", [], f"anchor occurs {len(positions)} times within locator; refine source span"
    if locator.block_index is not None:
        all_text = normalize_anchor("\n".join(s.content for s in document.segments))
        if len(_occurrences(all_text, anchor)) != 1:
            return "AMBIGUOUS_MAPPING", [], "block-local anchor repeats elsewhere; index lacks block offsets"
    eligible = chunks if locator.page_start is None else [c for c in chunks if c.page_start is not None
                   and c.page_start <= locator.page_end and c.page_end >= locator.page_start]
    complete = [[c.chunk_id] for c in eligible if anchor in normalize_anchor(c.content)]
    if complete:
        return "FULLY_MAPPED", sorted(complete), "anchor verified in source and complete chunk"
    loose = [c.chunk_id for c in eligible if _loose_hyphen(alternative.evidence_anchor) in _loose_hyphen(c.content)]
    if loose:
        return "ANCHOR_NORMALIZATION_FAILURE", [], "source anchor exists; decoded chunk lost hyphen line break; diagnostic chunks=" + ",".join(sorted(loose))
    # 只有能将 Chunk 连续正文唯一对齐回原文时，才证明跨块覆盖；不把词袋拼接当证据。
    full_text = normalize_anchor("\n".join(s.content for s in document.segments))
    full_positions = _occurrences(full_text, anchor)
    if len(full_positions) != 1:
        return "UNMAPPED", [], "source verified but full-document alignment is not unique"
    start, end = full_positions[0], full_positions[0] + len(anchor)
    intervals = []
    for chunk in eligible:
        content = normalize_anchor(chunk.content)
        offsets = _occurrences(full_text, content)
        if len(offsets) == 1 and offsets[0] < end and offsets[0] + len(content) > start:
            intervals.append((offsets[0], offsets[0] + len(content), chunk.chunk_id))
    cursor = start
    bundle = []
    while cursor < end:
        choices = [item for item in intervals if item[0] <= cursor and item[1] > cursor]
        if not choices:
            # 允许原文中仅空白的缝隙；不跨过任何缺失的实际字符。
            if cursor < len(full_text) and full_text[cursor].isspace():
                cursor += 1
                continue
            break
        chosen = max(choices, key=lambda item: (item[1], item[2]))
        cursor = chosen[1]
        bundle.append(chosen[2])
    if cursor >= end and bundle:
        return "FRAGMENTED_ACROSS_CHUNKS", [bundle], "union of source-aligned intervals covers complete anchor"
    if intervals:
        return "PARTIALLY_MAPPED", [], "source-aligned chunks cover only part of anchor"
    return "UNMAPPED", [], "source anchor verified; no exact chunk alignment; cause unresolved"


def map_dataset(dataset: EvaluationDatasetV2, documents: list[Document], chunks: list[DocumentChunk], *,
                index_version: str) -> MappingArtifact:
    """全量映射每题每组；替代证据成功即可满足 OR 合同。"""
    by_id = {doc.document_id: doc for doc in documents}
    if len(by_id) != len(documents) or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate source/chunk identity")
    rows = []
    for question in dataset.questions:
        for group in question.required_evidence_groups:
            outcomes = [_map_alternative(a, by_id.get(a.document_id), chunks) for a in group.alternatives]
            bundles = [bundle for _, sets, _ in outcomes for bundle in sets]
            if bundles:
                status = "FULLY_MAPPED" if any(len(bundle) == 1 for bundle in bundles) else "FRAGMENTED_ACROSS_CHUNKS"
            else:
                statuses = {status for status, _, _ in outcomes}
                status = next(iter(statuses)) if len(statuses) == 1 else "PARTIALLY_MAPPED"
            rows.append(GroupMapping(question_id=question.question_id, group_id=group.group_id,
                                     status=status, chunk_sets=bundles,
                                     reason="; ".join(f"alternative {i}: {status}: {reason}" for i, (status, _, reason) in enumerate(outcomes, 1))))
    return MappingArtifact(dataset_id=dataset.dataset_id, corpus_version=dataset.corpus_version,
                           index_version=index_version, groups=rows)
