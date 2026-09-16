"""Phase 15B 当前 active hybrid index 的只读 relevance projection。"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidenceflow.domain.chunk import DocumentChunk


def normalize_text(value: str) -> str:
    """按既有 Gold anchor 合同执行 NFKC、casefold 与空白折叠。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def project_active_relevance(
    chunks: Sequence[DocumentChunk],
    questions_v2: Mapping[str, Any],
) -> dict[str, Any]:
    """将冻结 evidence anchors 投影到 active Chunk，并返回稳定身份 Hash。"""

    chunks_by_document: dict[str, list[DocumentChunk]] = {}
    normalized_content = {
        chunk.chunk_id: normalize_text(chunk.content) for chunk in chunks
    }
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    groups_by_question: dict[str, list[dict[str, Any]]] = {}
    qrels: dict[str, list[str]] = {}
    for question in questions_v2["questions"]:
        question_id = question["question_id"]
        question_groups: list[dict[str, Any]] = []
        relevant_ids: set[str] = set()
        for group in question["required_evidence_groups"]:
            alternatives: list[dict[str, Any]] = []
            group_ids: set[str] = set()
            for alternative in group["alternatives"]:
                document_id = alternative["document_id"]
                anchor = alternative["evidence_anchor"]
                anchor_norm = normalize_text(anchor)
                matched = sorted(
                    chunk.chunk_id
                    for chunk in chunks_by_document.get(document_id, [])
                    if anchor_norm and anchor_norm in normalized_content[chunk.chunk_id]
                )
                group_ids.update(matched)
                alternatives.append(
                    {
                        "document_id": document_id,
                        "anchor": anchor,
                        "matched_chunk_ids": matched,
                        "indexable": bool(matched),
                    }
                )
            relevant_ids.update(group_ids)
            question_groups.append(
                {
                    "group_id": group["group_id"],
                    "alternatives": alternatives,
                    "matched_chunk_ids": sorted(group_ids),
                    "indexable": bool(group_ids),
                }
            )
        groups_by_question[question_id] = question_groups
        qrels[question_id] = sorted(relevant_ids)

    # Hash 只包含 active chunk 身份与 Gold 匹配结果，不包含时间戳或文件路径，
    # 使独立 replay 与 benchmark 能对同一 relevance projection 做字节级比对。
    hash_payload = {"qrels": qrels, "groups": groups_by_question}
    serialized = json.dumps(
        hash_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "groups_by_question": groups_by_question,
        "qrels": qrels,
        "active_relevance_projection_hash": hashlib.sha256(serialized).hexdigest(),
        "hash_serialization": "json.dumps(sort_keys=True,separators=(',', ':'))",
    }


def read_questions_v2(path: str | Path) -> dict[str, Any]:
    """读取冻结 v2 Gold 问题文件，供独立 replay 与 benchmark 共用。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))
