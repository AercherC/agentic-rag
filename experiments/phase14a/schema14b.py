"""Phase 14A-B 统一 StructuredDocument v14B（方案 §11 / §12）。

与 14A-A schema14a 的差异（本阶段新增，不修改生产 schema）：
- element 增加 ``semantic_inferred``（启发式推断必须显式标记，禁止 silent promote）；
- 文档级增加 ``cleaning_log``（可审计清洗动作）与 ``unresolved``（未决项）；
- ``retrieval_role`` 收敛为固定词表：body | structure | reference | furniture | hidden | metadata；
- source_ref 校验复用 schema14a：DOCX/XLSX 禁止伪造 page/bbox。

本模块只依赖标准库。
"""

from __future__ import annotations

from typing import Any, Iterable

from . import schema14a, sandbox14a

SCHEMA_VERSION = "phase14a-b-structured-document-v1"

# 14B retrieval_role 固定词表（方案 §11）。
RETRIEVAL_ROLES = frozenset({"body", "structure", "reference", "furniture", "hidden", "metadata"})

# element 字段（方案 §11 Common Element）。
ELEMENT_FIELDS_B = (
    "document_id", "element_id", "type", "raw_text", "normalized_text",
    "parent_id", "children", "retrieval_role", "source_ref", "source_parser",
    "semantic_inferred", "metadata",
)

# DOCX type → retrieval_role 映射（B0/B1 共用，属策略而非清洗）。
DOCX_ROLE_BY_TYPE = {
    "document_title": "structure",
    "heading": "structure",
    "paragraph": "body",
    "list_item": "body",
    "table": "structure",
    "table_cell": "structure",
    "caption": "body",
    "footnote": "reference",
    "endnote": "reference",
    "header": "furniture",
    "footer": "furniture",
    "image": "metadata",
}

# XLSX type → retrieval_role 映射；hidden cell 由 cleaner 单独覆盖为 hidden。
XLSX_ROLE_BY_TYPE = {
    "workbook_metadata": "metadata",
    "sheet": "structure",
    "table": "structure",
    "table_header": "structure",
    "row_group": "structure",
    "cell": "body",
    "key_value": "body",
    "note": "body",
}


def make_element14b(
    *,
    document_id: str,
    element_id: str,
    type: str,
    raw_text: str,
    source_format: str,
    source_ref: dict[str, Any],
    source_parser: str,
    retrieval_role: str,
    parent_id: str | None = None,
    children: Iterable[str] | None = None,
    semantic_inferred: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 14B element；retrieval_role 必须在固定词表内。"""

    if retrieval_role not in RETRIEVAL_ROLES:
        raise ValueError(f"retrieval_role '{retrieval_role}' not in 14B vocabulary")
    element = schema14a.make_element(
        document_id=document_id, element_id=element_id, type=type, raw_text=raw_text,
        source_format=source_format, source_ref=source_ref, source_parser=source_parser,
        parent_id=parent_id, children=children, retrieval_role=retrieval_role, metadata=metadata,
    )
    element["semantic_inferred"] = bool(semantic_inferred)
    return element


def make_document14b(
    *,
    document_id: str,
    source_format: str,
    source_file: str,
    elements: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    cleaning_log: list[dict[str, Any]] | None = None,
    unresolved: list[dict[str, Any]] | None = None,
    document_metadata: dict[str, Any] | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """装配 14B StructuredDocument（方案 §11 顶层结构）。"""

    payload = schema14a.make_document(
        document_id=document_id, source_format=source_format, source_file=source_file,
        elements=elements, relationships=relationships, document_metadata=document_metadata,
        source_sha256=source_sha256,
    )
    payload["schema_version"] = SCHEMA_VERSION
    payload["cleaning_log"] = list(cleaning_log or [])
    payload["unresolved"] = list(unresolved or [])
    return payload


def log_action(
    log: list[dict[str, Any]], *, action: str, target: str, reason: str,
    from_type: str | None = None, to_type: str | None = None,
    source_element_id: str | None = None, detail: dict[str, Any] | None = None,
) -> None:
    """追加一条可审计 cleaning_log（方案 §15：每次删除/重分类/合并/拆分/推断都必须留痕）。"""

    entry: dict[str, Any] = {
        "action": action,
        "target": target,
        "reason": reason,
    }
    if from_type is not None:
        entry["from"] = from_type
    if to_type is not None:
        entry["to"] = to_type
    if source_element_id is not None:
        entry["source_element_id"] = source_element_id
    if detail:
        entry["detail"] = detail
    log.append(entry)


def add_unresolved(unresolved: list[dict[str, Any]], *, issue: str, target: str,
                   detail: dict[str, Any] | None = None) -> None:
    """记录未决项（如 cached value 缺失、chart 未解析视觉语义）。"""

    entry: dict[str, Any] = {"issue": issue, "target": target}
    if detail:
        entry["detail"] = detail
    unresolved.append(entry)


def validate_document14b(document: dict[str, Any]) -> dict[str, Any]:
    """校验 14B 文档：字段完整性、role 词表、source_ref 格式、cleaning_log 结构、id 唯一。"""

    problems: list[dict[str, Any]] = []
    fmt = document.get("source_format")
    elements = document.get("elements") or []
    seen: set[str] = set()
    for index, el in enumerate(elements):
        for field in ELEMENT_FIELDS_B:
            if field not in el:
                problems.append({"scope": f"element[{index}]", "issue": f"missing field: {field}"})
        if el.get("retrieval_role") not in RETRIEVAL_ROLES:
            problems.append({"scope": f"element[{index}]", "issue": f"bad retrieval_role {el.get('retrieval_role')}"})
        if not isinstance(el.get("semantic_inferred"), bool):
            problems.append({"scope": f"element[{index}]", "issue": "semantic_inferred must be bool"})
        eid = el.get("element_id")
        if eid in seen:
            problems.append({"scope": f"element[{index}]", "issue": f"duplicate element_id {eid}"})
        seen.add(eid)
        for issue in schema14a.validate_source_ref(el.get("source_ref") or {}, fmt):
            problems.append({"scope": f"element[{index}]", "issue": issue})
    for index, entry in enumerate(document.get("cleaning_log") or []):
        if not entry.get("action") or not entry.get("reason"):
            problems.append({"scope": f"cleaning_log[{index}]", "issue": "action/reason required"})
    return {
        "valid": not problems,
        "problems": problems,
        "element_count": len(elements),
        "cleaning_log_count": len(document.get("cleaning_log") or []),
    }


def digest_of(document: dict[str, Any]) -> str:
    """确定性 digest（同输入同 digest；D-G7 / X-G9）。"""

    return sandbox14a.sha256_json(document)
