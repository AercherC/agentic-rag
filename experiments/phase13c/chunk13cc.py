# -*- coding: utf-8 -*-
"""Phase 13C-C：PDF chunk benchmark 的四臂切分实现（方案 PLAN.md §3）。

输入为 13C-B B2_fused ScientificDocument（frozen_inputs/b2_fused/*.json），
四臂共享统一约束：tokenizer=BGE-M3、target=512、add_special_tokens=False 口径、
chunk_id 含 arm 策略名。provenance 以 element_ids 记录（anchor 投影与 C-G4 的基础）。

复用说明：``experiments/phase14a/chunk14c.py`` 的 ``make_chunk`` 与 14A-C arm 表耦合
（STRATEGY_BY_ARM 查表），不修改归档代码，故此处本地实现 13C-C 版 make_chunk；
``token_len`` / ``_split_oversized_unit`` 为无状态纯函数，直接 import 复用。
窗口打包本地实现以支持方案要求的 ``\\n\\n`` 段间分隔符（14A-C pack_units 为 ``\\n``）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 复用归档实验的无状态纯函数（import 不复制、不修改）。
from phase14a.chunk14c import token_len, _split_oversized_unit  # noqa: E402

# arm → (策略名, overlap token 数)；target 统一 512（PLAN.md §3）。
ARM_SPECS: dict[str, tuple[str, int]] = {
    "PC0": ("fixed_token", 64),
    "PC1": ("section_aware", 0),
    "PC2": ("element_aware", 0),
    "PC3": ("section_aware_overlap", 64),
    # 13E 混合臂：standalone（caption/table/formula 独立）+ 不同 body 聚合。
    "PH1": ("hybrid_section_o0", 0),
    "PH2": ("hybrid_section_o64", 64),
    "PH3": ("hybrid_fixed_o64", 64),
}
# standalone 元素类型：各成独立 chunk 并从 body 流移除（13C-C PC2 语义，不重复入文）。
STANDALONE_TYPES = {"table", "table_caption", "figure_caption", "equation"}
TOKEN_TARGET = 512
# 元素串接分隔符：对齐生产 chunker 的 segment 间约定（PLAN.md §3）。
SEG_SEP = "\n\n"
CHUNK_SCHEMA_VERSION = "phase13c-c-chunk-v1"


def make_chunk(
    *,
    document_id: str,
    arm: str,
    seq: int,
    text: str,
    element_ids: list[str],
    token_count: int,
    page_start: int | None,
    page_end: int | None,
    heading_path: str | None = None,
    split: bool = False,
) -> dict[str, Any]:
    """构造 13C-C 统一 chunk 记录（chunk_id 含 document_id+arm+seq，跨臂不碰撞）。"""

    chunk: dict[str, Any] = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_id": f"{document_id}::{arm}::{seq:04d}",
        "document_id": document_id,
        "arm": arm,
        "strategy": ARM_SPECS[arm][0],
        "seq": seq,
        "text": text,
        "element_ids": list(element_ids),
        "token_count": token_count,
        "page_start": page_start,
        "page_end": page_end,
        "split": bool(split),
    }
    if heading_path is not None:
        chunk["heading_path"] = heading_path
    return chunk


def element_text(el: dict[str, Any]) -> str:
    """元素入索引文本：table 用序列化 markdown，其余用 normalized/line_preserved 文本。"""

    if el.get("type") == "table":
        tbl = el.get("table") or {}
        text = tbl.get("serialized_markdown") or tbl.get("primary_fallback_text") or ""
        return text.strip()
    for key in ("normalized_text", "line_preserved_text", "raw_text"):
        val = el.get(key)
        if val and val.strip():
            return val.strip()
    return ""


def included_elements(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """入索引元素：B2_fused 契约的 retrieval_element_ids 成员且文本非空，按 reading_order 排序。"""

    ids = set(doc.get("retrieval_element_ids") or [])
    els = [el for el in doc.get("elements", []) if el.get("element_id") in ids]
    out = [(el, element_text(el)) for el in els]
    # 文本为空的元素（如无文本的 figure 载体）不入索引；figure 语义由其 caption 元素承担。
    out = [(el, t) for el, t in out if t]
    out.sort(key=lambda pair: pair[0].get("reading_order") or 0)
    return out


def _pages(els: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """owner 元素真实页码的 min/max（无页码时为 None，不推测）。"""

    pages = [el.get("page") for el in els if isinstance(el.get("page"), int)]
    return (min(pages), max(pages)) if pages else (None, None)


def _make_window(
    cur: list[tuple[str, list[str], dict[str, Any], bool]], sep: str
) -> dict[str, Any]:
    """由 unit 序列构造窗口：text=sep 拼接、owners 去重保序、split 任一为 True。"""

    text = sep.join(t for t, _, _, _ in cur)
    owners: list[str] = []
    for _, os_, _, _ in cur:
        for oid in os_:
            if oid not in owners:
                owners.append(oid)
    els = [el for _, _, el, _ in cur]
    return {"text": text, "element_ids": owners, "elements": els, "split": any(s for _, _, _, s in cur)}


def pack_units_sep(
    units: list[tuple[str, list[str], dict[str, Any]]],
    counter: Callable[[str], int],
    target: int,
    overlap: int,
    sep: str = SEG_SEP,
) -> list[dict[str, Any]]:
    """unit 流 → 窗口（本地实现，支持 sep 分隔符与精确"拼接后计数"）。

    - unit token > target → 词级拆分（复用 14A-C ``_split_oversized_unit``，词级重叠近似）；
    - 顺序贪心打包，窗间 unit 级重叠：上一窗尾部 unit（拼接后 token 和 ≤ overlap）带入下一窗；
    - overlap=0 时不携带尾部（section 边界零重叠语义）。
    """

    flat: list[tuple[str, list[str], dict[str, Any], bool]] = []
    for text, owners, el in units:
        if not text.strip():
            continue
        n = token_len(text, counter)
        if n > target:
            # 超长 unit 拆分为伪 unit（owners 复制；词级重叠为近似口径，记录 split=True）。
            for piece in _split_oversized_unit(text, counter, target, overlap):
                flat.append((piece, owners, el, True))
        else:
            flat.append((text, owners, el, False))

    windows: list[dict[str, Any]] = []
    cur: list[tuple[str, list[str], dict[str, Any], bool]] = []

    def _joined(items: list[tuple[str, list[str], dict[str, Any], bool]]) -> str:
        return sep.join(t for t, _, _, _ in items)

    for unit in flat:
        text, owners, el, was_split = unit
        if cur and token_len(_joined(cur) + sep + text, counter) > target:
            # 关闭当前窗；重叠尾部仅当 overlap>0 时计算。
            tail: list[tuple[str, list[str], dict[str, Any], bool]] = []
            if overlap > 0:
                for prev in reversed(cur):
                    cand = [prev] + tail
                    if token_len(_joined(cand), counter) > overlap:
                        break
                    tail.insert(0, prev)
            # C-G5 硬预算优先于重叠：尾部与下一 unit 合并若超 target 则放弃尾部。
            if tail and token_len(_joined(tail) + sep + text, counter) > target:
                tail = []
            windows.append(_make_window(cur, sep))
            cur = list(tail)
        cur.append(unit)
    if cur:
        windows.append(_make_window(cur, sep))
    return windows


def _section_key(el: dict[str, Any]) -> tuple[str, ...]:
    """section 分组键（section_path 列表 → 元组；缺失用空元组）。"""

    path = el.get("section_path")
    return tuple(path) if isinstance(path, list) else ()


def _split_standalone(
    pairs: list[tuple[dict[str, Any], str]],
    counter: Callable[[str], int],
    overlap: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, list[str], dict[str, Any]]]]:
    """13E 混合臂共用：standalone 类型元素独立成 chunk，其余返回 body unit 流。"""

    standalone: list[dict[str, Any]] = []
    body_units: list[tuple[str, list[str], dict[str, Any]]] = []
    for el, t in pairs:
        if el.get("type") in STANDALONE_TYPES:
            standalone.extend(
                pack_units_sep([(t, [el["element_id"]], el)], counter, TOKEN_TARGET, overlap))
        else:
            body_units.append((t, [el["element_id"]], el))
    return standalone, body_units


def _group_sections(
    body_units: list[tuple[str, list[str], dict[str, Any]]],
) -> list[list[tuple[str, list[str], dict[str, Any]]]]:
    """按连续 section_path 分组（组边界强制断开）。"""

    groups: list[list[tuple[str, list[str], dict[str, Any]]]] = []
    prev_key: tuple[str, ...] | None = None
    for unit in body_units:
        key = _section_key(unit[2])
        if prev_key is None or key != prev_key:
            groups.append([])
            prev_key = key
        groups[-1].append(unit)
    return groups


def chunk_pdf(doc: dict[str, Any], arm: str, counter: Callable[[str], int]) -> list[dict[str, Any]]:
    """按 arm 策略切分单个 B2_fused 文档（PLAN.md §3 的四种策略）。"""

    _, overlap = ARM_SPECS[arm]
    pairs = included_elements(doc)
    doc_id = doc["document_id"]
    raw: list[dict[str, Any]] = []

    if arm == "PC0":
        # 固定窗口：全文档 unit 流（reading_order），段间 \n\n，滑切 C512/O64。
        units = [(t, [el["element_id"]], el) for el, t in pairs]
        raw = pack_units_sep(units, counter, TOKEN_TARGET, overlap)
    elif arm in ("PC1", "PC3"):
        # section-aware：连续同 section_path 的元素分组，组边界强制断开；组内 pack。
        groups: list[list[tuple[str, list[str], dict[str, Any]]]] = []
        prev_key: tuple[str, ...] | None = None
        for el, t in pairs:
            key = _section_key(el)
            if prev_key is None or key != prev_key:
                groups.append([])
                prev_key = key
            groups[-1].append((t, [el["element_id"]], el))
        for group in groups:
            path = _section_key(group[0][2])
            heading = " > ".join(path) if path else None
            for win in pack_units_sep(group, counter, TOKEN_TARGET, overlap):
                win["heading_path"] = heading
                raw.append(win)
    elif arm == "PC2":
        # element-aware：table / caption / equation 独立成 chunk（超长词级拆分），
        # 其余（paragraph、section_header、reference_entry 等）作为 body unit 流打包。
        standalone, body_units = _split_standalone(pairs, counter, overlap)
        raw.extend(standalone)
        raw.extend(pack_units_sep(body_units, counter, TOKEN_TARGET, overlap))
    elif arm in ("PH1", "PH2", "PH3"):
        # 混合策略：standalone 独立 chunk + body 按 section 聚合（PH1/PH2）或
        # 固定窗口（PH3，归因臂）——分离"caption 独立"与"section 聚合"的贡献。
        standalone, body_units = _split_standalone(pairs, counter, overlap)
        raw.extend(standalone)
        if arm == "PH3":
            raw.extend(pack_units_sep(body_units, counter, TOKEN_TARGET, overlap))
        else:
            for group in _group_sections(body_units):
                path = _section_key(group[0][2])
                heading = " > ".join(path) if path else None
                for win in pack_units_sep(group, counter, TOKEN_TARGET, overlap):
                    win["heading_path"] = heading
                    raw.append(win)
    else:  # 防御：未知 arm 直接报错，避免静默产出错误切分。
        raise ValueError(f"unknown arm: {arm}")

    # 统一构造 chunk 记录（seq 从 0 连续；页码取 owner 元素真实 min/max）。
    chunks: list[dict[str, Any]] = []
    for seq, win in enumerate(raw):
        ps, pe = _pages(win["elements"])
        chunks.append(make_chunk(
            document_id=doc_id, arm=arm, seq=seq, text=win["text"],
            element_ids=win["element_ids"], token_count=token_len(win["text"], counter),
            page_start=ps, page_end=pe, heading_path=win.get("heading_path"),
            split=win["split"],
        ))
    return chunks
