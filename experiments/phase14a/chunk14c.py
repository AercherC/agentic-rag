"""Phase 14A-C 切分模块：统一 chunk schema + 8 个切分 arm（方案 §1 重建执行版）。

重建说明：原 25 节方案在上下文压缩中丢失，本模块按压缩摘要 + 14A-B 惯例重建，
arm 定义采用 granularity 阶梯（报告与 status JSON 中标注 reconstructed）：

- DOCX（输入 = 冻结的 DOCX-B1 文档）：
  - D-C0 ``fixed_token``      ：全文档文本流，≤512 token 窗口，溢出重叠 64；无结构感知。
  - D-C1 ``heading_section``  ：按 heading/document_title 切 section，一节一 chunk（超长再切），
    chunk 前缀 heading path，表格 markdown 内联。
  - D-C2 ``element_atomic``   ：段落/列表/题注/脚注逐元素成 chunk（带 heading path 前缀），
    heading 归入其后首个正文 chunk；表格原子（超长按行拆）。
  - D-C3 ``heading_hybrid``   ：section 内正文打包 ≤512（多 chunk），表格独立原子 chunk，
    caption 紧邻 table 时并入 table chunk。
- XLSX（输入 = 冻结的 XLSX-B1 文档）：
  - X-C0 ``whole_sheet``      ：一 sheet 一 chunk（全部内容序列化，超长再切）。
  - X-C1 ``table_level``      ：一表一 chunk（header+rows），独立单元格/notes 一 chunk，kv 一 chunk。
  - X-C2 ``row_group_header8``：表内按 8 数据行一组切 chunk，每组重复 header 行。
  - X-C3 ``row_level_header`` ：每个数据行一个 chunk，重复 header 行。

统一约束（所有 arm 共享）：
- token 目标 512，超长才拆分，拆分重叠 64 token（方案冻结策略）；
- ``retrieval_role`` ∈ {furniture, hidden} 的元素不入索引（沉默索引 = 门失败）；
- 隐藏 sheet 的元素整体不入索引；XLSX 隐藏行/列 cell 不入索引；
- 每个 chunk 记录 ``element_ids``（provenance），anchor 投影与覆盖门依赖它；
- token counter 注入（生产用 bge-m3 tokenizer，测试用 fake），保持纯函数可测。
"""

from __future__ import annotations

import re
from typing import Any, Callable

# 冻结的 token 策略（方案 §13：target 512 / oversized split overlap 64）。
TOKEN_TARGET = 512
OVERLAP_TOKENS = 64

# 入索引的 retrieval_role 白名单（furniture/hidden/metadata 中除 image 外均不入索引）。
INDEXABLE_ROLES = frozenset({"body", "structure", "reference"})

CHUNK_SCHEMA_VERSION = "phase14a-c-chunk-v1"

# DOCX arm 名 → strategy 描述。
DOCX_ARMS = ("DC0", "DC1", "DC2", "DC3")
XLSX_ARMS = ("XC0", "XC1", "XC2", "XC3")
STRATEGY_BY_ARM = {
    "DC0": "fixed_token",
    "DC1": "heading_section",
    "DC2": "element_atomic",
    "DC3": "heading_hybrid",
    "XC0": "whole_sheet",
    "XC1": "table_level",
    "XC2": "row_group_header8",
    "XC3": "row_level_header",
}

# X-C2 的数据行窗口大小（每组 8 行，重复 header）。
ROW_GROUP_SIZE = 8

_COORD_RE = re.compile(r"^([A-Z]+)(\d+)$")


def column_index_of(coord: str) -> int:
    """Excel 列字母 → 1-based 列号（A=1）。用于 cell 排序与行列归属。"""

    m = _COORD_RE.match(coord)
    letters = m.group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def row_index_of(coord: str) -> int:
    """Excel 行号提取（A3 → 3）。"""

    return int(_COORD_RE.match(coord).group(2))


def make_chunk(
    *,
    document_id: str,
    arm: str,
    seq: int,
    text: str,
    element_ids: list[str],
    token_count: int,
    heading_path: str | None = None,
    split: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一 chunk 记录（provenance = element_ids，为 anchor 投影与覆盖门的基础）。"""

    chunk: dict[str, Any] = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_id": f"{document_id}::{arm}::{seq:04d}",
        "document_id": document_id,
        "arm": arm,
        "strategy": STRATEGY_BY_ARM[arm],
        "seq": seq,
        "text": text,
        "element_ids": list(element_ids),
        "token_count": token_count,
        "split": bool(split),
    }
    if heading_path is not None:
        chunk["heading_path"] = heading_path
    if extra:
        chunk["metadata"] = extra
    return chunk


# --------------------------------------------------------------------------------------
# 通用打包：unit = (text, owner_element_ids)；超长 unit 词级拆分（重叠 ~64 token），
# 窗口间 unit 级重叠（尾部 unit token 和 ≤ OVERLAP_TOKENS 时带入下一窗）。
# --------------------------------------------------------------------------------------

def token_len(text: str, counter: Callable[[str], int]) -> int:
    """token 计数（空文本计 0）。"""

    return counter(text) if text else 0


def _split_oversized_unit(
    text: str, counter: Callable[[str], int], target: int, overlap: int
) -> list[str]:
    """单个超长 unit 按词切分为 ≤target token 的窗口，相邻窗口重叠 ≈overlap token。

    重叠按词数近似：从上一窗尾部回退词直至 token 和 ≤ overlap（至少回退 0 词）。
    """

    words = text.split()
    if not words:
        return [text]
    windows: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for w in words:
        w_tokens = token_len(w + " ", counter)
        if cur and cur_tokens + w_tokens > target:
            windows.append(" ".join(cur))
            # 计算重叠尾部：从尾部回退词，token 和 ≤ overlap。
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(cur):
                t = token_len(prev + " ", counter)
                if tail and tail_tokens + t > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            cur = tail + [w]
            cur_tokens = token_len(" ".join(cur), counter)
        else:
            cur.append(w)
            cur_tokens += w_tokens
    if cur:
        windows.append(" ".join(cur))
    return windows


def pack_units(
    units: list[tuple[str, list[str]]],
    counter: Callable[[str], int],
    target: int = TOKEN_TARGET,
    overlap: int = OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """把 unit 流打包为窗口。

    返回窗口列表：{"text", "element_ids", "split"(bool 是否由超长拆分产生)}。
    - unit token > target → 先词级拆分为多个伪 unit（owners 复制到每个窗口，split=True）；
    - 顺序贪心打包，窗间重叠：上一窗尾部 unit（token 和 ≤ overlap）复制到下一窗开头；
    - 空文本 unit 跳过。
    """

    # 先展开超长 unit。
    flat: list[tuple[str, list[str], bool]] = []
    for text, owners in units:
        if not text.strip():
            continue
        n = token_len(text, counter)
        if n > target:
            for piece in _split_oversized_unit(text, counter, target, overlap):
                flat.append((piece, owners, True))
        else:
            flat.append((text, owners, False))

    windows: list[dict[str, Any]] = []
    cur: list[tuple[str, list[str], bool]] = []

    def _joined(items: list[tuple[str, list[str], bool]]) -> str:
        return "\n".join(t for t, _, _ in items)

    for unit in flat:
        text, owners, was_split = unit
        # 精确计 token：按"拼接后文本"计，避免分隔符（\n）开销导致超限。
        if cur and token_len(_joined(cur) + "\n" + text, counter) > target:
            # 关闭当前窗口；计算重叠尾部 unit（按拼接后 token 计）。
            tail: list[tuple[str, list[str], bool]] = []
            for prev in reversed(cur):
                cand = ([prev] + tail)
                if token_len(_joined(cand), counter) > overlap:
                    break
                tail.insert(0, prev)
            windows.append(_make_window(cur))
            cur = list(tail)
            # 尾部自身若已超 target（不可能，tail ≤ overlap）则防御性清空。
            if token_len(_joined(cur), counter) > target:
                cur = []
        cur.append(unit)
        # 单 unit 超长（flat 阶段应已拆分，这里兜底不再拆）。
    if cur:
        windows.append(_make_window(cur))
    return windows


def _make_window(cur: list[tuple[str, list[str], bool]]) -> dict[str, Any]:
    """由 unit 序列构造窗口（text = 行拼接；owners 去重保序；split 任一为 True）。"""

    text = "\n".join(t for t, _, _ in cur)
    owners: list[str] = []
    for _, os_, _ in cur:
        for oid in os_:
            if oid not in owners:
                owners.append(oid)
    return {"text": text, "element_ids": owners, "split": any(s for _, _, s in cur)}


# --------------------------------------------------------------------------------------
# DOCX 切分
# --------------------------------------------------------------------------------------

def _docx_included(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """入索引元素：role ∈ INDEXABLE_ROLES 且文本非空（table 用 metadata 序列化）。"""

    out = []
    for el in elements:
        if el.get("retrieval_role") not in INDEXABLE_ROLES:
            continue
        out.append(el)
    return out


def _docx_element_text(el: dict[str, Any]) -> str:
    """元素 → 文本行；table 用 serialized_markdown，缺失时从 metadata.cells 兜底
    （嵌套子表元素无 serialized_markdown，只有 cells 网格）。"""

    if el["type"] == "table":
        md = (el.get("metadata") or {}).get("serialized_markdown")
        if md:
            return md
        cells = (el.get("metadata") or {}).get("cells") or []
        if cells:
            # 简单网格序列化：首行为表头，其余为数据行（与表格 markdown 同风格）。
            lines = ["| " + " | ".join(str(x) for x in row) + " |" for row in cells]
            if len(lines) > 1:
                ncols = len(cells[0])
                lines.insert(1, "| " + " | ".join(["---"] * ncols) + " |")
            return "\n".join(lines)
        return ""
    return el.get("normalized_text") or ""


def _docx_heading_path_stack(el: dict[str, Any], stack: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """维护 heading 栈：document_title level=0，heading level≥1；返回新栈。"""

    t = el["type"]
    if t == "document_title":
        return [(0, el.get("normalized_text") or "")]
    if t == "heading":
        level = int((el.get("metadata") or {}).get("level") or 1)
        # 弹出所有 level ≥ 当前 的项（同级或更深）。
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, el.get("normalized_text") or ""))
    return stack


def _path_str(stack: list[tuple[int, str]]) -> str:
    return " > ".join(t for _, t in stack)


def chunk_docx(
    doc: dict[str, Any], arm: str, counter: Callable[[str], int]
) -> list[dict[str, Any]]:
    """DOCX 切分主入口（arm ∈ DC0..DC3）。"""

    els = _docx_included(doc["elements"])
    if arm == "DC0":
        return _chunk_docx_dc0(doc["document_id"], els, counter)
    if arm == "DC1":
        return _chunk_docx_dc1(doc["document_id"], els, counter)
    if arm == "DC2":
        return _chunk_docx_dc2(doc["document_id"], els, counter)
    if arm == "DC3":
        return _chunk_docx_dc3(doc["document_id"], els, counter)
    raise ValueError(f"unknown DOCX arm: {arm}")


def _emit(
    doc_id: str,
    arm: str,
    windows: list[dict[str, Any]],
    seq_start: int,
    heading_path: str | None,
    counter: Callable[[str], int],
) -> list[dict[str, Any]]:
    """窗口 → chunk 列表（token_count 取窗口文本真实计数）。"""

    out = []
    for i, w in enumerate(windows):
        out.append(
            make_chunk(
                document_id=doc_id,
                arm=arm,
                seq=seq_start + i,
                text=w["text"],
                element_ids=w["element_ids"],
                token_count=token_len(w["text"], counter),
                heading_path=heading_path,
                split=w["split"],
            )
        )
    return out


def _chunk_docx_dc0(
    doc_id: str, els: list[dict[str, Any]], counter: Callable[[str], int]
) -> list[dict[str, Any]]:
    """D-C0 fixed_token：无结构感知文本流。"""

    units = [(_docx_element_text(el), [el["element_id"]]) for el in els]
    return _emit(doc_id, "DC0", pack_units(units, counter), 0, None, counter)


def _chunk_docx_dc1(
    doc_id: str, els: list[dict[str, Any]], counter: Callable[[str], int]
) -> list[dict[str, Any]]:
    """D-C1 heading_section：一节一 chunk，prefix = heading path，表格内联，超长再切。"""

    chunks: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    section: list[dict[str, Any]] = []  # 当前 section 的元素
    seq = 0

    def flush() -> None:
        nonlocal seq
        if not section:
            return
        prefix = f"[{_path_str(stack)}]\n" if stack else ""
        units = [(_docx_element_text(el), [el["element_id"]]) for el in section]
        windows = pack_units(units, counter)
        for i, w in enumerate(windows):
            text = (prefix + w["text"]) if i == 0 or prefix else w["text"]
            if i > 0 and prefix:
                text = prefix + w["text"]
            chunks.append(
                make_chunk(
                    document_id=doc_id, arm="DC1", seq=seq, text=text,
                    element_ids=w["element_ids"],
                    token_count=token_len(text, counter),
                    heading_path=_path_str(stack), split=w["split"],
                )
            )
            seq += 1
        section.clear()

    for el in els:
        if el["type"] in ("document_title", "heading"):
            flush()
            stack = _docx_heading_path_stack(el, stack)
            section.append(el)
        else:
            section.append(el)
    flush()
    return chunks


def _chunk_docx_dc2(
    doc_id: str, els: list[dict[str, Any]], counter: Callable[[str], int]
) -> list[dict[str, Any]]:
    """D-C2 element_atomic：正文逐元素成 chunk（heading path 前缀），heading 挂到其后首个
    chunk；表格原子 chunk（超长按行拆）。"""

    chunks: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    pending_headings: list[str] = []  # 等待挂载的 heading element_id
    seq = 0

    def add_chunk(text: str, owners: list[str], split: bool, is_table: bool) -> None:
        nonlocal seq, pending_headings
        prefix = f"[{_path_str(stack)}]\n" if stack else ""
        full = prefix + text if prefix else text
        owners_full = list(pending_headings) + owners
        chunks.append(
            make_chunk(
                document_id=doc_id, arm="DC2", seq=seq, text=full,
                element_ids=owners_full, token_count=token_len(full, counter),
                heading_path=_path_str(stack), split=split,
                extra={"atomic_table": is_table},
            )
        )
        seq += 1
        pending_headings = []

    for el in els:
        if el["type"] in ("document_title", "heading"):
            stack = _docx_heading_path_stack(el, stack)
            pending_headings.append(el["element_id"])
        elif el["type"] == "table":
            md = _docx_element_text(el)
            # 表格原子：超长时按 markdown 行拆，每窗重复 prefix，owners 全表。
            if token_len(md, counter) > TOKEN_TARGET:
                lines = md.split("\n")
                units = [(ln, [el["element_id"]]) for ln in lines]
                windows = pack_units(units, counter)
                for w in windows:
                    add_chunk(w["text"], w["element_ids"], True, True)
            else:
                add_chunk(md, [el["element_id"]], False, True)
        else:
            text = _docx_element_text(el)
            if not text.strip():
                # 空文本元素（如空段落）也保证 heading 有挂载点，避免 heading 悬空。
                if pending_headings:
                    add_chunk(text, [el["element_id"]], False, False)
                continue
            add_chunk(text, [el["element_id"]], False, False)
    # 收尾：未挂载的 heading 自成 chunk。
    if pending_headings:
        prefix = f"[{_path_str(stack)}]\n" if stack else ""
        chunks.append(
            make_chunk(
                document_id=doc_id, arm="DC2", seq=seq, text=prefix.rstrip("\n"),
                element_ids=list(pending_headings),
                token_count=token_len(prefix, counter),
                heading_path=_path_str(stack), split=False,
            )
        )
    return chunks


def _chunk_docx_dc3(
    doc_id: str, els: list[dict[str, Any]], counter: Callable[[str], int]
) -> list[dict[str, Any]]:
    """D-C3 heading_hybrid：section 内正文打包，表格独立原子 chunk；
    caption 紧邻 table（前一个或后一个元素是 table）时并入 table chunk。"""

    chunks: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    seq = 0
    n = len(els)
    # 先标记 caption 归属：紧邻 table 的 caption 并入 table chunk。
    caption_owner: dict[int, int] = {}  # caption index → table index
    for i, el in enumerate(els):
        if el["type"] == "caption":
            if i + 1 < n and els[i + 1]["type"] == "table":
                caption_owner[i] = i + 1
            elif i > 0 and els[i - 1]["type"] == "table":
                caption_owner[i] = i - 1

    section_units: list[tuple[str, list[str]]] = []

    def flush_section() -> None:
        nonlocal seq
        if not section_units:
            return
        prefix = f"[{_path_str(stack)}]\n" if stack else ""
        windows = pack_units(section_units, counter)
        for w in windows:
            text = prefix + w["text"] if prefix else w["text"]
            chunks.append(
                make_chunk(
                    document_id=doc_id, arm="DC3", seq=seq, text=text,
                    element_ids=w["element_ids"], token_count=token_len(text, counter),
                    heading_path=_path_str(stack), split=w["split"],
                )
            )
            seq += 1
        section_units.clear()

    for i, el in enumerate(els):
        if el["type"] in ("document_title", "heading"):
            flush_section()
            stack = _docx_heading_path_stack(el, stack)
            # heading 归入 section 首个 chunk：作为 owner 放进 section units。
            section_units.append((el.get("normalized_text") or "", [el["element_id"]]))
        elif el["type"] == "table":
            flush_section()
            md = _docx_element_text(el)
            owners = [el["element_id"]]
            # 紧邻 caption 并入。
            for ci, ti in caption_owner.items():
                if ti == i:
                    owners.append(els[ci]["element_id"])
                    cap_text = els[ci].get("normalized_text") or ""
                    md = md + "\n" + cap_text if cap_text else md
            prefix = f"[{_path_str(stack)}]\n" if stack else ""
            if token_len(prefix + md, counter) > TOKEN_TARGET:
                units = [(ln, owners) for ln in md.split("\n")]
                windows = pack_units(units, counter)
                for w in windows:
                    text = prefix + w["text"] if prefix else w["text"]
                    chunks.append(
                        make_chunk(
                            document_id=doc_id, arm="DC3", seq=seq, text=text,
                            element_ids=w["element_ids"],
                            token_count=token_len(text, counter),
                            heading_path=_path_str(stack), split=True,
                            extra={"atomic_table": True},
                        )
                    )
                    seq += 1
            else:
                text = prefix + md if prefix else md
                chunks.append(
                    make_chunk(
                        document_id=doc_id, arm="DC3", seq=seq, text=text,
                        element_ids=list(owners), token_count=token_len(text, counter),
                        heading_path=_path_str(stack), split=False,
                        extra={"atomic_table": True},
                    )
                )
                seq += 1
        elif i in caption_owner:
            continue  # 已并入 table chunk
        else:
            text = _docx_element_text(el)
            if text.strip():
                section_units.append((text, [el["element_id"]]))
    flush_section()
    return chunks


# --------------------------------------------------------------------------------------
# XLSX 切分
# --------------------------------------------------------------------------------------

def build_xlsx_view(doc: dict[str, Any]) -> dict[str, Any]:
    """把 XLSX-B1 文档解析为切分视图（sheets/tables/standalone/kv），隐藏内容整棵剔除。

    - cell → table 归属：source_ref.(sheet, range) 与 table element 一致；
    - header 行数 = max(table_header.metadata.hierarchy 长度)（层级数即表头行数）；
    - 隐藏 sheet（metadata.hidden / state != visible）的全部元素剔除；
    - cell 的 is_hidden_row / is_hidden_col 剔除。
    """

    elements = doc["elements"]
    # sheet 可见性。
    sheet_hidden: dict[str, bool] = {}
    sheet_els: dict[str, dict[str, Any]] = {}
    for el in elements:
        if el["type"] == "sheet":
            md = el.get("metadata") or {}
            name = md.get("name") or (el.get("source_ref") or {}).get("sheet")
            hidden = bool(md.get("hidden")) or md.get("state") != "visible"
            sheet_hidden[name] = hidden
            sheet_els[name] = el

    def cell_hidden(el: dict[str, Any]) -> bool:
        md = el.get("metadata") or {}
        sheet = (el.get("source_ref") or {}).get("sheet")
        if sheet_hidden.get(sheet, False):
            return True
        return bool(md.get("is_hidden_row")) or bool(md.get("is_hidden_col"))

    tables: list[dict[str, Any]] = []
    standalone: dict[str, list[dict[str, Any]]] = {}
    kv_by_sheet: dict[str, list[dict[str, Any]]] = {}
    claimed: set[str] = set()  # 归属表的 cell element_id

    table_els = [el for el in elements if el["type"] == "table"]
    header_els = [el for el in elements if el["type"] == "table_header"]
    cell_els = [el for el in elements if el["type"] in ("cell", "key_value", "note")]

    for tbl in table_els:
        ref = tbl.get("source_ref") or {}
        sheet, rng = ref.get("sheet"), ref.get("range")
        if sheet_hidden.get(sheet, False):
            continue
        md = tbl.get("metadata") or {}
        header_count = 1
        for h in md.get("header_hierarchies") or []:
            header_count = max(header_count, len(h))
        # 通过 header 元素 hierarchy 长度推断表头行数。
        for h in header_els:
            href = h.get("source_ref") or {}
            if href.get("sheet") == sheet and href.get("range") == rng:
                header_count = max(header_count, len((h.get("metadata") or {}).get("hierarchy") or [1]))
        members = [
            c for c in cell_els
            if c.get("source_ref", {}).get("sheet") == sheet
            and c.get("source_ref", {}).get("range") == rng
            and not cell_hidden(c)
        ]
        headers: dict[int, dict[str, Any]] = {}
        for h in header_els:
            href = h.get("source_ref") or {}
            if href.get("sheet") == sheet and href.get("range") == rng:
                headers[int((h.get("metadata") or {}).get("column_index") or 0)] = h
        # 行分组：member cell 按 (row, col) 排序；表头行的 cell（与 header 同文本/行）
        # 由 header_count 判定。
        data_rows: dict[int, list[dict[str, Any]]] = {}
        for c in members:
            coord = c.get("source_ref", {}).get("cell")
            if not coord:
                continue
            r = row_index_of(coord)
            if r <= header_count:
                continue
            data_rows.setdefault(r, []).append(c)
        for r in data_rows:
            data_rows[r].sort(key=lambda c: column_index_of(c["source_ref"]["cell"]))
        # 表头行 cell 也 claimed（不进 standalone）。
        for c in members:
            coord = c.get("source_ref", {}).get("cell")
            if coord and row_index_of(coord) <= header_count:
                claimed.add(c["element_id"])
            elif coord:
                claimed.add(c["element_id"])
        tables.append(
            {
                "table_el": tbl,
                "sheet": sheet,
                "range": rng,
                "name": md.get("table_name") or rng,
                "named": bool(md.get("named")),
                "inferred": bool(md.get("inferred")),
                "headers": dict(sorted(headers.items())),
                "header_cells": [
                    c for c in members
                    if c.get("source_ref", {}).get("cell")
                    and row_index_of(c["source_ref"]["cell"]) <= header_count
                ],
                "data_rows": [data_rows[r] for r in sorted(data_rows)],
            }
        )

    for c in cell_els:
        if c["element_id"] in claimed:
            continue
        ref = c.get("source_ref") or {}
        sheet = ref.get("sheet")
        if sheet is None or sheet_hidden.get(sheet, False):
            continue
        if cell_hidden(c):
            continue
        if c["type"] == "key_value":
            kv_by_sheet.setdefault(sheet, []).append(c)
        else:
            standalone.setdefault(sheet, []).append(c)

    return {
        "sheet_hidden": sheet_hidden,
        "sheet_els": sheet_els,
        "tables": tables,
        "standalone": standalone,
        "kv_by_sheet": kv_by_sheet,
    }


def _xlsx_header_line(tbl: dict[str, Any]) -> tuple[str, list[str]]:
    """表的 header 行（含表名/sheet/层级展开）。

    owners 含 table 元素、table_header 元素与表头行的 cell 元素
    （header 行 cell 是 B1 的真实元素，必须被覆盖且可作 anchor 投影）。
    """

    parts = []
    for _col, h in sorted(tbl["headers"].items()):
        hier = (h.get("metadata") or {}).get("hierarchy") or [h.get("normalized_text") or ""]
        parts.append("/".join(str(x) for x in hier if x != ""))
    line = f"Table {tbl['name']} @ {tbl['sheet']} | Header: " + " | ".join(parts)
    owners = [tbl["table_el"]["element_id"]] + [
        h["element_id"] for _c, h in sorted(tbl["headers"].items())
    ] + [c["element_id"] for c in tbl.get("header_cells", [])]
    return line, owners


def _xlsx_row_line(tbl: dict[str, Any], row_cells: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """数据行序列化：``Row {首列值}: v1 | v2 | ...``；owners = 行内 cell element_ids。"""

    values = [c.get("normalized_text") or "" for c in row_cells]
    label = values[0] if values else ""
    line = f"Row {label}: " + " | ".join(values)
    return line, [c["element_id"] for c in row_cells]


def _xlsx_standalone_lines(sheet: str, cells: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """独立单元格/notes 序列化：``sheet!cell: text``。"""

    out = []
    for c in cells:
        coord = (c.get("source_ref") or {}).get("cell") or "?"
        out.append((f"{sheet}!{coord}: {c.get('normalized_text') or ''}", [c["element_id"]]))
    return out


def _xlsx_kv_lines(sheet: str, kvs: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """kv 对序列化（key_value 元素已配对，逐元素一行）。"""

    out = []
    for c in kvs:
        out.append((f"{sheet} kv: {c.get('normalized_text') or ''}", [c["element_id"]]))
    return out


def attach_sheet_elements(chunks: list[dict[str, Any]], doc: dict[str, Any], view: dict[str, Any]) -> None:
    """把未覆盖的可见 sheet 元素附加到该 sheet 内容的首个 chunk（就地修改）。

    背景：XC1/2/3 的表 chunk 不一定引用 sheet 元素，导致 sheet 元素覆盖缺失、
    以 sheet 为 anchor 的查询不可投影。XC0 已自带 sheet 行，无需处理。
    """

    covered = {eid for c in chunks for eid in c["element_ids"]}
    # 元素 → sheet 映射（按 source_ref.sheet）。
    el_sheet: dict[str, str] = {}
    for el in doc["elements"]:
        s = (el.get("source_ref") or {}).get("sheet")
        if s:
            el_sheet[el["element_id"]] = s
    for sheet, sheet_el in view["sheet_els"].items():
        if view["sheet_hidden"].get(sheet, False):
            continue
        sid = sheet_el["element_id"]
        if sid in covered:
            continue
        for c in chunks:
            if any(el_sheet.get(eid) == sheet for eid in c["element_ids"]):
                c["element_ids"].append(sid)
                break


def chunk_xlsx(
    doc: dict[str, Any], arm: str, counter: Callable[[str], int]
) -> list[dict[str, Any]]:
    """XLSX 切分主入口（arm ∈ XC0..XC3）。"""

    view = build_xlsx_view(doc)
    doc_id = doc["document_id"]
    if arm == "XC0":
        return _chunk_xlsx_xc0(doc, view, counter)
    if arm == "XC1":
        return _chunk_xlsx_xc1(doc, view, counter)
    if arm == "XC2":
        return _chunk_xlsx_xc2(doc, view, counter)
    if arm == "XC3":
        return _chunk_xlsx_xc3(doc, view, counter)
    raise ValueError(f"unknown XLSX arm: {arm}")


def _chunk_xlsx_xc0(
    doc: dict[str, Any], view: dict[str, Any], counter: Callable[[str], int]
) -> list[dict[str, Any]]:

    doc_id = doc["document_id"]
    """X-C0 whole_sheet：一 sheet 一 chunk（超长再切，unit 级重叠）。"""

    chunks: list[dict[str, Any]] = []
    seq = 0
    for sheet, sheet_el in view["sheet_els"].items():
        if view["sheet_hidden"].get(sheet, False):
            continue
        units: list[tuple[str, list[str]]] = [(f"Sheet {sheet}", [sheet_el["element_id"]])]
        for tbl in view["tables"]:
            if tbl["sheet"] != sheet:
                continue
            hl, ho = _xlsx_header_line(tbl)
            units.append((hl, ho))
            for row in tbl["data_rows"]:
                units.append(_xlsx_row_line(tbl, row))
        units += _xlsx_standalone_lines(sheet, view["standalone"].get(sheet, []))
        units += _xlsx_kv_lines(sheet, view["kv_by_sheet"].get(sheet, []))
        for w in pack_units(units, counter):
            chunks.append(
                make_chunk(
                    document_id=doc_id, arm="XC0", seq=seq, text=w["text"],
                    element_ids=w["element_ids"], token_count=token_len(w["text"], counter),
                    split=w["split"], extra={"sheet": sheet},
                )
            )
            seq += 1
    return chunks


def _non_table_chunks(
    doc_id: str, arm: str, view: dict[str, Any], seq: int,
    counter: Callable[[str], int],
) -> tuple[list[dict[str, Any]], int]:
    """非表内容：独立单元格/notes 一 chunk，kv 一 chunk（按 sheet 分组）。"""

    out: list[dict[str, Any]] = []
    for sheet in view["sheet_els"]:
        if view["sheet_hidden"].get(sheet, False):
            continue
        sa = view["standalone"].get(sheet, [])
        if sa:
            units = [(f"Sheet {sheet}", [view["sheet_els"][sheet]["element_id"]])]
            units += _xlsx_standalone_lines(sheet, sa)
            for w in pack_units(units, counter):
                out.append(
                    make_chunk(
                        document_id=doc_id, arm=arm, seq=seq, text=w["text"],
                        element_ids=w["element_ids"],
                        token_count=token_len(w["text"], counter), split=w["split"],
                        extra={"sheet": sheet, "content": "standalone"},
                    )
                )
                seq += 1
        kvs = view["kv_by_sheet"].get(sheet, [])
        if kvs:
            units = _xlsx_kv_lines(sheet, kvs)
            for w in pack_units(units, counter):
                out.append(
                    make_chunk(
                        document_id=doc_id, arm=arm, seq=seq, text=w["text"],
                        element_ids=w["element_ids"],
                        token_count=token_len(w["text"], counter), split=w["split"],
                        extra={"sheet": sheet, "content": "kv"},
                    )
                )
                seq += 1
    return out, seq


def _table_chunks(
    doc_id: str, arm: str, tbl: dict[str, Any], seq: int,
    counter: Callable[[str], int], rows_per_chunk: int | None,
) -> list[dict[str, Any]]:
    """表 chunk 生成。

    - rows_per_chunk=None：整表一 chunk，超长按行拆（重复 header 行）；
    - rows_per_chunk=8：固定行窗口（X-C2）；
    - rows_per_chunk=1：每行一 chunk（X-C3）；所有 chunk 均重复 header 行。
    """

    hl, ho = _xlsx_header_line(tbl)
    rows = tbl["data_rows"]
    out: list[dict[str, Any]] = []
    if rows_per_chunk is None:
        units = [(hl, ho)] + [_xlsx_row_line(tbl, r) for r in rows]
        for w in pack_units(units, counter):
            out.append(
                make_chunk(
                    document_id=doc_id, arm=arm, seq=seq, text=w["text"],
                    element_ids=w["element_ids"],
                    token_count=token_len(w["text"], counter), split=w["split"],
                    extra={"sheet": tbl["sheet"], "table": tbl["name"]},
                )
            )
            seq += 1
        return out

    if not rows:
        text = hl
        out.append(
            make_chunk(
                document_id=doc_id, arm=arm, seq=seq, text=text,
                element_ids=list(ho), token_count=token_len(text, counter),
                split=False, extra={"sheet": tbl["sheet"], "table": tbl["name"]},
            )
        )
        return out

    for i in range(0, len(rows), rows_per_chunk):
        group = rows[i : i + rows_per_chunk]
        units = [(hl, ho)] + [_xlsx_row_line(tbl, r) for r in group]
        # 行窗口很小不会超长，直接拼接（仍走 pack 以防极端宽表）。
        for w in pack_units(units, counter):
            out.append(
                make_chunk(
                    document_id=doc_id, arm=arm, seq=seq, text=w["text"],
                    element_ids=w["element_ids"],
                    token_count=token_len(w["text"], counter), split=w["split"],
                    extra={"sheet": tbl["sheet"], "table": tbl["name"],
                           "row_window": [i + 1, i + len(group)]},
                )
            )
            seq += 1
    return out


def _chunk_xlsx_xc1(
    doc: dict[str, Any], view: dict[str, Any], counter: Callable[[str], int]
) -> list[dict[str, Any]]:

    doc_id = doc["document_id"]
    """X-C1 table_level：一表一 chunk；独立单元格/notes、kv 各自成 chunk。"""

    chunks: list[dict[str, Any]] = []
    seq = 0
    for tbl in view["tables"]:
        cs = _table_chunks(doc_id, "XC1", tbl, seq, counter, rows_per_chunk=None)
        chunks += cs
        seq += len(cs)
    non_table, seq2 = _non_table_chunks(doc_id, "XC1", view, seq, counter)
    chunks += non_table
    attach_sheet_elements(chunks, doc, view)
    return chunks


def _chunk_xlsx_xc2(
    doc: dict[str, Any], view: dict[str, Any], counter: Callable[[str], int]
) -> list[dict[str, Any]]:

    doc_id = doc["document_id"]
    """X-C2 row_group_header8：每 8 数据行一组，每组重复 header。"""

    chunks: list[dict[str, Any]] = []
    seq = 0
    for tbl in view["tables"]:
        cs = _table_chunks(doc_id, "XC2", tbl, seq, counter, rows_per_chunk=ROW_GROUP_SIZE)
        chunks += cs
        seq += len(cs)
    non_table, _ = _non_table_chunks(doc_id, "XC2", view, seq, counter)
    chunks += non_table
    attach_sheet_elements(chunks, doc, view)
    return chunks


def _chunk_xlsx_xc3(
    doc: dict[str, Any], view: dict[str, Any], counter: Callable[[str], int]
) -> list[dict[str, Any]]:

    doc_id = doc["document_id"]
    """X-C3 row_level_header：每个数据行一个 chunk，重复 header。"""

    chunks: list[dict[str, Any]] = []
    seq = 0
    for tbl in view["tables"]:
        cs = _table_chunks(doc_id, "XC3", tbl, seq, counter, rows_per_chunk=1)
        chunks += cs
        seq += len(cs)
    non_table, _ = _non_table_chunks(doc_id, "XC3", view, seq, counter)
    chunks += non_table
    attach_sheet_elements(chunks, doc, view)
    return chunks
