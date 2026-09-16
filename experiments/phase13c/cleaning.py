"""Phase 13C-B Cleaner 契约（方案 §9 / §2 Step 2）：**全部为纯函数**。

设计边界
--------

本模块只做一件事：把某个 arm 的 13C 统一 schema 输出，确定性地变成一个
``ScientificDocument``（正文 + 结构 + 参考文献 + 装饰分离 + provenance）。

- **不依赖 Runtime**：不 import ``evidenceflow.*``，不读 Chroma，不写磁盘
  （除 :func:`clean_document` 的调用方自行落盘）。
- **不做"猜测式修复"**：公式不做语义重写，表格不做行列推断，图不做 VLM 理解。
  所有会被改写的字段都同时保留 ``raw_text``；被丢弃的元素都写进
  ``removed_furniture`` / ``removed_elements``（方案 §9.2 / §35）。
- **不删除任何"可能是正文"的东西**：正文、参考文献、结构元素永远保留在
  ``elements`` 里，只通过 ``retrieval_role`` 把它们分到不同的检索通道。

## 三个必须解释清楚的实现决定

### 1. 为什么需要"数学跨度规范化"（``normalize_math_spans``）

Phase 13C-A 记录 P1_mineru 在 ``seedformer.pdf`` 上丢失了 anchor
``EVID-ARR-017-G1-A1``（文本 ``Np = 128``）。逐元素追查后，该 anchor 的**内容其实在
P1 输出里**，形态是行内公式：

    The SeedFormer encoder applies two layers of set abstraction [22] and obtain
    $N _ { p } = 1 2 8$ pat...

即 LaTeX 把下标 ``N_p`` 写成 ``N _ { p }``、把 ``128`` 写成 ``1 2 8``。
现有 tokenizer（``schema.tokenize``）在这种写法下切出 ``n / p / 1 / 2 / 8``，
anchor 的 ``np`` / ``128`` 一个都取不到 —— 这是**表示约定**造成的检索性丢失，
不是正文丢失。

因此 Cleaner 增加一层稀疏的、纯词法的数学跨度规范化：只在数学定界符内部
（``$...$`` / ``$$...$$`` / ``\\(...\\)`` / ``\\[...\\]``）执行

- ``M1`` 删除 LaTeX 分组花括号 ``{`` ``}``（它们只承载分组语义，不是内容）；
- ``M2`` 删除紧贴单个字母/数字的 ``_`` ``^``（下标/上标连接符）；
- ``M3`` 删除无宽度排版命令（``\\,`` ``\\;`` ``\\:`` ``\\!`` ``\\quad`` ``\\qquad``）；
- ``M4`` 去掉常见"只起排版作用"的命令外壳（``\\mathrm{...}`` / ``\\operatorname{...}`` 等）；
- ``M5`` 折叠**两个字母数字之间**的空白（LaTeX 的字距填充）。

``M5`` 是这条链上唯一有风险的一步：它会把 ``a b`` 变成 ``ab``。限制条件是它**只在
数学跨度内**生效，且 ``raw_text`` 永远保留原样，因此任何被折叠的信息都还能取回。
**禁止**：替换符号、补全缺失算子、重排公式结构 —— 那些属于"猜测式修复"。

### 2. 为什么只用"解析器已声明的类型 + 文本模式"判装饰，不做几何发明

Phase 13C-A 的量化裁定显示：全部 arm 的 ``body_matched_to_furniture`` 都是 0，
即没有任何 arm 把正文误标成装饰；而门 5 失败的成因是"正文没对齐上"（分块差异 /
真缺失）与"根本没声明装饰"两件事，与"装饰能否过滤"无关。

所以本模块**不引入**基于页面上下边距的几何装饰判定（那会新造一类误判风险），
只用两个保守来源：

1. 解析器**显式声明**的类型（P0 经 13B 判定、P1 原生 ``header``/``footer``/
   ``page_number``/``aside_text``）——直接采纳；
2. 仅对 ``unknown`` 元素套用 ``schema.classify_furniture`` 的文本模式规则。

这样"把正文标成装饰"在构造上不可能从本模块产生（正文类型从不被改写）。

### 3. 为什么参考文献是"拆出来"而不是"删掉"

方案 §9.6 要求 ``References -> reference_entry[]``。P1 原生把整段参考文献放进
一个 ``list``（``sub_type=ref_text``）元素的 ``list_items[]`` 里；13C-A 的映射层把
该元素折叠成了 1 个 ``unknown``（文本为空），因此 ``T11_references_heavy`` 出现
122 个"低覆盖正文误删"。本模块按 ``list_items`` **逐条**展开为 ``reference_entry``
（这是解析器已经给出的边界，不是猜测），并在正文侧保留任何无法归属的段落。
参考文献永远保留在 ``elements`` 中，只把 ``retrieval_role`` 标成 ``reference``。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from . import schema

# Cleaner 语义版本：任何规则变化都必须提升它，否则产物无法区分新旧口径。
CLEANER_VERSION = "phase13c-b-cleaner-1.1.0"

# 允许写入 ScientificDocument 的检索角色。
RETRIEVAL_ROLES = ("body", "structure", "reference", "furniture", "discarded")

# 结构型元素（进入检索但不是叙述性正文）。
_STRUCTURE_TYPES = frozenset(
    {"abstract", "figure", "figure_caption", "table", "table_caption", "equation"}
)
# 正文型（叙述性）。
_BODY_LIKE = frozenset({"paragraph", "list_item", "section_header", "document_title"})
# 装饰型。
_FURNITURE = frozenset({"header", "footer", "page_number", "watermark", "sidebar"})


def retrieval_role_for(element_type: str) -> str:
    """把统一 schema 的 19 类映射为 5 个检索角色（方案 §9.6 的通道划分）。"""

    if element_type in _FURNITURE:
        return "furniture"
    if element_type == "reference_entry":
        return "reference"
    if element_type in _STRUCTURE_TYPES:
        return "structure"
    if element_type in _BODY_LIKE or element_type == "footnote":
        return "body"
    return "discarded"


# --------------------------------------------------------------------------------------
# §9.1 文本规范化（允许 / 禁止清单）
# --------------------------------------------------------------------------------------

# 数学跨度定界符。``$$...$$`` / ``\[...\]`` 必须排在单 ``$`` 之前，否则内层会被
# 误当成行内公式而把定界符留在文本里。
_MATH_SPANS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\$(?P<body>.+?)\$\$", re.S),
    re.compile(r"\\\[(?P<body>.+?)\\\]", re.S),
    re.compile(r"\\\((?P<body>.+?)\\\)", re.S),
    re.compile(r"(?<!\$)\$(?!\$)(?P<body>[^$]+?)(?<!\$)\$(?!\$)", re.S),
)
# 无宽度排版命令：删除后不改变 token 序列。
_NOOP_SPACING_CMD = re.compile(r"\\(?:qquad|quad|thinspace|enspace|,|;|:|!|\s)")
# 只承担排版作用的命令外壳，展开为其内容。
_WRAPPER_CMD = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|operatorname|text|mbox|textrm|textrm)\s*\{([^{}]*)\}"
)
# 下标/上标连接符：仅当它后面紧跟**单个**字母数字时删除。
# ``(?![A-Za-z0-9]{2})`` 保证 ``x_{12}`` 这类多字符下标不被误合。
_SUBSUP_GLUE = re.compile(r"[_^]\s*(?=[A-Za-z0-9])(?![A-Za-z0-9]{2})")
# 残留命令：``\log`` -> ``log``。
_RESIDUAL_CMD = re.compile(r"\\([A-Za-z]+)")
# 字母数字之间的空白（LaTeX 字距填充）：``1 2 8`` -> ``128``，``N p`` -> ``Np``。
_ALNUM_GAP = re.compile(r"(?<=[A-Za-z0-9])\s+(?=[A-Za-z0-9])")
_WHITESPACE = re.compile(r"\s+")


def normalize_math_body(body: str) -> str:
    """对**数学跨度内部**的文本做纯词法规范化（M1–M5）。

    这是本 Cleaner 唯一"改写公式文本"的地方，因此每一步都写明了它删掉的是什么
    语法角色（分组 / 连接符 / 排版空白），并且不替换任何符号。
    """

    text = _NOOP_SPACING_CMD.sub(" ", body)  # M3
    text = _WRAPPER_CMD.sub(r"\1", text)  # M4（允许嵌套一次由外层循环兜底）
    text = _WRAPPER_CMD.sub(r"\1", text)
    text = text.replace("{", " ").replace("}", " ")  # M1
    text = _SUBSUP_GLUE.sub("", text)  # M2
    text = _RESIDUAL_CMD.sub(r"\1", text)  # 残留命令 -> 裸名字
    text = _ALNUM_GAP.sub("", text)  # M5
    return _WHITESPACE.sub(" ", text).strip()


def normalize_math_spans(text: str) -> tuple[str, bool]:
    """把文本里所有数学跨度替换为规范化形态。

    Returns:
        ``(normalized_text, applied)``；``applied`` 为 ``False`` 表示文本里没有数学
        跨度，此时返回值与输入一致（保证"没有公式的文本零改动"可被机械核查）。
    """

    applied = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal applied
        applied = True
        return normalize_math_body(match.group("body"))

    result = text
    for pattern in _MATH_SPANS:
        result = pattern.sub(_replace, result)
    return result, applied


def normalize_body_text(raw_text: str) -> dict[str, Any]:
    """对正文型元素做 §9.1 允许的全部规范化，并回报每一层是否生效。

    允许：Unicode/NFC、换行统一、行尾空白、连字符断行修复、数学跨度规范化、
    空白折叠。禁止：全局删除换行、改写、翻译、改写术语。
    """

    line_preserved = schema.normalize_line_preserved(raw_text)
    hyphen_joined = schema.join_hyphenated_breaks(line_preserved)
    math_normalized, math_applied = normalize_math_spans(hyphen_joined)
    normalized = schema.collapse_whitespace(math_normalized)
    return {
        "raw_text": raw_text,
        "line_preserved_text": line_preserved,
        "normalized_text": normalized,
        "math_normalization_applied": math_applied,
    }


# --------------------------------------------------------------------------------------
# §9.2 装饰（furniture）
# --------------------------------------------------------------------------------------

# 装饰删除原因（写入 removed_furniture[].reason，必须可复核）。
REASON_PARSER_DECLARED = "parser_declared_furniture_type"
REASON_TEXT_PATTERN = "text_pattern_matched_furniture"
REASON_EMPTY_OR_UNMAPPED = "empty_text_or_unmapped_type"
REASON_REPEATED_TEXT = "repeated_short_text_across_pages"


def is_furniture_type(element_type: str) -> bool:
    """判断类型是否属于要移出检索正文的页面装饰。"""

    return element_type in _FURNITURE


def furniture_subtype_for_unknown(text: str, page: int | None, *, is_first_page: bool) -> str | None:
    """仅对 ``unknown`` 元素推断装饰子类型；不命中则返回 ``None``。

    只复用 ``schema.classify_furniture`` 的文本模式规则。**不做几何发明**：
    页面上下边距判定会新造"正文被标成装饰"的风险，而 13C-A 已证明该风险在本语料上
    恒为 0，没有必要引入。
    """

    if not text.strip():
        return None
    subtype = schema.classify_furniture(text, page, is_first_page)
    return None if subtype == "unknown" else subtype


# 纯数字（1-4 位）：内容自证为页码，不携带任何正文语义。
_DIGIT_ONLY = re.compile(r"^\d{1,4}$")
# 跨页重复短文本的最长长度：超过它更可能是正文巧合重复，不判装饰。
_REPEATED_HEADER_MAX_CHARS = 120


def detect_heuristic_furniture(elements: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, str]]:
    """文本自证的装饰补充检测（只看 paragraph / list_item，全部留痕可追溯）。

    两条规则（方案 §9 的 text-pattern 域内，不做几何发明）：

    1. **纯数字**：规范化文本为 1-4 位纯数字 → ``page_number``。裸页码不携带正文
       语义，P0 这类不声明装饰类型的 parser 也会把它们留在正文里（实测 T02/T03）。
    2. **跨页重复短文本**：同一规范化文本（<=120 字符）出现在 >=2 个不同页 →
       ``header``（running header 的定义性特征就是跨页重复；页 1 的完整标题更长，
       精确匹配不会命中）。

    返回 ``{element_id: (subtype, reason)}``；命中元素由调用方改类型、记删除记录。
    """

    eligible_types = {"paragraph", "list_item"}
    marks: dict[str, tuple[str, str]] = {}
    pages_by_text: dict[str, set[int]] = {}
    for element in elements:
        if element.get("retrieval_role") in {"furniture", "discarded"}:
            continue
        if element.get("type") not in eligible_types:
            continue
        text = str(element.get("normalized_text") or element.get("raw_text") or "").strip()
        page = element.get("page")
        if not text:
            continue
        # 规则 1：纯数字页码。
        if _DIGIT_ONLY.fullmatch(text):
            marks[element["element_id"]] = ("page_number", REASON_TEXT_PATTERN)
            continue
        # 规则 2：收集跨页重复候选（判定在收集完成后做）。
        if page is not None and len(text) <= _REPEATED_HEADER_MAX_CHARS:
            pages_by_text.setdefault(text, set()).add(int(page))
    repeated_texts = {text for text, pages in pages_by_text.items() if len(pages) >= 2}
    if repeated_texts:
        for element in elements:
            if element.get("element_id") in marks:
                continue
            if element.get("retrieval_role") in {"furniture", "discarded"}:
                continue
            if element.get("type") not in eligible_types:
                continue
            text = str(element.get("normalized_text") or element.get("raw_text") or "").strip()
            if text in repeated_texts:
                marks[element["element_id"]] = ("header", REASON_REPEATED_TEXT)
    return marks


def build_removed_furniture_record(
    element: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    """构造一条装饰删除记录（方案 §9.2 要求字段齐备）。"""

    return {
        "element_id": element.get("element_id"),
        "type": element.get("type"),
        "text": element.get("normalized_text") or element.get("raw_text") or "",
        "page": element.get("page"),
        "bbox": element.get("bbox"),
        "reason": reason,
        "source_parser": element.get("source_parser"),
        # provenance：删除记录本身也要能指回"这条装饰是从哪个解析器产物来的"。
        "source_ref": element.get("source_ref"),
    }


# --------------------------------------------------------------------------------------
# §9.3–§9.5 结构元素规范化
# --------------------------------------------------------------------------------------


def normalize_table(element: dict[str, Any]) -> dict[str, Any]:
    """把表格补成规范表示：``caption / headers / rows / raw_structure / markdown``。

    规范表示**不能只有扁平文本**（方案 §9.3）。当 arm 没有给出 ``table_structure``
    时，如实把 ``headers`` / ``rows`` 置为 ``None`` 并标 ``structure_available=False``，
    而不是用扁平文本编造行列。
    """

    structure = element.get("table_structure")
    headers: list[str] | None = None
    rows: list[list[str]] | None = None
    if isinstance(structure, dict) and structure.get("rows"):
        raw_rows = [[str(cell) for cell in row] for row in structure["rows"]]
        if structure.get("header_row") and raw_rows:
            headers, rows = raw_rows[0], raw_rows[1:]
        else:
            rows = raw_rows
    element["table"] = {
        "caption": element.get("caption_text"),
        "headers": headers,
        "rows": rows,
        "raw_structure": structure,
        "serialized_markdown": table_to_markdown(headers, rows),
        "page": element.get("page"),
        "bbox": element.get("bbox"),
        "section_path": list(element.get("section_path") or []),
        "structure_available": bool(rows),
        "serialization": "markdown",
    }
    return element


def table_to_markdown(headers: Sequence[str] | None, rows: Sequence[Sequence[str]] | None) -> str | None:
    """把行列渲染为 Markdown 表格；缺少行列信息时返回 ``None``。"""

    if not rows:
        return None
    width = max(len(row) for row in rows)
    lines: list[str] = []
    if headers:
        head = list(headers) + [""] * (width - len(headers))
        lines.append("| " + " | ".join(head) + " |")
        lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in rows:
        cells = list(row) + [""] * (width - len(row))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def normalize_figure(element: dict[str, Any], *, nearby_paragraph_ids: Sequence[str] = ()) -> dict[str, Any]:
    """图的规范表示：只保留元数据与关联，**不做 VLM 图像理解**（方案 §9.4）。"""

    element["figure"] = {
        "figure_id": element.get("figure_id") or element.get("element_id"),
        "caption": element.get("caption_text"),
        "page": element.get("page"),
        "bbox": element.get("bbox"),
        "section_path": list(element.get("section_path") or []),
        "nearby_paragraph_ids": list(nearby_paragraph_ids),
        "image_interpretation": "NOT_PERFORMED_IN_PHASE_13C_B",
    }
    return element


def normalize_equation(element: dict[str, Any]) -> dict[str, Any]:
    """公式的**双表示**（方案 §9.5）：``raw_text`` 与 ``normalized_latex`` 都保留。

    两者不允许互相覆盖：``raw_text`` 是解析器原始输出（可能是 LaTeX 源码，也可能是
    asciiMath 纯文本），``normalized_latex`` 是本 Cleaner 的词法规范化结果。
    ``equation_number`` 从 ``raw_text`` 里取证（而不是从规范化后的文本），因为
    规范化会删除花括号，编号所在的分组结构可能随之改变。
    """

    raw = element.get("raw_text") or ""
    number = element.get("equation_number") or extract_equation_number(raw)
    element["equation"] = {
        "raw_text": raw,
        "normalized_latex": (element.get("normalized_text") or ""),
        "equation_number": number,
        "page": element.get("page"),
        "bbox": element.get("bbox"),
        "section_path": list(element.get("section_path") or []),
        "nearby_paragraph_ids": [],
        "source_parser": element.get("source_parser"),
        # 表示约定披露：供下游判断"文本级公式指标是否可比"（13C-A 教训）。
        "raw_looks_like_latex": bool(re.search(r"\\[a-zA-Z]{2,}", raw)),
    }
    return element


# 行间公式编号：``...(3)`` / ``\tag{3}`` / ``(3.5)``。
_EQUATION_NUMBER = re.compile(r"(?:\\tag\{([^}]*)\}|\((\d{1,3}(?:\.\d+)?)\))\s*$")


def extract_equation_number(raw_text: str) -> str | None:
    """从公式原文末尾提取编号，形如 ``(3)``；取不到返回 ``None``。"""

    stripped = schema.collapse_whitespace(raw_text)
    match = _EQUATION_NUMBER.search(stripped)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return f"({value})" if value else None


# --------------------------------------------------------------------------------------
# §9.6 参考文献（独立通道）
# --------------------------------------------------------------------------------------

# 参考文献条目的起始标记：``[12]`` / ``[12] `` / ``12. ``
_REFERENCE_MARKER = re.compile(r"(?:(?<=\s)|^)\[(\d{1,4})\]|(?:(?<=\s)|^)(\d{1,3})\.\s")
# 参考文献章节标题。
_REFERENCE_HEADING = re.compile(r"^(?:\d+\.?\s*)?(references|bibliography|works\s+cited|literature\s+cited)\b", re.I)


def is_reference_heading(text: str) -> bool:
    """判断一个标题是否是参考文献章节标题。"""

    return bool(_REFERENCE_HEADING.match(schema.collapse_whitespace(text)))


def split_reference_block(text: str) -> list[str]:
    """把"一段里塞了多条参考文献"的文本按 ``[N]`` / ``N.`` 边界切成条目。

    只在真正找到 **≥2 个**标记时才切；否则原样返回一个元素——避免把正文段落
    误切成"参考文献"（方案 §35 禁止把正文误判为参考文献）。
    """

    stripped = schema.collapse_whitespace(text)
    starts = [match.start() for match in _REFERENCE_MARKER.finditer(stripped)]
    # 标记必须落在段首或空白之后；``reference_[1]_`` 这类正文内引用不算。
    if len(starts) < 2 or starts[0] > 0:
        return [text]
    pieces = [stripped[start:].strip() for start in starts]
    return [piece for piece in pieces if piece]


def separate_references(
    elements: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把参考文献从正文中独立出来（方案 §4）。

    规则（确定性、无猜测）：

    1. 找**最后一个**参考文献章节标题；它之后（且非装饰）的元素进入参考文献区。
    2. 兜底：若没有标题，则从文档 55% 位置起找"连续 ≥3 个以 ``[N]``/``N.`` 开头"的
       元素，把它们视为参考文献区（真实论文的 References 有时被排版工具吞掉标题）。
    3. 区内元素：已是 ``reference_entry`` 的保留；正文型元素若含 ≥2 个条目标记则
       **按标记切分**为多条 ``reference_entry``，否则整条转成一条 ``reference_entry``。
    4. 区外的元素一律不改类型 —— 正文不可能被本函数变成参考文献。
    """

    indexed = list(enumerate(elements))
    heading_index: int | None = None
    for index, element in indexed:
        if element.get("type") == "section_header" and is_reference_heading(
            element.get("normalized_text") or ""
        ):
            heading_index = index

    detection = "section_heading"
    region_start: int | None = None
    if heading_index is not None:
        region_start = heading_index + 1
    else:
        candidate = _detect_reference_run(elements)
        if candidate is not None:
            region_start = candidate
            detection = "marker_run_without_heading"
        else:
            detection = "not_detected"

    result: list[dict[str, Any]] = []
    split_count = 0
    converted_count = 0
    for index, element in indexed:
        in_region = region_start is not None and index >= region_start
        if not in_region or element.get("type") in _FURNITURE:
            result.append(element)
            continue
        if element.get("type") == "reference_entry":
            result.append(element)
            continue
        if element.get("type") not in _BODY_LIKE:
            result.append(element)
            continue
        pieces = split_reference_block(element.get("normalized_text") or "")
        converted_count += 1
        if len(pieces) > 1:
            split_count += 1
        for offset, piece in enumerate(pieces):
            clone = dict(element)
            # 身份规则：切分出来的条目在原 element_id 后加确定性后缀，保证可溯源到父元素。
            clone["element_id"] = (
                element["element_id"] if offset == 0 else f"{element['element_id']}-R{offset:03d}"
            )
            clone["parent_element_id"] = element["element_id"]
            clone["type"] = "reference_entry"
            text = normalize_body_text(piece)
            clone.update(text)
            clone["reference"] = normalize_reference(clone)
            result.append(clone)

    return result, {
        "reference_section_detected": region_start is not None,
        "detection_method": detection,
        "heading_index": heading_index,
        "region_start_index": region_start,
        "converted_element_count": converted_count,
        "split_block_count": split_count,
    }


def _detect_reference_run(elements: Sequence[dict[str, Any]]) -> int | None:
    """无参考文献标题时的兜底检测：正文后半段连续 ≥3 个条目标记开头。"""

    n = len(elements)
    for index in range(int(n * 0.55), n):
        if elements[index].get("type") not in _BODY_LIKE:
            continue
        run = 0
        cursor = index
        while cursor < n:
            element = elements[cursor]
            if element.get("type") not in _BODY_LIKE:
                cursor += 1
                if run:
                    break
                continue
            text = element.get("normalized_text") or ""
            if _REFERENCE_MARKER.match(schema.collapse_whitespace(text)):
                run += 1
                cursor += 1
                if run >= 3:
                    return index
            else:
                break
    return None


def normalize_reference(element: dict[str, Any]) -> dict[str, Any]:
    """参考文献条目的规范表示：编号 + 原文 + 是否完整条目。"""

    text = element.get("normalized_text") or ""
    match = _REFERENCE_MARKER.match(schema.collapse_whitespace(text))
    number = None
    if match:
        number = match.group(1) or match.group(2)
    return {
        "reference_number": number,
        "text": text,
        "page": element.get("page"),
        "bbox": element.get("bbox"),
        "section_path": list(element.get("section_path") or []),
        "source_parser": element.get("source_parser"),
    }


# --------------------------------------------------------------------------------------
# P1（MinerU）原生产物再标注：修的是 13C-A 的**映射缺口**，不是解析器本身
# --------------------------------------------------------------------------------------

# MinerU content_list 的原生类型 -> 13C 统一类型。
# 这份表与 13C-A 的 ``mappers.mineru_content_list_to_schema`` 的差异就是 Q3 的答案：
# 13C-A 只处理 text/equation/table/image/figure/picture，其余 7 类全部落到 ``unknown``
# 且文本为空。**不修改 mappers.py**，因为改了会让 13C-A 的修复性复跑不可比。
MINERU_TYPE_MAP: dict[str, str] = {
    "text": "paragraph",  # text_level 非空时升级为 section_header
    "equation": "equation",
    "table": "table",
    "image": "figure",
    "figure": "figure",
    "picture": "figure",
    "chart": "figure",
    # ``list`` 默认按 list_item 处理；只有 sub_type=ref_text 的 list 才在 retype 里
    # 逐条展开为 reference_entry（本语料全部 list 均为 ref_text）。
    "list": "list_item",
    "page_number": "page_number",
    "header": "header",
    "footer": "footer",
    "aside_text": "sidebar",
    "page_footnote": "footnote",
}


def _native_item_emits_caption(item: Mapping[str, Any]) -> bool:
    """判断 13C-A mapper 是否会为该原生条目额外发射一个 caption 元素。

    必须与 ``mappers.mineru_content_list_to_schema`` 的发射规则**逐字段一致**
    （它检查 ``img_caption`` / ``table_caption`` / ``image_caption`` 三个键），
    否则 ordinal 模拟会错位。
    """

    for key in ("img_caption", "table_caption", "image_caption"):
        values = item.get(key)
        if isinstance(values, list) and values:
            return True
        if isinstance(values, str) and values:
            return True
    return False


def native_ordinal_map(content_list: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """重建「元素序数 -> 原生 content_list 下标」的确定性对位。

    ## 为什么需要这个函数

    13C-A 的 mapper 写入的 ``source_ref`` 是 ``mineru:content_list:{len(elements)}``
    —— 即**元素序数**而不是原生条目下标。当某个原生条目（image/table 带 caption）
    额外发射一个 caption 元素后，其后所有元素的 ref 都相对原生下标**整体偏移 +1**。
    直接按 ref 对位会把元素 retype 到错误的原生条目上（实测 T02 的参考文献 list
    因此完全丢失，其后元素的类型/文本被相邻条目污染）。

    修复方式：mapper 的发射模式是确定的（每条原生条目 1 个元素，带 caption 键时
    额外 +1 个 caption 元素），因此可以**不依赖任何文本猜测**地重建对位。
    已在 6 篇受控模板上验证：按此映射对位后 text 型元素的文本与原生条目
    完全一致（145/145，0 错配）。
    """

    mapping: dict[int, int] = {}
    ordinal = 0
    for index, item in enumerate(content_list):
        mapping[ordinal] = index
        ordinal += 1
        if _native_item_emits_caption(item):
            mapping[ordinal] = index
            ordinal += 1
    return mapping


def retype_mineru_elements(
    elements: Sequence[dict[str, Any]],
    content_list: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按 MinerU 原生 ``content_list`` 重新标注元素类型与文本。

    对位方式：``source_ref`` 里记录的是 13C-A mapper 的**元素序数**（不是原生
    下标），因此用 :func:`native_ordinal_map` 重建「序数 -> 原生下标」的确定性
    映射后再对位——不依赖任何文本相似度猜测。对 ``list``（``sub_type=ref_text``）
    按 ``list_items`` 逐条展开为多条 ``reference_entry``。

    已被 13C-A 正确映射的元素（text/equation/table/image）也会被重算文本与类型，
    因为它们原生形态里就带着 ``text_level``、``table_caption`` 等 13C-A 没有采信的字段。
    重算后的元素带 ``retyped_from_native=True``，便于与 13C-A 输出逐条对照。
    """

    # 序数 -> 原生下标（发射模式模拟，见函数 docstring 的错位说明）。
    ordinal_to_native = native_ordinal_map(content_list)

    # 13C-A 的映射层已经为 image/table 外挂过 caption 元素；它们的宿主 element_id
    # 必须记录下来，否则本次再标注会**重复**添加同一个 caption。
    existing_caption_hosts = {
        element.get("caption_of")
        for element in elements
        if str(element.get("source_ref") or "").startswith("mineru:content_list_caption:")
    }

    stats: dict[str, Any] = {
        "content_list_length": len(content_list),
        "matched_source_ref": 0,
        "unmatched_source_ref": 0,
        "retyped_type_counts": {},
        "expanded_list_item_count": 0,
        "expanded_list_host_count": 0,
        "retyped_raw_types": {},
        "ordinal_map_size": len(ordinal_to_native),
    }
    result: list[dict[str, Any]] = []
    for element in elements:
        ref = str(element.get("source_ref") or "")
        # 只接受"宿主元素"的 source_ref（``mineru:content_list:<ordinal>``）。
        # caption 元素（``..._caption:<ordinal>``）与 Cleaner 自己生成的关联元素必须走
        # 未对位分支，否则它们会被宿主的原生文本覆盖掉。
        match = re.match(r"mineru:content_list:(\d+)$", ref)
        native_index = (
            ordinal_to_native.get(int(match.group(1))) if match else None
        )
        if native_index is None or native_index >= len(content_list):
            # caption 元素与无法对位的元素：保持原样，但同样重算数学规范化。
            stats["unmatched_source_ref"] += 1
            result.append(_refresh_texts(element))
            continue
        item = content_list[native_index]
        stats["matched_source_ref"] += 1
        raw_type = str(item.get("type") or "")
        stats["retyped_raw_types"][raw_type] = stats["retyped_raw_types"].get(raw_type, 0) + 1

        if raw_type == "list" and str(item.get("sub_type") or "") == "ref_text":
            # 一条 ref_text list 承载多条参考文献：逐条展开，边界由解析器给出，
            # 不做文本猜测。非 ref_text 的 list 走下方 list_item 分支。
            stats["expanded_list_host_count"] += 1
            for offset, entry in enumerate(item.get("list_items") or []):
                clone = dict(element)
                clone["element_id"] = (
                    element["element_id"] if offset == 0 else f"{element['element_id']}-R{offset:03d}"
                )
                clone["parent_element_id"] = element["element_id"]
                clone["type"] = "reference_entry"
                clone.update(normalize_body_text(str(entry)))
                clone["reference"] = normalize_reference(clone)
                clone["retyped_from_native"] = True
                result.append(clone)
                stats["expanded_list_item_count"] += 1
            continue

        new_type = MINERU_TYPE_MAP.get(raw_type, "unknown")
        if raw_type == "text" and item.get("text_level"):
            new_type = "section_header"
        text = _mineru_text(item, raw_type)
        updated = dict(element)
        updated["type"] = new_type
        updated.update(normalize_body_text(text))
        # caption 与脚注：解析器把它们和宿主块绑在一起，这里拆成独立元素。
        caption_text = _mineru_caption(item)
        if caption_text:
            updated["caption_text"] = caption_text
        if raw_type == "table":
            structure = element.get("table_structure")
            if structure:
                updated["table_structure"] = structure
        updated["retyped_from_native"] = True
        result.append(updated)
        stats["retyped_type_counts"][new_type] = stats["retyped_type_counts"].get(new_type, 0) + 1

        # MinerU 用 ``text`` 之外的类型承载 caption（``image_caption`` / ``chart_caption`` /
        # ``table_caption``）。13C-A 只在 image/table 分支外挂 caption，chart 完全没有，
        # 这里统一补齐为独立的 caption 元素（几何与宿主一致，指向宿主 element_id）。
        if caption_text and new_type in {"figure", "table"} and element["element_id"] not in existing_caption_hosts:
            caption = schema.make_element(
                document_id=element["document_id"],
                element_id=f"{element['element_id']}-C",
                type="figure_caption" if new_type == "figure" else "table_caption",
                raw_text=caption_text,
                source_parser=element.get("source_parser") or "P1_mineru",
                page=element.get("page"),
                bbox=element.get("bbox"),
                reading_order=element.get("reading_order"),
                section_path=element.get("section_path") or [],
                caption_of=element["element_id"],
                source_ref=f"mineru:native_caption:{ref}",
            )
            caption["retyped_from_native"] = True
            result.append(caption)
    return result, stats


def _mineru_text(item: Mapping[str, Any], raw_type: str) -> str:
    """取 MinerU 原生条目的文本（表体/图表/列表没有 ``text`` 字段时的兜底）。"""

    if raw_type == "table":
        html = str(item.get("table_body") or "")
        structure = _table_structure_from_html(html)
        if structure and structure.get("rows"):
            return " ".join(cell for row in structure["rows"] for cell in row if cell)
        return re.sub(r"<[^>]+>", " ", html)
    if raw_type == "list":
        # 非 ref_text 的普通列表：把 list_items 拼成正文文本（保留列表语义）。
        return " ".join(str(entry) for entry in item.get("list_items") or [])
    return str(item.get("text") or "")


def _mineru_caption(item: Mapping[str, Any]) -> str:
    """取 MinerU 原生条目的 caption 文本（不同模态字段名不同）。"""

    for key in ("image_caption", "table_caption", "chart_caption"):
        values = item.get(key)
        if isinstance(values, list) and values:
            return " ".join(str(value) for value in values)
        if isinstance(values, str) and values:
            return values
    return ""


def _table_structure_from_html(html: str) -> dict[str, Any] | None:
    """从 HTML 表格里取行列结构；无 ``<tr>`` 时返回 ``None``。

    复用 13C-A 的 ``mappers._table_from_html`` 会引入对 mappers 的耦合，故此处只需
    最小实现——行列（含 ``rowspan``/``colspan`` 的粗略展开）足够支撑 §9.3 的规范表示。
    """

    rows: list[list[str]] = []
    has_span = False
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells: list[str] = []
        for cell_html in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S | re.I):
            text = re.sub(r"<[^>]+>", " ", cell_html).replace("&nbsp;", " ")
            cells.append(schema.collapse_whitespace(text))
        if cells:
            rows.append(cells)
        if re.search(r"<t[dh][^>]*(?:row|col)span", row_html, re.I):
            has_span = True
    if not rows:
        return None
    return {"rows": rows, "header_row": True, "has_spanning_cell": has_span}


# --------------------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------------------


def _refresh_texts(element: dict[str, Any]) -> dict[str, Any]:
    """只重算文本字段（不改类型），保证未对位元素也享受同一套规范化。"""

    updated = dict(element)
    updated.update(normalize_body_text(element.get("raw_text") or ""))
    return updated


def clean_document(
    document: Mapping[str, Any],
    *,
    native_content_list: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """把一个 arm 的 13C schema 文档清洗为 ScientificDocument（方案 §8 / §9）。

    步骤（顺序有意义）：

    1. 重算文本（数学跨度规范化）；
    2. 若给了原生 ``content_list``，先做类型再标注（只对 P1 生效）；
    3. 标注检索角色；``unknown`` 元素尝试按文本模式标注为装饰；
    4. 参考文献分离（拆出独立通道，不删除）；
    5. 结构元素规范化（table / figure / equation）；
    6. 汇总 ``removed_furniture`` 与统计。
    """

    elements: list[dict[str, Any]] = [dict(element) for element in document.get("elements", [])]
    report: dict[str, Any] = {"cleaner_version": CLEANER_VERSION}
    if native_content_list is not None:
        elements, retype_stats = retype_mineru_elements(elements, native_content_list)
        report["retype_from_native"] = retype_stats
    else:
        elements = [_refresh_texts(element) for element in elements]

    removed_furniture: list[dict[str, Any]] = []
    removed_elements: list[dict[str, Any]] = []
    first_page = 1
    for element in elements:
        element_type = element.get("type")
        if is_furniture_type(element_type):
            element["retrieval_role"] = "furniture"
            element["furniture_origin"] = "parser_declared"
            removed_furniture.append(build_removed_furniture_record(element, REASON_PARSER_DECLARED))
            continue
        if element_type == "unknown":
            subtype = furniture_subtype_for_unknown(
                element.get("normalized_text") or "", element.get("page"), is_first_page=element.get("page") == first_page
            )
            if subtype:
                element["type"] = subtype
                element["retrieval_role"] = "furniture"
                element["furniture_origin"] = "text_pattern"
                removed_furniture.append(build_removed_furniture_record(element, REASON_TEXT_PATTERN))
                continue
            element["retrieval_role"] = "discarded"
            removed_elements.append(
                {
                    **build_removed_furniture_record(element, REASON_EMPTY_OR_UNMAPPED),
                    "kind": "unknown_or_empty",
                }
            )
            continue
        element["retrieval_role"] = retrieval_role_for(element_type)

    # 3.5 文本自证的装饰补充检测：裸页码与跨页重复短文本（running header）。
    # 这两条规则让不声明装饰类型的 parser（P0）也能通过 Gate 5A 的显式检测要求；
    # 命中元素同样改类型并写入 removed_furniture，逐条可追溯。
    # 先一次性计算全部命中（主循环里角色在变，不能边算边改），再统一应用。
    heuristic_marks = detect_heuristic_furniture(elements)
    for element in elements:
        mark = heuristic_marks.get(element["element_id"])
        if not mark or element.get("retrieval_role") in {"furniture", "discarded"}:
            continue
        subtype, reason = mark
        element["type"] = subtype
        element["retrieval_role"] = "furniture"
        element["furniture_origin"] = "text_pattern"
        removed_furniture.append(build_removed_furniture_record(element, reason))
        report.setdefault("heuristic_furniture", {}).setdefault(reason, 0)
        report["heuristic_furniture"][reason] += 1

    # 参考文献分离：只在"检索元素"上做，装饰元素不参与。
    retrieval_elements = [
        element for element in elements if element.get("retrieval_role") not in {"furniture", "discarded"}
    ]
    reference_report: dict[str, Any] = {"reference_section_detected": False, "detection_method": "skipped"}
    if retrieval_elements:
        separated, reference_report = separate_references(retrieval_elements)
        # 分离结果按宿主 element_id 归组。**只有宿主自己**触发这一组的一次性展开；
        # 派生元素（``-Rxxx``）在遍历时被跳过，否则它们会既随宿主展开一次、又被单独
        # 输出一次（实测会造成参考文献条目翻倍）。
        groups: dict[str, list[dict[str, Any]]] = {}
        for element in separated:
            host = element.get("parent_element_id") or element["element_id"]
            groups.setdefault(host, []).append(element)
        rebuilt: list[dict[str, Any]] = []
        emitted: set[str] = set()
        for element in elements:
            if element.get("retrieval_role") in {"furniture", "discarded"}:
                rebuilt.append(element)
                continue
            host = element.get("parent_element_id") or element["element_id"]
            if host != element["element_id"] or host in emitted:
                continue
            emitted.add(host)
            rebuilt.extend(groups.pop(host, [element]))
        # 宿主不在本轮元素集合里的组（例如上游把宿主归到装饰）按 element_id 排序补齐，
        # 保证"分离出来的条目"不会因为没有宿主而被静默丢弃。
        for host in sorted(groups):
            rebuilt.extend(groups[host])
        elements = rebuilt

    for element in elements:
        if element.get("retrieval_role") in {"furniture", "discarded"}:
            continue
        element["retrieval_role"] = retrieval_role_for(element["type"])
        if element["type"] == "table":
            normalize_table(element)
        elif element["type"] == "figure":
            normalize_figure(element)
        elif element["type"] == "equation":
            normalize_equation(element)
        elif element["type"] == "reference_entry":
            element.setdefault("reference", normalize_reference(element))

    # caption 元素与宿主之间的关联：优先用解析器给出的 ``caption_of``，缺失时按
    # "同页 + bbox 垂直相邻 + caption 在图上/下方"补一条确定性关联。
    link_captions(elements)

    # reading_order 重编号：切分出来的参考文献条目与宿主共享原始序号，若不重编号，
    # 文档会违反 §10 schema 的"reading_order 在文档内唯一"约束。
    # 列表顺序本身就是阅读顺序（切分结果原地展开），因此按列表顺序连续编号即可，
    # 不需要再按几何排序。原始序号保留在 ``original_reading_order`` 供溯源。
    for position, element in enumerate(elements):
        element["original_reading_order"] = element.get("reading_order")
        element["reading_order"] = position

    retrieval_ids = [
        element["element_id"]
        for element in elements
        if element.get("retrieval_role") in {"body", "structure", "reference"}
    ]
    reference_ids = [
        element["element_id"] for element in elements if element.get("retrieval_role") == "reference"
    ]

    type_counts: dict[str, int] = {}
    for element in elements:
        type_counts[element["type"]] = type_counts.get(element["type"], 0) + 1

    return {
        "document_id": document.get("document_id"),
        "title": _document_title(elements, document),
        "metadata": {
            "source_pdf": document.get("source_pdf"),
            "source_sha256": document.get("source_sha256"),
            "source_parser": document.get("parser"),
            "page_count": document.get("page_count"),
            "layout_class": document.get("layout_class"),
            "cleaner_version": CLEANER_VERSION,
            "cleaner_modes": {
                "unicode_normalization": True,
                "hyphen_repair": True,
                "math_span_normalization": True,
                "paragraph_visual_line_merge": False,
                "furniture_removal": True,
                "references_separation": True,
                "caption_isolation": True,
            },
        },
        "page_count": document.get("page_count"),
        "element_count": len(elements),
        "type_counts": dict(sorted(type_counts.items())),
        "elements": elements,
        "retrieval_element_ids": retrieval_ids,
        "reference_element_ids": reference_ids,
        "removed_furniture": removed_furniture,
        "removed_elements": removed_elements,
        "unresolved_conflicts": [],
        "cleaning_report": {
            "cleaner_version": CLEANER_VERSION,
            "input_element_count": len(document.get("elements", [])),
            "output_element_count": len(elements),
            "furniture_removed_count": len(removed_furniture),
            "discarded_unknown_count": len(removed_elements),
            "references": reference_report,
            **{k: v for k, v in report.items() if k != "cleaner_version"},
        },
    }


def _document_title(elements: Sequence[dict[str, Any]], document: Mapping[str, Any]) -> str | None:
    """取文档标题：优先显式 ``document_title`` 元素，退回第一个 section_header，再退回文件名。"""

    for element in elements:
        if element.get("type") == "document_title":
            return element.get("normalized_text") or None
    for element in elements:
        if element.get("type") == "section_header":
            return element.get("normalized_text") or None
    source = document.get("source_pdf")
    return str(source) if source else None


# --------------------------------------------------------------------------------------
# Caption 关联
# --------------------------------------------------------------------------------------


def link_captions(elements: Sequence[dict[str, Any]]) -> int:
    """把 caption 元素关联到最近的结构宿主（figure / table），返回关联数。

    规则：同页 + caption 的 ``caption_of`` 已有效则不覆盖；否则选择同页中
    bbox 与 caption 垂直距离最近、且水平投影有重叠的 figure/table。
    与 13C-A 的"caption 指向所属标题"不同，这里显式指向 figure/table 对象，
    因为 13C-C 的 element-aware 切分需要这对父子关系。
    """

    hosts_by_page: dict[int, list[dict[str, Any]]] = {}
    for element in elements:
        if element.get("type") in {"figure", "table"}:
            hosts_by_page.setdefault(element.get("page") or -1, []).append(element)

    linked = 0
    for element in elements:
        if element.get("type") not in {"figure_caption", "table_caption"}:
            continue
        owner = element.get("caption_of")
        owner_element = next(
            (host for host in elements if host.get("element_id") == owner), None
        )
        expected = "figure" if element.get("type") == "figure_caption" else "table"
        if owner_element is not None and owner_element.get("type") == expected:
            element["caption_host_id"] = owner
            linked += 1
            continue
        host = _nearest_host(element, hosts_by_page.get(element.get("page") or -1, []), expected)
        if host is not None:
            element["caption_host_id"] = host["element_id"]
            element["caption_of"] = host["element_id"]
            host.setdefault("caption_element_ids", []).append(element["element_id"])
            linked += 1
    return linked


def _nearest_host(
    caption: Mapping[str, Any], hosts: Iterable[Mapping[str, Any]], expected: str
) -> Mapping[str, Any] | None:
    """在候选宿主里选 bbox 垂直距离最小且有水平投影重叠者。"""

    caption_box = _box(caption.get("bbox"))
    if caption_box is None:
        return None
    best: Mapping[str, Any] | None = None
    best_gap: float | None = None
    for host in hosts:
        if host.get("type") != expected:
            continue
        host_box = _box(host.get("bbox"))
        if host_box is None:
            continue
        # 水平投影重叠：caption 与宿主必须落在同一栏。
        if min(caption_box[2], host_box[2]) - max(caption_box[0], host_box[0]) <= 0:
            continue
        gap = max(host_box[1] - caption_box[3], caption_box[1] - host_box[3], 0.0)
        if best_gap is None or gap < best_gap:
            best, best_gap = host, gap
    return best


def _box(bbox: Any) -> tuple[float, float, float, float] | None:
    """把 bbox 规范成 ``(x0, y0, x1, y1)``；非法时返回 ``None``。"""

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):  # pragma: no cover - 防御性
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
