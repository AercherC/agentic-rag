"""PDF 结构解析：block 级元素化、章节归属与页面装饰移除。

Phase 13F 晋升的移植层：全部纯函数与判定规则**逐字**移植自已归档实验脚本，
保证与冻结产物同一口径（不得在本文件内"顺手改进"任何阈值或正则）：

- ``detect_headings`` / ``resolve_section`` / ``detect_document_furniture`` /
  ``page_level_furniture_spans``：来自 ``experiments/run_phase13_kb_chunk_audit.py``
  （Phase 13A 审计，规则含实测误检的四层排除）；
- ``parse_page_blocks`` / ``_classify_block_text`` / ``_is_table_body`` /
  ``build_structured_document`` 及元素级装饰移除：来自
  ``experiments/run_phase13b_controlled_rebuild.py``（Phase 13B 受控重建 Step 1/2）。

为什么在 block 级而非整页文本上解析：``"".join(block_text)`` 与 ``page.get_text()``
在受控语料 145 页中 144 页严格相等，block 元素与官方抽取文本共享同一字符坐标系，
且自带 bbox（纯文本抽取所没有的信息）。
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pymupdf

from evidenceflow.domain.document import Document, DocumentSegment

# --------------------------------------------------------------------------------------
# 文本结构识别规则（与 13A 完全同口径，直接移植其正则，避免第二套定义）。
#
# 关键约束（Phase 13 实测结论）：PyMuPDF 把双栏页压成一条文本流，caption 常常与正文
# 同处一行；而 BGE-M3（XLM-R SentencePiece）decode 会把换行折叠为空格。
# 因此不能只在行首匹配 caption，否则会漏掉全部噪声。是否真正位于行首单独记录
# （at_line_start），用于区分"高置信 caption"与"仅出现在正文里的引用"。
# --------------------------------------------------------------------------------------

_FIGURE_ANY = re.compile(r"(?:Figure|Fig\.)\s*(\d{1,2})\b")
_FIGURE_CAPTION_FORM = re.compile(r"(?:Figure|Fig\.)\s*(\d{1,2})\s*[:.]\s*[A-Z]")
_TABLE_ANY = re.compile(r"(?:Table|Tab\.)\s*(\d{1,2})\b")
_TABLE_CAPTION_FORM = re.compile(r"(?:Table|Tab\.)\s*(\d{1,2})\s*[:.]\s*[A-Z]")
_ALGORITHM_BLOCK = re.compile(r"Algorithm\s*(\d{1,2})\b")
# 原始 LaTeX 命令泄漏：部分 PDF 的公式文本层直接保存 LaTeX 源，抽取出后无法语义化。
LATEX_COMMAND = re.compile(r"\\[a-zA-Z]{2,}")
# 参考文献条目：行首 "[N]" 形态（13B 分类时逐行 match 使用）。
_REFERENCE_ENTRY = re.compile(r"(?m)^[ \t]*\[\d{1,3}\]")
_EQUATION_NUMBER = re.compile(r"(?m)^[ \t]*\(\d{1,3}\)[ \t]*$")
_ARXIV_HEADER = re.compile(r"(?im)^\s*arXiv:\d{4}\.\d{4,5}v\d+")
_DOI = re.compile(r"(?i)\bdoi:\s*10\.\d{4,9}/\S+")
_COPYRIGHT = re.compile(r"(?i)(copyright|©|all rights reserved|permission to make digital)")
_CONFERENCE_HEADER = re.compile(
    r"(?i)(proceedings of|conference on|international conference|"
    r"computer vision and pattern recognition|neural information processing)"
)

# 章节标题：编号型（1 / 2.1 / 3.2.1）与非编号具名型。
_NUMBERED_HEADING = re.compile(
    r"(?m)^[ \t]*(\d{1,2}(?:\.\d{1,2}){0,3})[ \t]*\.?[ \t]+([A-Z][^\n]{1,80}?)[ \t]*$"
)
_NAMED_HEADINGS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental results",
    "experiment",
    "results",
    "results and analysis",
    "ablation study",
    "ablation",
    "discussion",
    "conclusion",
    "conclusions",
    "conclusion and future work",
    "limitations",
    "references",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "supplementary material",
]
_NAMED_HEADING = re.compile(
    r"(?m)^[ \t]*(" + "|".join(_NAMED_HEADINGS) + r")[ \t]*$",
    re.IGNORECASE,
)

# 章节内出现的 "N. Title" 形式也可能是列表项；用这些模式排除明显不是标题的行。
# 引用特征与数学符号：用于排除"参考文献条目 / 表格数值行"被误判成章节标题。
_CITATION_LIKE = re.compile(r"(?i)\bet al\b|,\s*[A-Z]\.|\(\d{4}\)|\b(?:19|20)\d{2}\b")
_HEADING_MATH = re.compile(r"[×≤≥↑↓≈∈∑∏±]")

_HEADING_REJECT = re.compile(
    r"(?i)(^|[,\s])(?:table|figure|fig\.|eq\.|equation|algorithm)\s*\d"
    r"|[;=<>]|\b(?:is|are|was|were|can|we|our|the)\b"
)

# 页级装饰的精确模式（与"跨页重复串"口径互补；逐页变化的页码必须单独定位）。
_PAGE_NUMBER_LINE = re.compile(r"^[ \t]*\d{1,5}[ \t]*$")
_OPEN_ACCESS_WATERMARK = re.compile(
    r"(?i)this\s+\w+\s+paper is the open access version"
    r"|provided by the computer vision foundation"
    r"|final published version of the proceedings"
    r"|is available on ieee xplore"
)

# --------------------------------------------------------------------------------------
# 13B 元素分类常量（与 13A 共享正则，阈值来自 Phase 13A/13B 实测校准）。
# --------------------------------------------------------------------------------------

# 结构化元素的受控类型（13B 口径；hybrid Chunker 只消费 furniture 移除后的元素）。
ELEMENT_TYPES = (
    "title",
    "paragraph",
    "figure_caption",
    "table_caption",
    "table",
    "equation",
    "reference_entry",
    "furniture",
)
# 元素级清理删除的装饰类型。
FURNITURE_ELEMENT_TYPES = {"furniture"}

# 公式编号行：整行只有 "(N)"。
_EQUATION_NUMBER_ONLY = re.compile(r"\(\d{1,3}\)")
# 全大写 "TABLE N" 独占首行的 caption 风格（adapointr 等，无冒号）。
_TABLE_ALLCAPS_HEADING = re.compile(r"TABLE\s+\d{1,2}\b")

# table 体判定：紧跟 caption 的块若满足"多短行（每行 ≤2 词）"或"数值密集"任一条件，
# 即视为表格体。表格头行（如 "Methods / Plane / Cabinet"）全为标签、数值占比低，必须靠
# "多短行"信号才能识别——这是 Phase 13A 已证"表格行列关系丢失"的直接修复点。
TABLE_CELL_MAX_WORDS_PER_LINE = 2.5
TABLE_CELL_MIN_LINES = 3
TABLE_NUMERIC_RATIO = 0.30
TABLE_NUMERIC_MIN_TOKENS = 4


# --------------------------------------------------------------------------------------
# 章节标题检测与归属（13A 移植）
# --------------------------------------------------------------------------------------


def detect_headings(segments: Sequence[DocumentSegment]) -> list[dict[str, Any]]:
    """在文本中定位章节标题，返回按原文顺序排列的标题列表。

    只识别"独立成行的编号标题"与"独立成行的具名标题"。为避免把正文/表格/参考文献误判成
    章节，叠加四层排除规则（每一条都来自 Phase 13 实测误检）：

    1. 编号标题：主编号 <= 20、标题词数 <= 8、不含年份/引用特征/数学符号、不以标点结尾；
       同一文档内一级编号不得重复（实测表格单元格被读成 "4 Note that, ..."）。
    2. 具名标题：必须非全小写，且在整篇文档中只出现一次
       （实测 "Methods" 是 Table 列头，共出现 7 次，全部排除）。
    3. 同位置同时命中编号与具名规则时只保留编号标题。
    4. 标题只在其所属 Segment 内生效（Segment 边界即 PDF 页边界）。
    """

    numbered: list[dict[str, Any]] = []
    named: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        content = segment.content
        for match in _NUMBERED_HEADING.finditer(content):
            number, title = match.group(1), match.group(2).strip()
            major = int(number.split(".")[0])
            # 规则 1：编号与标题形态过滤。
            if major > 20 or len(title) < 3 or len(title.split()) > 8:
                continue
            if _HEADING_REJECT.search(title) or _CITATION_LIKE.search(title):
                continue
            if _HEADING_MATH.search(title):
                continue
            if title[-1] in ",.;:":
                continue
            numbered.append(
                {
                    "segment_index": segment_index,
                    "char_offset": match.start(),
                    "page_number": segment.page_number,
                    "level": number.count(".") + 1,
                    "number": number,
                    "title": f"{number} {title}",
                    "kind": "numbered",
                }
            )
        for match in _NAMED_HEADING.finditer(content):
            raw = match.group(1)
            # 规则 2：全小写行几乎不可能是章节标题。
            if raw.islower():
                continue
            named.append(
                {
                    "segment_index": segment_index,
                    "char_offset": match.start(),
                    "page_number": segment.page_number,
                    "level": 1,
                    "number": None,
                    "title": raw.title() if raw.isupper() else raw,
                    "kind": "named",
                }
            )

    # 规则 1b：同一文档内一级编号不得重复。
    seen_major: set[str] = set()
    kept_numbered: list[dict[str, Any]] = []
    for heading in numbered:
        if heading["level"] == 1:
            if heading["number"] in seen_major:
                continue
            seen_major.add(heading["number"])
        kept_numbered.append(heading)

    # 规则 2b：具名标题必须在整篇文档中唯一出现，否则视为表格列头/页眉噪声。
    title_counts = Counter(heading["title"].lower() for heading in named)
    kept_named = [heading for heading in named if title_counts[heading["title"].lower()] == 1]

    headings = kept_numbered + kept_named
    # 规则 3 / 4：按原文顺序排序，并对同位置的编号/具名重复做去重。
    headings.sort(
        key=lambda item: (
            item["segment_index"],
            item["char_offset"],
            item["kind"] != "numbered",
        )
    )
    deduped: list[dict[str, Any]] = []
    for heading in headings:
        if (
            deduped
            and deduped[-1]["segment_index"] == heading["segment_index"]
            and abs(deduped[-1]["char_offset"] - heading["char_offset"]) <= 2
            and deduped[-1]["kind"] == "numbered"
        ):
            continue
        deduped.append(heading)
    return deduped


def resolve_section(
    headings: Sequence[dict[str, Any]], segment_index: int, char_offset: int
) -> dict[str, Any]:
    """返回给定字符位置所属的 section 层级路径。

    规则：取该位置之前最近的编号/具名标题作为 section；其下最近的更深标题作为 subsection。
    标题只在其所属 Segment 内生效（Segment 边界即 PDF 页边界，标题不会跨页延续）。
    """

    applicable = [
        heading
        for heading in headings
        if heading["segment_index"] <= segment_index
        and (heading["segment_index"] < segment_index or heading["char_offset"] <= char_offset)
    ]
    if not applicable:
        return {"section": None, "subsection": None, "section_path": []}

    top = None
    sub = None
    for heading in applicable:
        if heading["level"] <= 1:
            top = heading
            sub = None
        else:
            sub = heading
    section_title = top["title"] if top else None
    subsection_title = sub["title"] if sub else None
    path = [item for item in (section_title, subsection_title) if item]
    return {"section": section_title, "subsection": subsection_title, "section_path": path}


# --------------------------------------------------------------------------------------
# 页面装饰检测（13A 移植）
# --------------------------------------------------------------------------------------


def detect_document_furniture(documents: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """检测每篇 PDF 的重复页面装饰（页眉/页脚/arXiv/DOI/版权/会议抬头/页码）。

    方法：统计规范化后（数字折叠为 ``#``）在至少 3 页且不少于总页数 30% 的行，
    作为"重复装饰行"，用于后续 header/footer contamination 判定。
    """

    furniture: dict[str, list[str]] = {}
    for document in documents:
        cleaned_lines: list[Counter[str]] = []
        for segment in document["segments"]:
            counter: Counter[str] = Counter()
            for line in segment.content.split("\n"):
                stripped = line.strip()
                if not stripped or len(stripped) > 160:
                    continue
                normalized = re.sub(r"\d+", "#", stripped).lower()
                counter[normalized] += 1
            cleaned_lines.append(counter)
        page_count = max(len(cleaned_lines), 1)
        threshold = max(3, math.ceil(page_count * 0.30))
        overall: Counter[str] = Counter()
        for counter in cleaned_lines:
            overall.update(counter.keys())
        repeated = {
            line
            for line, pages in overall.items()
            if pages >= threshold and len(line) >= 3
        }
        # 补充模式型装饰（arXiv 抬头 / DOI / 版权 / 会议抬头 / 纯页码）。
        for segment in document["segments"]:
            for line in segment.content.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if (
                    _ARXIV_HEADER.search(stripped)
                    or _DOI.search(stripped)
                    or _COPYRIGHT.search(stripped)
                    or _CONFERENCE_HEADER.search(stripped)
                    or re.fullmatch(r"\d{1,3}", stripped)
                ):
                    repeated.add(re.sub(r"\d+", "#", stripped).lower())
        furniture[document["document_id"]] = sorted(repeated)
    return furniture


def page_level_furniture_spans(
    document: dict[str, Any], repeated_furniture: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """返回页级装饰元素的精确字符区间：页码 / 重复页眉页脚 / arXiv 侧栏 / 版权 / 水印。

    判定规则（确定性、可复算）：
      - 页码：该页首非空行或末非空行且整行只由 1–5 位数字组成。只看首尾两行，
        避免把页脚紧邻的表格数值行误判成页码。
      - 重复页眉页脚：整行规范化（数字→#）后命中 ``detect_document_furniture`` 的结果集。
        先排除形如参考文献 / 表格行的重复串（含年份、"pp."、"et al" 等）。
      - arXiv 侧栏 / 版权 / 开放获取水印：按模式匹配，记录精确区间。
    """

    spans: list[dict[str, Any]] = []
    # 只保留含字母的重复串：数字折叠会把 "1"/"3.1" 这类页码或小节号归一化为 "#"/"#.#"，
    # 它们不是页眉页脚，且页码已由 PAGE_NUMBER 规则更精确地覆盖。
    furniture_set = {
        entry for entry in repeated_furniture if re.search(r"[a-z]", entry)
    }

    def record(
        kind: str,
        segment_index: int,
        page_number: int | None,
        start: int,
        end: int,
        text: str,
        line_position: str | None = None,
    ) -> None:
        entry = {
            "kind": kind,
            "document_id": document["document_id"],
            "segment_index": segment_index,
            "page_number": page_number,
            "char_start": start,
            "char_end": end,
            "text": text.strip()[:120],
        }
        if line_position:
            entry["line_position"] = line_position
        spans.append(entry)

    for segment_index, segment in enumerate(document["segments"]):
        content = segment.content
        # 逐行记录绝对字符偏移，保证区间可直接与 block 元素的字符区间相交。
        line_offsets: list[tuple[int, int, str]] = []
        cursor = 0
        for line in content.split("\n"):
            line_offsets.append((cursor, cursor + len(line), line))
            cursor += len(line) + 1
        non_empty = [item for item in line_offsets if item[2].strip()]
        if not non_empty:
            continue

        candidates: list[tuple[int, int, str]] = [non_empty[0]]
        if non_empty[-1] is not non_empty[0]:
            candidates.append(non_empty[-1])
        page_number_lines: set[tuple[int, int]] = set()
        for start, end, line in candidates:
            if _PAGE_NUMBER_LINE.match(line):
                page_number_lines.add((start, end))
                record(
                    "PAGE_NUMBER",
                    segment_index,
                    segment.page_number,
                    start,
                    end,
                    line,
                    "first" if (start, end, line) == non_empty[0] else "last",
                )

        # 重复页眉页脚：页级按行匹配，命中即记录精确区间。
        if furniture_set:
            for start, end, line in line_offsets:
                stripped = line.strip()
                if not stripped or len(stripped) > 160 or (start, end) in page_number_lines:
                    continue
                if _CITATION_LIKE.search(stripped) or _REFERENCE_ENTRY.match(stripped):
                    continue
                if re.sub(r"\d+", "#", stripped).lower() in furniture_set:
                    record(
                        "REPEATED_FURNITURE",
                        segment_index,
                        segment.page_number,
                        start,
                        end,
                        line,
                    )

        for kind, pattern in (
            ("ARXIV_SIDEBAR", _ARXIV_HEADER),
            ("COPYRIGHT", _COPYRIGHT),
            ("OPEN_ACCESS_WATERMARK", _OPEN_ACCESS_WATERMARK),
        ):
            for match in pattern.finditer(content):
                record(
                    kind,
                    segment_index,
                    segment.page_number,
                    match.start(),
                    match.end(),
                    match.group(0),
                )

    # 去重：同一字符区间只保留一个类别（按记录顺序，页码/页眉优先）。
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for element in spans:
        key = (element["segment_index"], element["char_start"], element["char_end"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(element)
    return deduped


# --------------------------------------------------------------------------------------
# block 级元素化与元素构造（13B 移植）
# --------------------------------------------------------------------------------------


def _normalize_block_text(raw: str) -> str:
    """对单个 block 文本做保守规范化：只统一 Unicode/换行并去掉行尾空白。

    刻意不做跨行折叠、不去掉块首尾空行——因为 block 的行结构正是要保住的东西。
    """

    text = unicodedata.normalize("NFC", raw).replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    return re.sub(r"[ \t]+(?=\n)", "", text)


def parse_page_blocks(page: Any) -> list[dict[str, Any]]:
    """把一个 PDF 页解析为 block 级元素（保留 bbox、行结构、页内字符区间）。"""

    blocks: list[dict[str, Any]] = []
    cursor = 0
    for block_index, block in enumerate(page.get_text("blocks")):
        x0, y0, x1, y1, raw_text, _block_no, block_type = block
        text = _normalize_block_text(str(raw_text))
        start = cursor
        cursor += len(text)
        blocks.append(
            {
                "block_index": block_index,
                "char_start": start,
                "char_end": cursor,
                "bbox": [
                    round(float(x0), 2),
                    round(float(y0), 2),
                    round(float(x1), 2),
                    round(float(y1), 2),
                ],
                "raw_text": text,
                # 图像块（block_type=1）不产生文本元素，但保留占位信息供审计。
                "is_text": block_type == 0,
            }
        )
    return blocks


def _block_lines(text: str) -> list[str]:
    """返回 block 内的原始行（保留换行边界，仅丢弃空行用于模式匹配）。"""

    return [line for line in text.split("\n") if line.strip()]


def _digit_ratio(text: str) -> float:
    """统计文本的数值 token 占比，用于识别表格体。"""

    tokens = [tok for tok in re.split(r"\s+", text) if tok]
    if not tokens:
        return 0.0
    numeric = sum(
        1
        for tok in tokens
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?", tok.strip("()[],;"))
    )
    return numeric / len(tokens)


def _line_stats(text: str) -> tuple[int, float]:
    """返回 (非空行数, 平均每行词数)；用于识别"多短行"的表格体块。"""

    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0, 0.0
    words = [len(line.split()) for line in lines]
    return len(lines), sum(words) / len(lines)


def _is_table_body(text: str) -> bool:
    """判定一个紧跟 caption 的块是否为表格体。

    两条互斥信号（任一命中即可）：
      1. 多短行：≥3 个非空行且平均每行 ≤2.5 词（表格头行 / 单元格短值）；
      2. 数值密集：≥4 个 token 且数值占比 ≥0.30（纯数值数据行）。
    正文段落两个信号都不会命中。
    """

    line_count, avg_words = _line_stats(text)
    if line_count >= TABLE_CELL_MIN_LINES and avg_words <= TABLE_CELL_MAX_WORDS_PER_LINE:
        return True
    tokens = [tok for tok in re.split(r"\s+", text) if tok]
    if len(tokens) >= TABLE_NUMERIC_MIN_TOKENS and _digit_ratio(text) >= TABLE_NUMERIC_RATIO:
        return True
    return False


def _classify_block_text(text: str) -> str:
    """按单块文本判定元素类型（不含 table 体延续判定，那需要上下文）。

    优先级：caption > 参考文献条目 > 公式 > 正文。"caption 形态"刻意优先于
    "是否含公式编号"，因为实测 caption 中常出现 ``(1)`` 这类编号。
    """

    lines = _block_lines(text)
    if not lines:
        return "paragraph"
    if any(_FIGURE_CAPTION_FORM.search(line) for line in lines):
        return "figure_caption"
    if any(_TABLE_CAPTION_FORM.search(line) for line in lines):
        return "table_caption"
    # 全大写 "TABLE N" 独占首行（adapointr 等论文的 caption 风格，无冒号）：
    # 必须在参考文献/公式判定之前拦截，否则会被误判成 equation 或 furniture。
    if _TABLE_ALLCAPS_HEADING.match(lines[0].strip()):
        return "table_caption"
    if any(_REFERENCE_ENTRY.match(line) for line in lines):
        return "reference_entry"
    joined = " ".join(lines)
    if LATEX_COMMAND.search(joined):
        # 公式文本层直接保存了 LaTeX 源，无法语义化；单列一类以便审计。
        return "equation"
    if any(_EQUATION_NUMBER_ONLY.fullmatch(line.strip()) for line in lines):
        return "equation"
    # 短块且以数学运算符为主，通常是行间公式。
    if len(joined) <= 200 and len(re.findall(r"[=+\-×÷∑∈≤≥]", joined)) >= 3:
        return "equation"
    return "paragraph"


def _heading_lookup(headings: Sequence[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    """把标题按 (segment_index, char_offset) 建索引，供 block 元素继承标题身份。"""

    return {(h["segment_index"], h["char_offset"]): h for h in headings}


def _element_id(document_id: str, page_number: int, block_index: int) -> str:
    """生成元素稳定 ID：文档 + 页 + 块序号的 SHA-256 前 24 位。"""

    payload = f"{document_id}|p{page_number}|b{block_index}"
    return "el-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _column_of(bbox: Sequence[float], page_width: float) -> str:
    """按 block 的水平位置判定栏属：L / R / F(full-width)。"""

    x0, _y0, x1, _y1 = bbox
    if x0 / page_width >= 0.42 and x1 / page_width <= 1.0 and x0 / page_width > 0.45:
        return "R" if x1 / page_width > 0.75 else "F"
    if x1 / page_width <= 0.58:
        return "L"
    return "F"


def _vertical_band(bbox: Sequence[float], page_height: float) -> str:
    """按 block 的垂直位置判定页内区域：top / body / bottom。"""

    _x0, y0, _x1, y1 = bbox
    if y1 / page_height <= 0.12:
        return "top"
    if y0 / page_height >= 0.88:
        return "bottom"
    return "body"


def _governing_heading_id(
    heading_records: Sequence[dict[str, Any]], segment_index: int, char_offset: int
) -> int | None:
    """返回该位置所属 level<=1 标题在标题列表中的下标（无归属返回 None）。"""

    governing = None
    for index, heading in enumerate(heading_records):
        if heading["segment_index"] < segment_index or (
            heading["segment_index"] == segment_index and heading["char_offset"] <= char_offset
        ):
            if heading["level"] <= 1:
                governing = index
    return governing


def build_structured_document(
    document_id: str,
    file_name: str,
    pdf_path: Path,
    canonical_segments: Sequence[DocumentSegment],
    heading_records: Sequence[dict[str, Any]],
    furniture_spans: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """把一篇 PDF 解析为 DocumentElement 列表（结构化中间表示）。

    每个元素都带：page / bbox（版面身份）、char_start/char_end（页内字符区间）、
    section / subsection / section_path / section_heading_id（章节归属）、
    parent_id（正文元素指向所属标题元素；表格体指向其 table caption）。

    三遍处理：
      1. 类型分类 + 装饰候选标记 + 标题对齐；
      2. 紧跟 table_caption 的"多短行/数值密集"块升级为 table 体（可跨块延续），
         表格头行即使撞上"跨表重复"的装饰候选也必须是 table；
      3. 章节归属 + parent_id + 稳定元素 ID；仍未被表格/标题覆盖的装饰候选判为 furniture。
    """

    heading_index = _heading_lookup(heading_records)
    # 页级装饰区间按 segment_index 建索引，用于把 block 标记为 furniture。
    furniture_by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for span in furniture_spans:
        furniture_by_segment[span["segment_index"]].append(span)

    page_segments = {segment.page_number: index for index, segment in enumerate(canonical_segments)}
    elements: list[dict[str, Any]] = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index in range(pdf.page_count):
            page_number = page_index + 1
            segment_index = page_segments.get(page_number)
            if segment_index is None:
                # 空页不产生元素（与 canonical_segments 的构建口径一致）。
                continue
            page = pdf[page_index]
            blocks = parse_page_blocks(page)
            page_width = float(page.rect.width) or 1.0
            page_height = float(page.rect.height) or 1.0
            page_furniture = furniture_by_segment.get(segment_index, [])

            # 第一遍：类型与装饰标记。
            staged: list[dict[str, Any]] = []
            for ordinal, block in enumerate(blocks):
                if not block["is_text"]:
                    continue
                text = block["raw_text"]
                element_type = _classify_block_text(text)
                # 装饰判定：与页级装饰区间的精确字符区间相交。这里只记"候选"，
                # 最终判定留到表格体升级之后——紧跟 table_caption 的表格头行
                # 会因方法名跨表重复被误判成页眉，必须让表格体判定优先。
                on_furniture = any(
                    block["char_start"] < span["char_end"] and span["char_start"] < block["char_end"]
                    for span in page_furniture
                )
                # 标题判定：block 整体与某个已识别标题的字符起点对齐。
                heading = heading_index.get((segment_index, block["char_start"]))
                if heading is not None and element_type == "paragraph":
                    element_type = "title"
                staged.append(
                    {
                        "block_index": block["block_index"],
                        "ordinal": ordinal,
                        "text": text,
                        "char_start": block["char_start"],
                        "char_end": block["char_end"],
                        "bbox": block["bbox"],
                        "element_type": element_type,
                        "on_furniture": on_furniture,
                        "heading": heading,
                        "column": _column_of(block["bbox"], page_width),
                        "vertical": _vertical_band(block["bbox"], page_height),
                    }
                )

            # 第二遍：紧跟 table_caption 的表格体升级（可跨块延续，直到非表格体块为止）。
            in_table_body = False
            for position, item in enumerate(staged):
                previous = staged[position - 1] if position > 0 else None
                follows_caption = (
                    previous is not None and previous["element_type"] == "table_caption"
                )
                if in_table_body or follows_caption:
                    if _is_table_body(item["text"]):
                        item["element_type"] = "table"
                        item["on_furniture"] = False
                        in_table_body = True
                        continue
                if item["element_type"] == "table":
                    in_table_body = False
                else:
                    in_table_body = False

            # 第三遍准备：对仍标记装饰候选且未被表格/标题覆盖的块，最终判为 furniture。
            for item in staged:
                if item["on_furniture"] and item["element_type"] == "paragraph":
                    item["element_type"] = "furniture"

            # 第三遍：章节归属 + parent_id + 稳定元素 ID。
            caption_stack: dict[str, str] = {}
            section_element_id: str | None = None
            for item in staged:
                resolved = resolve_section(heading_records, segment_index, item["char_start"])
                item["section"] = resolved["section"]
                item["subsection"] = resolved["subsection"]
                item["section_path"] = resolved["section_path"]
                item["section_heading_id"] = _governing_heading_id(
                    heading_records, segment_index, item["char_start"]
                )
                element_id = _element_id(document_id, page_number, item["block_index"])
                item["element_id"] = element_id
                item["page"] = page_number
                item["document_id"] = document_id
                item["file_name"] = file_name

                if item["element_type"] == "title":
                    section_element_id = element_id
                    item["parent_id"] = None
                elif item["element_type"] == "table":
                    item["parent_id"] = caption_stack.get("table")
                elif item["element_type"] in {"figure_caption", "table_caption"}:
                    item["parent_id"] = section_element_id
                    caption_stack[item["element_type"].split("_")[0]] = element_id
                else:
                    item["parent_id"] = section_element_id
                elements.append(item)

    return {
        "document_id": document_id,
        "file_name": file_name,
        "element_count": len(elements),
        "elements": elements,
    }


def remove_furniture_elements(elements: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """元素级装饰移除（13B Step 2 口径）：删除 furniture，保留 section/caption/table/公式。

    这样切分输入流里不再出现页码、arXiv 侧栏、版权与水印。
    """

    return [element for element in elements if element["element_type"] not in FURNITURE_ELEMENT_TYPES]


# --------------------------------------------------------------------------------------
# 生产入口：Document → 结构化元素（hybrid Chunker 的直接输入）
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredParseResult:
    """单篇 PDF 的结构解析结果及可审计统计。"""

    document_id: str
    file_name: str
    # furniture 移除后的入索引元素（13B 元素 dict，字段见 build_structured_document）。
    elements: list[dict[str, Any]] = field(default_factory=list)
    heading_count: int = 0
    removed_element_count: int = 0
    section_attribution_rate: float = 0.0


def parse_structured_document(document: Document, pdf_path: Path) -> StructuredParseResult:
    """对一篇已加载的 PDF 执行完整结构解析（13B Step 1 + Step 2 生产口径）。

    解析在原始页文本的 block 视图上执行（canonical segments = 逐 block 规范化后按序
    拼接，与官方抽取文本产生的标题列表实测逐项一致），不依赖 Cleaner 输出。
    """

    # 自有规范文本：逐 block 规范化后按序拼接，作为标题/装饰检测的坐标系。
    canonical_segments: list[DocumentSegment] = []
    with pymupdf.open(pdf_path) as pdf:
        for page_index in range(pdf.page_count):
            blocks = parse_page_blocks(pdf[page_index])
            text = "".join(block["raw_text"] for block in blocks)
            if text.strip():
                canonical_segments.append(
                    DocumentSegment(content=text, page_number=page_index + 1, metadata={})
                )

    headings = detect_headings(canonical_segments)
    repeated = detect_document_furniture(
        [{"document_id": document.document_id, "segments": canonical_segments}]
    )[document.document_id]
    furniture_spans = page_level_furniture_spans(
        {"document_id": document.document_id, "segments": canonical_segments}, repeated
    )
    structured = build_structured_document(
        document_id=document.document_id,
        file_name=document.file_name,
        pdf_path=pdf_path,
        canonical_segments=canonical_segments,
        heading_records=headings,
        furniture_spans=furniture_spans,
    )
    kept = remove_furniture_elements(structured["elements"])

    # 统计信息供 Gate 审计（section 归属率应与 13B 冻结值同量级）。
    attributed = sum(1 for element in kept if element["section"] is not None)
    return StructuredParseResult(
        document_id=document.document_id,
        file_name=document.file_name,
        elements=kept,
        heading_count=len(headings),
        removed_element_count=structured["element_count"] - len(kept),
        section_attribution_rate=round(attributed / max(1, len(kept)), 6),
    )
