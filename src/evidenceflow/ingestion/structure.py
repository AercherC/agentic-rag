"""PDF structural parsing: block-level elementization, section attribution and page-furniture removal.

Porting layer promoted by Phase 13F: every pure function and decision rule is ported **verbatim** from the archived
experiment scripts so the behaviour matches the frozen artifacts (no threshold or regex may be "improved along the way"
in this file). The source scripts below were deleted in Stage A A4; if a comparison is needed they can be restored from
the git tag ``stage-a-start``:

- ``detect_headings`` / ``resolve_section`` / ``detect_document_furniture`` /
  ``page_level_furniture_spans``: from ``experiments/run_phase13_kb_chunk_audit.py``
  (Phase 13A audit; the rules include the four-layer exclusion derived from observed false positives);
- ``parse_page_blocks`` / ``_classify_block_text`` / ``_is_table_body`` /
  ``build_structured_document`` and element-level furniture removal: from
  ``experiments/run_phase13b_controlled_rebuild.py`` (Phase 13B controlled rebuild Step 1/2).

Why parse at block level rather than on whole-page text: ``"".join(block_text)`` is strictly equal to
``page.get_text()`` on 144 of the 145 controlled-corpus pages, so block elements share the same character coordinate
system as the official extracted text and additionally carry a bbox (information plain text extraction lacks).
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
# Text-structure recognition rules (the same definition as 13A; its regexes are ported directly
# to avoid a second definition).
#
# Key constraint (a Phase 13 measurement): PyMuPDF flattens a two-column page into one text
# stream, so captions often share a line with body text; and the BGE-M3 (XLM-R SentencePiece)
# decode collapses newlines into spaces.
# Therefore caption matching must not be anchored to the line start only, or all that noise is
# missed. Whether the match really sits at the line start is recorded separately
# (at_line_start), to distinguish a "high-confidence caption" from "a citation that merely
# appears in body text".
# --------------------------------------------------------------------------------------

_FIGURE_ANY = re.compile(r"(?:Figure|Fig\.)\s*(\d{1,2})\b")
_FIGURE_CAPTION_FORM = re.compile(r"(?:Figure|Fig\.)\s*(\d{1,2})\s*[:.]\s*[A-Z]")
_TABLE_ANY = re.compile(r"(?:Table|Tab\.)\s*(\d{1,2})\b")
_TABLE_CAPTION_FORM = re.compile(r"(?:Table|Tab\.)\s*(\d{1,2})\s*[:.]\s*[A-Z]")
_ALGORITHM_BLOCK = re.compile(r"Algorithm\s*(\d{1,2})\b")
# Raw LaTeX command leakage: some PDFs store the LaTeX source directly in the formula text layer,
# which cannot be made semantic after extraction.
LATEX_COMMAND = re.compile(r"\\[a-zA-Z]{2,}")
# Reference entry: the "[N]" form at line start (used by 13B when matching line by line during
# classification).
_REFERENCE_ENTRY = re.compile(r"(?m)^[ \t]*\[\d{1,3}\]")
_EQUATION_NUMBER = re.compile(r"(?m)^[ \t]*\(\d{1,3}\)[ \t]*$")
_ARXIV_HEADER = re.compile(r"(?im)^\s*arXiv:\d{4}\.\d{4,5}v\d+")
_DOI = re.compile(r"(?i)\bdoi:\s*10\.\d{4,9}/\S+")
_COPYRIGHT = re.compile(r"(?i)(copyright|©|all rights reserved|permission to make digital)")
_CONFERENCE_HEADER = re.compile(
    r"(?i)(proceedings of|conference on|international conference|"
    r"computer vision and pattern recognition|neural information processing)"
)

# Section headings: numbered ("1" / "2.1" / "3.2.1") and unnumbered named ones.
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

# An "N. Title" form inside a section may also be a list item; these patterns exclude lines that
# are obviously not headings.
# Citation features and math symbols: used to exclude "reference entries / table numeric rows"
# being misjudged as section headings.
_CITATION_LIKE = re.compile(r"(?i)\bet al\b|,\s*[A-Z]\.|\(\d{4}\)|\b(?:19|20)\d{2}\b")
_HEADING_MATH = re.compile(r"[×≤≥↑↓≈∈∑∏±]")

_HEADING_REJECT = re.compile(
    r"(?i)(^|[,\s])(?:table|figure|fig\.|eq\.|equation|algorithm)\s*\d"
    r"|[;=<>]|\b(?:is|are|was|were|can|we|our|the)\b"
)

# Exact patterns for page-level furniture (complementary to the "string repeated across pages"
# definition; a page number that changes page by page must be located separately).
_PAGE_NUMBER_LINE = re.compile(r"^[ \t]*\d{1,5}[ \t]*$")
_OPEN_ACCESS_WATERMARK = re.compile(
    r"(?i)this\s+\w+\s+paper is the open access version"
    r"|provided by the computer vision foundation"
    r"|final published version of the proceedings"
    r"|is available on ieee xplore"
)

# --------------------------------------------------------------------------------------
# 13B element classification constants (regexes shared with 13A; thresholds calibrated on
# Phase 13A/13B measurements).
# --------------------------------------------------------------------------------------

# Controlled types of structural elements (13B definition; the hybrid Chunker consumes only the
# elements left after furniture removal).
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
# Furniture types deleted by element-level cleaning.
FURNITURE_ELEMENT_TYPES = {"furniture"}

# Equation-number lines: the whole line is just "(N)".
_EQUATION_NUMBER_ONLY = re.compile(r"\(\d{1,3}\)")
# All-caps "TABLE N" caption style occupying the first line (adapointr and similar, with no colon).
_TABLE_ALLCAPS_HEADING = re.compile(r"TABLE\s+\d{1,2}\b")

# Table-body detection: a block immediately following a caption counts as table body if it
# satisfies either "many short lines (<=2 words per line)" or "numeric density".
# A table header row (such as "Methods / Plane / Cabinet") is all labels with a low numeric
# share, so only the "many short lines" signal can detect it - this is the direct fix for the
# "table row/column relations are lost" finding already proven in Phase 13A.
TABLE_CELL_MAX_WORDS_PER_LINE = 2.5
TABLE_CELL_MIN_LINES = 3
TABLE_NUMERIC_RATIO = 0.30
TABLE_NUMERIC_MIN_TOKENS = 4


# --------------------------------------------------------------------------------------
# Section heading detection and attribution (ported from 13A)
# --------------------------------------------------------------------------------------


def detect_headings(segments: Sequence[DocumentSegment]) -> list[dict[str, Any]]:
    """Locate section headings in the text and return them in document order.

    It recognizes only "numbered headings on their own line" and "named headings on their own line". To avoid misjudging
    body text, tables or references as sections, four exclusion layers are applied (each one comes from a false positive
    observed in Phase 13):

    1. Numbered headings: major number <= 20, at most 8 words in the title, no year/citation feature/math symbol, and no
       trailing punctuation; within one document a level-1 number must not repeat (a table cell was actually read as
       "4 Note that, ...").
    2. Named headings: must not be all lowercase, and must occur only once in the whole document
       ("Methods" was actually a Table column header appearing 7 times, all excluded).
    3. When the numbered and named rules hit the same position, keep only the numbered heading.
    4. A heading is only valid inside the Segment it belongs to (a Segment boundary is a PDF page boundary).
    """

    numbered: list[dict[str, Any]] = []
    named: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        content = segment.content
        for match in _NUMBERED_HEADING.finditer(content):
            number, title = match.group(1), match.group(2).strip()
            major = int(number.split(".")[0])
            # Rule 1: filter on numbering and title shape.
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
            # Rule 2: an all-lowercase line is almost never a section heading.
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

    # Rule 1b: within one document a level-1 number must not repeat.
    seen_major: set[str] = set()
    kept_numbered: list[dict[str, Any]] = []
    for heading in numbered:
        if heading["level"] == 1:
            if heading["number"] in seen_major:
                continue
            seen_major.add(heading["number"])
        kept_numbered.append(heading)

    # Rule 2b: a named heading must occur uniquely in the whole document, otherwise it is treated
    # as a table column
    # header or header noise.
    title_counts = Counter(heading["title"].lower() for heading in named)
    kept_named = [heading for heading in named if title_counts[heading["title"].lower()] == 1]

    headings = kept_numbered + kept_named
    # Rules 3 / 4: sort in document order and deduplicate numbered/named hits at the same position.
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
    """Return the section hierarchy path that owns the given character position.

    Rule: the nearest numbered/named heading before that position is the section; the nearest deeper heading below it is
    the subsection. A heading is only valid inside its own Segment (a Segment boundary is a PDF page boundary, so a
    heading never continues across pages).
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
# Page furniture detection (ported from 13A)
# --------------------------------------------------------------------------------------


def detect_document_furniture(documents: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Detect the repeated page furniture of each PDF (header/footer/arXiv/DOI/copyright/conference banner/page number).

    Method: count the lines that, after normalization (digits folded to ``#``), appear on at least 3 pages and on no less
    than 30% of the total pages; those become "repeated furniture lines", used later for the header/footer contamination
    check.
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
        # Also add pattern-based furniture (arXiv banner / DOI / copyright / conference banner /
        # bare page number).
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
    """Return the exact character spans of page-level furniture elements: page number / repeated header-footer / arXiv sidebar / copyright / watermark.

    Decision rules (deterministic and recomputable):
      - Page number: the first non-empty line or the last non-empty line of the page, consisting only of 1-5 digits.
        Only these two lines are examined, so that a table numeric row adjacent to the footer is not misjudged as a page number.
      - Repeated header/footer: the whole line, normalized (digits -> #), hits the result set of
        ``detect_document_furniture``. Strings shaped like reference entries / table rows (containing a year, "pp.",
        "et al", etc.) are excluded first.
      - arXiv sidebar / copyright / open-access watermark: matched by pattern, and the exact span is recorded.
    """

    spans: list[dict[str, Any]] = []
    # Keep only repeated strings containing letters: digit folding would normalize page numbers or
    # section numbers such as "1"/"3.1" into "#"/"#.#", which are not headers/footers, and page
    # numbers are already covered more precisely by the PAGE_NUMBER rule.
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
        # Record the absolute character offset of every line, so the spans can be intersected
        # directly with block spans.
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

        # Repeated header/footer: match per line at page level and record the exact span on a hit.
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

    # Deduplicate: keep only one category per character span (in record order, page number/header
    # first).
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
# Block-level elementization and element construction (ported from 13B)
# --------------------------------------------------------------------------------------


def _normalize_block_text(raw: str) -> str:
    """Apply conservative normalization to a single block's text: only unify Unicode/newlines and strip trailing
    whitespace.

    Deliberately no cross-line folding and no removal of leading/trailing blank lines - the line structure of a block is
    exactly what must be preserved.
    """

    text = unicodedata.normalize("NFC", raw).replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    return re.sub(r"[ \t]+(?=\n)", "", text)


def parse_page_blocks(page: Any) -> list[dict[str, Any]]:
    """Parse one PDF page into block-level elements (keeping bbox, line structure and in-page character spans)."""

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
                # Image blocks (block_type=1) produce no text element, but placeholder information
                # is kept for auditing.
                "is_text": block_type == 0,
            }
        )
    return blocks


def _block_lines(text: str) -> list[str]:
    """Return the raw lines inside the block (keeping newline boundaries; only empty lines are dropped for pattern matching)."""

    return [line for line in text.split("\n") if line.strip()]


def _digit_ratio(text: str) -> float:
    """Compute the share of numeric tokens in the text, used to identify table bodies."""

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
    """Return (non-empty line count, average words per line); used to identify "many short lines" table-body blocks."""

    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0, 0.0
    words = [len(line.split()) for line in lines]
    return len(lines), sum(words) / len(lines)


def _is_table_body(text: str) -> bool:
    """Decide whether a block immediately following a caption is a table body.

    Two mutually independent signals (either one suffices):
      1. Many short lines: >=3 non-empty lines averaging <=2.5 words per line (table header rows / short cell values);
      2. Numeric density: >=4 tokens with a numeric share >=0.30 (purely numeric data rows).
    A body-text paragraph triggers neither signal.
    """

    line_count, avg_words = _line_stats(text)
    if line_count >= TABLE_CELL_MIN_LINES and avg_words <= TABLE_CELL_MAX_WORDS_PER_LINE:
        return True
    tokens = [tok for tok in re.split(r"\s+", text) if tok]
    if len(tokens) >= TABLE_NUMERIC_MIN_TOKENS and _digit_ratio(text) >= TABLE_NUMERIC_RATIO:
        return True
    return False


def _classify_block_text(text: str) -> str:
    """Decide the element type from a single block's text (excluding the table-body continuation rule, which needs context).

    Priority: caption > reference entry > equation > body text. The "caption shape" check is deliberately given priority
    over "whether it contains an equation number", because captions were actually observed to contain numbers such as
    ``(1)``.
    """

    lines = _block_lines(text)
    if not lines:
        return "paragraph"
    if any(_FIGURE_CAPTION_FORM.search(line) for line in lines):
        return "figure_caption"
    if any(_TABLE_CAPTION_FORM.search(line) for line in lines):
        return "table_caption"
    # An all-caps "TABLE N" occupying the first line (the caption style of papers such as
    # adapointr, with no colon):
    # this must be intercepted before the reference/equation checks, otherwise it is misjudged as
    # equation or furniture.
    if _TABLE_ALLCAPS_HEADING.match(lines[0].strip()):
        return "table_caption"
    if any(_REFERENCE_ENTRY.match(line) for line in lines):
        return "reference_entry"
    joined = " ".join(lines)
    if LATEX_COMMAND.search(joined):
        # The formula text layer stores the LaTeX source directly and cannot be made semantic; keep
        # it in its own class
        # for auditing.
        return "equation"
    if any(_EQUATION_NUMBER_ONLY.fullmatch(line.strip()) for line in lines):
        return "equation"
    # A short block dominated by math operators is usually a display equation.
    if len(joined) <= 200 and len(re.findall(r"[=+\-×÷∑∈≤≥]", joined)) >= 3:
        return "equation"
    return "paragraph"


def _heading_lookup(headings: Sequence[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    """Index the headings by (segment_index, char_offset) so block elements can inherit heading identity."""

    return {(h["segment_index"], h["char_offset"]): h for h in headings}


def _element_id(document_id: str, page_number: int, block_index: int) -> str:
    """Build a stable element ID: the first 24 hex chars of the SHA-256 of document + page + block ordinal."""

    payload = f"{document_id}|p{page_number}|b{block_index}"
    return "el-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _column_of(bbox: Sequence[float], page_width: float) -> str:
    """Decide the column from the block's horizontal position: L / R / F (full-width)."""

    x0, _y0, x1, _y1 = bbox
    if x0 / page_width >= 0.42 and x1 / page_width <= 1.0 and x0 / page_width > 0.45:
        return "R" if x1 / page_width > 0.75 else "F"
    if x1 / page_width <= 0.58:
        return "L"
    return "F"


def _vertical_band(bbox: Sequence[float], page_height: float) -> str:
    """Decide the in-page region from the block's vertical position: top / body / bottom."""

    _x0, y0, _x1, y1 = bbox
    if y1 / page_height <= 0.12:
        return "top"
    if y0 / page_height >= 0.88:
        return "bottom"
    return "body"


def _governing_heading_id(
    heading_records: Sequence[dict[str, Any]], segment_index: int, char_offset: int
) -> int | None:
    """Return the index in the heading list of the level<=1 heading owning this position (None when there is none)."""

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
    """Parse one PDF into a list of DocumentElement entries (the structured intermediate representation).

    Every element carries: page / bbox (layout identity), char_start/char_end (in-page character span),
    section / subsection / section_path / section_heading_id (section attribution), and
    parent_id (body elements point at their heading element; a table body points at its table caption).

    Three passes:
      1. type classification + furniture candidate marking + heading alignment;
      2. a "many short lines / numeric density" block right after a table_caption is upgraded to table body (the body may
         continue across blocks), and a table header row must stay a table even if it collides with a "repeated across
         tables" furniture candidate;
      3. section attribution + parent_id + stable element ID; furniture candidates still not covered by table/heading are
         finally judged furniture.
    """

    heading_index = _heading_lookup(heading_records)
    # Index the page-level furniture spans by segment_index, so blocks can be marked as furniture.
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
                # No elements for an empty page (consistent with how canonical_segments is built).
                continue
            page = pdf[page_index]
            blocks = parse_page_blocks(page)
            page_width = float(page.rect.width) or 1.0
            page_height = float(page.rect.height) or 1.0
            page_furniture = furniture_by_segment.get(segment_index, [])

            # First pass: types and furniture marking.
            staged: list[dict[str, Any]] = []
            for ordinal, block in enumerate(blocks):
                if not block["is_text"]:
                    continue
                text = block["raw_text"]
                element_type = _classify_block_text(text)
                # Furniture check: intersect the exact character span with the page-level furniture
                # spans. Only a
                # "candidate" is marked here; the final decision waits until after table-body
                # upgrade, because a table
                # header row right after a table_caption would otherwise be misjudged as a page
                # header when method names
                # repeat across tables - the table-body rule must take priority.
                on_furniture = any(
                    block["char_start"] < span["char_end"] and span["char_start"] < block["char_end"]
                    for span in page_furniture
                )
                # Heading check: the block as a whole aligns with the character start of an already
                # detected heading.
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

            # Second pass: upgrade table bodies right after a table_caption (the body may continue
            # across blocks until a
            # non-table-body block appears).
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

            # Preparation for the third pass: blocks still marked as furniture candidates and not
            # covered by
            # table/heading are finally judged furniture.
            for item in staged:
                if item["on_furniture"] and item["element_type"] == "paragraph":
                    item["element_type"] = "furniture"

            # Third pass: section attribution + parent_id + stable element ID.
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
    """Element-level furniture removal (13B Step 2 definition): delete furniture and keep section/caption/table/equation.

    The chunking input stream therefore no longer contains page numbers, arXiv sidebars, copyright lines or watermarks.
    """

    return [element for element in elements if element["element_type"] not in FURNITURE_ELEMENT_TYPES]


# --------------------------------------------------------------------------------------
# Production entry point: Document -> structured elements (the direct input of the hybrid Chunker)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredParseResult:
    """Structural parse result of a single PDF plus auditable statistics."""

    document_id: str
    file_name: str
    # Indexed elements after furniture removal (13B element dicts; fields documented on
    # build_structured_document).
    elements: list[dict[str, Any]] = field(default_factory=list)
    heading_count: int = 0
    removed_element_count: int = 0
    section_attribution_rate: float = 0.0


def parse_structured_document(document: Document, pdf_path: Path) -> StructuredParseResult:
    """Run the full structural parse on one already loaded PDF (the production definition of 13B Step 1 + Step 2).

    Parsing runs on the block view of the raw page text (canonical segments = each block normalized then concatenated in
    order, whose heading list was measured to match the one produced from the officially extracted text item by item); it
    does not depend on Cleaner output.
    """

    # Our own canonical text: each block normalized then concatenated in order, serving as the
    # coordinate system for
    # heading/furniture detection.
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

    # The statistics feed the Gate audit (the section attribution rate should be of the same order
    # as the frozen 13B value).
    attributed = sum(1 for element in kept if element["section"] is not None)
    return StructuredParseResult(
        document_id=document.document_id,
        file_name=document.file_name,
        elements=kept,
        heading_count=len(headings),
        removed_element_count=structured["element_count"] - len(kept),
        section_attribution_rate=round(attributed / max(1, len(kept)), 6),
    )
