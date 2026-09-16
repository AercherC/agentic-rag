"""Phase 14A-B DOCX native 抽取 + 清洗 runner（子进程，B0/B1 共用抽取层）。

抽取层把 python-docx 对象转成纯数据 native 流（cleaners14b 的输入）：
- 文档顺序：按 OOXML body 顺序遍历 <w:p> 与 <w:tbl>（禁止 paragraphs+tables 拼接，§13）
- 段落携带 inference features：style / bold / size_pt / numPr
- 表格：cell matrix（含合并展开）、merged_regions（同 tc 归组）、嵌套子表（XML 级抽取）
- header/footer/footnote/textbox/page break

运行：python -m phase14a.runners.docx_clean14b --docx <f> --out <f.json> --arm B0|B1
"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.document import Document as _Doc
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from phase14a import cleaners14b, sandbox14a
from phase14a.runners.docx_native_arm import _read_footnotes, _read_textboxes, iter_block_items

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _para_features(p: Paragraph) -> dict:
    """段落推理特征：样式 / 首个加粗 / 最大字号 / numPr。"""

    bold = any(bool(run.bold) for run in p.runs if run.bold is not None)
    size_pt = None
    for run in p.runs:
        if run.font.size is not None:
            size_pt = max(size_pt or 0.0, run.font.size.pt)
    numbering = p._p.find(qn("w:pPr")) is not None and p._p.find(qn("w:pPr")).find(qn("w:numPr")) is not None
    return {"bold": bold, "size_pt": size_pt, "numbering": numbering}


def _extract_table(tbl: Table, table_index: int) -> dict:
    rows = tbl.rows
    n_rows = len(rows)
    n_cols = len(rows[0].cells) if n_rows else 0
    cells: list[list[str]] = []
    for row in rows:
        cells.append([(cell.text or "") for cell in row.cells])

    # merged regions：从 OOXML 原生 gridSpan / vMerge 属性重建（不依赖对象身份比较，
    # 避免 lxml 代理 id 复用造成的假分组）
    merged_regions = []
    open_vmerges: dict[int, dict] = {}  # start_col -> {start_row, last_row, grid_span}
    for r, tr in enumerate(tbl._tbl.findall(qn("w:tr"))):
        c = 0
        for tc in tr.findall(qn("w:tc")):
            tcPr = tc.find(qn("w:tcPr"))
            grid_span, vmerge = 1, None
            if tcPr is not None:
                gs = tcPr.find(qn("w:gridSpan"))
                if gs is not None:
                    grid_span = int(gs.get(qn("w:val")) or 1)
                vm = tcPr.find(qn("w:vMerge"))
                if vm is not None:
                    vmerge = vm.get(qn("w:val")) or "continue"
            if vmerge == "restart":
                open_vmerges[c] = {"start_row": r, "last_row": r, "grid_span": grid_span}
            elif vmerge == "continue" and c in open_vmerges:
                open_vmerges[c]["last_row"] = r  # 纵向延续
            elif grid_span > 1:
                merged_regions.append({"anchor": f"R{r + 1}C{c + 1}",
                                       "row_span": 1, "col_span": grid_span})
            c += grid_span
    # 收尾：纵向 merge 的实际跨度 = last_row - start_row + 1（支持中途结束）
    for c, info in open_vmerges.items():
        if info["last_row"] > info["start_row"] or info["grid_span"] > 1:
            merged_regions.append({"anchor": f"R{info['start_row'] + 1}C{c + 1}",
                                   "row_span": info["last_row"] - info["start_row"] + 1,
                                   "col_span": info["grid_span"]})
    merged_regions.sort(key=lambda m: (m["anchor"], m["row_span"], m["col_span"]))

    # 嵌套子表（XML 级，跨 python-docx 版本安全）
    nested = []
    for row_el in tbl._tbl.findall(qn("w:tr")):
        for tc_el in row_el.findall(qn("w:tc")):
            for sub in tc_el.findall(qn("w:tbl")):
                sub_rows = []
                for tr in sub.findall(qn("w:tr")):
                    sub_cells = []
                    for tc in tr.findall(qn("w:tc")):
                        sub_cells.append("".join(t.text or "" for t in tc.iter(qn("w:t"))))
                    sub_rows.append(sub_cells)
                nested.append({
                    "rows": len(sub_rows),
                    "cols": len(sub_rows[0]) if sub_rows else 0,
                    "cells": sub_rows,
                })

    style = None
    try:
        style = tbl.style.name if tbl.style else None
    except Exception:
        style = None
    return {"kind": "table", "table_index": table_index, "rows": n_rows, "cols": n_cols,
            "cells": cells, "merged_regions": merged_regions, "nested": nested,
            "style": style}


def extract_native(docx_path: Path) -> dict:
    doc = Document(str(docx_path))
    body: list[dict] = []
    page_breaks: list[int] = []
    pid = 0
    table_idx = -1
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            feat = _para_features(block)
            has_pic = "a:blip" in block._p.xml or "pic:pic" in block._p.xml
            style = (block.style.name if block.style else "Normal") or "Normal"
            body.append({"kind": "paragraph", "para_index": pid, "text": block.text or "",
                         "style": style, "has_pic": has_pic, "is_textbox": False, **feat})
            if "w:br" in block._p.xml and 'type="page"' in block._p.xml:
                page_breaks.append(pid)
            pid += 1
        elif isinstance(block, Table):
            table_idx += 1
            body.append(_extract_table(block, table_idx))

    # 浮动 textbox：作为 body 内的独立段落记录（保持 14A-A W7 语义）
    textboxes = _read_textboxes(docx_path)
    for bi, text in enumerate(textboxes):
        body.append({"kind": "paragraph", "para_index": pid + bi, "text": text,
                     "style": "Textbox", "bold": False, "size_pt": None,
                     "numbering": False, "has_pic": False, "is_textbox": True})

    return {
        "doc_id": docx_path.stem,
        "source_file": docx_path.name,
        "source_sha256": sandbox14a.sha256_file(docx_path),
        "body": body,
        "headers": [p.text for sec in doc.sections for p in sec.header.paragraphs if p.text.strip()],
        "footers": [p.text for sec in doc.sections for p in sec.footer.paragraphs if p.text.strip()],
        "footnotes": _read_footnotes(docx_path),
        "page_breaks": page_breaks,
    }


def main() -> int:
    docx_path = Path(sys.argv[sys.argv.index("--docx") + 1])
    out_path = Path(sys.argv[sys.argv.index("--out") + 1])
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "B0"
    native = extract_native(docx_path)
    document = cleaners14b.clean_docx(native, semantic=(arm == "B1"))
    sandbox14a.save_json(document, out_path)
    print(f"DOCX-{arm} parsed {docx_path.name}: {document['element_count']} elements, "
          f"log={len(document['cleaning_log'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
