"""Phase 14A-B 受控语料生成器（子进程 runner，运行于隔离沙盒）。

复用 14A-A build_corpus_arm 的构造函数（W1..W8 / X1..X10 同一写法保证可比），
扩展支持 14B 新 block kind：
- DOCX：``merged_table``（=table+merges）、table 的 ``nested``（cell 内嵌套子表）
- XLSX：``number_formats``、``auto_filter``、``kv_pairs``

运行：python -m phase14a.runners.build_corpus14b --out <dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from phase14a import corpus14a, corpus14b, sandbox14a
from phase14a.runners.build_corpus_arm import _build_docx_table, _build_xlsx

from docx import Document
from docx.oxml import ns as _docx_ns

# python-docx nsmap 缺少 wps 前缀（wordprocessingShape），导致 14A-A 的 _add_textbox
# 一进 try 即 KeyError → 永久退化为 1x1 表格 callout（W7 浮动文本框从未真实生成）。
# 注册后真实 txbxContent 路径可用；callout 仅作兜底保留。
_docx_ns.nsmap.setdefault("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")


def _build_docx14b(case: dict, out_path: Path) -> None:
    """在 14A-A 构造器之上扩展 merged_table / nested table。"""

    doc = Document()
    footnotes: list[str] = []
    for block in case["blocks"]:
        kind = block["kind"]
        if kind == "merged_table":
            # merged_table 与 table 同构（headers/rows/merges）
            _build_docx_table(doc, {**block, "kind": "table"})
        elif kind == "table" and block.get("nested"):
            _build_docx_table(doc, block)
            # 在最后一个表格的首个数据 cell 内嵌入子表（XML 级，python-docx 无公开 API）
            tbl = doc.tables[-1]
            cell = tbl.cell(1, 0)
            nested = block["nested"]
            nt = cell.add_table(rows=1 + len(nested["rows"]), cols=len(nested["headers"][0]))
            for ci, h in enumerate(nested["headers"][0]):
                nt.cell(0, ci).text = str(h)
            for ri, row in enumerate(nested["rows"], start=1):
                for ci, v in enumerate(row):
                    nt.cell(ri, ci).text = str(v)
        else:
            # 其余 kind 委托 14A-A 构造器（title/heading/paragraph/list/table/
            # caption/image/footnote/header/footer/page_break/textbox/bad_heading）
            _build_docx(block, doc, footnotes)
    doc.save(str(out_path))
    if footnotes:
        from phase14a.runners.build_corpus_arm import _inject_footnotes
        _inject_footnotes(out_path, footnotes)


def _build_docx(block: dict, doc: Document, footnotes: list[str]) -> None:
    """逐 block 构造（从 build_corpus_arm._build_docx 抽出的单 block 逻辑）。"""

    from docx.shared import Pt
    kind = block["kind"]
    if kind == "title":
        doc.add_heading(block["text"], level=0)
    elif kind == "heading":
        doc.add_heading(block["text"], level=block.get("level", 1))
    elif kind == "paragraph":
        doc.add_paragraph(block["text"])
    elif kind == "list_item":
        doc.add_paragraph(block["text"], style=block.get("style", "List Bullet"))
    elif kind == "table":
        _build_docx_table(doc, block)
    elif kind == "caption":
        doc.add_paragraph(block["text"], style="Caption")
    elif kind == "image":
        from phase14a.runners.build_corpus_arm import _png_stream
        from docx.shared import Inches
        doc.add_picture(_png_stream(), width=Inches(block.get("width", 2.0)))
    elif kind == "footnote":
        footnotes.append(block["text"])
    elif kind == "header":
        sec = doc.sections[0]
        (sec.header.paragraphs[0] if sec.header.paragraphs else sec.header.add_paragraph()).text = block["text"]
    elif kind == "footer":
        sec = doc.sections[0]
        (sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()).text = block["text"]
    elif kind == "page_break":
        doc.add_page_break()
    elif kind == "textbox":
        from phase14a.runners.build_corpus_arm import _add_textbox
        _add_textbox(doc, block["text"])
    elif kind == "bad_heading":
        p = doc.add_paragraph()
        run = p.add_run(block.get("number", "") + (" " if block.get("number") else "") + block["text"])
        run.bold = block.get("bold", True)
        if block.get("size"):
            run.font.size = Pt(block["size"])
    else:
        doc.add_paragraph(block.get("text", ""))


def _build_xlsx14b(case: dict, out_path: Path) -> None:
    """在 14A-A 构造器之上扩展 number_formats / auto_filter / kv_pairs。

    顺序敏感：必须先 openpyxl 补结构、再补公式 cached <v> 注入——openpyxl
    load/save 往返会丢弃已注入的 cached 值，且会丢弃 chart（本路径只用于
    无 chart 的 14B 新用例，X10 的 chart 用例不走此分支）。
    """

    from openpyxl import load_workbook
    from phase14a.runners.build_corpus_arm import _inject_cached_values

    # 去掉 14B 新增键后交给 14A-A 构造器（notes/tables/hidden/named/formula/chart）
    base_case = {**case, "sheets": [{k: v for k, v in s.items()
                                     if k not in ("number_formats", "auto_filter", "kv_pairs")}
                                    for s in case["sheets"]]}
    _build_xlsx(base_case, out_path)

    need_patch = any(
        s.get("auto_filter") or s.get("number_formats") or s.get("kv_pairs")
        for s in case["sheets"]
    )
    if need_patch:
        wb = load_workbook(str(out_path))
        for sheet in case["sheets"]:
            ws = wb[sheet["name"]]
            for ref, fmt in (sheet.get("number_formats") or {}).items():
                ws[ref].number_format = fmt
            af = sheet.get("auto_filter")
            if af:
                ws.auto_filter.ref = af["ref"]
            for kv in sheet.get("kv_pairs", []):
                ws[kv["cell"]] = kv["text"]
        wb.save(str(out_path))

    # openpyxl 往返后重新注入 formula cached 值（保持 formula+cached 双表示）
    injections: dict[str, dict[str, object]] = {}
    for sheet in case["sheets"]:
        for table in sheet.get("tables", []):
            fc = table.get("formula_cells")
            if fc:
                injections.setdefault(sheet["name"], {}).update(
                    {k: v[1] for k, v in fc.items()}
                )
    if injections:
        _inject_cached_values(out_path, injections)


def main() -> int:
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else (
        sandbox14a.ARTIFACT_ROOT / "phase14a_b" / "corpus"
    )
    docx_dir = out_dir / "controlled_docx"
    xlsx_dir = out_dir / "controlled_xlsx"
    sandbox14a.ensure_dir(docx_dir)
    sandbox14a.ensure_dir(xlsx_dir)

    docx_cases = corpus14a.DOCX_CASES + corpus14b.DOCX_CASES_B
    xlsx_cases = corpus14a.XLSX_CASES + corpus14b.XLSX_CASES_B

    built = {"docx": [], "xlsx": []}
    for case in docx_cases:
        p = docx_dir / f"{case['case_id']}.docx"
        _build_docx14b(case, p)
        built["docx"].append(p.name)
    for case in xlsx_cases:
        p = xlsx_dir / f"{case['case_id']}.xlsx"
        _build_xlsx14b(case, p)
        built["xlsx"].append(p.name)

    summary = {
        "artifact_type": "phase14a_b_corpus_build",
        "docx_count": len(built["docx"]),
        "xlsx_count": len(built["xlsx"]),
        "docx": built["docx"],
        "xlsx": built["xlsx"],
    }
    sandbox14a.save_json(summary, out_dir / "build_summary.json")
    print(sandbox14a.sha256_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
