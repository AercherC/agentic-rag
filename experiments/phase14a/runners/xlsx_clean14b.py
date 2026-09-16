"""Phase 14A-B XLSX native 抽取 + 清洗 runner（子进程，B0/B1 共用抽取层）。

抽取层把 openpyxl 对象转成纯数据 native 流（cleaners14b 的输入）：
- 两次只读加载：data_only=False 取 formula，data_only=True 取 cached（§14 禁止自算）
- 每个非空 cell：coord / raw / formula / cached / data_type / number_format / hidden
- named tables / merged ranges（含 anchor）/ auto filter / chart·image presence

运行：python -m phase14a.runners.xlsx_clean14b --xlsx <f> --out <f.json> --arm B0|B1
"""

from __future__ import annotations

import sys
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from phase14a import cleaners14b, sandbox14a


def extract_native(xlsx_path: Path) -> dict:
    wb_f = load_workbook(str(xlsx_path), data_only=False)
    wb_v = load_workbook(str(xlsx_path), data_only=True)
    sheets = []
    for order, sheet_name in enumerate(wb_f.sheetnames):
        ws_f = wb_f[sheet_name]
        ws_v = wb_v[sheet_name]
        cells = []
        for row in ws_f.iter_rows():
            for cell in row:
                if cell.value is None:
                    continue
                coord = cell.coordinate
                is_formula = cell.data_type == "f"
                # formula 与 cached 分开取，绝不互相覆盖（X-G2）
                cached = ws_v[coord].value if is_formula else None
                hidden_row = bool(ws_f.row_dimensions[cell.row].hidden) if cell.row in ws_f.row_dimensions else False
                col_letter = get_column_letter(cell.column)
                hidden_col = bool(ws_f.column_dimensions[col_letter].hidden) if col_letter in ws_f.column_dimensions else False
                cells.append({
                    "coord": coord, "row": cell.row, "col": cell.column,
                    "raw_value": cell.value,
                    "formula": cell.value if is_formula else None,
                    "cached": cached,
                    "data_type": cell.data_type,
                    "number_format": cell.number_format,
                    "hidden_row": hidden_row, "hidden_col": hidden_col,
                })
        merged = []
        anchor_cells = {}
        for m in ws_f.merged_cells.ranges:
            merged.append(str(m))
            # anchor = merge 左上角 cell
            anchor_cells[str(m)] = f"{get_column_letter(m.bounds[0])}{m.bounds[1]}"
        # 隐藏行/列作为 sheet 级事实（空行/空列的隐藏无法通过 cell flag 证明）
        hidden_rows = sorted(r for r, dim in ws_f.row_dimensions.items() if dim.hidden)
        hidden_cols = sorted(k for k, dim in ws_f.column_dimensions.items() if dim.hidden)
        named = []
        for tname, tbl in ws_f.tables.items():
            ref = tbl.ref if hasattr(tbl, "ref") else str(tbl)
            named.append({"name": tname, "ref": ref})
        # chart / image：presence-only（14A-B 不解析视觉语义）
        charts = []
        for ch in getattr(ws_f, "_charts", []) or []:
            anchor = getattr(ch, "anchor", None)
            anchor_desc = None
            frm = getattr(anchor, "_from", None)
            if frm is not None:
                anchor_desc = f"{get_column_letter(frm.col + 1)}{frm.row + 1}"
            charts.append({"chart_type": type(ch).__name__, "anchor": anchor_desc})
        images = []
        for img in getattr(ws_f, "_images", []) or []:
            anchor = getattr(img, "anchor", None)
            frm = getattr(anchor, "_from", None)
            images.append({"anchor": f"{get_column_letter(frm.col + 1)}{frm.row + 1}" if frm is not None else None})
        sheets.append({
            "name": sheet_name, "order": order, "state": ws_f.sheet_state,
            "merged_ranges": merged, "anchor_cells": anchor_cells,
            "hidden_rows": hidden_rows, "hidden_cols": hidden_cols,
            "auto_filter_ref": ws_f.auto_filter.ref if ws_f.auto_filter and ws_f.auto_filter.ref else None,
            "cells": cells, "named_tables": named, "charts": charts, "images": images,
        })
    wb_f.close()
    wb_v.close()
    return {
        "doc_id": xlsx_path.stem,
        "source_file": xlsx_path.name,
        "source_sha256": sandbox14a.sha256_file(xlsx_path),
        "sheets": sheets,
    }


def main() -> int:
    xlsx_path = Path(sys.argv[sys.argv.index("--xlsx") + 1])
    out_path = Path(sys.argv[sys.argv.index("--out") + 1])
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "B0"
    native = extract_native(xlsx_path)
    document = cleaners14b.clean_xlsx(native, semantic=(arm == "B1"))
    sandbox14a.save_json(document, out_path)
    print(f"XLSX-{arm} parsed {xlsx_path.name}: {document['element_count']} elements, "
          f"log={len(document['cleaning_log'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
