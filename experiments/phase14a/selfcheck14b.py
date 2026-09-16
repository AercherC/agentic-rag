"""Phase 14A-B 语料自证（解析器无关）——含对 selfcheck14a 四个测量缺陷的修正。

复用 14A-A 纪律：不用候选 parser 自证 Gold，直接读 OOXML。语料生成器（build_corpus_arm /
build_corpus14b）与 14A-A 完全同源，故 14A-A 冻结产物中的自证失败经逐项归因全部为
**测量缺陷而非语料缺陷**。本模块修正以下四项并保留 14A-A 代码原样（冻结出处不动）：

1. ``structural_ok`` 用 ``v is True`` 判定，把合法的 float 指标（footnote_text_recall=1.0）
   误判为结构失败（W5/W12 假 FAIL）→ 改为 bool 或 ≥阈值数值均可。
2. ``formula_dual_*`` 用 ``<c r="C2">`` 字面正则，而 _inject_cached_values 的 ET 重写会把
   默认命名空间改写为 ``ns0:`` 前缀 → 数据完好但正则失明（X6/X13）→ 改为命名空间无关正则。
3. ``hidden_sheet_*`` 把正则字符串当**子串**做 ``in raw_all``，永远匹配不上（X7/X14）→ 改 re.search。
4. cell token 召回对 float 做精确字符串匹配，IEEE 754 序列化（8.7 → "8.699999999999999"）
   必然假阴性（X1/X8）→ 数值 token 用浮点容差归一后比对。

XLSX 额外核对 14B 新增事实：kv_pairs 落盘、auto_filter ref。
"""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from . import corpus14a, corpus14b, sandbox14a
from .selfcheck14a import _docx_all_text, _docx_raw_xml, _tokenize, _xlsx_all_values

_MEASUREMENT_NOTES = [
    "structural_ok: bool-or-numeric(>=threshold) 判定，修复 selfcheck14a `v is True` 对 float 指标的误判（W5/W12）",
    "formula_dual: 命名空间无关正则，修复 ET 重写 ns0: 前缀导致的失明（X6/X13）",
    "hidden_sheet: re.search 替代「正则当子串」（X7/X14）",
    "cell token recall: float 容差归一，修复 IEEE 754 序列化假阴性（X1/X8）",
]


def _docx_check_fixed(case: dict, path: Path, threshold: float = 0.9) -> dict[str, Any]:
    """与 selfcheck14a.check_docx_case 同一逻辑，仅修正 structural_ok 判定。"""

    raw = _docx_raw_xml(path)
    all_text = _docx_all_text(path)
    vocab = _tokenize(all_text)

    min_recall = 1.0
    for block in case["blocks"]:
        gt = block.get("text")
        if not gt:
            continue
        gt_tokens = _tokenize(gt)
        if gt_tokens:
            min_recall = min(min_recall, sum(1 for t in gt_tokens if t in vocab) / len(gt_tokens))

    expected_table = any(b["kind"] == "table" for b in case["blocks"])
    expected_footnote = any(b["kind"] == "footnote" for b in case["blocks"])
    expected_header = any(b["kind"] == "header" for b in case["blocks"])
    expected_footer = any(b["kind"] == "footer" for b in case["blocks"])
    expected_image = any(b["kind"] == "image" for b in case["blocks"])
    expected_textbox = any(b["kind"] == "textbox" for b in case["blocks"])
    expected_heading_style = any(b["kind"] in ("heading", "title") for b in case["blocks"])

    struct: dict[str, Any] = {}
    if expected_table:
        struct["table_present"] = "w:tbl" in raw
    if expected_footnote:
        with zipfile.ZipFile(path) as z:
            fn = "word/footnotes.xml" in z.namelist()
            ftext = z.read("word/footnotes.xml").decode("utf-8", "replace") if fn else ""
        fvocab = _tokenize(ftext)
        fn_tokens_present = fn and all(
            _tokenize(b["text"]) <= fvocab for b in case["blocks"] if b["kind"] == "footnote")
        struct["footnote_part_present"] = bool(fn_tokens_present) or (min_recall >= threshold)
    if expected_header:
        struct["header_present"] = bool(re.search(r"word/header\d*\.xml", raw)) or "<w:header" in raw
    if expected_footer:
        struct["footer_present"] = bool(re.search(r"word/footer\d*\.xml", raw)) or "<w:footer" in raw
    if expected_image:
        struct["image_present"] = "a:blip" in raw or "pic:pic" in raw
    if expected_textbox:
        tb_texts = [b["text"] for b in case["blocks"] if b["kind"] == "textbox"]
        struct["textbox_present"] = "txbxContent" in raw or all(
            _tokenize(t) <= vocab for t in tb_texts)
    if expected_heading_style:
        struct["heading_style_present"] = bool(re.search(r'w:val="Heading\d"', raw)) or 'w:val="Title"' in raw

    # 修正点 1：float 指标（如 footnote_text_recall）合法，仅 False/None/低于阈值视为失败
    structural_ok = all(
        (v is True) or (isinstance(v, (int, float)) and not isinstance(v, bool) and v >= threshold)
        for v in struct.values()) if struct else True
    passes = min_recall >= threshold and structural_ok
    return {"case_id": case["case_id"], "passes": passes,
            "min_token_recall": round(min_recall, 4), "structural": struct}


def _ns_any(tag: str) -> str:
    """命名空间无关的元素匹配片段：``<(?:[A-Za-z0-9]+:)?c``。"""

    return rf"<(?:[A-Za-z0-9]+:)?{tag}"


def _float_norm_vocab(values: set[str]) -> set[str]:
    """数值容差词表：float 可解析的值同时登记 round(x,6) 归一形式（修正点 4）。"""

    vocab: set[str] = set()
    for v in values:
        vocab.add(v)
        try:
            vocab.add(str(round(float(v), 6)))
        except ValueError:
            pass
    return vocab


def _xlsx_check_fixed(case: dict, path: Path, threshold: float = 0.9) -> dict[str, Any]:
    """与 selfcheck14a.check_xlsx_case 同一逻辑，修正 2/3/4 号测量缺陷。"""

    with zipfile.ZipFile(path) as z:
        raw_all = "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith(".xml"))
    values = _xlsx_all_values(path)
    # 修正点 4：数值归一后的词表（8.7 vs 8.699999999999999 视为同值）
    vocab = _float_norm_vocab(_tokenize(" ".join(values)))

    expected_tokens: set[str] = set()
    for sheet in case["sheets"]:
        for note in sheet.get("notes", []) + sheet.get("side_notes", []):
            expected_tokens |= _tokenize(note["text"])
        for kv in sheet.get("kv_pairs", []):
            expected_tokens |= _tokenize(str(kv["text"]))
        for table in sheet.get("tables", []):
            for h in table.get("headers", []) or []:
                if isinstance(h, list):
                    for x in h:
                        expected_tokens |= _tokenize(str(x))
                else:
                    expected_tokens |= _tokenize(str(h))
            for row in table.get("rows", []):
                for x in row:
                    if x is not None:
                        expected_tokens |= _tokenize(str(x))
            for _cell, (formula, cached) in (table.get("formula_cells") or {}).items():
                expected_tokens |= _tokenize(str(cached))

    covered = (sum(1 for t in expected_tokens
                   if t in vocab or str(round(float(t), 6)) in vocab) / len(expected_tokens)
               if expected_tokens else 1.0)
    min_recall = covered

    struct: dict[str, Any] = {}
    # 修正点 2：命名空间无关的 formula 双表示检查
    for sheet in case["sheets"]:
        for table in sheet.get("tables", []):
            for cell_ref, (formula, cached) in (table.get("formula_cells") or {}).items():
                m = re.search(rf'{_ns_any("c")}[^>]*r="{cell_ref}"[^>]*>.*?</(?:[A-Za-z0-9]+:)?c>',
                              raw_all, re.S)
                if m:
                    cell_xml = m.group(0)
                    has_f = re.search(rf'{_ns_any("f")}[^>]*>{re.escape(formula.lstrip("="))}<', cell_xml)
                    has_v = re.search(rf'{_ns_any("v")}[^>]*>{re.escape(str(cached))}<', cell_xml)
                    struct[f"formula_dual_{cell_ref}"] = bool(has_f and has_v)
                else:
                    struct[f"formula_dual_{cell_ref}"] = False

    for sheet in case["sheets"]:
        for table in sheet.get("tables", []):
            for merge in table.get("merges", []):
                ref = f"{merge[0]}:{merge[1]}"
                struct[f"merge_{ref}"] = f'ref="{ref}"' in raw_all
            if table.get("is_named"):
                struct[f"named_table_{table.get('table_name')}"] = (
                    f'name="{table.get("table_name")}"' in raw_all)
        # 修正点 3：re.search（selfcheck14a 把正则当子串，恒 False）
        if sheet.get("state") == "hidden":
            struct[f"hidden_sheet_{sheet['name']}"] = bool(
                re.search(rf'name="{sheet["name"]}"[^>]*state="hidden"', raw_all))
        for r in sheet.get("hidden_rows", []):
            struct[f"hidden_row_{r}"] = 'hidden="1"' in raw_all
        for cl in sheet.get("hidden_cols", []):
            struct[f"hidden_col_{cl}"] = 'hidden="1"' in raw_all
        af = sheet.get("auto_filter")
        if af:
            struct[f"auto_filter_{af['ref']}"] = f'ref="{af["ref"]}"' in raw_all

    structural_ok = all(v is True for v in struct.values()) if struct else True
    passes = min_recall >= threshold and structural_ok
    return {"case_id": case["case_id"], "passes": passes,
            "min_token_recall": round(min_recall, 4), "structural": struct,
            "all_structural_ok": structural_ok}


def check_corpus(docx_dir: Path, xlsx_dir: Path) -> dict[str, Any]:
    docx_cases = corpus14a.DOCX_CASES + corpus14b.DOCX_CASES_B
    xlsx_cases = corpus14a.XLSX_CASES + corpus14b.XLSX_CASES_B

    docx_results = []
    for case in docx_cases:
        blocks = [{**b, "kind": "table" if b["kind"] == "merged_table" else b["kind"]}
                  for b in case["blocks"]]
        p = docx_dir / f"{case['case_id']}.docx"
        if p.exists():
            docx_results.append(_docx_check_fixed({**case, "blocks": blocks}, p))
        else:
            docx_results.append({"case_id": case["case_id"], "passes": False, "missing_file": True})

    xlsx_results = []
    for case in xlsx_cases:
        p = xlsx_dir / f"{case['case_id']}.xlsx"
        if p.exists():
            result = _xlsx_check_fixed(case, p)
            # 14B 增量：kv_pairs 值必须真实落盘（值级而非 token 级）
            if any(s.get("kv_pairs") for s in case["sheets"]):
                vals = _float_norm_vocab(_xlsx_all_values(p))
                for s in case["sheets"]:
                    for kv in s.get("kv_pairs", []):
                        v = str(kv["text"])
                        result["structural"][f"kv_{kv['cell']}"] = v in vals or str(round(float(v), 6)) in vals
                result["all_structural_ok"] = all(v is True for v in result["structural"].values())
                result["passes"] = result["passes"] and result["all_structural_ok"]
            xlsx_results.append(result)
        else:
            xlsx_results.append({"case_id": case["case_id"], "passes": False, "missing_file": True})

    return {
        "artifact_type": "phase14a_b_corpus_self_check",
        "method": "解析器无关 OOXML 自证（14A-A 纪律 + 四项测量修正，见 measurement_notes）",
        "measurement_notes": _MEASUREMENT_NOTES,
        "docx_passing": sum(1 for r in docx_results if r.get("passes")),
        "docx_total": len(docx_results),
        "xlsx_passing": sum(1 for r in xlsx_results if r.get("passes")),
        "xlsx_total": len(xlsx_results),
        "all_passed": all(r.get("passes") for r in docx_results + xlsx_results),
        "docx": docx_results,
        "xlsx": xlsx_results,
    }
