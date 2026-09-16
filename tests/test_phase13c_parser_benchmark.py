"""Phase 13C-A Parser Benchmark 的纯逻辑与回归测试。

覆盖范围（全部离线、无网络、不加载任何模型）：

1. ``phase13c.schema`` —— §10 统一 schema、派生字段、文档校验、装饰分类；
2. ``phase13c.corpus`` 的排版与测量内核 —— 词级换行、文本框高度测量、
   **溢出必须抛异常**（回归）、**超长 token 必须抛异常**（回归）；
3. ``phase13c.align`` —— 多重集合相似度、单元完整性（完整/碎片/缺失/误合并），
   其中"单 token 装饰单元不得制造误合并"是明确的回归用例；
4. ``phase13c.selfcheck`` —— 与解析器无关的 PDF 文本层自证，含"页码/bbox/幽灵文字"
   三类失败模式的构造性验证；
5. ``phase13c.metrics`` —— 文本相似度基础指标、表格结构化辅助函数，以及
   **figure/table 的几何检出（bbox IoU）**——后者是"无文本结构元素"的唯一有效判据。

纪律：不访问网络、不加载真实模型、不写项目目录（所有 PDF 都在内存或 tmp_path 中生成）。
"""

from __future__ import annotations

import json
import re

import pymupdf
import pytest

from experiments.phase13c import adapters, align, corpus, metrics, schema, selfcheck

# 门判定的纯函数（anchor_hit_map / validation_for_arm / evaluate_gate）住在入口脚本里，
# 这里直接导入以做语义回归；该模块导入不加载任何模型、不产生写副作用。
from experiments import run_phase13c_parser_benchmark as run_benchmark

# --------------------------------------------------------------------------------------
# 公共构造helper
# --------------------------------------------------------------------------------------


def gt_element(element_id: str, type_: str, text: str, page: int = 1, bbox=None) -> dict:
    """构造一个 Ground Truth 风格的元素（``text`` + ``page`` + ``type``）。"""

    return {
        "element_id": element_id,
        "type": type_,
        "text": text,
        "page": page,
        "bbox": list(bbox) if bbox else [0.0, 0.0, 10.0, 10.0],
    }


def parsed_element(element_id: str, type_: str, text: str, page: int = 1, order: int | None = None) -> dict:
    """构造一个 parser 输出风格的元素（``normalized_text`` + ``element_id``）。"""

    return schema.make_element(
        document_id="doc-test",
        element_id=element_id,
        type=type_,
        raw_text=text,
        source_parser="test",
        page=page,
        bbox=[0.0, 0.0, 10.0, 10.0],
        reading_order=order,
    )


# --------------------------------------------------------------------------------------
# 1. schema
# --------------------------------------------------------------------------------------


def test_element_type_whitelist_is_19_and_contains_furniture():
    assert len(schema.ELEMENT_TYPES) == 19
    assert schema.FURNITURE_TYPES <= set(schema.ELEMENT_TYPES)
    # 表格与图注必须是"结构化类型"，否则 13C-B 的 ELEMENT_AWARE 切分无从判定。
    assert {"table", "figure_caption", "table_caption"} <= schema.STRUCTURED_TYPES


def test_make_element_derives_normalized_text_by_joining_hyphenated_breaks():
    element = schema.make_element(
        document_id="doc-1",
        element_id="el-1",
        type="paragraph",
        raw_text="multi-\ncolumn layouts\n lose structure.",
        source_parser="test",
    )
    # 连字符断行必须被合并，换行则折叠为空格。
    assert element["normalized_text"] == "multicolumn layouts lose structure."
    # 逐行保真文本保留原始换行，供"行结构是否保留"类指标使用。
    assert "\n" in element["line_preserved_text"]


def test_join_hyphenated_breaks_only_applies_between_letters():
    # 数字-字母之间的连字符是真实语义（如 arXiv:2609.01234v1 这类编号），不得合并。
    assert schema.join_hyphenated_breaks("2609.0123-\n4v1") == "2609.0123-\n4v1"
    assert schema.join_hyphenated_breaks("state-\nof-the-art") == "stateof-the-art"


def test_make_element_rejects_unknown_type():
    with pytest.raises(ValueError):
        schema.make_element(
            document_id="doc-1", element_id="el-1", type="paragraphx", raw_text="x", source_parser="test"
        )


def test_validate_document_flags_duplicate_element_ids_and_bad_type():
    good = parsed_element("el-1", "paragraph", "hello world")
    duplicate = parsed_element("el-1", "paragraph", "other text")
    bad_type = dict(parsed_element("el-2", "paragraph", "text"), type="not-a-type")
    document = schema.make_document(
        document_id="doc-1",
        source_pdf="x.pdf",
        source_sha256="0" * 64,
        parser="test",
        page_count=1,
        elements=[good, duplicate, bad_type],
    )
    result = schema.validate_document(document)
    assert result["valid"] is False
    issues = " ".join(problem["issue"] for problem in result["problems"])
    assert "duplicate" in issues
    assert "unknown" in issues or "illegal" in issues or "type" in issues


def test_validate_document_accepts_wellformed_document():
    document = schema.make_document(
        document_id="doc-1",
        source_pdf="x.pdf",
        source_sha256="0" * 64,
        parser="test",
        page_count=1,
        elements=[parsed_element("el-1", "paragraph", "hello world")],
    )
    assert schema.validate_document(document)["valid"] is True


def test_classify_furniture_rules():
    assert schema.classify_furniture("12", page=1, is_first_page=False) == "page_number"
    assert schema.classify_furniture("DRAFT - do not distribute", page=1, is_first_page=True) == "watermark"
    assert schema.classify_furniture("arXiv:2609.01234v1", page=1, is_first_page=True) == "sidebar"
    # 运行页眉（非首页、非纯数字）归 header；首页同位置归 footer 之外的 header 也允许，
    # 这里只断言"会落到装饰家族里"，避免把渲染样式写死进测试。
    assert schema.classify_furniture("Some Journal Vol. 3", page=2, is_first_page=False) in schema.FURNITURE_TYPES


# --------------------------------------------------------------------------------------
# 2. corpus 排版内核
# --------------------------------------------------------------------------------------


def test_wrap_text_keeps_every_word_and_respects_width():
    text = " ".join(f"word{i}" for i in range(40))
    lines = corpus.wrap_text(text, "tiro", 10.0, 200.0)
    assert lines
    for line in lines:
        assert pymupdf.get_text_length(line, fontname="tiro", fontsize=10.0) <= 200.0
    # 词一个都不能丢。
    assert " ".join(lines).split() == text.split()


def test_textbox_height_matches_pymupdf_own_measurement():
    """解析公式（textbox_height）必须与 PyMuPDF 自己测出的高度一致。

    这条不变量是"静默丢字"问题的根：一旦两者不一致，按公式预留的矩形就会让
    ``insert_textbox`` 返回负值并一个字都不写。
    """

    fontsize = 9.0
    lineheight = 1.20
    for lines in (1, 2, 5):
        payload = "\n".join(f"line {index}" for index in range(lines))
        analytic = corpus.textbox_height("tiro", fontsize, lines, lineheight)
        measured = corpus.measure_textbox_height(
            200.0, payload, fontname="tiro", fontsize=fontsize, lineheight=lineheight
        )
        assert analytic == pytest.approx(measured, abs=1e-6)


def test_measure_textbox_height_uses_pymupdf_wrapping():
    # 宽度越小 -> 需要越多行 -> 需要越高。
    payload = " ".join(["alpha"] * 30)
    narrow = corpus.measure_textbox_height(80.0, payload, fontname="tiro", fontsize=10.0, lineheight=1.2)
    wide = corpus.measure_textbox_height(400.0, payload, fontname="tiro", fontsize=10.0, lineheight=1.2)
    assert narrow > wide > 0.0


def test_insert_textbox_checked_raises_instead_of_silently_dropping_text():
    """回归：矩形高度不足时 PyMuPDF 返回负值且不写任何文字。

    首版语料没有检查返回值，导致表格体与页码在 PDF 文本层完全消失。
    """

    document = pymupdf.open()
    try:
        page = document.new_page(width=200, height=200)
        # 9pt Times 单行实际需要 9*1.2 + 9*0.281 = 13.33pt；13pt 的框放不下。
        with pytest.raises(RuntimeError, match="textbox overflow"):
            corpus.insert_textbox_checked(
                page,
                pymupdf.Rect(0.0, 0.0, 100.0, 13.0),
                "1",
                fontname="tiro",
                fontsize=9.0,
                lineheight=1.2,
            )
        assert page.get_text("words") == []
    finally:
        document.close()


def test_insert_textbox_checked_writes_when_rect_is_adequate():
    document = pymupdf.open()
    try:
        page = document.new_page(width=200, height=200)
        height = corpus.measure_textbox_height(100.0, "1", fontname="tiro", fontsize=9.0, lineheight=1.2)
        corpus.insert_textbox_checked(
            page,
            pymupdf.Rect(0.0, 0.0, 100.0, height + 1.0),
            "1",
            fontname="tiro",
            fontsize=9.0,
            lineheight=1.2,
        )
        assert [item[4] for item in page.get_text("words")] == ["1"]
    finally:
        document.close()


def test_ensure_tokens_fit_raises_for_token_wider_than_box():
    """回归：PyMuPDF 会把放不下的单词硬拆成两个 token。

    arXiv 侧栏的 ``arXiv:2609.01234v1`` 在 6pt 下宽 49.83pt，放进 48pt 的框后被拆成
    ``arXiv:2609.01234v`` + ``1``，GT 的 token ``2609.01234v1`` 因此永久失配。
    """

    with pytest.raises(RuntimeError, match="longest token"):
        corpus.ensure_tokens_fit(
            "arXiv:2609.01234v1",
            fontname="tiro",
            fontsize=6.0,
            width=48.0,
            label="test sidebar",
        )


def test_ensure_tokens_fit_accepts_when_box_is_wide_enough():
    corpus.ensure_tokens_fit(
        "arXiv:2609.01234v1", fontname="tiro", fontsize=6.0, width=60.0, label="test sidebar"
    )


# --------------------------------------------------------------------------------------
# 3. align
# --------------------------------------------------------------------------------------


def test_multiset_similarity_basics():
    reference = align.token_counter("alpha beta gamma")
    assert align.multiset_recall(reference, align.token_counter("alpha beta gamma")) == 1.0
    assert align.multiset_recall(reference, align.token_counter("alpha beta")) == pytest.approx(2 / 3)
    assert align.multiset_recall(reference, align.token_counter("")) == 0.0
    assert align.multiset_f1(reference, align.token_counter("alpha beta gamma")) == 1.0


def test_unit_integrity_complete_and_fragmented_and_missing():
    paragraph = "The quick brown fox jumps over the lazy dog near the river bank today"
    ground_truth = [gt_element("gt-1", "paragraph", paragraph)]
    # 完整：单个 parser 元素覆盖全部 token。
    complete = align.analyze_unit_integrity(ground_truth, [parsed_element("pr-1", "paragraph", paragraph)])
    assert complete["complete_unit_count"] == 1
    assert complete["fragmented"] == []
    # 碎片：同一段落被切成两半，并集覆盖 100%。
    halves = paragraph.split()
    split = [
        parsed_element("pr-1", "paragraph", " ".join(halves[:7])),
        parsed_element("pr-2", "paragraph", " ".join(halves[7:])),
    ]
    fragmented = align.analyze_unit_integrity(ground_truth, split)
    assert len(fragmented["fragmented"]) == 1
    assert fragmented["fragmented"][0]["union_coverage"] == 1.0
    # 缺失：只抽到一半内容。
    incomplete = align.analyze_unit_integrity(ground_truth, [parsed_element("pr-1", "paragraph", " ".join(halves[:7]))])
    assert len(incomplete["incomplete"]) == 1
    assert incomplete["incomplete"][0]["best_single_coverage"] < align.COMPLETE_COVERAGE


def test_single_token_furniture_does_not_create_phantom_merges():
    """回归：页码 GT 只有 1 个 token，任何含该数字的正文块都会"完整包含"它。

    未修复时，作者单位里的 "1 Science Park Road" 会因为含 token ``1`` 而被判为
    "合并了页码 1"，使 T01 的误合并率虚高到 0.46。
    """

    ground_truth = [
        gt_element("gt-1", "author_block", "Wei Zhang, Ming Li - Institute of Document Engineering, 1 Science Park Road"),
        gt_element("gt-2", "page_number", "1"),
    ]
    parsed = [
        parsed_element(
            "pr-1",
            "paragraph",
            "Wei Zhang, Ming Li - Institute of Document Engineering, 1 Science Park Road",
        )
    ]
    integrity = align.analyze_unit_integrity(ground_truth, parsed)
    assert integrity["merged"] == []


def test_real_merge_of_heading_and_paragraph_is_reported():
    """标题 + 紧随段落被粘成一个 parser 元素，且标题编号 token 独占 -> 判为误合并。"""

    ground_truth = [
        gt_element("gt-1", "section_header", "2 Related Work"),
        gt_element(
            "gt-2",
            "paragraph",
            "Chunking strategies have evolved from fixed token windows to structure aware segmentation.",
        ),
    ]
    parsed = [
        parsed_element(
            "pr-1",
            "paragraph",
            "2 Related Work Chunking strategies have evolved from fixed token windows to structure aware segmentation.",
        )
    ]
    integrity = align.analyze_unit_integrity(ground_truth, parsed)
    assert len(integrity["merged"]) == 1
    record = integrity["merged"][0]
    assert record["merged_count"] == 2
    # dominance 是占比，绝不应超过 1。
    assert 0.0 <= record["dominance"] <= 1.0
    assert record["size_ratio"] <= align.MERGED_MAX_SIZE_RATIO


def test_long_block_incidentally_covering_short_units_is_not_a_merge():
    """一个装了少量其它内容的超长块顺带覆盖两个短 GT，不应判为误合并。"""

    ground_truth = [
        gt_element("gt-1", "paragraph", "alpha beta gamma delta epsilon zeta"),
        gt_element("gt-2", "paragraph", "eta theta iota kappa lambda mu"),
    ]
    long_text = " ".join(
        ["alpha beta gamma delta epsilon zeta", "eta theta iota kappa lambda mu"]
        + [f"filler{index}" for index in range(60)]
    )
    parsed = [parsed_element("pr-1", "paragraph", long_text)]
    integrity = align.analyze_unit_integrity(ground_truth, parsed)
    assert integrity["merged"] == []


def test_merged_detection_excludes_furniture_types():
    """装饰元素（header/footer/page_number/watermark/sidebar）不参与误合并判定。"""

    ground_truth = [
        gt_element("gt-1", "footer", "Copyright 2026 by the Association for Document Engineering."),
        gt_element("gt-2", "watermark", "DRAFT do not distribute this version"),
    ]
    parsed = [
        parsed_element(
            "pr-1",
            "paragraph",
            "Copyright 2026 by the Association for Document Engineering. DRAFT do not distribute this version",
        )
    ]
    assert align.analyze_unit_integrity(ground_truth, parsed)["merged"] == []


def test_align_elements_reports_missing_and_spurious():
    ground_truth = [
        gt_element("gt-1", "paragraph", "the model learns a compact representation of documents"),
        gt_element("gt-2", "paragraph", "tables are flattened into unreadable token streams by naive parsers"),
    ]
    parsed = [
        parsed_element("pr-1", "paragraph", "the model learns a compact representation of documents"),
        parsed_element("pr-2", "paragraph", "a completely unrelated sentence about weather forecasts in the north"),
    ]
    alignment = align.align_elements(ground_truth, parsed)
    assert [item["gt_element_id"] for item in alignment["missing"]] == ["gt-2"]
    assert [item["parsed_element_id"] for item in alignment["spurious"]] == ["pr-2"]
    assert len(alignment["matches"]) == 1


def test_detect_column_interleave_flags_row_major_two_column_reading():
    """双栏被逐行交叉读取时必须被检出（方案 §11.2 的硬门依据）。"""

    ground_truth = [
        {**gt_element("gt-l1", "paragraph", "left one", page=1), "column": "L", "reading_order": 0},
        {**gt_element("gt-l2", "paragraph", "left two", page=1), "column": "L", "reading_order": 1},
        {**gt_element("gt-r1", "paragraph", "right one", page=1), "column": "R", "reading_order": 2},
        {**gt_element("gt-r2", "paragraph", "right two", page=1), "column": "R", "reading_order": 3},
    ]
    # 正确顺序：左1 左2 右1 右2
    correct = [
        parsed_element("pr-l1", "paragraph", "left one", order=0),
        parsed_element("pr-l2", "paragraph", "left two", order=1),
        parsed_element("pr-r1", "paragraph", "right one", order=2),
        parsed_element("pr-r2", "paragraph", "right two", order=3),
    ]
    good = align.detect_column_interleave(
        ground_truth, correct, align.align_elements(ground_truth, correct)
    )
    assert good["comparable_pairs"] == 6
    assert good["interleave_detected"] is False
    # 交叉顺序：左1 右1 左2 右2
    interleaved = [
        parsed_element("pr-l1", "paragraph", "left one", order=0),
        parsed_element("pr-r1", "paragraph", "right one", order=1),
        parsed_element("pr-l2", "paragraph", "left two", order=2),
        parsed_element("pr-r2", "paragraph", "right two", order=3),
    ]
    bad = align.detect_column_interleave(
        ground_truth, interleaved, align.align_elements(ground_truth, interleaved)
    )
    assert bad["interleave_detected"] is True
    assert bad["interleave_rate"] > bad["threshold"]


# --------------------------------------------------------------------------------------
# 4. selfcheck（与解析器无关的语料自证）
# --------------------------------------------------------------------------------------


def _render_page(tmp_path, text: str, bbox, *, extra: str | None = None):
    """渲染一个单页 PDF，返回 (path, gt_document)。"""

    document = pymupdf.open()
    page = document.new_page(width=300, height=300)
    rect = pymupdf.Rect(*bbox)
    height = corpus.measure_textbox_height(rect.width, text, fontname="tiro", fontsize=10.0, lineheight=1.2)
    corpus.insert_textbox_checked(
        page,
        pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + height + 1.0),
        text,
        fontname="tiro",
        fontsize=10.0,
        lineheight=1.2,
    )
    if extra:
        corpus.insert_textbox_checked(
            page,
            pymupdf.Rect(rect.x0, rect.y0 + height + 20.0, rect.x1, rect.y0 + height + 60.0),
            extra,
            fontname="tiro",
            fontsize=10.0,
            lineheight=1.2,
        )
    path = tmp_path / "page.pdf"
    document.save(str(path))
    document.close()
    return path, {
        "template_id": "TP",
        "rasterized": False,
        "page_count": 1,
        "elements": [
            {
                "element_id": "gt-0001",
                "type": "paragraph",
                "text": text,
                "page": 1,
                "bbox": [rect.x0, rect.y0, rect.x1, rect.y0 + height + 1.0],
            }
        ],
    }


def test_selfcheck_passes_on_a_faithful_ground_truth(tmp_path):
    path, gt_document = _render_page(tmp_path, "alpha beta gamma delta", (20.0, 20.0, 200.0, 120.0))
    check = selfcheck.check_document(path, gt_document)
    assert check["excluded"] is False
    assert check["passes"] is True
    assert check["min_page_token_recall"] == 1.0
    assert check["min_bbox_token_recall"] == 1.0
    assert check["max_unclaimed_ratio"] == 0.0


def test_selfcheck_detects_ground_truth_claiming_the_wrong_page(tmp_path):
    path, gt_document = _render_page(tmp_path, "alpha beta gamma delta", (20.0, 20.0, 200.0, 120.0))
    gt_document["elements"][0]["page"] = 2
    check = selfcheck.check_document(path, gt_document)
    assert check["passes"] is False
    assert check["min_page_token_recall"] < 0.95


def test_selfcheck_detects_ground_truth_bbox_that_misses_the_text(tmp_path):
    path, gt_document = _render_page(tmp_path, "alpha beta gamma delta", (20.0, 20.0, 200.0, 120.0))
    # bbox 缩到左下角一小块，声称的文本不在其中。
    gt_document["elements"][0]["bbox"] = [20.0, 20.0, 40.0, 30.0]
    check = selfcheck.check_document(path, gt_document)
    assert check["passes"] is False
    assert check["min_bbox_token_recall"] < 0.95


def test_selfcheck_detects_undeclared_ghost_text(tmp_path):
    path, gt_document = _render_page(
        tmp_path, "alpha beta gamma delta", (20.0, 20.0, 200.0, 120.0), extra="ghost content never declared"
    )
    check = selfcheck.check_document(path, gt_document)
    assert check["passes"] is False
    assert check["max_unclaimed_ratio"] > selfcheck.UNCLAIMED_TOLERANCE


def test_selfcheck_excludes_rasterized_template_without_opening_the_pdf(tmp_path):
    # 传入不存在的路径：栅格化模板必须在打开 PDF 之前就被排除，否则扫描模板永远无法自证。
    check = selfcheck.check_document(
        tmp_path / "does-not-exist.pdf",
        {"template_id": "T12", "rasterized": True, "page_count": 3, "elements": []},
    )
    assert check["excluded"] is True
    assert "rasterized" in check["reason"]


# --------------------------------------------------------------------------------------
# 5. metrics 的基础工具
# --------------------------------------------------------------------------------------


def test_character_error_rate_and_exact_match():
    assert metrics.character_error_rate("hello world", "hello world")["cer"] == 0.0
    assert metrics.character_error_rate("hello world", "")["cer"] == 1.0
    assert 0.0 < metrics.character_error_rate("hello world", "hello word")["cer"] < 0.2
    # normalized_exact_match 只做空白折叠，不做大小写归一。
    assert metrics.normalized_exact_match("Alpha   Beta", "Alpha Beta") is True
    assert metrics.normalized_exact_match("alpha", "beta") is False


def test_token_prf_reports_precision_recall_and_f1():
    result = metrics.token_prf("alpha beta gamma", "alpha beta gamma delta")
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(3 / 4)
    assert 0.0 < result["f1"] < 1.0


def test_table_helpers_normalise_cells_and_render_html():
    assert metrics._norm_cell("  F1  ") == "f1"
    html = metrics.tables_to_html({"rows": [["a", "b"], ["c", "d"]], "header_row": True})
    assert html is not None
    assert html.count("<tr>") == 2
    # 表头行渲染为 th，数据行渲染为 td。
    assert html.count("<th>") == 2
    assert html.count("<td>") == 2
    assert metrics.tables_to_html(None) is None
    assert metrics.tables_to_html({"rows": []}) is None


def test_aggregate_metric_dicts_ignores_none_and_uses_mean():
    aggregated = metrics.aggregate_metric_dicts([{"a": 1.0, "b": None}, {"a": 0.0, "b": None}, {"a": None, "b": None}])
    assert aggregated["a"] == pytest.approx(0.5)
    # 全部为 None 的指标必须保持 None（语义：该 parser 未提供计算所需信息），
    # 而不是被当作 0 拉低均值。
    assert aggregated["b"] is None


# --------------------------------------------------------------------------------------
# §14 Parser Selection Gate —— 门 1 / 门 10 的语义回归
#
# 这两条曾经实现错：
#   门 1 用绝对阈值 mean_recall >= 0.995 判定"无新增丢失"，导致 baseline 自身的既有失败
#       把全部候选 arm 一起判死；
#   门 10 用跨 arm 聚合的 schema 校验结果，任一 arm 不合法会让所有 arm 一起 FAIL。
# 下面的用例把正确语义钉死。
# --------------------------------------------------------------------------------------


def _gate_arm_record(*, valid_documents: bool = True) -> dict:
    """构造一个最小可判定的 arm_record，用于门 10 的 per-arm 校验。"""

    return {
        "controlled_metrics": {
            "none::T01": {"document_validation": {"valid": valid_documents, "problem_count": 0 if valid_documents else 3}}
        },
        "real_metrics": {},
    }


def _gate_generic_metrics() -> dict:
    """只满足门 3/4/5 的受控指标骨架（门 1/2/10 由参数单独控制）。"""

    return {
        "none::T01": {
            "is_two_column": False,
            "reading_order": {"column_interleave": {"interleave_detected": False}},
            "provenance": {"page_provenance_coverage": 1.0},
            "table": {"gt_table_count": 0, "table_detected": None},
            "figure_caption": {
                "gt_figure_count": 1,
                "figure_detection": 0.9,
                "gt_figure_caption_count": 1,
                "caption_detection": 0.9,
            },
            "furniture": {
                "parsed_furniture_count": 2,
                "furniture_false_removal": 0.0,
            },
        }
    }


def test_anchor_hit_map_flattens_and_ors_duplicate_anchor_ids():
    real_metrics = {
        "a.pdf": {"anchors": {"detail": [{"anchor_id": "EVID-Q1-G1-A1", "hit": False}]}},
        "b.pdf": {"anchors": {"detail": [{"anchor_id": "EVID-Q1-G1-A1", "hit": True}, {"anchor_id": "EVID-Q2-G1-A1", "hit": False}]}},
    }
    hits = run_benchmark.anchor_hit_map(real_metrics)
    # 同一锚点多篇明细：任一命中即算命中。
    assert hits["EVID-Q1-G1-A1"] is True
    assert hits["EVID-Q2-G1-A1"] is False


def test_validation_for_arm_is_per_arm_not_global():
    good = run_benchmark.validation_for_arm(_gate_arm_record(valid_documents=True))
    bad = run_benchmark.validation_for_arm(_gate_arm_record(valid_documents=False))
    assert good["valid"] is True
    assert good["graded_documents"] == 1
    # 坏 arm 自己的判定必须为 False…… 且不能反过来污染好 arm（这就是跨 arm 聚合的 bug）。
    assert bad["valid"] is False


def test_gate1_no_new_loss_is_relative_to_baseline_not_absolute():
    """baseline 已失败的锚点不算"新增丢失"；只有 baseline 命中而本 arm 未命中才 FAIL。"""

    # baseline 命中 A1，未命中 A2（既有失败，例如 ARR-007 那类）。
    baseline_hits = {"EVID-Q1-G1-A1": True, "EVID-Q1-G1-A2": False}
    # 候选 arm 命中 A1、仍然未命中 A2 → 没有新增丢失 → PASS（绝对阈值会把它判 FAIL）。
    same = run_benchmark.evaluate_gate(
        controlled_metrics=_gate_generic_metrics(),
        anchor_hits={"EVID-Q1-G1-A1": True, "EVID-Q1-G1-A2": False},
        baseline_anchor_hits=baseline_hits,
        document_validation={"valid": True, "problems": [], "graded_documents": 1},
        source_pdfs_unchanged=True,
        external_llm_declaration="none",
        failure_records=[],
        reproducibility_record={"reproducible": True, "template_id": "T02", "ocr_mode": "none"},
    )
    gate1 = next(check for check in same["checks"] if check["gate"] == "canonical_anchor_no_new_loss")
    assert gate1["passed"] is True
    assert gate1["evidence"]["new_loss_count"] == 0

    # 候选 arm 把 baseline 命中的 A1 丢掉了 → 新增丢失 → FAIL。
    worse = run_benchmark.evaluate_gate(
        controlled_metrics=_gate_generic_metrics(),
        anchor_hits={"EVID-Q1-G1-A1": False, "EVID-Q1-G1-A2": False},
        baseline_anchor_hits=baseline_hits,
        document_validation={"valid": True, "problems": [], "graded_documents": 1},
        source_pdfs_unchanged=True,
        external_llm_declaration="none",
        failure_records=[],
        reproducibility_record={"reproducible": True, "template_id": "T02", "ocr_mode": "none"},
    )
    gate1_worse = next(
        check for check in worse["checks"] if check["gate"] == "canonical_anchor_no_new_loss"
    )
    assert gate1_worse["passed"] is False
    assert gate1_worse["evidence"]["new_loss_anchor_ids"] == ["EVID-Q1-G1-A1"]


def test_gate1_is_pending_without_baseline_hits():
    """没有 baseline 命中集合时不得伪造 PASS，必须 PENDING（None）。"""

    result = run_benchmark.evaluate_gate(
        controlled_metrics=_gate_generic_metrics(),
        anchor_hits={},
        baseline_anchor_hits=None,
        document_validation={"valid": True, "problems": [], "graded_documents": 1},
        source_pdfs_unchanged=True,
        external_llm_declaration="none",
        failure_records=[],
        reproducibility_record={"reproducible": True, "template_id": "T02", "ocr_mode": "none"},
    )
    gate1 = next(check for check in result["checks"] if check["gate"] == "canonical_anchor_no_new_loss")
    assert gate1["passed"] is None
    assert result["all_gates_decided"] is False


def test_gate10_fails_only_for_the_arm_with_invalid_schema():
    """门 10 必须是 per-arm：坏 arm FAIL，好 arm 仍应 PASS。"""

    def run(validation: dict) -> bool:
        result = run_benchmark.evaluate_gate(
            controlled_metrics=_gate_generic_metrics(),
            anchor_hits={"A1": True},
            baseline_anchor_hits={"A1": True},
            document_validation=validation,
            source_pdfs_unchanged=True,
            external_llm_declaration="none",
            failure_records=[],
            reproducibility_record={"reproducible": True, "template_id": "T02", "ocr_mode": "none"},
        )
        return next(
            check for check in result["checks"] if check["gate"] == "maps_to_unified_schema"
        )["passed"]

    assert run({"valid": True, "problems": [], "graded_documents": 1}) is True
    assert run({"valid": False, "problems": [2], "graded_documents": 1}) is False


def test_gate6_reproducibility_is_tri_state():
    """"未采集"必须是 PENDING，"采集到但不一致"才是 FAIL。

    回归：早期实现写成 ``bool(record.get("reproducible"))``，把 reproducible=None
    （复现模板未参与本次运行）误判成 FAIL——那是"没能验证"而不是"验证不通过"。
    """

    def gate6(record) -> bool | None:
        result = run_benchmark.evaluate_gate(
            controlled_metrics=_gate_generic_metrics(),
            anchor_hits={"A1": True},
            baseline_anchor_hits={"A1": True},
            document_validation={"valid": True, "problems": [], "graded_documents": 1},
            source_pdfs_unchanged=True,
            external_llm_declaration="none",
            failure_records=[],
            reproducibility_record=record,
        )
        return next(check for check in result["checks"] if check["gate"] == "reproducible_output")[
            "passed"
        ]

    # 未采集 → PENDING
    assert gate6({"checked": False, "reproducible": None, "reason": "首次运行未产出 T02"}) is None
    # record 整体缺失 → PENDING
    assert gate6(None) is None
    # 采集且一致 → PASS
    assert gate6({"checked": True, "reproducible": True, "digest_first": "ab" * 32}) is True
    # 采集但不一致 → FAIL
    assert gate6({"checked": True, "reproducible": False, "digest_first": "ab" * 32}) is False


# --------------------------------------------------------------------------------------
# arm 的 OCR 档位定义 —— "原生文本层档必须存在"与"环境阻塞档必须显式记录"
# --------------------------------------------------------------------------------------


def test_mineru_offers_native_tier_so_none_mode_never_comes_back_empty():
    """回归：缺少 "none" 档会让 --ocr-modes none 被过滤成空列表 → arm 静默零运行。

    曾经的症状是 `arm P1_mineru 完成，用时 0.4s` 且 structured/ 下没有任何产物——
    看起来像"跑完了"，实际是一个文档都没解析。
    """

    mineru = adapters.ARM_INDEX["P1_mineru"]
    assert "none" in mineru.ocr_modes
    resolved = run_benchmark.resolve_ocr_modes(mineru, ["none"])
    assert resolved == ["none"], "原生文本层档必须可被请求，否则 arm 会静默零运行"


def test_marker_vlm_ocr_tiers_are_explicitly_blocked_not_silently_dropped():
    """Marker 的 auto/force 需要 Docker/llama.cpp，本机不可用——必须显式记录原因。

    如果只是把它们从 ocr_modes 里删掉，产物会看起来像"Marker 天生只有原生档"，
    把环境限制误读成工具能力边界。
    """

    marker = adapters.ARM_INDEX["P3_marker"]
    assert marker.ocr_modes == ("none",)
    assert set(marker.blocked_ocr_modes) == {"auto", "force"}
    # 原因必须可机器核查地指向 docker / llama-server
    reason = marker.blocked_ocr_modes["auto"]
    assert "docker" in reason and "llama-server" in reason

    # 只报告"被明确请求过"的档位；用默认档时不产生噪音。
    assert set(adapters.blocked_ocr_modes_for(marker, ["none", "auto"])) == {"auto"}
    assert adapters.blocked_ocr_modes_for(marker, None) == {}
    assert adapters.blocked_ocr_modes_for(marker, ["none"]) == {}


# --------------------------------------------------------------------------------------
# figure / table 的几何检出（bbox IoU）—— "无文本的结构元素只能用几何判据"
#
# 这一组用例把受控矩阵暴露的三类真实缺陷钉死：
#   ① 用文本相似度判定图/表检出 → GT 图的 text 是空串（矢量绘制），相似度在构造上恒为 0，
#      实测所有 arm 的 figure_detection 都是 0.0（含 baseline），把唯一有效的区分信号抹掉；
#   ② 缺同页约束 → 模板化语料里不同页的同类元素坐标高度相似，会互相误配；
#   ③ 页码缺失被直接判死 → "找不到元素"与"元素找到了但页码不可验证"是两件事，
#      后者属于 provenance 缺陷，必须分开报告。
# --------------------------------------------------------------------------------------


def geom_gt(element_id: str, type_: str, page, bbox, text: str = "") -> dict:
    """构造 GT 风格的结构元素：``text`` 默认为空（矢量图 / 表格结构体）。"""

    return {
        "element_id": element_id,
        "type": type_,
        "text": text,
        "page": page,
        "bbox": list(bbox) if bbox is not None else None,
    }


def geom_parsed(element_id: str, type_: str, page, bbox, text: str = "", **extra) -> dict:
    """构造 parser 风格的带 bbox 元素（走 schema 保证字段齐全）。"""

    element = schema.make_element(
        document_id="doc-geom",
        element_id=element_id,
        type=type_,
        raw_text=text,
        source_parser="test",
        page=page,
        bbox=list(bbox) if bbox is not None else None,
        reading_order=0,
    )
    element.update(extra)
    return element


def test_bbox_iou_primitive_semantics():
    assert metrics.bbox_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0
    assert metrics.bbox_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0
    # 半重叠：交 5x5=25，并 100+100-25=175。
    assert metrics.bbox_iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)) == pytest.approx(25 / 175)
    # 任一框缺失时返回 0，而不是抛异常或返回未定义的比值。
    assert metrics.bbox_iou(None, (0.0, 0.0, 10.0, 10.0)) == 0.0


def test_element_box_rejects_degenerate_and_malformed_boxes():
    assert metrics.element_box({"bbox": [0, 0, 10, 10]}) == (0.0, 0.0, 10.0, 10.0)
    assert metrics.element_box({"bbox": [0, 0, 0, 10]}) is None, "零宽框无面积，不能参与 IoU"
    assert metrics.element_box({"bbox": [0, 0, 10]}) is None
    assert metrics.element_box({"bbox": [0, "x", 10, 10]}) is None
    assert metrics.element_box({"bbox": [0, 0, float("inf"), 10]}) is None
    assert metrics.element_box({"bbox": None}) is None
    assert metrics.element_box({}) is None


def test_geometric_detection_is_one_to_one_and_honours_threshold():
    """贪心一对一：一个 parsed 元素不能同时"检出"两个 GT；低于阈值不算命中。"""

    gt = [
        geom_gt("gt-fig-1", "figure", 1, [0, 0, 100, 100]),
        geom_gt("gt-fig-2", "figure", 1, [0, 0, 100, 100]),  # 与 gt-fig-1 完全重合
    ]
    parsed = [geom_parsed("pr-fig-1", "figure", 1, [0, 0, 100, 100])]
    result = metrics.geometric_detection(gt, parsed, gt_type="figure")
    assert result["match_count"] == 1
    assert result["detection"] == pytest.approx(0.5)

    # IoU = 10/190 ≈ 0.053，远低于 0.5 → 不算命中。
    close_but_no = metrics.geometric_detection(
        [geom_gt("gt-fig-1", "figure", 1, [0, 0, 100, 100])],
        [geom_parsed("pr-fig-1", "figure", 1, [90, 0, 190, 100])],
        gt_type="figure",
    )
    assert close_but_no["match_count"] == 0


def test_geometric_detection_rejects_cross_page_coincidence():
    """同页约束：模板化语料里不同页的同类元素坐标高度相似，跨页命中必须被否决。"""

    result = metrics.geometric_detection(
        [geom_gt("gt-fig-1", "figure", 2, [50, 100, 500, 400])],
        [geom_parsed("pr-fig-1", "figure", 7, [50, 100, 500, 400])],
        gt_type="figure",
    )
    assert result["match_count"] == 0
    assert result["detection"] == 0.0


def test_geometric_detection_flags_unverified_page_matches():
    """页码缺失时应命中但标记"未验证"，而不是静默丢弃或静默算作已验证。"""

    result = metrics.geometric_detection(
        [geom_gt("gt-fig-1", "figure", 2, [50, 100, 500, 400])],
        [geom_parsed("pr-fig-1", "figure", None, [50, 100, 500, 400])],
        gt_type="figure",
    )
    assert result["match_count"] == 1
    assert result["page_unverified_match_count"] == 1
    assert result["matches"][0]["page_verified"] is False


def test_figure_detection_uses_bbox_iou_when_gt_text_is_empty():
    """回归：GT 图没有文本时，文本判据恒为 0，只有几何判据能识别出检出。"""

    gt = [
        geom_gt("gt-fig-1", "figure", 2, [50, 100, 500, 400]),
        geom_gt("gt-fig-2", "figure", 3, [50, 120, 500, 380]),
    ]
    parsed = [
        geom_parsed("pr-fig-1", "figure", 2, [52, 98, 498, 402]),
        geom_parsed("pr-fig-2", "figure", 3, [49, 121, 501, 379]),
    ]
    result = metrics.figure_caption_metrics(gt, parsed, align.align_elements(gt, parsed))
    assert result["gt_figure_count"] == 2
    # 文本判据必然为 0——两个 GT 图的 text 都是空串（这就是旧实现恒判 0 的原因）。
    assert result["figure_detection_by_text"] == 0.0
    # 几何判据必须全部认出。
    assert result["figure_detection_by_geometry"] == 1.0
    assert result["figure_detection"] == 1.0


def test_table_detected_uses_bbox_iou_when_table_text_lives_in_structure():
    """回归：Docling 的表格文本在 ``table_structure`` 里、扁平文本为空，文本判据曾判 0。"""

    gt = [geom_gt("gt-tbl-1", "table", 2, [40, 200, 560, 500])]
    parsed = [
        {
            "element_id": "pr-tbl-1",
            "type": "table",
            "text": "",
            "normalized_text": "",
            "page": 2,
            "bbox": [41, 199, 559, 501],
            "reading_order": 0,
            "table_structure": {"rows": [["h1", "h2"], ["a", "b"]], "header_row": True},
        }
    ]
    result = metrics.table_metrics(gt, parsed, align.align_elements(gt, parsed))
    assert result["table_detected_by_text"] == 0.0
    assert result["table_detected_by_geometry"] == 1.0
    assert result["table_detected"] == 1.0


def test_geometric_precision_exposes_over_labelling():
    """精确率暴露「把图统统标成表」：召回可以满分，精确率立刻掉下来。

    实测来源：Marker 在 `T10_figure_heavy`（GT 有 2 表 5 图）输出了 6 个 table 块，
    其中 4 个实际是图。只看召回会得到"表格检出 1.0"的假象。
    """

    gt = [geom_gt("gt-tbl-1", "table", 1, [0, 0, 100, 100])]
    parsed = [
        geom_parsed("pr-tbl-1", "table", 1, [0, 0, 100, 100]),      # 真表
        geom_parsed("pr-tbl-2", "table", 1, [200, 200, 300, 300]),  # 实为图
        geom_parsed("pr-tbl-3", "table", 1, [400, 400, 500, 500]),  # 实为图
    ]
    result = metrics.geometric_detection(gt, parsed, gt_type="table")
    assert result["detection"] == 1.0, "召回口径只看 GT 侧，这里应当是满分"
    assert result["match_count"] == 1
    assert result["precision"] == pytest.approx(1 / 3), "精确率必须暴露 2/3 的误标"
    assert result["parsed_with_bbox_count"] == 3


def test_detection_precision_is_surfaced_in_type_metrics():
    """两个类型指标的 payload 都必须带几何精确率，否则报告看不到误标。"""

    gt = [
        geom_gt("gt-tbl-1", "table", 1, [0, 0, 100, 100]),
        geom_gt("gt-fig-1", "figure", 1, [0, 200, 100, 300]),
    ]
    parsed = [
        geom_parsed("pr-tbl-1", "table", 1, [0, 0, 100, 100]),
        geom_parsed("pr-tbl-2", "table", 1, [400, 400, 500, 500]),
        geom_parsed("pr-fig-1", "figure", 1, [0, 200, 100, 300]),
    ]
    alignment = align.align_elements(gt, parsed)
    table = metrics.table_metrics(gt, parsed, alignment)
    figure = metrics.figure_caption_metrics(gt, parsed, alignment)
    assert table["table_detection_precision"] == pytest.approx(0.5)
    assert figure["figure_detection_precision"] == pytest.approx(1.0)


def test_detection_union_keeps_text_evidence_and_geometry_denominator_is_bbox_only():
    """两条证据链取并集；几何分母只算「有合法 bbox 的 GT」，GT 无 bbox 时返回 None。"""

    gt = [
        geom_gt("gt-tbl-1", "table", 2, None, text="Model Accuracy F1 Baseline 0.82 0.85")
    ]
    parsed = [
        geom_parsed(
            "pr-tbl-1", "table", 2, [40, 200, 560, 500], text="Model Accuracy F1 Baseline 0.82 0.85"
        )
    ]
    result = metrics.table_metrics(gt, parsed, align.align_elements(gt, parsed))
    # GT 无 bbox → 几何判据无从计算，必须返回 None（"未采集"）而不是 0（"未检出"）。
    assert result["table_detected_by_geometry"] is None
    # 文本判据仍然成立 → 并集判为已检出。
    assert result["table_detected_by_text"] == 1.0
    assert result["table_detected"] == 1.0


# --------------------------------------------------------------------------------------
# 公式：文本约定不可比 + 分子分母必须同处一个计数空间
#
# 存在的理由：实测 baseline 在 T09 只把 1 个元素标成 equation，却因 11 个 GT 公式各自
# 匹配到一个"段落"而算出 equation_detection_precision = 11.0（> 1 在数学上不可能）；
# 同时 MinerU 检出了全部 11 个公式，却因为输出原始 LaTeX（GT 是 asciiMath）而被算成
# 0.0707。前者是计数空间缺陷，后者是表示约定缺陷——两个都必须由实现消除，
# 否则报告会同时出现"不可能的数字"和"把检出成功读成失败"。
# --------------------------------------------------------------------------------------


def test_bbox_overlap_coefficient_primitive_semantics():
    """重叠系数对"框大框小"不敏感：一个框完全含于另一个即为 1.0，这正是公式需要的性质。"""

    box = (0.0, 0.0, 10.0, 10.0)
    assert metrics.bbox_overlap_coefficient(box, box) == 1.0
    assert metrics.bbox_overlap_coefficient(box, (20.0, 20.0, 30.0, 30.0)) == 0.0
    # 小框完全落在大框内部：IoU 只有 25/100=0.25，重叠系数却是 1.0。
    inner = (2.5, 2.5, 7.5, 7.5)
    assert metrics.bbox_iou(box, inner) == pytest.approx(25 / 100)
    assert metrics.bbox_overlap_coefficient(box, inner) == 1.0
    assert metrics.bbox_overlap_coefficient(None, box) == 0.0


def test_equation_geometry_uses_overlap_because_gt_box_is_line_level():
    """GT 公式框是行级框、parser 是字形框：IoU 判不中，重叠系数判得中。

    这条用例把阈值选择固定下来——若有人把公式改回 IoU 判据，它会立刻失败。
    """

    # GT：行级框，占满栏宽（含右侧公式编号的区域）。
    gt = [geom_gt("g1", "equation", 1, (56.0, 725.91, 540.28, 741.67), text="E = m c^2 (1)")]
    # parser：只框住公式字形本身，宽度约为 GT 的三分之一。
    parsed = [geom_parsed("p1", "equation", 1, (210.72, 730.01, 385.56, 739.17), text="E = m c^2 (1)")]

    overlap = metrics.geometric_detection(gt, parsed, gt_type="equation", parsed_type="equation",
                                          criterion="overlap")
    assert overlap["criterion"] == "overlap"
    assert overlap["threshold"] == metrics.GEOMETRIC_OVERLAP_THRESHOLD
    assert overlap["match_count"] == 1
    assert overlap["detection"] == 1.0
    # 同一对元素用 IoU 判据必然判不中——这就是公式必须换判据的理由。
    iou = metrics.geometric_detection(gt, parsed, gt_type="equation", parsed_type="equation",
                                      criterion="iou")
    assert iou["match_count"] == 0
    assert iou["detection"] == 0.0
    # 命中记录同时保留得分与原始 IoU，便于解释"IoU 很低却仍算命中"。
    assert overlap["matches"][0]["score"] == 1.0
    assert overlap["matches"][0]["iou"] < metrics.GEOMETRIC_IOU_THRESHOLD


def test_geometric_detection_rejects_unknown_criterion():
    """判据名拼错时必须显式报错，不能静默退化成 IoU 而让产物口径不明。"""

    with pytest.raises(ValueError):
        metrics.geometric_detection([], [], gt_type="equation", criterion="overlap-coefficient")


def test_equation_hits_require_equation_type_on_both_sides():
    """把公式输出成"段落"不得计入公式检出——否则分子分母异空间，精确率会 > 1。"""

    # GT 有 3 个公式，文本各不相同；parser 把这 3 个公式都输出成了 paragraph。
    gt = [
        geom_gt("g1", "equation", 1, (0.0, 0.0, 100.0, 10.0), text="alpha = beta + gamma"),
        geom_gt("g2", "equation", 1, (0.0, 20.0, 100.0, 30.0), text="delta = epsilon / zeta"),
        geom_gt("g3", "equation", 1, (0.0, 40.0, 100.0, 50.0), text="eta = theta - iota"),
    ]
    parsed = [
        geom_parsed("p1", "paragraph", 1, (0.0, 0.0, 100.0, 10.0), text="alpha = beta + gamma"),
        geom_parsed("p2", "paragraph", 1, (0.0, 20.0, 100.0, 30.0), text="delta = epsilon / zeta"),
        geom_parsed("p3", "paragraph", 1, (0.0, 40.0, 100.0, 50.0), text="eta = theta - iota"),
    ]
    result = metrics.equation_metrics(gt, parsed, align.align_elements(gt, parsed))
    # 元素数量与文本都齐，但没有一个是 equation 类型 → 检出为 0，而不是 1.0。
    assert result["parsed_equation_count"] == 0
    assert result["equation_true_positive"] == 0
    assert result["equation_detection_recall"] == 0.0
    assert result["equation_detection_precision"] is None


def test_equation_precision_never_exceeds_one_when_gt_outnumbers_parsed():
    """退化用例：GT 公式多、parser 公式少时，旧实现会算出 precision > 1。"""

    gt = [
        geom_gt("g1", "equation", 1, (0.0, 0.0, 100.0, 10.0), text="a = b + c"),
        geom_gt("g2", "equation", 1, (0.0, 20.0, 100.0, 30.0), text="d = e + f"),
        geom_gt("g3", "equation", 1, (0.0, 40.0, 100.0, 50.0), text="g = h + i"),
    ]
    # parser 只输出了 1 个公式元素；另外把 g2/g3 输出成了段落。
    parsed = [
        geom_parsed("p1", "equation", 1, (0.0, 0.0, 100.0, 10.0), text="a = b + c"),
        geom_parsed("p2", "paragraph", 1, (0.0, 20.0, 100.0, 30.0), text="d = e + f"),
        geom_parsed("p3", "paragraph", 1, (0.0, 40.0, 100.0, 50.0), text="g = h + i"),
    ]
    result = metrics.equation_metrics(gt, parsed, align.align_elements(gt, parsed))
    assert result["equation_true_positive"] == 1
    for key in ("equation_detection_recall", "equation_detection_precision",
                "equation_detection_by_geometry", "equation_detection_by_text"):
        value = result[key]
        assert value is not None and 0.0 <= value <= 1.0, (key, value)


def test_equation_latex_convention_is_disclosed_not_inferred():
    """GT 纯文本、parser 输出 LaTeX 时必须显式标记"约定不可比"，而不是给一个 0 分。"""

    gt = [geom_gt("g1", "equation", 1, (0.0, 0.0, 100.0, 10.0), text="L_enc = - sum_i log P (1)")]
    parsed = [
        geom_parsed(
            "p1", "equation", 1, (10.0, 2.0, 90.0, 8.0),
            text=r"$$ \mathrm { L } \operatorname { e n c } = - \operatorname { s u m } _ { i } $$",
        )
    ]
    result = metrics.equation_metrics(gt, parsed, align.align_elements(gt, parsed))
    assert result["latex_convention"]["gt_latex_like_count"] == 0
    assert result["latex_convention"]["parsed_latex_like_count"] == 1
    assert result["latex_convention"]["comparable"] is False
    # 文本判据因约定不同而失效，但几何判据仍如实判定"检出了"。
    assert result["equation_detection_by_text"] == 0.0
    assert result["equation_detection_by_geometry"] == 1.0
    assert result["equation_detection_recall"] == 1.0


def test_equation_number_and_display_only_read_equation_typed_partners():
    """编号保留 / 行内行间判别不得去读一个被错标成段落的元素。"""

    gt = [
        {
            "element_id": "g1", "type": "equation", "page": 1,
            "bbox": [0.0, 0.0, 100.0, 10.0], "text": "a = b + c",
            "equation_number": "(7)", "display_mode": "display",
        }
    ]
    parsed = [
        geom_parsed("p1", "paragraph", 1, (0.0, 0.0, 100.0, 10.0),
                    text="a = b + c (7)", display_mode="display")
    ]
    result = metrics.equation_metrics(gt, parsed, align.align_elements(gt, parsed))
    # 类型不符时这两个指标的分母为 0 → None（"无法评估"），而不是拿段落去凑一个 1.0。
    assert result["formula_number_preservation"] is None
    assert result["display_vs_inline_accuracy"] is None


# --------------------------------------------------------------------------------------
# --reuse-native 的语义：mapper 修复后复用原生产物，但"到底跑没跑解析"必须能从产物读出来
#
# 存在的理由：mapper / 指标修复不改变 parser 的原生产物，重跑解析纯属浪费（受控矩阵一轮近
# 50 分钟）。复用是安全的（复用不会伪造"我跑过解析"），但"本轮哪些文档是被复用的"
# 必须落进产物，否则后续复核无从判断，也无法解释 §30 的耗时为什么还是旧值。
# --------------------------------------------------------------------------------------


def test_native_has_files_distinguishes_empty_from_missing(tmp_path):
    """空目录（进程跑过但零产出）必须与"根本没跑过"区分开，否则会复用一份空的"产物"。"""

    # 目录不存在 = 该文档从未被解析过。
    assert run_benchmark._native_has_files(tmp_path / "missing") is False
    # 目录存在但一个文件都没有 = 进程起了却零产出（实测出现过 arm 0.4s「完成」的假成功）。
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_benchmark._native_has_files(empty) is False
    # 文件写在子目录里也算数：parser 普遍会嵌套输出，只看顶层文件会漏判。
    nested = empty / "sub"
    nested.mkdir()
    (nested / "out.json").write_text("{}", encoding="utf-8")
    assert run_benchmark._native_has_files(empty) is True


def test_reusable_native_requires_completed_mapping(tmp_path):
    """「有原生产物」还不够，还必须有已完成的映射结果，否则可能复用被中断的半成品。

    为什么要这第二层判据：原生目录在**启动解析之前**就被创建，解析过程中也会持续写入。
    一次被 Ctrl-C 打断的解析同样留下"有文件"的目录；复用它会让映射层读到不完整输入，
    而产物看起来完全正常——这是最危险的一类静默错误。
    """

    native = tmp_path / "native"
    native.mkdir()
    result = tmp_path / "structured" / "doc.json"

    # 情形一：根本没有原生产物 → 不可复用。
    reusable, reason = run_benchmark._reusable_native(tmp_path / "nothing", result)
    assert reusable is False
    assert "没有 parser 产物" in reason

    # 情形二：原生有文件，但缺少已完成映射结果 → 无法证明其完整 → 不可复用（本次新增语义）。
    (native / "blocks.json").write_text("{}", encoding="utf-8")
    reusable, reason = run_benchmark._reusable_native(native, result)
    assert reusable is False
    assert "缺少已完成的映射结果" in reason

    # 情形三：两层判据都满足 → 可复用。注意 result 的父目录不必存在，
    # 判据是"文件是否存在"，而不是"目录结构是否铺好"。
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("{}", encoding="utf-8")
    reusable, reason = run_benchmark._reusable_native(native, result)
    assert reusable is True
    assert reason


def test_previous_parse_seconds_reads_back_last_measured_cost(tmp_path):
    """重映射必须复用上一轮实测耗时，否则 §30 的成本指标会被重映射清空。"""

    result = tmp_path / "doc.json"
    assert run_benchmark._previous_parse_seconds(result) is None
    # 直接用标准库写盘：sandbox.save_json 有写入根目录守卫（会拒绝 pytest 的 tmp_path），
    # 这里要测的是读回逻辑，不该绕过那道守卫。
    result.write_text(json.dumps({"arm_run": {"elapsed_seconds": 12.5}}), encoding="utf-8")
    assert run_benchmark._previous_parse_seconds(result) == 12.5
    result.write_text(json.dumps({"arm_run": {}}), encoding="utf-8")
    assert run_benchmark._previous_parse_seconds(result) is None
    result.write_text(json.dumps({"elapsed_seconds": 3.0}), encoding="utf-8")
    assert run_benchmark._previous_parse_seconds(result) is None
    # 产物损坏时返回 None，而不是抛异常中断整轮重映射。
    result.write_text("{ not json", encoding="utf-8")
    assert run_benchmark._previous_parse_seconds(result) is None


def test_force_and_reuse_native_are_mutually_exclusive():
    """两个开关语义相反（强制重跑解析 / 优先复用原生产物），同时给出必须报错。"""

    with pytest.raises(SystemExit):
        run_benchmark.main(["--force", "--reuse-native", "--scope", "controlled"])


# --------------------------------------------------------------------------------------
# §35 指标异常检查 —— 比例越界只可能是实现缺陷，必须变成可归因的失败记录
#
# 触发这组用例的真实事件：`section_header_f1 = 1.0474`（F1 > 1 数学上不可能），
# 根因是分母用了 set(element_id) 把重复 id 的多个实例折叠成一个。
# --------------------------------------------------------------------------------------


def test_metric_range_violations_flags_impossible_ratio_and_ignores_legit_over_one():
    payload = {
        "section": {"section_header_f1": 1.0474, "section_path_accuracy": None},
        "table": {"table_detected": 0.5},
        # CER 语义上可以 > 1（编辑距离可能超过参考长度），不能被泛化判据误报。
        "text_fidelity": {"character_error_rate_mean": 1.8},
    }
    violations = run_benchmark.metric_range_violations(payload)
    assert [v["metric"] for v in violations] == ["section.section_header_f1"]
    assert violations[0]["value"] == 1.0474


def test_collect_metric_anomalies_is_per_document_and_covers_arm_summary():
    arm = {
        "controlled_metrics": {
            "none::T01": {"section": {"section_header_f1": 1.0474}},
            "none::T02": {"section": {"section_header_f1": 0.9}},
        },
        "real_metrics": {"a.pdf": {"anchors": {"canonical_anchor_recall": 1.2}}},
        "controlled_summary": {"figure_detection": -0.1},
    }
    anomalies = run_benchmark.collect_metric_anomalies(arm)
    kinds = {(r["scope"], r["document"], r["detail"]["metric"]) for r in anomalies}
    assert ("controlled", "T01", "section.section_header_f1") in kinds
    assert ("real", "a.pdf", "anchors.canonical_anchor_recall") in kinds
    # arm 级均值也是比例，同样必须查——否则均值把单文档缺陷抹平后就查不到了。
    assert ("controlled_summary", None, "figure_detection") in kinds
    # 正常文档不得被连带报告。
    assert all(r["document"] != "T02" for r in anomalies)
    # 门 8 要求每条失败都能归因到 PDF/page：三个字段必须齐全。
    assert all({"document", "page", "kind"} <= set(r) for r in anomalies)
    assert all(r["kind"] == "METRIC_OUT_OF_RANGE" for r in anomalies)


# --------------------------------------------------------------------------------------
# §4.1「正文误删」成因分解
#
# 触发这组用例的真实事件：报告里的「读法」结论（"正文被标成装饰全部为 0，
# 因此门 5 对该维度没有区分力"）一度是写死的文案。产物里该字段为 `n/a`（未采集）时，
# 报告仍然宣称了一个数据不支持的事实。结论必须由数据推出来。
# --------------------------------------------------------------------------------------


def test_body_loss_reading_note_distinguishes_zero_from_not_collected():
    """「合计为 0」与「未采集」必须给出不同结论——前者支持"门 5 无区分力"，后者什么都不支持。"""

    zero = run_benchmark._body_loss_reading_note(collected=True, matched_total=0, has_low_coverage=False)
    missing = run_benchmark._body_loss_reading_note(collected=False, matched_total=0, has_low_coverage=False)
    nonzero = run_benchmark._body_loss_reading_note(collected=True, matched_total=7, has_low_coverage=False)

    assert "全部为 0" in zero
    assert "没有区分力" in zero
    # 未采集时不得出现任何"没有区分力/全部为 0"式的结论，只能声明无法判断。
    assert "无法" in missing
    assert "全部为 0" not in missing
    assert "没有区分力" not in missing
    # 非 0 时必须改变结论方向，不能继续宣称"全部为 0"。
    assert "7" in nonzero
    assert "全部为 0" not in nonzero


def test_body_loss_reading_note_mentions_template_attribution_only_when_present():
    """低覆盖存在时才提示按模板归因；否则不该引入一个空指向。"""

    with_low = run_benchmark._body_loss_reading_note(collected=True, matched_total=0, has_low_coverage=True)
    without_low = run_benchmark._body_loss_reading_note(collected=True, matched_total=0, has_low_coverage=False)
    assert "低覆盖集中处" in with_low
    assert "低覆盖集中处" not in without_low


# --------------------------------------------------------------------------------------
# §4.1 聚合（summarise_body_loss）
#
# 触发这组用例的真实事件：报告同一行里，计数是元素级汇总（高 5 / 总 45），
# 占比却复用了指标 JSON 的**逐文档均值**（0.8333）——读者按 5/45 算出 0.11，与 0.83 矛盾。
# 另一个事件：环境阻塞的 arm 没有受控指标，却把「正文被标成装饰」的判定拖成"未采集"。
# --------------------------------------------------------------------------------------


def test_summarise_body_loss_pooled_ratio_agrees_with_same_row_counts():
    """占比必须由同一行的计数现算（元素级），不能沿用逐文档均值。"""

    arms = [
        {
            "arm_id": "A",
            "controlled_metrics": {
                # 5 个模板各 1 个高覆盖元素；1 个模板独占 40 个低覆盖元素（扫描页形态）。
                **{f"none::T{i:02d}": {"furniture": {"body_loss_by_coverage": {"high": 1, "medium": 0, "low": 0}}}
                   for i in range(1, 6)},
                "none::T12": {"furniture": {"body_loss_by_coverage": {"high": 0, "medium": 0, "low": 40}}},
            },
            "controlled_summary": {
                # 逐文档均值：6 个有误删的模板 -> (1.0*5 + 0.0)/6 = 0.8333
                "body_loss_high_coverage_ratio": 0.833333,
                "body_loss_low_coverage_ratio": 0.166667,
                "body_matched_to_furniture": 0.0,
            },
            "controlled": {},
            "cost": {"controlled_runs": 6},
        }
    ]
    result = run_benchmark.summarise_body_loss(arms)
    row = result["rows"][0]
    # 计数：高 5 / 低 40 / 合计 45
    assert row["n"] == 45
    # 占比是元素级：5/45 与 40/45，与同一行的计数自洽（不能是 0.8333）。
    assert row["hi"] == "0.1111"
    assert row["lo"] == "0.8889"
    assert row["focus"].startswith("T12")
    assert result["has_low_coverage"] is True
    assert result["collected"] is True
    assert result["matched_total"] == 0.0


def test_summarise_body_loss_excludes_blocked_arm_from_totals():
    """环境阻塞的 arm 以 `—` 占位，且不得参与 collected/matched_total 的判定。"""

    arms = [
        {
            "arm_id": "OK",
            "controlled_metrics": {
                "none::T01": {"furniture": {"body_loss_by_coverage": {"high": 0, "medium": 0, "low": 0}}}
            },
            "controlled_summary": {"body_matched_to_furniture": 0.0},
            "cost": {"controlled_runs": 1},
        },
        {"arm_id": "BLOCKED", "controlled_metrics": {}, "controlled_summary": {}, "cost": {}},
    ]
    result = run_benchmark.summarise_body_loss(arms)
    blocked_row = result["rows"][1]
    assert blocked_row["n"] == "—"
    assert blocked_row["runs"] == 0
    # 阻塞 arm 没有该字段，但它是"未评估"而非"未采集"——不得把整体判定降级。
    assert result["collected"] is True


def test_summarise_body_loss_reports_not_collected_when_field_absent():
    """已评估的 arm 缺该字段时才是"未采集"，此时结论必须降级。"""

    arms = [
        {
            "arm_id": "OLD",
            "controlled_metrics": {
                "none::T01": {"furniture": {"body_loss_by_coverage": {"high": 1, "medium": 0, "low": 0}}}
            },
            "controlled_summary": {},
            "cost": {"controlled_runs": 1},
        }
    ]
    result = run_benchmark.summarise_body_loss(arms)
    assert result["collected"] is False
    assert result["rows"][0]["mis"] == "n/a"


def test_furniture_metrics_splits_body_loss_by_document_coverage():
    """正文误删必须按"文档级 token 覆盖率"分层，否则分块差异会被误读成内容缺失。"""
    # GT：一段正文（其 token 在文档别处也出现） + 一段正文（token 只此一处，且 parser 完全没输出）。
    ground_truth = [
        gt_element("gt-1", "paragraph", "alpha beta gamma delta"),
        gt_element("gt-2", "paragraph", "uniqueword secretword"),
        gt_element("gt-3", "section_header", "Introduction", page=1),
    ]
    # parser 把 gt-1 与后面一大段不相干的文字合并成了一个块：token 一个不少，但 F1 被精确率压到 0.5 以下，
    # 因此对不上任何单个 GT 元素——这正是"分块/合并差异"的形态（内容在，只是没对齐）。
    filler = " ".join(f"filler{i}" for i in range(1, 11))
    parsed = [
        parsed_element("p-1", "paragraph", f"alpha beta gamma delta {filler}"),
        parsed_element("p-2", "section_header", "Introduction"),
    ]
    alignment = align.align_elements(ground_truth, parsed)
    furniture = metrics.furniture_metrics(ground_truth, parsed, alignment)

    split = furniture["body_loss_by_coverage"]
    # gt-1 的 token 全部能在文档中出现（p-1 含 alpha..delta）→ 高覆盖，属"分块/合并差异"。
    # gt-2 的 token 在 parser 输出里完全不出现 → 低覆盖，属"真实缺失"。
    assert split["high"] >= 1
    assert split["low"] >= 1
    assert split["thresholds"]["high"] == metrics.BODY_LOSS_HIGH_COVERAGE
    # 三层之和必须等于误删总数，否则分解漏掉了元素。
    assert (split["high"] + split["medium"] + split["low"]) == furniture["furniture_false_removal_count"]
    # 三个占比之和恒为 1（有误删时），可用于自证分解没有重复计数。
    assert abs(
        split["high_coverage_ratio"] + split["medium_coverage_ratio"] + split["low_coverage_ratio"] - 1.0
    ) < 1e-9
    # note 必须披露聚合口径（arm 级按均值合并 → 各文档等权）：否则读者会拿逐文档均值去推断
    # 元素级严重程度，实测两者可相差数倍（某文档独占几十个元素时）。
    assert "均值" in split["note"]
    assert "medium" in split["note"]
    # 本用例没有任何"文本对得上却被标成装饰"的元素。
    assert furniture["body_matched_to_furniture"] == 0


def test_body_matched_to_furniture_counts_text_matches_labelled_as_furniture():
    """正文被标成装饰时应计入 `body_matched_to_furniture`，且**不**计入 false_removal。

    这正是 `furniture_false_removal` 测不到的维度：它只看"有没有对齐"，不看"对齐到了什么类型"。
    """

    ground_truth = [gt_element("gt-1", "paragraph", "the quick brown fox jumps over the lazy dog")]
    # parser 把这段正文输出了出来，但类型判成了 footer（装饰）。
    parsed = [parsed_element("p-1", "footer", "the quick brown fox jumps over the lazy dog")]
    alignment = align.align_elements(ground_truth, parsed)
    furniture = metrics.furniture_metrics(ground_truth, parsed, alignment)

    assert furniture["body_matched_to_furniture"] == 1
    # 文本对得上 → 不算"丢失"，因此 false_removal 为 0：门 5 对这类错误完全无感。
    assert furniture["furniture_false_removal_count"] == 0
    assert furniture["furniture_false_removal"] == 0.0




# --------------------------------------------------------------------------------------
# §8 门判定的呈现
#
# 触发这组用例的真实事件：报告把环境阻塞的 `P4_grobid` 写成 `### P4_grobid — FAIL`，
# 与同一份报告里"本机无 Java/Docker，故 P4 未获得任何结构化输出"的声明自相矛盾，
# 会让人把"本机跑不起来"读成"GROBID 能力不达标"。
# --------------------------------------------------------------------------------------


def _gate_arm(arm_id: str, status: str, checks: list[dict]):
    """构造 §8 渲染所需的最小 arm 记录。"""

    passed = sum(1 for c in checks if c["passed"])
    decidable = sum(1 for c in checks if c["passed"] is not None)
    return {
        "arm_id": arm_id,
        "status": status,
        "gate": {
            "checks": checks,
            "passed_count": passed,
            "decidable_count": decidable,
            "gate_passed": passed == decidable and decidable > 0,
        },
    }


def test_blocked_arm_section_is_labelled_blocked_not_fail():
    """环境阻塞的 arm 不得渲染成 FAIL——那是能力结论，阻塞不是。"""

    arm = _gate_arm(
        "P4_grobid",
        "BLOCKED",
        [{"gate": "availability", "passed": False, "evidence": {"available": False}}],
    )
    text = "\n".join(run_benchmark.arm_gate_section(arm))
    assert "BLOCKED（未评估，非工具能力结论）" in text
    assert "— FAIL" not in text
    # 单条门也必须是 BLOCKED，否则表体与标题口径不一致。
    assert "| availability | BLOCKED |" in text
    # 阻塞 arm 不报"通过 x/y 条可判定硬门"，避免把 1 条可用性检查伪装成门矩阵。
    assert "条可判定硬门" not in text


def test_evaluated_arm_section_keeps_pass_fail_and_pending_verdicts():
    """已评估 arm 的三态必须保持 PASS / FAIL / PENDING。"""

    arm = _gate_arm(
        "P0_current",
        "OK",
        [
            {"gate": "g_pass", "passed": True, "evidence": {}},
            {"gate": "g_fail", "passed": False, "evidence": {}},
            {"gate": "g_pending", "passed": None, "evidence": {}},
        ],
    )
    text = "\n".join(run_benchmark.arm_gate_section(arm))
    assert "| g_pass | PASS |" in text
    assert "| g_fail | FAIL |" in text
    assert "| g_pending | PENDING |" in text
    assert "BLOCKED" not in text
    assert "通过 1/2 条可判定硬门。" in text


# --------------------------------------------------------------------------------------
# 报告生成器的端到端回归
#
# 触发这组用例的真实事件：报告里用了一个不存在的 `_format_numeric`，
# 于是整个 §4.1 会在生成时抛 `NameError`。当时只跑完了解析矩阵才发现，
# 只能靠手工冒烟脚本确认。报告生成器是本项目最容易出细节错误的部件
# （未定义名、表格列数错位、结论硬编码），却一直没有端到端测试。
# --------------------------------------------------------------------------------------


def _misaligned_table_rows(text: str) -> list[tuple[int, str, int, int]]:
    """返回表格中列数与表头不一致的行：``(行号, 类型, 表头列数, 实际列数)``。"""

    lines = text.splitlines()
    problems: list[tuple[int, str, int, int]] = []
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        columns = len(line.strip().strip("|").split("|"))
        is_separator = bool(re.fullmatch(r"[\s\-:|]+", line.strip().strip("|")))
        if is_separator:
            back = index - 1
            while back >= 0 and not lines[back].startswith("|"):
                back -= 1
            expected = len(lines[back].strip().strip("|").split("|")) if back >= 0 else -1
            if expected != columns:
                problems.append((index + 1, "SEP", expected, columns))
            continue
        back = index - 1
        while back >= 0 and lines[back].startswith("|"):
            back -= 1
        if back >= 0 and lines[back + 1].startswith("|"):
            expected = len(lines[back + 1].strip().strip("|").split("|"))
            if expected != columns:
                problems.append((index + 1, "ROW", expected, columns))
    return problems


def _minimal_report_arm(arm_id: str, *, blocked: bool = False) -> dict:
    """构造报告渲染所需的最小 arm 记录（一个已评估、一个环境阻塞）。"""

    if blocked:
        return {
            "arm_id": arm_id,
            "tool_version": None,
            "availability": {"available": False, "reason": "本机无 JVM/Docker"},
            "blocked_ocr_modes": {},
            "status": "BLOCKED",
            "blocked_reason": "本机无 JVM/Docker",
            "controlled_summary": {},
            "real_summary": {},
            "controlled_metrics": {},
            "cost": {},
            "gate": {
                "checks": [{"gate": "availability", "passed": False, "evidence": {"available": False}}],
                "passed_count": 0,
                "decidable_count": 1,
                "gate_passed": False,
            },
        }
    return {
        "arm_id": arm_id,
        "tool_version": "tool==1.0",
        "availability": {"available": True, "reason": "venv=sandbox"},
        "blocked_ocr_modes": {},
        "status": "OK",
        "controlled_summary": {
            "paragraph_text_recall": 0.9,
            "furniture_false_removal": 0.05,
            # 全 0 -> 报告应走"该维度无区分力"的分支
            "body_matched_to_furniture": 0.0,
        },
        "real_summary": {"mean_anchor_recall": 1.0, "min_anchor_recall": 1.0, "mean_page_provenance": 1.0},
        "controlled_metrics": {
            "none::T01": {"furniture": {"body_loss_by_coverage": {"high": 1, "medium": 0, "low": 0}}},
            "none::T02": {"furniture": {"body_loss_by_coverage": {"high": 0, "medium": 0, "low": 3}}},
        },
        "cost": {
            "controlled_seconds_per_page": 0.5,
            "real_seconds_per_page": 0.1,
            "peak_ram_mb": 100.0,
            "gpu_required": False,
            "pages_processed": 20,
            "controlled_runs": 2,
        },
        "gate": {
            "checks": [
                {"gate": "g_pass", "passed": True, "evidence": {}},
                {"gate": "g_fail", "passed": False, "evidence": {}},
                {"gate": "g_pending", "passed": None, "evidence": {}},
            ],
            "passed_count": 1,
            "decidable_count": 2,
            "gate_passed": False,
        },
    }


def _minimal_report_payload() -> dict:
    """构造报告生成器的完整最小 payload（字段齐全，值不重要）。"""

    return {
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "sandbox_root": "sandbox",
        "artifact_root": "artifacts",
        "status": "PARTIAL",
        "run_mode": {"reuse_native": True, "remapped_run_count": 2, "parsed_run_count": 0},
        "headline": "一句话结论。",
        "arms": [_minimal_report_arm("A_ok"), _minimal_report_arm("B_blocked", blocked=True)],
        "corpus_self_check": {
            "method": "PDF 文本层自证",
            "passing_template_count": 1,
            "graded_template_count": 2,
            "threshold": 0.98,
            "why_not_p0_based": "不依赖任何 parser",
            "checks": [
                {
                    "template_id": "T01",
                    "excluded": False,
                    "min_page_token_recall": 1.0,
                    "min_bbox_token_recall": 1.0,
                    "mean_bbox_token_recall": 1.0,
                    "max_unclaimed_ratio": 0.0,
                    "passes": True,
                },
                # 扫描页模板必须走"排除（无文本层）"分支，不能被算成失败。
                {"template_id": "T12", "excluded": True},
            ],
        },
        "focus_failure_cases": [
            {
                "question_id": "ARR-007",
                "anchor_id": "A1",
                "file_name": "doc.pdf",
                "page": 3,
                "text": "anchor text",
                "per_arm": [{"hit": True, "best_recall": 1.0}, {"hit": False, "best_recall": 0.0}],
            }
        ],
        "reproducibility": {"template_id": "T01", "reproducible_arm_count": 1, "graded_arm_count": 1},
        "selected_parsers": [{"arm_id": "A_ok", "conditional": True, "why": "有条件候选"}],
        "limitations": ["限制一。"],
    }


def test_report_renders_end_to_end_without_error_and_tables_align(tmp_path, monkeypatch):
    """端到端渲染：不得抛异常（抓未定义名一类缺陷），且所有表格列数自洽。"""

    written: dict[str, str] = {}

    def fake_save_text(text: str, path):
        written["text"] = text
        return path

    monkeypatch.setattr(run_benchmark, "BASE", tmp_path)
    monkeypatch.setattr(run_benchmark.sandbox, "save_text", fake_save_text)

    payload = _minimal_report_payload()
    run_benchmark.write_parser_comparison_report(payload)
    text = written["text"]

    # 章节齐全：缺任何一节都说明渲染中途短路了。
    for heading in ("## 0. ", "## 1. ", "## 2. ", "## 3. ", "## 4. ", "### 4.1 ", "## 5. ",
                    "## 6. ", "## 7. ", "## 8. ", "## 9. ", "## 10. "):
        assert heading in text, heading

    assert _misaligned_table_rows(text) == []


def test_report_marks_blocked_arm_and_reads_body_loss_from_data(tmp_path, monkeypatch):
    """阻塞 arm 标 BLOCKED（非 FAIL），且 §4.1 结论由数据推出。"""

    written: dict[str, str] = {}

    monkeypatch.setattr(run_benchmark, "BASE", tmp_path)
    monkeypatch.setattr(
        run_benchmark.sandbox, "save_text", lambda text, path: written.setdefault("text", text) and path
    )

    run_benchmark.write_parser_comparison_report(_minimal_report_payload())
    text = written["text"]

    # 环境阻塞不是能力结论。
    assert "### `B_blocked` — BLOCKED（未评估，非工具能力结论）" in text
    assert "### `B_blocked` — FAIL" not in text
    # 已评估 arm 保持 PASS/FAIL 语义。
    assert "### `A_ok` — FAIL" in text

    # §4.1：本用例的两个 arm 里，被标成装饰的均值合计为 0 → 应走"无区分力"分支。
    assert "全部为 0" in text
    assert "没有区分力" in text
    # 占比必须与同一行的计数自洽：A_ok 高 1 / 低 3 / 合计 4。
    assert "| `A_ok` | 0.0500 | 4 | 2 | 0.2500 | 0.7500 | 0.0000 |" in text
    # 阻塞 arm 不得用 0 混进统计（0 会被读成"零损失"）。
    assert "| `B_blocked` | — | — | 0 | — | — | — | — |" in text
