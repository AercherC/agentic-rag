"""受控模板语料（Controlled Template Corpus）生成与元素级 Ground Truth（方案 §6）。

## 为什么必须自建语料

真实论文没有可信的元素级 Ground Truth：即使用 GROBID/Docling 作为"参考答案"，也只是
把某个 parser 的判断当成真值，无法判定该 parser 自身的错误。因此方案 §6 要求用**同一份
已知内容**渲染多种版式，从而得到精确的 element-level Ground Truth。

## 实现方式

用 PyMuPDF 逐元素手工排版：引擎自己完成词级换行测量（``pymupdf.get_text_length``），
每个 Ground Truth 元素对应一次 ``insert_textbox`` 调用，因此：

- GT 的 ``text`` 是**渲染前的原文**，不受换行影响；
- GT 的 ``bbox`` 是**实际使用的矩形**；
- GT 的 ``reading_order`` 是排版顺序；双栏时为"左栏自上而下 → 右栏自上而下"；
- GT 的 ``column`` 为 ``L``/``R``（栏内）/ ``F``（跨栏），供双栏交叉检测使用。

这套对应关系有一个可自证的副作用：**P0（当前工程解析器）在这套语料上的保真度必须接近
满分**，否则说明语料本身排版有缺陷。orchestrator 会把这个自证作为语料有效性检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pymupdf

from . import sandbox

# --------------------------------------------------------------------------------------
# 版式模板定义（方案 §6.1 要求的 12 类）
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateSpec:
    """一个论文版式模板的几何与字体参数。"""

    template_id: str
    label: str
    columns: int
    page_size: tuple[float, float]
    body_font: str
    body_size: float
    title_size: float
    header_size: float
    caption_size: float
    table_size: float
    bold_font: str = "tibo"
    leading_ratio: float = 1.22
    margin_left: float = 56.0
    margin_right: float = 56.0
    margin_top: float = 56.0
    margin_bottom: float = 60.0
    gutter: float = 18.0
    running_head: str | None = None
    copyright_footer: str | None = None
    arxiv_sidebar: str | None = None
    watermark: str | None = None
    decorate_left_margin: float = 0.0
    extra_tables: int = 0
    extra_equations: int = 0
    extra_figures: int = 0
    extra_references: int = 0
    rasterize: bool = False

    @property
    def usable_width(self) -> float:
        """去掉左右页边后的可用宽度。"""

        return self.page_size[0] - self.margin_left - self.margin_right

    @property
    def column_width(self) -> float:
        """单栏宽度（跨栏元素用整宽）。"""

        if self.columns == 1:
            return self.usable_width - self.decorate_left_margin
        return (self.usable_width - self.gutter) / 2

    @property
    def full_width(self) -> float:
        """跨栏宽度。"""

        return self.usable_width - self.decorate_left_margin

    @property
    def body_left(self) -> float:
        """正文起始横坐标。"""

        return self.margin_left + self.decorate_left_margin

    @property
    def page_height(self) -> float:
        """页高。"""

        return self.page_size[1]


# 12 类模板（方案 §6.1）。字体只用 PDF 内建 base-14，保证渲染完全确定、无外部字体依赖。
TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        template_id="T01_single_column_plain",
        label="普通单栏论文",
        columns=1,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=10.5,
        title_size=17.0,
        header_size=13.0,
        caption_size=9.0,
        table_size=8.5,
        copyright_footer="Copyright 2026 by the Association for Document Engineering.",
    ),
    TemplateSpec(
        template_id="T02_ieee_two_column",
        label="IEEE-style 双栏",
        columns=2,
        page_size=(612.0, 792.0),
        body_font="tiro",
        body_size=9.5,
        title_size=18.0,
        header_size=11.0,
        caption_size=8.0,
        table_size=8.0,
        running_head="Structure-Aware Scientific Document Ingestion",
        copyright_footer="978-1-6654-0000-0/26/$31.00 (c)2026 IEEE",
    ),
    TemplateSpec(
        template_id="T03_cvpr_two_column",
        label="CVPR/ICCV-style 双栏",
        columns=2,
        page_size=(612.0, 792.0),
        body_font="tiro",
        body_size=10.0,
        title_size=17.0,
        header_size=12.0,
        caption_size=8.5,
        table_size=8.0,
        running_head="CVPR 2026 Submission #1234. Confidential Review Copy.",
        watermark="Preprint. Under review.",
        extra_figures=1,
    ),
    TemplateSpec(
        template_id="T04_acm_two_column",
        label="ACM conference 双栏",
        columns=2,
        page_size=(612.0, 792.0),
        body_font="tiro",
        body_size=9.5,
        title_size=16.0,
        header_size=11.0,
        caption_size=8.0,
        table_size=8.0,
        copyright_footer="ACM ISBN 978-1-4503-XXXX-X/26/07",
    ),
    TemplateSpec(
        template_id="T05_springer_lncs",
        label="Springer LNCS 类单栏",
        columns=1,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=10.0,
        title_size=15.0,
        header_size=12.0,
        caption_size=8.5,
        table_size=8.0,
        copyright_footer="Springer Nature Switzerland AG 2026, LNCS 15000, pp. 1-15, 2026.",
    ),
    TemplateSpec(
        template_id="T06_elsevier_journal",
        label="Elsevier / Journal 类双栏",
        columns=2,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=9.5,
        title_size=16.0,
        header_size=11.0,
        caption_size=8.0,
        table_size=8.0,
        running_head="Pattern Recognition Letters 200 (2026) 1-12",
        copyright_footer="0167-8655 (c) 2026 Elsevier B.V. All rights reserved.",
    ),
    TemplateSpec(
        template_id="T07_arxiv_preprint",
        label="arXiv preprint 单栏（含 arXiv 侧栏）",
        columns=1,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=10.5,
        title_size=16.0,
        header_size=13.0,
        caption_size=9.0,
        table_size=8.5,
        arxiv_sidebar="arXiv:2609.01234v1  [cs.CV]  14 Sep 2026",
        # 侧栏框宽 = decorate_left_margin - 4.0。取 56 而不是 52：token
        # ``arXiv:2609.01234v1`` 在 6pt Times 下宽 49.83pt，52 会给 48pt 框，
        # PyMuPDF 会把它硬拆成两个 token，GT 与文本层因此失配。
        decorate_left_margin=56.0,
    ),
    TemplateSpec(
        template_id="T08_table_heavy",
        label="Table-heavy（4 张表）",
        columns=1,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=10.0,
        title_size=15.0,
        header_size=12.0,
        caption_size=8.5,
        table_size=8.0,
        extra_tables=2,
    ),
    TemplateSpec(
        template_id="T09_formula_heavy",
        label="Formula-heavy（11 个行间公式）",
        columns=1,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=10.5,
        title_size=15.0,
        header_size=12.0,
        caption_size=9.0,
        table_size=8.5,
        extra_equations=8,
    ),
    TemplateSpec(
        template_id="T10_figure_heavy",
        label="Figure-heavy（5 张图）",
        columns=2,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=10.0,
        title_size=15.0,
        header_size=12.0,
        caption_size=8.5,
        table_size=8.0,
        extra_figures=3,
    ),
    TemplateSpec(
        template_id="T11_references_heavy",
        label="References-heavy（72 条参考文献）",
        columns=1,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=10.0,
        title_size=15.0,
        header_size=12.0,
        caption_size=8.5,
        table_size=8.0,
        extra_references=60,
    ),
    TemplateSpec(
        template_id="T12_scanned_image_only",
        label="扫描/纯图像 PDF（无文本层）",
        columns=1,
        page_size=(595.28, 841.89),
        body_font="tiro",
        body_size=10.5,
        title_size=17.0,
        header_size=13.0,
        caption_size=9.0,
        table_size=8.5,
        rasterize=True,
    ),
)

TEMPLATE_INDEX = {spec.template_id: spec for spec in TEMPLATES}

TEMPLATE_ROLE = {
    "T01_single_column_plain": "单栏出生数字 PDF（对照组）",
    "T02_ieee_two_column": "双栏 reading order 主测试",
    "T03_cvpr_two_column": "双栏 + 水印 + 跨栏图",
    "T04_acm_two_column": "双栏 + 版权页脚",
    "T05_springer_lncs": "单栏窄版心 + 出版社页脚",
    "T06_elsevier_journal": "双栏 + 期刊页眉",
    "T07_arxiv_preprint": "单栏 + arXiv 侧栏（furniture 主测试）",
    "T08_table_heavy": "表格结构压力测试",
    "T09_formula_heavy": "公式检测压力测试",
    "T10_figure_heavy": "图与图注压力测试",
    "T11_references_heavy": "参考文献切分压力测试",
    "T12_scanned_image_only": "无文本层 -> OCR 触发条件测试",
}

# --------------------------------------------------------------------------------------
# 内容规格（方案 §6.2 要求的最小结构集合，所有模板共用同一份内容）
# --------------------------------------------------------------------------------------

# 表格结构：``rows`` 为 row-major 单元格文本；``header_row=True`` 表示首行是表头。
TABLE_1: dict[str, Any] = {
    "table_id": "tbl-1",
    "rows": [
        ["Method", "CD", "F1"],
        ["PCN", "8.20", "0.855"],
        ["PoinTr", "7.83", "0.871"],
        ["AdaPoinTr", "7.41", "0.885"],
        ["Ours", "7.05", "0.901"],
    ],
    "header_row": True,
    "has_spanning_cell": False,
    "spanning_rows": [],
}

# 表 2 含跨列单元格（方案 §11.7 要求测 spanning cell）。
TABLE_2: dict[str, Any] = {
    "table_id": "tbl-2",
    "rows": [
        ["Loss ablation on the decoding head", "", ""],
        ["Setting", "CD", "F1"],
        ["Full model", "7.05", "0.901"],
        ["w/o reconstruction loss", "7.62", "0.874"],
        ["w/o consistency loss", "7.71", "0.868"],
    ],
    "header_row": True,
    "has_spanning_cell": True,
    "spanning_rows": [0],
}

TABLE_EXTRA: tuple[dict[str, Any], ...] = (
    {
        "table_id": "tbl-3",
        "rows": [
            ["Backbone", "Params (M)", "CD"],
            ["Transformer-S", "12.4", "7.55"],
            ["Transformer-B", "24.8", "7.05"],
            ["Transformer-L", "48.1", "6.98"],
        ],
        "header_row": True,
        "has_spanning_cell": False,
        "spanning_rows": [],
    },
    {
        "table_id": "tbl-4",
        "rows": [
            ["Resolution", "Points", "CD"],
            ["512", "2048", "7.44"],
            ["1024", "8192", "7.05"],
            ["2048", "32768", "7.01"],
        ],
        "header_row": True,
        "has_spanning_cell": False,
        "spanning_rows": [],
    },
)

# 追加公式（formula-heavy 模板使用），全部为 ASCII 数学，避免 base-14 字体缺字。
EXTRA_EQUATIONS: tuple[tuple[str, str], ...] = (
    ("L_rec = (1/N) sum_i min_j || x_i - y_j ||_2", "(4)"),
    ("L_con = (1/M) sum_k || z_k - z_k_prime ||_2", "(5)"),
    ("A = softmax( (Q K^T) / sqrt(d_k) ) V", "(6)"),
    ("p(y | x) = prod_t p(y_t | y_<t, x)", "(7)"),
    ("G = argmax_G sum_(v in G) w_v", "(8)"),
    ("E = - sum_c q_c log p_c", "(9)"),
    ("d(X, Y) = min_pi sum_i || x_i - y_pi(i) ||_2", "(10)"),
    ("theta_star = argmin_theta L(theta) + lambda || theta ||_2", "(11)"),
)


def _paragraph(text: str, **extra: Any) -> dict[str, Any]:
    """构造一个 paragraph 内容块。"""

    return {"type": "paragraph", "text": text, **extra}


# 核心内容块：按"人类阅读顺序"排列；渲染器负责把它们放进栏/页。
CORE_BLOCKS: tuple[dict[str, Any], ...] = (
    {
        "type": "document_title",
        "text": "Structure-Aware Scientific Document Ingestion for Retrieval-Augmented Question Answering",
    },
    {
        "type": "author_block",
        "text": "Wei Zhang, Ming Li, Sara Novak, Daniel Kim - Institute of Document Engineering, 1 Science Park Road, 100190",
    },
    {
        "type": "abstract",
        "text": (
            "Scientific document ingestion is the bottleneck of retrieval-augmented question "
            "answering over research literature. Born-digital PDFs still lose line structure, mix "
            "captions into body text, and flatten tables into unreadable streams. In this work we "
            "present a controlled benchmark that separates parser errors from chunker errors, and we "
            "introduce an element-level schema that makes reading order, furniture, captions, tables "
            "and equations comparable across five ingestion pipelines. Our experiments on twelve "
            "paper templates show that native text extraction remains the most reliable route for "
            "born-digital PDFs, while conditional page-level OCR is required only for scanned pages."
        ),
    },
    {"type": "section_header", "text": "1 Introduction", "level": 1},
    _paragraph(
        "Retrieval-augmented generation over scientific literature depends on the quality of the "
        "ingested corpus. When a PDF is converted into fixed-size token windows, the retrieval unit "
        "no longer corresponds to a knowledge unit: page numbers, running heads, watermarks and "
        "broken LaTeX fragments enter the index, while figure captions are diluted into paragraphs "
        "that discuss unrelated content. Prior work has shown that ranking metrics degrade whenever "
        "these structural artefacts dominate the embedding of a chunk [3].",
        citation_callouts=["[3]"],
    ),
    _paragraph(
        "The contribution of this paper is threefold. First, we define a single schema that all "
        "ingestion pipelines must emit, and we show that this schema makes heterogeneous parsers "
        "directly comparable. Second, we build a controlled template corpus with element-level ground "
        "truth so that text fidelity, reading order and table structure can be measured without "
        "relying on any parser as a reference. Third, we evaluate conditional OCR strategies and "
        "report when native text extraction should be preferred [12], [7].",
        citation_callouts=["[12]", "[7]"],
    ),
    {"type": "section_header", "text": "2 Related Work", "level": 1},
    _paragraph(
        "Document understanding pipelines such as MinerU, Docling and Marker combine layout "
        "detection with structured serialization. These systems recover section hierarchy, tables "
        "and formulas, but they differ substantially in their treatment of furniture and in whether "
        "they require an external vision-language model [5], [9].",
    ),
    _paragraph(
        "Chunking strategies have evolved from fixed token windows to structure-aware and "
        "parent-child schemes. Earlier work in this project showed that element-aware chunking "
        "isolates captions and therefore repairs caption dilution failures, at the cost of a much "
        "larger number of small retrieval units [2].",
    ),
    {"type": "section_header", "text": "3 Method", "level": 1},
    _paragraph(
        "We model a document as an ordered sequence of typed elements. Each element carries a "
        "reading order, a page index, a bounding box and a section path. Let the page width be W and "
        "the gutter be G; an element is assigned to the left column whenever its box centre c_x "
        "satisfies c_x < (W - G) / 2 [4].",
        inline_math=["c_x < (W - G) / 2"],
    ),
    {"type": "list_item", "text": "(1) Parse the PDF into raw blocks with explicit bounding boxes."},
    {"type": "list_item", "text": "(2) Normalize Unicode and repair hyphenated line breaks conservatively."},
    {"type": "list_item", "text": "(3) Emit typed elements and drop page furniture from the embedding corpus."},
    {"type": "section_header", "text": "3.1 Encoder", "level": 2},
    _paragraph(
        "The encoder consumes the concatenated element text and produces a dense representation. "
        "Each element is embedded separately so that a caption is never averaged together with the "
        "paragraph that follows it [6].",
    ),
    {
        "type": "equation",
        "text": "L_enc = - sum_i log P( e_i | e_less_i )",
        "equation_number": "(1)",
        "display_mode": "display",
    },
    _paragraph(
        "This objective encourages the encoder to model element boundaries rather than surface "
        "fluency, which is what retrieval actually needs.",
        footnote="1",
    ),
    {"type": "section_header", "text": "3.2 Decoder", "level": 2},
    _paragraph(
        "The decoder reconstructs a compact document representation that preserves section "
        "boundaries. Long tables are split row-wise and the header row is repeated in every part.",
    ),
    {
        "type": "equation",
        "text": "H = LayerNorm( x + MultiHead( x, x, x ) )",
        "equation_number": "(2)",
        "display_mode": "display",
    },
    {"type": "figure", "figure_id": "fig-1", "text": ""},
    {
        "type": "figure_caption",
        "text": "Figure 1: Overview of the proposed ingestion pipeline. Raw PDF pages are converted into typed elements, cleaned of page furniture, and chunked along section and element boundaries.",
        "caption_of": "fig-1",
    },
    _paragraph(
        "Figure 1 summarizes the pipeline. The parser produces typed elements, the cleaner removes "
        "page furniture, and the chunker merges sibling paragraphs while isolating captions and "
        "tables [8]."
    ),
    {"type": "section_header", "text": "4 Experiments", "level": 1},
    _paragraph(
        "We evaluate five ingestion pipelines on twelve templates and on a real corpus of ten "
        "point-cloud completion papers. All retrieval experiments use the same dense encoder, the "
        "same query set and the same candidate depth so that only ingestion changes [10].",
    ),
    {"type": "section_header", "text": "4.1 Dataset", "level": 2},
    _paragraph(
        "The controlled corpus contains one logical document rendered under twelve templates with "
        "identical textual content. Every element is annotated with its type, section path and "
        "reading order, which makes parser errors attributable to a specific page and box.",
    ),
    {
        "type": "table_caption",
        "text": "Table 1: Point cloud completion accuracy on the controlled benchmark. Lower CD is better.",
        "caption_of": "tbl-1",
    },
    {"type": "table", "table_structure": TABLE_1, "caption_of": "tbl-1"},
    {"type": "section_header", "text": "4.2 Ablation", "level": 2},
    _paragraph(
        "Table 2 isolates the contribution of each loss term. Removing the consistency loss costs "
        "0.6 CD points, which confirms that structural supervision is not redundant."
    ),
    {
        "type": "equation",
        "text": "L_total = L_cd + lambda_1 L_rec + lambda_2 L_con",
        "equation_number": "(3)",
        "display_mode": "display",
    },
    {
        "type": "table_caption",
        "text": "Table 2: Ablation on the decoding head. The first row spans all columns.",
        "caption_of": "tbl-2",
    },
    {"type": "table", "table_structure": TABLE_2, "caption_of": "tbl-2"},
    {"type": "figure", "figure_id": "fig-2", "text": ""},
    {
        "type": "figure_caption",
        "text": "Figure 2: Qualitative comparison of completion results. Our method preserves thin structures while the baseline produces blobs, and the residual band shows the difference clearly.",
        "caption_of": "fig-2",
        # 该图注是 ARR-007 失败模式的受控复现：caption 短、信息密度低，
        # 一旦与相邻长正文合并成一个 chunk，其 embedding 就被稀释。
        "regression_role": "ARR007_CAPTION_DILUTION_ANALOG",
    },
    _paragraph(
        "Figure 2 shows that qualitative gains are consistent with the quantitative table. We "
        "attribute the improvement to the geometry-aware attention introduced in Section 3.2."
    ),
    {"type": "section_header", "text": "5 Conclusion", "level": 1},
    _paragraph(
        "We presented a schema-first benchmark for scientific document ingestion. The controlled "
        "corpus turns an otherwise subjective question into a measurable one, and it shows that "
        "structure-preserving ingestion is a prerequisite for retrieval quality [1]."
    ),
    {"type": "section_header", "text": "References", "level": 1},
)

# 参考文献条目（受控语料共 12 条；references-heavy 模板追加 60 条）。
CORE_REFERENCES: tuple[str, ...] = (
    '[1] W. Zhang and M. Li, "Structure-aware chunking for retrieval," Journal of Document Engineering, vol. 12, no. 3, pp. 210-225, 2025.',
    '[2] S. Novak, "Element-level evaluation of ingestion pipelines," in Proc. ACM Symposium on Document Engineering, 2024, pp. 45-58.',
    '[3] D. Kim, "Caption dilution in fixed-size chunking," Information Retrieval Journal, vol. 27, pp. 1-19, 2026.',
    '[4] A. Bauer and L. Chen, "Reading order recovery in multi-column PDFs," in Proc. ICDAR, 2023, pp. 301-315.',
    '[5] OpenDataLab, "MinerU: an open-source document parsing toolkit," GitHub repository, 2025.',
    '[6] IBM Research, "Docling: a document conversion library with hierarchical chunking," 2025.',
    '[7] Datalab, "Marker: converting PDF to markdown with layout and OCR models," 2025.',
    '[8] GROBID Team, "GROBID: machine learning software for extracting information from scholarly documents," 2025.',
    '[9] InfiniFlow, "DeepDoc: visual document understanding for RAG pipelines," 2025.',
    '[10] X. Yu et al., "AdaPoinTr: diverse point cloud completion with adaptive geometry-aware transformers," IEEE TPAMI, 2023.',
    '[11] Y. Rao et al., "HyperPCN: hypergraph point completion network," in Proc. CVPR, 2024.',
    '[12] M. Li et al., "Point cloud completion networks: a survey," Pattern Recognition, vol. 150, 2025.',
)

_EXTRA_REFERENCE_TEMPLATE = (
    '[{index}] R. Author, "A survey of structured document understanding, part {index}," '
    "Journal of Applied Document Analysis, vol. {volume}, no. {issue}, pp. {start}-{end}, 20{year}."
)


def _extra_references(count: int) -> list[str]:
    """生成确定性的追加参考文献条目（不用随机数，保证可复现）。"""

    refs: list[str] = []
    for offset in range(count):
        index = 13 + offset
        refs.append(
            _EXTRA_REFERENCE_TEMPLATE.format(
                index=index,
                volume=10 + (offset % 37),
                issue=1 + (offset % 4),
                start=100 + 3 * offset,
                end=100 + 3 * offset + 14,
                year=70 + (offset % 30),
            )
        )
    return refs


def _extra_equations(count: int) -> list[dict[str, Any]]:
    """生成确定性的追加行间公式。"""

    equations: list[dict[str, Any]] = []
    for offset in range(count):
        body, number = EXTRA_EQUATIONS[offset % len(EXTRA_EQUATIONS)]
        text = body if offset < len(EXTRA_EQUATIONS) else f"{body} + variant_{offset}"
        equations.append(
            {
                "type": "equation",
                "text": text,
                "equation_number": number,
                "display_mode": "display",
            }
        )
    return equations


def _extra_figures(count: int) -> list[dict[str, Any]]:
    """生成确定性的追加图与图注。"""

    blocks: list[dict[str, Any]] = []
    for offset in range(count):
        figure_id = f"fig-extra-{offset + 1}"
        blocks.append({"type": "figure", "figure_id": figure_id, "text": ""})
        blocks.append(
            {
                "type": "figure_caption",
                "text": f"Figure {3 + offset}: Additional qualitative result for configuration {offset + 1}. Better viewed when zoomed.",
                "caption_of": figure_id,
            }
        )
    return blocks


def _insert_before_heading(
    blocks: list[dict[str, Any]], needle: str, payload: Sequence[dict[str, Any]]
) -> None:
    """在指定标题之前插入若干块（原地修改）。"""

    for index, block in enumerate(blocks):
        if block.get("type") == "section_header" and needle in block.get("text", ""):
            blocks[index:index] = list(payload)
            return
    blocks.extend(payload)


def build_block_list(spec: TemplateSpec) -> list[dict[str, Any]]:
    """按模板附加量扩展核心内容块，返回该模板的完整排版顺序。"""

    blocks: list[dict[str, Any]] = list(CORE_BLOCKS)

    if spec.extra_equations:
        _insert_before_heading(blocks, "4.2 Ablation", _extra_equations(spec.extra_equations))

    if spec.extra_tables:
        payload: list[dict[str, Any]] = []
        for table in TABLE_EXTRA[: spec.extra_tables]:
            number = int(str(table["table_id"]).split("-")[1])
            payload.append(
                {
                    "type": "table_caption",
                    "text": f"Table {number}: Additional configuration on the controlled benchmark with a longer caption to stress caption isolation.",
                    "caption_of": table["table_id"],
                }
            )
            payload.append(
                {"type": "table", "table_structure": table, "caption_of": table["table_id"]}
            )
        _insert_before_heading(blocks, "References", payload)

    if spec.extra_figures:
        _insert_before_heading(blocks, "5 Conclusion", _extra_figures(spec.extra_figures))

    references: list[str] = list(CORE_REFERENCES)
    if spec.extra_references:
        references.extend(_extra_references(spec.extra_references))
    blocks.extend({"type": "reference_entry", "text": text} for text in references)

    return blocks


# --------------------------------------------------------------------------------------
# 排版引擎
# --------------------------------------------------------------------------------------


def wrap_text(text: str, font: str, size: float, width: float) -> list[str]:
    """按实际字形宽度做词级换行；超长单词允许溢出但不丢弃。"""

    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if pymupdf.get_text_length(trial, fontname=font, fontsize=size) <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [""]


@dataclass
class LayoutCursor:
    """排版游标：当前页 / 当前栏 / 当前纵坐标。"""

    spec: TemplateSpec
    pdf: Any
    page: Any
    page_number: int
    column: int
    y: float

    @property
    def y_top(self) -> float:
        """正文区上边界。"""

        return self.spec.margin_top

    @property
    def y_limit(self) -> float:
        """正文区下边界。"""

        return self.spec.page_height - self.spec.margin_bottom

    def x_left_for(self, span: str) -> float:
        """元素起始横坐标。"""

        if span == "full" or self.spec.columns == 1:
            return self.spec.body_left
        if self.column == 0:
            return self.spec.body_left
        return self.spec.body_left + self.spec.column_width + self.spec.gutter

    def width_for(self, span: str) -> float:
        """元素可用宽度。"""

        return self.spec.full_width if span == "full" else self.spec.column_width

    def new_page(self) -> None:
        """开新页并回到第一栏顶部。"""

        self.page = self.pdf.new_page(width=self.spec.page_size[0], height=self.spec.page_size[1])
        self.page_number += 1
        self.column = 0
        self.y = self.y_top

    def ensure_space(self, height: float, span: str, *, force_new_page: bool = False) -> None:
        """保证当前栏有足够空间；不足则换栏或换页。"""

        if force_new_page:
            self.new_page()
        elif self.y + height > self.y_limit:
            if self.spec.columns == 2 and self.column == 0:
                self.column = 1
                self.y = self.y_top
                if self.y + height > self.y_limit:
                    self.new_page()
            else:
                self.new_page()
        # 跨栏元素只能在页面顶部（栏 0 起点）出现。
        if span == "full" and (self.column != 0 or self.y > self.y_top + 0.01):
            self.new_page()


# --------------------------------------------------------------------------------------
# PyMuPDF 文本框尺寸与安全写入
# --------------------------------------------------------------------------------------

# 表格单元格的行距与底部内边距。行高由 textbox_height() 反推，不再手写经验系数。
CELL_LINEHEIGHT = 1.30
CELL_PADDING_BOTTOM = 3.0

# 页眉/页脚/页码/水印/侧栏的行距。显式给定，保证渲染可复现（不依赖 PyMuPDF 缺省值）。
FURNITURE_LINEHEIGHT = 1.20

# 图内标注文字的行距（图内容最终会被栅格化进位图，仅影响"图长什么样"）。
FIGURE_LINEHEIGHT = 1.20

# ``insert_textbox`` 所需矩形高度的精确公式（由探针实测反推，误差 < 1e-6）：
#
#     required = n_lines * fontsize * lineheight + fontsize * |descender|
#
# 关键事实：**矩形高度不足时 insert_textbox 返回负值，并且一个字都不写**。首版语料
# 正是因为没有检查返回值，让表格体（行高比需求小约 2.4pt）与页码（14pt 框需 14.535pt）
# 在文本层静默消失，自证因此把"排版丢失内容"误报成了其它问题。
#
# 修复策略分两层：
#   1) ``measure_textbox_height`` —— 把"需要多高"交给 PyMuPDF 自己测量（草稿页 + 反推
#      返回值），不再用解析公式猜。这样预排版与最终绘制由同一个引擎决定，不可能不一致。
#   2) ``insert_textbox_checked`` —— 绘制时再次检查返回值，放不下就抛异常，fail fast。
_FONT_CACHE: dict[str, Any] = {}

# 草稿页高度：任何单元素的文本块都不会接近这个值。
_MEASURE_PROBE_HEIGHT = 20000.0


def _font_metrics(fontname: str) -> Any:
    """带缓存的字体度量对象（base-14 字体，度量完全确定）。"""

    if fontname not in _FONT_CACHE:
        _FONT_CACHE[fontname] = pymupdf.Font(fontname)
    return _FONT_CACHE[fontname]


def textbox_height(fontname: str, fontsize: float, lines: int, lineheight: float) -> float:
    """``insert_textbox`` 在该行数下需要的最小矩形高度（解析近似式）。

    仅在需要"先算后画"且已知确切行数时使用；不确定行数时请用
    :func:`measure_textbox_height`。
    """

    return lines * fontsize * lineheight + fontsize * abs(_font_metrics(fontname).descender)


def measure_textbox_height(
    width: float,
    text: str,
    *,
    fontname: str,
    fontsize: float,
    align: int = 0,
    lineheight: float | None = None,
) -> float:
    """用 PyMuPDF 自己的排版引擎测量文本在给定宽度下需要的矩形高度。

    做法是把文本先写到一个高度极大的草稿页上，再用返回值 ``unused`` 反推：
    ``required = probe_height - unused``。草稿页用完即弃，不会污染目标文档。
    这样"实际会画多高"与"预留多高"由同一个排版引擎给出，不存在估算误差。
    """

    kwargs: dict[str, Any] = {"fontname": fontname, "fontsize": fontsize, "align": align}
    if lineheight is not None:
        kwargs["lineheight"] = lineheight
    scratch = pymupdf.open()
    try:
        page = scratch.new_page(width=max(width, 1.0) + 1.0, height=_MEASURE_PROBE_HEIGHT)
        unused = page.insert_textbox(
            pymupdf.Rect(0.0, 0.0, width, _MEASURE_PROBE_HEIGHT), text, **kwargs
        )
    finally:
        scratch.close()
    if unused is None:
        return 0.0
    if unused < 0:
        raise RuntimeError(
            f"measure_textbox_height failed: width={width} font={fontname} "
            f"fontsize={fontsize} unused={unused:.3f}"
        )
    return _MEASURE_PROBE_HEIGHT - unused


def insert_textbox_checked(
    page: Any,
    rect: pymupdf.Rect,
    text: str,
    *,
    fontname: str,
    fontsize: float,
    align: int = 0,
    lineheight: float | None = None,
    color: Any = None,
) -> None:
    """``insert_textbox`` 的安全封装：放不下就抛异常，绝不静默丢字。

    语料生成阶段一旦出现"文本没写进 PDF"，GT 与 PDF 就会永久失配，而后续所有
    parser 指标都建立在这份 GT 上。因此必须 fail fast，而不是依赖自证事后发现。
    """

    kwargs: dict[str, Any] = {"fontname": fontname, "fontsize": fontsize, "align": align}
    if lineheight is not None:
        kwargs["lineheight"] = lineheight
    if color is not None:
        kwargs["color"] = color
    unused = page.insert_textbox(rect, text, **kwargs)
    if unused is not None and unused < 0:
        lines = text.count("\n") + 1
        raise RuntimeError(
            "textbox overflow: "
            f"rect=({rect.x0:.2f}, {rect.y0:.2f}, {rect.x1:.2f}, {rect.y1:.2f}) "
            f"font={fontname} fontsize={fontsize} lines={lines} unused={unused:.3f}"
        )


def ensure_tokens_fit(text: str, *, fontname: str, fontsize: float, width: float, label: str) -> None:
    """确保文本中最长的 token 能完整放进给定宽度，否则直接抛异常。

    PyMuPDF 在单个 token 放不下时会**硬拆 token**（例：``arXiv:2609.01234v1`` 被拆成
    ``arXiv:2609.01234v`` + ``1``）。拆出来的片段与 GT 的 token 不同，会让 GT 与 PDF
    文本层永久失配——自证会表现为 page/bbox 召回不足，但那时语料已经生成。
    因此在渲染之前显式拦截这类"静默拆词"。
    """

    widest = max(
        (pymupdf.get_text_length(token, fontname=fontname, fontsize=fontsize) for token in text.split()),
        default=0.0,
    )
    if widest > width + 1e-6:
        raise RuntimeError(
            f"{label}: longest token needs {widest:.2f}pt but box width is {width:.2f}pt "
            f"(font={fontname}, size={fontsize})"
        )


def _place_text(
    cursor: LayoutCursor,
    text: str,
    *,
    font: str,
    size: float,
    align: int = 0,
    span: str = "col",
    space_before: float = 0.0,
    space_after: float = 2.0,
    force_new_page: bool = False,
) -> tuple[float, float, float, float]:
    """把一个文本元素排入当前栏，返回其实际矩形。"""

    spec = cursor.spec
    width = cursor.width_for(span)
    ensure_tokens_fit(
        text, fontname=font, fontsize=size, width=width, label=f"[{spec.template_id}] text element"
    )
    # 先按栏宽做词级预换行，再把结果交给 PyMuPDF 测量所需高度——预换行保证
    # "排版引擎认为的换行点"是确定的，测量保证"预留高度 == 实际绘制高度"。
    payload = "\n".join(wrap_text(text, font, size, width))
    y_height = measure_textbox_height(
        width, payload, fontname=font, fontsize=size, align=align, lineheight=spec.leading_ratio
    )

    cursor.ensure_space(space_before + y_height + space_after, span, force_new_page=force_new_page)
    cursor.y += space_before

    x_left = cursor.x_left_for(span)
    y_top = cursor.y
    y_bottom = y_top + y_height
    # 文本框比测量宽度宽 1pt：只会减少所需高度，不会造成溢出。GT 的 bbox 直接采用这个
    # "实际占用的排版矩形"——早期实现用 max(行宽) 收缩 x1，隐含假设文本左对齐，对
    # align=1（居中）的元素会切掉右侧一半，居中标题因此被判为 bbox 不完整。
    box_width = width + 1.0
    rect = pymupdf.Rect(x_left, y_top, x_left + box_width, y_bottom + 0.5)
    insert_textbox_checked(
        cursor.page,
        rect,
        payload,
        fontname=font,
        fontsize=size,
        align=align,
        lineheight=spec.leading_ratio,
    )

    cursor.y = y_bottom + space_after
    return x_left, y_top, x_left + box_width, y_bottom


def _reserve_rect(
    cursor: LayoutCursor, height: float, *, span: str, space_before: float = 3.0, space_after: float = 4.0
) -> pymupdf.Rect:
    """为图形/表格预留一块区域并返回其矩形。"""

    cursor.ensure_space(space_before + height + space_after, span)
    cursor.y += space_before
    x_left = cursor.x_left_for(span)
    rect = pymupdf.Rect(x_left, cursor.y, x_left + cursor.width_for(span), cursor.y + height)
    cursor.y = rect.y1 + space_after
    return rect


def _paint_figure_vector(page: Any, rect: pymupdf.Rect, figure_id: str) -> None:
    """在给定页面的 rect 内绘制矢量示意图（不引入任何外部图片资源）。"""

    grey = (0.35, 0.35, 0.35)
    black = (0, 0, 0)

    if figure_id == "fig-1":
        # 架构图：四个方框 + 箭头。
        box_w = rect.width / 4.0
        box_h = 30.0
        y0 = rect.y0 + 18.0
        labels = ["PDF", "Parser", "Cleaner", "Chunker"]
        for index, label in enumerate(labels):
            x0 = rect.x0 + index * (box_w + 6.0)
            page.draw_rect(pymupdf.Rect(x0, y0, x0 + box_w, y0 + box_h), color=black, width=0.8)
            insert_textbox_checked(
                page,
                pymupdf.Rect(x0, y0 + 8.0, x0 + box_w, y0 + box_h),
                label,
                fontname="helv",
                fontsize=7.5,
                align=1,
                lineheight=FIGURE_LINEHEIGHT,
            )
            if index < len(labels) - 1:
                arrow_x = x0 + box_w + 1.0
                page.draw_line(
                    pymupdf.Point(arrow_x, y0 + box_h / 2),
                    pymupdf.Point(arrow_x + 4.5, y0 + box_h / 2),
                    color=grey,
                    width=0.8,
                )
        page.draw_rect(
            pymupdf.Rect(rect.x0, y0 + box_h + 12.0, rect.x1, rect.y1 - 6.0), color=grey, width=0.6
        )
        insert_textbox_checked(
            page,
            pymupdf.Rect(rect.x0, y0 + box_h + 16.0, rect.x1, rect.y1 - 8.0),
            "typed elements -> section aware chunks",
            fontname="helv",
            fontsize=7.0,
            align=1,
            lineheight=FIGURE_LINEHEIGHT,
        )
        return

    # 定性对比网格：2 行 x 3 列，每格画不同密度的点阵。
    cell_w = rect.width / 3.0
    cell_h = rect.height / 2.0
    for row in range(2):
        for col in range(3):
            cell = pymupdf.Rect(
                rect.x0 + col * cell_w,
                rect.y0 + row * cell_h,
                rect.x0 + (col + 1) * cell_w,
                rect.y0 + (row + 1) * cell_h,
            )
            page.draw_rect(cell, color=grey, width=0.5)
            for step in range(3):
                page.draw_circle(
                    pymupdf.Point(cell.x0 + 10 + step * 12, cell.y0 + 14 + row * 6),
                    1.6 + 0.6 * (row + 1),
                    color=black,
                    fill=black,
                )


def _draw_figure(cursor: LayoutCursor, figure_id: str, *, span: str = "col", height: float = 118.0) -> tuple[float, float, float, float]:
    """放置一张图。

    图以**嵌入位图**形式进入 PDF，而不是直接画矢量图形。这不是审美选择而是公平性要求：
    真实论文的图通常是嵌入的栅格图或复杂矢量图，而 layout 模型（Docling / MinerU / Marker）
    都是在真实论文上训练的，"一张图是一块位图对象"是它们识别 picture 区域的重要线索。
    若用简单矢量图形，所有 arm 的 figure 检出都会异常偏低，指标就失去了区分力。

    实现方式是先在一个离屏页面上绘制矢量示意，再栅格化为位图贴回主文档。
    """

    rect = _reserve_rect(cursor, height, span=span)
    offscreen = pymupdf.open()
    try:
        off_page = offscreen.new_page(width=rect.width, height=rect.height)
        _paint_figure_vector(off_page, pymupdf.Rect(0, 0, rect.width, rect.height), figure_id)
        pixmap = off_page.get_pixmap(dpi=150)
    finally:
        offscreen.close()
    cursor.page.insert_image(rect, pixmap=pixmap)
    return rect.x0, rect.y0, rect.x1, rect.y1


def _draw_table(
    cursor: LayoutCursor,
    structure: dict[str, Any],
    *,
    span: str = "col",
    font_size: float,
) -> tuple[float, float, float, float, list[list[list[float]]]]:
    """画一个真实网格表格，返回表格矩形与每个单元格的 bbox。"""

    rows: list[list[str]] = [list(row) for row in structure["rows"]]
    column_count = max(len(row) for row in rows)
    cell_lineheight = CELL_LINEHEIGHT
    # 单元格文本自带的上下内边距。
    cell_inset_x = 2.0
    cell_inset_top = 1.0

    # 列宽按该列最长单元格文本宽度分配，剩余宽度均分给各列。
    available = cursor.width_for(span) - 2.0
    natural: list[float] = []
    for col in range(column_count):
        widest = 0.0
        for row in rows:
            cell = row[col] if col < len(row) else ""
            widest = max(widest, pymupdf.get_text_length(cell, fontname="tiro", fontsize=font_size))
        natural.append(widest + 8.0)
    total_natural = sum(natural) or 1.0
    widths = [value / total_natural * available for value in natural]

    heights: list[float] = []
    wrapped: list[list[list[str]]] = []
    for row_index, row in enumerate(rows):
        row_lines: list[list[str]] = []
        row_needed = 0.0
        for col in range(column_count):
            cell = row[col] if col < len(row) else ""
            cell_width = max(widths[col] - 2.0 * cell_inset_x, 12.0)
            ensure_tokens_fit(
                cell,
                fontname="tiro",
                fontsize=font_size,
                width=cell_width,
                label=f"table cell (row {row_index}, col {col})",
            )
            lines = wrap_text(cell, "tiro", font_size, cell_width)
            row_lines.append(lines)
            # 行高必须容纳 insert_textbox 的真实需求，否则它会静默丢弃整格文字：
            # 早期实现用 max_lines*fontsize*1.30 + 3.0，比真实需求小约 2.4pt，表格体
            # 因此在 PDF 文本层完全消失（自证能发现，但那时已生成了一份"空表"语料）。
            # 这里改为把每个单元格的实际文本交给 PyMuPDF 自己测量，取本行最大值。
            payload = "\n".join(lines) if lines else " "
            row_needed = max(
                row_needed,
                measure_textbox_height(
                    cell_width,
                    payload,
                    fontname="tiro",
                    fontsize=font_size,
                    lineheight=cell_lineheight,
                ),
            )
        wrapped.append(row_lines)
        heights.append(row_needed + cell_inset_top + CELL_PADDING_BOTTOM)

    total_height = sum(heights)
    rect = _reserve_rect(cursor, total_height, span=span, space_before=4.0, space_after=6.0)
    page = cursor.page
    cell_boxes: list[list[list[float]]] = []

    y = rect.y0
    for row_index, row in enumerate(rows):
        row_height = heights[row_index]
        x = rect.x0
        row_boxes: list[list[float]] = []
        for col in range(column_count):
            col_width = widths[col]
            cell_rect = pymupdf.Rect(x, y, x + col_width, y + row_height)
            page.draw_rect(cell_rect, color=(0.55, 0.55, 0.55), width=0.5)
            # 跨列单元格：首行的空单元格不文本化（视觉上是一个合并单元格）。
            text_lines = wrapped[row_index][col]
            if not (row_index in structure.get("spanning_rows", []) and col > 0):
                insert_textbox_checked(
                    page,
                    pymupdf.Rect(
                        x + cell_inset_x,
                        y + cell_inset_top,
                        x + col_width - cell_inset_x,
                        y + row_height,
                    ),
                    "\n".join(text_lines),
                    fontname="tiro",
                    fontsize=font_size,
                    align=0,
                    lineheight=cell_lineheight,
                )
            row_boxes.append([round(x, 2), round(y, 2), round(x + col_width, 2), round(y + row_height, 2)])
            x += col_width
        cell_boxes.append(row_boxes)
        y += row_height

    return rect.x0, rect.y0, rect.x1, rect.y1, cell_boxes


# --------------------------------------------------------------------------------------
# 元素 → Ground Truth
# --------------------------------------------------------------------------------------


def _row_major_text(structure: dict[str, Any]) -> str:
    """表格体的线性化文本（一个朴素文本抽取器会看到的内容）。"""

    return " ".join(cell for row in structure["rows"] for cell in row if cell)


def _gt_element(
    *,
    template_id: str,
    index: int,
    type: str,
    text: str,
    page: int,
    bbox: tuple[float, float, float, float],
    column: str,
    section_path: Sequence[str],
    block: dict[str, Any] | None,
    on_furniture: bool = False,
) -> dict[str, Any]:
    """构造一个 Ground Truth 元素。"""

    block = block or {}
    element: dict[str, Any] = {
        "element_id": f"gt-{template_id}-{index:04d}",
        "type": type,
        "text": text,
        "reading_order": index,
        "page": page,
        "bbox": [round(float(v), 2) for v in bbox],
        "column": column,
        "section_path": list(section_path),
        "on_furniture": on_furniture,
        "caption_of": block.get("caption_of"),
        "figure_id": block.get("figure_id"),
        "table_id": block.get("table_id") or (block.get("table_structure") or {}).get("table_id"),
        "equation_number": block.get("equation_number"),
        "display_mode": block.get("display_mode"),
        "citation_callouts": list(block.get("citation_callouts") or []),
        "footnote": block.get("footnote"),
        "level": block.get("level"),
        "regression_role": block.get("regression_role"),
    }
    if block.get("table_structure"):
        structure = block["table_structure"]
        element["table_structure"] = {
            "rows": [list(row) for row in structure["rows"]],
            "header_row": bool(structure.get("header_row")),
            "has_spanning_cell": bool(structure.get("has_spanning_cell")),
            "spanning_rows": list(structure.get("spanning_rows") or []),
        }
    return element


def _render_furniture(
    cursor_page: Any,
    spec: TemplateSpec,
    page_number: int,
    total_pages: int,
    elements: list[dict[str, Any]],
    template_id: str,
) -> None:
    """在页边距内绘制页眉/页脚/页码/侧栏/水印，并登记为 GT 装饰元素。

    装饰元素的 GT 类型按 13C schema 细分（header / footer / page_number / sidebar /
    watermark），这样"哪一类装饰最难被正确剔除"这个结论才有粒度。
    """

    page = cursor_page
    body_top = spec.margin_top
    body_bottom = spec.page_height - spec.margin_bottom
    lineheight = FURNITURE_LINEHEIGHT

    def register(type: str, text: str, rect: pymupdf.Rect) -> None:
        elements.append(
            _gt_element(
                template_id=template_id,
                index=len(elements),
                type=type,
                text=text,
                page=page_number,
                bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                column="F",
                section_path=[],
                block=None,
                on_furniture=True,
            )
        )

    def place(
        type: str, text: str, *, left: float, top: float, width: float, fontsize: float, align: int, color: Any = None
    ) -> pymupdf.Rect:
        """在固定位置写入一段装饰文本，返回实际使用的矩形。

        高度按"真实换行后的行数"反推，而不是手写常数——①首版给页码写死了 14.0pt，
        而 9pt Times 单行实际需要 14.535pt；②窄框（如 arXiv 侧栏 48pt 宽）会把长文本
        折成多行，写死高度同样会静默丢字。
        """

        lines = wrap_text(text, spec.body_font, fontsize, width)
        payload = "\n".join(lines) if lines else " "
        ensure_tokens_fit(
            text,
            fontname=spec.body_font,
            fontsize=fontsize,
            width=width,
            label=f"[{spec.template_id}] furniture {type}",
        )
        height = (
            measure_textbox_height(
                width, payload, fontname=spec.body_font, fontsize=fontsize, align=align, lineheight=lineheight
            )
            + 2.0
        )
        rect = pymupdf.Rect(left, top, left + width, top + height)
        insert_textbox_checked(
            page,
            rect,
            payload,
            fontname=spec.body_font,
            fontsize=fontsize,
            align=align,
            lineheight=lineheight,
            color=color,
        )
        register(type, text, rect)
        return rect

    # 页码：底部居中。
    if page_number >= 1:
        place(
            "page_number",
            str(page_number),
            left=spec.page_size[0] / 2 - 12.0,
            top=body_bottom + 16.0,
            width=24.0,
            fontsize=9.0,
            align=1,
        )

    # 运行页眉：除首页外每页。
    if spec.running_head and page_number > 1:
        place(
            "header",
            spec.running_head,
            left=spec.body_left,
            top=body_top - 32.0,
            width=spec.full_width,
            fontsize=7.5,
            align=0,
            color=(0.45, 0.45, 0.45),
        )

    # 版权页脚：仅首页。
    if spec.copyright_footer and page_number == 1:
        place(
            "footer",
            spec.copyright_footer,
            left=spec.body_left,
            top=body_bottom + 4.0,
            width=spec.full_width,
            fontsize=6.5,
            align=0,
        )

    # arXiv 侧栏：仅首页，位于左侧页边。
    if spec.arxiv_sidebar and page_number == 1:
        place(
            "sidebar",
            spec.arxiv_sidebar,
            left=spec.margin_left - spec.decorate_left_margin + 2.0,
            top=body_top - 6.0,
            width=spec.decorate_left_margin - 4.0,
            fontsize=6.0,
            align=0,
        )

    # 水印行：仅首页，位于页脚上方。
    if spec.watermark and page_number == 1:
        place(
            "watermark",
            spec.watermark,
            left=spec.body_left,
            top=body_bottom + 20.0,
            width=spec.full_width,
            fontsize=7.0,
            align=1,
            color=(0.6, 0.6, 0.6),
        )


def render_template(spec: TemplateSpec, *, out_dir: Path) -> dict[str, Any]:
    """渲染一个模板，返回 ``{"pdf_path", "ground_truth"}``。"""

    blocks = build_block_list(spec)
    pdf = pymupdf.open()
    first_page = pdf.new_page(width=spec.page_size[0], height=spec.page_size[1])
    cursor = LayoutCursor(
        spec=spec, pdf=pdf, page=first_page, page_number=1, column=0, y=spec.margin_top
    )

    elements: list[dict[str, Any]] = []
    current_path: list[str] = []

    for block in blocks:
        block_type = block["type"]
        span = "full" if (spec.columns == 2 and block_type in {"document_title", "author_block"}) else "col"

        if block_type == "section_header":
            if block.get("level") == 1:
                current_path = [block["text"]]
            else:
                current_path = [current_path[0] if current_path else "Front Matter", block["text"]]
            size = spec.header_size
            rect = _place_text(
                cursor,
                block["text"],
                font=spec.bold_font,
                size=size,
                span=span,
                space_before=8.0,
                space_after=3.0,
            )
            elements.append(
                _gt_element(
                    template_id=spec.template_id,
                    index=len(elements),
                    type="section_header",
                    text=block["text"],
                    page=cursor.page_number,
                    bbox=rect,
                    column=_column_of(cursor, span),
                    section_path=current_path,
                    block=block,
                )
            )
            continue

        if block_type == "document_title":
            rect = _place_text(
                cursor,
                block["text"],
                font=spec.bold_font,
                size=spec.title_size,
                align=1,
                span=span,
                space_before=6.0,
                space_after=8.0,
            )
        elif block_type == "author_block":
            rect = _place_text(
                cursor,
                block["text"],
                font=spec.body_font,
                size=spec.body_size,
                align=1,
                span=span,
                space_before=2.0,
                space_after=10.0,
            )
        elif block_type == "abstract":
            rect = _place_text(
                cursor,
                block["text"],
                font=spec.body_font,
                size=spec.body_size - 0.5,
                span=span,
                space_before=4.0,
                space_after=8.0,
            )
        elif block_type in {"paragraph", "list_item", "reference_entry"}:
            rect = _place_text(
                cursor,
                block["text"],
                font=spec.body_font,
                size=spec.body_size,
                span=span,
                space_before=1.5 if block_type != "reference_entry" else 1.0,
                space_after=3.0,
            )
        elif block_type in {"figure_caption", "table_caption"}:
            rect = _place_text(
                cursor,
                block["text"],
                font=spec.body_font,
                size=spec.caption_size,
                span=span,
                space_before=3.0,
                space_after=4.0,
            )
        elif block_type == "equation":
            # 行间公式与其编号同处一行（真实论文的排版习惯）。GT 的 text 用"朴素抽取器
            # 看到的内容"= 公式体 + 编号：编号若只画不登记，会在文本层成为 GT 未声明的
            # 幽灵文字（自证的第 3 条判据会命中）；结构化访问仍走 equation_number 字段。
            line_text = f"{block['text']}      {block.get('equation_number', '')}".rstrip()
            rect = _place_text(
                cursor,
                line_text,
                font=spec.body_font,
                size=spec.body_size,
                align=1,
                span=span,
                space_before=5.0,
                space_after=5.0,
            )
            block = {**block, "text": line_text}
        elif block_type == "figure":
            rect = _draw_figure(cursor, block["figure_id"], span="col")
        elif block_type == "table":
            rect = _draw_table(cursor, block["table_structure"], span="col", font_size=spec.table_size)[:4]
            block = {**block, "table_id": block["table_structure"]["table_id"]}
            # 表格体的 GT 文本使用行优先线性化结果（朴素抽取器看到的内容）。
            elements.append(
                _gt_element(
                    template_id=spec.template_id,
                    index=len(elements),
                    type="table",
                    text=_row_major_text(block["table_structure"]),
                    page=cursor.page_number,
                    bbox=rect,
                    column=_column_of(cursor, "col"),
                    section_path=current_path,
                    block=block,
                )
            )
            continue
        else:  # pragma: no cover - 内容规格里不存在未知类型
            raise ValueError(f"Unsupported content block type: {block_type}")

        gt_type = block_type if block_type != "figure" else "figure"
        elements.append(
            _gt_element(
                template_id=spec.template_id,
                index=len(elements),
                type=gt_type,
                text=block.get("text", ""),
                page=cursor.page_number,
                bbox=rect,
                column=_column_of(cursor, span),
                section_path=current_path,
                block=block,
            )
        )

    total_pages = pdf.page_count
    for index in range(total_pages):
        _render_furniture(
            pdf[index], spec, index + 1, total_pages, elements, spec.template_id
        )

    # 装饰元素的 reading_order 在其登记顺序之后；重排为"按页 + 登记序"，
    # 但不能破坏正文元素之间原有的先后关系，因此装饰元素统一排在文档末尾。
    for order, element in enumerate(elements):
        element["reading_order"] = order

    out_dir = sandbox.ensure_dir(out_dir)
    pdf_path = out_dir / f"{spec.template_id}.pdf"
    pdf.save(str(pdf_path), garbage=3, deflate=True)
    pdf.close()

    if spec.rasterize:
        _rasterize_in_place(pdf_path)

    return {
        "template_id": spec.template_id,
        "label": spec.label,
        "role": TEMPLATE_ROLE.get(spec.template_id),
        "columns": spec.columns,
        "page_size": list(spec.page_size),
        "rasterized": spec.rasterize,
        "page_count": total_pages,
        "element_count": len(elements),
        "elements": elements,
        "pdf_path": str(pdf_path),
        "sha256": sandbox.sha256_file(pdf_path),
    }


def _column_of(cursor: LayoutCursor, span: str) -> str:
    """返回元素所属栏标记：F（跨栏/单栏）、L（左栏）、R（右栏）。"""

    if span == "full" or cursor.spec.columns == 1:
        return "F"
    return "L" if cursor.column == 0 else "R"


def _rasterize_in_place(pdf_path: Path) -> None:
    """把整份 PDF 栅格化为纯图像页，去掉文本层（模拟扫描件）。

    先写到临时文件再替换原文件：源文档在栅格化期间仍需保持打开读取，
    直接覆盖同一路径会因文件句柄占用而失败。
    """

    source = pymupdf.open(pdf_path)
    target = pymupdf.open()
    for page in source:
        pixmap = page.get_pixmap(dpi=200)
        new_page = target.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, pixmap=pixmap)
    temporary = pdf_path.with_suffix(".raster.tmp.pdf")
    target.save(str(temporary), garbage=3, deflate=True)
    target.close()
    source.close()
    temporary.replace(pdf_path)


# --------------------------------------------------------------------------------------
# 语料装配入口
# --------------------------------------------------------------------------------------


def build_controlled_corpus(*, out_dir: Path) -> dict[str, Any]:
    """渲染全部 12 个模板，返回语料清单（含 Ground Truth 与 PDF 哈希）。"""

    documents: list[dict[str, Any]] = []
    ground_truth: dict[str, Any] = {}
    for spec in TEMPLATES:
        document = render_template(spec, out_dir=out_dir)
        documents.append(document)
        ground_truth[spec.template_id] = {
            "template_id": spec.template_id,
            "label": spec.label,
            "role": TEMPLATE_ROLE.get(spec.template_id),
            "columns": spec.columns,
            "rasterized": spec.rasterize,
            "has_text_layer": not spec.rasterize,
            "page_count": document["page_count"],
            "element_count": document["element_count"],
            "elements": document["elements"],
            "content_key_count": len(build_block_list(spec)),
        }

    return {
        "artifact_type": "phase13c_controlled_template_corpus",
        "created_at_utc": sandbox.utc_now(),
        "template_count": len(documents),
        "core_content_version": "core-v1",
        "shared_content": True,
        "note": (
            "12 个模板共用同一份核心内容（title/abstract/5 节/2 图/2 表/3 行间公式/1 脚注/"
            "引用标注/12 条参考文献）。table-heavy / formula-heavy / figure-heavy / "
            "references-heavy 模板在其上追加对应类型的元素，以形成单类压力测试；"
            "核心元素的 GT 文本在所有模板间完全一致，因此跨模板可比较。"
        ),
        "templates": documents,
        "ground_truth": ground_truth,
    }


def ground_truth_text_corpus(document: dict[str, Any]) -> str:
    """把一篇语料的 GT 文本按阅读顺序拼成单一字符串。"""

    ordered = sorted(document["elements"], key=lambda e: e["reading_order"])
    return " ".join(e["text"] for e in ordered if e["text"])


def export_ground_truth(corpus: dict[str, Any], path: Path) -> Path:
    """导出 Ground Truth 到独立文件（供其他阶段复用与人工审阅）。"""

    payload = {
        "artifact_type": "phase13c_ground_truth",
        "created_at_utc": sandbox.utc_now(),
        "schema": (
            "element_id, type, text, reading_order, page, bbox, column, section_path, "
            "caption_of, figure_id, table_id, equation_number, display_mode, "
            "citation_callouts, footnote, level, regression_role, table_structure"
        ),
        "documents": corpus["ground_truth"],
    }
    return sandbox.save_json(payload, path)
