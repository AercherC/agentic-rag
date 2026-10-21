"""派生式题集工具（B3a-2′ D1）：句池、span、answer_keys、接地检查与配额（纯函数）。

本模块只做确定性派生，不读取文件、不发任何请求：
- 句池与 span 选择规则来自 SPEC §13.1(c)(c′) 与 `Claude outputs/06_AUTOMATED_EVAL_DESIGN.md` §2.3；
- 落盘前唯一允许的规范化是 NFKC（`normalize_source_text`），大小写与空白处理统一交给
  `key_match.normalize_text`（§13.1(d)），避免两套规范化漂移；
- `build_evidence_alternative` 复用 `eval/stable_mapping.py` 的 `hyphen_rules` + `normalize_mapping`，
  与 canonical Gold v3 的锚点定位算法保持一致，不修改 `stable_mapping`。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from evidenceflow.domain.evaluation_v2 import (
    ContractModel,
    EvidenceAlternative,
    EvidenceGroup,
    EvidenceLocator,
)
from evidenceflow.eval.e2e.dataset import (
    FALSE_PREMISE_TAG,
    EvalCase,
    ForbiddenClaim,
    Provenance,
    RequiredClaim,
)
from evidenceflow.eval.e2e.key_match import normalize_text
from evidenceflow.eval.stable_mapping import hyphen_rules, normalize_mapping

# 候选句长度窗口（设计 §2.1 的“自足候选句”口径）。
MIN_SENTENCE_CHARS = 60
MAX_SENTENCE_CHARS = 320
# §2.3 派生表：超过 40 字符的长短语不派生别名（key_matchable=false）。
MAX_KEY_SPAN_CHARS = 40
# SPEC §13.1(c′)③：claim 规范化后至少 10 字符。
MIN_CLAIM_CHARS = 10
# SPEC §13.1(c′)④：题干与 claim 的最长公共连续词序列不得超过 8 个词。
MAX_COMMON_WORD_RUN = 8
# §13.1(c′)④ 与 G5 都用规范化后的空白切词，这里统一只按空白切。
_WHITESPACE = re.compile(r"\s+")
# 句子边界：只有标点后跟空白才算边界，避免把小数 `0.0012` 或 `7.21` 切成两段。
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# PDF 抽取会保留印刷断词（`archi-\ntecture`）。必须在压空白**之前**折叠掉「连字符 + 换行 + 缩进」，
# 否则换行先被压成空格会留下 `archi- tecture`，此后任何规范化都无法还原成 `architecture`
# （`key_match` 的 `_LINE_END_HYPHEN` 要求连字符后跟换行，形态已不存在）。
_LINE_END_HYPHEN = re.compile(r"-\s*\n\s*")
# 指代/连接词开头的句子缺少自足语义，不能作为独立题源（设计 §2.1）。
_BAD_SENTENCE_STARTS = frozenset(
    {"this", "these", "those", "it", "they", "we", "our", "such", "however", "therefore", "thus", "here"}
)
# 参考文献条目：`2 [2] Iro Armeni, Sasha Sax, ...` / `5, 7 [8] Thibault Groueix, ...`。
# 这类句子的"答案"是作者名或文献编号，不是论文事实，探针能答中但题目无意义
# （小样校准里 5 条 G6 失败全来自这类句子与 4 条小节标题）。实测占句池 12.2%。
_REFERENCE_ENTRY = re.compile(r"^[\d,\s]{0,8}\[\d+\]\s")
# 公式/表格残片：PDF 抽取会把 LaTeX 公式（`\f rac {\ s }_s`、`s _ i ^l`）与表格行
# （大量数字、行内分隔）切成一"句"。这类文本不是自然语言陈述，接地检查必然失败
# （小样校准里 `claim_grounded` 全部来自公式残片与表头），必须整句排除。
_FORMULA_MARKERS = re.compile(r"\\f\s*rac|\^\{|_\{|\\s\b|\$|\\mathrm|\\mathbf")
# 非字母数字字符占比过高（公式/符号行）与数字过密（表头/结果表）也是残片信号。
_MAX_SYMBOL_RATIO = 0.22
_MAX_DIGIT_RATIO = 0.20
# 小节标题开头：`Experiments In this section, ...` / `Discussion We have ...`。
# PDF 把小节标题与正文首句粘成一行，切句后标题成了句首词；它不含事实，也不能作为答案。
_SECTION_HEADING_STARTS = frozenset({
    "abstract", "introduction", "background", "related", "method", "methods", "approach",
    "experiments", "experiment", "results", "result", "discussion", "conclusion", "conclusions",
    "ablation", "implementation", "appendix", "references", "acknowledgements", "acknowledgments",
    "supplementary",
})
# 语篇连接词开头的句子只是在指代上文，单独拿出来无法自足成题（与 `_BAD_SENTENCE_STARTS` 同源，
# 但这些词常出现在句中，只在句首时排除）。
# 只收"纯过渡、后接指代/结论"的词；`additionally`/`finally`/`overall` 之后常接真实事实句
# （`Additionally, we randomly set 1% ...` 是合法题源），因此不能整句排除。
_BAD_SENTENCE_STARTS_EXTRA = frozenset({
    "further", "furthermore", "moreover", "meanwhile", "similarly",
    "consequently", "besides", "instead", "otherwise",
})
# 专名 token：大写开头、至少 3 字符（`[A-Z][A-Za-z0-9+\-]{2,}` 的 token 化版本）。
_PROPER_TOKEN = re.compile(r"[A-Z][A-Za-z0-9+\-]{1,}")
# 任意字母开头的 token，用于判断“连续大写开头”的专名序列是否中断。
_WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*")
# 数字 + 可选单位。单位按“最长优先”排列：任务书给出的顺序里 `m` 在 `ms` 之前，
# 会让 `2ms` 被截成 `2m`；这里只调整顺序以保持最长匹配，匹配语义（单位集合）不变。
NUMERIC_UNIT_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mm|ms|cm|Hz|GB|MB|%|m|s|K|M|B)?"
)
# 派生别名时拆分数值与单位，单位集合与 `NUMERIC_UNIT_PATTERN` 一致（最长优先）。
_NUMERIC_WITH_UNIT = re.compile(
    r"^(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>mm|ms|cm|Hz|GB|MB|%|m|s|K|M|B)$"
)
# 题型分层用的关键词表（设计 §2.3“按题型词表分层抽样”）；顺序即优先级。
# `implementation` 必须排在 `method` **之前**：两类高度重叠（`we use`、`based on` 都出现），
# 若 `method` 先匹配，`implementation` 永远拿不到题，而 Gate B3 要求 8 类题型各 ≥4 题
# （实测未加该条目时 `implementation` 候选数为 0）。顺序即优先级，具体的先判。
_QUESTION_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("conclusion_limitation", ("limitation", "fail", "cannot", "can not", "drawback", "struggle", "bottleneck")),
    ("ablation", ("ablation", "without", "variant", "baseline")),
    ("experiment_setup", ("dataset", "batch size", "learning rate", "epoch", "optimizer", "trained on", "gpu")),
    ("motivation_concept", ("motivation", "challenge", "problem", "aim", "goal", "we propose", "introduce", "address")),
    ("implementation", (
        "implement", "source code", "we use", "built on", "based on", "framework", "library",
        "pytorch", "tensorflow", "trained with", "in practice", "details", "hyperparameter",
    )),
    ("method", ("architecture", "module", "layer", "network", "algorithm", "design", "we use", "consists of")),
    ("quantitative_results", ("improve", "outperform", "achiev", "accuracy", "percent", "%")),
)
# 规范要求的 8 类题型（SPEC §13.1(a) 的 canonical 切片），配额按全集校验而不是只看出现过的类型。
QUESTION_TYPES: tuple[str, ...] = (
    "method",
    "motivation_concept",
    "quantitative_results",
    "implementation",
    "experiment_setup",
    "ablation",
    "multi_document",
    "conclusion_limitation",
)
# 难度三档：由调用方通过 tags 或单独字段传入（gold chunk 的 Dense 排名，只作标签）。
DIFFICULTY_TAGS: tuple[str, ...] = ("easy", "medium", "hard")
# 两类不可答题的取值（与 `dataset.UnanswerableKind` 一致；这里导出便于构造器与报告共用）。
UNANSWERABLE_NEAR_MISS = "near_miss"
UNANSWERABLE_OUT_OF_CORPUS = "out_of_corpus"
# 不能作为答案的"专名"：句首虚词/连接词，以及图注与小节引用词。
# 实体正则只要求"大写开头 + 至少 3 字符"，因此 `The`（T+he）、`Figure`、`Table` 都会被误判为专名；
# 它们不含任何事实值，一旦成为 answer_keys 会让 key_match 结构性失败。
ENTITY_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "this", "these", "those", "it", "we", "our", "as", "at", "in", "on", "of",
    "for", "and", "or", "but", "if", "since", "when", "where", "while", "with", "from", "by",
    "figure", "fig", "table", "section", "eq", "equation", "appendix", "algorithm", "sec",
    # 语篇连接词与表格/章节标签同样不含事实值（校准里 `Here`、`Further`、`Complexity` 都曾成为答案）。
    "here", "there", "further", "furthermore", "moreover", "additionally", "finally", "meanwhile",
    "similarly", "consequently", "overall", "besides", "instead", "notably", "specifically",
    "complexity", "analysis", "overview", "setup", "results", "method", "methods",
})
# 图注/小节引用词：紧跟其后的纯数字是编号（`Figure 2`、`Section 3`），不是答案。
_REFERENCE_LABELS: frozenset[str] = frozenset({
    "figure", "fig", "table", "section", "sec", "eq", "equation", "appendix", "algorithm",
})
# 实体 span 的原始形态：大写开头、后接至少 2 个字母数字或 `+`/`-`。
_ENTITY_SPAN = re.compile(r"[A-Z][A-Za-z0-9+\-]{2,}")


class SpanCandidate(ContractModel):
    """一个候选答案 span；sentence 为 NFKC（并压成单行）后的源句。"""

    sentence: str
    span: str
    kind: Literal["numeric", "entity"]


class DerivedClaim(ContractModel):
    """一条由源句派生的 claim：text 恒等于源句，answer_keys 为空表示 key_matchable=false。"""

    text: str
    span: str
    answer_keys: list[list[str]]


class QuotaReport(ContractModel):
    """批级配额校验结果；`ok=false` 时 `violations` 逐条给出未达标项。"""

    ok: bool
    violations: list[str]
    counts: dict[str, Any]


def normalize_source_text(text: str) -> str:
    """落盘前唯一允许的规范化：``unicodedata.normalize("NFKC", text)``。

    不做大小写、空白或标点处理：SPEC §13.1(c′) 明确要求落盘文本只做 NFKC，
    大小写/空白差异交给 (d) 的 `normalize_text` 统一消除，否则锚点会与原文对不上。
    """

    return unicodedata.normalize("NFKC", text)


def split_sentences(text: str) -> list[str]:
    """按 ``.!?`` + 空白切句，并把句内换行/多空白压成单个空格。

    处理顺序固定为：NFKC → **折叠行尾断词** → 按 ``.!?`` 切句 → 压空白 → 去首尾空白。
    该顺序不可调换：PDF 抽取保留印刷断词（`archi-\ntecture`），若先压空白会得到
    `archi- tecture`，而 `key_match.normalize_text` 的折叠规则要求「连字符后跟换行」，
    换行既已消失就再也无法还原成 `architecture`，接地检查（§13.1(c′)①）会必然失败。
    实测句池 1951 条候选句中有 572 条（29.3%）受此影响，所以先折叠是正确性问题而非优化。

    切分只在“标点后跟空白”处发生，因此 `0.0012s`、`7.21` 这类小数不会被拆开；
    跨行排版的句子（`often\\nincomplete`）合并成一条，落盘后仍是单行文本。
    该函数同时供 ``pool`` 统计“句数”与 ``sentence_candidates`` 使用。

    已知局限（与 §13.1(d) 一致，由 `tests/test_e2e_key_match.py` 的 xfail 固定）：
    合法连字符复合词若恰在行尾断开（`self-\nattention`），折叠后成为 `selfattention`。
    """

    normalized = normalize_source_text(text)
    # 断词是抽取噪声，不是词义的一部分；必须在压空白之前消除，否则形态被破坏后无法恢复。
    normalized = _LINE_END_HYPHEN.sub("", normalized)
    sentences: list[str] = []
    for raw in _SENTENCE_BOUNDARY.split(normalized):
        # 行内换行属于排版噪声，不是句子内容；压成单空格后锚点才能与 normalize_mapping 对齐。
        sentence = _WHITESPACE.sub(" ", raw).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def _has_proper_noun_run(sentence: str, *, min_run: int = 2) -> bool:
    """是否存在 >= ``min_run`` 个“连续大写开头”的专名 token（数字开头的 token 会中断序列）。"""

    run = 0
    for token in _WORD_TOKEN.findall(sentence):
        if _PROPER_TOKEN.fullmatch(token):
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def _first_word(sentence: str) -> str:
    """取首个小写词，用于过滤指代/连接词开头的非自足句。"""

    match = re.search(r"[A-Za-z]+", sentence)
    return match.group(0).lower() if match else ""


def _is_reference_or_heading(sentence: str) -> bool:
    """判断句子是否为参考文献条目或小节标题粘连句，这类句子不能作为题源。

    参考文献条目（`2 [2] Iro Armeni, Sasha Sax, ...`）的"答案"是作者名/文献号；
    小节标题粘连句（`Experiments In this section, ...`）的"答案"是标题词。
    两者都能被探针答中，但都不是论文事实——校准显示它们会产出
    "第一作者的名是什么"这类没有评测意义的题，且干扰块极易命中（`ambiguous_distractor`）。
    """

    if _REFERENCE_ENTRY.match(sentence):
        return True
    # 只看首词：小节标题被 PDF 抽取粘到正文首句前面，因此首词就是标题本身。
    first_word = _first_word(sentence)
    return first_word in _SECTION_HEADING_STARTS


def _is_noise_fragment(sentence: str) -> bool:
    """判断句子是否为公式/表格残片，这类文本不能作为题源。

    PDF 抽取会把 LaTeX 公式与表格行切成"句子"：
    `average based on inverse distances [22] ... s _ i ^l = \\f rac {\\ s }_s Here, ...`、
    `Complexity Analysis w/ Position Extractor (Params), theoretical computation cost (FLOPs), ...`。
    它们不是自然语言陈述，接地检查（§13.1(c′)①）会因公式被重排而必然失败，
    而且"答案"往往是符号而非事实。用三重信号判定：LaTeX 残留、非字母数字符号占比、
    数字占比——任一超限即排除，避免只靠单条正则漏掉变形写法。
    """

    if _FORMULA_MARKERS.search(sentence):
        return True
    # 占比按"非空白字符"计：空白长度不携带语义，算进去会稀释真实密度。
    visible = [char for char in sentence if not char.isspace()]
    if not visible:
        return True
    symbols = sum(1 for char in visible if not char.isalnum())
    digits = sum(1 for char in visible if char.isdigit())
    if symbols / len(visible) > _MAX_SYMBOL_RATIO:
        return True
    return digits / len(visible) > _MAX_DIGIT_RATIO


def sentence_candidates(text: str) -> list[str]:
    """从 chunk 正文抽候选句（设计 §2.1 / §2.3）。

    规则：NFKC → 折叠行尾断词 → 按 ``.!?`` 切句 → 去首尾空白并压成单行；保留 60–320 字符、
    首词不是指代词/连接词或小节标题（大小写不敏感）、不是参考文献条目、
    且“含数字 或 含 >=2 个连续大写开头的专名 token”的句子；同一句只出现一次（保留首次出现顺序）。
    """

    candidates: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        if not MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS:
            continue
        first_word = _first_word(sentence)
        if first_word in _BAD_SENTENCE_STARTS or first_word in _BAD_SENTENCE_STARTS_EXTRA:
            continue
        # 参考文献与小节标题不是事实陈述，排除后才能保证"答案"是有意义的论文内容。
        if _is_reference_or_heading(sentence):
            continue
        # 公式/表格残片不是自然语言陈述：接地检查会失败，答案也常是符号。
        if _is_noise_fragment(sentence):
            continue
        # 自足性：数字（数值事实）或连续专名（方法/数据集名）至少满足一个，否则题干无法单句可答。
        if not re.search(r"\d", sentence) and not _has_proper_noun_run(sentence):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        candidates.append(sentence)
    return candidates


def numeric_unit_spans(sentence: str) -> list[str]:
    """返回句中全部“数字 + 单位”的原始子串（必须带单位，供 pool 统计使用）。

    与 ``select_spans`` 不同：这里用于统计“含数字+单位句数”（设计 §2.1 的 132 个 chunk 口径），
    裸数字（如 `7.21`）不算，否则统计会把普通数值句也算进去。
    """

    spans: list[str] = []
    for match in NUMERIC_UNIT_PATTERN.finditer(sentence):
        span = match.group(0).strip()
        if span and _NUMERIC_WITH_UNIT.match(span):
            spans.append(span)
    return spans


def select_spans(sentence: str) -> list[SpanCandidate]:
    """从单句选出至多 2 个答案 span（数值优先），每个 span 都是句子的字面子串。

    - 数值：取 ``NUMERIC_UNIT_PATTERN`` 的第一个**可用**匹配，原始子串长度需 >= 2；
      方括号内的纯数字（`[38]`、`[60]`）是引用编号而非事实值，跳过并继续找下一个匹配。
    - 实体：没有可用数值时，取第一个**非停用词**的专名序列 ``[A-Z][A-Za-z0-9+\\-]{2,}``；
      句首虚词（`The`）与图注词（`Figure`/`Table`）不是可命名的答案，必须跳过。
    - 同时找到数值与实体时返回两者（数值在前），否则只返回找到的一个。

    为什么必须过滤：实体正则 `[A-Z][A-Za-z0-9+\\-]{2,}` 会把句首 `The`（`T` + `he`）和
    `Figure` 当成专名。当句内没有长度 >= 2 的数值时（如 `includes 2 PN layers` 的 `2` 被长度规则弃掉），
    返回值就会是 `The` / `Figure`，而它们根本不是答案 —— 小样校准里 10 条有 3 条因此 G6 必败。
    """

    candidates: list[SpanCandidate] = []
    numeric: SpanCandidate | None = None
    for match in NUMERIC_UNIT_PATTERN.finditer(sentence):
        # 正则里的 `\s*` 在无单位时会吞掉尾随空白，去掉后再确认仍是字面子串。
        span = match.group(0).strip()
        if len(span) < 2:
            continue
        # 引用编号 `[38]` 是该论断引用的文献序号，不是被断言的事实值，不能作为答案。
        if _is_citation_number(sentence, match.start(), match.end()):
            continue
        numeric = SpanCandidate(sentence=sentence, span=span, kind="numeric")
        break
    if numeric is not None:
        candidates.append(numeric)

    entity = _first_entity_span(sentence)
    if entity is not None and entity != (numeric.span if numeric else None):
        candidates.append(SpanCandidate(sentence=sentence, span=entity, kind="entity"))
    return candidates[:2]


def _is_citation_number(sentence: str, start: int, end: int) -> bool:
    """判断 ``sentence[start:end]`` 是否是方括号引用编号（`[38]`、`[29,36]`）。

    只认紧邻的方括号：`[38]` 是文献序号，`2 PN layers` 里的 `2` 不是。
    """

    before = sentence[:start].rstrip()
    after = sentence[end:].lstrip()
    return before.endswith("[") and after.startswith(("]", ","))


def _is_distinctive_entity(span: str) -> bool:
    """判断一个词是否带有"专名特征"，而不只是句首被大写的普通词。

    句首位置无法靠大小写区分专名与普通词（`Each`、`Following`、`Despite`、`Let`、`Tenth`
    都是首字母大写化的普通词，却与 `PointNet++`、`PCN-CD`、`MLP` 满足同一个正则）。
    因此要求至少带一个"非句法大写"的特征：
    - 首字母之外还有大写（`SnowflakeNet`、`PointTransformer`）；
    - 含数字或 `+`/`-`（`PN2`、`3D-EPN`、`PointNet++`、`Set-to-Set`）；
    - 全大写缩写且长度 >= 2（`MLP`、`SSG`）。

    句中出现时大小写本身已是强信号，不需要这条规则。
    """

    if any(char.isdigit() for char in span) or "+" in span or "-" in span:
        return True
    if span[1:] != span[1:].lower():
        return True
    return span.isupper() and len(span) >= 2


def _first_entity_span(sentence: str) -> str | None:
    """取第一个可作为答案的专名 span；跳过句首虚词与图注词，跳过图/表编号。

    - 停用词（`The`、`Figure`）不含事实值，跳过；
    - 紧跟在 `Figure`/`Table`/`Section` 等引用词后的纯数字是编号，跳过；
    - **句首位置的候选必须带专名特征**（`_is_distinctive_entity`）：句首必然大写，
      `Each`/`Following`/`Despite`/`Let`/`Tenth` 这些普通词会与真专名混淆；
      只有当它带内部大写、数字或符号时才可信（`PointNet++`、`PCN-CD`、`MLP` 都能通过）。
    没有合格候选时返回 ``None``（调用方据此丢弃该候选，而不是造一个注定命不中的键）。
    """

    for match in _ENTITY_SPAN.finditer(sentence):
        span = match.group(0)
        if span.casefold() in ENTITY_STOPWORDS:
            continue
        # 位置在句首（允许前导引号/括号）时，大写可能只来自句法位置，需额外证据。
        at_start = not sentence[: match.start()].strip(" \t\"'(（[《")
        if at_start and not _is_distinctive_entity(span):
            continue
        # 引用词后紧跟的纯数字（`Figure 2` 的 `2`）是编号；只对纯数字生效，避免误伤 `3D` 这类真值。
        if span.isdigit():
            previous = _ENTITY_SPAN.findall(sentence[: match.start()])
            if previous and previous[-1].casefold() in _REFERENCE_LABELS:
                continue
        return span
    return None


def _dedupe_aliases(aliases: Iterable[str]) -> list[str]:
    """按 ``normalize_text`` 后的形式去重（保留首次出现的原始写法），并丢弃过短/空别名。"""

    kept: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        if not alias.strip():
            continue
        normalized = normalize_text(alias)
        # SPEC §13.1(c)③：规范化后长度 < 2 的别名没有判别力（裸 `1` 会命中所有年份/编号）。
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(alias)
    return kept


def derive_answer_keys(span: str, sentence: str, *, query_language: str = "en") -> list[list[str]]:
    """按派生表为 span 生成一个别名组（组内“或”），返回 ``[[...]]`` 或 ``[]``。

    派生规则（设计 §2.3，顺序固定）：
    1. 原样 span；
    2. 数值 + 单位：去单位（`1.5mm`→`1.5`）与单位前加空格（`1.5 mm`）；
    3. 百分比：`{n}%`→`{n} percent`；``query_language == "zh"`` 时再加 `百分之{n}`；
    4. 含 `-` 的 span：加折叠形（去 `-`）与空格形（`-`→空格）；
    5. span 长度 > 40 字符 ⇒ 返回 ``[]``（key_matchable=false）。

    去重按 ``normalize_text`` 后形式；丢弃规范化后长度 < 2 的别名；结果至少 1 个别名，
    否则返回 ``[]``（调用方仍可建题，但要统计 key_matchable 比例）。

    已知限制（SPEC §13.1(c)③）：裸 `1` 这类长度 1 的别名会被丢弃，所以 `1%` 只能派生出
    `1%` / `1 percent`，不能再用裸数字接住自然回答；`PointNet++` 这类尾部修饰符也不单独剥离
    （任务书派生表只列了连字符、单位与百分比三类）。
    """

    if len(span) > MAX_KEY_SPAN_CHARS:
        return []
    aliases: list[str] = [span]
    numeric = _NUMERIC_WITH_UNIT.match(span)
    if numeric is not None:
        number, unit = numeric.group("number"), numeric.group("unit")
        aliases.append(number)
        # 单位前加空格是自然回答的常见写法（`0.0012 s`）。
        aliases.append(f"{number} {unit}")
        if unit == "%":
            aliases.append(f"{number} percent")
            if query_language == "zh":
                aliases.append(f"百分之{number}")
    # 复合词：PDF 行尾断词时原文可能是 `view-\npoints`，折叠后为 `viewpoints`；
    # 也保留空格形，接住模型把连字符写成空格的情况。
    if "-" in span:
        aliases.append(span.replace("-", ""))
        aliases.append(span.replace("-", " "))
    kept = _dedupe_aliases(aliases)
    # 兜底：所有别名都被判为过短时视为不可派生，避免生成“永远无法命中”的空组。
    return [kept] if kept else []


def three_gram_jaccard(left: str, right: str) -> float:
    """规范化后按空白切词，计算 3-gram 集合的 Jaccard；任一侧不足 3 词时退化为词集合。

    两侧都为空返回 0.0。该函数只用于 G5 的“题干两两去重”，
    不用于题干与 claim 的泄漏判断（那由 ``longest_common_word_run`` 负责，SPEC §13.1(c′)④）。
    """

    left_words = normalize_text(left).split()
    right_words = normalize_text(right).split()
    if not left_words or not right_words:
        return 0.0
    if len(left_words) < 3 or len(right_words) < 3:
        left_set: set[Any] = set(left_words)
        right_set: set[Any] = set(right_words)
    else:
        left_set = {tuple(left_words[index:index + 3]) for index in range(len(left_words) - 2)}
        right_set = {tuple(right_words[index:index + 3]) for index in range(len(right_words) - 2)}
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def longest_common_word_run(left: str, right: str) -> int:
    """返回两段文本规范化后“最长公共连续词序列”的词数（动态规划，O(n*m)）。

    SPEC §13.1(c′)④：> 8 个词即判定题干复述原文。停用词也计入，不做词干化或同义替换；
    任一侧为空返回 0。用连续词长而不是 Jaccard，是因为数值题必须带上原文单位/对象，
    Jaccard 会把“题干与原文共享长数值上下文”误判为泄漏。
    """

    left_words = normalize_text(left).split()
    right_words = normalize_text(right).split()
    if not left_words or not right_words:
        return 0
    # 只保留上一行 DP 结果，空间 O(m)；长度相同则说明字符级连续，等价于词级连续。
    previous = [0] * (len(right_words) + 1)
    best = 0
    for left_word in left_words:
        current = [0] * (len(right_words) + 1)
        for index, right_word in enumerate(right_words, start=1):
            if left_word == right_word:
                current[index] = previous[index - 1] + 1
                if current[index] > best:
                    best = current[index]
        previous = current
    return best


def classify_question_type(sentence: str, span_kind: str) -> str:
    """按题型关键词表给派生题分层（数值 span 一律 quantitative_results）。

    关键词表来自设计 §2.3 的“按题型词表分层抽样”，只用于配额切片，不参与真值判定；
    未命中任何词表时回落到 ``method``（方法描述是语料主体）。
    """

    if span_kind == "numeric":
        return "quantitative_results"
    lowered = sentence.casefold()
    for question_type, keywords in _QUESTION_TYPE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return question_type
    return "method"


def stratified_sample(
    candidates: Sequence[dict[str, Any]],
    *,
    limit: int,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """按「文档 × 题型 × 语言」分层抽样，替代 ``candidates[:limit]`` 的前缀截断。

    为什么不能用前缀截断：候选按 chunk 遍历顺序排列，取前 N 条等于只取前几篇论文的句子。
    实测 1951 条句池取前 200 条时，有两篇论文只有 1 条与 3 条候选，而 Gate B3 要求
    「每篇论文 ≥6 题」——无论生成多少条都会直接判配额失败。分层抽样让有限的 ``limit``
    在各层之间轮转，才可能同时满足每篇与每类的配额。

    为什么用 sha256 而不是 ``random``：题目落盘后要能被复算与审计。``hash()`` 受进程哈希种子
    影响、``random`` 受全局状态影响，两者都会让同一输入在不同进程产出不同样本；
    ``sha256(f"{seed}:{candidate_id}")`` 在任何进程、任何平台都给出同一个排序。

    参数：
    - ``candidates``：``pool`` 产出的候选句行（需含 ``candidate_id``/``document_id``/``sentence``）。
    - ``limit``：要取的总条数；大于总数时返回全部（不重复）。
    - ``seed``：复跑用的独立抽样种子，决定层内排序。

    层键为 ``(document_id, 题型, query_language)``；题型用 ``classify_question_type`` 复算，
    取不到合法 span 的候选单独归入 ``no_span`` 层（它们注定被丢弃，但不该挤占可出题的层）。
    返回顺序稳定：同一输入两次调用逐元素相同。
    """

    if limit < 0:
        raise ValueError("limit must be >= 0")
    if limit == 0 or not candidates:
        return []

    # 先按层分桶，层内用确定性哈希排序；这样同一层内的取用顺序与进程无关。
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        sentence = str(candidate.get("sentence", ""))
        spans = select_spans(sentence)
        question_type = classify_question_type(sentence, spans[0].kind) if spans else "no_span"
        language = str(candidate.get("query_language") or "")
        key = (str(candidate.get("document_id") or ""), question_type, language)
        buckets[key].append(candidate)
    for key, rows in buckets.items():
        rows.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row.get('candidate_id', '')}".encode("utf-8")
        ).hexdigest())

    # 层间轮转取用：每轮各层各取一条，直到取满或全部耗尽。
    # 层顺序同样用 sha256 打散，而不是按 (document_id, 题型) 字典序：字典序会让轮转在样本量
    # 较小时反复回到同一篇论文（实测 limit=10 只覆盖 2 篇），打散后小样本也能跨文档铺开。
    ordered_keys = sorted(
        buckets, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    )
    positions = {key: 0 for key in ordered_keys}
    chosen: list[dict[str, Any]] = []
    while len(chosen) < limit:
        progressed = False
        for key in ordered_keys:
            index = positions[key]
            if index >= len(buckets[key]):
                continue
            chosen.append(buckets[key][index])
            positions[key] = index + 1
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    return chosen


class MultiDocPair(ContractModel):
    """一对可组合成 T3 多文档题的候选（两条来自不同论文、同题型、都可 key_match）。"""

    question_type: str
    candidate_a: dict[str, Any]
    candidate_b: dict[str, Any]


def pair_multi_document_candidates(
    items: Sequence[dict[str, Any]],
    *,
    limit: int,
    seed: int = 1,
) -> list[MultiDocPair]:
    """把单证据候选两两配对成 T3 多文档题的材料（设计 §2.4）。

    配对约束（任一不满足即不配对）：
    1. 两条 `question_type` 相同——设计 §2.4 要求「同题型」组合，否则题干难以自然合并；
    2. 两条 `document_id` 不同——T3 的意义就是"必须两篇证据"；
    3. 两条都 `key_matchable`（`answer_keys` 非空）——否则无法验证"单块只中一半"；
    4. 同一 `sentence` / 同一 `span` 不在输出里复用（G5 已限制同一 span 每篇最多 2 次）。

    排序用 ``sha256(f"{seed}:{doc_a}:{doc_b}")``：同一输入在任何进程都得到同一配对，
    换 ``seed`` 得到独立第二遍。找不到可配对项时返回空列表，由报告统计缺口，不抛错。
    """

    if limit <= 0:
        return []
    # 按题型分桶，桶内只保留可 key_match 的候选；数值型与实体型混排不影响正确性，
    # 因为题干生成只看 sentence 与 span，不看 span 类型。
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("decision") != "accepted":
            continue
        if not item.get("answer_keys"):
            continue
        question_type = str(item.get("question_type") or "")
        if not question_type or question_type == "multi_document":
            # multi_document 是 T3 的结果类型，不能再作为 T3 的输入题型。
            continue
        buckets[question_type].append(item)

    used_sentences: set[str] = set()
    used_spans: set[str] = set()
    pairs: list[MultiDocPair] = []
    # 题型之间轮转，避免配额被单一题型占满（与 stratified_sample 同一思路）。
    ordered_types = sorted(
        buckets, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    )
    cursors = {key: 0 for key in ordered_types}
    while len(pairs) < limit:
        progressed = False
        for question_type in ordered_types:
            if len(pairs) >= limit:
                break
            rows = buckets[question_type]
            index = cursors[question_type]
            while index < len(rows):
                first = rows[index]
                index += 1
                if str(first.get("sentence")) in used_sentences or str(first.get("span")) in used_spans:
                    continue
                partner = _find_partner(
                    rows[index:], first, used_sentences=used_sentences, used_spans=used_spans
                )
                if partner is None:
                    continue
                pairs.append(MultiDocPair(question_type=question_type, candidate_a=first, candidate_b=partner))
                for chosen in (first, partner):
                    used_sentences.add(str(chosen.get("sentence")))
                    used_spans.add(str(chosen.get("span")))
                progressed = True
                break
            cursors[question_type] = index
        if not progressed:
            break
    return pairs


def _find_partner(
    rows: Sequence[dict[str, Any]],
    first: dict[str, Any],
    *,
    used_sentences: set[str],
    used_spans: set[str],
) -> dict[str, Any] | None:
    """在同题型候选里找第一条不同文档、未被占用、且可 key_match 的搭档。"""

    for row in rows:
        if row is first:
            continue
        if str(row.get("document_id")) == str(first.get("document_id")):
            continue
        if not row.get("answer_keys"):
            continue
        if str(row.get("sentence")) in used_sentences or str(row.get("span")) in used_spans:
            continue
        return row
    return None


def replacement_number(value: str, *, context_text: str, max_attempts: int = 3) -> str:
    """按 SPEC §13.1(f) 的数字替换规则生成一个"看起来合理但与原值不同"的数，用于 T6 错误前提题。

    规则逐条对齐合同（这是 B5b 反事实与 T6 共用的确定性构造，不允许自创算法）：
    1. 有小数：最后一位小数 +1；`9` 变 `0` 并向前进位，**位数不变**（`0.19`→`0.20`、`1.99`→`2.00`）；
    2. 整数 >= 10：+1；
    3. **< 10 的整数不选**（如 `7`、`1%`）：替换后与邻近值太接近，容易被模型当噪声忽略；
    4. 新值若已出现在 ``context_text`` 中则再应用一次，最多 ``max_attempts`` 次，仍冲突抛 ``ValueError``；
    5. 保留单位与 `%` 后缀（`1.5mm`→`1.6mm`、`50%`→`51%`）。

    为什么必须保证新值不在原文：T6 的"错误前提"必须能被原文证伪；若篡改值原本就在页文本里，
    题目就同时存在两个"正确"值，D8 的 `forbidden_claims` 也失去了矛盾锚点。
    """

    # 单位/百分号与数值分开处理：派生 span 常带单位（`0.0012s`、`1.5mm`），替换只作用于数值本身。
    match = re.fullmatch(r"(?P<number>\d+(?:\.\d+)?)(?P<suffix>\s*[A-Za-z%]*)", value.strip())
    if match is None:
        raise ValueError(f"not a plain number: {value!r}")
    number, suffix = match.group("number"), match.group("suffix")

    candidate = number
    for _ in range(max_attempts):
        candidate = _increment_number(candidate)
        rendered = f"{candidate}{suffix}"
        # 规范化后再比对：原文里的 `1.6` 与生成值 `1.6` 写法一致，但上下文可能带换行或标签。
        if normalize_text(rendered) not in normalize_text(context_text):
            return rendered
    raise ValueError(f"could not find a non-conflicting replacement for {value!r} in {max_attempts} attempts")


def _increment_number(body: str) -> str:
    """对纯数字串做一次 §13.1(f) 的增量；小数保持位数，整数 < 10 直接拒绝。

    小数用"定点加法"实现：把 `1.99` 视作 199（尺度 100），加 1 得 200，再按原小数位数
    插回小数点得 `2.00`。这样进位与位数保持同时成立，不需要逐位手写进位逻辑。
    """

    if "." in body:
        integer, fraction = body.split(".", 1)
        scale = 10 ** len(fraction)
        scaled = int(integer) * scale + int(fraction) + 1
        # 位数保持不变：整数部分按原值 +1 的进位结果自然增长，小数位固定回原长度。
        whole, rest = divmod(scaled, scale)
        return f"{whole}.{str(rest).rjust(len(fraction), '0')}"
    number = int(body)
    if number < 10:
        raise ValueError(f"integers below 10 are not selected for replacement: {body!r}")
    return str(number + 1)


def build_out_of_corpus_case(
    *,
    case_id: str,
    query: str,
    query_language: str,
    approval_basis: str,
    source_question_id: str,
    tags: list[str] | None = None,
) -> EvalCase:
    """构造 T4 out_of_corpus 不可答题：题目所问实体/事实在整个语料中不存在。

    与 near_miss 的区别只在"实体是否存在"：T4 里连实体都不在语料中（如 `Point-BERT`、
    `A100`），T5 里实体在但属性缺失。两者在 `EvalCase` 上的形态相同——
    `expected_status=insufficient_context`、`acceptable_abstention=False`、
    `unanswerable_kind` 取各自的值、**不得有证据组**（`dataset.py` 的 `unanswerable_shape`）。
    """

    case_tags = list(tags or [])
    marker = f"unanswerable:{UNANSWERABLE_OUT_OF_CORPUS}"
    for tag in (marker, UNANSWERABLE_OUT_OF_CORPUS, f"source:{source_question_id}"):
        if tag not in case_tags:
            case_tags.append(tag)
    return EvalCase(
        case_id=case_id,
        family_id=f"phase12_{source_question_id}",
        query=query,
        query_language=query_language,
        question_type="out_of_corpus",
        tags=case_tags,
        critical=False,
        expected_status="insufficient_context",
        acceptable_abstention=False,
        required_evidence_groups=[],
        required_claims=[],
        forbidden_claims=[],
        unanswerable_kind=UNANSWERABLE_OUT_OF_CORPUS,
        provenance=Provenance(
            drafted_by="phase12",
            approved_by="machine",
            approval_basis=approval_basis,
            source="phase12_unanswerable",
        ),
    )


def build_near_miss_case(
    *,
    case_id: str,
    query: str,
    query_language: str,
    question_type: str,
    entity: str,
    attribute: str,
    approval_basis: str,
    tags: list[str] | None = None,
) -> EvalCase:
    """构造 T5 near_miss 不可答题：实体在语料里，但它与属性的共现为 0。

    SPEC §13.1(a) 的不可答题形态：
    - ``expected_status=insufficient_context``、``acceptable_abstention=False``（D7 强制）；
    - ``unanswerable_kind="near_miss"``；
    - **不得有 ``required_evidence_groups``**（`dataset.py` 的 `unanswerable_shape` 规则明确禁止），
      因此该类题不参与证据映射，也不进入上下文层指标的分母；
    - ``tags`` 带 ``unanswerable:near_miss`` 便于分片统计。

    为什么题干要写明实体：near_miss 的难点是"实体存在但属性缺失"，题干必须让系统能检索到
    该实体所在的文档，否则系统会因"完全找不到"而拒答——那是 out_of_corpus 的行为，
    测不到"检索到了但上下文不足以回答"这个真正要测的能力。
    """

    case_tags = list(tags or [])
    marker = f"unanswerable:{UNANSWERABLE_NEAR_MISS}"
    if marker not in case_tags:
        case_tags.append(marker)
    if UNANSWERABLE_NEAR_MISS not in case_tags:
        case_tags.append(UNANSWERABLE_NEAR_MISS)
    return EvalCase(
        case_id=case_id,
        family_id=f"derived_t5_{attribute.replace(' ', '_')}",
        query=query,
        query_language=query_language,
        question_type=question_type,
        tags=case_tags,
        critical=False,
        expected_status="insufficient_context",
        acceptable_abstention=False,
        required_evidence_groups=[],
        required_claims=[],
        forbidden_claims=[],
        unanswerable_kind=UNANSWERABLE_NEAR_MISS,
        provenance=Provenance(
            drafted_by="agent",
            approved_by="machine",
            approval_basis=approval_basis,
            source="derived_t5",
        ),
    )


def near_miss_query(candidate: NearMissCandidate) -> str:
    """生成 near_miss 题干：点名实体 + 询问未在语料中与它共现的属性。

    题干必须点名实体（见 ``build_near_miss_case`` 的理由），且不得复制源句超过 8 个连续词
    （§13.1(c′)④）——这里只拼实体与属性词，天然不会复制源句。
    """

    if candidate.query_language.lower().startswith("zh"):
        return f"{candidate.entity} 在训练时使用的 {candidate.attribute} 是什么？"
    return f"What {candidate.attribute} does {candidate.entity} use during training?"


def build_false_premise_case(
    *,
    case_id: str,
    family_id: str,
    query: str,
    query_language: str,
    question_type: str,
    sentence: str,
    span: str,
    answer_keys: list[list[str]],
    anchor: Any,
    approval_basis: str,
    false_value: str,
    tags: list[str] | None = None,
    source: str = "derived_t6",
) -> EvalCase:
    """构造 D8 错误前提题：题干含被篡改的值，required claim 写纠正后的原文事实。

    D8 口径（SPEC §13.1(a)）：按**可答题**计分 —— ``expected_status=answered``、
    ``acceptable_abstention=True``（拒答算状态正确）、``tags`` 含 ``false_premise``、
    ``unanswerable_kind=None``；``forbidden_claims`` 至少 1 条，其锚点是与该错误说法矛盾的原文。

    为什么 required claim 用原句：模型纠正前提并给出正确事实才算答对；原句就是那个正确事实，
    因此 ``key_match`` 仍按原 ``answer_keys`` 判定，不需要另一套匹配逻辑。
    """

    case_tags = list(tags or [])
    if FALSE_PREMISE_TAG not in case_tags:
        case_tags.append(FALSE_PREMISE_TAG)
    return EvalCase(
        case_id=case_id,
        family_id=family_id,
        query=query,
        query_language=query_language,
        question_type=question_type,
        tags=case_tags,
        critical=False,
        expected_status="answered",
        acceptable_abstention=True,
        required_evidence_groups=[EvidenceGroup(group_id="G1", alternatives=[anchor])],
        required_claims=[
            RequiredClaim(
                claim_id="C1",
                text=sentence,
                required=True,
                evidence_anchors=[anchor],
                answer_keys=answer_keys,
            )
        ],
        forbidden_claims=[
            ForbiddenClaim(
                claim_id="F1",
                # 错误说法写成与原值的对比形式，便于人工抽查时一眼看出被篡改的是哪个值。
                text=f"the value is {false_value} instead of {span}",
                evidence_anchors=[anchor],
            )
        ],
        unanswerable_kind=None,
        provenance=Provenance(
            drafted_by="agent",
            approved_by="machine",
            approval_basis=approval_basis,
            source=source,
        ),
    )


# T5（near_miss）派生用的"实体 × 属性"属性词表（设计 §2.4 / 计划 B3 表）。
# 选词原则：这些属性在点云补全类论文里**可能被讨论也可能没被讨论**，所以"实体存在但属性缺失"
# 是真实存在的近失情形；不能用论文不可能涉及的属性（那会退化成 out_of_corpus）。
NEAR_MISS_ATTRIBUTES: tuple[str, ...] = (
    "optimizer", "learning rate", "batch size", "epoch", "gpu model", "training time",
    "license", "author affiliation", "hyperparameter", "random seed", "data augmentation",
)


class NearMissCandidate(ContractModel):
    """一条 near_miss 候选：实体来自已有 T1 题，属性是语料中未与该实体共现的邻近属性。"""

    base_candidate_id: str
    entity: str
    attribute: str
    question_type: str
    query_language: str


def plan_near_miss_candidates(
    items: Sequence[dict[str, Any]],
    *,
    corpus_text: str,
    limit: int,
    seed: int = 1,
) -> list[NearMissCandidate]:
    """从已有 T1 题派生 near_miss 候选（T5）：实体存在，但"实体 × 属性"在语料中共现为 0。

    为什么必须从 T1 题派生而不是凭空造：near_miss 的定义是「实体在语料里，缺的只是那个属性」。
    凭空造的实体（如 `Point-BERT`）属于 `out_of_corpus`；两者的区别正是**实体是否出现**，
    这条规则把分类变成可复算的判断，而不是靠作者声明（计划 B3 表 T4/T5）。

    确定性要求：
    - ``corpus_text`` 是 817 个 chunk 的规范化全文，共现判定用 ``normalize_text`` 后的字面子串；
    - 同层（实体）内按 ``sha256(f"{seed}:{entity}:{attribute}")`` 排序，跨实体轮转，
      保证同一输入在同一进程/不同进程都产出同样顺序；
    - 同一实体只出一个候选（避免近失题扎堆在同一篇论文的同一实体）。

    返回空列表是合法结果（说明该批题里没有可用的近失组合），由调用方决定是否补抽。
    """

    if limit <= 0 or not corpus_text:
        return []
    normalized_corpus = normalize_text(corpus_text)
    # 按实体聚合并去重：同一实体重复出现时只保留第一条（按 candidate_id 排序保证确定性）。
    by_entity: dict[str, dict[str, Any]] = {}
    for item in sorted(items, key=lambda row: str(row.get("candidate_id", ""))):
        if item.get("decision") != "accepted" or item.get("source") != "derived_t1":
            continue
        span = str(item.get("span") or "").strip()
        if not span or span.casefold() in ENTITY_STOPWORDS:
            continue
        by_entity.setdefault(span, item)

    chosen: list[NearMissCandidate] = []
    for entity in sorted(by_entity, key=lambda name: hashlib.sha256(
        f"{seed}:{name}".encode("utf-8")).hexdigest()):
        if len(chosen) >= limit:
            break
        # 实体必须有判别力：`Feature`/`Natural`/`Part`/`Failure` 这类普通名词虽被大写，
        # 问"它用什么 optimizer"没有意义，也不是真实的"实体存在"情形。
        # 复用 `_is_distinctive_entity`，与 select_spans 对句首伪专名的判定同一口径。
        if not _is_distinctive_entity(entity):
            continue
        # 实体自身必须确实出现在语料里，否则这题是 out_of_corpus 而不是 near_miss。
        if normalize_text(entity) not in normalized_corpus:
            continue
        item = by_entity[entity]
        for attribute in NEAR_MISS_ATTRIBUTES:
            # 共现判定：实体与属性必须在同一个窗口内出现才算"语料讨论过"。
            # 用 `entity ... attribute` 的宽松顺序对（两者都出现即可视为可能共现），
            # 保守起见要求两者同时出现在同一 chunk 的合并文本里——这里用整语料近似，
            # 偏严：整语料都没有同时出现，才是确定的"共现为 0"。
            if _co_occurs(entity, attribute, corpus_text):
                continue
            chosen.append(
                NearMissCandidate(
                    base_candidate_id=str(item.get("candidate_id", "")),
                    entity=entity,
                    attribute=attribute,
                    question_type=str(item.get("question_type") or "method"),
                    query_language=str(item.get("query_language") or "en"),
                )
            )
            break
    return chosen


def _co_occurs(entity: str, attribute: str, corpus_text: str, *, window: int = 200) -> bool:
    """实体与属性是否在语料中以不超过 ``window`` 字符的距离共现。

    用邻近窗口而不是整语料同时出现：整语料粒度太粗，一篇 100 页论文里"提过实体"和
    "提过 optimizer"几乎必然同时成立，会让 near_miss 候选全被误判为"语料讨论过"。
    窗口按规范化后的字符数计，200 字符约等于一个 chunk 的邻域。
    """

    normalized = normalize_text(corpus_text)
    entity_norm = normalize_text(entity)
    attribute_norm = normalize_text(attribute)
    if not entity_norm or not attribute_norm:
        return False
    start = 0
    while True:
        position = normalized.find(entity_norm, start)
        if position < 0:
            return False
        if attribute_norm in normalized[max(0, position - window): position + len(entity_norm) + window]:
            return True
        start = position + 1


def _case_value(case: Any, name: str, default: Any = None) -> Any:
    """从 EvalCase 或普通 dict 中取字段，便于配额同时服务落盘题集与 JSONL 候选。"""
    if isinstance(case, dict):
        return case.get(name, default)
    return getattr(case, name, default)


def _case_document_ids(case: Any) -> list[str]:
    """题涉及的论文：EvalCase 从证据组锚点取，dict 直接读 ``document_id``。"""

    if isinstance(case, dict):
        document_id = case.get("document_id")
        return [document_id] if document_id else []
    document_ids: list[str] = []
    for group in getattr(case, "required_evidence_groups", []):
        for alternative in group.alternatives:
            if alternative.document_id not in document_ids:
                document_ids.append(alternative.document_id)
    return document_ids


def _case_difficulty(case: Any) -> str | None:
    """难度优先读单独字段，其次读 tags 中的 easy/medium/hard。"""

    explicit = _case_value(case, "difficulty")
    if explicit in DIFFICULTY_TAGS:
        return explicit
    tags = _case_value(case, "tags", []) or []
    for tag in tags:
        if tag in DIFFICULTY_TAGS:
            return tag
    return None


def plan_quotas(
    cases: Sequence[Any],
    *,
    per_document_min: int = 6,
    per_type_min: int = 4,
    hard_ratio_min: float = 0.10,
) -> QuotaReport:
    """统计并校验批级配额，不抛错（设计 §2.5）。

    - 每篇论文 >= ``per_document_min``（一道多文档题计入其涉及的每篇论文）；
    - 8 类 ``question_type`` 每类 >= ``per_type_min``（按规范全集校验，缺失类型同样算违规）；
    - 中英各约一半：``|zh-en| <= max(2, 10% 总数)``；
    - 难度三档各 >= ``hard_ratio_min``（难度由调用方通过 tags 或单独字段传入；
      完全没有难度标签时只报一条“缺标签”，不伪造三档比例）。

    ``counts`` 记录总数与各维度分布，供 ``B3_gate_report.json`` 直接引用。
    """

    total = len(cases)
    document_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    for case in cases:
        for document_id in _case_document_ids(case):
            document_counts[document_id] += 1
        type_counts[str(_case_value(case, "question_type", "") or "")] += 1
        language_counts[str(_case_value(case, "query_language", "") or "")] += 1
        difficulty_counts[_case_difficulty(case) or "unknown"] += 1

    violations: list[str] = []
    for document_id in sorted(document_counts):
        if document_counts[document_id] < per_document_min:
            violations.append(
                f"document {document_id}: {document_counts[document_id]} < {per_document_min}"
            )
    for question_type in QUESTION_TYPES:
        if type_counts.get(question_type, 0) < per_type_min:
            violations.append(
                f"question_type {question_type}: {type_counts.get(question_type, 0)} < {per_type_min}"
            )
    # 语言配额：只把 zh/en 分别计数；无法识别的语言也单独提示，避免被静默算进某一侧。
    zh_count = sum(count for language, count in language_counts.items() if language.lower().startswith("zh"))
    en_count = sum(count for language, count in language_counts.items() if language.lower().startswith("en"))
    language_tolerance = max(2, int(total * 0.1))
    if abs(zh_count - en_count) > language_tolerance:
        violations.append(
            f"language balance: |zh-en| = {abs(zh_count - en_count)} > {language_tolerance}"
        )
    unknown_languages = [
        language for language in language_counts if not language.lower().startswith(("zh", "en"))
    ]
    for language in sorted(unknown_languages):
        violations.append(f"unknown query_language: {language!r}")
    # 难度配额：分母用“有标签的题”，否则缺标签会被误算成 hard；全无标签时只报一条。
    labeled = total - difficulty_counts.get("unknown", 0)
    if total and labeled == 0:
        violations.append("difficulty labels missing: pass tags easy/medium/hard or a difficulty field")
    elif labeled:
        for tier in DIFFICULTY_TAGS:
            ratio = difficulty_counts.get(tier, 0) / labeled
            if ratio < hard_ratio_min:
                violations.append(
                    f"difficulty {tier}: {ratio:.3f} < {hard_ratio_min:.3f} "
                    f"({difficulty_counts.get(tier, 0)}/{labeled})"
                )
    counts = {
        "total": total,
        "documents": dict(sorted(document_counts.items())),
        "question_types": dict(sorted(type_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "difficulty": dict(sorted(difficulty_counts.items())),
    }
    return QuotaReport(ok=not violations, violations=violations, counts=counts)


def page_text(document: Any, page_start: int | None, page_end: int | None) -> str:
    """取 locator 页范围原文；页码为 None（MD/TXT）时取整篇，与 §13.1(c) 口径一致。

    页文本用 ``"\\n"`` 拼接 Segment，与 ``validate_source`` 的 ``raw`` 完全一致，
    这样接地检查与锚点定位看到的是同一份文本。
    """

    if page_start is None:
        return "\n".join(segment.content for segment in document.segments)
    return "\n".join(
        segment.content
        for segment in document.segments
        if segment.page_number is not None and page_start <= segment.page_number <= page_end
    )


def _occurrence_contexts(text: str, start: int, end: int) -> tuple[str | None, str | None]:
    """取锚点左右各 20 字符上下文；空白一侧返回 None（模型要求非空或 null）。"""

    left = text[max(0, start - 20):start].rstrip()
    right = text[end:end + 20].lstrip()
    return left or None, right or None


def build_evidence_alternative(
    document: Any,
    *,
    sentence: str,
    page_start: int | None,
    page_end: int | None,
) -> EvidenceAlternative:
    """在 document 的指定页范围定位 sentence，返回带 locator 的锚点。

    规范化算法与 ``stable_mapping.validate_source`` 完全相同：``hyphen_rules`` 只折叠
    “原页换行断词 + 锚点完整词”共同证明的连字符，``normalize_mapping`` 再做 NFKC、casefold
    与空白折叠。返回的 ``occurrence`` 是规范化页文本中的首次出现序号（1-based），
    ``source_span_start/end`` 是规范化文本中的半开区间，``left_context/right_context``
    各取 20 字符。

    定位不到、或同一句在页内出现多次且 20 字符上下文完全相同（无法区分是哪一处）时抛
    ``ValueError``，由调用方丢弃该候选；绝不猜测位置。
    """

    raw = page_text(document, page_start, page_end)
    rules = hyphen_rules(raw, sentence)
    text = normalize_mapping(raw, rules)
    anchor = normalize_mapping(sentence, rules)
    if not anchor:
        raise ValueError("sentence is empty after normalization")
    positions = [match.start() for match in re.finditer(f"(?={re.escape(anchor)})", text)]
    if not positions:
        raise ValueError(f"sentence not found in pages {page_start}..{page_end}")
    start, end = positions[0], positions[0] + len(anchor)
    left, right = _occurrence_contexts(text, start, end)
    if len(positions) > 1:
        # 出现多次时，只有上下文能把 occurrence 1 与其他出现区分开；完全相同则无法证明定位含义。
        first = (left, right)
        for other in positions[1:]:
            if _occurrence_contexts(text, other, other + len(anchor)) == first:
                raise ValueError(
                    f"sentence appears {len(positions)} times with indistinguishable context"
                )
    locator = EvidenceLocator(
        page_start=page_start,
        page_end=page_end,
        occurrence=1,
        source_span_start=start,
        source_span_end=end,
        left_context=left,
        right_context=right,
    )
    return EvidenceAlternative(
        document_id=document.document_id,
        content_version=document.content_hash,
        locator=locator,
        evidence_anchor=sentence,
    )


def build_case(
    *,
    case_id: str,
    family_id: str,
    query: str,
    query_language: str,
    question_type: str,
    sentence: str,
    span: str,
    answer_keys: list[list[str]],
    anchor: EvidenceAlternative,
    approval_basis: str,
    tags: list[str] | None = None,
    source: str = "derived_t1",
    critical: bool = False,
    extra_evidence_groups: Sequence[EvidenceGroup] = (),
    extra_claims: Sequence[RequiredClaim] = (),
) -> EvalCase:
    """把派生结果组装成 ``EvalCase``（D3 机器验收口径）。

    默认形态（单证据派生题）：``expected_status=answered``、``acceptable_abstention=False``、
    ``critical=False``、证据组 ``G1``、required claim ``C1``（text = 源句、带锚点与 answer_keys）。
    provenance 固定为 ``drafted_by="agent"``、``approved_by="machine"``，``approval_basis``
    由调用方传入（指向闸门报告与 sha256）；``source`` 区分派生方式（如 ``derived_t1``/``derived_t2``）。

    ``extra_evidence_groups`` / ``extra_claims`` 用于多证据题（T3）：它们追加在 ``G1`` / ``C1`` 之后，
    ID 由调用方给定（形如 ``G2`` / ``C2``）。组与 claim 的 ID 必须题内唯一——重复时直接报错而不是
    静默去重，因为 ``validate_dataset`` 会把重复 ID 判为不可冻结，静默去重会掩盖上游配对错误。

    ``span`` 已用于 ``derive_answer_keys``，EvalCase schema 不单独存 span；这里只做
    “span 必须仍在源句中”的防御性检查，避免把错位派生写进题集。
    """

    if span not in sentence:
        raise ValueError(f"span {span!r} is not a substring of the source sentence")
    evidence_groups = [EvidenceGroup(group_id="G1", alternatives=[anchor]), *extra_evidence_groups]
    claims = [
        RequiredClaim(
            claim_id="C1",
            text=sentence,
            required=True,
            evidence_anchors=[anchor],
            answer_keys=answer_keys,
        ),
        *extra_claims,
    ]
    # §13.1(a) 要求组 ID 与 claim ID 题内唯一；在这里提前拒绝，避免写出注定无法冻结的题集。
    for kind, ids in (
        ("evidence group", [group.group_id for group in evidence_groups]),
        ("claim", [claim.claim_id for claim in claims]),
    ):
        duplicated = sorted({item for item in ids if ids.count(item) > 1})
        if duplicated:
            raise ValueError(f"duplicate {kind} id(s): {duplicated}")
    return EvalCase(
        case_id=case_id,
        family_id=family_id,
        query=query,
        query_language=query_language,
        question_type=question_type,
        tags=list(tags or []),
        critical=critical,
        expected_status="answered",
        acceptable_abstention=False,
        required_evidence_groups=evidence_groups,
        required_claims=claims,
        forbidden_claims=[],
        unanswerable_kind=None,
        provenance=Provenance(
            drafted_by="agent",
            approved_by="machine",
            approval_basis=approval_basis,
            source=source,
        ),
    )
