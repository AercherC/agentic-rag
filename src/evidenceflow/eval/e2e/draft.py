"""派生式题集工具（B3a-2′ D1）：句池、span、answer_keys、接地检查与配额（纯函数）。

本模块只做确定性派生，不读取文件、不发任何请求：
- 句池与 span 选择规则来自 SPEC §13.1(c)(c′) 与 `Claude outputs/06_AUTOMATED_EVAL_DESIGN.md` §2.3；
- 落盘前唯一允许的规范化是 NFKC（`normalize_source_text`），大小写与空白处理统一交给
  `key_match.normalize_text`（§13.1(d)），避免两套规范化漂移；
- `build_evidence_alternative` 复用 `eval/stable_mapping.py` 的 `hyphen_rules` + `normalize_mapping`，
  与 canonical Gold v3 的锚点定位算法保持一致，不修改 `stable_mapping`。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from evidenceflow.domain.evaluation_v2 import (
    ContractModel,
    EvidenceAlternative,
    EvidenceGroup,
    EvidenceLocator,
)
from evidenceflow.eval.e2e.dataset import EvalCase, Provenance, RequiredClaim
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
# 指代/连接词开头的句子缺少自足语义，不能作为独立题源（设计 §2.1）。
_BAD_SENTENCE_STARTS = frozenset(
    {"this", "these", "those", "it", "they", "we", "our", "such", "however", "therefore", "thus", "here"}
)
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
_QUESTION_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("conclusion_limitation", ("limitation", "fail", "cannot", "can not", "drawback", "struggle", "bottleneck")),
    ("ablation", ("ablation", "without", "variant", "baseline")),
    ("experiment_setup", ("dataset", "batch size", "learning rate", "epoch", "optimizer", "trained on", "gpu")),
    ("motivation_concept", ("motivation", "challenge", "problem", "aim", "goal", "we propose", "introduce", "address")),
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

    切分只在“标点后跟空白”处发生，因此 `0.0012s`、`7.21` 这类小数不会被拆开；
    跨行排版的句子（`often\\nincomplete`）合并成一条，落盘后仍是单行文本。
    该函数同时供 ``pool`` 统计“句数”与 ``sentence_candidates`` 使用。
    """

    normalized = normalize_source_text(text)
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


def sentence_candidates(text: str) -> list[str]:
    """从 chunk 正文抽候选句（设计 §2.1 / §2.3）。

    规则：NFKC → 按 ``.!?`` 切句 → 去首尾空白并压成单行；保留 60–320 字符、
    首词不是指代词/连接词（大小写不敏感）、且“含数字 或 含 >=2 个连续大写开头的专名 token”
    的句子；同一句只出现一次（保留首次出现顺序）。
    """

    candidates: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        if not MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS:
            continue
        if _first_word(sentence) in _BAD_SENTENCE_STARTS:
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

    - 数值：取 ``NUMERIC_UNIT_PATTERN`` 的第一个匹配，原始子串长度需 >= 2；
    - 实体：没有可用数值时，取第一个专名序列 ``[A-Z][A-Za-z0-9+\\-]{2,}``；
    - 同时找到数值与实体时返回两者（数值在前），否则只返回找到的一个。
    """

    candidates: list[SpanCandidate] = []
    numeric: SpanCandidate | None = None
    for match in NUMERIC_UNIT_PATTERN.finditer(sentence):
        # 正则里的 `\s*` 在无单位时会吞掉尾随空白，去掉后再确认仍是字面子串。
        span = match.group(0).strip()
        if len(span) >= 2:
            numeric = SpanCandidate(sentence=sentence, span=span, kind="numeric")
            break
    if numeric is not None:
        candidates.append(numeric)

    entity_match = re.search(r"[A-Z][A-Za-z0-9+\-]{2,}", sentence)
    if entity_match is not None and entity_match.group(0) != (numeric.span if numeric else None):
        candidates.append(SpanCandidate(sentence=sentence, span=entity_match.group(0), kind="entity"))
    return candidates[:2]


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
) -> EvalCase:
    """把派生结果组装成 ``EvalCase``（D3 机器验收口径）。

    固定形态：``expected_status=answered``、``acceptable_abstention=False``、``critical=False``、
    证据组 ``G1``、required claim ``C1``（text = 源句、带锚点与 answer_keys）。
    provenance 固定为 ``drafted_by="agent"``、``approved_by="machine"``，``approval_basis``
    由调用方传入（指向闸门报告与 sha256）；``source`` 区分派生方式（如 ``derived_t1``/``derived_t2``）。

    ``span`` 已用于 ``derive_answer_keys``，EvalCase schema 不单独存 span；这里只做
    “span 必须仍在源句中”的防御性检查，避免把错位派生写进题集。
    """

    if span not in sentence:
        raise ValueError(f"span {span!r} is not a substring of the source sentence")
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
        forbidden_claims=[],
        unanswerable_kind=None,
        provenance=Provenance(
            drafted_by="agent",
            approved_by="machine",
            approval_basis=approval_basis,
            source=source,
        ),
    )
