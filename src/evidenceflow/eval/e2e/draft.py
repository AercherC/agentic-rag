"""Derived question-set tooling (B3a-2′ D1): sentence pool, spans, answer_keys, grounding checks, quotas (pure functions).

This module performs deterministic derivation only; it reads no files and issues no requests:
- The sentence-pool and span selection rules come from SPEC §13.1(c)(c′) and
  `Claude outputs/06_AUTOMATED_EVAL_DESIGN.md` §2.3;
- The only normalisation allowed before persisting is NFKC (`normalize_source_text`); case and
  whitespace handling is delegated uniformly to `key_match.normalize_text` (§13.1(d)), so the two
  normalisations never drift apart;
- `build_evidence_alternative` reuses `hyphen_rules` + `normalize_mapping` from
  `eval/stable_mapping.py`, keeping the same anchor-location algorithm as canonical Gold v3 without
  modifying `stable_mapping`.
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

# Candidate-sentence length window (design §2.1's "self-contained candidate sentence" convention).
MIN_SENTENCE_CHARS = 60
MAX_SENTENCE_CHARS = 320
# §2.3 derivation table: a phrase over 40 characters gets no derived aliases (key_matchable=false).
MAX_KEY_SPAN_CHARS = 40
# SPEC §13.1(c′)③: a claim must be at least 10 characters after normalisation.
MIN_CLAIM_CHARS = 10
# SPEC §13.1(c′)④: the longest common consecutive word run of stem and claim is capped at 8 words.
MAX_COMMON_WORD_RUN = 8
# Both §13.1(c′)④ and G5 tokenise on normalised whitespace, so tokenisation is whitespace-only
# here.
_WHITESPACE = re.compile(r"\s+")
# Sentence boundary: only punctuation followed by whitespace counts, so decimals such as
# `0.0012` or `7.21` are never split into two halves.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# PDF extraction preserves printed hyphenation (`archi-\ntecture`). "Hyphen + newline + indent"
# must
# be folded **before** collapsing whitespace, otherwise collapsing first leaves `archi- tecture`
# and
# no later normalisation can restore `architecture` (`key_match`'s `_LINE_END_HYPHEN` requires a
# newline after the hyphen, and that shape no longer exists).
_LINE_END_HYPHEN = re.compile(r"-\s*\n\s*")
# A sentence starting with a pronoun or connective lacks self-contained meaning, so it cannot be an
# independent question source (design §2.1).
_BAD_SENTENCE_STARTS = frozenset(
    {"this", "these", "those", "it", "they", "we", "our", "such", "however", "therefore", "thus", "here"}
)
# Reference entries: `2 [2] Iro Armeni, Sasha Sax, ...` / `5, 7 [8] Thibault Groueix, ...`.
# For these sentences the "answer" is an author name or reference number, not a paper fact; a probe
# can hit them, but the question is meaningless (in the small-sample calibration all 5 G6 failures
# came from such sentences plus 4 section headings). They are 12.2% of the sentence pool.
_REFERENCE_ENTRY = re.compile(r"^[\d,\s]{0,8}\[\d+\]\s")
# Formula/table fragments: PDF extraction turns LaTeX formulas (`\f rac {\ s }_s`, `s _ i ^l`) and
# table rows (many digits, inline separators) into one "sentence". Such text is not a
# natural-language statement, so the grounding check necessarily fails (in the small-sample
# calibration every `claim_grounded` failure came from formula fragments and table headers); the
# whole sentence must be excluded.
_FORMULA_MARKERS = re.compile(r"\\f\s*rac|\^\{|_\{|\\s\b|\$|\\mathrm|\\mathbf")
# A high share of non-alphanumeric characters (formula/symbol lines) and dense digits
# (headers/result tables) are also fragment signals.
_MAX_SYMBOL_RATIO = 0.22
_MAX_DIGIT_RATIO = 0.20
# Section-heading start: `Experiments In this section, ...` / `Discussion We have ...`.
# PDF glues a section heading onto the first body sentence, so after splitting the heading becomes
# the first word; it holds no fact and cannot serve as an answer.
_SECTION_HEADING_STARTS = frozenset({
    "abstract", "introduction", "background", "related", "method", "methods", "approach",
    "experiments", "experiment", "results", "result", "discussion", "conclusion", "conclusions",
    "ablation", "implementation", "appendix", "references", "acknowledgements", "acknowledgments",
    "supplementary",
})
# A sentence starting with a discourse connective only refers back to the preceding text and cannot
# stand alone as a question (same origin as `_BAD_SENTENCE_STARTS`, but these words often appear
# mid-sentence, so they are excluded only at the start).
# Only "purely transitional, followed by a reference or a conclusion" words are collected;
# `additionally`/`finally`/`overall` are often followed by a real factual sentence
# (`Additionally, we randomly set 1% ...` is a legitimate question source), so they must not be
# excluded wholesale.
_BAD_SENTENCE_STARTS_EXTRA = frozenset({
    "further", "furthermore", "moreover", "meanwhile", "similarly",
    "consequently", "besides", "instead", "otherwise",
})
# Proper-noun token: starts uppercase, at least 3 characters (the tokenised form of
# `[A-Z][A-Za-z0-9+\-]{2,}`).
_PROPER_TOKEN = re.compile(r"[A-Z][A-Za-z0-9+\-]{1,}")
# A token starting with any letter, used to tell whether a run of consecutive capitalised proper
# nouns is interrupted.
_WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*")
# Number + optional unit. Units are ordered "longest first": the task brief lists `m` before `ms`,
# which would truncate `2ms` to `2m`; only the order is adjusted here to keep longest-match, and
# the matching semantics (the unit set) are unchanged.
NUMERIC_UNIT_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mm|ms|cm|Hz|GB|MB|%|m|s|K|M|B)?"
)
# Deriving aliases splits the value from the unit, with the same unit set as `NUMERIC_UNIT_PATTERN`
# (longest first).
_NUMERIC_WITH_UNIT = re.compile(
    r"^(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>mm|ms|cm|Hz|GB|MB|%|m|s|K|M|B)$"
)
# Keyword table for question-type stratification (design §2.3, "stratified sampling by
# question-type
# vocabulary"); order is priority.
# `implementation` must come **before** `method`: the two overlap heavily (`we use` and `based on`
# both appear), and if `method` matches first, `implementation` never gets a question, while Gate
# B3
# requires >= 4 questions for each of the 8 types (measured: without this entry the
# `implementation`
# candidate count is 0). Order is priority, so the specific type is decided first.
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
# The 8 question types the spec requires (the canonical slice of SPEC §13.1(a)); quotas are
# validated against the full set, not only the types that occurred.
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
# Three difficulty tiers, supplied by the caller via tags or a dedicated field (the Dense rank of
# the gold chunk; a label only).
DIFFICULTY_TAGS: tuple[str, ...] = ("easy", "medium", "hard")
# Values of the two unanswerable kinds (consistent with `dataset.UnanswerableKind`; exported here
# so the builders and the report share them).
UNANSWERABLE_NEAR_MISS = "near_miss"
UNANSWERABLE_OUT_OF_CORPUS = "out_of_corpus"
# "Proper nouns" that cannot be answers: sentence-initial function words/connectives, plus caption
# and section-reference words.
# The entity regex requires only "starts uppercase + at least 3 characters", so `The` (T+he),
# `Figure`, and `Table` are all misread as proper nouns; none of them carries a factual value, and
# once they become answer_keys they make key_match fail structurally.
ENTITY_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "this", "these", "those", "it", "we", "our", "as", "at", "in", "on", "of",
    "for", "and", "or", "but", "if", "since", "when", "where", "while", "with", "from", "by",
    "figure", "fig", "table", "section", "eq", "equation", "appendix", "algorithm", "sec",
    # Discourse connectives and table/section labels likewise carry no factual value (in the
    # calibration `Here`, `Further`, and `Complexity` all became answers).
    "here", "there", "further", "furthermore", "moreover", "additionally", "finally", "meanwhile",
    "similarly", "consequently", "overall", "besides", "instead", "notably", "specifically",
    "complexity", "analysis", "overview", "setup", "results", "method", "methods",
})
# Caption/section reference words: a bare number right after them is an index (`Figure 2`,
# `Section 3`), not an answer.
_REFERENCE_LABELS: frozenset[str] = frozenset({
    "figure", "fig", "table", "section", "sec", "eq", "equation", "appendix", "algorithm",
})
# Raw shape of an entity span: starts uppercase, followed by at least 2 alphanumerics or `+`/`-`.
_ENTITY_SPAN = re.compile(r"[A-Z][A-Za-z0-9+\-]{2,}")


class SpanCandidate(ContractModel):
    """One candidate answer span; `sentence` is the source sentence after NFKC (collapsed to one line)."""

    sentence: str
    span: str
    kind: Literal["numeric", "entity"]


class DerivedClaim(ContractModel):
    """One claim derived from a source sentence.

    `text` always equals the source sentence, and empty `answer_keys` means key_matchable=false.
    """

    text: str
    span: str
    answer_keys: list[list[str]]


class QuotaReport(ContractModel):
    """Batch-level quota validation result; when `ok=false`, `violations` lists each unmet requirement."""

    ok: bool
    violations: list[str]
    counts: dict[str, Any]


def normalize_source_text(text: str) -> str:
    """The only normalisation allowed before persisting: ``unicodedata.normalize("NFKC", text)``.

    Case, whitespace, and punctuation are deliberately untouched: SPEC §13.1(c′) requires persisted
    text to be NFKC-only, and case/whitespace differences are eliminated uniformly by the
    `normalize_text` of (d); otherwise anchors would not line up with the original text.
    """

    return unicodedata.normalize("NFKC", text)


def split_sentences(text: str) -> list[str]:
    """Split on ``.!?`` + whitespace and collapse in-sentence newlines/multiple spaces into one space.

    The order is fixed: NFKC -> **fold end-of-line hyphenation** -> split on ``.!?`` -> collapse
    whitespace -> trim. The order cannot be exchanged: PDF extraction preserves printed hyphenation
    (`archi-\ntecture`), and collapsing whitespace first yields `archi- tecture`, while the folding
    rule in `key_match.normalize_text` requires "hyphen followed by newline"; once the newline is
    gone, `architecture` can never be restored and the grounding check (§13.1(c′)①) necessarily fails.
    Measured on a 1951-sentence pool, 572 candidates (29.3%) are affected, so folding first is a
    correctness requirement, not an optimisation.

    Splitting happens only where "punctuation is followed by whitespace", so decimals such as
    `0.0012s` and `7.21` are never broken; a sentence laid out across lines (`often\\nincomplete`) is
    merged into one entry and stays a single line after persisting. The function serves both the
    ``pool`` sentence count and ``sentence_candidates``.

    Known limitation (consistent with §13.1(d), pinned by an xfail in
    `tests/test_e2e_key_match.py`): a legitimate hyphenated compound that happens to break at a line
    end (`self-\nattention`) becomes `selfattention` after folding.
    """

    normalized = normalize_source_text(text)
    # Hyphenation is extraction noise, not part of the word meaning; it must be removed before
    # collapsing whitespace, otherwise the shape is broken beyond recovery.
    normalized = _LINE_END_HYPHEN.sub("", normalized)
    sentences: list[str] = []
    for raw in _SENTENCE_BOUNDARY.split(normalized):
        # Inline newlines are layout noise rather than sentence content; only after collapsing them
        # to single spaces can the anchor align with normalize_mapping.
        sentence = _WHITESPACE.sub(" ", raw).strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def _has_proper_noun_run(sentence: str, *, min_run: int = 2) -> bool:
    """Whether there are >= ``min_run`` consecutive capitalised proper-noun tokens.

    A token starting with a digit interrupts the run.
    """

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
    """Return the first lowercase word, used to filter out non-self-contained sentences that open with a pronoun or connective."""

    match = re.search(r"[A-Za-z]+", sentence)
    return match.group(0).lower() if match else ""


def _is_reference_or_heading(sentence: str) -> bool:
    """Decide whether a sentence is a reference entry or a heading glued to body text.

    Neither can serve as a question source. The "answer" of a reference entry
    (`2 [2] Iro Armeni, Sasha Sax, ...`) is an author name or a reference number, and the "answer" of
    a heading-glued sentence (`Experiments In this section, ...`) is a heading word. A probe can hit
    both, yet neither is a paper fact - calibration showed they produce questions like "what is the
    first author's name" with no evaluation value, and their distractor blocks hit very easily
    (`ambiguous_distractor`).
    """

    if _REFERENCE_ENTRY.match(sentence):
        return True
    # Only the first word is inspected: PDF extraction glues a section heading onto the front of
    # the
    # first body sentence, so the first word is the heading itself.
    first_word = _first_word(sentence)
    return first_word in _SECTION_HEADING_STARTS


def _is_noise_fragment(sentence: str) -> bool:
    """Decide whether a sentence is a formula/table fragment; such text cannot be a question source.

    PDF extraction turns LaTeX formulas and table rows into "sentences": `average based on inverse
    distances [22] ... s _ i ^l = \\f rac {\\ s }_s Here, ...`, `Complexity Analysis w/ Position
    Extractor (Params), theoretical computation cost (FLOPs), ...`. They are not natural-language
    statements; the grounding check (§13.1(c′)①) necessarily fails because formulas get reordered, and
    the "answer" is usually a symbol rather than a fact. Three signals are used: residual LaTeX, the
    share of non-alphanumeric symbols, and the share of digits; exceeding any limit excludes the
    sentence, so one regex alone cannot let a re-spelled fragment through.
    """

    if _FORMULA_MARKERS.search(sentence):
        return True
    # The share is computed over "non-whitespace characters": whitespace length carries no
    # meaning and would dilute the real density.
    visible = [char for char in sentence if not char.isspace()]
    if not visible:
        return True
    symbols = sum(1 for char in visible if not char.isalnum())
    digits = sum(1 for char in visible if char.isdigit())
    if symbols / len(visible) > _MAX_SYMBOL_RATIO:
        return True
    return digits / len(visible) > _MAX_DIGIT_RATIO


def sentence_candidates(text: str) -> list[str]:
    """Extract candidate sentences from chunk body text (design §2.1 / §2.3).

    Rules: NFKC -> fold end-of-line hyphenation -> split on ``.!?`` -> trim and collapse to one line;
    keep sentences of 60-320 characters whose first word is not a pronoun/connective or section
    heading (case-insensitive), that are not reference entries, and that contain a digit or >= 2
    consecutive capitalised proper-noun tokens; each sentence appears once (first-occurrence order
    preserved).
    """

    candidates: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        if not MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS:
            continue
        first_word = _first_word(sentence)
        if first_word in _BAD_SENTENCE_STARTS or first_word in _BAD_SENTENCE_STARTS_EXTRA:
            continue
        # References and section headings are not factual statements; only after excluding them is
        # the "answer" meaningful paper content.
        if _is_reference_or_heading(sentence):
            continue
        # Formula/table fragments are not natural-language statements: grounding fails and the
        # answer is often a symbol.
        if _is_noise_fragment(sentence):
            continue
        # Self-containedness: a number (a numeric fact) or a proper-noun run (method/dataset name)
        # must hold at least once, otherwise the stem cannot be answered from one sentence.
        if not re.search(r"\d", sentence) and not _has_proper_noun_run(sentence):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        candidates.append(sentence)
    return candidates


def numeric_unit_spans(sentence: str) -> list[str]:
    """Return every raw "number + unit" substring in the sentence (unit mandatory; used for pool statistics).

    Unlike ``select_spans`` this feeds the count of "sentences containing number + unit" (the 132-chunk
    convention of design §2.1); bare numbers such as `7.21` do not count, otherwise ordinary numeric
    sentences would be included.
    """

    spans: list[str] = []
    for match in NUMERIC_UNIT_PATTERN.finditer(sentence):
        span = match.group(0).strip()
        if span and _NUMERIC_WITH_UNIT.match(span):
            spans.append(span)
    return spans


def select_spans(sentence: str) -> list[SpanCandidate]:
    """Select at most 2 answer spans from one sentence (numbers first); each span is a literal substring.

    - Numbers: take the first **usable** match of ``NUMERIC_UNIT_PATTERN``, and the raw substring must
      be >= 2 characters; a bare number inside square brackets (`[38]`, `[60]`) is a citation index
      rather than a factual value, so it is skipped and the next match is tried.
    - Entities: when no usable number exists, take the first **non-stopword** proper-noun run
      ``[A-Z][A-Za-z0-9+\\-]{2,}``; sentence-initial function words (`The`) and caption words
      (`Figure`/`Table`) are not nameable answers and must be skipped.
    - When both a number and an entity are found, return both (number first); otherwise return the
      single one found.

    Why this filter is mandatory: the entity regex `[A-Z][A-Za-z0-9+\\-]{2,}` reads sentence-initial
    `The` (`T` + `he`) and `Figure` as proper nouns. When the sentence holds no value of length >= 2
    (e.g. `includes 2 PN layers`, where the length rule discards `2`), the result would be `The` /
    `Figure`, which are not answers at all - in the small-sample calibration 3 of 10 cases failed G6
    for this reason.
    """

    candidates: list[SpanCandidate] = []
    numeric: SpanCandidate | None = None
    for match in NUMERIC_UNIT_PATTERN.finditer(sentence):
        # In the regex, `\s*` swallows trailing whitespace when no unit follows; strip it and then
        # confirm the result is still a literal substring.
        span = match.group(0).strip()
        if len(span) < 2:
            continue
        # A citation index like `[38]` is the reference number the claim cites, not the asserted
        # factual value, so it cannot be an answer.
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
    """Whether ``sentence[start:end]`` is a square-bracket citation index (`[38]`, `[29,36]`).

    Only an adjacent bracket pair counts: `[38]` is a reference number, the `2` in `2 PN layers` is
    not.
    """

    before = sentence[:start].rstrip()
    after = sentence[end:].lstrip()
    return before.endswith("[") and after.startswith(("]", ","))


def _is_distinctive_entity(span: str) -> bool:
    """Whether a word carries "proper-noun traits" rather than being merely a capitalised ordinary word.

    Sentence-initial position cannot be resolved by case alone (`Each`, `Following`, `Despite`, `Let`,
    `Tenth` are all capitalised ordinary words that satisfy the same regex as `PointNet++`, `PCN-CD`,
    `MLP`). At least one "non-syntactic capitalisation" trait is therefore required:
    - an extra capital beyond the first letter (`SnowflakeNet`, `PointTransformer`);
    - a digit or `+`/`-` (`PN2`, `3D-EPN`, `PointNet++`, `Set-to-Set`);
    - an all-caps abbreviation of length >= 2 (`MLP`, `SSG`).

    Mid-sentence, capitalisation is already a strong signal and this rule is unnecessary.
    """

    if any(char.isdigit() for char in span) or "+" in span or "-" in span:
        return True
    if span[1:] != span[1:].lower():
        return True
    return span.isupper() and len(span) >= 2


def _first_entity_span(sentence: str) -> str | None:
    """Take the first proper-noun span usable as an answer; skip function words, caption words, and figure/table numbers.

    - Stopwords (`The`, `Figure`) carry no factual value and are skipped;
    - A bare number immediately after a reference word such as `Figure`/`Table`/`Section` is an index
      and is skipped;
    - **A sentence-initial candidate must carry proper-noun traits** (`_is_distinctive_entity`):
      sentence-initial position always capitalises, so ordinary words like
      `Each`/`Following`/`Despite`/`Let`/`Tenth` are indistinguishable from real proper nouns; only
      internal capitals, digits, or symbols make it trustworthy (`PointNet++`, `PCN-CD`, `MLP` all
      pass).
    Return ``None`` when no candidate qualifies (the caller then drops the candidate instead of
    inventing a key that can never hit).
    """

    for match in _ENTITY_SPAN.finditer(sentence):
        span = match.group(0)
        if span.casefold() in ENTITY_STOPWORDS:
            continue
        # When the position is sentence-initial (leading quotes/brackets allowed), the
        # capitalisation may come from syntax alone and needs extra evidence.
        at_start = not sentence[: match.start()].strip(" \t\"'(（[《")
        if at_start and not _is_distinctive_entity(span):
            continue
        # A bare number right after a reference word (`Figure 2`'s `2`) is an index; this applies
        # to
        # bare numbers only, so real values such as `3D` are not damaged.
        if span.isdigit():
            previous = _ENTITY_SPAN.findall(sentence[: match.start()])
            if previous and previous[-1].casefold() in _REFERENCE_LABELS:
                continue
        return span
    return None


def _dedupe_aliases(aliases: Iterable[str]) -> list[str]:
    """Deduplicate by the ``normalize_text`` form, keeping the first original spelling.

    Aliases that are too short or empty are dropped.
    """

    kept: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        if not alias.strip():
            continue
        normalized = normalize_text(alias)
        # SPEC §13.1(c)③: an alias shorter than 2 characters after normalisation has no
        # discriminative power (a bare `1` would hit every year and index).
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(alias)
    return kept


def derive_answer_keys(span: str, sentence: str, *, query_language: str = "en") -> list[list[str]]:
    """Build one alias group for the span from the derivation table (OR inside the group), returning ``[[...]]`` or ``[]``.

    Derivation rules (design §2.3, fixed order):
    1. the span verbatim;
    2. number + unit: drop the unit (`1.5mm`->`1.5`) and put a space before it (`1.5 mm`);
    3. percentage: `{n}%`->`{n} percent`; when ``query_language == "zh"`` also add the Chinese
       percent form;
    4. a span containing `-`: add the folded form (with `-` removed) and the spaced form
       (`-`->space);
    5. span longer than 40 characters => return ``[]`` (key_matchable=false).

    Deduplication uses the ``normalize_text`` form; aliases shorter than 2 characters after
    normalisation are dropped; the result must hold at least 1 alias, otherwise ``[]`` is returned
    (the caller can still build the question but must report the key_matchable ratio).

    Known limitation (SPEC §13.1(c)③): a 1-character alias such as a bare `1` is dropped, so `1%` can
    only derive `1%` / `1 percent` and no bare number can catch a natural answer; trailing modifiers
    such as `PointNet++` are not stripped separately either (the task derivation table lists only
    hyphens, units, and percentages).
    """

    if len(span) > MAX_KEY_SPAN_CHARS:
        return []
    aliases: list[str] = [span]
    numeric = _NUMERIC_WITH_UNIT.match(span)
    if numeric is not None:
        number, unit = numeric.group("number"), numeric.group("unit")
        aliases.append(number)
        # Putting a space before the unit is a common way to write natural answers (`0.0012 s`).
        aliases.append(f"{number} {unit}")
        if unit == "%":
            aliases.append(f"{number} percent")
            if query_language == "zh":
                aliases.append(f"百分之{number}")
    # Compound words: with end-of-line hyphenation the source may be `view-\npoints`, which folds
    # to
    # `viewpoints`; the spaced form is kept too, catching the case where the model writes the
    # hyphen
    # as a space.
    if "-" in span:
        aliases.append(span.replace("-", ""))
        aliases.append(span.replace("-", " "))
    kept = _dedupe_aliases(aliases)
    # Fallback: when every alias is judged too short the span counts as non-derivable, avoiding an
    # empty group that could never hit.
    return [kept] if kept else []


def three_gram_jaccard(left: str, right: str) -> float:
    """Tokenise on whitespace after normalisation and compute the Jaccard of the 3-gram sets.

    Falls back to word sets when either side has fewer than 3 words; both sides empty returns 0.0.
    This function is only used for the pairwise "stem deduplication" of G5, not for leak detection
    between a stem and a claim (that is ``longest_common_word_run``, SPEC §13.1(c′)④).
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
    """Return the word count of the longest common consecutive word run (dynamic programming, O(n*m)).

    SPEC §13.1(c′)④: more than 8 words means the stem restates the source. Stopwords count too; no
    stemming or synonym substitution; either side empty returns 0. Consecutive-word length is used
    instead of Jaccard because a numeric question must carry the source unit/object, and Jaccard would
    misread "the stem shares a long numeric context with the source" as a leak.
    """

    left_words = normalize_text(left).split()
    right_words = normalize_text(right).split()
    if not left_words or not right_words:
        return 0
    # Only the previous DP row is kept, giving O(m) space; equal lengths mean the match is
    # character-contiguous, which is equivalent to word-contiguous.
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
    """Stratify a derived question by question-type keywords (numeric spans are always quantitative_results).

    The keyword table comes from "stratified sampling by question-type vocabulary" in design §2.3; it
    only feeds quota slicing and never participates in ground-truth decisions. When no list matches,
    the type falls back to ``method`` (method descriptions dominate the corpus).
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
    """Stratified sampling by document x question type x language, replacing the prefix truncation of ``candidates[:limit]``.

    Why prefix truncation is unusable: candidates follow chunk iteration order, so taking the first N
    rows means taking sentences from only the first few papers. Measured on a 1951-sentence pool,
    taking the first 200 left two papers with only 1 and 3 candidates, while Gate B3 requires
    ">= 6 questions per paper" - however many are generated, the quota fails outright. Stratified
    sampling rotates the limited ``limit`` across layers, which is the only way to satisfy the
    per-paper and per-type quotas at the same time.

    Why sha256 rather than ``random``: after the questions are persisted they must be recomputable and
    auditable. ``hash()`` depends on the process hash seed and ``random`` on global state, so both
    make the same input yield different samples in different processes, whereas
    ``sha256(f"{seed}:{candidate_id}")`` gives the same ordering in any process and on any platform.

    Parameters:
    - ``candidates``: the candidate-sentence rows produced by ``pool`` (must contain
      ``candidate_id``/``document_id``/``sentence``).
    - ``limit``: total rows to take; when larger than the total, everything is returned (no
      duplicates).
    - ``seed``: an independent sampling seed for reruns, deciding the within-layer order.

    The layer key is ``(document_id, question type, query_language)``; the question type is recomputed
    by ``classify_question_type``, and candidates with no usable span go into a separate ``no_span``
    layer (they are doomed anyway, but they must not crowd out the layers that can produce questions).
    Return order is stable: two calls on the same input are element-wise identical.
    """

    if limit < 0:
        raise ValueError("limit must be >= 0")
    if limit == 0 or not candidates:
        return []

    # Bucket by layer first, then sort within each layer by a deterministic hash, so the take order
    # inside a layer is independent of the process.
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

    # Round-robin across layers: each round takes one row from every layer until the limit is
    # filled
    # or all layers are exhausted.
    # The layer order is also scattered with sha256 rather than sorted by (document_id, question
    # type): dictionary order makes the rotation keep returning to the same paper when the sample
    # is
    # small (measured: limit=10 covered only 2 papers); after scattering, a small sample still
    # spreads across documents.
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
    """A pair that can be combined into a T3 multi-document question.

    The two candidates come from different papers, share a question type, and both are key_matchable.
    """

    question_type: str
    candidate_a: dict[str, Any]
    candidate_b: dict[str, Any]


def pair_multi_document_candidates(
    items: Sequence[dict[str, Any]],
    *,
    limit: int,
    seed: int = 1,
) -> list[MultiDocPair]:
    """Pair single-evidence candidates two by two into the material of a T3 multi-document question (design §2.4).

    Pairing constraints (any failure means no pair):
    1. both have the same `question_type` - design §2.4 requires a "same question type" combination,
       otherwise the stem cannot be merged naturally;
    2. both have different `document_id` - the whole point of T3 is "two pieces of evidence are
       required";
    3. both are `key_matchable` (`answer_keys` non-empty) - otherwise "a single block hits only half"
       cannot be verified;
    4. the same `sentence` / the same `span` is not reused in the output (G5 already limits one span to
       at most 2 uses per paper).

    Ordering uses ``sha256(f"{seed}:{doc_a}:{doc_b}")``: the same input gives the same pairing in any
    process, and changing ``seed`` gives an independent second pass. When nothing can be paired an
    empty list is returned and the report counts the shortfall; no error is raised.
    """

    if limit <= 0:
        return []
    # Bucket by question type, keeping only key_matchable candidates in each bucket; mixing numeric
    # and entity candidates does not affect correctness, because stem generation only looks at
    # `sentence` and `span`, not at the span type.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get("decision") != "accepted":
            continue
        if not item.get("answer_keys"):
            continue
        question_type = str(item.get("question_type") or "")
        if not question_type or question_type == "multi_document":
            # multi_document is an output type of T3, so it cannot also be an input type for T3.
            continue
        buckets[question_type].append(item)

    used_sentences: set[str] = set()
    used_spans: set[str] = set()
    pairs: list[MultiDocPair] = []
    # Round-robin across question types so the quota is not filled by a single type (the same idea
    # as in stratified_sample).
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
    """Find the first same-type partner that is a different document, unused, and key_matchable."""

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
    """Generate a number that "looks plausible but differs from the original" per SPEC §13.1(f), for T6 false-premise questions.

    The rules follow the contract item by item (this is the deterministic construction shared by the
    B5b counterfactual and T6; no invented algorithm):
    1. decimals: increment the last fractional digit; `9` becomes `0` with a carry, and the **digit
       count is unchanged** (`0.19`->`0.20`, `1.99`->`2.00`);
    2. integers >= 10: +1;
    3. integers **< 10 are not chosen** (e.g. `7`, `1%`): the replacement lands too close to
       neighbouring values and is easily ignored by the model as noise;
    4. if the new value already occurs in ``context_text``, apply the rule again up to
       ``max_attempts`` times, and raise ``ValueError`` if it still collides;
    5. keep the unit and the `%` suffix (`1.5mm`->`1.6mm`, `50%`->`51%`).

    Why the new value must not appear in the source: the T6 "false premise" must be falsifiable from
    the source; if the tampered value was already on the page, the question holds two "correct" values
    at once and D8's `forbidden_claims` lose their contradiction anchor.
    """

    # Units/percent signs are handled separately from the number: derived spans often carry a unit
    # (`0.0012s`, `1.5mm`), and the replacement applies to the number alone.
    match = re.fullmatch(r"(?P<number>\d+(?:\.\d+)?)(?P<suffix>\s*[A-Za-z%]*)", value.strip())
    if match is None:
        raise ValueError(f"not a plain number: {value!r}")
    number, suffix = match.group("number"), match.group("suffix")

    candidate = number
    for _ in range(max_attempts):
        candidate = _increment_number(candidate)
        rendered = f"{candidate}{suffix}"
        # Compare after normalisation: the `1.6` in the source and the generated `1.6` are
        # spelled the same, but the context may carry newlines or tags.
        if normalize_text(rendered) not in normalize_text(context_text):
            return rendered
    raise ValueError(f"could not find a non-conflicting replacement for {value!r} in {max_attempts} attempts")


def _increment_number(body: str) -> str:
    """Apply one §13.1(f) increment to a pure digit string; decimals keep their digit count, integers < 10 are rejected.

    Decimals use "fixed-point addition": treat `1.99` as 199 (scale 100), add 1 to get 200, then
    reinsert the decimal point with the original number of fractional digits to get `2.00`. Carry and
    digit-count preservation therefore hold together without hand-written per-digit carry logic.
    """

    if "." in body:
        integer, fraction = body.split(".", 1)
        scale = 10 ** len(fraction)
        scaled = int(integer) * scale + int(fraction) + 1
        # The digit count is preserved: the integer part naturally grows from the +1 carry, and the
        # fractional digits are padded back to the original length.
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
    """Build a T4 out_of_corpus unanswerable question: the entity/fact asked about is absent from the corpus.

    The only difference from near_miss is "whether the entity exists": in T4 even the entity is absent
    from the corpus (e.g. `Point-BERT`, `A100`), while in T5 the entity exists but the attribute is
    missing. Both have the same shape on `EvalCase`: `expected_status=insufficient_context`,
    `acceptable_abstention=False`, `unanswerable_kind` set to the respective value, and **no evidence
    groups** (`unanswerable_shape` in `dataset.py`).
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
    """Build a T5 near_miss unanswerable question: the entity is in the corpus but co-occurs zero times with the attribute.

    The unanswerable shape of SPEC §13.1(a):
    - ``expected_status=insufficient_context``, ``acceptable_abstention=False`` (enforced by D7);
    - ``unanswerable_kind="near_miss"``;
    - **no ``required_evidence_groups``** (explicitly forbidden by the `unanswerable_shape` rule in
      `dataset.py`), so this kind takes part in no evidence mapping and enters no context-layer metric
      denominator;
    - ``tags`` carries ``unanswerable:near_miss`` for sharded statistics.

    Why the stem must name the entity: the difficulty of near_miss is "the entity exists but the
    attribute is missing", so the stem must let the system retrieve the document holding that entity;
    otherwise the system refuses because it "cannot find anything at all" - that is out_of_corpus
    behaviour and does not measure the capability this question exists to measure, namely "retrieved,
    but the context is insufficient to answer".
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
    """Generate the near_miss stem: name the entity, then ask for an attribute that never co-occurs with it.

    The stem must name the entity (see the rationale in ``build_near_miss_case``) and must not copy
    more than 8 consecutive words of the source sentence (§13.1(c′)④) - here only the entity and the
    attribute word are concatenated, so no source sentence is copied by construction.
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
    """Build a D8 false-premise question: the stem carries a tampered value and the required claim states the correction.

    D8 convention (SPEC §13.1(a)): scored as an **answerable** question - ``expected_status=answered``,
    ``acceptable_abstention=True`` (abstaining counts as a correct status), ``tags`` contains
    ``false_premise``, ``unanswerable_kind=None``; ``forbidden_claims`` has at least 1 entry, whose
    anchor is the source passage contradicting the false statement.

    Why the required claim uses the original sentence: the model must correct the premise and give the
    right fact; the original sentence is that fact, so ``key_match`` still judges against the original
    ``answer_keys`` and no second matching logic is needed.
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
                # The false statement is written as a contrast against the original value, so a
                # manual reviewer sees at a glance which value was tampered with.
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


# The "entity x attribute" attribute vocabulary used to derive T5 (near_miss) questions
# (design §2.4 / plan B3 table).
# Selection principle: these attributes **may or may not be discussed** in point-cloud-completion
# papers, so "the entity exists but the attribute is missing" is a genuinely occurring near-miss
# situation; attributes a paper could never touch must not be used (that would degenerate into
# out_of_corpus).
NEAR_MISS_ATTRIBUTES: tuple[str, ...] = (
    "optimizer", "learning rate", "batch size", "epoch", "gpu model", "training time",
    "license", "author affiliation", "hyperparameter", "random seed", "data augmentation",
)


class NearMissCandidate(ContractModel):
    """One near_miss candidate.

    The entity comes from an existing T1 question, and the attribute is a neighbouring attribute that
    never co-occurs with that entity in the corpus.
    """

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
    """Derive near_miss candidates (T5) from existing T1 questions: "entity x attribute" co-occurs zero times.

    Why it must be derived from T1 questions rather than invented: near_miss is defined as "the entity
    is in the corpus and only that attribute is missing". An invented entity (e.g. `Point-BERT`) is
    `out_of_corpus`; the distinction is exactly **whether the entity occurs**, and this rule turns the
    classification into a recomputable decision instead of an author's declaration (plan B3 table,
    T4/T5).

    Determinism requirements:
    - ``corpus_text`` is the normalised full text of the 817 chunks, and co-occurrence is decided on
      literal substrings of ``normalize_text`` output;
    - within a layer (entity) rows are ordered by ``sha256(f"{seed}:{entity}:{attribute}")`` and
      rotated across entities, so the same input produces the same order within one process and
      across processes;
    - each entity yields only one candidate (so near-miss questions do not pile onto the same entity
      in the same paper).

    An empty list is a legitimate result (no usable near-miss combination in this batch), and the
    caller decides whether to draw more.
    """

    if limit <= 0 or not corpus_text:
        return []
    normalized_corpus = normalize_text(corpus_text)
    # Aggregate and deduplicate by entity: when the same entity appears repeatedly only the first
    # row is kept (sorted by candidate_id to stay deterministic).
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
        # The entity must be discriminative: ordinary nouns such as `Feature`/`Natural`/`Part`/
        # `Failure` are capitalised, but asking "what optimizer does it use" is meaningless and is
        # not a real "entity exists" situation.
        # `_is_distinctive_entity` is reused, the same convention select_spans applies to
        # sentence-initial pseudo-proper-nouns.
        if not _is_distinctive_entity(entity):
            continue
        # The entity itself must actually appear in the corpus, otherwise the question is
        # out_of_corpus rather than near_miss.
        if normalize_text(entity) not in normalized_corpus:
            continue
        item = by_entity[entity]
        for attribute in NEAR_MISS_ATTRIBUTES:
            # Co-occurrence rule: entity and attribute must appear inside the same window to count
            # as "discussed in the corpus".
            # The loose ordered pair `entity ... attribute` is used (both appearing is enough to
            # consider possible co-occurrence), and conservatively both must appear in the merged
            # text of one chunk - approximated here by the whole corpus, which errs strict: only
            # when they never co-occur anywhere in the corpus is "co-occurrence is 0" certain.
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
    """Whether entity and attribute co-occur in the corpus within at most ``window`` characters.

    A proximity window is used rather than "co-occurring anywhere in the corpus": whole-corpus
    granularity is too coarse, since in a 100-page paper "mentions the entity" and "mentions the
    optimizer" almost always both hold, which would misjudge every near_miss candidate as "discussed in
    the corpus". The window counts normalised characters, and 200 characters is roughly one chunk's
    neighbourhood.
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
    """Read a field from either an EvalCase or a plain dict.

    This lets the quotas serve both the persisted dataset and the JSONL candidates.
    """
    if isinstance(case, dict):
        return case.get(name, default)
    return getattr(case, name, default)


def _case_document_ids(case: Any) -> list[str]:
    """The papers a question touches: EvalCase reads them from anchors, a dict reads ``document_id``."""

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
    """Difficulty reads the dedicated field first, then easy/medium/hard from tags."""

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
    """Compute and validate the batch-level quotas without raising (design §2.5).

    - every paper >= ``per_document_min`` (a multi-document question counts for each paper it touches);
    - each of the 8 ``question_type`` values >= ``per_type_min`` (validated against the full spec set,
      so a missing type is a violation too);
    - roughly half Chinese and half English: ``|zh-en| <= max(2, 10% of total)``;
    - each of the three difficulty tiers >= ``hard_ratio_min`` (difficulty comes from the caller via
      tags or a dedicated field; with no difficulty labels at all only one "missing labels"
      violation is reported, and no tier ratio is fabricated).

    ``counts`` records the total and the per-dimension distribution, for direct citation by
    ``B3_gate_report.json``.
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
    # Language quota: only zh/en are counted separately; an unrecognised language is reported
    # separately too, so it is never silently counted on one side.
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
    # Difficulty quota: the denominator uses "questions that carry a label", otherwise a missing
    # label would be miscounted as hard; with no labels at all only one violation is reported.
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
    """Return the original text of the locator's page range; None page numbers mean the whole document.

    That matches §13.1(c), where MD/TXT material has no page numbers. Page text joins Segments with
    ``"\\n"``, exactly as ``validate_source``'s ``raw`` does, so the grounding check and anchor
    location see the same text.
    """

    if page_start is None:
        return "\n".join(segment.content for segment in document.segments)
    return "\n".join(
        segment.content
        for segment in document.segments
        if segment.page_number is not None and page_start <= segment.page_number <= page_end
    )


def _occurrence_contexts(text: str, start: int, end: int) -> tuple[str | None, str | None]:
    """Return 20 characters of context on each side of the anchor.

    A whitespace-only side returns None (the schema requires a non-empty string or null).
    """

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
    """Locate the sentence within the document's given page range and return an anchor with a locator.

    The normalisation algorithm is exactly that of ``stable_mapping.validate_source``: ``hyphen_rules``
    folds only hyphens jointly proven by "an original-page newline break + the anchor's complete word",
    and ``normalize_mapping`` then applies NFKC, casefold, and whitespace collapsing. The returned
    ``occurrence`` is the 1-based index of the first occurrence in the normalised page text,
    ``source_span_start/end`` is a half-open interval in the normalised text, and
    ``left_context/right_context`` take 20 characters each.

    When the sentence cannot be located, or it occurs several times on the page with identical
    20-character contexts (so the occurrence cannot be identified), ``ValueError`` is raised and the
    caller drops the candidate; the position is never guessed.
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
        # When it occurs several times, only the context can distinguish occurrence 1 from the
        # others; if they are identical the locator meaning cannot be proven.
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
    """Assemble the derivation result into an ``EvalCase`` (the D3 machine-acceptance convention).

    Default shape (single-evidence derived question): ``expected_status=answered``,
    ``acceptable_abstention=False``, ``critical=False``, evidence group ``G1``, required claim ``C1``
    (text = the source sentence, with anchor and answer_keys). provenance is fixed to
    ``drafted_by="agent"`` and ``approved_by="machine"``; ``approval_basis`` is passed in by the caller
    (pointing at the gate report and its sha256); ``source`` distinguishes the derivation method (e.g.
    ``derived_t1``/``derived_t2``).

    ``extra_evidence_groups`` / ``extra_claims`` serve multi-evidence questions (T3): they are appended
    after ``G1`` / ``C1`` with IDs supplied by the caller (of the form ``G2`` / ``C2``). Group and claim
    IDs must be unique within a question - duplicates raise immediately instead of being silently
    deduplicated, because ``validate_dataset`` judges duplicate IDs as unfreezable and silent dedup
    would hide an upstream pairing error.

    ``span`` has already been used by ``derive_answer_keys`` and the EvalCase schema does not store it
    separately; only a defensive "the span is still inside the source sentence" check is done here, so
    a misaligned derivation never reaches the dataset.
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
    # §13.1(a) requires group IDs and claim IDs to be unique within a question; reject here early
    # so
    # a dataset that could never freeze is never written.
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
