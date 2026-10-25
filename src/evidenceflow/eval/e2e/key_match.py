"""Key-element matching `key_match` (PROJECT_SPEC §13.1(d)); pure functions that depend
only on the standard library.

Rule summary:
- Normalization: NFKC -> fold trailing hyphens -> strip `[S\\d+]` citation tags -> casefold ->
  collapse consecutive whitespace into one space and trim both ends;
- Normalize aliases and text the same way before substring matching;
- When an alias starts/ends with an ASCII letter or digit, the characters immediately outside
  the hit must not be ASCII letters or digits (so `1.5` does not hit `11.5`, and `ViT` does
  not hit `ViTs`); CJK and other non-ASCII characters need no boundary;
- No number parsing, unit conversion, or fuzzy matching; a different spelling must be listed
  as an explicit alias;
- A claim hits if and only if every element group hits; any alias hit makes its group hit.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

# The citation syntax in contract §10 recognizes only uppercase `[S<digits>]`; remove it after
# NFKC and before casefold, so a full-width spelling also becomes `[S1]` and is stripped.
_CITATION_TAG = re.compile(r"\[S\d+\]")
# PDF extraction preserves printed hyphenation: a trailing hyphen and newline must be folded,
# otherwise the natural answer `viewpoints` can never align with the source text. Fold only the
# certain shape "hyphen + end-of-line newline"; do not guess at other hyphens.
_LINE_END_HYPHEN = re.compile(r"-\s*\n\s*")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text according to §13.1(d).

    Ordering rationale: fold trailing hyphens first, strip citation tags next, and collapse
    whitespace last. Folding must happen before whitespace collapsing; otherwise a broken word
    would first become `view- points` and can never be restored. Tag removal must also happen
    before whitespace collapsing, because tags are usually surrounded by spaces (for example,
    `CD [S2]` followed by more text), and removing a tag leaves two consecutive spaces.
    Collapsing first and removing tags later would leave a stray space, so an alias that spans
    the tag would not hit. Tags are replaced with the empty string rather than a space, so the
    text on both sides of a tight form such as `A[S2]B` joins directly. Known limitation: a valid
    hyphenated compound that happens to break at a line end (for example, `self-` ending one line
    and `attention` starting the next) becomes `selfattention` after folding, so aliases should
    use the folded form or choose another alias.
    """

    normalized = unicodedata.normalize("NFKC", text)
    # End-of-line hyphenation is extraction noise, not part of the word's meaning; remove it
    # first so the later whitespace collapsing is not misled.
    normalized = _LINE_END_HYPHEN.sub("", normalized)
    normalized = _CITATION_TAG.sub("", normalized)
    normalized = normalized.casefold()
    # Collapse consecutive whitespace (including newlines and tabs) into one space and trim
    # both ends, so layout differences do not affect matching.
    return _WHITESPACE.sub(" ", normalized).strip()


def _is_ascii_alnum(char: str) -> bool:
    """ASCII letters and digits alone participate in the word-boundary check; CJK characters,
    punctuation, and whitespace do not."""

    return char.isascii() and char.isalnum()


def _spans_in_normalized(alias_norm: str, text_norm: str) -> list[tuple[int, int]]:
    """Find every hit span in already-normalized text that satisfies the ASCII boundary rule
    (overlapping starts are allowed)."""

    if not alias_norm:
        return []
    # A boundary check is needed only on a side where the alias itself starts/ends with an
    # ASCII letter or digit.
    check_left = _is_ascii_alnum(alias_norm[0])
    check_right = _is_ascii_alnum(alias_norm[-1])
    spans: list[tuple[int, int]] = []
    start = text_norm.find(alias_norm)
    while start != -1:
        end = start + len(alias_norm)
        left_ok = not (check_left and start > 0 and _is_ascii_alnum(text_norm[start - 1]))
        right_ok = not (check_right and end < len(text_norm) and _is_ascii_alnum(text_norm[end]))
        if left_ok and right_ok:
            spans.append((start, end))
        # Continue from the next character so a valid hit overlapping a rejected position is
        # not missed.
        start = text_norm.find(alias_norm, start + 1)
    return spans


def find_alias_spans(alias: str, text: str) -> list[tuple[int, int]]:
    """Return every hit span `[start, end)` of the alias in the **normalized text**.

    The span coordinates are based on `normalize_text(text)`, not on the original text; the B5b
    numeric replacement uses them to locate positions in the normalized text. An alias that
    normalizes to empty returns an empty list.
    """

    return _spans_in_normalized(normalize_text(alias), normalize_text(text))


def alias_hits(alias: str, text: str) -> bool:
    """Whether a single alias hits the text according to §13.1(d)."""

    return bool(find_alias_spans(alias, text))


def group_hits(group: Sequence[str], text: str) -> bool:
    """Aliases within an element group are OR-ed: any alias hit is enough; an empty group
    never hits."""

    text_norm = normalize_text(text)
    return any(_spans_in_normalized(normalize_text(alias), text_norm) for alias in group)


def claim_hits(answer_keys: Sequence[Sequence[str]], text: str) -> bool:
    """Element groups are AND-ed: the claim hits only when every group hits.

    An empty `answer_keys` means the claim is not eligible for key-element matching
    (`key_matchable=false`), and callers must not score it. This function therefore raises
    `ValueError` directly rather than mistaking "cannot decide" for "hit".
    """

    if not answer_keys:
        raise ValueError("answer_keys is empty: claim is not key_matchable")
    return all(group_hits(group, text) for group in answer_keys)
