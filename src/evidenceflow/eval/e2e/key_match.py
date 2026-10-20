"""关键要素匹配 `key_match`（PROJECT_SPEC §13.1(d)），纯函数、只依赖标准库。

规则摘要：
- 规范化：NFKC → 行尾连字符折叠 → 去掉 `[S\\d+]` 引用标签 → casefold → 连续空白压成一个空格并去首尾空白；
- 别名与文本同样规范化后做子串匹配；
- 别名首/尾字符是 ASCII 字母或数字时，命中位置外侧相邻字符不得是 ASCII 字母或数字
  （防止 `1.5` 命中 `11.5`、`ViT` 命中 `ViTs`）；CJK 等非 ASCII 字符不要求边界；
- 不做数字解析、单位换算或模糊匹配；不同写法只能列为显式别名；
- claim 命中 ⇔ 所有要素组都命中；组内任一别名命中即该组命中。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

# 合同 §10 的 Citation 语法只识别大写 `[S<数字>]`；在 NFKC 之后、casefold 之前去除，
# 这样全角 `［Ｓ１］` 经 NFKC 变成 `[S1]` 后也会被去掉。
_CITATION_TAG = re.compile(r"\[S\d+\]")
# PDF 抽取会保留印刷断词（`view-\npoints`）：行尾连字符与换行必须折叠，否则自然回答的
# `viewpoints` 永远无法与原文对齐。只在“连字符 + 行尾换行”这一确定形态上折叠，不猜测其他连字符。
_LINE_END_HYPHEN = re.compile(r"-\s*\n\s*")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """按 §13.1(d) 规范化文本。

    顺序说明：先折叠行尾连字符、再去引用标签、最后压空白。折叠必须发生在压空白之前，
    否则断词会先变成 `view- points` 而无法再还原；去标签必须在压空白之前，因为标签两侧
    常带空格（如 `CD [S2] 值`），去掉标签后会留下两个连续空格，若先压空白再去标签就会残留
    多余空白，使别名 `CD 值` 无法命中。标签替换为空串而不是空格，使 `CD[S2]值` 这类紧贴
    写法的两侧文本能直接拼接。已知局限：合法连字符复合词若恰在行尾断开（`self-\nattention`），
    折叠后成为 `selfattention`，因此别名应使用折叠后的形式或改用其他别名。
    """

    normalized = unicodedata.normalize("NFKC", text)
    # 行尾断词是抽取噪声，不是词义的一部分；先消除它，后续空白折叠才不会被误导。
    normalized = _LINE_END_HYPHEN.sub("", normalized)
    normalized = _CITATION_TAG.sub("", normalized)
    normalized = normalized.casefold()
    # 连续空白（含换行、制表符）压成一个空格，并去掉首尾空白，避免排版差异影响匹配。
    return _WHITESPACE.sub(" ", normalized).strip()


def _is_ascii_alnum(char: str) -> bool:
    """ASCII 字母或数字才参与词边界判断；CJK、标点、空白都不算。"""

    return char.isascii() and char.isalnum()


def _spans_in_normalized(alias_norm: str, text_norm: str) -> list[tuple[int, int]]:
    """在已规范化文本中找出满足 ASCII 边界规则的全部命中区间（允许重叠起点）。"""

    if not alias_norm:
        return []
    # 只有别名首/尾本身是 ASCII 字母数字时才需要检查对应一侧的边界。
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
        # 从下一个字符继续找，保证不会漏掉与被拒位置重叠的合法命中。
        start = text_norm.find(alias_norm, start + 1)
    return spans


def find_alias_spans(alias: str, text: str) -> list[tuple[int, int]]:
    """返回别名在**规范化后文本**中的全部命中区间 `[start, end)`。

    区间坐标基于 `normalize_text(text)`，不是原文坐标；B5b 的数字替换在规范化文本上
    定位时使用。规范化后为空的别名返回空列表。
    """

    return _spans_in_normalized(normalize_text(alias), normalize_text(text))


def alias_hits(alias: str, text: str) -> bool:
    """单个别名是否按 §13.1(d) 命中文本。"""

    return bool(find_alias_spans(alias, text))


def group_hits(group: Sequence[str], text: str) -> bool:
    """要素组内别名之间是“或”：任一别名命中即可；空组视为不命中。"""

    text_norm = normalize_text(text)
    return any(_spans_in_normalized(normalize_text(alias), text_norm) for alias in group)


def claim_hits(answer_keys: Sequence[Sequence[str]], text: str) -> bool:
    """要素组之间是“与”：所有组都命中，claim 才算命中。

    `answer_keys` 为空表示该 claim 不可做关键要素匹配（`key_matchable=false`），调用方不应
    对它计分，因此这里直接抛 `ValueError`，避免把“无法判定”误当成“命中”。
    """

    if not answer_keys:
        raise ValueError("answer_keys is empty: claim is not key_matchable")
    return all(group_hits(group, text) for group in answer_keys)
