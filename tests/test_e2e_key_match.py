"""验证 SPEC §13.1(d) 关键要素匹配 key_match 的规范化与边界规则。"""

import pytest

from evidenceflow.eval.e2e.key_match import (
    alias_hits,
    claim_hits,
    find_alias_spans,
    group_hits,
    normalize_text,
)


def test_decimal_numbers_respect_ascii_boundaries():
    """1.5 不得命中 11.5 或 1.55，但可命中独立出现的 1.5。"""
    assert not alias_hits("1.5", "the value is 11.5 here")
    assert not alias_hits("1.5", "the value is 1.55 here")
    assert alias_hits("1.5", "the value is 1.5.")
    assert alias_hits("1.5", "误差为1.5，")


def test_ascii_word_boundary_for_names():
    """ViT 不命中 ViTs；`-` 不是 ASCII 字母数字，所以命中 ViT-B。"""
    assert not alias_hits("ViT", "several ViTs are compared")
    assert alias_hits("ViT", "we use ViT-B as backbone")


def test_cjk_alias_needs_no_boundary():
    """CJK 别名在中文句中直接命中，不要求两侧边界。"""
    assert alias_hits("倒角距离", "本文采用倒角距离作为评价指标")
    assert alias_hits("倒角距离", "使用CD倒角距离评测")


def test_nfkc_and_case_insensitive():
    """全角数字与字母经 NFKC 后命中；大小写不敏感。"""
    assert alias_hits("PCN", "在ＰＣＮ数据集上")
    assert alias_hits("12.5", "误差为１２．５")
    assert alias_hits("chamfer distance", "CHAMFER   Distance")


def test_citation_tags_removed_before_whitespace_collapse():
    """去掉 [S1] 标签后两侧文本拼接命中；多余空白被压成一个空格。"""
    assert normalize_text("CD[S2] 值") == "cd 值"
    assert normalize_text("CD [S2] 值") == "cd 值"
    assert alias_hits("CD 值", "CD[S2] 值较低")
    assert alias_hits("cd值", "CD[S2]值较低")
    assert alias_hits("point cloud", "point [S1]\n cloud completion")


def test_line_end_hyphen_is_folded_before_whitespace_collapse():
    """PDF 印刷断词 `view-\\npoints` 折叠成 viewpoints，自然回答的 viewpoints 才能命中。"""
    assert normalize_text("view-\npoints") == "viewpoints"
    assert normalize_text("view-\r\npoints") == "viewpoints"
    assert normalize_text("view-  \n  points") == "viewpoints"
    assert alias_hits("viewpoints", "source has view-\npoints")
    # 引用标签仍在折叠之后被删除，两种顺序不冲突。
    assert normalize_text("view-\npoints [S2]") == "viewpoints"


@pytest.mark.xfail(strict=True, reason="已知局限：行尾断开的合法复合词 point-cloud 被折叠成 pointcloud")
def test_line_end_hyphen_breaks_legal_compound_words():
    """已知局限的固定证据：合法连字符复合词恰在行尾断开时，折叠会删掉连接符。"""
    assert alias_hits("point-cloud", "a point-\ncloud is mentioned")


def test_group_or_and_claim_and():
    """组内“或”、组间“与”。"""
    text = "SnowflakeNet reaches 7.21 on PCN"
    assert group_hits(["SeedFormer", "SnowflakeNet"], text)
    assert not group_hits(["SeedFormer", "PoinTr"], text)
    assert not group_hits([], text)
    assert claim_hits([["SnowflakeNet"], ["7.21", "7.2"]], text)
    assert not claim_hits([["SnowflakeNet"], ["6.5"]], text)


def test_empty_answer_keys_raise():
    """不可匹配的 claim 不应被调用。"""
    with pytest.raises(ValueError):
        claim_hits([], "anything")


def test_find_alias_spans_on_normalized_text():
    """区间基于规范化文本，被边界拒绝的位置不返回。"""
    text = "1.5 and 11.5 and 1.5"
    assert find_alias_spans("1.5", text) == [(0, 3), (17, 20)]
    assert normalize_text(text)[17:20] == "1.5"
    assert find_alias_spans("x", "") == []
    assert find_alias_spans("   ", "abc") == []
