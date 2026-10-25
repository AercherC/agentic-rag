"""Verify the normalisation and boundary rules of SPEC §13.1(d) key-element matching (key_match)."""

import pytest

from evidenceflow.eval.e2e.key_match import (
    alias_hits,
    claim_hits,
    find_alias_spans,
    group_hits,
    normalize_text,
)


def test_decimal_numbers_respect_ascii_boundaries():
    """1.5 must not hit 11.5 or 1.55, but may hit a standalone 1.5."""
    assert not alias_hits("1.5", "the value is 11.5 here")
    assert not alias_hits("1.5", "the value is 1.55 here")
    assert alias_hits("1.5", "the value is 1.5.")
    assert alias_hits("1.5", "误差为1.5，")


def test_ascii_word_boundary_for_names():
    """ViT does not hit ViTs; `-` is not an ASCII alphanumeric, so ViT-B does hit."""
    assert not alias_hits("ViT", "several ViTs are compared")
    assert alias_hits("ViT", "we use ViT-B as backbone")


def test_cjk_alias_needs_no_boundary():
    """A CJK alias hits directly inside a Chinese sentence and requires no boundary on either side."""
    assert alias_hits("倒角距离", "本文采用倒角距离作为评价指标")
    assert alias_hits("倒角距离", "使用CD倒角距离评测")


def test_nfkc_and_case_insensitive():
    """Full-width digits and letters hit after NFKC; matching is case-insensitive."""
    assert alias_hits("PCN", "在ＰＣＮ数据集上")
    assert alias_hits("12.5", "误差为１２．５")
    assert alias_hits("chamfer distance", "CHAMFER   Distance")


def test_citation_tags_removed_before_whitespace_collapse():
    """With the [S1] tag removed the two sides join and hit; redundant whitespace collapses to one space."""
    assert normalize_text("CD[S2] 值") == "cd 值"
    assert normalize_text("CD [S2] 值") == "cd 值"
    assert alias_hits("CD 值", "CD[S2] 值较低")
    assert alias_hits("cd值", "CD[S2]值较低")
    assert alias_hits("point cloud", "point [S1]\n cloud completion")


def test_line_end_hyphen_is_folded_before_whitespace_collapse():
    """The PDF-printed line break `view-\\npoints` folds into viewpoints, so only a natural answer of viewpoints can hit."""
    assert normalize_text("view-\npoints") == "viewpoints"
    assert normalize_text("view-\r\npoints") == "viewpoints"
    assert normalize_text("view-  \n  points") == "viewpoints"
    assert alias_hits("viewpoints", "source has view-\npoints")
    # Citation tags are still removed after the folding, so the two orders do not conflict.
    assert normalize_text("view-\npoints [S2]") == "viewpoints"


@pytest.mark.xfail(strict=True, reason="已知局限：行尾断开的合法复合词 point-cloud 被折叠成 pointcloud")
def test_line_end_hyphen_breaks_legal_compound_words():
    """Pinned evidence of the known limitation: when a legitimate hyphenated compound happens to break at a line end, the folding deletes the hyphen."""
    assert alias_hits("point-cloud", "a point-\ncloud is mentioned")


def test_group_or_and_claim_and():
    """OR within a group, AND across groups."""
    text = "SnowflakeNet reaches 7.21 on PCN"
    assert group_hits(["SeedFormer", "SnowflakeNet"], text)
    assert not group_hits(["SeedFormer", "PoinTr"], text)
    assert not group_hits([], text)
    assert claim_hits([["SnowflakeNet"], ["7.21", "7.2"]], text)
    assert not claim_hits([["SnowflakeNet"], ["6.5"]], text)


def test_empty_answer_keys_raise():
    """A claim that cannot be matched must not be passed in."""
    with pytest.raises(ValueError):
        claim_hits([], "anything")


def test_find_alias_spans_on_normalized_text():
    """Spans are based on the normalised text, and positions rejected by the boundary rule are not returned."""
    text = "1.5 and 11.5 and 1.5"
    assert find_alias_spans("1.5", text) == [(0, 3), (17, 20)]
    assert normalize_text(text)[17:20] == "1.5"
    assert find_alias_spans("x", "") == []
    assert find_alias_spans("   ", "abc") == []
