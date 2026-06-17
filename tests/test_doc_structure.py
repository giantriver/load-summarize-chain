"""Tests for doc_structure.py — covers plan test cases 1, 2, 2b, 3."""

import pytest
from doc_structure import (
    HeadingCandidate,
    LineRecord,
    analyze_structure_profile,
    build_line_records,
    clean_line_records,
    detect_heading_candidates,
    filter_toc_and_noise,
    find_toc_regions,
    infer_levels,
    validate_sequence,
    build_heading_tree,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

CASE1_TEXT = """\
1. 綜合概述
1.1 概論
1.1.1 緣由及目的
(1) 緣由
內文...
(2) 目的
內文...
1.1.2 專有名詞
(1).核子反應器設施
內文...\
"""

CASE2_TEXT = """\
一、目的
內文...
二、適用時機
內文...
（一）二級開設
內文...
（1）境外部分
內文...\
"""

CASE2B_TEXT = """\
三、作業程序
（一）二級開設
1. 開設時機
（1）境外部分
（2）邊境部分
2. 參與機關\
"""

CASE4_TEXT = """\
（一）總則
一、目的
內文...
二、適用範圍
內文...
（二）作業程序
一、啟動時機
內文...
二、任務分工
內文...\
"""

CASE5_TEXT = """\
(A) 總則
(a) 目的
(1) 細項
內文...
(B) 作業程序
(a) 啟動時機
(1) 細項
內文...\
"""

CASE3_TEXT = """\
目錄
1. 綜合概述..................................................1
1.1 概論......................................................1
24410 輻射劑量評估及輻射防護措施
附圖目錄
圖 1-1：ZPRL位置圖
1. 綜合概述
1.1 概論\
"""


def _pipeline(text: str) -> tuple[list[HeadingCandidate], dict]:
    """Run the full detection pipeline on a single-page text, return (candidates, profile)."""
    pages = [text]
    records = build_line_records(pages)
    records = clean_line_records(records)
    all_cands = detect_heading_candidates(records)
    toc_regions = find_toc_regions(records)
    cands = filter_toc_and_noise(all_cands, toc_regions)
    profile = analyze_structure_profile(cands)
    cands = infer_levels(cands, profile)
    cands = validate_sequence(cands)
    return cands, profile


# ─── Test 1: Technical report format ─────────────────────────────────────────

class TestCase1:
    def test_profile(self):
        _, profile = _pipeline(CASE1_TEXT)
        assert profile["has_decimal"] is True
        assert profile["max_decimal_depth"] == 3

    def test_levels(self):
        cands, _ = _pipeline(CASE1_TEXT)
        # Key by (style, num) to avoid collision: decimal "1" vs paren_num "1"
        sn = {(c.style, c.num): c.level for c in cands}

        assert sn[("decimal", "1")] == 1
        assert sn[("decimal", "1.1")] == 2
        assert sn[("decimal", "1.1.1")] == 3
        assert sn[("decimal", "1.1.2")] == 3

    def test_paren_levels(self):
        cands, _ = _pipeline(CASE1_TEXT)
        paren_cands = [c for c in cands if c.style == "paren_num"]
        # deepest=3, so paren_num → L4
        assert all(c.level == 4 for c in paren_cands)

    def test_tree_structure(self):
        cands, _ = _pipeline(CASE1_TEXT)
        tree = build_heading_tree(cands)
        # Root should have one child: 1. 綜合概述
        assert len(tree["children"]) == 1
        ch1 = tree["children"][0]
        assert ch1["num"] == "1"
        # 1.1 is child of 1
        assert any(c["num"] == "1.1" for c in ch1["children"])


# ─── Test 2: Pure Chinese legal (no decimal) ─────────────────────────────────

class TestCase2:
    def test_profile(self):
        _, profile = _pipeline(CASE2_TEXT)
        assert profile["has_decimal"] is False
        assert profile["max_decimal_depth"] == 0
        assert profile["cjk_style_order"] == ["cjk_comma", "cjk_paren"]
        assert profile["cjk_style_order_source"] == "parent_child_evidence"
        assert profile["cjk_parent_scores"]["cjk_comma_parent"] > profile["cjk_parent_scores"]["cjk_paren_parent"]
        assert profile["style_order"][:2] == ["cjk_comma", "cjk_paren"]
        assert profile["style_order_source"] == "parent_child_evidence"

    def test_levels(self):
        cands, _ = _pipeline(CASE2_TEXT)
        # Key by (style, num) to avoid collision between cjk_comma "一" and cjk_paren "一"
        sn_level = {(c.style, c.num): c.level for c in cands}

        assert sn_level[("cjk_comma", "一")] == 1
        assert sn_level[("cjk_comma", "二")] == 1

    def test_cjk_paren_level(self):
        cands, _ = _pipeline(CASE2_TEXT)
        cjk_paren = [c for c in cands if c.style == "cjk_paren"]
        assert all(c.level == 2 for c in cjk_paren)

    def test_paren_num_level(self):
        cands, _ = _pipeline(CASE2_TEXT)
        # deepest = 2 + 0 = 2; paren_num → L3
        paren_num = [c for c in cands if c.style == "paren_num"]
        assert all(c.level == 3 for c in paren_num)


# ─── Test 2b: Legal with single-level decimal ─────────────────────────────────

class TestCase2b:
    def test_profile(self):
        _, profile = _pipeline(CASE2B_TEXT)
        assert profile["has_decimal"] is False
        assert profile["max_decimal_depth"] == 1

    def test_levels(self):
        cands, _ = _pipeline(CASE2B_TEXT)
        # Key by (style, num) to avoid collisions
        sn = {(c.style, c.num): c for c in cands}

        assert sn[("cjk_comma", "三")].level == 1
        assert sn[("cjk_paren", "一")].level == 2
        assert sn[("decimal", "1")].level == 3
        assert sn[("decimal", "2")].level == 3

    def test_paren_num_under_decimal(self):
        cands, _ = _pipeline(CASE2B_TEXT)
        # deepest = 2 + 1 = 3; paren_num → L4
        paren_num = [c for c in cands if c.style == "paren_num"]
        assert paren_num, "Should have paren_num candidates"
        assert all(c.level == 4 for c in paren_num)


# ─── Test 4: CJK paren as outer section ──────────────────────────────────────

class TestCase4:
    def test_profile_infers_paren_first_order(self):
        _, profile = _pipeline(CASE4_TEXT)
        assert profile["has_decimal"] is False
        assert profile["cjk_style_order"] == ["cjk_paren", "cjk_comma"]
        assert profile["cjk_style_order_source"] == "parent_child_evidence"
        assert profile["cjk_parent_scores"]["cjk_paren_parent"] > profile["cjk_parent_scores"]["cjk_comma_parent"]
        assert profile["style_order"][:2] == ["cjk_paren", "cjk_comma"]

    def test_levels_follow_detected_cjk_order(self):
        cands, _ = _pipeline(CASE4_TEXT)
        sn = {(c.style, c.num, c.title): c for c in cands}

        assert sn[("cjk_paren", "一", "總則")].level == 1
        assert sn[("cjk_paren", "二", "作業程序")].level == 1
        assert sn[("cjk_comma", "一", "目的")].level == 2
        assert sn[("cjk_comma", "二", "適用範圍")].level == 2


# ─── Test 5: Non-CJK style order from evidence ───────────────────────────────

class TestCase5:
    def test_profile_infers_non_cjk_style_order(self):
        _, profile = _pipeline(CASE5_TEXT)
        assert profile["has_decimal"] is False
        assert profile["style_order"] == ["paren_upper", "paren_lower", "paren_num"]
        assert profile["style_order_source"] == "parent_child_evidence"

    def test_levels_follow_non_cjk_style_order(self):
        cands, _ = _pipeline(CASE5_TEXT)
        st = {(c.style, c.num, c.title): c for c in cands}

        assert st[("paren_upper", "A", "總則")].level == 1
        assert st[("paren_upper", "B", "作業程序")].level == 1
        assert st[("paren_lower", "a", "目的")].level == 2
        assert st[("paren_num", "1", "細項")].level == 3


# ─── Test 3: TOC noise filtering ─────────────────────────────────────────────

class TestCase3:
    def _run(self):
        pages = [CASE3_TEXT]
        records = build_line_records(pages)
        records = clean_line_records(records)
        all_cands = detect_heading_candidates(records)
        toc_regions = find_toc_regions(records)
        return filter_toc_and_noise(all_cands, toc_regions)

    def test_only_body_headings_pass(self):
        cands = self._run()
        nums = [c.num for c in cands]
        # Only 1. 綜合概述 and 1.1 概論 at the end should pass
        assert "1" in nums
        assert "1.1" in nums

    def test_toc_lines_excluded(self):
        cands = self._run()
        raws = [c.raw for c in cands]
        # Dot-lead TOC lines should be excluded
        assert not any("....." in r for r in raws)

    def test_large_num_excluded(self):
        cands = self._run()
        nums = [c.num for c in cands]
        assert "24410" not in nums

    def test_figure_caption_excluded(self):
        cands = self._run()
        raws = [c.raw for c in cands]
        assert not any("圖 1-1" in r for r in raws)

    def test_validate_sequence_rejects_24410(self):
        """Even if 24410 somehow passed noise filter, validate_sequence must reject it."""
        # Simulate by injecting it
        fake = HeadingCandidate(
            page=1, line_no=99, char_start=0, char_end=0,
            style="decimal", num="24410",
            title="輻射劑量評估及輻射防護措施",
            raw="24410 輻射劑量評估及輻射防護措施",
            level=1,
        )
        good1 = HeadingCandidate(
            page=1, line_no=100, char_start=0, char_end=0,
            style="decimal", num="1",
            title="綜合概述", raw="1. 綜合概述", level=1,
        )
        result = validate_sequence([fake, good1])
        nums = [c.num for c in result]
        assert "24410" not in nums
        assert "1" in nums


# ─── Additional edge cases ────────────────────────────────────────────────────

class TestValidateSequence:
    def _make(self, num, title, style="decimal", level=1, line_no=0):
        return HeadingCandidate(
            page=1, line_no=line_no, char_start=0, char_end=0,
            style=style, num=num, title=title,
            raw=f"{num} {title}", level=level,
        )

    def test_normal_sequence_accepted(self):
        cands = [
            self._make("1", "一", line_no=0),
            self._make("2", "二", line_no=1),
            self._make("3", "三", line_no=2),
        ]
        result = validate_sequence(cands)
        assert len(result) == 3

    def test_large_jump_rejected(self):
        cands = [
            self._make("1904", "年份事件", line_no=0),
        ]
        result = validate_sequence(cands)
        assert len(result) == 0

    def test_child_without_parent_rejected(self):
        cands = [
            self._make("2.24", "x10", line_no=0),
        ]
        result = validate_sequence(cands)
        assert len(result) == 0

    def test_child_accepted_with_parent(self):
        cands = [
            self._make("1", "章一", line_no=0, level=1),
            self._make("1.1", "節一", line_no=1, level=2),
        ]
        result = validate_sequence(cands)
        assert len(result) == 2
