"""Tests for doc_structure.py — covers plan test cases 1, 2, 2b, 3."""

import re

import pytest
from doc_structure import (
    HeadingCandidate,
    HeadingPatternSpec,
    LineRecord,
    analyze_structure_profile,
    build_documents_from_pages,
    build_line_records,
    build_marker,
    build_source_ref,
    clean_line_records,
    detect_heading_candidates,
    filter_toc_and_noise,
    find_toc_regions,
    format_chunk_source,
    get_spec,
    infer_levels,
    register_pattern_spec,
    unregister_pattern_spec,
    validate_sequence,
    build_heading_tree,
    _merge_metadata,
)


def _candidate(num, title, pattern="decimal", level=0, line_no=0, raw=None):
    """Build a HeadingCandidate the way detection does (marker is source of truth)."""
    raw = raw if raw is not None else f"{num} {title}"
    spec = get_spec(pattern)
    return HeadingCandidate(
        page=1, line_no=line_no, char_start=0, char_end=0,
        pattern_id=spec.pattern_id,
        marker=build_marker(spec, num, raw), num=num, title=title,
        raw=raw, level=level,
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
        sn = {(c.pattern_id, c.num): c.level for c in cands}

        assert sn[("decimal", "1")] == 1
        assert sn[("decimal", "1.1")] == 2
        assert sn[("decimal", "1.1.1")] == 3
        assert sn[("decimal", "1.1.2")] == 3

    def test_paren_levels(self):
        cands, _ = _pipeline(CASE1_TEXT)
        paren_cands = [c for c in cands if c.pattern_id == "paren_num"]
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
        assert profile["anchor_order"][:2] == ["cjk_comma", "cjk_paren"]
        assert profile["anchor_order_source"] == "parent_child_evidence"
        # evidence favours cjk_comma as the outer (parent) pattern
        assert profile["parent_child_scores"].get("cjk_comma->cjk_paren", 0) > 0

    def test_levels(self):
        cands, _ = _pipeline(CASE2_TEXT)
        # Key by (style, num) to avoid collision between cjk_comma "一" and cjk_paren "一"
        sn_level = {(c.pattern_id, c.num): c.level for c in cands}

        assert sn_level[("cjk_comma", "一")] == 1
        assert sn_level[("cjk_comma", "二")] == 1

    def test_cjk_paren_level(self):
        cands, _ = _pipeline(CASE2_TEXT)
        cjk_paren = [c for c in cands if c.pattern_id == "cjk_paren"]
        assert all(c.level == 2 for c in cjk_paren)

    def test_paren_num_level(self):
        cands, _ = _pipeline(CASE2_TEXT)
        # deepest = 2 + 0 = 2; paren_num → L3
        paren_num = [c for c in cands if c.pattern_id == "paren_num"]
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
        sn = {(c.pattern_id, c.num): c for c in cands}

        assert sn[("cjk_comma", "三")].level == 1
        assert sn[("cjk_paren", "一")].level == 2
        assert sn[("decimal", "1")].level == 3
        assert sn[("decimal", "2")].level == 3

    def test_paren_num_under_decimal(self):
        cands, _ = _pipeline(CASE2B_TEXT)
        # deepest = 2 + 1 = 3; paren_num → L4
        paren_num = [c for c in cands if c.pattern_id == "paren_num"]
        assert paren_num, "Should have paren_num candidates"
        assert all(c.level == 4 for c in paren_num)


# ─── Test 4: CJK paren as outer section ──────────────────────────────────────

class TestCase4:
    def test_profile_infers_paren_first_order(self):
        _, profile = _pipeline(CASE4_TEXT)
        assert profile["has_decimal"] is False
        assert profile["anchor_order"][:2] == ["cjk_paren", "cjk_comma"]
        assert profile["anchor_order_source"] == "parent_child_evidence"

    def test_levels_follow_detected_cjk_order(self):
        cands, _ = _pipeline(CASE4_TEXT)
        sn = {(c.pattern_id, c.num, c.title): c for c in cands}

        assert sn[("cjk_paren", "一", "總則")].level == 1
        assert sn[("cjk_paren", "二", "作業程序")].level == 1
        assert sn[("cjk_comma", "一", "目的")].level == 2
        assert sn[("cjk_comma", "二", "適用範圍")].level == 2


# ─── Test 5: Non-CJK style order from evidence ───────────────────────────────

class TestCase5:
    def test_profile_infers_non_cjk_style_order(self):
        _, profile = _pipeline(CASE5_TEXT)
        assert profile["has_decimal"] is False
        assert profile["anchor_order"] == ["paren_upper", "paren_lower", "paren_num"]
        assert profile["anchor_order_source"] == "parent_child_evidence"

    def test_levels_follow_non_cjk_style_order(self):
        cands, _ = _pipeline(CASE5_TEXT)
        st = {(c.pattern_id, c.num, c.title): c for c in cands}

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
        fake = _candidate(
            "24410", "輻射劑量評估及輻射防護措施", line_no=99, level=1,
            raw="24410 輻射劑量評估及輻射防護措施",
        )
        good1 = _candidate("1", "綜合概述", line_no=100, level=1, raw="1. 綜合概述")
        result = validate_sequence([fake, good1])
        nums = [c.num for c in result]
        assert "24410" not in nums
        assert "1" in nums


# ─── Additional edge cases ────────────────────────────────────────────────────

class TestValidateSequence:
    def _make(self, num, title, style="decimal", level=1, line_no=0):
        return _candidate(num, title, pattern=style, level=level, line_no=line_no)

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


# ─── Chunk metadata: contained_sections + list-based heading_tree ─────────────

def _single_chunk_meta(text: str) -> dict:
    docs, _ = build_documents_from_pages(
        pages=[text], chunk_size=1000, overlap_size=0, source_file="要點.pdf",
    )
    assert len(docs) == 1, f"expected one chunk, got {len(docs)}"
    return docs[0].metadata


class TestContainedSections:
    def test_single_section(self):
        meta = _single_chunk_meta("三、作業程序")
        assert meta["contained_sections"] == ["三、作業程序"]

    def test_two_subsections(self):
        meta = _single_chunk_meta("四、任務分工\n（一）災防辦\n（二）核安會")
        assert meta["contained_sections"] == [
            "四、任務分工 > （一）災防辦",
            "四、任務分工 > （二）核安會",
        ]

    def test_three_levels(self):
        meta = _single_chunk_meta(
            "三、作業程序\n（二）一級開設\n1. 開設時機\n2. 進駐機關"
        )
        assert meta["contained_sections"] == [
            "三、作業程序 > （二）一級開設 > 1. 開設時機",
            "三、作業程序 > （二）一級開設 > 2. 進駐機關",
        ]


def _walk_nodes(tree: dict):
    """Yield every node in a structure_tree (root + descendants)."""
    if not tree:
        return
    yield tree
    for child in tree.get("items") or []:
        yield from _walk_nodes(child)


def _count_tokens_safe(text: str) -> int:
    from doc_structure import _count_tokens
    return _count_tokens(text)


class TestStructureTreeShape:
    def test_items_is_list_with_pattern_id(self):
        meta = _single_chunk_meta("三、作業程序\n（二）一級開設\n1. 開設時機")
        tree = meta["structure_tree"]
        assert tree["section_key"] == "cjk_comma:三"
        assert tree["pattern_id"] == "cjk_comma"
        assert isinstance(tree["items"], list)
        child = tree["items"][0]
        assert child["section_key"] == "cjk_paren:二"
        assert child["pattern_id"] == "cjk_paren"
        leaf = child["items"][0]
        assert leaf["section_key"] == "decimal:1"
        assert leaf["pattern_id"] == "decimal"
        assert leaf["title"] == "開設時機"  # trailing colon stripped from title

    def test_no_kind_or_role_field_anywhere(self):
        meta = _single_chunk_meta("三、作業程序\n（二）一級開設\n1. 開設時機")
        for node in _walk_nodes(meta["structure_tree"]):
            assert "kind" not in node
            assert "role" not in node          # role removed from the data model
            assert "pattern_id" in node

    def test_no_legacy_fields(self):
        meta = _single_chunk_meta("三、作業程序")
        for stale in ("heading_path", "contains_sections", "breadcrumb"):
            assert stale not in meta


class TestMergeTrees:
    def test_merge_keyed_by_section_key_preserves_order(self):
        m1 = _single_chunk_meta("四、任務分工\n（一）災防辦")
        m2 = _single_chunk_meta("四、任務分工\n（二）核安會")
        merged = _merge_metadata(m1, m2)
        keys = [c["section_key"] for c in merged["structure_tree"]["items"]]
        assert keys == ["cjk_paren:一", "cjk_paren:二"]
        assert merged["contained_sections"] == [
            "四、任務分工 > （一）災防辦",
            "四、任務分工 > （二）核安會",
        ]
        # merged metadata records that a merge happened via the warnings field
        assert "small_chunk_merged" in merged["warnings"]
        assert "merge_applied" not in merged

    def test_merge_does_not_mutate_sources(self):
        m1 = _single_chunk_meta("四、任務分工\n（一）災防辦")
        m2 = _single_chunk_meta("四、任務分工\n（二）核安會")
        _merge_metadata(m1, m2)
        assert len(m1["structure_tree"]["items"]) == 1


class TestSourceFormatting:
    def test_source_ref_single_page(self):
        meta = {"source_file": "要點.pdf", "page_start": 3, "page_end": 3, "chunk_index": 8}
        assert build_source_ref(meta) == "要點.pdf#p3#chunk8"

    def test_source_ref_multi_page(self):
        meta = {"source_file": "要點.pdf", "page_start": 3, "page_end": 4, "chunk_index": 8}
        assert build_source_ref(meta) == "要點.pdf#p3-p4#chunk8"

    def test_format_chunk_source_with_sections(self):
        meta = {
            "source_file": "要點.pdf", "page_start": 3, "page_end": 3, "chunk_index": 8,
            "contained_sections": ["三、作業程序 > （二）一級開設 > 1. 開設時機"],
        }
        out = format_chunk_source(meta)
        assert "來源：要點.pdf，第 3 頁，chunk 8" in out
        assert "本段包含的結構路徑：" in out
        assert "- 三、作業程序 > （二）一級開設 > 1. 開設時機" in out

    def test_format_chunk_source_no_sections(self):
        meta = {
            "source_file": "要點.pdf", "page_start": 3, "page_end": 3, "chunk_index": 8,
            "contained_sections": [],
        }
        out = format_chunk_source(meta)
        assert out == "來源：要點.pdf，第 3 頁，chunk 8"


# ─── Plan Test 1: pipeline no longer depends on role ──────────────────────────

class TestNoRoleDependency:
    def test_profile_uses_anchor_order_not_structural_enum(self):
        docs, structure_info = build_documents_from_pages(
            pages=["四、任務分工\n（一）災防辦\n1. 協助督導"],
            chunk_size=1000, overlap_size=0, source_file="要點.pdf",
        )
        profile = structure_info["profile"]
        assert "anchor_order" in profile
        assert "structural_order" not in profile
        assert "enum_order" not in profile
        assert "style_order" not in profile

    def test_candidate_has_no_role_attribute(self):
        records = clean_line_records(build_line_records(["一、目的\n（一）範圍"]))
        cands = detect_heading_candidates(records)
        assert cands and not hasattr(cands[0], "role")


# ─── Plan Test 2: every marker is an anchor (no structural/enum gate) ─────────

class TestAllMarkersAreAnchors:
    def test_all_marker_kinds_detected(self):
        text = "四、任務分工：\n（一）災防辦：\n1. 協助督導。\n2. 彙整資訊。"
        records = clean_line_records(build_line_records([text]))
        pids = {c.pattern_id for c in detect_heading_candidates(records)}
        assert {"cjk_comma", "cjk_paren", "decimal"} <= pids

    def test_every_anchor_enters_structure_tree(self):
        # decimal AND paren_num both become nodes — no role excludes them.
        meta = _single_chunk_meta("三、作業程序\n（一）一級開設\n1. 開設時機\n(1) 子項")
        keys = {n["section_key"] for n in _walk_nodes(meta["structure_tree"])}
        assert "cjk_comma:三" in keys
        assert "cjk_paren:一" in keys
        assert "decimal:1" in keys
        assert "paren_num:1" in keys


# ─── Plan Test 3 / 4: delayed splitting ───────────────────────────────────────

TASK_TEXT = (
    "四、任務分工：\n（一）災防辦：\n1. 協助督導各部會應變處置。\n"
    "2. 彙整相關資訊。\n3. 其他交辦事項。\n（二）核安會：\n"
    "1. 辦理輻射監測。\n2. 提供技術支援。"
)


class TestDelayedSplitting:
    def test_large_chunk_size_keeps_one_chunk(self):
        docs, _ = build_documents_from_pages(
            pages=[TASK_TEXT], chunk_size=2000, overlap_size=0, source_file="x.pdf")
        assert len(docs) == 1  # fits + few top anchors → not split fine-grained

    def test_small_chunk_size_splits_at_outer_anchor_first(self):
        # chunk_size forces a split of the whole, but each （）group still fits.
        docs, _ = build_documents_from_pages(
            pages=[TASK_TEXT], chunk_size=80, overlap_size=0, source_file="x.pdf")
        # Should split into （一） / （二） groups, NOT one chunk per "1."
        joined = [d.page_content for d in docs]
        assert len(docs) >= 2
        # the 災防辦 group keeps its three numbered items together
        assert any("災防辦" in c and "1. 協助督導各部會應變處置" in c
                   and "3. 其他交辦事項" in c for c in joined)
        # we did not explode into a chunk that is just a lone "1." item
        lone = [c for c in joined if c.strip().startswith("1.") and "災防辦" not in c]
        assert not lone


# ─── Plan Test 5 / 6: contained_sections + structure_tree ─────────────────────

class TestStructureTreeMetadata:
    def test_contained_sections_and_tree(self):
        meta = _single_chunk_meta("四、任務分工：\n（一）災防辦：\n1. 協助督導。\n2. 彙整資訊。")
        assert meta["contained_sections"] == [
            "四、任務分工 > （一）災防辦 > 1. 協助督導",
            "四、任務分工 > （一）災防辦 > 2. 彙整資訊",
        ]
        tree = meta["structure_tree"]
        assert tree["pattern_id"] == "cjk_comma"
        assert tree["items"][0]["pattern_id"] == "cjk_paren"
        assert tree["items"][0]["items"][0]["pattern_id"] == "decimal"

    def test_lean_metadata_fields_only(self):
        meta = _single_chunk_meta("三、作業程序")
        # Final metadata is the lean public set; diagnostics are not exposed.
        assert set(meta) == {
            "source_file", "doc_title", "page_start", "page_end", "chunk_index",
            "contained_sections", "structure_tree", "warnings",
        }
        for dropped in ("heading_tree", "structure_type", "split_strategy",
                        "structure_confidence", "anchor_order_confidence",
                        "anchor_order_support_count", "merge_applied", "merge_reason"):
            assert dropped not in meta


# ─── Plan Test 7: MapReduce-friendly — no flood of tiny chunks ────────────────

class TestNoTinyChunkFlood:
    def test_small_chunks_are_merged_and_marked(self):
        docs, _ = build_documents_from_pages(
            pages=[TASK_TEXT], chunk_size=60, overlap_size=0, source_file="x.pdf")
        min_tokens = max(80, 60 // 10)
        tiny = [d for d in docs if _count_tokens_safe(d.page_content) < min_tokens]
        # any surviving tiny chunk is rare; merged ones are flagged in warnings
        assert len(tiny) <= 2
        merged = [d for d in docs if "small_chunk_merged" in (d.metadata.get("warnings") or [])]
        assert merged, "expected at least one merged chunk for this tiny chunk_size"
        for d in docs:
            assert "merge_applied" not in d.metadata
            assert "merge_reason" not in d.metadata


# ─── Plan Test 4 / 5 (anchor order): evidence beats first-seen ────────────────

class TestAnchorOrderEvidence:
    def test_inner_pattern_first_does_not_mislead(self):
        text = ("（一）前言說明\n一、正式章節\n（一）正式子章節\n"
                "1. 開設時機\n二、第二章節")
        _, profile = _pipeline(text)
        assert profile["anchor_order"][0] == "cjk_comma"
        assert profile["anchor_order"][:2] == ["cjk_comma", "cjk_paren"]

    def test_evidence_overrides_first_seen(self):
        text = ("（一）摘要\n一、目的\n（一）適用範圍\n二、作業程序\n"
                "（一）處理流程\n（二）通報流程")
        _, profile = _pipeline(text)
        assert profile["anchor_order"] == ["cjk_comma", "cjk_paren"]
        assert profile["anchor_order_source"] == "parent_child_evidence"

    def test_resetting_inner_counter_not_mistaken_for_outer(self):
        text = (
            "三、作業程序\n"
            "（一）二級開設\n1. 開設時機\n2. 參與機關\n"
            "（二）一級開設\n1. 開設時機\n2. 參與機關\n"
        )
        _, profile = _pipeline(text)
        assert profile["anchor_order"] == ["cjk_comma", "cjk_paren", "decimal"]

    def test_resetting_inner_builds_correct_tree(self):
        meta = _single_chunk_meta(
            "三、作業程序\n（一）二級開設\n1. 開設時機\n2. 參與機關"
        )
        assert meta["contained_sections"] == [
            "三、作業程序 > （一）二級開設 > 1. 開設時機",
            "三、作業程序 > （一）二級開設 > 2. 參與機關",
        ]
        tree = meta["structure_tree"]
        assert tree["level"] == 1                       # 三
        assert tree["items"][0]["level"] == 2           # （一）
        assert tree["items"][0]["items"][0]["level"] == 3  # 1.
