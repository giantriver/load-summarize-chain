"""
doc_structure.py — Anchor-based document structure detection and chunk splitting.

Every detected marker is just an *anchor candidate*: we never classify it as a
"section" vs a "list item" by its regex. Per document we infer each anchor's
LEVEL (from parent/child evidence) and split lazily — a range stays whole until
it is too big or spans too many top-level anchors. Chunk metadata carries
contained_sections + structure_tree (detected structure, not a strict ToC).

Pipeline:
  build_line_records → clean_line_records → detect_heading_candidates
  → find_toc_regions → filter_toc_and_noise → find_table_regions
  → filter_table_regions → analyze_structure_profile (anchor_order)
  → infer_levels → validate_sequence → split_by_candidate_lines
  → repair_small_chunks_docs
"""

import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None

# ─── Heading pattern registry ────────────────────────────────────────────────

# NOTE: every matched marker is just an *anchor candidate*. We deliberately do
# NOT classify a pattern as "structural" vs "enum" — the same marker shape plays
# different roles in different documents (e.g. （一）/（二） can be a real section
# or a mere paragraph list; (1)(2)(3) can head English sections). The role is
# decided per-document by anchor LEVEL + delayed splitting, never by the regex.


@dataclass(frozen=True)
class HeadingPatternSpec:
    """One anchor-marker rule. Adding an anchor shape = adding one spec here.

    pattern_id    : stable identity of the rule (also used in section_key).
    regex         : compiled pattern with named groups (?P<num>) (?P<title>).
    priority      : detection order — lowest value tried first, first match wins.
    num_type      : arabic / arabic_chain / cjk / latin_upper / latin_lower.
    marker_style  : how build_marker derives prefix/suffix —
                    "decimal" (trailing separator from raw),
                    "paren"   (bracket width detected from raw),
                    "affix"   (literal prefix/suffix taken from this spec).
    prefix/suffix : literal affixes used only by the "affix" marker_style.
    level_strategy: "flat" (single level) or "decimal_chain" (depth from dots).
    examples      : documentation only.
    """
    pattern_id: str
    regex: "re.Pattern[str]"
    priority: int
    num_type: str
    marker_style: str = "affix"
    prefix: str = ""
    suffix: str = ""
    level_strategy: str = "flat"
    examples: tuple = ()


_DEFAULT_PATTERN_SPECS: list[HeadingPatternSpec] = [
    HeadingPatternSpec(
        pattern_id="decimal",
        regex=re.compile(r"^(?P<num>\d+(?:[.．]\d+)*)[.．、]?\s*(?P<title>[一-鿿A-Za-z].+)$"),
        priority=10, num_type="arabic",
        marker_style="decimal", level_strategy="decimal_chain",
        examples=("1.", "1.1", "2.3.4", "10.3.5"),
    ),
    HeadingPatternSpec(
        pattern_id="paren_num",
        regex=re.compile(r"^[\(（](?P<num>\d+)[\)）][.．]?\s*(?P<title>.+)$"),
        priority=20, num_type="arabic", marker_style="paren",
        examples=("(1)", "（2）", "(1)."),
    ),
    HeadingPatternSpec(
        pattern_id="paren_upper",
        regex=re.compile(r"^[\(（](?P<num>[A-Z])[\)）][.．]?\s*(?P<title>.+)$"),
        priority=30, num_type="latin_upper", marker_style="paren",
        examples=("(A)", "（B）"),
    ),
    HeadingPatternSpec(
        pattern_id="paren_lower",
        regex=re.compile(r"^[\(（](?P<num>[a-z])[\)）][.．]?\s*(?P<title>.+)$"),
        priority=40, num_type="latin_lower", marker_style="paren",
        examples=("(a)", "（b）"),
    ),
    HeadingPatternSpec(
        pattern_id="cjk_comma",
        regex=re.compile(r"^(?P<num>[一二三四五六七八九十百]+)、\s*(?P<title>.+)$"),
        priority=50, num_type="cjk",
        marker_style="affix", suffix="、",
        examples=("一、目的", "十二、附則"),
    ),
    HeadingPatternSpec(
        pattern_id="cjk_paren",
        regex=re.compile(r"^[（(](?P<num>[一二三四五六七八九十百]+)[）)]\s*(?P<title>.+)$"),
        priority=60, num_type="cjk", marker_style="paren",
        examples=("（一）目的", "（十）附則"),
    ),
]

# Delayed split: even if a range fits in chunk_size, keep splitting while it
# spans more than this many top-level anchors (distinct outermost sections).
MAX_ANCHORS_PER_CHUNK = 6
# …but the anchor-count cap is token-aware: it only forces a split once a range
# is reasonably large. A range that fits well under this fraction of chunk_size
# is kept whole even with many top anchors (e.g. a 7-item list that is still
# small) — splitting it would only create tiny fragments that repair merges back.
ANCHOR_SPLIT_MIN_RATIO = 0.5
# Below this much parent/child evidence, the anchor order is weakly supported.
MIN_ANCHOR_SUPPORT = 3

# Active registry — mutable so callers can register extra specs at runtime.
_PATTERN_SPECS: list[HeadingPatternSpec] = list(_DEFAULT_PATTERN_SPECS)
_SPEC_BY_ID: dict[str, HeadingPatternSpec] = {s.pattern_id: s for s in _PATTERN_SPECS}


def get_pattern_specs() -> list[HeadingPatternSpec]:
    """All registered specs, in registration order."""
    return list(_PATTERN_SPECS)


def get_spec(pattern_id: str) -> HeadingPatternSpec:
    return _SPEC_BY_ID[pattern_id]


def iter_specs_by_priority() -> list[HeadingPatternSpec]:
    """Specs ordered for detection: lowest priority value first."""
    return sorted(_PATTERN_SPECS, key=lambda s: s.priority)


def default_anchor_order() -> list[str]:
    """Priority-ordered pattern ids; used only as an anchor-order tie-break."""
    return [s.pattern_id for s in iter_specs_by_priority()]


def is_decimal_chain_spec(pattern_id: str) -> bool:
    spec = _SPEC_BY_ID.get(pattern_id)
    return bool(spec and spec.level_strategy == "decimal_chain")


def register_pattern_spec(spec: HeadingPatternSpec, *, replace: bool = False) -> None:
    """Add (or replace) a spec in the active registry."""
    if spec.pattern_id in _SPEC_BY_ID:
        if not replace:
            raise ValueError(f"pattern_id already registered: {spec.pattern_id}")
        unregister_pattern_spec(spec.pattern_id)
    _PATTERN_SPECS.append(spec)
    _SPEC_BY_ID[spec.pattern_id] = spec


def unregister_pattern_spec(pattern_id: str) -> None:
    """Remove a spec (no-op if absent). Mainly for test isolation."""
    spec = _SPEC_BY_ID.pop(pattern_id, None)
    if spec is not None:
        _PATTERN_SPECS.remove(spec)


# bare number: 序號與標題被 PDF 拆成兩行
BARE_NUMBER_RE = re.compile(r"^(?P<num>\d+(?:[.．]\d+)*)[.．、]$")

PAGE_NUMBER_RE = re.compile(r"第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁")
SENTENCE_END_RE = re.compile(r"[。！？；;：:]$")
FIGURE_TABLE_RE = re.compile(r"^[圖表]\s*\d+([-.–]\d+)?\s*[：:]")

# Table caption (表N-N：…) and column-header row (項次/編號/序號…) anchor a table
# region. Numbered lines beneath such an anchor are table rows, not headings.
TABLE_CAPTION_RE = re.compile(r"^表\s*\d+(?:[-.–—]\d+)*\s*[：:]")
TABLE_HEADER_RE = re.compile(r"^(?:項\s*次|編\s*號|序\s*號|項\s*目)\s")
TOC_TITLE_RE = re.compile(
    r"^(目\s*錄|頁\s*次|附\s*圖\s*目\s*錄|附\s*表\s*目\s*錄|圖\s*目\s*錄|表\s*目\s*錄)$"
)

MIN_MULTI_DECIMAL = 3   # multi-level decimals needed to call has_decimal=True
MIN_DEPTH_SUPPORT = 2   # min occurrences for a depth to count toward max_decimal_depth
MAX_JUMP = 5            # allow this many skipped numbers before rejecting

# Table-region detection: a "bare integer" row is `1 文字` with NO separator —
# distinct from a real section `1. 文字`, which carries a period. A run of such
# rows is treated as a table (rows suppressed as headings) when it sits under a
# caption/header anchor, or when the run alone is long enough to be unambiguous.
TABLE_GAP_TOLERANCE = 4      # max blank/continuation lines allowed between rows
TABLE_ANCHOR_LOOKBACK = 3    # non-blank lines to scan above a run for an anchor
TABLE_ANCHORED_MIN = 2       # min rows for an anchored run to count as a table
TABLE_STANDALONE_MIN = 5     # min rows for an unanchored run to count as a table


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class LineRecord:
    page: int
    text: str
    char_start: int = 0
    char_end: int = 0


@dataclass
class HeadingCandidate:
    """A detected anchor marker. Just an anchor candidate — no structural/enum
    role is assigned; its place is decided per-document by `level` + splitting."""
    page: int
    line_no: int
    char_start: int
    char_end: int
    pattern_id: str
    marker: dict
    num: str
    title: str
    raw: str
    level: int = 0


# ─── Marker ──────────────────────────────────────────────────────────────────

def build_marker(spec: HeadingPatternSpec, num: str, raw: str) -> dict:
    """Build the marker descriptor (text / num / num_type / prefix / suffix).

    All format knowledge comes from the spec — this never branches on a specific
    pattern_id, only on the spec's marker_style.
    """
    raw = (raw or "").strip()
    num_type = spec.num_type
    prefix = ""
    suffix = ""

    if spec.marker_style == "decimal":
        if "." in num or "．" in num:
            num_type = "arabic_chain"
        # Capture a trailing separator directly following the number (e.g. "1.")
        after = raw[len(num):len(num) + 1] if raw.startswith(num) else ""
        if after in {".", "．", "、"}:
            suffix = after
    elif spec.marker_style == "paren":
        prefix = "（" if raw[:1] == "（" else "("
        suffix = "）" if "）" in raw else ")"
    else:  # "affix": literal prefix/suffix from the spec (e.g. 第…條, …、)
        prefix = spec.prefix
        suffix = spec.suffix

    return {
        "text": f"{prefix}{num}{suffix}",
        "num": num,
        "num_type": num_type,
        "prefix": prefix,
        "suffix": suffix,
    }


# ─── Token counting ──────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return len(text) // 4


# ─── Heading line utilities ──────────────────────────────────────────────────

def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if BARE_NUMBER_RE.match(stripped):
        return True
    return any(spec.regex.match(stripped) for spec in get_pattern_specs())


def is_standalone_section_heading(line: str) -> bool:
    stripped = line.strip()
    for spec in get_pattern_specs():
        if spec.pattern_id in ("cjk_comma", "cjk_paren") and spec.regex.match(stripped):
            return True
    return False


# ─── Build line records ──────────────────────────────────────────────────────

def build_line_records(pages: list[str]) -> list[LineRecord]:
    records: list[LineRecord] = []
    cursor = 0
    for page_idx, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            char_start = cursor
            char_end = cursor + len(line)
            records.append(LineRecord(
                page=page_idx, text=line,
                char_start=char_start, char_end=char_end,
            ))
            cursor = char_end + 1  # +1 for the newline separator
    return records


# ─── Clean line records ──────────────────────────────────────────────────────

def _should_keep_line_break(prev: str, nxt: str) -> bool:
    if not prev or not nxt:
        return True
    # Keep break after any heading so its body text isn't merged into the heading
    if is_heading_line(prev):
        return True
    if is_heading_line(nxt):
        return True
    # Keep TOC section titles and figure/table captions on their own line
    if TOC_TITLE_RE.match(nxt) or FIGURE_TABLE_RE.match(nxt):
        return True
    return bool(SENTENCE_END_RE.search(prev))


def clean_line_records(records: list[LineRecord]) -> list[LineRecord]:
    """Strip page-number patterns, collapse blanks, and unwrap soft line-breaks."""
    # Pass 1: strip page numbers and whitespace
    step1 = [
        LineRecord(page=r.page, text=PAGE_NUMBER_RE.sub("", r.text).strip(),
                   char_start=r.char_start, char_end=r.char_end)
        for r in records
    ]

    # Pass 2: collapse consecutive blank lines
    step2: list[LineRecord] = []
    prev_blank = False
    for r in step1:
        if not r.text:
            if not prev_blank:
                step2.append(r)
            prev_blank = True
        else:
            step2.append(r)
            prev_blank = False

    # Pass 3: unwrap – merge continuation lines
    step3: list[LineRecord] = []
    for r in step2:
        if not r.text:
            if step3 and step3[-1].text:
                step3.append(r)
            continue
        if not step3 or not step3[-1].text:
            step3.append(r)
            continue
        prev = step3[-1]
        if _should_keep_line_break(prev.text, r.text):
            step3.append(r)
        else:
            step3[-1] = LineRecord(
                page=prev.page,
                text=prev.text + r.text,
                char_start=prev.char_start,
                char_end=r.char_end,
            )

    # Pass 4: rebuild contiguous char positions
    cursor = 0
    final: list[LineRecord] = []
    for r in step3:
        char_start = cursor
        char_end = cursor + len(r.text)
        final.append(LineRecord(page=r.page, text=r.text,
                                char_start=char_start, char_end=char_end))
        cursor = char_end + 1
    return final


# ─── Detect heading candidates ───────────────────────────────────────────────

def _match_spec(text: str) -> tuple[HeadingPatternSpec, "re.Match[str]"] | None:
    """Return the first (by priority) spec whose regex matches text, else None."""
    for spec in iter_specs_by_priority():
        m = spec.regex.match(text)
        if m:
            return spec, m
    return None


def detect_heading_candidates(line_records: list[LineRecord]) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    n = len(line_records)

    for i, rec in enumerate(line_records):
        line = rec.text.strip()
        if not line:
            continue

        # Try bare number merge (PDF split across two lines)
        bare = BARE_NUMBER_RE.match(line)
        if bare and i + 1 < n:
            nxt = line_records[i + 1]
            nxt_text = nxt.text.strip()
            if nxt_text and not is_heading_line(nxt_text):
                merged = line + " " + nxt_text
                hit = _match_spec(merged)
                if hit:
                    spec, m = hit
                    num = m.group("num")
                    candidates.append(HeadingCandidate(
                        page=rec.page, line_no=i,
                        char_start=rec.char_start, char_end=nxt.char_end,
                        pattern_id=spec.pattern_id,
                        marker=build_marker(spec, num, merged), num=num,
                        title=m.group("title"), raw=merged,
                    ))
                continue

        hit = _match_spec(line)
        if hit:
            spec, m = hit
            num = m.group("num")
            candidates.append(HeadingCandidate(
                page=rec.page, line_no=i,
                char_start=rec.char_start, char_end=rec.char_end,
                pattern_id=spec.pattern_id,
                marker=build_marker(spec, num, line), num=num,
                title=m.group("title"), raw=line,
            ))

    return candidates


# ─── Noise filter ────────────────────────────────────────────────────────────

def is_noise_line(c: HeadingCandidate) -> bool:
    line = c.raw.strip()

    # 1. Bare page number (Arabic or Roman)
    if re.match(r"^(i{1,4}|vi{0,3}|xi{0,3}|[IVX]+|\d+)$", line, re.IGNORECASE):
        return True

    # 2. TOC/index section titles
    if TOC_TITLE_RE.match(line):
        return True

    # 3. TOC lines – dot-leader style OR no-dot-leader style (title ends with
    #    a bare page number, e.g. "1. 綜合概述 1" or "1.1 概論 1")
    if re.search(r"[.．…]{3,}", line):
        return True
    if c.pattern_id == "decimal" and re.search(r"\s+\d+\s*$", c.title.strip()):
        return True

    # 4. Figure / table captions
    if FIGURE_TABLE_RE.match(line):
        return True

    # 5. A single-integer decimal. Under the anchor model a numbered line IS a
    #    valid (deep) anchor — there is no list-vs-section split — so we do NOT try
    #    to second-guess "prose vs list item" here (neither a trailing 。！？ nor an
    #    internal comma disqualifies it; e.g. "1. 協助督導。" and "1. 全國…，…。" are
    #    both kept). The only reject is leading-zero numbers (004), which are
    #    codes/IDs rather than section/anchor numbers.
    if c.pattern_id == "decimal" and "." not in c.num and "．" not in c.num:
        # Leading-zero numbers (004) are codes/IDs, never section/list anchors.
        # NOTE: there is deliberately NO comma-based filter. Under the anchor model
        # a numbered line is just an anchor candidate — there is no list-vs-section
        # split — so comma-bearing list items (very common in regulations) must not
        # be dropped at detection. Whether such an anchor matters is decided later
        # by level + delayed splitting, not by guessing "prose vs item" here.
        if len(c.num) > 1 and c.num[0] == "0":
            return True

    return False


# ─── TOC region detection ────────────────────────────────────────────────────

def find_toc_regions(line_records: list[LineRecord]) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) for TOC regions in line_records."""
    toc_regions: list[tuple[int, int]] = []
    in_toc = False
    toc_start = -1

    for i, rec in enumerate(line_records):
        text = rec.text.strip()
        if not text:
            continue

        # Additional TOC titles while already in a TOC section are absorbed
        if TOC_TITLE_RE.match(text):
            if not in_toc:
                in_toc = True
                toc_start = i
            continue  # keep accumulating even if already in_toc

        if not in_toc:
            continue

        # We are inside a TOC region — check if this line is the real body start.
        # A "real body heading" must: (a) match a heading pattern, (b) have no dot
        # leader, (c) title must NOT end with a bare page number (which would mean
        # it is a no-dot-leader TOC entry like "1. 綜合概述 1"), and (d) for decimal
        # style the leading number must be ≤ MAX_JUMP+1 (to exclude codes like 24410).
        is_real_body = False
        for spec in get_pattern_specs():
            m = spec.regex.match(text)
            if not m or re.search(r"[.．…]{3,}", text):
                continue
            # TOC entries without dot-leaders end with a bare page number
            if re.search(r"\s+\d+\s*$", m.group("title")):
                continue
            if spec.pattern_id == "decimal":
                parts = _parse_decimal_parts(m.group("num"))
                if parts and parts[0] <= MAX_JUMP + 1:
                    is_real_body = True
                    break
            else:
                # CJK and paren headings are unambiguous body starts
                is_real_body = True
                break

        if is_real_body:
            toc_regions.append((toc_start, i - 1))
            in_toc = False
            toc_start = -1

    if in_toc and toc_start >= 0:
        toc_regions.append((toc_start, len(line_records) - 1))

    return toc_regions


def find_body_start_page(line_records: list[LineRecord]) -> int:
    for rec in line_records:
        text = rec.text.strip()
        if not text:
            continue
        for spec in get_pattern_specs():
            if spec.regex.match(text) and not re.search(r"[.．…]{3,}", text):
                return rec.page
    return 1


def filter_toc_and_noise(
    candidates: list[HeadingCandidate],
    toc_regions: list[tuple[int, int]],
) -> list[HeadingCandidate]:
    toc_line_nos: set[int] = set()
    for start, end in toc_regions:
        toc_line_nos.update(range(start, end + 1))

    return [
        c for c in candidates
        if c.line_no not in toc_line_nos and not is_noise_line(c)
    ]


# ─── Table region detection ──────────────────────────────────────────────────

def _bare_integer_row(line: str) -> int | None:
    """Return the integer if `line` is a bare single-integer row (`1 文字`).

    Returns None for real decimal sections (`1. 文字` / `1.1 文字`): a trailing
    period/comma separator or a multi-level number means it is a heading, not a
    table row. The separator is the key signal that protects real sections.
    """
    m = get_spec("decimal").regex.match(line)
    if not m:
        return None
    num = m.group("num")
    if "." in num or "．" in num:
        return None
    after = line[len(num):len(num) + 1]
    if after in {".", "．", "、"}:
        return None
    try:
        return int(num)
    except ValueError:
        return None


def _find_table_anchor(line_records: list[LineRecord], start_idx: int) -> int | None:
    """Scan up to TABLE_ANCHOR_LOOKBACK non-blank lines above start_idx for a
    table caption or a column-header row. Return its line index, else None."""
    seen = 0
    j = start_idx - 1
    while j >= 0 and seen < TABLE_ANCHOR_LOOKBACK:
        text = line_records[j].text.strip()
        if not text:
            j -= 1
            continue
        if TABLE_CAPTION_RE.match(text) or TABLE_HEADER_RE.match(text):
            return j
        seen += 1
        j -= 1
    return None


def find_table_regions(line_records: list[LineRecord]) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) spans that are tables, so the numbered rows
    inside them are not mistaken for section headings.

    Bare-integer rows are grouped into runs (tolerating continuation lines); a
    run becomes a table region when it is anchored by a caption/header above it,
    or is long enough to stand alone.
    """
    rows = [
        (i, v)
        for i, rec in enumerate(line_records)
        if rec.text.strip() and (v := _bare_integer_row(rec.text.strip())) is not None
    ]
    if not rows:
        return []

    regions: list[tuple[int, int]] = []

    def _finalize(run: list[tuple[int, int]]) -> None:
        if not run:
            return
        start_idx, end_idx = run[0][0], run[-1][0]
        anchor = _find_table_anchor(line_records, start_idx)
        anchored = anchor is not None and len(run) >= TABLE_ANCHORED_MIN
        standalone = len(run) >= TABLE_STANDALONE_MIN
        if anchored or standalone:
            regions.append((anchor if anchor is not None else start_idx, end_idx))

    run = [rows[0]]
    for idx, val in rows[1:]:
        if idx - run[-1][0] <= TABLE_GAP_TOLERANCE + 1:
            run.append((idx, val))
        else:
            _finalize(run)
            run = [(idx, val)]
    _finalize(run)

    return regions


def filter_table_regions(
    candidates: list[HeadingCandidate],
    table_regions: list[tuple[int, int]],
) -> list[HeadingCandidate]:
    """Drop heading candidates that fall inside a detected table region."""
    if not table_regions:
        return candidates
    table_lines: set[int] = set()
    for start, end in table_regions:
        table_lines.update(range(start, end + 1))
    return [c for c in candidates if c.line_no not in table_lines]


# ─── Structure profile ───────────────────────────────────────────────────────

def _num_type_of(c: HeadingCandidate) -> str:
    spec = _SPEC_BY_ID.get(c.pattern_id)
    return spec.num_type if spec else c.marker.get("num_type", "")


def analyze_structure_profile(candidates: list[HeadingCandidate]) -> dict:
    pattern_counts: Counter = Counter(c.pattern_id for c in candidates)
    decimal_levels: Counter = Counter()

    for c in candidates:
        if is_decimal_chain_spec(c.pattern_id):
            depth = c.num.count(".") + c.num.count("．") + 1
            decimal_levels[depth] += 1

    significant = [d for d, cnt in decimal_levels.items() if cnt >= MIN_DEPTH_SUPPORT]
    max_decimal_depth = max(significant) if significant else 0

    multi_decimal = sum(cnt for d, cnt in decimal_levels.items() if d >= 2)
    has_decimal = multi_decimal >= MIN_MULTI_DECIMAL
    has_cjk = any(_num_type_of(c) == "cjk" for c in candidates)

    # Every present pattern is an anchor; order them outer→inner by evidence.
    anchor_order, source, confidence, scores = infer_anchor_order_by_evidence(
        candidates, default_anchor_order())
    support_count = sum(scores.values())

    return {
        "pattern_counts": dict(pattern_counts),
        "decimal_levels": decimal_levels,
        "max_decimal_depth": max_decimal_depth,
        "has_decimal": has_decimal,
        "has_cjk": has_cjk,
        "anchor_order": anchor_order,
        "anchor_order_source": source,
        "anchor_order_confidence": confidence,
        "anchor_order_support_count": support_count,
        "parent_child_scores": scores,
    }


# ─── Level inference ─────────────────────────────────────────────────────────

def _present_patterns(candidates: list[HeadingCandidate]) -> list[str]:
    """Distinct pattern_ids in first-appearance order."""
    present: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c.pattern_id not in seen:
            seen.add(c.pattern_id)
            present.append(c.pattern_id)
    return present


def _evidence_scores(candidates: list[HeadingCandidate], present: list[str]) -> dict:
    """Directed outer→inner evidence between patterns, scored per transition.

    For each adjacent pair of differing patterns A→B in document order, the
    direction is decided by whether B *continues its own counter*:

      • B's value increased vs B's previous occurrence  → B is resuming an
        ongoing (outer) level → evidence (B is outer than A): score[(B, A)] += 1.
      • B is fresh (first seen, or its counter restarted) → we are descending
        into a child → evidence (A is outer than B): score[(A, B)] += 1.

    This distinguishes "parent resumes after a child run" (e.g. …2. （二）…,
    where （二） continues the （）counter) from "parent introduces a child"
    (e.g. （一） 1.) — the two look identical to a plain adjacency count and were
    the source of inverted decimal/cjk_paren ordering.
    """
    scores: Counter = Counter()
    last_value: dict[str, tuple] = {}
    prev: HeadingCandidate | None = None

    for c in candidates:
        cur_value = _marker_value(c)
        if prev is not None and prev.pattern_id != c.pattern_id:
            prev_b = last_value.get(c.pattern_id)
            if prev_b is not None and cur_value > prev_b:
                scores[(c.pattern_id, prev.pattern_id)] += 1   # B continuing → B outer
            else:
                scores[(prev.pattern_id, c.pattern_id)] += 1   # B fresh → A outer
        last_value[c.pattern_id] = cur_value
        prev = c
    return dict(scores)


def _parent_child_evidence(
    candidates: list[HeadingCandidate], parent_id: str, child_id: str,
) -> int:
    """Combined follower+bracket evidence that parent_id is outer to child_id."""
    return _evidence_scores(candidates, _present_patterns(candidates)).get(
        (parent_id, child_id), 0)


def _marker_value(c: HeadingCandidate) -> tuple:
    """Comparable numeric value of a marker, for detecting counter resets."""
    nt = _num_type_of(c)
    if nt in ("arabic", "arabic_chain"):
        parts = _parse_decimal_parts(c.num)
        return tuple(parts) if parts else (0,)
    if nt == "cjk":
        return (_cjk_to_int(c.num),)
    if nt == "latin_upper" and c.num:
        return (ord(c.num[:1].upper()) - ord("A") + 1,)
    if nt == "latin_lower" and c.num:
        return (ord(c.num[:1].lower()) - ord("a") + 1,)
    return (0,)


def _order_confidence(order: list[str], scores: dict) -> float:
    """Fraction of evidence consistent with the chosen order (0..1)."""
    pos = {p: i for i, p in enumerate(order)}
    support = contradict = 0
    for (a, b), n in scores.items():
        if a in pos and b in pos:
            if pos[a] < pos[b]:
                support += n
            else:
                contradict += n
    total = support + contradict
    return round(support / total, 4) if total else 0.0


def _topological_order(
    present: list[str], scores: dict, tie_key,
) -> list[str]:
    """Order patterns outer→inner by a topological sort of net parent→child edges.

    An edge A→B (A is outer) exists when net evidence net(A,B) = score(A,B) −
    score(B,A) > 0. Kahn's algorithm peels sources (no remaining parent),
    breaking ties with `tie_key`. Any cycle leftover is appended by tie_key so a
    total order is always produced.
    """
    net: dict[tuple[str, str], int] = {}
    for (a, b), n in scores.items():
        net[(a, b)] = net.get((a, b), 0) + n

    indeg = {p: 0 for p in present}
    adj: dict[str, list[str]] = {p: [] for p in present}
    for a in present:
        for b in present:
            if a >= b:
                continue
            w = net.get((a, b), 0) - net.get((b, a), 0)
            if w > 0:
                adj[a].append(b)
                indeg[b] += 1
            elif w < 0:
                adj[b].append(a)
                indeg[a] += 1

    order: list[str] = []
    placed: set[str] = set()
    while len(order) < len(present):
        ready = [p for p in present if p not in placed and indeg[p] == 0]
        if not ready:  # cycle — release the best remaining node by tie_key
            ready = [p for p in present if p not in placed]
        node = sorted(ready, key=tie_key)[0]
        order.append(node)
        placed.add(node)
        for b in adj[node]:
            indeg[b] -= 1
    return order


def infer_anchor_order_by_evidence(
    candidates: list[HeadingCandidate], default_order: list[str],
) -> tuple[list[str], str, float, dict]:
    """Order ALL anchor patterns outer→inner from directional parent/child evidence.

    Evidence direction per transition is decided by whether a pattern continues
    its own counter (see _evidence_scores), then a topological sort turns the
    pairwise net evidence into a total order. first_seen / registry priority are
    only tie-breaks, so a stray inner marker appearing first cannot mislead it.

    Returns (anchor_order, source, confidence, scores_str). Patterns are NOT
    split by role here — every present pattern is ranked together.

    Returns (order, source, confidence, scores_str).
    """
    present = _present_patterns(candidates)
    first_seen: dict[str, int] = {}
    for i, c in enumerate(candidates):
        first_seen.setdefault(c.pattern_id, i)

    scores = _evidence_scores(candidates, present)
    scores_str = {f"{a}->{b}": n for (a, b), n in scores.items()}

    if len(present) <= 1:
        source = "single_style" if present else "none"
        return list(present), source, (1.0 if present else 0.0), scores_str

    def tie_key(p: str) -> tuple[int, int]:
        idx = default_order.index(p) if p in default_order else len(default_order)
        return (idx, first_seen[p])

    order = _topological_order(present, scores, tie_key)

    if sum(scores.values()) == 0:
        return order, "default_order", 0.0, scores_str
    return order, "parent_child_evidence", _order_confidence(order, scores), scores_str


def infer_levels(candidates: list[HeadingCandidate], profile: dict) -> list[HeadingCandidate]:
    """Assign each anchor a `level` from its position in `anchor_order`.

    `level` is purely the anchor's depth in *this* document — it does NOT mean
    the anchor is a formal section. A decimal_chain pattern stretches by its
    dotted depth; patterns ranked after it are pushed below its deepest level so
    they keep nesting under it.
    """
    max_decimal_depth: int = profile["max_decimal_depth"]

    anchor_order = list(profile.get("anchor_order") or [])
    if not anchor_order:
        anchor_order, *_ = infer_anchor_order_by_evidence(candidates, default_anchor_order())

    base_levels = {pid: idx + 1 for idx, pid in enumerate(anchor_order)}

    # A decimal_chain pattern occupies extra levels (its dotted depth); shift any
    # pattern ranked after it down so it stays below the deepest decimal level.
    chain_idx = next(
        (i for i, pid in enumerate(anchor_order) if is_decimal_chain_spec(pid)), None)
    extra = max(max_decimal_depth, 1) - 1
    if chain_idx is not None and extra:
        for pid in anchor_order[chain_idx + 1:]:
            base_levels[pid] += extra

    fallback_level = len(anchor_order) + extra + 1

    def _level(c: HeadingCandidate) -> int:
        pid = c.pattern_id
        if pid not in base_levels:
            return fallback_level
        base = base_levels[pid]
        if is_decimal_chain_spec(pid):
            depth = c.num.count(".") + c.num.count("．") + 1
            return base + depth - 1
        return base

    return [
        HeadingCandidate(
            page=c.page, line_no=c.line_no,
            char_start=c.char_start, char_end=c.char_end,
            pattern_id=c.pattern_id,
            marker=c.marker, num=c.num, title=c.title, raw=c.raw,
            level=_level(c),
        )
        for c in candidates
    ]


# ─── Sequence validation ─────────────────────────────────────────────────────

def _parse_decimal_parts(num: str) -> list[int]:
    parts = re.split(r"[.．]", num)
    try:
        return [int(p) for p in parts if p]
    except ValueError:
        return []


_CJK_DIGITS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "百": 100,
}


def _cjk_to_int(num: str) -> int:
    """Convert CJK numeral string to int (best-effort for common cases)."""
    # Handle 十X forms like 十一, 十二
    if num.startswith("十"):
        tail = num[1:]
        return 10 + (_CJK_DIGITS.get(tail, 0) if tail else 0)
    return sum(_CJK_DIGITS.get(ch, 0) for ch in num) or 1


def validate_sequence(candidates: list[HeadingCandidate]) -> list[HeadingCandidate]:
    """Remove candidates that fail numeric-sequence continuity checks."""
    accepted: list[HeadingCandidate] = []

    # For decimal: track last accepted number at each prefix tuple
    last_at_prefix: dict[tuple, int] = {}
    accepted_prefixes: set[tuple] = set()
    # Level of the depth-1 decimals currently being tracked. A shallower
    # (parent-level) non-decimal anchor later resets the counter — see below.
    decimal_base_level: int | None = None

    for c in candidates:
        if is_decimal_chain_spec(c.pattern_id):
            parts = _parse_decimal_parts(c.num)
            if not parts:
                continue

            depth = len(parts)
            if depth == 1:
                decimal_base_level = c.level
                n = parts[0]
                last = last_at_prefix.get((), 0)
                # Accept if n==1, or within MAX_JUMP of last accepted (or first occurrence ≤ MAX_JUMP+1)
                if n == 1 or (last > 0 and 1 <= n - last <= MAX_JUMP) or (last == 0 and n <= MAX_JUMP + 1):
                    last_at_prefix[()] = n
                    accepted_prefixes.add((n,))
                    accepted.append(c)
                # else: skip (e.g. 24410, 1904, 50)
            else:
                parent = tuple(parts[:-1])
                child_n = parts[-1]
                if parent in accepted_prefixes:
                    last = last_at_prefix.get(parent, 0)
                    if child_n == 1 or (last > 0 and 1 <= child_n - last <= MAX_JUMP):
                        last_at_prefix[parent] = child_n
                        accepted_prefixes.add(tuple(parts))
                        accepted.append(c)
                # else: parent not accepted → reject
        else:
            # A non-decimal anchor shallower than the tracked decimals starts a new
            # structural context (a different parent). Reset the counter so it does
            # NOT leak across the boundary — otherwise a missing first item ("1.")
            # makes the next list's 2./3. fail continuity (they sit below the prior
            # section's last number). Inner anchors (e.g. paren_num below a decimal)
            # are deeper, so they never trigger a reset.
            if decimal_base_level is not None and c.level < decimal_base_level:
                last_at_prefix = {}
                accepted_prefixes = set()
                decimal_base_level = None
            # CJK, paren_* styles are unambiguous enough; accept unconditionally
            accepted.append(c)

    return accepted


# ─── Heading tree ────────────────────────────────────────────────────────────

def build_heading_tree(candidates: list[HeadingCandidate]) -> dict:
    root: dict = {"level": 0, "title": "ROOT", "num": "", "pattern_id": "", "children": []}
    stack = [root]
    for h in candidates:
        level = h.level
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        node = {
            "level": level, "pattern_id": h.pattern_id,
            "num": h.num, "title": h.title,
            "page": h.page, "line_no": h.line_no,
            "char_start": h.char_start, "raw": h.raw,
            "children": [],
        }
        stack[-1]["children"].append(node)
        stack.append(node)
    return root


# ─── Metadata helpers ────────────────────────────────────────────────────────

def _clean_title(title: str) -> str:
    """Trim trailing label/sentence punctuation so display titles read cleanly.

    A heading like "1. 開設時機：" or list item "1. 協助督導。" carries the
    trailing ：。！？ as a separator/terminator, not as part of the title. The raw
    line keeps it; the title and breadcrumbs don't.
    """
    return (title or "").strip().rstrip("：:。！？").strip()


def section_key(pattern_id: str, num: str) -> str:
    """Stable identity for a heading node: `{pattern_id}:{num}` (e.g. `cjk_paren:二`).

    Used to match nodes when merging trees. Unlike section_id (the bare number),
    it never collides across patterns — `一、` and `（一）` both have num `一` but
    distinct keys `cjk_comma:一` / `cjk_paren:一`.
    """
    return f"{pattern_id}:{num}"


def _make_node(
    level: int, pattern_id: str, num: str, title: str, raw: str, marker: dict,
) -> dict:
    return {
        "level": level,
        "section_key": section_key(pattern_id, num),
        "section_id": num,
        "pattern_id": pattern_id,
        "title": _clean_title(title),
        "raw": (raw or "").strip(),
        "marker": marker,
        "items": [],
    }


def _format_section_label(node: dict) -> str:
    """Render a heading node as `<marker> <title>` for breadcrumbs.

    CJK markers (三、 / （二）) butt directly against the title; arabic/latin
    markers (1. / (1)) take a separating space — matching how the source reads.
    """
    marker = node.get("marker") or {}
    text = marker.get("text") or node.get("section_id") or ""
    title = _clean_title(node.get("title") or "")
    if not text:
        return title
    if not title:
        return text
    sep = "" if marker.get("num_type") == "cjk" else " "
    return f"{text}{sep}{title}"


def _contained_sections_from_tree(tree: dict) -> list[str]:
    """Flatten a heading_tree into leaf-first breadcrumb strings for display.

    Each entry is the full path from the root section to a leaf, e.g.
    "三、作業程序 > （二）一級開設 > 1. 開設時機". A node with no children is
    itself a leaf, so a chunk holding only a parent yields a single breadcrumb.
    """
    if not tree:
        return []
    result: list[str] = []

    def _walk(node: dict, prefix: list[str]) -> None:
        path = prefix + [_format_section_label(node)]
        children = node.get("items") or []
        if not children:
            result.append(" > ".join(p for p in path if p))
        else:
            for child in children:
                _walk(child, path)

    _walk(tree, [])
    return result


def _make_anchor_path(stack: list[HeadingCandidate]) -> list[dict]:
    return [
        {"level": h.level, "pattern_id": h.pattern_id, "num": h.num,
         "title": h.title, "raw": h.raw, "marker": h.marker}
        for h in stack
    ]


def _build_structure_tree_meta(
    anchor_path: list[dict],
    contained: list[HeadingCandidate],
) -> dict:
    """Build the recursive structure tree stored in chunk metadata.

    The ancestor chain (anchor_path) supplies shared parent context; `contained`
    are the anchors physically inside the chunk. Nodes are nested by level into
    each node's ordered `items` list (preserving document order). EVERY anchor
    becomes a node — there is no structural/enum gate; the tree is the detected
    structure, not a guaranteed table of contents.

    Dedup is *per-parent* (path-aware): an anchor is only collapsed into an
    existing node when a sibling with the same section_key already sits under the
    same parent. A global `(level, section_key)` dedup would wrongly merge list
    items that restart their numbering under different parents — e.g. （1）（2）（3）
    recurring beneath each of 1./2./3. — dropping every repeat after the first.
    """
    entries: list[tuple[int, str, str, str, str, dict]] = [
        (h["level"], h["pattern_id"], h["num"], h["title"], h.get("raw", ""), h["marker"])
        for h in anchor_path
    ] + [
        (c.level, c.pattern_id, c.num, c.title, c.raw, c.marker)
        for c in contained
    ]

    root: dict = {}
    stack: list[dict] = []
    for level, pattern_id, num, title, raw, marker in entries:
        key = section_key(pattern_id, num)
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        parent = stack[-1] if stack else None

        if parent is not None:
            existing = _find_child(parent["items"], key)
        elif root and root.get("section_key") == key:
            existing = root
        else:
            existing = None

        if existing is not None:
            stack.append(existing)
            continue

        node = _make_node(level, pattern_id, num, title, raw, marker)
        if parent is not None:
            parent["items"].append(node)
        elif not root:
            root = node
        else:
            # Defensive: chunk spanning two top-level sections — attach so no
            # heading is lost (merge gating normally prevents this).
            root["items"].append(node)
        stack.append(node)
    return root


def _doc_title_from_source(source_file: str) -> str:
    if not source_file:
        return ""
    name = source_file.replace("\\", "/").rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


# ─── Split by candidate lines ────────────────────────────────────────────────

def split_by_candidate_lines(
    line_records: list[LineRecord],
    candidates: list[HeadingCandidate],
    toc_line_nos: set[int],
    chunk_size: int,
    overlap_size: int,
    source_file: str = "",
    *,
    base_warnings: list[str] | None = None,
) -> list[Document]:
    """Delayed anchor-recursive split.

    A range stays whole while it (a) fits in chunk_size AND (b) spans no more
    than MAX_ANCHORS_PER_CHUNK *top-level* anchors. Only when a range is too big
    OR spans too many top-level anchors do we split at its outermost anchor level
    and recurse. RecursiveCharacterTextSplitter is the last resort (no anchors
    left but still too big).
    """
    fallback = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=overlap_size,
    )
    n = len(line_records)
    base_warnings = list(base_warnings or [])

    def _get_text(start: int, end: int) -> str:
        return "\n".join(
            line_records[i].text
            for i in range(start, end + 1)
            if i not in toc_line_nos
        ).strip()

    def _get_pages(start: int, end: int) -> list[int]:
        return [
            line_records[i].page
            for i in range(start, end + 1)
            if i not in toc_line_nos and line_records[i].text.strip()
        ]

    def _make_doc(
        text: str,
        pages: list[int],
        stack: list[HeadingCandidate],
        contained: list[HeadingCandidate],
        *,
        extra_warnings: list[str] | None = None,
    ) -> Document:
        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else 1
        anchor_path = _make_anchor_path(stack)
        warnings = base_warnings + list(extra_warnings or [])
        meta = _build_metadata(
            source_file, page_start, page_end, pages, anchor_path, contained,
            warnings=warnings,
        )
        return Document(page_content=text, metadata=meta)

    def _is_only_current_parent_heading(start: int, end: int, stack: list[HeadingCandidate]) -> bool:
        if not stack:
            return False
        lines = [
            line_records[i].text.strip()
            for i in range(start, end + 1)
            if i not in toc_line_nos and line_records[i].text.strip()
        ]
        return len(lines) == 1 and lines[0] == stack[-1].raw.strip()

    def _fallback_docs(start: int, end: int, stack, contained, text: str) -> list[Document]:
        pages = _get_pages(start, end)
        base_meta = _make_doc(
            "", pages, stack, contained, extra_warnings=["fallback_splitter_used"],
        ).metadata
        return [
            Document(page_content=sub, metadata=deepcopy(base_meta))
            for sub in fallback.split_text(text)
            if sub.strip()
        ]

    def _split_range(
        start: int,
        end: int,
        cands: list[HeadingCandidate],  # sorted by line_no, all within [start, end]
        stack: list[HeadingCandidate],
    ) -> list[Document]:
        text = _get_text(start, end)
        if not text:
            return []

        tokens = _count_tokens(text)
        fits = tokens <= chunk_size
        min_level = min((c.level for c in cands), default=0)
        top_count = sum(1 for c in cands if c.level == min_level) if cands else 0

        # Delayed-split stop condition. The MAX_ANCHORS_PER_CHUNK cap is token-aware:
        # it only forces a split once the range is reasonably large. A range that
        # fits AND is below ANCHOR_SPLIT_MIN_RATIO·chunk_size is kept whole even when
        # it spans many top anchors (e.g. a 7-item list that is still small), so a
        # coherent list isn't shattered into fragments that repair just merges back.
        anchor_cap_applies = tokens > chunk_size * ANCHOR_SPLIT_MIN_RATIO
        if fits and (top_count <= MAX_ANCHORS_PER_CHUNK or not anchor_cap_applies):
            return [_make_doc(text, _get_pages(start, end), stack, cands)]

        # No anchors left to split on → character fallback (only reached when big).
        if not cands:
            return _fallback_docs(start, end, stack, [], text)

        # Split at the outermost anchor level present in this range.
        top_cands = [c for c in cands if c.level == min_level]

        segments: list[tuple[int, int, HeadingCandidate | None]] = []
        prev = start
        for i, cand in enumerate(top_cands):
            segment_start = cand.line_no
            if cand.line_no > prev:
                if _is_only_current_parent_heading(prev, cand.line_no - 1, stack):
                    segment_start = prev
                else:
                    segments.append((prev, cand.line_no - 1, None))
            next_start = top_cands[i + 1].line_no if i + 1 < len(top_cands) else end + 1
            segments.append((segment_start, min(next_start - 1, end), cand))
            prev = next_start
        if prev <= end:
            segments.append((prev, end, None))

        docs: list[Document] = []
        for seg_s, seg_e, cand in segments:
            if seg_s > seg_e:
                continue
            sub_cands = [c for c in cands if c.level > min_level and seg_s <= c.line_no <= seg_e]
            new_stack = list(stack)
            if cand is not None:
                while new_stack and new_stack[-1].level >= cand.level:
                    new_stack.pop()
                new_stack.append(cand)
            docs.extend(_split_range(seg_s, seg_e, sub_cands, new_stack))
        return docs

    body_cands = sorted(candidates, key=lambda c: c.line_no)

    if not body_cands:
        text = _get_text(0, n - 1)
        if not text:
            return []
        if _count_tokens(text) <= chunk_size:
            return [_make_doc(text, _get_pages(0, n - 1), [], [])]
        return _fallback_docs(0, n - 1, [], [], text)

    return _split_range(0, n - 1, body_cands, [])


def _build_metadata(
    source_file: str,
    page_start: int,
    page_end: int,
    pages: list[int],
    anchor_path: list[dict],
    contained: list[HeadingCandidate] | None = None,
    *,
    warnings: list[str] | None = None,
) -> dict:
    # Lean public metadata. Diagnostics that were once emitted (heading_tree alias,
    # structure_type, split_strategy, structure/anchor_order confidence + support,
    # merge_applied/merge_reason) are intentionally NOT exposed — nothing downstream
    # reads them, and `warnings` already records merge/fallback events. structure_tree
    # is kept (the detected anchor structure of THIS chunk — not a guaranteed ToC).
    structure_tree = _build_structure_tree_meta(anchor_path, contained or [])
    return {
        "source_file": source_file,
        "doc_title": _doc_title_from_source(source_file),
        "page_start": page_start,
        "page_end": page_end,
        "contained_sections": _contained_sections_from_tree(structure_tree),
        "structure_tree": structure_tree,
        "warnings": list(warnings or []),
    }


# ─── Source display ──────────────────────────────────────────────────────────

def build_source_ref(meta: dict) -> str:
    """Compact source reference string, e.g. `要點.pdf#p3-p4#chunk8`.

    Derived on demand from chunk-level fields — never stored in metadata.
    """
    src = meta.get("source_file") or ""
    page_start = meta.get("page_start")
    page_end = meta.get("page_end")
    chunk_index = meta.get("chunk_index")
    if page_start == page_end or page_end is None:
        page = f"p{page_start}"
    else:
        page = f"p{page_start}-p{page_end}"
    return f"{src}#{page}#chunk{chunk_index}"


def format_chunk_source(meta: dict) -> str:
    """Human-readable source block for RAG answers / summary attribution.

        來源：境外核災處理作業要點.pdf，第 3 頁，chunk 8
        本段包含：
        - 三、作業程序 > （二）一級開設 > 1. 開設時機

    The structure-path list is omitted when no anchors were detected.
    """
    src = meta.get("source_file") or ""
    page_start = meta.get("page_start")
    page_end = meta.get("page_end")
    chunk_index = meta.get("chunk_index")
    if page_start == page_end or page_end is None:
        page_label = f"第 {page_start} 頁"
    else:
        page_label = f"第 {page_start}–{page_end} 頁"

    lines = [f"來源：{src}，{page_label}，chunk {chunk_index}"]
    sections = meta.get("contained_sections") or []
    if sections:
        lines.append("本段包含的結構路徑：")
        lines.extend(f"- {s}" for s in sections)
    return "\n".join(lines)


# ─── Repair small chunks ──────────────────────────────────────────────────────

def _get_tree(meta: dict) -> dict:
    return meta.get("structure_tree") or meta.get("heading_tree") or {}


def _should_merge_doc(doc: Document, min_tokens: int = 80) -> bool:
    text = doc.page_content.strip()
    return bool(text) and _count_tokens(text) < min_tokens


# Merge-priority tiers (lower = preferred). See _merge_tier_reason.
_MERGE_REASONS = {
    1: "small_chunk_same_parent",
    2: "small_chunk_same_top_anchor",
    3: "small_chunk_cross_anchor",
}


def _bc_parent(breadcrumb: str) -> str:
    parts = breadcrumb.split(" > ")
    return " > ".join(parts[:-1])


def _bc_top(breadcrumb: str) -> str:
    return breadcrumb.split(" > ")[0]


def _merge_tier_reason(left_meta: dict, right_meta: dict) -> tuple[int, str]:
    """Classify how related two adjacent chunks are (for merge priority).

    1 same parent path → 2 same top-level anchor → 3 cross anchor (last resort).
    A chunk with no detected anchors is freely absorbable (treated as tier 2).
    """
    left = left_meta.get("contained_sections") or []
    right = right_meta.get("contained_sections") or []
    if not left or not right:
        return 2, _MERGE_REASONS[2]
    lp, rp = _bc_parent(left[-1]), _bc_parent(right[0])
    if lp and lp == rp:
        return 1, _MERGE_REASONS[1]
    if _bc_top(left[-1]) == _bc_top(right[0]):
        return 2, _MERGE_REASONS[2]
    return 3, _MERGE_REASONS[3]


def _find_child(items: list[dict], key: str) -> dict | None:
    for child in items:
        if child.get("section_key") == key:
            return child
    return None


def _merge_trees(t1: dict, t2: dict) -> dict:
    """Recursively union two structure trees, matched by section_key.

    Child order is preserved: existing children keep their position, newly seen
    children are appended in source order. Matching on section_key (not the bare
    section_id) avoids collisions like `cjk_comma:一` vs `cjk_paren:一`.
    """
    if not t1:
        return deepcopy(t2) if t2 else {}
    if not t2:
        return deepcopy(t1)
    if t1.get("section_key") != t2.get("section_key"):
        # Different roots — nest the second under the first so nothing is lost.
        merged = deepcopy(t1)
        items = merged.setdefault("items", [])
        if _find_child(items, t2.get("section_key")) is None:
            items.append(deepcopy(t2))
        return merged

    merged = deepcopy(t1)
    target_items = merged.setdefault("items", [])
    for source_child in t2.get("items", []):
        target_child = _find_child(target_items, source_child.get("section_key"))
        if target_child is None:
            target_items.append(deepcopy(source_child))
        else:
            idx = target_items.index(target_child)
            target_items[idx] = _merge_trees(target_child, source_child)
    return merged


def _merge_metadata(m1: dict, m2: dict) -> dict:
    result = dict(m1)
    result["page_start"] = min(m1.get("page_start", 99999), m2.get("page_start", 99999))
    result["page_end"] = max(m1.get("page_end", 0), m2.get("page_end", 0))
    tree = _merge_trees(_get_tree(m1), _get_tree(m2))
    result["structure_tree"] = tree
    result["contained_sections"] = _contained_sections_from_tree(tree)
    warnings = list(m1.get("warnings") or [])
    # "small_chunk_merged" in warnings is the only signal that a merge happened —
    # merge_applied/merge_reason are no longer emitted.
    for w in (m2.get("warnings") or []) + ["small_chunk_merged"]:
        if w not in warnings:
            warnings.append(w)
    result["warnings"] = warnings
    return result


def repair_small_chunks_docs(docs: list[Document], chunk_size: int) -> list[Document]:
    """Merge sub-threshold chunks into the most-related neighbour.

    Priority: same parent path > same top anchor > cross anchor (last resort).
    For each small chunk we score both neighbours and merge into the lowest-tier
    (most related) one that still fits, recording merge_applied / merge_reason.
    """
    # Merge any chunk whose token count is below ~10% of chunk_size (floor 80).
    min_tokens = max(80, chunk_size // 10)
    repaired: list[Document] = []
    i = 0
    while i < len(docs):
        doc = docs[i]
        text = doc.page_content.strip()
        if not text:
            i += 1
            continue

        if _should_merge_doc(doc, min_tokens):
            # Candidate merges: (tier, direction) for whichever still fits. The tier
            # (relatedness) drives the choice; the reason string is no longer stored.
            options = []
            if i + 1 < len(docs):
                merged_text = text + "\n" + docs[i + 1].page_content.strip()
                if _count_tokens(merged_text) <= chunk_size:
                    tier, _ = _merge_tier_reason(doc.metadata, docs[i + 1].metadata)
                    options.append((tier, "right", merged_text))
            if repaired:
                merged_text = repaired[-1].page_content + "\n" + text
                if _count_tokens(merged_text) <= chunk_size:
                    tier, _ = _merge_tier_reason(repaired[-1].metadata, doc.metadata)
                    options.append((tier, "left", merged_text))

            if options:
                options.sort(key=lambda o: (o[0], 0 if o[1] == "right" else 1))
                tier, direction, merged_text = options[0]
                if direction == "right":
                    meta = _merge_metadata(doc.metadata, docs[i + 1].metadata)
                    repaired.append(Document(page_content=merged_text, metadata=meta))
                    i += 2
                else:
                    meta = _merge_metadata(repaired[-1].metadata, doc.metadata)
                    repaired[-1] = Document(page_content=merged_text, metadata=meta)
                    i += 1
                continue

        repaired.append(doc)
        i += 1
    return repaired


# ─── Main entry point ────────────────────────────────────────────────────────

def build_documents_from_pages(
    pages: list[str],
    chunk_size: int,
    overlap_size: int,
    source_file: str = "",
) -> tuple[list[Document], dict]:
    """
    Full pipeline from page text list to LangChain Documents.
    Returns (docs, structure_info).
    structure_info keys: toc_pages_excluded, preview_headings, profile, candidate_count.
    """
    line_records = build_line_records(pages)
    line_records = clean_line_records(line_records)

    all_candidates = detect_heading_candidates(line_records)

    toc_regions = find_toc_regions(line_records)
    toc_line_nos: set[int] = set()
    for start, end in toc_regions:
        toc_line_nos.update(range(start, end + 1))

    candidates = filter_toc_and_noise(all_candidates, toc_regions)

    table_regions = find_table_regions(line_records)
    candidates = filter_table_regions(candidates, table_regions)

    profile = analyze_structure_profile(candidates)
    candidates = infer_levels(candidates, profile)
    candidates = validate_sequence(candidates)

    # Document-level warning: the anchor order rests on little evidence.
    base_warnings: list[str] = []
    if (profile.get("anchor_order_support_count", 0) < MIN_ANCHOR_SUPPORT
            and len(profile.get("anchor_order") or []) > 1):
        base_warnings.append("low_anchor_order_support")

    docs = split_by_candidate_lines(
        line_records, candidates, toc_line_nos,
        chunk_size, overlap_size, source_file,
        base_warnings=base_warnings,
    )

    for _ in range(3):
        repaired = repair_small_chunks_docs(docs, chunk_size)
        if len(repaired) == len(docs):
            break
        docs = repaired

    # Assign sequential chunk index after all merging is done
    for idx, doc in enumerate(docs):
        doc.metadata["chunk_index"] = idx

    toc_pages = sorted({line_records[i].page for i in toc_line_nos if i < len(line_records)})
    preview = [f"page={c.page} level={c.level} {c.raw}" for c in candidates[:20]]

    structure_info = {
        "toc_pages_excluded": toc_pages,
        "preview_headings": preview,
        "profile": profile,
        "candidate_count": len(candidates),
        # Promote anchor-order diagnostics to the top level for convenience.
        "anchor_order": profile.get("anchor_order", []),
        "anchor_order_source": profile.get("anchor_order_source"),
        "anchor_order_confidence": profile.get("anchor_order_confidence"),
        "anchor_order_support_count": profile.get("anchor_order_support_count"),
        "parent_child_scores": profile.get("parent_child_scores", {}),
    }

    return docs, structure_info
