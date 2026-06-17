"""
doc_structure.py — Document structure detection and chunk splitting.

Pipeline:
  build_line_records → clean_line_records → detect_heading_candidates
  → find_toc_regions → filter_toc_and_noise → analyze_structure_profile
  → infer_levels → validate_sequence → build_heading_tree
  → split_by_candidate_lines → repair_small_chunks_docs
"""

import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None

# ─── Heading patterns ────────────────────────────────────────────────────────

HEADING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 1 / 1.1 / 1.1.1 / 10.3.5；分隔符容許 . ． 、
    ("decimal", re.compile(
        r"^(?P<num>\d+(?:[.．]\d+)*)[.．、]?\s*(?P<title>[一-鿿A-Za-z].+)$"
    )),
    # (1) (1). （1）
    ("paren_num", re.compile(
        r"^[\(（](?P<num>\d+)[\)）][.．]?\s*(?P<title>.+)$"
    )),
    # (A) (A). （A）
    ("paren_upper", re.compile(
        r"^[\(（](?P<num>[A-Z])[\)）][.．]?\s*(?P<title>.+)$"
    )),
    # (a) （a）
    ("paren_lower", re.compile(
        r"^[\(（](?P<num>[a-z])[\)）][.．]?\s*(?P<title>.+)$"
    )),
    # 一、目的
    ("cjk_comma", re.compile(
        r"^(?P<num>[一二三四五六七八九十百]+)、\s*(?P<title>.+)$"
    )),
    # （一）目的
    ("cjk_paren", re.compile(
        r"^[（(](?P<num>[一二三四五六七八九十百]+)[）)]\s*(?P<title>.+)$"
    )),
]

# bare number: 序號與標題被 PDF 拆成兩行
BARE_NUMBER_RE = re.compile(r"^(?P<num>\d+(?:[.．]\d+)*)[.．、]$")

PAGE_NUMBER_RE = re.compile(r"第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁")
SENTENCE_END_RE = re.compile(r"[。！？；;：:]$")
FIGURE_TABLE_RE = re.compile(r"^[圖表]\s*\d+([-.–]\d+)?\s*[：:]")
TOC_TITLE_RE = re.compile(
    r"^(目\s*錄|頁\s*次|附\s*圖\s*目\s*錄|附\s*表\s*目\s*錄|圖\s*目\s*錄|表\s*目\s*錄)$"
)

MIN_MULTI_DECIMAL = 3   # multi-level decimals needed to call has_decimal=True
MIN_DEPTH_SUPPORT = 2   # min occurrences for a depth to count toward max_decimal_depth
MAX_JUMP = 5            # allow this many skipped numbers before rejecting
DEFAULT_STYLE_ORDER = [
    "cjk_comma",
    "cjk_paren",
    "decimal",
    "paren_num",
    "paren_upper",
    "paren_lower",
]


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class LineRecord:
    page: int
    text: str
    char_start: int = 0
    char_end: int = 0


@dataclass
class HeadingCandidate:
    page: int
    line_no: int
    char_start: int
    char_end: int
    style: str
    num: str
    title: str
    raw: str
    level: int = 0


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
    return any(pat.match(stripped) for _, pat in HEADING_PATTERNS)


def is_standalone_section_heading(line: str) -> bool:
    stripped = line.strip()
    for style, pat in HEADING_PATTERNS:
        if style in ("cjk_comma", "cjk_paren") and pat.match(stripped):
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
                for style, pat in HEADING_PATTERNS:
                    m = pat.match(merged)
                    if m:
                        candidates.append(HeadingCandidate(
                            page=rec.page, line_no=i,
                            char_start=rec.char_start, char_end=nxt.char_end,
                            style=style, num=m.group("num"),
                            title=m.group("title"), raw=merged,
                        ))
                        break
                continue

        for style, pat in HEADING_PATTERNS:
            m = pat.match(line)
            if m:
                candidates.append(HeadingCandidate(
                    page=rec.page, line_no=i,
                    char_start=rec.char_start, char_end=rec.char_end,
                    style=style, num=m.group("num"),
                    title=m.group("title"), raw=line,
                ))
                break

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

    # 3. TOC lines – contain 3+ consecutive dots OR end with a page number
    if re.search(r"[.．…]{3,}", line):
        return True

    # 4. Figure / table captions
    if FIGURE_TABLE_RE.match(line):
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
        # leader, and (c) for decimal style the leading number must be ≤ MAX_JUMP+1
        # (to exclude concatenated TOC codes like 24410).
        is_real_body = False
        for style, pat in HEADING_PATTERNS:
            m = pat.match(text)
            if not m or re.search(r"[.．…]{3,}", text):
                continue
            if style == "decimal":
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
        for _, pat in HEADING_PATTERNS:
            if pat.match(text) and not re.search(r"[.．…]{3,}", text):
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


# ─── Structure profile ───────────────────────────────────────────────────────

def analyze_structure_profile(candidates: list[HeadingCandidate]) -> dict:
    style_counts: Counter = Counter(c.style for c in candidates)
    decimal_levels: Counter = Counter()

    for c in candidates:
        if c.style == "decimal":
            depth = c.num.count(".") + c.num.count("．") + 1
            decimal_levels[depth] += 1

    significant = [d for d, cnt in decimal_levels.items() if cnt >= MIN_DEPTH_SUPPORT]
    max_decimal_depth = max(significant) if significant else 0

    multi_decimal = sum(cnt for d, cnt in decimal_levels.items() if d >= 2)
    has_decimal = multi_decimal >= MIN_MULTI_DECIMAL
    cjk_order, cjk_order_source, cjk_parent_scores = _infer_cjk_style_order(candidates)
    style_order, style_order_source, style_parent_scores = _infer_style_order(candidates)

    return {
        "style_counts": style_counts,
        "decimal_levels": decimal_levels,
        "max_decimal_depth": max_decimal_depth,
        "has_decimal": has_decimal,
        "has_cjk": style_counts.get("cjk_comma", 0) > 0 or style_counts.get("cjk_paren", 0) > 0,
        "style_order": style_order,
        "style_order_source": style_order_source,
        "style_parent_scores": style_parent_scores,
        "cjk_style_order": cjk_order,
        "cjk_style_order_source": cjk_order_source,
        "cjk_parent_scores": cjk_parent_scores,
    }


# ─── Level inference ─────────────────────────────────────────────────────────

def _parent_child_evidence(
    candidates: list[HeadingCandidate],
    parent_style: str,
    child_style: str,
) -> int:
    """Count direct child-style evidence inside consecutive parent-style headings."""
    parent_indexes = [
        i for i, c in enumerate(candidates)
        if c.style == parent_style
    ]
    if not parent_indexes:
        return 0

    score = 0
    for idx, start in enumerate(parent_indexes):
        end = parent_indexes[idx + 1] if idx + 1 < len(parent_indexes) else len(candidates)
        direct_child_style = None
        for c in candidates[start + 1:end]:
            if c.style != parent_style:
                direct_child_style = c.style
                break
        if direct_child_style == child_style:
            score += 1
    return score


def _infer_cjk_style_order(candidates: list[HeadingCandidate]) -> tuple[list[str], str, dict[str, int]]:
    """Infer whether cjk_comma or cjk_paren is the outer CJK section style."""
    cjk_parent_scores = {
        "cjk_comma_parent": 0,
        "cjk_paren_parent": 0,
    }
    present = {
        c.style for c in candidates
        if c.style in {"cjk_comma", "cjk_paren"}
    }
    if not present:
        return [], "none", cjk_parent_scores
    if present == {"cjk_comma"}:
        return ["cjk_comma"], "single_style", cjk_parent_scores
    if present == {"cjk_paren"}:
        return ["cjk_paren"], "single_style", cjk_parent_scores

    comma_parent_score = _parent_child_evidence(candidates, "cjk_comma", "cjk_paren")
    paren_parent_score = _parent_child_evidence(candidates, "cjk_paren", "cjk_comma")
    cjk_parent_scores = {
        "cjk_comma_parent": comma_parent_score,
        "cjk_paren_parent": paren_parent_score,
    }

    if paren_parent_score > comma_parent_score:
        return ["cjk_paren", "cjk_comma"], "parent_child_evidence", cjk_parent_scores
    if comma_parent_score > paren_parent_score:
        return ["cjk_comma", "cjk_paren"], "parent_child_evidence", cjk_parent_scores
    return ["cjk_comma", "cjk_paren"], "default_tie_break", cjk_parent_scores


def _infer_style_order(candidates: list[HeadingCandidate]) -> tuple[list[str], str, dict[str, int]]:
    """Infer heading-style hierarchy from pairwise parent/child evidence."""
    first_seen: dict[str, int] = {}
    for idx, c in enumerate(candidates):
        if c.style in DEFAULT_STYLE_ORDER and c.style not in first_seen:
            first_seen[c.style] = idx
    present = sorted(first_seen, key=lambda style: first_seen[style])
    parent_scores = {style: 0 for style in present}
    if not present:
        return [], "none", parent_scores
    if len(present) == 1:
        return present, "single_style", parent_scores

    decisive = False
    for parent in present:
        for child in present:
            if parent == child:
                continue
            score = _parent_child_evidence(candidates, parent, child)
            parent_scores[parent] += score
            if score > 0 and first_seen[parent] < first_seen[child]:
                decisive = True

    ordered = present
    source = "parent_child_evidence" if decisive else "default_tie_break"
    return ordered, source, parent_scores


def infer_levels(candidates: list[HeadingCandidate], profile: dict) -> list[HeadingCandidate]:
    has_decimal: bool = profile["has_decimal"]
    max_decimal_depth: int = profile["max_decimal_depth"]

    if has_decimal:
        deepest = max(max_decimal_depth, 1)

        def _level(c: HeadingCandidate) -> int:
            if c.style == "decimal":
                return c.num.count(".") + c.num.count("．") + 1
            if c.style == "cjk_comma":
                return 1
            if c.style == "cjk_paren":
                return 2
            if c.style == "paren_num":
                return deepest + 1
            if c.style == "paren_upper":
                return deepest + 2
            if c.style == "paren_lower":
                return deepest + 3
            return deepest + 1
    else:
        inferred_order, _, _ = _infer_style_order(candidates)
        style_order = profile.get("style_order") or inferred_order
        style_levels = {
            style: idx + 1
            for idx, style in enumerate(style_order)
        }
        deepest = len(style_order) + max_decimal_depth

        def _level(c: HeadingCandidate) -> int:  # type: ignore[misc]
            if c.style == "decimal" and c.style in style_levels:
                depth = c.num.count(".") + c.num.count("．") + 1
                return style_levels[c.style] + depth - 1
            if c.style in style_levels:
                return style_levels[c.style]
            if c.style == "decimal":
                depth = c.num.count(".") + c.num.count("．") + 1
                return len(style_order) + depth
            if c.style == "paren_num":
                return deepest + 1
            if c.style == "paren_upper":
                return deepest + 2
            if c.style == "paren_lower":
                return deepest + 3
            return deepest + 1

    return [
        HeadingCandidate(
            page=c.page, line_no=c.line_no,
            char_start=c.char_start, char_end=c.char_end,
            style=c.style, num=c.num, title=c.title, raw=c.raw,
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

    for c in candidates:
        if c.style == "decimal":
            parts = _parse_decimal_parts(c.num)
            if not parts:
                continue

            depth = len(parts)
            if depth == 1:
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
            # CJK, paren_* styles are unambiguous enough; accept unconditionally
            accepted.append(c)

    return accepted


# ─── Heading tree ────────────────────────────────────────────────────────────

def build_heading_tree(candidates: list[HeadingCandidate]) -> dict:
    root: dict = {"level": 0, "title": "ROOT", "num": "", "style": "", "children": []}
    stack = [root]
    for h in candidates:
        level = h.level
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        node = {
            "level": level, "style": h.style,
            "num": h.num, "title": h.title,
            "page": h.page, "line_no": h.line_no,
            "char_start": h.char_start, "raw": h.raw,
            "children": [],
        }
        stack[-1]["children"].append(node)
        stack.append(node)
    return root


# ─── Metadata helpers ────────────────────────────────────────────────────────

# Structural sub-sections (named, worth listing in metadata) vs. enumeration
# items (1. 2. 3. / (1) (2) — too granular, collapsed to a boolean flag).
_SECTION_STYLES = {"cjk_comma", "cjk_paren", "paren_upper", "paren_lower"}
_ITEM_STYLES = {"decimal", "paren_num"}


def _make_heading_path(stack: list[HeadingCandidate]) -> list[dict]:
    return [
        {"level": h.level, "style": h.style, "num": h.num, "title": h.title}
        for h in stack
    ]


def _format_heading(style: str, num: str, title: str) -> str:
    if style in ("paren_num", "paren_upper", "paren_lower"):
        return f"({num}) {title}"
    if style == "cjk_paren":
        return f"（{num}）{title}"
    if style == "cjk_comma":
        return f"{num}、{title}"
    sep = "." if "." not in num and "．" not in num else ""
    return f"{num}{sep} {title}"


def _format_heading_dict(heading: dict) -> str:
    return _format_heading(
        str(heading.get("style", "")),
        str(heading.get("num", "")),
        str(heading.get("title", "")),
    )


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
) -> list[Document]:
    """Top-down hierarchical split.

    Tries to keep each top-level section as one chunk. Only recurses into
    the next heading level when a section exceeds chunk_size. Falls back to
    RecursiveCharacterTextSplitter when no headings remain.
    """
    fallback = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=overlap_size,
    )
    n = len(line_records)

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
    ) -> Document:
        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else 1
        heading_path = _make_heading_path(stack)
        meta = _build_metadata(
            source_file, page_start, page_end, pages, heading_path, contained,
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

    def _split_range(
        start: int,
        end: int,
        cands: list[HeadingCandidate],  # sorted by line_no, all within [start, end]
        stack: list[HeadingCandidate],
    ) -> list[Document]:
        text = _get_text(start, end)
        if not text:
            return []

        # Fits in one chunk — keep it whole regardless of heading boundaries
        if _count_tokens(text) <= chunk_size:
            return [_make_doc(text, _get_pages(start, end), stack, cands)]

        # Too big but no headings left → use fallback text splitter
        if not cands:
            pages = _get_pages(start, end)
            base_meta = _make_doc("", pages, stack, []).metadata
            return [
                Document(page_content=sub, metadata=dict(base_meta))
                for sub in fallback.split_text(text)
                if sub.strip()
            ]

        # Split at the top-most (lowest level number) headings available
        min_level = min(c.level for c in cands)
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
        metadata = _build_metadata(
            source_file=source_file,
            page_start=line_records[0].page if line_records else 1,
            page_end=line_records[-1].page if line_records else 1,
            pages=sorted({r.page for r in line_records}) if line_records else [1],
            heading_path=[],
            contained=[],
        )
        return [
            Document(page_content=c, metadata=dict(metadata))
            for c in fallback.split_text(text)
            if c.strip()
        ]

    return _split_range(0, n - 1, body_cands, [])


def _partition_sections(
    all_sections: list[tuple[int, str]],
) -> tuple[list[str], list[str]]:
    """Split ordered structural headings into parent context and covered sections.

    The "branch level" is the shallowest level that holds 2+ sections — the point
    where the chunk starts spanning siblings. Everything shallower is shared parent
    context; the branch level and deeper are covered sub-sections
    (contains_sections). With no branching it's a nested chain: all but the deepest
    section are context, the deepest is the single covered section.
    """
    if not all_sections:
        return [], []

    level_counts = Counter(level for level, _ in all_sections)
    branch_level = min((lvl for lvl, cnt in level_counts.items() if cnt >= 2), default=None)

    if branch_level is None:
        section_path = [title for _, title in all_sections[:-1]]
        contains_sections = [all_sections[-1][1]]
    else:
        section_path = [title for level, title in all_sections if level < branch_level]
        contains_sections = [title for level, title in all_sections if level >= branch_level]

    # Dedupe while preserving order (defensive — chain and inside shouldn't overlap)
    section_path = list(dict.fromkeys(section_path))
    contains_sections = list(dict.fromkeys(contains_sections))
    return section_path, contains_sections


def _build_metadata(
    source_file: str,
    page_start: int,
    page_end: int,
    pages: list[int],
    heading_path: list[dict],
    contained: list[HeadingCandidate] | None = None,
) -> dict:
    # Only named structural sections feed contains_sections.
    # Enumeration items (1. 2. / (1)) are conveyed solely by contains_subitems.
    contained = contained or []

    structural_chain = [h for h in heading_path if h["style"] in _SECTION_STYLES]

    # All structural sections relevant to this chunk, in document order:
    #   - the ancestor/leaf chain (heading_path) provides parent context
    #   - `contained` are sibling/child sections physically inside the chunk
    chain = [
        (h["level"], _format_heading(h["style"], h["num"], h["title"]))
        for h in structural_chain
    ]
    inside = [
        (c.level, _format_heading(c.style, c.num, c.title))
        for c in contained
        if c.style in _SECTION_STYLES
    ]
    all_sections = chain + inside

    _, contains_sections = _partition_sections(all_sections)
    contains_subitems = any(c.style in _ITEM_STYLES for c in contained) or any(
        h["style"] in _ITEM_STYLES for h in heading_path
    )

    return {
        "source_file": source_file,
        "doc_title": _doc_title_from_source(source_file),
        "page_start": page_start,
        "page_end": page_end,
        "heading_path": heading_path,
        "contains_sections": contains_sections,
        "contains_subitems": contains_subitems,
    }


# ─── Repair small chunks ──────────────────────────────────────────────────────

def _should_merge_doc(doc: Document, min_tokens: int = 80) -> bool:
    text = doc.page_content.strip()
    return bool(text) and _count_tokens(text) < min_tokens


def _top_section(meta: dict) -> str | None:
    for heading in meta.get("heading_path", []):
        if heading.get("style") in _SECTION_STYLES:
            return _format_heading_dict(heading)
    return None


def _can_merge_docs(left: Document, right: Document) -> bool:
    left_top = _top_section(left.metadata)
    right_top = _top_section(right.metadata)
    return not (left_top and right_top and left_top != right_top)


def _common_prefix(a: list, b: list) -> list:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def _merge_metadata(m1: dict, m2: dict) -> dict:
    result = dict(m1)
    result["page_start"] = min(m1.get("page_start", 99999), m2.get("page_start", 99999))
    result["page_end"] = max(m1.get("page_end", 0), m2.get("page_end", 0))

    # Keep the common structured heading path. Any diverging structural heading
    # becomes part of contains_sections.
    hp1: list[dict] = m1.get("heading_path", [])
    hp2: list[dict] = m2.get("heading_path", [])
    common_heading_path = _common_prefix(hp1, hp2)
    extra: list[str] = []
    if len(hp1) > len(common_heading_path) and hp1[len(common_heading_path)].get("style") in _SECTION_STYLES:
        extra.append(_format_heading_dict(hp1[len(common_heading_path)]))
    if len(hp2) > len(common_heading_path) and hp2[len(common_heading_path)].get("style") in _SECTION_STYLES:
        formatted = _format_heading_dict(hp2[len(common_heading_path)])
        if formatted not in extra:
            extra.append(formatted)
    result["heading_path"] = common_heading_path

    # Union contained sections: diverging headings + both chunks' own contains_sections
    merged_sections: list[str] = list(extra)
    for s in m1.get("contains_sections", []) + m2.get("contains_sections", []):
        if s not in merged_sections:
            merged_sections.append(s)
    result["contains_sections"] = merged_sections
    result["contains_subitems"] = (
        m1.get("contains_subitems", False) or m2.get("contains_subitems", False)
    )
    return result


def repair_small_chunks_docs(docs: list[Document], chunk_size: int) -> list[Document]:
    # Target: merge any chunk whose token count is below ~10% of chunk_size.
    # Floor at 80 so small chunk_size values still catch heading-only fragments.
    min_tokens = max(80, chunk_size // 10)
    repaired: list[Document] = []
    i = 0
    while i < len(docs):
        doc = docs[i]
        text = doc.page_content.strip()
        if not text:
            i += 1
            continue

        if _should_merge_doc(doc, min_tokens) and i + 1 < len(docs):
            merged_text = text + "\n" + docs[i + 1].page_content.strip()
            if _can_merge_docs(doc, docs[i + 1]) and _count_tokens(merged_text) <= chunk_size:
                meta = _merge_metadata(doc.metadata, docs[i + 1].metadata)
                repaired.append(Document(page_content=merged_text, metadata=meta))
                i += 2
                continue

        if _should_merge_doc(doc, min_tokens) and repaired:
            merged_text = repaired[-1].page_content + "\n" + text
            if _can_merge_docs(repaired[-1], doc) and _count_tokens(merged_text) <= chunk_size:
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
    profile = analyze_structure_profile(candidates)
    candidates = infer_levels(candidates, profile)
    candidates = validate_sequence(candidates)

    docs = split_by_candidate_lines(
        line_records, candidates, toc_line_nos,
        chunk_size, overlap_size, source_file,
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
    }

    return docs, structure_info
