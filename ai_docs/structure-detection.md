# 文件結構偵測邏輯

本文整理目前 `doc_structure.py` 如何從 PDF / 文字內容偵測章節結構，並產生 chunk metadata。

---

## 入口

主要入口：

```python
build_documents_from_pages(
    pages: list[str],
    chunk_size: int,
    overlap_size: int,
    source_file: str = "",
)
```

PDF 與貼上文字目前都會走這個入口：

- PDF：`pages` 是每頁文字清單。
- 貼上文字：會被包成單頁 `pages=[transcript]`。

回傳：

```python
docs, structure_info
```

其中 `docs` 是 LangChain `Document` list；`structure_info` 主要用於 debug，例如被排除的目錄頁、前幾個 heading preview、結構 profile。

---

## 整體流程

目前流程如下：

```text
pages
↓
build_line_records()
↓
clean_line_records()
↓
detect_heading_candidates()
↓
find_toc_regions()
↓
filter_toc_and_noise()
↓
analyze_structure_profile()
↓
infer_levels()
↓
validate_sequence()
↓
split_by_candidate_lines()
↓
repair_small_chunks_docs()
↓
補 chunk_index
↓
Document list
```

---

## 1. 建立行資料

位置：`build_line_records()`

系統先把每頁文字拆成一行一筆 `LineRecord`：

```python
@dataclass
class LineRecord:
    page: int
    text: str
    char_start: int = 0
    char_end: int = 0
```

每行會保留：

- `page`：來自 PDF 第幾頁。
- `text`：該行文字。
- `char_start` / `char_end`：在合併後文件中的大致字元位置。

目前 chunk 頁碼不是用「每頁獨立切分」決定，而是由 chunk 內包含的 `LineRecord.page` 推得：

```python
page_start = min(pages)
page_end = max(pages)
```

---

## 2. 清理與硬換行還原

位置：`clean_line_records()`

清理分四步：

1. 移除 PDF 頁碼格式：

```text
第1頁/共7頁
第 2 頁 / 共 7 頁
```

2. 去除每行前後空白。
3. 合併連續空白行。
4. 嘗試把 PDF/OCR 的硬換行接回同一句。

是否保留換行由 `_should_keep_line_break()` 決定。

會保留換行的情況：

- 前一行或下一行是空白。
- 前一行是 heading。
- 下一行是 heading。
- 下一行是目錄標題或圖表標題。
- 前一行以句尾標點結束。

例如：

```text
災害防救辦
公室
```

若不符合保留換行條件，會被接成：

```text
災害防救辦公室
```

---

## 3. Heading Candidate 偵測

位置：`detect_heading_candidates()`

系統使用 `HEADING_PATTERNS` 偵測可能的章節標題：

```python
HEADING_PATTERNS = [
    ("decimal", ...),      # 1. / 1.1 / 1.1.1
    ("paren_num", ...),    # (1) / （1）
    ("paren_upper", ...),  # (A) / （A）
    ("paren_lower", ...),  # (a) / （a）
    ("cjk_comma", ...),    # 一、目的
    ("cjk_paren", ...),    # （一）二級開設
]
```

每個候選會變成 `HeadingCandidate`：

```python
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
```

欄位含義：

- `style`：標題樣式，例如 `cjk_comma`、`cjk_paren`、`decimal`。
- `num`：序號，例如 `三`、`一`、`1`。
- `title`：標題文字，例如 `作業程序`。
- `raw`：原始標題行，例如 `三、作業程序`。
- `level`：後續推斷出的層級。

### Bare Number 修補

有些 PDF 會把條列編號和內容拆成兩行：

```text
2.
參與機關（單位、團體）：...
```

目前用：

```python
BARE_NUMBER_RE = re.compile(r"^(?P<num>\d+(?:[.．]\d+)*)[.．、]$")
```

若偵測到單獨的 `2.`，且下一行不是 heading，會先合併成：

```text
2. 參與機關（單位、團體）：...
```

再重新套用 heading pattern。

---

## 4. 目錄與雜訊排除

位置：

- `find_toc_regions()`
- `filter_toc_and_noise()`
- `is_noise_line()`

### 目錄區塊

系統會找出目錄區域，例如：

```text
目錄
...
```

並排除目錄區中的 heading candidate，避免目錄項目被誤當正文標題。

目錄判斷會參考：

- `目錄`、`頁次`、`圖目錄`、`表目錄` 等標題。
- dot leader，例如 `......`。
- 後續是否出現真正正文 heading。

### 雜訊行

以下候選會被排除：

- 單獨頁碼或羅馬數字。
- 目錄標題。
- 含有大量 dot leader 的目錄項。
- 圖表標題，例如 `圖 1：...`、`表 2：...`。

---

## 5. 文件結構 Profile

位置：`analyze_structure_profile()`

系統會先統計候選標題的型態，用來決定後續 level 推斷策略。

主要統計：

```python
style_counts
decimal_levels
max_decimal_depth
has_decimal
has_cjk
style_order
style_order_source
style_parent_scores
cjk_style_order
cjk_style_order_source
cjk_parent_scores
```

其中：

- `has_decimal`：文件是否大量使用 `1.1`、`1.1.1` 這種 decimal 階層。
- `has_cjk`：文件是否使用 `一、`、`（一）` 這類中文章節。
- `max_decimal_depth`：decimal heading 的最大有效深度。
- `style_order`：依 style 首次出現順序建立層級骨架，並用包覆證據確認父子關係。
- `style_order_source`：通用 style 順序的判定來源，例如 `parent_child_evidence`。
- `style_parent_scores`：各 style 作為父層時的包覆證據分數。
- `cjk_style_order`：只針對 `一、` 與 `（一）` 的相容/debug 子集。
- `cjk_style_order_source`：CJK 層級順序的判定來源。
- `cjk_parent_scores`：兩種 CJK 樣式分別作為父層時的包覆證據分數。

目前只有當多層 decimal 出現數量達到門檻時，才會把文件視為 decimal 主導。

---

## 6. Level 推斷

位置：`infer_levels()`

### Decimal 主導文件

如果文件大量使用 decimal heading：

```text
1.
1.1
1.1.1
```

level 會依 decimal 深度決定：

```text
1       → level 1
1.1     → level 2
1.1.1   → level 3
```

其他樣式會被放在 decimal 層級之後。

### 中文法規 / 條列文件

如果文件不是 decimal 主導，會根據所有 heading style 的首次出現順序與直接子層包覆證據推斷層級順序，而不只處理 `一、` 與 `（一）`。

目前做法是：

1. 先依 heading style 在正文中首次出現的順序建立 `style_order`。
2. 再計算直接子層包覆證據，例如在兩個 `一、` 之間第一個不同 style 是否為 `（一）`。
3. 若首次順序中的父子關係有包覆證據支撐，`style_order_source` 會是 `parent_child_evidence`。
4. 若沒有足夠包覆證據，仍會保留首次出現順序，但 source 會是 `default_tie_break` 或 `single_style`。

常見法規格式如下：

```text
一、目的        → level 1
（一）二級開設  → level 2
1. 開設時機     → level 3
（1）境外部分   → level 4
```

如果文件使用相反格式，例如：

```text
（一）總則
一、目的
二、適用範圍

（二）作業程序
一、啟動時機
二、任務分工
```

系統會從 `（一）... 一、... 二、... （二）...` 這種包覆模式推斷：

```text
（一）總則   → level 1
一、目的     → level 2
二、適用範圍 → level 2
```

也就是說，`cjk_comma` 與 `cjk_paren` 的 level 不再寫死；常見的 `一、` 外層格式本身也是透過包覆證據判定，而不是 fallback。

例如 `境外核災處理作業要點.pdf` 中可以觀察到：

```text
三、作業程序
（一）二級開設
（二）一級開設
（三）...
```

這會讓 `cjk_comma` 作為父層的分數高於 `cjk_paren` 作為父層，因此判定：

```text
一、 → level 1
（一） → level 2
```

只有在兩種樣式都存在但包覆證據平手時，才會用 `一、 → （一）` 作為 tie-break 預設。

同樣的方式也適用於其他樣式。例如：

```text
(A) 總則
(a) 目的
(1) 細項

(B) 作業程序
(a) 啟動時機
(1) 細項
```

系統會依包覆證據推斷：

```text
(A) / (B) → level 1
(a)      → level 2
(1)      → level 3
```

如果多個 style 之間沒有足夠包覆證據，才會使用預設 tie-break 順序：

```text
優先保留正文首次出現順序；若仍需固定順序參照，使用：
cjk_comma → cjk_paren → decimal → paren_num → paren_upper → paren_lower
```

---

## 7. 序號連續性驗證

位置：`validate_sequence()`

對 decimal 樣式，系統會檢查序號是否合理連續。

目的：避免把正文中的年份、電話、編號誤判為標題，例如：

```text
1030008776
1904
24410
```

驗證邏輯大致是：

- 第一層 decimal 可以從 `1` 開始。
- 允許少量跳號，跳號上限由 `MAX_JUMP` 控制。
- 多層 decimal 必須有已接受的父層，例如要接受 `2.1`，必須先有合理的 `2`。

中文樣式與括號樣式目前較明確，會直接接受。

---

## 8. 結構切分

位置：`split_by_candidate_lines()`

這是 chunk 產生的核心。

### 基本策略

系統採用 top-down 切分：

```text
先看整段是否 <= chunk_size
↓
若可容納，整段保留為一個 chunk
↓
若超過，找目前範圍內最上層 heading 切開
↓
每個子範圍再遞迴判斷
↓
若沒有 heading 可切，fallback 到 RecursiveCharacterTextSplitter
```

也就是說，heading 是優先結構邊界，但 `chunk_size` 仍是保護上限。

### 父標題不獨立成 chunk

針對這種情況：

```text
三、作業程序
（一）二級開設
...
```

系統會避免產生只有：

```text
三、作業程序
```

的獨立 chunk。

如果切分時發現子節前面的 prelude 只有目前父標題，會讓子節 segment 從父標題開始，因此結果會是：

```text
三、作業程序
（一）二級開設
...
```

這是由 `_is_only_current_parent_heading()` 與 `_split_range()` 內的 segment start 調整完成。

---

## 9. Metadata 產生

位置：

- `_make_heading_path()`
- `_build_metadata()`
- `_partition_sections()`

目前 metadata 主要欄位：

```python
{
    "source_file": "...pdf",
    "doc_title": "...",
    "page_start": 1,
    "page_end": 2,
    "heading_path": [...],
    "contains_sections": [...],
    "contains_subitems": True,
    "chunk_index": 3
}
```

### doc_title

`doc_title` 由 `source_file` 推得：

```text
境外核災處理作業要點.pdf
↓
境外核災處理作業要點
```

目前不是從 PDF 內文第一行抓。

### heading_path

`heading_path` 表示 chunk 起始位置所在的完整標題階層。

例如：

```text
三、作業程序
（一）二級開設
1. 開設時機...
```

metadata：

```python
"heading_path": [
    {"level": 1, "style": "cjk_comma", "num": "三", "title": "作業程序"},
    {"level": 2, "style": "cjk_paren", "num": "一", "title": "二級開設"}
]
```

`1. 開設時機` 屬於細項條列，不會進 `heading_path`；是否包含細項由 `contains_subitems` 表示。

### contains_sections

`contains_sections` 表示 chunk 實際涵蓋的「具名結構章節」。

目前具名結構樣式：

```python
_SECTION_STYLES = {
    "cjk_comma",
    "cjk_paren",
    "paren_upper",
    "paren_lower",
}
```

細項條列樣式：

```python
_ITEM_STYLES = {
    "decimal",
    "paren_num",
}
```

細項不會放進 `contains_sections`，只會讓：

```python
"contains_subitems": True
```

#### 單一章節

若 chunk 是：

```text
一、目的
...
```

因為它沒有更下一層具名子節，所以它本身就是涵蓋的章節：

```python
"heading_path": [
    {"level": 1, "style": "cjk_comma", "num": "一", "title": "目的"}
],
"contains_sections": ["一、目的"]
```

#### 父章 + 單一子節

若 chunk 是：

```text
三、作業程序
（一）二級開設
1. ...
2. ...
```

父章 `三、作業程序` 只作為定位脈絡，具體涵蓋單位是 `（一）二級開設`：

```python
"heading_path": [
    {"level": 1, "style": "cjk_comma", "num": "三", "title": "作業程序"},
    {"level": 2, "style": "cjk_paren", "num": "一", "title": "二級開設"}
],
"contains_sections": ["（一）二級開設"]
```

因此 `三、作業程序` 不會再重複放入 `contains_sections`。

#### 父章 + 多個同層子節

若 chunk 是：

```text
五、其他
（一）...
（二）...
（三）...
（四）...
```

`heading_path` 退回共同父章：

```python
"heading_path": [
    {"level": 1, "style": "cjk_comma", "num": "五", "title": "其他"}
]
```

`contains_sections` 保留實際涵蓋的子節：

```python
"contains_sections": [
    "（一）...",
    "（二）...",
    "（三）...",
    "（四）..."
]
```

---

## 10. 短 Chunk 修補

位置：`repair_small_chunks_docs()`

切完後，系統會修補過短 chunk。

過短門檻：

```python
min_tokens = max(80, chunk_size // 10)
```

合併原則：

1. 優先嘗試與下一個 chunk 合併。
2. 若不行，再嘗試與前一個 chunk 合併。
3. 合併後不可超過 `chunk_size`。
4. 不合併不同第一層章節，避免 `二、適用時機` 被併到 `三、作業程序`。
5. 最多修補 3 輪。

合併 metadata 時：

- `page_start` 取較小頁碼。
- `page_end` 取較大頁碼。
- `heading_path` 保留兩個 chunk 的共同前綴。
- 分歧出去的具名章節會放入 `contains_sections`。
- `contains_subitems` 只要任一 chunk 為 true，合併後就是 true。

---

## 目前限制

1. `doc_title` 只從檔名推得，尚未從文件正文推斷。
2. 目前中文數字轉換只支援常見範圍，特殊中文編號可能不完整。
3. `壹、貳、參`、`甲、乙、丙` 尚未納入主要 heading pattern。
4. PDF/OCR 硬換行修復是啟發式規則，仍可能誤接或漏接。
5. 若文件沒有可辨識 heading，會直接 fallback 成一般 token chunk。
6. 若 `chunk_size` 太小，長條列仍可能被 fallback splitter 硬切。
