# Chunk 切分流程與 Metadata 完整參考

> 對象：`doc_structure.py`（全部邏輯）。本文件鉅細靡遺列出每一個篩選、搜尋、
> 判定規則與門檻值，供評估「正確性」與「泛化性」。所有行號對應 `doc_structure.py`。
>
> **核心模型（anchor-based）**：每個命中的標號都只是一個 *anchor candidate*。系統**不**用
> regex 判斷它是「章節」還是「條列」（同一種標號在不同文件可能扮演不同角色）。改為：
> 1. 對每個 anchor 以 `pattern_id`（哪條規則）+ `marker`（標號格式）描述，**沒有 role、沒有 kind**；
> 2. 依**方向感知父子證據 + 拓樸排序**推出 `anchor_order`，再給每個 anchor 一個 `level`；
> 3. **延遲切分**：區塊只在「太大」或「頂層 anchor 太多」時才往更深層切；
> 4. metadata 用 `contained_sections` + `structure_tree`（偵測到的結構，非嚴格目錄）。

---

## 0. 總覽

入口 `build_documents_from_pages(pages, chunk_size, overlap_size, source_file)`
（[doc_structure.py:1529](../doc_structure.py)）回傳 `(docs, structure_info)`。

```
pages: list[str]（每頁原始文字）
  │
  ├─(1) build_line_records          → list[LineRecord]  逐行 + 字元座標
  ├─(2) clean_line_records          → 去頁碼 / 併空行 / 反換行 / 重算座標
  ├─(3) detect_heading_candidates   → list[HeadingCandidate]（registry 比對 + 裸號併行）
  │
  ├─(4) find_toc_regions            → 目錄區段 [(start,end)]
  ├─(5) filter_toc_and_noise        → 移除目錄行 + is_noise_line
  ├─(6) find_table_regions          → 表格區段 [(start,end)]
  ├─(7) filter_table_regions        → 移除表格內假 anchor
  │
  ├─(8) analyze_structure_profile   → pattern_counts + anchor_order（證據式）
  ├─(9) infer_levels                → 每個 anchor 指定 level（依 anchor_order）
  ├─(10) validate_sequence          → 數字連續性檢查（僅 decimal_chain）
  │
  ├─(11) split_by_candidate_lines   → 延遲 anchor 遞迴切 chunk → list[Document]
  ├─(12) repair_small_chunks_docs   → 最多 3 輪小 chunk 合併（依關聯度）
  └─(13) chunk_index 連續編號        → 寫入 metadata
```

`structure_info` 鍵：`toc_pages_excluded`、`preview_headings`、`profile`、
`candidate_count`，外加頂層 anchor 診斷 `anchor_order`、`anchor_order_source`、
`anchor_order_confidence`、`anchor_order_support_count`、`parent_child_scores`
（除錯 / UI 顯示用，不進 chunk metadata）。

---

## 1. 資料結構

### LineRecord（[doc_structure.py:198](../doc_structure.py)）
```python
LineRecord(page: int, text: str, char_start: int = 0, char_end: int = 0)
```
`char_start/char_end` 是「全文串接後」的字元偏移（每行尾 +1 當換行）。目前僅在切分
內部當行索引用，**不會**寫入 chunk metadata。

### HeadingCandidate（[doc_structure.py:206](../doc_structure.py)）
```python
HeadingCandidate(page, line_no, char_start, char_end,
                 pattern_id: str, marker: dict,
                 num: str, title: str, raw: str, level: int = 0)
```
- `pattern_id`：命中哪一條 registry 規則（如 `decimal` / `cjk_comma` / `article_cjk`）。
- `marker`：標號格式的結構化資料（見 §3）。
- `level`：該 anchor 在**本文件**中的深度（由 `anchor_order` 決定），**不代表它一定是正式章節**。
- **沒有 `role`、沒有 `kind`**——主流程只靠 `pattern_id` / `marker` / `level`。

---

## 2. Pattern Registry（anchor 規則集中地，[doc_structure.py:42](../doc_structure.py)）

新增 / 修改 anchor 規則的**唯一位置**。每條規則是一個 `HeadingPatternSpec`（frozen dataclass）：

```python
HeadingPatternSpec(
    pattern_id: str,        # 規則身分（也用於 section_key）
    regex: re.Pattern,      # 需含具名群組 (?P<num>) (?P<title>)
    priority: int,          # 偵測順序：數字小者先試，第一個命中即停
    num_type: str,          # arabic / arabic_chain / cjk / latin_upper / latin_lower
    marker_style: str = "affix",     # decimal / paren / affix（決定 prefix/suffix 怎麼來）
    prefix: str = "",       # affix 樣式專用（字面前綴，如 第）
    suffix: str = "",       # affix 樣式專用（字面後綴，如 條 / 、）
    level_strategy: str = "flat",    # flat / decimal_chain（用小數點深度算層級）
    examples: tuple = (),   # 文件用
)
```

> **沒有 `role` 欄位**。是否進 tree、是否切分，皆**不**由 spec 決定，而是 per-document 由
> anchor level + 延遲切分決定（§9–§12）。

### 內建規則（`_DEFAULT_PATTERN_SPECS`，[doc_structure.py:68](../doc_structure.py)）

| pattern_id | priority | num_type | marker_style | regex 重點 | 範例 |
|---|---|---|---|---|---|
| `decimal` | 10 | arabic | decimal（+`decimal_chain`） | `^(\d+(?:[.．]\d+)*)[.．、]?\s*([一-鿿A-Za-z].+)$` | `1.` `1.1` `2.3.4` |
| `paren_num` | 20 | arabic | paren | `^[\(（](\d+)[\)）][.．]?\s*(.+)$` | `(1)` `（2）` |
| `paren_upper` | 30 | latin_upper | paren | `^[\(（]([A-Z])[\)）][.．]?\s*(.+)$` | `(A)` |
| `paren_lower` | 40 | latin_lower | paren | `^[\(（]([a-z])[\)）][.．]?\s*(.+)$` | `(a)` |
| `cjk_comma` | 50 | cjk | affix（suffix=`、`） | `^([一二三四五六七八九十百]+)、\s*(.+)$` | `一、` `十二、` |
| `cjk_paren` | 60 | cjk | paren | `^[（(]([一二三四五六七八九十百]+)[）)]\s*(.+)$` | `（一）` |

> 這 6 條都只是 anchor 規則；例如 `（一）/（二）` 在某文件是章節、在另一文件是段落條列，
> 系統不在 registry 層決定，交給 level + 延遲切分。

### Registry API（[doc_structure.py:125–165](../doc_structure.py)）
- `get_pattern_specs()`：全部 spec（註冊順序）。
- `get_spec(pattern_id)`：取單一 spec。
- `iter_specs_by_priority()`（[:134](../doc_structure.py)）：依 priority 升冪（偵測用順序）。
- `default_anchor_order()`（[:139](../doc_structure.py)）：priority 順序的 id 清單，僅當 anchor order 的 tie-break。
- `is_decimal_chain_spec(pattern_id)`（[:144](../doc_structure.py)）：是否 `level_strategy=="decimal_chain"`。
- `register_pattern_spec(spec, *, replace=False)` / `unregister_pattern_spec(id)`
  （[:149](../doc_structure.py)）：runtime 增刪規則（測試或擴充用）。

### 新增一條規則的範例（例如「第十條」）
```python
register_pattern_spec(HeadingPatternSpec(
    pattern_id="article_cjk",
    regex=re.compile(r"^第(?P<num>[一二三四五六七八九十百千萬零〇]+)條\s*(?P<title>.+)$"),
    priority=5, num_type="cjk", marker_style="affix", prefix="第", suffix="條",
))
```
→ 自動產生 `pattern_id="article_cjk"`、marker
`{text:"第十條", num:"十", num_type:"cjk", prefix:"第", suffix:"條"}`、
`section_key="article_cjk:十"`，並自動進入 detect / order / level / tree，**不必改其他地方**。

### 裸號併行（PDF 把序號和標題拆成兩行）
`BARE_NUMBER_RE = ^(\d+(?:[.．]\d+)*)[.．、]$`（[doc_structure.py:167](../doc_structure.py)）

`detect_heading_candidates`（[doc_structure.py:380](../doc_structure.py)）：若某行只是裸號
（如 `1.`），且**下一行**非 anchor，則合併成 `「1. 下一行內容」` 再以
`_match_spec`（[doc_structure.py:371](../doc_structure.py)）依 priority 重新比對；合併後的 candidate
`char_start` 取本行、`char_end` 取下一行。

**關鍵限制（泛化性注意點）**
- decimal title 必須以 `[一-鿿A-Za-z]`（中日韓統一表意文字或英文字母）開頭；以數字或符號
  開頭的標題不會被視為 decimal anchor。
- CJK 數字僅支援 `一二三四五六七八九十百`（內建規則無「千、萬、零、廿」等）。
- 分隔符容許 `.`（半形點）`．`（全形點）`、`（頓號）。
- `paren_*` / `cjk_paren` 容許半形 `()` 與全形 `（）` 混用。

---

## 3. marker（[doc_structure.py:223](../doc_structure.py)）

簽名 `build_marker(spec: HeadingPatternSpec, num: str, raw: str) -> dict`：
```jsonc
{ "text": "（二）", "num": "二", "num_type": "cjk", "prefix": "（", "suffix": "）" }
```
**所有格式知識都來自 spec**，函式只依 `spec.marker_style` 分派，不再 `if pattern_id == ...`：

- `marker_style == "decimal"`：`num_type` 取 `spec.num_type`（含 `.`/`．` 時升級為
  `arabic_chain`）；suffix 取緊跟在 num 後的 `.`/`．`/`、`（若有）。
- `marker_style == "paren"`：prefix 依 raw 首字判半形 `(` 或全形 `（`；suffix 同理。
- `marker_style == "affix"`：prefix / suffix 直接取自 `spec.prefix` / `spec.suffix`
  （如 `第…條`、cjk_comma 的 `、`）。
- `text = f"{prefix}{num}{suffix}"`（重建用完整前綴）。

> 舊的 `role` / `_TREE_KINDS` / `_ENUM_KINDS` / `heading_kind()` 已全部移除。

---

## 4. 清理：clean_line_records（[doc_structure.py:315](../doc_structure.py)）

四個 pass：

1. **去頁碼**：`PAGE_NUMBER_RE = 第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁` 整段移除，再 strip。
2. **併空行**：連續空行壓成一行。
3. **反換行（unwrap）**：判斷是否保留斷行 → `_should_keep_line_break(prev, nxt)`：
   - 任一方為 anchor 行（`is_heading_line`）→ 保留斷行。
   - `nxt` 是 TOC 標題或圖表 caption → 保留。
   - `prev` 以句末符號結尾（`SENTENCE_END_RE = [。！？；;：:]$`）→ 保留。
   - 否則視為軟換行 → 把 `nxt` 併入 `prev`。
4. **重算座標**：合併後重建連續 `char_start/char_end`。

> 影響：標題的「換行續行」會被併回；但若內文被誤判為續行而併入，可能改變 chunk 邊界。
> anchor 行一律保護不被併（前後都檢查 `is_heading_line`）。

`is_heading_line`（[doc_structure.py:267](../doc_structure.py)）：裸號或任一 registry 規則命中。
`is_standalone_section_heading`（[doc_structure.py:274](../doc_structure.py)）：僅 `cjk_comma`/`cjk_paren`。

---

## 5. 目錄偵測：find_toc_regions（[doc_structure.py:469](../doc_structure.py)）

狀態機，回傳 `[(start_idx, end_idx)]`。

- `TOC_TITLE_RE = ^(目錄|頁次|附圖目錄|附表目錄|圖目錄|表目錄)$`（容許字間空白）→ 進入 TOC。
  TOC 內再遇 TOC 標題會被吸收（持續累積）。
- **結束條件**：在 TOC 內遇到「真正的 body anchor」才結束，其定義為同時滿足：
  (a) 命中某 registry 規則；
  (b) 不含 dot-leader（`[.．…]{3,}`）；
  (c) title 結尾不是裸頁碼（`\s+\d+\s*$`，排除 `1. 綜合概述 1` 這種無點目錄列）；
  (d) `pattern_id == "decimal"` 時，首段數字必須 `≤ MAX_JUMP + 1`（=6，排除 `24410` 之類代碼）。
- 文件結束仍在 TOC → 該段延伸到最後一行。

`find_body_start_page`（[doc_structure.py:524](../doc_structure.py)）：回傳第一個非 dot-leader anchor
所在頁（輔助用）。

`filter_toc_and_noise`（[doc_structure.py:535](../doc_structure.py)）：剔除落在 TOC 行號集合內、或
`is_noise_line` 為真的 candidate。

---

## 6. 雜訊過濾：is_noise_line（[doc_structure.py:426](../doc_structure.py)）

對單一 candidate 的 `raw` / `title` / `pattern_id` 判斷，命中任一即丟棄：

1. **裸頁碼**：`^(i{1,4}|vi{0,3}|xi{0,3}|[IVX]+|\d+)$`（忽略大小寫，含羅馬數字 i–xiii 級別）。
2. **目錄/索引標題**：`TOC_TITLE_RE`。
3. **dot-leader 目錄列**：含 `[.．…]{3,}`。
4. **無點目錄列**：`pattern_id == "decimal"` 且 title 以 `\s+\d+\s*$`（空白+數字）結尾，如 `1. 概論 1`。
5. **圖表 caption**：`FIGURE_TABLE_RE = ^[圖表]\s*\d+([-.–]\d+)?\s*[：:]`。
6. **單整數 decimal 的前導零**（僅 `pattern_id == "decimal"` 且 num 為**單一整數**、不含小數點時）：
   - num 長度 > 1 且首位為 `0`（前導零，如 `001`、`004`）→ 代碼/編號，非 anchor → 雜訊。

> **anchor 模型的調整**：規則 6 **不再**因「title 以 `。！？` 結尾」或「title 含逗號」就丟棄。
> 在 anchor 模型下標號行只是 anchor candidate（**不分 list / section**）：`1. 協助督導。` 與
> `1. 全國環境輻射監測…，提供…。` 都是合法（深層）anchor —— 是否重要、是否該細切由
> level + 延遲切分決定，**不在偵測階段猜「散文 vs 清單」**。
> **逗號過濾已整層移除**：它是舊「structural / enum」思維的殘留；法規清單項目幾乎都含逗號，
> 舊版「含逗號就丟」會整條漏掉合法清單（曾導致 `（二）核安會` 下 `1./2./3.` 消失）。
> 前導零過濾保留（與 list/section 之分無關）。規則 4、6 只作用於 `pattern_id == "decimal"`；
> `paren_*`、`cjk_*` 不受影響。

---

## 7. 表格偵測：find_table_regions（[doc_structure.py:590](../doc_structure.py)）

目的：表格內「裸整數列」(`1 文字`，無分隔符) 不可被當成 anchor。

- **裸整數列判定** `_bare_integer_row`（[doc_structure.py:551](../doc_structure.py)）：以
  `get_spec("decimal").regex` 比對，但 num 無小數點、且 num 後緊跟的字元**不是**
  `.`/`．`/`、`（有分隔符＝真章節 `1.`，回傳 None）。
- 把所有裸整數列依行號分群成 run，群內可容忍 `TABLE_GAP_TOLERANCE = 4` 行間隔（即行號差 `≤ 5`）。
- run 成為表格區段的條件（`_finalize`）：
  - **有錨點**：上方 `TABLE_ANCHOR_LOOKBACK = 3` 個非空行內有表格標題
    （`TABLE_CAPTION_RE = ^表\s*\d+(?:[-.–—]\d+)*\s*[：:]`）或欄位標題
    （`TABLE_HEADER_RE = ^(?:項次|編號|序號|項目)\s`），且 run 長度 `≥ TABLE_ANCHORED_MIN = 2`；
    區段起點往上含錨點行。
  - **獨立成表**：run 長度 `≥ TABLE_STANDALONE_MIN = 5`（無錨點也算）。

`filter_table_regions`（[doc_structure.py:630](../doc_structure.py)）：剔除落在表格行號集合內的 candidate。

---

## 8. 結構統計：analyze_structure_profile（[doc_structure.py:650](../doc_structure.py)）

對**所有** anchor（不分角色）一起統計，回傳 dict：
- `pattern_counts`：各 `pattern_id` 次數。
- `decimal_levels`：各 decimal_chain 深度（小數點數+1）的出現次數。
- `max_decimal_depth`：出現次數 `≥ MIN_DEPTH_SUPPORT = 2` 的最大深度（否則 0）。
- `has_decimal`：深度 `≥ 2` 的 decimal 總數 `≥ MIN_MULTI_DECIMAL = 3`。
- `has_cjk`：任一 candidate 的 `num_type == "cjk"`。
- `anchor_order`：所有出現過的 pattern，依外層→內層排序（見 §9）。
- `anchor_order_source`：`parent_child_evidence` / `single_style` / `default_order` / `none`。
- `anchor_order_confidence`：0..1，排序與證據的一致程度。
- `anchor_order_support_count`：父子證據總量（`sum(parent_child_scores.values())`），
  避免「只有 1 條證據卻 confidence=1.0」被當可靠。
- `parent_child_scores`：debug 用，如 `{"cjk_comma->cjk_paren": 3, "cjk_paren->decimal": 5}`。

範例：
```jsonc
{
  "pattern_counts": { "cjk_comma": 2, "cjk_paren": 1, "decimal": 2 },
  "anchor_order": ["cjk_comma", "cjk_paren", "decimal"],
  "anchor_order_source": "parent_child_evidence",
  "anchor_order_confidence": 1.0,
  "anchor_order_support_count": 4,
  "parent_child_scores": { "cjk_comma->cjk_paren": 1, "cjk_paren->decimal": 1 }
}
```

> 不再有 `structural_order` / `enum_order` / `style_order` / `style_counts` /
> `cjk_style_order` —— 全部併為單一 `anchor_order`。

---

## 9. anchor order 推斷：infer_anchor_order_by_evidence（[doc_structure.py:810](../doc_structure.py)）

> 對**所有** anchor（不分角色）一起排序。用**方向感知父子證據 + 拓樸排序**，不以首見順序為主，
> 也不用全域 rank 加總（會讓中間層被它「身為某父層的子」的強邊拖到比更內層還深）。

### 方向感知證據 `_evidence_scores`（[doc_structure.py:698](../doc_structure.py)）
逐一掃描相鄰且 pattern 不同的轉換 `A → B`，用「B 是否延續自己的計數」決定方向：
- **B 的值比 B 上一次出現大**（延續中的外層計數，如 `…2. （二）…`，`（二）` 接 `（一）`）
  → `B` 是外層 → `score[(B, A)] += 1`。
- **B 是全新的**（首次出現，或計數重啟，如 `（一）1.` 的 `1.`）→ 正在往子層下探
  → `A` 是外層 → `score[(A, B)] += 1`。

這直接區分「父層在子序列跑完後 resume」與「父層帶出第一個子」——兩者在單純鄰接計數下長得
一模一樣，正是先前 decimal/cjk_paren 反序的根因。`_marker_value`（[doc_structure.py:739](../doc_structure.py)）
依 num_type 把標號轉成可比較的整數 tuple（cjk 用 `_cjk_to_int`、decimal 用小數點 tuple、
latin 用字母序）。

### 拓樸排序 `_topological_order`（[doc_structure.py:768](../doc_structure.py)）
- 對每對 `(A,B)` 算淨證據 `net = score(A,B) − score(B,A)`；`net>0` → 加有向邊 `A→B`（A 外層）。
- Kahn 演算法逐一取出「沒有未排父層」的節點（source）；多個 source 時用
  `tie_key = (registry priority index, first_seen)` 決定先後。
- 若殘留環（無 source），釋放 tie_key 最小者打破，保證一定產生全序。

### 信心值 `_order_confidence`（[doc_structure.py:754](../doc_structure.py)）
最終順序中，與證據一致（外層在前）的分數占總證據的比例（0..1）。

### 為什麼能避免首見順序誤導 + 反序（對應測試）
- **內層 pattern 先出現**：開頭 preamble `（一）` 讓 first_seen 誤判 cjk_paren 先；但
  `（一）` 後接 `一、` 時 `一、` 是全新計數 → 證據判 cjk_comma 外層，平手由 priority
  tie-break 仍選 cjk_comma → `anchor_order[0] == "cjk_comma"`。
- **計數重啟的內層**（真實 PDF：每個 `（）` 下 `1. 2.` 重啟）：`…2. （二）…` 因 `（二）`
  延續計數而正確記為 `cjk_paren→decimal`，不再誤記為 `decimal→cjk_paren` →
  `anchor_order = [cjk_comma, cjk_paren, decimal]`（confidence 1.0）。

`_parent_child_evidence(cands, parent, child)`（[doc_structure.py:731](../doc_structure.py)）保留為單對
查詢的輔助函式（回傳 `_evidence_scores` 中該對的值）。

---

## 10. 層級指定：infer_levels（[doc_structure.py:848](../doc_structure.py)）

純依 `profile["anchor_order"]` 指定 level（缺時用 `infer_anchor_order_by_evidence` 重算）：

- 基本：`anchor_levels[pattern_id] = anchor_order 中的 index + 1`。
- `level_strategy == "decimal_chain"` 的 pattern：實際 level = `base + 小數點深度 − 1`。
- 為避免排在 decimal_chain 之後的 pattern 與深層 decimal 撞層，會把它們往下位移
  `max(max_decimal_depth, 1) − 1`。
- 不在 anchor_order 的 pattern → `len(anchor_order) + extra + 1`（最深）。

例如 `anchor_order = ["cjk_comma","cjk_paren","decimal"]`：
```
一、       level 1
（一）     level 2
1.        level 3
1.1       level 4
1.1.1     level 5
```

> `level` 只代表本文件 anchor 的深度，**不代表正式章節層級**。

---

## 11. 序號連續性：validate_sequence（[doc_structure.py:924](../doc_structure.py)）

僅對 `is_decimal_chain_spec(pattern_id)` 為真者做檢查；CJK 與 paren_* 一律接受。

- 以 `_parse_decimal_parts` 切成整數陣列；`last_at_prefix` 記錄各層 prefix 最後接受的號碼。
- **第一層（depth 1）** 接受條件（任一）：`n == 1`；或已有 last 且 `1 ≤ n − last ≤ MAX_JUMP(=5)`；
  或首次出現（last==0）且 `n ≤ MAX_JUMP + 1 (=6)` → 排除 `24410`、`1904`、`50` 之類跳號。
- **子層（depth ≥ 2）**：parent prefix 必須已被接受；child 號碼須 `==1` 或在 MAX_JUMP 內遞增。
  parent 未接受 → 整個子節點拒絕（孤兒）。
- **父層邊界重置**：以 `decimal_base_level` 記錄目前追蹤的 depth-1 decimal 所在 level；
  之後遇到「比它更淺（level 較小）的非 decimal anchor」（代表換到另一個父層，如 `（二）核安會`、
  `四、…`）即清空 `last_at_prefix` / `accepted_prefixes`。**避免序號計數跨父層外溢**——否則某段
  清單若首項 `1.` 被上游濾掉，下一段的 `2./3.` 會因「接在前一段最後號碼之後」而連續性失敗被誤刪。
  比 decimal 更深的內層 anchor（如 decimal 底下的 `paren_num`）不觸發重置。

CJK 轉整數 `_cjk_to_int`（[doc_structure.py:915](../doc_structure.py)）僅 best-effort，支援 `十X`（十一…）
與個位加總；目前 validate_sequence **未**對 CJK 做連續性驗證（但 `_marker_value` 用它判斷
anchor order 的計數延續）。

---

## 12. 延遲切分：split_by_candidate_lines（[doc_structure.py:1155](../doc_structure.py)）

由上而下遞迴 `_split_range(start, end, cands, stack)`。**延遲切分的停止條件**：

```
token_count ≤ chunk_size
且（頂層 anchor 數 ≤ MAX_ANCHORS_PER_CHUNK(=6)　或　token_count ≤ ANCHOR_SPLIT_MIN_RATIO(=0.5)·chunk_size）
```
- 「頂層 anchor 數」= 範圍內 level 最小（最外層）的 anchor 個數，代表這一塊跨越幾個**最外層**
  區段（一個區段內含多少子項不算）。
- **anchor 數上限是 token-aware**：只有當區塊「偏大」（token > `ANCHOR_SPLIT_MIN_RATIO·chunk_size`）時，
  anchor 數超過 6 才強制往下切；**純項目多但 token 小**（≤ 一半 chunk_size）的區塊照樣整塊保留。
  例：`（二）核安會` 下 `1.`–`7.` 共 7 項（>6）但只有 ~286 tokens（chunk_size=1000），不再被拆成
  「`1.` 一塊、`2.`–`7.` 一塊」，而是整段留著（避免切碎後又被 repair 併回）。
- **符合停止條件** → 整塊保留為一個 chunk（即使內部還有更深 anchor）。
- **太大，或（偏大且頂層 anchor 太多）** → 取目前最外層 anchor level 當切點切 segments：
  - anchor 之前的前言區段（除非該區段僅含當前父 anchor 自己，則併入下一段）；
  - 每個 anchor 到下一個同級 anchor 之間為一段，遞迴往**下一層** anchor 切；
  - `stack`（祖先鏈）隨遞迴維護，供 metadata 的 anchor_path。
- **太大且無 anchor 可切** → 才 fallback `RecursiveCharacterTextSplitter`
  （`_fallback_docs`，[doc_structure.py:1221](../doc_structure.py)，並加 warning `fallback_splitter_used`）。

token 計數 `_count_tokens`（[doc_structure.py:259](../doc_structure.py)）：有 tiktoken 用
`cl100k_base`，否則 `len//4` 近似。

效果：
- 「四、任務分工」整段若 token 不大、頂層只有它一個 anchor → **整塊保留**，不會把每個 `1.` 拆開。
- 同父層下的長清單（如 `（二）核安會` 的 `1.`–`7.`）即使項目數 >6，只要 token 偏小也整塊保留。
- 若太大 → 先切成 `（一）/（二）`，仍太大才用 `1.2.3.` 繼續切。

`_make_doc`（[doc_structure.py:1193](../doc_structure.py)）：
- `page_start/end = min/max(pages)`（pages 只計非空、非 TOC 行的頁碼）。
- `anchor_path = _make_anchor_path(stack)`（祖先鏈，含 `pattern_id`）。
- `contained` = 該範圍內的 anchors。
- → `_build_metadata`（只帶入 `warnings`；不再傳 confidence / support_count）。

---

## 13. 小 chunk 修復：repair_small_chunks_docs（[doc_structure.py:1476](../doc_structure.py)）

最多跑 3 輪（外層 [doc_structure.py:1572](../doc_structure.py)），直到數量不變。

- 門檻 `min_tokens = max(80, chunk_size // 10)`；token `< min_tokens` 視為過小。
- **合併優先序**（`_merge_tier_reason`，[doc_structure.py:1403](../doc_structure.py)）——挑最相關且合併後仍
  `≤ chunk_size` 的鄰居（左右皆評估，同分優先右鄰）：
  1. `small_chunk_same_parent`：兩 chunk 的 breadcrumb 父路徑相同（如同屬 `四 > （一）`）。
  2. `small_chunk_same_top_anchor`：頂層 anchor 相同（一方無 anchor 也歸此類）。
  3. `small_chunk_cross_anchor`：跨頂層 anchor（**最後手段**，仍允許）。
- 合併 metadata `_merge_metadata`（[doc_structure.py:1459](../doc_structure.py)）：`page_start/end` 取 min/max；
  `structure_tree` 以 `_merge_trees`（[doc_structure.py:1428](../doc_structure.py)）依 `section_key` 遞迴聯集、
  保序、deepcopy 不污染來源；`contained_sections` 由合併後的 tree 重算；warnings 聯集並加
  `small_chunk_merged`（此 warning 即「曾合併」的唯一標記，不再輸出 `merge_applied` / `merge_reason`）。
  `_merge_tier_reason` 仍回傳 tier + reason，但 reason 僅供挑選優先序、**不寫入 metadata**。

> 用 `section_key`（含 `pattern_id`，非裸 `section_id`）做合併判定，避免
> `一、`（`cjk_comma:一`）與 `（一）`（`cjk_paren:一`）碰撞。

---

## 14. Metadata 欄位（最終輸出）

每個 `Document.metadata`（`_build_metadata`，[doc_structure.py:1304](../doc_structure.py)）：

```jsonc
{
  "source_file": "境外核災處理作業要點.pdf",
  "doc_title":   "境外核災處理作業要點",     // 去副檔名（_doc_title_from_source）
  "page_start":  3,
  "page_end":    3,
  "chunk_index": 8,                          // 全部合併後才連續編號（0-based）
  "contained_sections": [                    // 本 chunk 偵測到的結構路徑（展示 / 引用用）
    "三、作業程序 > （二）一級開設 > 1. 開設時機",
    "三、作業程序 > （二）一級開設 > 2. 進駐機關"
  ],
  "structure_tree": { ... },                 // 偵測到的結構樹（見下）
  "warnings": []
}
```

> 這 **8 個欄位就是全部輸出**（含 `chunk_index`）。`chunk_index` 在
> [doc_structure.py:1579](../doc_structure.py) 於所有合併完成後才寫入，確保連續。

### 語意定義（重要）
- `contained_sections` **不再保證**都是正式章節；它是本 chunk 內「依標號規則偵測到的結構路徑」。
- `structure_tree` **不是嚴格目錄樹**；它是依 anchor level 建立的文件結構樹。
- 不要假設每個節點都是正式 heading。

### 已移除的診斷欄位（不再輸出）
為了讓 metadata 精簡，以下欄位**已不再寫入**（先前曾輸出、下游無人使用）：
`heading_tree`（structure_tree 的相容別名）、`structure_type`、`split_strategy`、
`structure_confidence`、`anchor_order_confidence`、`anchor_order_support_count`、
`merge_applied`、`merge_reason`。
- 仍需 anchor order 信心 / 支持度時：看 `build_documents_from_pages` 回傳的 `structure_info`
  （§0），那裡仍有 `anchor_order_confidence` / `anchor_order_support_count` 等診斷。
- 是否發生過小 chunk 合併：看 `warnings` 是否含 `small_chunk_merged`（取代舊的
  `merge_applied` / `merge_reason`）。
- 內部仍以 `structure_tree` 進行合併（`_get_tree` 讀 `structure_tree`），不依賴上述欄位。

### structure_tree 節點（`_make_node`，[doc_structure.py:1022](../doc_structure.py)）

```jsonc
{
  "level": 1,
  "section_key": "cjk_comma:三",   // = f"{pattern_id}:{marker.num}"，跨 pattern 不碰撞
  "section_id": "三",              // 純號碼（= marker.num）
  "pattern_id": "cjk_comma",       // 命中的規則 id
  "title": "作業程序",             // 已去尾端 ：。！？（_clean_title）
  "raw": "三、作業程序",            // 原始行（保留標點）
  "marker": { "text": "三、", "num": "三", "num_type": "cjk",
              "prefix": "", "suffix": "、" },
  "items": [ ...子節點... ]        // ★ list（保序），非 dict
}
```

- **每個 anchor 都建節點**（沒有 structural/enum 的過濾）；節點**無 `role`、無 `kind`**。
- `items` 為 **ordered list**，保留原文順序。
- **去重為 per-parent（path-aware）**：用 stack 邊建樹邊比對，只有當「同一父節點底下已有相同
  `section_key` 的兄弟」才合併（重用 `_find_child`）。**不可**用全域 `(level, section_key)` 去重——
  那會把在不同父層下**重啟編號**的清單項目誤判為重複，例如 `1./2./3.` 各自底下重複出現的
  `（1）（2）（3）`，會在第一組之後全被丟棄（只剩各父層編號唯一的尾巴如 `（4）`、`（5）`）。
- 由 `_make_anchor_path`（祖先鏈）+ `contained`（本 chunk 內 anchors）共同建成
  （`_build_structure_tree_meta`，[doc_structure.py:1086](../doc_structure.py)）；per-parent 去重同時
  正確處理 `anchor_path` 尾節點與 `contained` 首節點重疊（同 path）的情況。

### section_key（[doc_structure.py:1012](../doc_structure.py)）
```python
section_key = f"{pattern_id}:{marker['num']}"   # cjk_comma:三 / cjk_paren:二 / decimal:1
```

### contained_sections（`_contained_sections_from_tree`，[doc_structure.py:1054](../doc_structure.py)）
- 從 tree DFS，輸出每個 **leaf** 的完整 breadcrumb（root→leaf）。
- 無子節點的節點本身即 leaf（只含父節點時輸出單一 breadcrumb）。
- label 格式 `_format_section_label`（[doc_structure.py:1037](../doc_structure.py)）：CJK marker 與 title
  緊貼（`三、作業程序`）；arabic/latin marker 與 title 間加空白（`1. 開設時機`）。
- 無 anchor → `[]`。

### 標題清理 `_clean_title`（[doc_structure.py:1002](../doc_structure.py)）
去掉標題尾端的 `：。！？`（label/句末符號），讓 `1. 協助督導。` 的 title 顯示為 `協助督導`；
`raw` 仍保留完整原文。

### 來源字串 helpers
- `build_source_ref(meta)`（[doc_structure.py:1333](../doc_structure.py)）：
  `要點.pdf#p3#chunk8`（單頁）或 `要點.pdf#p3-p4#chunk8`（跨頁）。**不儲存**，動態產生。
- `format_chunk_source(meta)`（[doc_structure.py:1349](../doc_structure.py)）：RAG / 摘要引用顯示用：
  ```
  來源：境外核災處理作業要點.pdf，第 3 頁，chunk 8
  本段包含的結構路徑：
  - 三、作業程序 > （二）一級開設 > 1. 開設時機
  ```
  跨頁顯示 `第 3–4 頁`；`contained_sections` 為空時省略「本段包含的結構路徑」區塊。

### warnings
metadata 唯一的診斷欄位即 `warnings`（list）。可能值：
- `fallback_splitter_used`：該 chunk 由字元切分（`RecursiveCharacterTextSplitter`）產生。
- `low_anchor_order_support`：文件級——`anchor_order_support_count < MIN_ANCHOR_SUPPORT` 且 anchor 種類 > 1。
- `small_chunk_merged`：由 repair 合併而來（取代舊的 `merge_applied` / `merge_reason`）。

> anchor order 的 `structure_type` / `split_strategy` / 信心值 / 支持度等不再進 metadata；
> 需要時改看 `structure_info`（§0，`build_documents_from_pages` 的第二個回傳值）。

### 已淘汰欄位 / 名稱
metadata 不再輸出：`kind`、`role`、`heading_path`、`breadcrumb`、`top_breadcrumb`、
`start_heading`、`contains_sections`、`contains_headings`、`primary_section_path`、`style`、
`source_ref`，以及精簡後移除的 `heading_tree`、`structure_type`、`split_strategy`、
`structure_confidence`、`anchor_order_confidence`、`anchor_order_support_count`、
`merge_applied`、`merge_reason`。原始碼也已移除：`HEADING_PATTERNS`、`_PATTERN_BY_NAME`、`_NUM_TYPE_BY_PATTERN`、
`heading_kind()`、`_TREE_KINDS`、`_ENUM_KINDS`、`ROLE_STRUCTURAL` / `ROLE_ENUM`、
`structural_pattern_ids()` / `enum_pattern_ids()`、`DEFAULT_STYLE_ORDER`、
`infer_style_order_by_evidence`（改名 `infer_anchor_order_by_evidence`）、profile 的
`structural_order` / `enum_order` / `style_order` / `style_counts` / `cjk_style_order`。

---

## 15. 常數總表

| 常數 | 值 | 用途 |
|---|---|---|
| `MAX_ANCHORS_PER_CHUNK` | 6 | 延遲切分：區塊偏大時頂層 anchor 數超過即繼續往下切（建議 5–8） |
| `ANCHOR_SPLIT_MIN_RATIO` | 0.5 | anchor 數上限的 token 門檻；token ≤ 此比例·chunk_size 的小區塊不因 anchor 數而切 |
| `MIN_ANCHOR_SUPPORT` | 3 | 低於此父子證據量 → `low_anchor_order_support` warning |
| `MIN_MULTI_DECIMAL` | 3 | `has_decimal` 門檻（多層 decimal 數） |
| `MIN_DEPTH_SUPPORT` | 2 | 某深度要計入 `max_decimal_depth` 的最少出現次數 |
| `MAX_JUMP` | 5 | 序號可跳過的最大數；亦用於 TOC body 判定（`≤ MAX_JUMP+1`） |
| `TABLE_GAP_TOLERANCE` | 4 | 表格列之間容許的空白/續行數 |
| `TABLE_ANCHOR_LOOKBACK` | 3 | 往上找表格錨點掃描的非空行數 |
| `TABLE_ANCHORED_MIN` | 2 | 有錨點時成表的最少列數 |
| `TABLE_STANDALONE_MIN` | 5 | 無錨點時成表的最少列數 |
| repair `min_tokens` | `max(80, chunk_size//10)` | 小 chunk 合併門檻 |
| repair rounds | 3 | 合併最多輪數 |

---

## 16. 已知假設與泛化性風險（供評估）

1. **CJK 數字字典有限**：內建規則只認 `一二三四五六七八九十百`；`_cjk_to_int` 對 `廿`、`卅`、
   `千`、`萬`、阿拉伯混排無解，超過 `十X`（11–19）後較大數可能不準。新增 spec 擴充字符集時，
   `_cjk_to_int` / `_marker_value` 仍需同步處理，才能正確判斷 anchor order 的計數延續與序號連續性。
2. **anchor order 為啟發式**：方向感知證據 + 拓樸排序在目前測試集（含對抗案例與真實 PDF）正確；
   但仍依賴「下一個標號是否延續自己計數」這個局部訊號。樣本極少、或父層與子層計數行為相似的
   極端文件仍可能誤排。`anchor_order_confidence` + `anchor_order_support_count` 可作為信任度指標
   （並會觸發 `low_anchor_order_support` warning）。
3. **`MAX_ANCHORS_PER_CHUNK` 用「頂層 anchor 數」且 token-aware**：anchor 數上限只在區塊「偏大」
   （token > `ANCHOR_SPLIT_MIN_RATIO·chunk_size`）時才強制往下切；token 偏小的長清單整段保留，
   避免切碎後又被 repair 併回（曾導致 `（二）核安會` 的 `1.` 與 `2.`–`7.` 被分到兩段）。閾值
   6 與 0.5 皆為經驗值；調高 ratio → 更傾向保留大區塊，調低 → 更早因 anchor 數而切。
4. **單整數 decimal 仍可能誤收**：偵測階段不再用 `。！？` 結尾或逗號來猜「散文 vs 清單」
   （§6 規則 6 已整層移除逗號過濾），散文型 `1. 句子。`、`1.如仍須…，應先獲…` 都會被當 anchor
   （成為深層節點）。這是 anchor 模型的刻意取捨：寧可多收（多半無害——延遲切分不會因此爆切，
   只是 `contained_sections` 可能多出雜訊路徑），也不要漏掉大量含逗號的合法清單。
   仍靠前導零、表格、TOC 過濾把關。
5. **僅 decimal_chain 做序號連續性驗證**：CJK / paren_* 無 `validate_sequence`，亂序或誤判的
   CJK anchor 不會被剔除。decimal 的連續性計數現會在父層邊界重置（§11），不再跨父層外溢；
   但同一父層內若首項缺失且該段只剩 `2./3.`，仍依「首次出現 `≤ MAX_JUMP+1`」規則接受。
6. **unwrap 可能誤併**：非句末結尾的內文行會被併入前行；若 OCR 漏標點，chunk 邊界與
   anchor title 可能被汙染。
7. **TOC 偵測依賴標題詞 + dot-leader**：無「目錄」標題或無點引導的目錄頁可能漏判；
   反之正文出現 `目錄` 字樣可能誤入。
8. **token 計數 fallback**：無 tiktoken 時用 `len//4`，與實際模型 token 數差距大，
   會影響 chunk_size 判斷與 repair 門檻。
9. **page_start/end 來自行頁碼**：跨頁併行的 chunk 取 min/max；若清理階段頁碼歸屬有誤，
   來源頁碼會跟著偏。

---

## 17. 對應測試（tests/test_doc_structure.py，49 passed）

- `TestCase1/2/2b`：技術報告 decimal、純中文法規（cjk）、含單層 decimal 的法規 → profile / level
  （斷言用 `c.pattern_id`、`profile["anchor_order"]`、`parent_child_scores`）。
- `TestCase4`：`（一）` 為外層的 CJK 順序推斷。
- `TestCase5`：全 paren（A/a/數字）→ `anchor_order == [paren_upper, paren_lower, paren_num]`。
- `TestCase3`：TOC 雜訊過濾（dot-leader、`24410`、圖 caption）。
- `TestValidateSequence`：跳號、孤兒子節點、正常遞增。
- `TestContainedSections`：三情境 breadcrumb。
- `TestStructureTreeShape`：list items、`section_key`/`pattern_id`、title 去標點、
  **整棵樹無 `kind`、無 `role`**、無舊欄位。
- `TestMergeTrees`：依 `section_key` 合併保序、不污染來源、warnings 含 `small_chunk_merged`（無 `merge_applied`）。
- `TestSourceFormatting`：`build_source_ref` 單/跨頁、`format_chunk_source`（含「本段包含的結構路徑」）。
- `TestNoRoleDependency`：profile 有 `anchor_order`、無 `structural_order`/`enum_order`/`style_order`；
  candidate 無 `role` 屬性。
- `TestAllMarkersAreAnchors`：cjk_comma/cjk_paren/decimal 皆被偵測；decimal 與 paren_num 都進 tree。
- `TestDelayedSplitting`：大 chunk_size 保留整塊；小 chunk_size 先切 `（一）/（二）` 而非每個 `1.`。
- `TestStructureTreeMetadata`：`contained_sections` + `structure_tree`；並驗證 metadata 為精簡
  8 欄位集合（無 `heading_tree` / `structure_type` / `split_strategy` / `merge_*` 等診斷欄）。
- `TestNoTinyChunkFlood`：MapReduce 友善——無大量極短 chunk；合併者於 `warnings` 帶 `small_chunk_merged`。
- `TestAnchorOrderEvidence`：內層 pattern 先出現不誤導；父子證據勝過 first_seen；
  **計數重啟的內層不被誤判為外層**（對應真實 PDF 反序 bug 的迴歸測試）。
- `TestPatternRegistry`：runtime 註冊 `article_cjk`（`第十條`）可被偵測、marker 正確、可反註冊。
