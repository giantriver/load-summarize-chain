# 文件切分邏輯說明

本文記錄目前系統從 PDF / 文字輸入到 Map-Reduce 摘要前的切分流程。

---

## 目標

切分策略以「文件結構優先，token 長度保護」為原則：

1. 先清理 PDF 頁碼等雜訊。
2. 將 PDF 每頁文字合併成連續文件後再切分。
3. 盡量修復 PDF/OCR 造成的硬換行。
4. 優先依中文法規常見章節標題切分。
5. 若章節仍超過 token 上限，再依下一層條號切分。
6. 若仍超過 token 上限，才 fallback 到 `RecursiveCharacterTextSplitter`。
7. 最後修補過短 chunk，避免標題或斷句片段獨立成無意義 chunk。

---

## PDF 文字前處理

位置：`pdf_ocr.py`

### 擷取流程

1. 使用 `pypdfium2` 優先讀取 PDF 內建文字層。
2. 若沒有文字層，才將 PDF 頁面轉成圖片並使用 PaddleOCR。
3. PaddleOCR 使用較輕量的 mobile 模型：
   - `PP-OCRv5_mobile_det`
   - `PP-OCRv5_mobile_rec`
4. 為避開 PaddlePaddle CPU oneDNN / PIR runtime 問題，於 import PaddleOCR 前設定：

```python
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
```

### 頁碼清理

PDF 常見頁首 / 頁尾：

```text
第1頁/共7頁
第 2 頁 / 共 7 頁
```

目前用以下 regex 移除：

```python
PAGE_NUMBER_RE = re.compile(r"第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁")
```

清理發生在 `_normalize_extracted_text()`，因此不論文字來自 PDF 文字層或 OCR 結果，都會套用。

---

## 摘要前文字清理

位置：`doc_structure.py`

`clean_document_text()` 會在切分前再次處理文字：

1. 移除 `第N頁/共M頁`。
2. 統一換行：`\r\n` / `\r` 轉成 `\n`。
3. 去除每行頭尾空白。
4. 合併連續空白行，避免分隔符造成過多空 chunk。
5. 呼叫 `unwrap_pdf_line_breaks()`，盡量還原 PDF/OCR 的硬換行。

這層清理是保險：即使文字不是從 `pdf_ocr.py` 來，也能清掉常見頁碼。

### PDF 硬換行還原

PDF 文字層或 OCR 常把同一句切成多行：

```text
為提供國內相關機關
啟動境外核災應變之作業程序
```

目前會在下列情況保留換行：

1. 前一行或下一行為空白行。
2. 前一行是獨立章節標題，例如 `一、目的`、短的 `（一）二級開設`。
3. 下一行是章節或條列標題。
4. 前一行以句尾標點結束：

```python
SENTENCE_END_RE = re.compile(r"[。！？；;：:]$")
```

其他情況會把兩行直接接在一起，降低這類斷行對切分和摘要的干擾：

```text
災害防救辦
公室
```

會盡量還原成：

```text
災害防救辦公室
```

---

## 結構切分規則

位置：`map_reduce_vllm.py`

目前支援多種中文與數字條號：

```python
HEADING_PATTERNS = [
    ("decimal", ...),      # 1. / 1.1 / 1.1.1
    ("paren_num", ...),    # (1) / （1）
    ("paren_upper", ...),  # (A) / （A）
    ("paren_lower", ...),  # (a) / （a）
    ("cjk_comma", ...),    # 一、目的
    ("cjk_paren", ...),    # （一）二級開設
]
BARE_NUMBER_RE = re.compile(r"^\d+[.．、]$")
```

對應例子：

```text
一、目的
二、適用時機
三、作業程序

（一）二級開設
（二）一級開設

1. 開設時機
2. 參與機關

（1）境外部分
（2）邊境部分
```

---

## 切分流程

入口：`convert_transcript_to_split_docs()`

### PDF 輸入

Streamlit 上傳 PDF 後會同時保存兩份內容：

1. `input_text`：所有頁面合併後顯示在 UI 的文字。
2. `input_pages`：每頁文字清單，用來回推 chunk 對應頁碼。

摘要或 chunk 測試時，若有 `page_texts`，目前流程是：

```text
page_texts
↓
build_line_records() 建立逐行資料與頁碼
↓
clean_line_records() 清理頁碼、空白與 PDF 硬換行
↓
detect_heading_candidates() 偵測標題候選
↓
排除目錄與雜訊標題，推斷 heading level
↓
split_by_candidate_lines() 依文件結構與 chunk_size 切分
↓
repair_small_chunks_docs() 修補過短 chunk
↓
建立 page_start / page_end、heading_path、contains_sections 等 metadata
↓
轉成 LangChain Document
```

重點：PDF 不再逐頁各自切分，而是以整份文件的行序與頁碼資訊做結構切分。頁碼只作為 metadata，不作為 chunk 邊界。

### 貼上文字輸入

沒有 `page_texts` 時，流程是：

```text
transcript
↓
包成單頁 pages=[transcript]
↓
doc_structure.build_documents_from_pages()
↓
轉成 LangChain Document
```

因此貼上文字與 PDF 預覽會使用同一套 metadata schema；差異是貼上文字會被視為單頁輸入。

### 結構切分

實際流程：

```text
doc_structure.build_documents_from_pages()
↓
依標題候選推斷層級，例如：一、二、三、（一）、1.、（1）
↓
優先在較高層級標題切分
↓
若單一段落超過 chunk_size，再往較低層級切
↓
若仍超過 chunk_size，fallback 到 RecursiveCharacterTextSplitter
↓
repair_small_chunks_docs()
↓
轉成 LangChain Document
```

系統會先從推斷出的較高層級標題切。只有當某一段超過 `chunk_size` 時，才往下一層切。
各種 heading style 會依文件中出現順序與包覆模式推斷父子層級；常見 `一、` 外層格式也是由證據判定，不是固定規則。只有證據平手時才使用預設 style 順序作為 tie-break。

例如 `chunk_size=1000` 時：

```text
三、作業程序
（一）二級開設
1. 開設時機
2. 參與機關
3. 直轄市、縣（市）政府
```

如果整個 `（一）二級開設` 未超過 1000 tokens，就會保留為一個 chunk，不會再切到 `1. 2. 3.`。

若 `chunk_size=300`，單一條列項可能超過上限，此時會繼續往下一層或 fallback，可能出現語意不理想的硬切。

---

## Fallback Token Splitter

當結構切分後某段仍超過 token 上限，會使用：

```python
RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size,
    chunk_overlap=overlap_size,
)
```

token 計算使用：

```python
tiktoken.get_encoding("cl100k_base")
```

注意：

1. `chunk_size` 是上限，不是每個 chunk 的固定大小。
2. fallback splitter 會依分隔符遞迴切分，可能產生小於 `chunk_size` 的 chunk。
3. 若某段文字沒有明顯分隔符，最後會以 token / 字元保護方式切開。

---

## 短 Chunk 修補

位置：`doc_structure.py` 的 `repair_small_chunks_docs()`

目的：避免以下內容獨立成 chunk：

```text
三、作業程序
（一）二級開設
```

或斷句尾巴：

```text
由核安會通知事故可能影響地區之
```

### 判斷條件

`repair_small_chunks_docs()` 會以 `max(80, chunk_size // 10)` 作為過短 chunk 門檻。

會被視為應合併的小 chunk：

1. 只有標題行。
2. 只有標點。
3. 包含標題，但非標題內容只有少量標點。
4. 結尾是常見未完成語氣或連接符：
5. 單獨的數字條列標記，例如 `2.`、`2、`。

```python
{"之", "及", "與", "和", "或", "、", "（", "，", "；", "："}
```

### 合併方向

修補邏輯會優先嘗試：

1. 合併到下一個 chunk。
2. 若不能合併到下一個 chunk，再嘗試合併到前一個 chunk。
3. 合併後不得超過 `chunk_size`。
4. 最多執行 3 輪，處理連續短 chunk。
5. 不合併不同第一層章節的 chunk，避免 `二、適用時機` 與 `三、作業程序` 這類相鄰主章被併在一起。

另外，切分階段會避免產生只有父標題的獨立 chunk。若某段只有目前父標題，例如：

```text
三、作業程序
```

且下一段是它的第一個子節，會直接把父標題併入該子節：

```text
三、作業程序
（一）二級開設
...
```

---

## Streamlit 預設值

位置：`app_streamlit.py`

目前 UI 預設：

```text
chunk_size = 1000
chunk_size max = 3000
overlap_size = 0
combine_batch_size = 6
map_max_tokens = 256
combine_max_tokens = 512
max_model_len = 4096
```

UI 另有 `只測試 chunk 切分（不執行 map-reduce）` toggle。開啟時只會執行 `convert_transcript_to_split_docs()` 並顯示 chunks，不會查詢 vLLM model，也不會執行 map-reduce。

對法規 / 作業要點類文件，建議：

```text
chunk_size = 800 ~ 1200
overlap_size = 0 ~ 50
```

目前預設使用：

```text
chunk_size = 1000
overlap_size = 0
```

原因：

1. 法規文件章節邊界清楚，不需要大量 overlap。
2. overlap 太大容易讓條文摘要重複。
3. `1000 tokens` 通常可容納完整小節，又不會讓 Map prompt 太長。

---

## 已知限制

1. 目前 heading regex 主要針對中文法規 / 條列文件。
2. PDF 硬換行還原是啟發式規則，可能誤接或漏接少數換行。
3. 若章節內容極長且沒有子標題，仍會 fallback 到一般 token splitter。
4. 頁碼清理目前只處理 `第N頁/共M頁` 格式。
5. 文件若使用特殊條號，例如 `壹、貳、參` 或 `A. B. C.`，目前不會作為優先結構切分點。
6. 當 `chunk_size` 太小時，fallback splitter 可能把長句或長條列硬切開，例如把 `核子事故或核彈爆炸事故` 切成兩個 chunk。

---

## Metadata 保存方式

目前 `Document.metadata` 保存來源、頁碼範圍、結構化標題路徑、包含的小節與條列狀態。

單頁 chunk：

```python
{
    "source_file": "境外核災處理作業要點.pdf",
    "doc_title": "境外核災處理作業要點",
    "page_start": 1,
    "page_end": 1,
    "heading_path": [
        {"level": 1, "style": "cjk_comma", "num": "三", "title": "作業程序"},
        {"level": 2, "style": "cjk_paren", "num": "一", "title": "二級開設"}
    ],
    "contains_sections": ["（一）二級開設"],
    "contains_subitems": True,
    "chunk_index": 3
}
```

跨頁 chunk：

```python
{
    "source_file": "境外核災處理作業要點.pdf",
    "doc_title": "境外核災處理作業要點",
    "page_start": 1,
    "page_end": 2,
    "heading_path": [
        {"level": 1, "style": "cjk_comma", "num": "三", "title": "作業程序"},
        {"level": 2, "style": "cjk_paren", "num": "一", "title": "二級開設"}
    ],
    "contains_sections": ["（一）二級開設"],
    "contains_subitems": True,
    "chunk_index": 3
}
```

Streamlit 顯示時會把跨頁格式顯示為：

```text
頁碼：1-2
```

Streamlit chunk preview 會顯示：

```text
頁碼：6-7 | 章節：五、其他
```

其中 `章節` 由 `heading_path` 即時格式化產生。完整 metadata 會放在每個 chunk 下方的 `Metadata JSON` 折疊區，方便檢查 `heading_path`、`contains_sections`、`page_start/page_end` 等欄位。

### 章節欄位

章節 metadata 主要使用 `heading_path`：

```python
[
    {"level": 1, "style": "cjk_comma", "num": "三", "title": "作業程序"},
    {"level": 2, "style": "cjk_paren", "num": "一", "title": "二級開設"}
]
```

`heading_path` 是 chunk 起始位置所在的完整標題階層，保留 `level`、`style`、`num`、`title`，UI 顯示時才格式化為：

```text
三、作業程序 > （一）二級開設
```

`contains_sections` 則記錄這個 chunk 內實際涵蓋的小節：

```python
{
    "heading_path": [
        {"level": 1, "style": "cjk_comma", "num": "五", "title": "其他"}
    ],
    "contains_sections": [
        "（一）...",
        "（二）...",
        "（三）...",
        "（四）..."
    ]
}
```

`contains_subitems` 是布林值，表示 chunk 內是否包含 `1. 2. 3.` 或 `（1）（2）` 這類更細條列。這些細項不另存路徑，避免 metadata 過度膨脹。

例如一個 chunk 包含整個 `五、其他`：

```text
五、其他
（一）...
（二）...
（三）...
（四）...
```

即使其中有多個 `（一）（二）（三）（四）`，它的主要章節也應該是：

```python
{
    "heading_path": [
        {"level": 1, "style": "cjk_comma", "num": "五", "title": "其他"}
    ],
    "contains_sections": ["（一）...", "（二）...", "（三）...", "（四）..."]
}
```

例如前一個 chunk 已讀到：

```text
三、作業程序
（一）二級開設
```

下一個 chunk 即使只有：

```text
3. 直轄市、縣（市）政府：...
```

仍會繼承：

```python
{
    "heading_path": [
        {"level": 1, "style": "cjk_comma", "num": "三", "title": "作業程序"},
        {"level": 2, "style": "cjk_paren", "num": "一", "title": "二級開設"}
    ]
}
```

貼上文字輸入沒有 PDF 頁碼，但仍可能有章節 metadata。若文字中也沒有可辨識章節，metadata 可能是空 dict：

```python
{}
```

---

## 後續可選改進

1. 支援 `壹、貳、參`、`甲、乙、丙`、`A. B. C.` 等更多條號格式。
2. 將章節 metadata 顯示做成 UI toggle，必要時可只顯示頁碼。
3. 加入 chunk preview 指標：每個 chunk 的 token 數、來源切分規則。
4. 改善 fallback splitter，避免把條號 `2.` 和後續文字拆開，或切在 `或 / 及 / 與 / 之` 之後。
5. 將結構切分策略抽成獨立模組，方便測試與替換。
