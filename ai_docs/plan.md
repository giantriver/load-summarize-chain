# Streamlit UI 計畫（貼文字直接摘要）

## 目標

做一個可用的 UI：貼入文字，按按鈕，回傳摘要，並即時顯示 Map-Reduce 過程。

---

## 範圍

1. 前端用 Streamlit。
2. 後端直接呼叫 vLLM 的 OpenAI-compatible API。
3. 單頁面，支援動態模型選擇，即時顯示 mapping / reduce 進度與中途摘要。

---

## Step 1：啟動 vLLM

```bash
docker run --rm -it --gpus all ^
  -p 8000:8000 ^
  --ipc=host ^
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 ^
  -v C:\Users\jerry\.cache\huggingface:/root/.cache/huggingface ^
  vllm/vllm-openai:latest ^
  Qwen/Qwen2.5-1.5B-Instruct ^
  --host 0.0.0.0 ^
  --port 8000 ^
  --gpu-memory-utilization 0.82 ^
  --max-model-len 2048
```

說明：
1. `Qwen2.5-1.5B-Instruct` 比 0.5B 品質更好，且比 3B 更容易在中小顯存啟動。
2. `max-model-len 2048` 讓 Reduce 階段有足夠的 context window 合併多份摘要，避免無限迴圈。
3. 你目前實際可用顯存約為 6.85/7.96 GiB，`gpu_memory_utilization` 必須低於這個比例，故先用 `0.82`。
4. 若啟動失敗（KV cache 不足），先嘗試降回 `--max-model-len 1024` 並同時降低 `--gpu-memory-utilization 0.75`；再不行就降回 `Qwen/Qwen2.5-0.5B-Instruct`。

驗證：

```bash
curl http://localhost:8000/v1/models
```

---

## Step 2：準備環境

```bash
uv sync
```

```bash
uv add streamlit
```

---

## Step 2.5：目前 Map-Reduce 程式碼流程（現況整理）

### 核心檔案

1. `map_reduce_vllm.py`：Map-Reduce 邏輯主體與 CLI。
2. `app_streamlit.py`：Streamlit UI，呼叫 `run_map_reduce()` 並顯示進度。

### map_reduce_vllm.py 流程

1. **分塊**
  - `convert_transcript_to_split_docs()` 使用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` 依 `chunk_size/overlap_size` 切分。
2. **語言偵測 + 輸出規則**
  - `detect_primary_language()`：粗略判斷中/英。
  - `get_script_rule()`：若偵測為繁中，要求輸出只用繁體。
3. **Map 步驟**
  - `MAP_TEMPLATE_TXT`：要求條列、固定語言。
  - 每個 chunk 呼叫 `try_invoke_with_backoff()`，會依 context 邊界調整 `max_tokens`。
  - `on_map_progress()` / `on_map_result()` 提供進度與中間輸出。
4. **Reduce 步驟（多輪合併）**
  - `reduce_summaries()` 用 `COMBINE_TEMPLATE_TXT` 合併。
  - 每輪依 `combine_batch_size` 分批合併，直到剩 1 筆。
  - `on_reduce_progress()` / `on_reduce_result()` 回傳進度與中間輸出。
5. **輸出修正**
  - `normalize_output_script()`（OpenCC 可用時）轉繁中。
  - 最終回傳 `output.replace("- -", "-")`。

### app_streamlit.py 流程

1. **輸入與設定**
  - `st.text_area` 貼文字。
  - Base URL：`http://localhost:8000/v1`。
  - `fetch_current_model()` 透過 `/models` 取第一個模型 ID。
  - Advanced settings：`chunk_size` / `overlap_size` / `combine_batch_size` / `map_max_tokens` / `combine_max_tokens` / `max_model_len`。
2. **Summarize 執行**
  - 建立進度條與兩個 expander（Mapping/Reduce 中間結果）。
  - 呼叫 `run_map_reduce()` 並透過 callback 更新 UI。
3. **完成狀態**
  - Mapping / Reduce 進度條顯示 done ✓ 與耗時。
  - 顯示最終摘要。

---

## Step 3：UI 腳本（app_streamlit.py）

### 輸入區
- 多行文字輸入框（貼文字）。
- vLLM Base URL 輸入欄（預設 `http://localhost:8000/v1`）。

### 模型選擇
- 自動從 vLLM `/v1/models` 取得模型清單（有快取，可手動 Reload）。
- 下拉選單選擇模型；可勾選「Use custom model name」改用手動輸入的模型名稱。

### 摘要執行
- 按下 **Summarize** 後顯示：
  1. **Mapping 進度條**：顯示 `chunk N / total`，完成後標示 done ✓。
  2. **Reducing 進度條**：顯示 `round N — M / total`，完成後標示 done ✓。
  3. **Mapping results（可折疊）**：每個 chunk 完成後即時附加該 chunk 的摘要。
  4. **Reduce results（可折疊）**：每個 reduce batch 完成後即時附加合併摘要。
- 全部完成後顯示最終摘要。

---

## Step 4：啟動 UI

```bash
uv run streamlit run app_streamlit.py
```

瀏覽器開啟後即可貼文字測試。

---

## Step 5：加入 ROUGE 評估（Map-Reduce 結束後）

### 目標

在摘要完成後，若提供參考摘要，計算 ROUGE-1 / ROUGE-2 / ROUGE-L 的 Precision / Recall / F1，並顯示在 UI。

### Scope

1. 不改動既有 Map-Reduce 流程，只在完成摘要後追加評估。
2. 支援中文文字：提供字元級 ROUGE，並預留斷詞級（可選）擴充。
3. 不強制需要參考摘要；沒有 reference 時跳過。

### 後端設計（evaluation.py）

1. 新增 `compute_rouge(candidate, reference)`，回傳 `rouge1/rouge2/rougeL` 的 `precision/recall/f1`。
2. 預設使用 **char-level**，不做 word-level（避免斷詞依賴與結果不一致）。
3. 新增 `normalize_for_rouge(text)` 做文字正規化（空白/換行統一）。
4. 設計回傳資料結構，讓 UI 可直接顯示：

```python
{
  "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
  "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
  "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
}
```

### UI 設計（app_streamlit.py）

1. 增加「Reference summary / 人工參考摘要（可選）」多行輸入框。
2. ROUGE 模式先固定為 **Character-level**（先不提供切換）。
3. Summarize 完成後：
  - 若 reference 有內容，顯示 ROUGE 表格（Precision / Recall / F1）。
  - 若無 reference，顯示提示："No reference summary provided; ROUGE skipped."。
4. 在 ROUGE 表格下加一句限制說明：
   - "ROUGE measures text overlap with the reference summary. It does not fully evaluate factual correctness, semantic equivalence, or readability."

### 依賴

1. `rouge-score`（或自寫簡化版）。
2. 先不加入 `jieba`。

---

## Step 6：加入 PaddleOCR 擷取 PDF 文字

### 目標

讓使用者可以上傳 PDF，系統先用 PaddleOCR 擷取 PDF 文字，再把擷取結果送進既有 Map-Reduce 摘要流程。圖片檔上傳先不納入本階段。

### Scope

1. 支援單一 PDF 檔案上傳。
2. PDF 文字擷取完成後，沿用目前 `run_map_reduce()` 的分塊、Map、Reduce、ROUGE 流程。
3. 初版以「可摘要」為優先，不先做版面還原、圖片輸出、表格結構化或多檔批次處理。
4. OCR 結果先以純文字/Markdown 形式顯示在 UI，讓使用者可檢查後再摘要。

### 建議方案

使用 PaddleOCR 3.x 的 `PPStructureV3`，原因是它直接支援 PDF 輸入，並可輸出 Markdown / JSON，適合後續接 LLM 摘要。若只需要逐行文字，也可以用 `PaddleOCR` 基礎 OCR，但 PDF 文件通常有段落、標題、表格與多欄版面，`PPStructureV3` 比較適合作為文件解析入口。

### 依賴

```bash
uv add "paddleocr[doc-parser]"
```

若安裝時缺少 PaddlePaddle runtime，依執行環境加裝 CPU 或 GPU 版本：

```bash
# CPU
uv add paddlepaddle

# GPU 版本需依 CUDA / 作業系統選擇 PaddlePaddle 官方對應 wheel
```

備註：
1. `paddleocr[doc-parser]` 包含文件解析能力，例如 `PPStructureV3`。
2. 第一次執行會下載模型，建議在 UI 顯示「首次載入較久」的提示。
3. 若部署在沒有 GPU 的機器，先使用 CPU 路線驗證功能，再評估效能。

### 後端設計（pdf_ocr.py）

新增 `pdf_ocr.py`，集中處理 PDF 暫存、OCR、文字整理，避免把 OCR 細節塞進 `app_streamlit.py`。

```python
from pathlib import Path
from tempfile import NamedTemporaryFile

from paddleocr import PPStructureV3


_pipeline: PPStructureV3 | None = None


def get_pdf_ocr_pipeline() -> PPStructureV3:
    global _pipeline
    if _pipeline is None:
        _pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _pipeline


def extract_pdf_text(pdf_bytes: bytes, filename: str = "upload.pdf") -> str:
    suffix = Path(filename).suffix or ".pdf"
    with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()

        pipeline = get_pdf_ocr_pipeline()
        output = pipeline.predict(input=tmp.name)

        pages: list[str] = []
        for page_index, result in enumerate(output, start=1):
            page_text = result_to_text(result)
            if page_text.strip():
                pages.append(f"# Page {page_index}\n\n{page_text.strip()}")

    return "\n\n".join(pages).strip()
```

`result_to_text(result)` 初版可採保守策略：
1. 優先使用 PaddleOCR result 物件提供的 Markdown 輸出能力（若可直接取得文字）。
2. 若 API 只方便寫檔，改用暫存資料夾呼叫 `result.save_to_markdown(...)` 後讀回 `.md`。
3. 若 Markdown 輸出不穩定，再退回 JSON 欄位，收集文字區塊並依頁面順序串接。

### UI 設計（app_streamlit.py）

1. 在原本 `st.text_area("輸入文字")` 上方加入輸入來源切換：
   - `貼上文字`
   - `上傳 PDF`
2. 選擇 `貼上文字` 時，維持目前行為。
3. 選擇 `上傳 PDF` 時：
   - 顯示 `st.file_uploader("上傳 PDF", type=["pdf"])`。
   - 按下「擷取 PDF 文字」後呼叫 `extract_pdf_text()`。
   - 把 OCR 結果存入 `st.session_state["input_text"]`，並顯示可編輯的文字區。
4. 「開始摘要」永遠讀取同一個 `input_text`，不需要改 `run_map_reduce()`。
5. OCR 失敗時顯示友善錯誤，例如：
   - PDF 無法讀取。
   - PaddleOCR 模型下載失敗。
   - 記憶體不足。

### 建議 UI 流程

1. 使用者選擇 `上傳 PDF`。
2. 上傳 PDF。
3. 點擊「擷取 PDF 文字」。
4. UI 顯示 OCR 結果文字區，使用者可微調。
5. 點擊「開始摘要」。
6. 顯示既有 Mapping / Reduce 進度與最終摘要。

### 實作步驟

1. 新增依賴：`paddleocr[doc-parser]` 與必要的 `paddlepaddle` runtime。
2. 新增 `pdf_ocr.py`：
   - `get_pdf_ocr_pipeline()`
   - `extract_pdf_text(pdf_bytes, filename)`
   - `result_to_text(result)`
3. 修改 `app_streamlit.py`：
   - 加入輸入來源切換。
   - 加入 PDF uploader。
   - 加入「擷取 PDF 文字」按鈕與 spinner。
   - OCR 結果寫入可編輯文字區。
4. 保持 `map_reduce_vllm.py` 不變，讓 OCR 只負責產生文字。
5. 加入簡單測試：
   - `extract_pdf_text()` 對空 bytes / 非 PDF 檔案要有可理解的錯誤。
   - 可用一份小型測試 PDF 驗證會回傳非空字串。

### 完成條件

1. Streamlit 可上傳 PDF。
2. 點擊「擷取 PDF 文字」後，文字區出現 PDF 內容。
3. 使用者可修改 OCR 文字後再摘要。
4. 摘要流程仍會顯示 Mapping / Reduce 進度。
5. 沒有提供 PDF 或 OCR 失敗時，UI 會顯示明確錯誤，不會直接崩潰。

### 後續可選

1. 優先讀取 PDF 既有文字層，沒有文字層時才跑 OCR，以節省時間。
2. 支援多 PDF 批次上傳並合併摘要。
3. 保留 Markdown 標題、頁碼與表格，提升摘要可讀性。
4. 加入 OCR 結果下載按鈕（txt / md）。
5. 加入頁數限制或進度顯示，避免大型 PDF 讓 UI 等太久。
6. 未來若要支援圖片，再把 `st.file_uploader` 的 type 擴充成 `["pdf", "png", "jpg", "jpeg"]`，並讓 `extract_pdf_text()` 拆成更通用的 `extract_document_text()`。

---

## 完成條件

1. 可開啟 Streamlit 頁面。
2. 貼入一段文字可成功回傳摘要。
3. 摘要過程中可即時看到 mapping / reduce 進度條與中途產生的摘要。
4. 不需 notebook、不需手動改程式碼參數。
5. 提供參考摘要時，會顯示 ROUGE-1/2/L 指標與 P/R/F1。
6. 上傳 PDF 後，可先用 PaddleOCR 擷取文字，再送入摘要流程。

---

## 後續可選

1. 加入摘要長度選項（短 / 中 / 長）。
2. 加入輸出下載按鈕（txt）。
3. 加入錯誤提示（vLLM 未啟動時顯示友善訊息）。
4. 將 PDF OCR 進一步包成背景任務，避免大型 PDF 擷取時阻塞 UI。
