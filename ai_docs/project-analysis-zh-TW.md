# load-summarize-chain 專案程式碼解析

## 1. 專案在做什麼

這個專案示範如何用 LangChain 的 `load_summarize_chain`，搭配本地 Ollama 模型（llama3.2）對長文本做摘要，並比較兩種常見策略：

- Map-Reduce：先分段摘要，再彙整成總摘要
- Refine：先產生初稿，再逐段修訂

主要資料來源是 `sample-text.txt`，輸出是 `summary-map-reduce.txt` 與 `summary-refine.txt`。

---

## 2. 專案檔案逐一說明

### 根目錄

1. `README.md`
- 專案介紹與安裝方式
- 說明 Map-Reduce / Refine 兩種摘要流程
- 提供依賴安裝指令

2. `pyproject.toml`
- Poetry 專案設定
- 定義 Python 版本（^3.12）
- 定義核心套件：langchain、langchain-ollama、tiktoken、jupyter 等

3. `requirements.txt`
- 完整鎖定版本的相依套件清單
- 用於 `pip install -r requirements.txt`

4. `sample-text.txt`
- 長篇原始文本（訪談逐字稿）
- 所有 notebook 都以此檔案為摘要輸入

5. `summary-map-reduce.txt`
- Map-Reduce 策略最終摘要結果

6. `summary-refine.txt`
- Refine 策略最終摘要結果

7. `map-reduce.ipynb`
- Map-Reduce 的教學版（含大量中間觀察）

8. `refine.ipynb`
- Refine 的教學版（含參數效果測試）

9. `map-reduce-wrap.ipynb`
- Map-Reduce 的封裝版（把流程打包在 `run(...)`）

10. `refine-wrap.ipynb`
- Refine 的封裝版（把流程打包在 `run(...)`）

11. `compare-summaries.ipynb`
- 讀取兩份摘要，轉成 HTML 並左右並排比較

12. `ai_docs/`
- 文件資料夾（本文件放在這裡）

---

## 3. Notebook 程式碼逐段解釋

以下以「你實際寫的程式區塊」來解釋每段在做什麼。

## 3.1 `map-reduce.ipynb`（教學版）

### A. 匯入與提示模板
- 匯入 `Document`、`PromptTemplate`、`load_summarize_chain`、`RecursiveCharacterTextSplitter`、`ChatOllama`、`tiktoken`
- `MAP_TEMPLATE_TXT`：給每個切塊的摘要提示（條列）
- `COMBINE_TEMPLATE_TXT`：把多個小摘要再整合成最終摘要

### B. 載入原文
- `with open("sample-text.txt")` 讀入 `transcript`

### C. Helper 函式
- `get_text_splitter(chunk_size, overlap_size)`：建立 tiktoken-aware 的切塊器
- `convert_text_to_tokens(text)`：用 tiktoken 估算 token 數

### D. Prepare docs
- 設定 `transcript_up_to = 50000`
- `chunk_size = 500`、`overlap_size = 0`
- 把文本包成 LangChain `Document`
- 切成 `split_docs`
- 印出每塊的字元數、token 數、字數，確認分塊結果

### E. 建立 LLM
- `ChatOllama(model="llama3.2", base_url="http://localhost:11434", ...)`
- 這裡調整 `num_ctx`（上下文長度）與 `num_predict`（輸出長度）

### F. Naive 跑 `load_summarize_chain`
- `chain_type = "map_reduce"`
- 先建立 map/combine 的 `PromptTemplate`
- `chain.invoke(split_docs[:12])`：先跑部分分塊做驗證

### G. Some maths
- 估算總 token 與 combine 階段可能需要的輸出上限
- 目的是理解：分塊越多，最後彙整 prompt 越大，`num_ctx` 需求越高

### H. Manual process（手動拆流程）
- 手動做 map：逐塊送入 `llm.invoke(...)`，收集 `summaries`
- 合併 `summaries` 成一段長文字
- 再做 combine：一次送進另一個 LLM
- 重新計算 `num_ctx`，確保 combine 提示塞得下

### I. 視覺化輸出
- 用 `IPython.display.HTML` + `markdown` 顯示摘要內容

重點：
- 這本 notebook 同時示範「LangChain 一鍵做」與「手動 map/combine 拆開做」
- 非常適合學習與除錯

---

## 3.2 `refine.ipynb`（教學版）

### A. 匯入與模板
- 與 map-reduce 類似，但模板換成：
  - `QUESTION_TEMPLATE_TXT`：初始摘要
  - `REFINE_TEMPLATE_TXT`：用新文本修訂既有摘要

### B. 載入原文 + Helper
- 同樣讀取 `sample-text.txt`
- 同樣提供切塊與 token 計算函式

### C. Prepare docs
- `chunk_size = 1000`（通常比 map-reduce 大）
- 切分後同樣做字元/token/字數統計

### D. LLM 參數
- `num_ctx` 預設拉到 2048
- 因 refine 每一步都帶入「已有摘要 + 新文本」，上下文通常較吃重

### E. Naive 跑 refine chain
- `chain_type = "refine"`
- 建立 `question_prompt` 與 `refine_prompt`
- `chain.invoke(split_docs[:5])` 做小規模驗證

### F. 測試 `num_predict` 效果
- 把 `num_ctx`、`num_predict` 調大，對全部 `split_docs` 執行
- 觀察摘要完整度與輸出長度變化

重點：
- Refine 的優勢在連貫性，缺點是順序式處理較慢

---

## 3.3 `map-reduce-wrap.ipynb`（封裝版）

這本把流程包成可重複呼叫的函式。

### 核心結構
1. 設定模板與參數（模型、chunk、temperature、predict）
2. helper：
- 切塊
- token 計算
- `get_larger_context_size(token_count)`：自動找最接近且大於需求的 `num_ctx`（1024 的倍數）
3. `convert_transcript_to_split_docs(...)`：統一建立分塊
4. `run(...)`：
- 先 map（迴圈跑每塊）
- 再 combine（把 map 結果再摘要）
- 依 token 自動調整 map/combine 階段的 `num_ctx`

### 為什麼這版實用
- 把教學版的手動步驟收斂成單一入口 `run(...)`
- 可直接重複套在其他文本

---

## 3.4 `refine-wrap.ipynb`（封裝版）

與 map-reduce-wrap 類似，也是做流程打包。

### 核心結構
1. 設定 `QUESTION_TEMPLATE_TXT` / `REFINE_TEMPLATE_TXT`
2. helper：切塊、token 計算、動態 context
3. `run(...)`：
- 建立 `PromptTemplate`
- 根據 `chunk_size` 與 `num_predict` 推估需要的 `num_ctx`
- 建立 `load_summarize_chain(..., chain_type="refine")`
- 回傳 `output_text`

### 與教學版差異
- 程式碼更短、更工程化
- 便於複用與批次執行

---

## 3.5 `compare-summaries.ipynb`

這本專門做「結果比對」。

流程：
1. 讀 `summary-map-reduce.txt`
2. 讀 `summary-refine.txt`
3. 用 `markdown.markdown(...)` 轉 HTML
4. 組成左右兩欄的 HTML（flex 版面）
5. `display(HTML(...))` 在 notebook 中並排顯示

用途：
- 快速比較兩種摘要策略的差異（完整性、條理、冗長度）

---

## 4. 兩種摘要策略在此專案的差異

1. Map-Reduce
- 計算模型：每塊獨立摘要，再做最終整併
- 優點：可分段、可平行化（概念上）
- 風險：合併階段可能壓縮掉細節

2. Refine
- 計算模型：在既有摘要上持續增量修訂
- 優點：上下文連續性通常較好
- 風險：完全順序依賴，速度較慢

---

## 5. 你可以如何使用這個專案

1. 學流程：先看教學版 `map-reduce.ipynb`、`refine.ipynb`
2. 跑實務：再用 `map-reduce-wrap.ipynb`、`refine-wrap.ipynb`
3. 做比較：最後打開 `compare-summaries.ipynb`
4. 改資料：把 `sample-text.txt` 換成你的長文，重跑即可

---

## 6. 小提醒（目前程式碼細節）

1. `map-reduce-wrap.ipynb` 的 `run(...)` 參數名稱是 `combine_prompt_text`，但中間又指定 `combine_prompt_text = COMBINE_TEMPLATE_TXT`，等於忽略呼叫時傳入值。
2. `refine.ipynb` 在 `PromptTemplate` 的 `input_variables` 寫法和 refine 模板佔位符有落差風險，實際使用時建議確認包含 `existing_answer`。
3. notebook 內有部分輸出顯示歷史錯誤訊息（stderr），若要正式展示，建議清除後重新執行一次。

以上不影響你理解流程，但若要做成穩定腳本，建議優先整理這三點。
