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
  --max-model-len 1024
```

說明：
1. `Qwen2.5-1.5B-Instruct` 比 0.5B 品質更好，且比 3B 更容易在中小顯存啟動。
2. 你目前遇到的是 KV cache 記憶體不足，`max-model-len 2048` 在此環境太吃記憶體。
3. 你目前實際可用顯存約為 6.85/7.96 GiB，`gpu_memory_utilization` 必須低於這個比例，故先用 `0.82`。
4. 若仍啟動失敗，先關閉其他占用 GPU 的程式；再不行就降回 `Qwen/Qwen2.5-0.5B-Instruct`。

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

## 完成條件

1. 可開啟 Streamlit 頁面。
2. 貼入一段文字可成功回傳摘要。
3. 摘要過程中可即時看到 mapping / reduce 進度條與中途產生的摘要。
4. 不需 notebook、不需手動改程式碼參數。

---

## 後續可選

1. 加入摘要長度選項（短 / 中 / 長）。
2. 加入輸出下載按鈕（txt）。
3. 加入錯誤提示（vLLM 未啟動時顯示友善訊息）。
