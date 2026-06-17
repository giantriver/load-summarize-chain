import time

import streamlit as st

from langchain_core.documents import Document

from evaluation import compute_rouge
from map_reduce_vllm import convert_transcript_to_split_docs, run_map_reduce
from pdf_ocr import PdfOcrDependencyError, PdfOcrError, extract_pdf_pages

st.set_page_config(page_title="文字摘要工具", page_icon="📝", layout="centered")
st.title("文字摘要工具")
st.caption("貼上文字或上傳 PDF，使用 vLLM 產生摘要")


def fetch_current_model(base_url: str) -> str | None:
    try:
        import requests
        url = base_url.rstrip("/") + "/models"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])
        ids = [item.get("id", "") for item in data if item.get("id")]
        return ids[0] if ids else None
    except Exception:
        return None


def format_heading_meta(heading: dict) -> str:
    style = heading.get("style")
    num = heading.get("num", "")
    title = heading.get("title", "")
    if style == "cjk_comma":
        return f"{num}、{title}"
    if style == "cjk_paren":
        return f"（{num}）{title}"
    if style in {"paren_num", "paren_upper", "paren_lower"}:
        return f"({num}) {title}"
    sep = "." if "." not in str(num) and "．" not in str(num) else ""
    return f"{num}{sep} {title}".strip()


def format_heading_path(headings: list[dict]) -> str | None:
    labels = [format_heading_meta(heading) for heading in headings if isinstance(heading, dict)]
    labels = [label for label in labels if label]
    return " > ".join(labels) if labels else None


if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "input_pages" not in st.session_state:
    st.session_state["input_pages"] = None

input_source = st.radio(
    "輸入來源",
    ["貼上文字", "上傳 PDF"],
    horizontal=True,
)

if input_source == "上傳 PDF":
    uploaded_pdf = st.file_uploader("上傳 PDF", type=["pdf"])
    st.caption("第一次使用 PaddleOCR 可能需要下載模型，會花比較久。")

    if uploaded_pdf is not None:
        if uploaded_pdf.file_id != st.session_state.get("last_pdf_id"):
            st.session_state["last_pdf_id"] = uploaded_pdf.file_id
            with st.spinner("正在擷取 PDF 文字..."):
                try:
                    extracted_pages = extract_pdf_pages(uploaded_pdf.getvalue(), uploaded_pdf.name)
                except PdfOcrDependencyError as exc:
                    st.error(str(exc))
                except PdfOcrError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"PDF 文字擷取失敗：{exc}")
                else:
                    extracted_text = "\n\n".join(extracted_pages).strip()
                    if extracted_text:
                        st.session_state["input_text"] = extracted_text
                        st.session_state["input_pages"] = extracted_pages
                        st.session_state.pop("summary", None)
                        st.success("PDF 文字擷取完成，可先檢查或修正文字後再摘要。")
                    else:
                        st.session_state["input_pages"] = None
                        st.warning("沒有從 PDF 擷取到可用文字。")
else:
    st.session_state["input_pages"] = None
    st.session_state.pop("last_pdf_id", None)

input_text = st.text_area(
    "輸入文字",
    height=280,
    placeholder="請貼上原文，或上傳 PDF 後擷取文字...",
    key="input_text",
)

chunk_only = st.toggle("只測試 chunk 切分（不執行 map-reduce）", value=False)

if not chunk_only:
    base_url = st.text_input("Base URL", value="http://localhost:8082/v1")
    model = fetch_current_model(base_url) or "local-model"
    st.caption(f"模型：`{model}`")

with st.expander("進階設定", expanded=False):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        chunk_size = st.number_input("分塊大小（tokens）", min_value=100, max_value=3000, value=1000, step=50)
    with adv_col2:
        overlap_size = st.number_input("重疊大小（tokens）", min_value=0, max_value=500, value=0, step=10)

    if not chunk_only:
        adv_col3, adv_col4, adv_col5 = st.columns(3)
        with adv_col3:
            combine_batch_size = st.number_input("合併批次大小", min_value=2, max_value=20, value=6, step=1)
        with adv_col4:
            map_max_tokens = st.number_input("Map 最多輸出 tokens", min_value=32, max_value=512, value=256, step=16)
        with adv_col5:
            combine_max_tokens = st.number_input("Reduce 最多輸出 tokens", min_value=32, max_value=2048, value=512, step=16)
        max_model_len = st.number_input("模型上下文長度", min_value=256, max_value=8192, value=4096, step=256)

run_button_label = "測試 chunk 切分" if chunk_only else "開始摘要"

if st.button(run_button_label, type="primary"):
    if not input_text.strip():
        st.warning("請先貼上原文。")
    else:
        chunks_expander = st.empty()

        def on_chunks(chunks: list[Document]) -> None:
            with chunks_expander.expander(f"原文分塊內容 — {len(chunks)} chunks", expanded=False):
                for i, doc in enumerate(chunks, start=1):
                    metadata = doc.metadata or {}

                    page_start = metadata.get("page_start")
                    page_end = metadata.get("page_end")
                    page_label = (
                        f"{page_start}-{page_end}"
                        if page_start is not None and page_end is not None and page_end != page_start
                        else str(page_start) if page_start is not None
                        else None
                    )

                    heading_path = metadata.get("heading_path") or []
                    contains_sections: list = metadata.get("contains_sections") or []
                    chapter_label = format_heading_path(heading_path) if isinstance(heading_path, list) else None
                    sections_label = "、".join(contains_sections) if len(contains_sections) > 1 else None

                    subitems_label = "是" if metadata.get("contains_subitems") else None

                    meta_parts = [
                        part
                        for part in [
                            f"頁碼：{page_label}" if page_label else None,
                            f"章節：{chapter_label}" if chapter_label else None,
                            f"包含小節：{sections_label}" if sections_label else None,
                            f"包含條列：{subitems_label}" if subitems_label else None,
                        ]
                        if part
                    ]
                    st.markdown(f"**Chunk {i} / {len(chunks)}**")
                    if meta_parts:
                        st.caption(" | ".join(meta_parts))
                    if metadata:
                        with st.expander("Metadata JSON", expanded=False):
                            st.json(metadata)
                    st.text(doc.page_content)
                    st.divider()

        try:
            source = uploaded_pdf.name if input_source == "上傳 PDF" and uploaded_pdf else "manual"

            if chunk_only:
                split_docs = convert_transcript_to_split_docs(
                    transcript=input_text,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    source=source,
                    page_texts=st.session_state.get("input_pages"),
                )
                on_chunks(split_docs)
                st.session_state.pop("summary", None)
                st.success(f"Chunk 切分完成，共 {len(split_docs)} chunks，未執行 map-reduce。")
            else:
                st.subheader("進度")
                map_label = st.empty()
                map_bar = st.progress(0)
                reduce_label = st.empty()
                reduce_bar = st.progress(0)

                with st.expander("Mapping 中間結果", expanded=False):
                    map_results_container = st.container()

                with st.expander("Reduce 中間結果", expanded=False):
                    reduce_results_container = st.container()

                timings: dict[str, float] = {}

                def on_map_progress(current: int, total: int) -> None:
                    if "map_start" not in timings:
                        timings["map_start"] = time.time()
                    map_label.markdown(f"**Mapping：** chunk {current} / {total}")
                    map_bar.progress(current / total)

                def on_map_result(index: int, total: int, text: str) -> None:
                    with map_results_container:
                        st.markdown(f"**Chunk {index} / {total}**")
                        st.markdown(text)
                        st.divider()

                def on_reduce_progress(round_index: int, current: int, total: int) -> None:
                    if "reduce_start" not in timings:
                        timings["reduce_start"] = time.time()
                    reduce_label.markdown(f"**Reducing 第 {round_index} 輪：** {current} / {total}")
                    reduce_bar.progress(current / total)

                def on_reduce_result(round_index: int, batch_index: int, text: str) -> None:
                    with reduce_results_container:
                        st.markdown(f"**第 {round_index} 輪 — batch {batch_index}**")
                        st.markdown(text)
                        st.divider()

                summary = run_map_reduce(
                    transcript=input_text,
                    model=model,
                    base_url=base_url,
                    api_key="EMPTY",
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    temperature=0.5,
                    map_max_tokens=map_max_tokens,
                    combine_max_tokens=combine_max_tokens,
                    combine_batch_size=combine_batch_size,
                    max_model_len=max_model_len,
                    token_safety_margin=96,
                    min_output_tokens=64,
                    source=source,
                    page_texts=st.session_state.get("input_pages"),
                    on_map_progress=on_map_progress,
                    on_reduce_progress=on_reduce_progress,
                    on_map_result=on_map_result,
                    on_reduce_result=on_reduce_result,
                    on_chunks=on_chunks,
                )
                timings["end"] = time.time()
                map_elapsed = timings.get("reduce_start", timings["end"]) - timings.get("map_start", timings["end"])
                reduce_elapsed = timings["end"] - timings.get("reduce_start", timings["end"])
                map_label.markdown(f"**Mapping：** 完成 ✓ ({map_elapsed:.1f}s)")
                map_bar.progress(1.0)
                reduce_label.markdown(f"**Reducing：** 完成 ✓ ({reduce_elapsed:.1f}s)")
                reduce_bar.progress(1.0)
                st.session_state["summary"] = summary
        except Exception as exc:
            st.error(f"執行失敗：{exc}")

if "summary" in st.session_state and st.session_state["summary"].strip():
    st.subheader("摘要")
    st.write(st.session_state["summary"])

    reference_text = st.text_area(
        "參考摘要（人工，可選）",
        height=160,
        placeholder="請貼上人工撰寫的參考摘要...",
        key="reference_text",
    )
    if st.button("清除參考摘要", key="clear_reference"):
        st.session_state["reference_text"] = ""
        reference_text = ""

    if reference_text.strip():
        rouge_scores = compute_rouge(st.session_state["summary"], reference_text)
        st.subheader("ROUGE 評估")
        table_rows = [
            {
                "指標": "ROUGE-1",
                "Precision": f"{rouge_scores['rouge1']['precision']:.3f}",
                "Recall": f"{rouge_scores['rouge1']['recall']:.3f}",
                "F1": f"{rouge_scores['rouge1']['f1']:.3f}",
            },
            {
                "指標": "ROUGE-2",
                "Precision": f"{rouge_scores['rouge2']['precision']:.3f}",
                "Recall": f"{rouge_scores['rouge2']['recall']:.3f}",
                "F1": f"{rouge_scores['rouge2']['f1']:.3f}",
            },
            {
                "指標": "ROUGE-L",
                "Precision": f"{rouge_scores['rougeL']['precision']:.3f}",
                "Recall": f"{rouge_scores['rougeL']['recall']:.3f}",
                "F1": f"{rouge_scores['rougeL']['f1']:.3f}",
            },
        ]
        st.table(table_rows)
        st.caption(
            "ROUGE 只衡量與參考摘要的文字重疊，"
            "無法完整評估事實正確性、語意等價或可讀性。"
        )
    else:
        st.info("可貼上人工參考摘要以計算 ROUGE。")
