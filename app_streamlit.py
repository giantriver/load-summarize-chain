import time

import streamlit as st

from map_reduce_vllm import run_map_reduce

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="centered")
st.title("Text Summarizer")
st.caption("Paste text and get a summary via vLLM")


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

input_text = st.text_area("Input text", height=280, placeholder="Paste your text here...")

base_url = st.text_input("Base URL", value="http://localhost:8000/v1")

model = fetch_current_model(base_url) or "Qwen/Qwen2.5-0.5B-Instruct"
st.caption(f"Model: `{model}`")

with st.expander("Advanced settings", expanded=False):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        chunk_size = st.number_input("Chunk size (tokens)", min_value=100, max_value=2000, value=600, step=50)
    with adv_col2:
        overlap_size = st.number_input("Overlap size (tokens)", min_value=0, max_value=500, value=0, step=10)
    with adv_col3:
        combine_batch_size = st.number_input("Combine batch size", min_value=2, max_value=20, value=6, step=1)
    adv_col4, adv_col5, adv_col6 = st.columns(3)
    with adv_col4:
        map_max_tokens = st.number_input("Map max tokens", min_value=32, max_value=512, value=256, step=16)
    with adv_col5:
        combine_max_tokens = st.number_input("Combine max tokens", min_value=32, max_value=2048, value=512, step=16)
    with adv_col6:
        max_model_len = st.number_input("Max model len", min_value=256, max_value=8192, value=4096, step=256)

if st.button("Summarize", type="primary"):
    if not input_text.strip():
        st.warning("Please paste some text first.")
    else:
        st.subheader("Progress")
        map_label = st.empty()
        map_bar = st.progress(0)
        reduce_label = st.empty()
        reduce_bar = st.progress(0)

        chunks_expander = st.empty()

        with st.expander("Mapping results (intermediate)", expanded=False):
            map_results_container = st.container()

        with st.expander("Reduce results (intermediate)", expanded=False):
            reduce_results_container = st.container()

        timings: dict[str, float] = {}

        def on_chunks(chunks: list[str]) -> None:
            with chunks_expander.expander(f"Chunk contents (original) — {len(chunks)} chunks", expanded=False):
                for i, text in enumerate(chunks, start=1):
                    st.markdown(f"**Chunk {i} / {len(chunks)}**")
                    st.text(text)
                    st.divider()

        def on_map_progress(current: int, total: int) -> None:
            if "map_start" not in timings:
                timings["map_start"] = time.time()
            map_label.markdown(f"**Mapping:** chunk {current} / {total}")
            map_bar.progress(current / total)

        def on_map_result(index: int, total: int, text: str) -> None:
            with map_results_container:
                st.markdown(f"**Chunk {index} / {total}**")
                st.markdown(text)
                st.divider()

        def on_reduce_progress(round_index: int, current: int, total: int) -> None:
            if "reduce_start" not in timings:
                timings["reduce_start"] = time.time()
            reduce_label.markdown(f"**Reducing round {round_index}:** {current} / {total}")
            reduce_bar.progress(current / total)

        def on_reduce_result(round_index: int, batch_index: int, text: str) -> None:
            with reduce_results_container:
                st.markdown(f"**Round {round_index} — batch {batch_index}**")
                st.markdown(text)
                st.divider()

        try:
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
                on_map_progress=on_map_progress,
                on_reduce_progress=on_reduce_progress,
                on_map_result=on_map_result,
                on_reduce_result=on_reduce_result,
                on_chunks=on_chunks,
            )
            timings["end"] = time.time()
            map_elapsed = timings.get("reduce_start", timings["end"]) - timings.get("map_start", timings["end"])
            reduce_elapsed = timings["end"] - timings.get("reduce_start", timings["end"])
            map_label.markdown(f"**Mapping:** done ✓ ({map_elapsed:.1f}s)")
            map_bar.progress(1.0)
            reduce_label.markdown(f"**Reducing:** done ✓ ({reduce_elapsed:.1f}s)")
            reduce_bar.progress(1.0)
            st.subheader("Summary")
            st.write(summary)
        except Exception as exc:
            st.error(f"Summarization failed: {exc}")
