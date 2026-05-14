import streamlit as st
import requests

from map_reduce_vllm import run_map_reduce

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="centered")
st.title("Text Summarizer")
st.caption("Paste text and get a summary via vLLM")


@st.cache_data(ttl=30)
def fetch_models(base_url: str) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    model_ids = [item.get("id", "") for item in data if item.get("id")]
    return sorted(model_ids)

input_text = st.text_area("Input text", height=280, placeholder="Paste your text here...")

col1, col2 = st.columns(2)
with col1:
    default_model = "Qwen/Qwen2.5-0.5B-Instruct"
with col2:
    base_url = st.text_input("Base URL", value="http://localhost:8000/v1")

left, right = st.columns([1, 1])
with left:
    refresh_models = st.button("Reload model list")
with right:
    use_custom_model = st.checkbox("Use custom model name", value=False)

if refresh_models:
    fetch_models.clear()

models: list[str] = []
model_load_error = None
try:
    models = fetch_models(base_url)
except Exception as exc:
    model_load_error = str(exc)

if model_load_error:
    st.warning(f"Cannot load models from vLLM: {model_load_error}")

if not models:
    models = [default_model]

selected_model = st.selectbox("Model", options=models, index=0)
custom_model = st.text_input("Custom model", value=selected_model, disabled=not use_custom_model)
model = custom_model.strip() if use_custom_model else selected_model

if st.button("Summarize", type="primary"):
    if not input_text.strip():
        st.warning("Please paste some text first.")
    else:
        st.subheader("Progress")
        map_label = st.empty()
        map_bar = st.progress(0)
        reduce_label = st.empty()
        reduce_bar = st.progress(0)

        with st.expander("Mapping results (intermediate)", expanded=False):
            map_results_container = st.container()

        with st.expander("Reduce results (intermediate)", expanded=False):
            reduce_results_container = st.container()

        def on_map_progress(current: int, total: int) -> None:
            map_label.markdown(f"**Mapping:** chunk {current} / {total}")
            map_bar.progress(current / total)

        def on_map_result(index: int, total: int, text: str) -> None:
            with map_results_container:
                st.markdown(f"**Chunk {index} / {total}**")
                st.markdown(text)
                st.divider()

        def on_reduce_progress(round_index: int, current: int, total: int) -> None:
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
                chunk_size=400,
                overlap_size=0,
                temperature=0.5,
                map_max_tokens=192,
                combine_max_tokens=384,
                combine_batch_size=3,
                max_model_len=1024,
                token_safety_margin=96,
                min_output_tokens=64,
                on_map_progress=on_map_progress,
                on_reduce_progress=on_reduce_progress,
                on_map_result=on_map_result,
                on_reduce_result=on_reduce_result,
            )
            map_label.markdown("**Mapping:** done ✓")
            map_bar.progress(1.0)
            reduce_label.markdown("**Reducing:** done ✓")
            reduce_bar.progress(1.0)
            st.subheader("Summary")
            st.write(summary)
        except Exception as exc:
            st.error(f"Summarization failed: {exc}")
