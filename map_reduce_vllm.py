import argparse
from collections.abc import Callable
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from tqdm import tqdm

try:
    from opencc import OpenCC
except Exception:
    OpenCC = None

MAP_TEMPLATE_TXT = """Write a detailed summary of this text section in bullet points.
Use '-' for bullet points and answer only the bullet points.
You must write the summary strictly in this language: {language_hint}.
Do not switch languages unless the input itself is mixed-language content.
Script rule: {script_rule}
Text:
{text}

SUMMARY:"""

COMBINE_TEMPLATE_TXT = """Combine these summaries into a final summary in bullet points.
Use '-' for bullet points and answer only the bullet points.
You must write the final summary strictly in this language: {language_hint}.
Use the source text language preference, not the intermediate summary language.
Script rule: {script_rule}
Text:
{text}

FINAL SUMMARY:"""


def detect_primary_language(text: str) -> str:
    # Minimal heuristic: if there are enough CJK chars, treat as Chinese.
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "Traditional Chinese" if cjk_count >= 20 else "English"


def get_script_rule(language_hint: str) -> str:
    if language_hint == "Traditional Chinese":
        return "If output is Chinese, use Traditional Chinese only. Do not use Simplified Chinese characters."
    return "Use standard spelling and punctuation for the selected language."


def normalize_output_script(text: str, language_hint: str) -> str:
    if language_hint != "Traditional Chinese":
        return text
    if OpenCC is None:
        return text
    converter = OpenCC("s2t")
    return converter.convert(text)


def count_tokens(text: str) -> int:
    # cl100k_base is a practical approximation for OpenAI-compatible chat models.
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def is_context_overflow_error(exc: Exception) -> bool:
    if not isinstance(exc, BadRequestError):
        return False
    message = str(exc).lower()
    return "maximum context length" in message and "requested" in message


def try_invoke_with_backoff(
    prompt_text: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    start_max_tokens: int,
    min_output_tokens: int,
) -> str | None:
    max_tokens = max(start_max_tokens, min_output_tokens)

    while max_tokens >= min_output_tokens:
        llm = build_llm(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            response = llm.invoke(prompt_text)
            return response.content.strip()
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise
            max_tokens -= 16

    return None


def get_text_splitter(chunk_size: int, overlap_size: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )


def convert_transcript_to_split_docs(
    transcript: str,
    chunk_size: int,
    overlap_size: int,
) -> list[Document]:
    docs = [Document(page_content=transcript)] # 把 transcript 包裝成 LangChain 能理解的 Document 格式
    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    return text_splitter.split_documents(docs)


def build_llm(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def reduce_summaries(
    summaries: list[str],
    combine_prompt: PromptTemplate,
    language_hint: str,
    script_rule: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    combine_max_tokens: int,
    combine_batch_size: int,
    max_model_len: int,
    token_safety_margin: int,
    min_output_tokens: int,
    on_reduce_progress: Callable[[int, int, int], None] | None = None,
    on_reduce_result: Callable[[int, int, str], None] | None = None,
) -> str:
    current_round = [s for s in summaries if s.strip()]
    if not current_round:
        return ""

    round_index = 1
    while len(current_round) > 1:
        next_round: list[str] = []
        i = 0
        pbar = tqdm(total=len(current_round), desc=f"Reducing round {round_index}")

        while i < len(current_round):
            local_batch_size = min(combine_batch_size, len(current_round) - i)
            reduced = False

            while local_batch_size >= 2:
                group = current_round[i : i + local_batch_size]
                combined_text = "\n".join(group)
                full_prompt = combine_prompt.format_prompt(
                    text=combined_text,
                    language_hint=language_hint,
                    script_rule=script_rule,
                )
                input_tokens = count_tokens(full_prompt.text)
                allowed_output = max_model_len - input_tokens - token_safety_margin

                if allowed_output < min_output_tokens:
                    local_batch_size -= 1
                    continue

                output_tokens = min(combine_max_tokens, allowed_output)
                reduced_output = try_invoke_with_backoff(
                    prompt_text=full_prompt.text,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    start_max_tokens=output_tokens,
                    min_output_tokens=min_output_tokens,
                )
                if reduced_output is None:
                    local_batch_size -= 1
                    continue

                reduced_output = normalize_output_script(reduced_output, language_hint)
                next_round.append(reduced_output)
                i += local_batch_size
                pbar.update(local_batch_size)
                if on_reduce_progress:
                    on_reduce_progress(round_index, i, len(current_round))
                if on_reduce_result:
                    on_reduce_result(round_index, len(next_round), reduced_output)
                reduced = True
                break

            if not reduced:
                pbar.close()
                raise RuntimeError(
                    "Reduce step cannot fit context window. "
                    "Try lowering --chunk-size / --map-max-tokens, or increasing vLLM --max-model-len."
                )

        pbar.close()
        print(f"Reduce round {round_index}: {len(current_round)} -> {len(next_round)}")
        current_round = next_round
        round_index += 1

    return current_round[0]


def run_map_reduce(
    transcript: str,
    model: str,
    base_url: str,
    api_key: str,
    chunk_size: int,
    overlap_size: int,
    temperature: float,
    map_max_tokens: int,
    combine_max_tokens: int,
    combine_batch_size: int,
    max_model_len: int,
    token_safety_margin: int,
    min_output_tokens: int,
    on_map_progress: Callable[[int, int], None] | None = None,
    on_reduce_progress: Callable[[int, int, int], None] | None = None,
    on_map_result: Callable[[int, int, str], None] | None = None,
    on_reduce_result: Callable[[int, int, str], None] | None = None,
) -> str:
    split_docs = convert_transcript_to_split_docs(
        transcript=transcript,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )

    print(f"Total chunks: {len(split_docs)}")

    language_hint = detect_primary_language(transcript)
    script_rule = get_script_rule(language_hint)

    map_prompt = PromptTemplate(template=MAP_TEMPLATE_TXT, input_variables=["text", "language_hint", "script_rule"])
    combine_prompt = PromptTemplate(
        template=COMBINE_TEMPLATE_TXT,
        input_variables=["text", "language_hint", "script_rule"],
    )

    summaries: list[str] = []
    for split_doc in tqdm(split_docs, desc="Mapping"):
        full_prompt = map_prompt.format_prompt(
            text=split_doc.page_content,
            language_hint=language_hint,
            script_rule=script_rule,
        )
        input_tokens = count_tokens(full_prompt.text)
        allowed_output = max_model_len - input_tokens - token_safety_margin
        if allowed_output < min_output_tokens:
            raise RuntimeError(
                "Map step cannot fit context window. "
                "Try lowering --chunk-size or increasing vLLM --max-model-len."
            )

        map_output_tokens = min(map_max_tokens, allowed_output)
        mapped_output = try_invoke_with_backoff(
            prompt_text=full_prompt.text,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            start_max_tokens=map_output_tokens,
            min_output_tokens=min_output_tokens,
        )
        if mapped_output is None:
            raise RuntimeError(
                "Map step cannot fit context window even after token backoff. "
                "Try lowering --chunk-size or increasing vLLM --max-model-len."
            )

        mapped_output = normalize_output_script(mapped_output, language_hint)
        summaries.append(mapped_output)
        if on_map_progress:
            on_map_progress(len(summaries), len(split_docs))
        if on_map_result:
            on_map_result(len(summaries), len(split_docs), mapped_output)

    output = reduce_summaries(
        summaries=summaries,
        combine_prompt=combine_prompt,
        language_hint=language_hint,
        script_rule=script_rule,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        combine_max_tokens=combine_max_tokens,
        combine_batch_size=combine_batch_size,
        max_model_len=max_model_len,
        token_safety_margin=token_safety_margin,
        min_output_tokens=min_output_tokens,
        on_reduce_progress=on_reduce_progress,
        on_reduce_result=on_reduce_result,
    )

    output = normalize_output_script(output, language_hint)
    return output.replace("- -", "-").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map-reduce summarization using vLLM OpenAI-compatible API.")
    parser.add_argument("--input", type=Path, default=Path("sample-text.txt"), help="Input text file path.")
    parser.add_argument("--output", type=Path, default=Path("summary-map-reduce.txt"), help="Output summary file path.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name served by vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for endpoint auth.")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size in tokens for splitting.")
    parser.add_argument("--overlap-size", type=int, default=0, help="Chunk overlap in tokens.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--map-max-tokens", type=int, default=256, help="Max output tokens per map step.")
    parser.add_argument("--combine-max-tokens", type=int, default=512, help="Max output tokens per reduce step.")
    parser.add_argument("--combine-batch-size", type=int, default=6, help="Summaries to combine per reduce batch.")
    parser.add_argument("--max-model-len", type=int, default=1024, help="Model context window size from vLLM.")
    parser.add_argument("--token-safety-margin", type=int, default=96, help="Reserved tokens to avoid edge overflow.")
    parser.add_argument("--min-output-tokens", type=int, default=64, help="Minimum output tokens per generation call.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transcript = args.input.read_text(encoding="utf-8")
    summary = run_map_reduce(
        transcript=transcript,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        temperature=args.temperature,
        map_max_tokens=args.map_max_tokens,
        combine_max_tokens=args.combine_max_tokens,
        combine_batch_size=args.combine_batch_size,
        max_model_len=args.max_model_len,
        token_safety_margin=args.token_safety_margin,
        min_output_tokens=args.min_output_tokens,
    )

    args.output.write_text(summary + "\n", encoding="utf-8")
    print(f"Summary written to: {args.output}")


if __name__ == "__main__":
    main()
