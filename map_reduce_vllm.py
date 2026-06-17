import argparse
import re
from collections.abc import Callable
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from tqdm import tqdm

from doc_structure import (
    HEADING_PATTERNS,
    BARE_NUMBER_RE as _DS_BARE_NUMBER_RE,
    build_documents_from_pages,
    is_heading_line as _ds_is_heading_line,
    is_standalone_section_heading as _ds_is_standalone_section_heading,
)

try:
    from opencc import OpenCC
except Exception:
    OpenCC = None

MAP_TEMPLATE_TXT = """Extract the key information from this text section.

Rules:
- Summarize ONLY the content inside <source_text>.
- Do not include, translate, or summarize these rules.
- Keep only important information.
- Remove repetition and minor details.
- Use concise bullet points.
- One idea per bullet point.
- Maximum 5 bullet points.
- If <source_text> has no useful information, output only: -

You must write the summary strictly in this language: {language_hint}.
Script rule: {script_rule}

<source_text>
{text}
</source_text>

KEY POINTS:"""

COMBINE_TEMPLATE_TXT = """Write a concise paragraph summary of the following summaries.

Rules:
- Merge overlapping ideas.
- Keep only the most important concepts.
- Remove repeated information.
- Omit examples and secondary details.
- Focus on the overall meaning instead of listing every point.
- Write only ONE short paragraph.

You must write the summary strictly in this language: {language_hint}.
Script rule: {script_rule}

Summaries:
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


def remove_prompt_echo(text: str) -> str:
    prompt_echo_markers = (
        "summarize only the content",
        "do not include, translate, or summarize",
        "keep only important information",
        "remove repetition and minor details",
        "use concise bullet points",
        "one idea per bullet point",
        "maximum 5 bullet points",
        "source_text",
        "key points",
        "只摘要",
        "不要包含",
        "不要翻譯",
        "保留重要資訊",
        "使用簡潔",
        "刪除重複",
        "重複和細節",
        "次要細節",
        "每個想法",
        "最多 5",
        "最多5",
        "五點",
        "5 點",
    )
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        normalized_line = line.strip().lstrip("-•*0123456789.、)） ").strip()
        lower_line = normalized_line.lower()
        if lower_line in {"rules", "規則", "key points"}:
            continue
        if any(marker in lower_line for marker in prompt_echo_markers):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def count_tokens(text: str) -> int:
    # cl100k_base is a practical approximation for OpenAI-compatible chat models.
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


PAGE_NUMBER_RE = re.compile(r"第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁")
# Legacy patterns kept for structural_split_text (plain-text path)
MAIN_HEADING_RE = re.compile(r"^[一二三四五六七八九十]+、.+$")
SUB_HEADING_RE = re.compile(r"^（[一二三四五六七八九十]+）.+$")
NUMBER_RE = re.compile(r"^\d+[.．、]\s*.+$")
BARE_NUMBER_RE = re.compile(r"^\d+[.．、]$")
PAREN_NUMBER_RE = re.compile(r"^（\d+）.+$")
SENTENCE_END_RE = re.compile(r"[。！？；;：:]$")


# Delegate heading detection to doc_structure so unwrap and structural split
# use the same patterns.
def is_heading_line(line: str) -> bool:
    return _ds_is_heading_line(line)


def is_standalone_section_heading(line: str) -> bool:
    return _ds_is_standalone_section_heading(line)


def should_keep_line_break(previous_line: str, next_line: str) -> bool:
    if not previous_line or not next_line:
        return True
    if is_heading_line(previous_line):
        return True
    if is_heading_line(next_line):
        return True
    return bool(SENTENCE_END_RE.search(previous_line))


def unwrap_pdf_line_breaks(text: str) -> str:
    lines = text.splitlines()
    unwrapped: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if unwrapped and unwrapped[-1]:
                unwrapped.append("")
            continue

        if not unwrapped or not unwrapped[-1]:
            unwrapped.append(stripped)
            continue

        previous = unwrapped[-1]
        if should_keep_line_break(previous, stripped):
            unwrapped.append(stripped)
        else:
            unwrapped[-1] = previous + stripped

    return "\n".join(unwrapped)


def clean_document_text(text: str) -> str:
    text = PAGE_NUMBER_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(line)
        previous_blank = False

    return unwrap_pdf_line_breaks("\n".join(cleaned_lines).strip())


def split_by_heading_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    lines = text.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if pattern.match(line.strip()) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append(current)

    return ["\n".join(section).strip() for section in sections if "\n".join(section).strip()]


def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    return any(
        pattern.match(stripped)
        for pattern in (MAIN_HEADING_RE, SUB_HEADING_RE, NUMBER_RE, BARE_NUMBER_RE, PAREN_NUMBER_RE)
    )


def should_merge_small_chunk(chunk: str, min_tokens: int = 80) -> bool:
    stripped = chunk.strip()
    if not stripped:
        return False
    if count_tokens(stripped) >= min_tokens:
        return False

    lines = [line for line in stripped.splitlines() if line.strip()]
    if lines and all(is_heading_line(line) for line in lines):
        return True
    if len(lines) == 1 and BARE_NUMBER_RE.match(lines[0].strip()):
        return True
    non_heading_text = "".join(line.strip() for line in lines if not is_heading_line(line))
    if any(is_heading_line(line) for line in lines) and non_heading_text in {"", "。", "，", "、"}:
        return True
    if stripped in {"。", "，", "、"}:
        return True
    return stripped[-1] in {"之", "及", "與", "和", "或", "、", "（", "，", "；", "："}


def repair_small_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    repaired: list[str] = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i].strip()
        if not chunk:
            i += 1
            continue

        if should_merge_small_chunk(chunk) and i + 1 < len(chunks):
            merged = chunk + "\n" + chunks[i + 1].strip()
            if count_tokens(merged) <= chunk_size:
                repaired.append(merged)
                i += 2
                continue

        if should_merge_small_chunk(chunk) and repaired:
            merged = repaired[-1] + "\n" + chunk
            if count_tokens(merged) <= chunk_size:
                repaired[-1] = merged
                i += 1
                continue

        repaired.append(chunk)
        i += 1

    return repaired


def structural_split_text(
    text: str,
    chunk_size: int,
    overlap_size: int,
    clean_text: bool = True,
) -> list[str]:
    fallback_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    patterns = [MAIN_HEADING_RE, SUB_HEADING_RE, NUMBER_RE, PAREN_NUMBER_RE]

    def split_with_fallback(section: str, pattern_index: int = 0) -> list[str]:
        if count_tokens(section) <= chunk_size:
            return [section]

        if pattern_index < len(patterns):
            pattern = patterns[pattern_index]
            parts = split_by_heading_pattern(section, pattern)
            if len(parts) > 1:
                first_line = parts[0].splitlines()[0].strip()
                if pattern_index > 0 and not pattern.match(first_line):
                    parts = [parts[0] + "\n" + parts[1], *parts[2:]]

                chunks: list[str] = []
                for part in parts:
                    chunks.extend(split_with_fallback(part, pattern_index + 1))
                return chunks

        return fallback_splitter.split_text(section)

    split_text = clean_document_text(text) if clean_text else text.strip()
    chunks = split_with_fallback(split_text)
    for _ in range(3):
        repaired = repair_small_chunks(chunks, chunk_size)
        if repaired == chunks:
            break
        chunks = repaired
    return [chunk for chunk in chunks if chunk.strip()]


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


def _build_documents_from_page_texts(
    page_texts: list[str],
    chunk_size: int,
    overlap_size: int,
    source: str = "",
) -> list[Document]:
    docs, structure_info = build_documents_from_pages(
        pages=page_texts,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        source_file=source,
    )
    excluded = structure_info.get("toc_pages_excluded", [])
    if excluded:
        print(f"[doc_structure] 已排除 {len(excluded)} 頁目錄（頁碼：{excluded}）")
    preview = structure_info.get("preview_headings", [])
    if preview:
        print(f"[doc_structure] 結構偵測前 {len(preview)} 個 heading：")
        for line in preview:
            print(f"  {line}")
    return docs


def convert_transcript_to_split_docs(
    transcript: str,
    chunk_size: int,
    overlap_size: int,
    source: str = "input",
    page_texts: list[str] | None = None,
) -> list[Document]:
    if page_texts:
        return _build_documents_from_page_texts(
            page_texts=page_texts,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            source=source,
        )

    docs, _ = build_documents_from_pages(
        pages=[transcript],
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        source_file=source,
    )
    return docs


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

            # A lone item at the end of a round cannot be merged; carry it forward as-is.
            if local_batch_size == 1:
                next_round.append(current_round[i])
                i += 1
                pbar.update(1)
                if on_reduce_progress:
                    on_reduce_progress(round_index, i, len(current_round))
                if on_reduce_result:
                    on_reduce_result(round_index, len(next_round), current_round[i - 1])
                continue

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
    source: str = "input",
    page_texts: list[str] | None = None,
    on_map_progress: Callable[[int, int], None] | None = None,
    on_reduce_progress: Callable[[int, int, int], None] | None = None,
    on_map_result: Callable[[int, int, str], None] | None = None,
    on_reduce_result: Callable[[int, int, str], None] | None = None,
    on_chunks: Callable[[list[Document]], None] | None = None,
) -> str:
    split_docs = convert_transcript_to_split_docs(
        transcript=transcript,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        source=source,
        page_texts=page_texts,
    )

    print(f"Total chunks: {len(split_docs)}")

    if on_chunks:
        on_chunks(split_docs)

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
        mapped_output = remove_prompt_echo(mapped_output)
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
    parser.add_argument("--model", default="local-model", help="Model name served by llama.cpp server.")
    parser.add_argument("--base-url", default="http://localhost:8082/v1", help="llama.cpp OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for endpoint auth.")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size in tokens for splitting.")
    parser.add_argument("--overlap-size", type=int, default=0, help="Chunk overlap in tokens.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--map-max-tokens", type=int, default=256, help="Max output tokens per map step.")
    parser.add_argument("--combine-max-tokens", type=int, default=512, help="Max output tokens per reduce step.")
    parser.add_argument("--combine-batch-size", type=int, default=6, help="Summaries to combine per reduce batch.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Model context window size from vLLM.")
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
