from __future__ import annotations

import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

# Avoid PaddlePaddle CPU oneDNN/PIR runtime errors with some PP-OCRv5 models.
# This must be set before importing paddle / paddleocr.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"


class PdfOcrError(RuntimeError):
    pass


class PdfOcrDependencyError(PdfOcrError):
    pass


_pipeline: Any | None = None
PAGE_NUMBER_RE = re.compile(r"第\s*\d+\s*頁\s*/\s*共\s*\d+\s*頁")


def get_pdf_ocr_pipeline() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise PdfOcrDependencyError(
            'PaddleOCR 尚未安裝。請先執行 `uv add "paddleocr[doc-parser]"`，'
            "若缺少 PaddlePaddle runtime，再依環境安裝 `paddlepaddle`。"
        ) from exc

    _pipeline = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
    )
    return _pipeline


def extract_pdf_pages(pdf_bytes: bytes, filename: str = "upload.pdf") -> list[str]:
    if not pdf_bytes:
        raise PdfOcrError("PDF 檔案是空的。")
    if not filename.lower().endswith(".pdf"):
        raise PdfOcrError("目前只支援 PDF 檔案。")

    text_layer_pages = extract_pdf_text_layer_pages(pdf_bytes)
    if text_layer_pages:
        return text_layer_pages

    try:
        output = predict_pdf_pages(pdf_bytes)
    except PdfOcrError:
        raise
    except Exception as exc:
        raise PdfOcrError(f"PaddleOCR 擷取 PDF 文字失敗：{exc}") from exc

    return output


def extract_pdf_text(pdf_bytes: bytes, filename: str = "upload.pdf") -> str:
    pages = extract_pdf_pages(pdf_bytes, filename=filename)
    return "\n\n".join(pages).strip()


def extract_pdf_text_layer_pages(pdf_input: bytes | str) -> list[str]:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return []

    pages: list[str] = []
    pdf = pdfium.PdfDocument(pdf_input)
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            textpage = page.get_textpage()
            try:
                page_text = textpage.get_text_range()
            finally:
                textpage.close()
                page.close()

            pages.append(_normalize_extracted_text(page_text))
    finally:
        pdf.close()

    return pages


def extract_pdf_text_layer(pdf_input: bytes | str) -> str:
    pages = extract_pdf_text_layer_pages(pdf_input)
    return "\n\n".join(page for page in pages if page).strip()


def predict_pdf_pages(pdf_input: bytes | str) -> list[str]:
    import numpy as np
    import pypdfium2 as pdfium

    ocr = get_pdf_ocr_pipeline()
    pdf = pdfium.PdfDocument(pdf_input)
    pages: list[str] = []

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            try:
                image = page.render(scale=2).to_pil()
                result = ocr.predict(input=np.array(image.convert("RGB")))
            finally:
                page.close()

            page_text = result_to_text(result)
            pages.append(page_text)
    finally:
        pdf.close()

    return pages


def result_to_text(result: Any) -> str:
    markdown_text = _result_to_markdown_text(result)
    if markdown_text:
        return markdown_text

    data = _result_to_data(result)
    text_values = list(_iter_text_values(data))
    return _normalize_extracted_text("\n".join(text_values))


def _result_to_markdown_text(result: Any) -> str:
    in_memory_text = _get_string_from_common_keys(
        result,
        keys=("markdown", "markdown_text", "md", "text", "content"),
    )
    if in_memory_text:
        return _normalize_extracted_text(in_memory_text)

    save_to_markdown = getattr(result, "save_to_markdown", None)
    if not callable(save_to_markdown):
        return ""

    with TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir)
        try:
            save_to_markdown(save_path=str(save_path))
        except TypeError:
            save_to_markdown(str(save_path))
        except Exception:
            return ""

        markdown_files = sorted(save_path.rglob("*.md"))
        markdown_parts = [
            path.read_text(encoding="utf-8", errors="ignore").strip()
            for path in markdown_files
            if path.is_file()
        ]
        return _normalize_extracted_text("\n\n".join(part for part in markdown_parts if part))


def _result_to_data(result: Any) -> Any:
    if isinstance(result, (dict, list, tuple, str)):
        return result

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass

    json_value = getattr(result, "json", None)
    if callable(json_value):
        try:
            return json.loads(json_value())
        except Exception:
            pass

    return getattr(result, "__dict__", result)


def _iter_text_values(data: Any) -> list[str]:
    if data is None:
        return []

    if isinstance(data, str):
        return [data] if data.strip() else []

    if isinstance(data, dict):
        values: list[str] = []
        for key, value in data.items():
            if key in {"text", "rec_text", "recognized_text", "content", "markdown"}:
                if isinstance(value, str) and value.strip():
                    values.append(value)
                else:
                    values.extend(_iter_text_values(value))
            elif key in {"rec_texts", "texts"} and isinstance(value, list):
                values.extend(str(item) for item in value if str(item).strip())
            else:
                values.extend(_iter_text_values(value))
        return values

    if isinstance(data, (list, tuple)):
        values = []
        for item in data:
            values.extend(_iter_text_values(item))
        return values

    return []


def _get_string_from_common_keys(data: Any, keys: tuple[str, ...]) -> str:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    for key in keys:
        value = getattr(data, key, None)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def _normalize_extracted_text(text: str) -> str:
    text = PAGE_NUMBER_RE.sub("", text)
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    compact_lines: list[str] = []
    previous_blank = False

    for line in lines:
        if not line:
            if not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue

        compact_lines.append(line)
        previous_blank = False

    return "\n".join(compact_lines).strip()
