import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pdf_ocr import extract_pdf_pages
from map_reduce_vllm import convert_transcript_to_split_docs

PDF = "境外核災處理作業要點.pdf"

with open(PDF, "rb") as f:
    data = f.read()

pages = extract_pdf_pages(data, PDF)
print(f"# pages extracted: {len(pages)}")
full = "\n\n".join(pages)
print(f"# total chars: {len(full)}")
print("=" * 70)
print("RAW PAGE PREVIEW (first 1500 chars):")
print(full[:1500])
print("=" * 70)

docs = convert_transcript_to_split_docs(
    transcript=full,
    chunk_size=1000,
    overlap_size=0,
    source=PDF,
    page_texts=pages,
)

print(f"\n# CHUNKS: {len(docs)}\n")

def node_label(node):
    m = node.get("marker") or {}
    return f"{m.get('text','') } {node.get('title','')}".strip()

def walk(tree, depth=0, out=None):
    if out is None:
        out = []
    if not tree:
        return out
    out.append("  " * depth + f"[{tree.get('section_id')}] {node_label(tree)}")
    for c in (tree.get("items") or {}).values():
        walk(c, depth + 1, out)
    return out

for i, d in enumerate(docs):
    md = d.metadata
    text = d.page_content
    print(f"\n{'#'*70}")
    print(f"CHUNK {i}  | pages {md.get('page_start')}-{md.get('page_end')} | chars={len(text)} | content_type={md.get('content_type')}")
    print("heading_tree:")
    for line in walk(md.get("heading_tree")):
        print("   " + line)
    print("-" * 40 + " TEXT:")
    print(text[:700] + ("..." if len(text) > 700 else ""))
