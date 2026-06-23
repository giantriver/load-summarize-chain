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
docs = convert_transcript_to_split_docs(
    transcript="\n\n".join(pages), chunk_size=1000, overlap_size=0,
    source=PDF, page_texts=pages)

print("FULL METADATA of CHUNK 10 (cjk_comma > cjk_paren > decimal):")
print(json.dumps(docs[10].metadata, ensure_ascii=False, indent=2))

# sanity checks
issues = []
for i, d in enumerate(docs):
    md = d.metadata
    if not d.page_content.strip():
        issues.append(f"chunk {i}: empty content")
    if md.get("page_start", 0) > md.get("page_end", 0):
        issues.append(f"chunk {i}: page_start>page_end")
    for k in ("heading_path", "contains_sections", "contains_subitems", "style"):
        if k in md:
            issues.append(f"chunk {i}: stale field {k}")
    if "contained_sections" not in md:
        issues.append(f"chunk {i}: missing contained_sections")
    # enum markers must never be section nodes in the tree
    def check(tree):
        if not tree:
            return
        m = tree.get("marker") or {}
        if m.get("prefix") in ("(", "（") and m.get("num_type") == "arabic":
            issues.append(f"chunk {i}: enum marker {m.get('text')} leaked into tree")
        for c in (tree.get("items") or []):
            check(c)
    check(md.get("heading_tree"))

# chunk_index continuity
idxs = [d.metadata.get("chunk_index") for d in docs]
if idxs != list(range(len(docs))):
    issues.append(f"chunk_index not sequential: {idxs}")

print("\nISSUES:", issues if issues else "NONE")
