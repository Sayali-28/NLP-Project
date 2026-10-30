"""PDF text extraction and cleaning."""
import re
from collections import Counter

import fitz  # PyMuPDF


def extract_pages(file_bytes: bytes) -> list[str]:
    """Return the raw text of every page of a PDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def _remove_headers_footers(pages: list[str]) -> list[str]:
    """Drop lines that repeat on at least half of the pages (headers, footers, page numbers)."""
    if len(pages) < 3:
        return pages
    counts = Counter()
    for p in pages:
        for line in {l.strip() for l in p.splitlines() if l.strip()}:
            counts[line] += 1
    threshold = max(3, len(pages) // 2)
    common = {l for l, c in counts.items() if c >= threshold and len(l) < 100}
    cleaned = []
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip() not in common]
        cleaned.append("\n".join(lines))
    return cleaned


def clean_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)          # join hyphenated words
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # single newlines -> space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_paper(file_bytes: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, cleaned_text) for pages that contain text."""
    pages = _remove_headers_footers(extract_pages(file_bytes))
    result = []
    for i, p in enumerate(pages, start=1):
        t = clean_text(p)
        if t:
            result.append((i, t))
    return result
