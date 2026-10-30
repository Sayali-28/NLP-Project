"""Split the paper into overlapping word chunks, remembering page numbers."""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    page: int


def make_chunks(pages: list[tuple[int, str]], size: int = 150, overlap: int = 30) -> list[Chunk]:
    words = []  # (word, page)
    for page_no, text in pages:
        words.extend((w, page_no) for w in text.split())
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        part = words[start:start + size]
        if len(part) < 20 and chunks:  # merge a tiny tail into nothing
            break
        chunks.append(Chunk(" ".join(w for w, _ in part), part[0][1]))
        if start + size >= len(words):
            break
    return chunks
