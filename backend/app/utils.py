from typing import List
from PyPDF2 import PdfReader
import re

def extract_text_from_pdf(buf) -> str:
    reader = PdfReader(buf)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n\n".join(pages)

# Simple, layout-agnostic chunker
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]
