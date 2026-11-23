"""build.py - load documents, chunk them, build the vector (FAISS) and keyword (BM25) indexes.

Put your PDFs / .txt / .md files in data/raw/, then run from the project root:
    python build.py
"""
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
CHUNKS_FILE = Path("data/chunks.json")
INDEX_DIR = Path("index")

CHUNK_WORDS = 300      # experiment with 200 / 300 / 400 and compare in eval.py
OVERLAP_WORDS = 50
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)   # rejoin words hyphenated across lines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pages():
    """One record per PDF page, or per text/markdown file."""
    for path in sorted(RAW_DIR.glob("**/*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_no, page in enumerate(reader.pages, start=1):
                text = clean(page.extract_text() or "")
                if text:
                    yield {"source": path.name, "page": page_no, "text": text}
        elif suffix in {".txt", ".md"}:
            text = clean(path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                yield {"source": path.name, "page": 1, "text": text}


def split_words(text: str, size: int, overlap: int):
    words = text.split()
    step = size - overlap
    for start in range(0, len(words), step):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            break


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def main():
    # 1) chunk
    chunks = []
    for page in load_pages():
        for piece in split_words(page["text"], CHUNK_WORDS, OVERLAP_WORDS):
            chunks.append({
                "id": f"c{len(chunks):04d}",
                "source": page["source"],
                "page": page["page"],
                "text": piece,
            })
    if not chunks:
        raise SystemExit(f"No documents found in {RAW_DIR}/ (add PDF, .txt or .md files).")
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {CHUNKS_FILE}")

    # 2) vector index (normalized embeddings, so inner product == cosine similarity)
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # 3) keyword index
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"Indexes saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
