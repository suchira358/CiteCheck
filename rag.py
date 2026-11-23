"""rag.py - retrieval (vector / BM25 / hybrid / hybrid + rerank), grounded answers with
citations, a self-check, and a "not found" fallback.

Setup:  export GEMINI_API_KEY=your_key      (Windows: set GEMINI_API_KEY=your_key)
Run:    python rag.py "your question"          or just   python rag.py   for a chat loop
"""
import json
import os
import pickle
import re
import sys
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from google import genai
from sentence_transformers import CrossEncoder, SentenceTransformer
import time

CHUNKS_FILE = Path("data/chunks.json")
INDEX_DIR = Path("index")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gemini-4-hydrogen")

CANDIDATES = 20   # how many chunks go into the reranker
FINAL_K = 4       # how many chunks the LLM sees
RRF_K = 60        # standard constant for reciprocal rank fusion
MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
NOT_FOUND_MSG = "Not found in the documents."


# ---------- loading (done once) ----------
@lru_cache(maxsize=1)
def resources():
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    return chunks, index, bm25, embedder


@lru_cache(maxsize=1)
def reranker():
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def client():
    return genai.Client()   # reads GEMINI_API_KEY from the environment


FALLBACK_MODELS = [LLM_MODEL, "gemini-2.5-flash"]
RETRY_ON = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL",
    "disconnected", "RemoteProtocolError", "timeout", "timed out", "Connection",
)


def llm(prompt, max_tokens=800):
    last_error = None
    for model in FALLBACK_MODELS:
        for attempt in range(3):
            try:
                resp = client().models.generate_content(model=model, contents=prompt)
                return (resp.text or "").strip()
            except Exception as e:
                last_error = e
                msg = f"{type(e).__name__} {e}"
                if "404" in msg or "NOT_FOUND" in msg:
                    break                          # this model name isn't available: next model
                if any(token in msg for token in RETRY_ON):
                    time.sleep(2 * (attempt + 1))  # busy or connection dropped: wait, retry
                    continue
                raise                              # bad key, bad request, etc.: show it
    raise last_error


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


# ---------- retrieval ----------
def vector_ids(query, k):
    _, index, _, embedder = resources()
    q = embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    _, ids = index.search(np.asarray(q, dtype="float32"), k)
    return [int(i) for i in ids[0] if i != -1]


def bm25_ids(query, k):
    _, _, bm25, _ = resources()
    scores = bm25.get_scores(tokenize(query))
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def rrf(rankings, k=RRF_K):
    """Reciprocal rank fusion: combine ranked lists using ranks, not raw scores
    (BM25 scores and cosine scores are on different scales)."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve(query, mode="hybrid_rerank", final_k=FINAL_K):
    """Return the top chunks (dicts with a 'score' key; score is None unless reranked)."""
    chunks = resources()[0]
    if mode == "vector":
        return [dict(chunks[i], score=None) for i in vector_ids(query, final_k)]
    if mode == "bm25":
        return [dict(chunks[i], score=None) for i in bm25_ids(query, final_k)]

    fused = rrf([vector_ids(query, CANDIDATES), bm25_ids(query, CANDIDATES)])
    if mode == "hybrid":
        return [dict(chunks[i], score=None) for i in fused[:final_k]]

    # hybrid_rerank: a cross-encoder re-scores each (query, chunk) pair more carefully
    candidates = fused[:CANDIDATES]
    pairs = [(query, chunks[i]["text"]) for i in candidates]
    scores = reranker().predict(pairs)
    order = np.argsort(scores)[::-1][:final_k]
    return [dict(chunks[candidates[j]], score=float(scores[j])) for j in order]


# ---------- generation + self-check ----------
ANSWER_PROMPT = """You answer questions using ONLY the context below.
Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- After each claim, cite the chunk id in square brackets, like [c0012].
- If the context does not contain the answer, reply with exactly: NOT_FOUND

Context:
{context}

Question: {question}"""

VERIFY_PROMPT = """Context:
{context}

Answer:
{answer}

Is every claim in the answer directly supported by the context above?
Reply with exactly YES or NO."""

REWRITE_PROMPT = """Rewrite this question as a short, keyword-rich search query for a document
search engine. Output only the query.

Question: {question}"""

@lru_cache(maxsize=1)
def client():
    return genai.Client()   # reads GEMINI_API_KEY from the environment


def llm(prompt, max_tokens=800):
    resp = client().models.generate_content(model=LLM_MODEL, contents=prompt)
    return (resp.text or "").strip()


def format_context(hits):
    return "\n\n".join(
        f"[{h['id']}] ({h['source']}, page {h['page']})\n{h['text']}" for h in hits
    )


def ask(question, mode="hybrid_rerank", self_check=True):
    """Full pipeline. Returns {'answer', 'sources', 'rewritten_query', 'grounded'}."""

    def attempt(search_query):
        hits = retrieve(search_query, mode)
        context = format_context(hits)
        answer = llm(ANSWER_PROMPT.format(context=context, question=question))
        ok = "NOT_FOUND" not in answer
        if ok and self_check:
            verdict = llm(VERIFY_PROMPT.format(context=context, answer=answer), max_tokens=5)
            ok = verdict.upper().startswith("YES")
        return answer, hits, ok

    answer, hits, ok = attempt(question)
    rewritten = None
    if not ok and self_check:                       # one retry with a rewritten query
        rewritten = llm(REWRITE_PROMPT.format(question=question), max_tokens=60)
        answer, hits, ok = attempt(rewritten)
    if not ok:
        answer = NOT_FOUND_MSG
    return {"answer": answer, "sources": hits, "rewritten_query": rewritten, "grounded": ok}

CLAIM_PROMPT = """You are a fact checker. Judge the statement ONLY using the context below.

Context:
{context}

Statement: {claim}

Reply in exactly this format:
VERDICT: <TRUE, FALSE, or NOT_ENOUGH_INFO>
EXPLANATION: <one or two sentences explaining why, citing chunk ids like [c0012]>

Rules:
- TRUE only if the context clearly supports the whole statement.
- FALSE only if the context clearly contradicts part of it; say what the context says instead.
- NOT_ENOUGH_INFO if the context does not settle it. Do not use outside knowledge."""


def check_claim(claim, hits=None, mode="hybrid_rerank"):
    """Judge one statement against the documents."""
    if hits is None:
        hits = retrieve(claim, mode)
    text = llm(CLAIM_PROMPT.format(context=format_context(hits), claim=claim))
    m = re.search(r"VERDICT:\s*(TRUE|FALSE|NOT_ENOUGH_INFO)", text, re.I)
    verdict = m.group(1).upper() if m else "NOT_ENOUGH_INFO"
    e = re.search(r"EXPLANATION:\s*(.*)", text, re.S | re.I)
    explanation = e.group(1).strip() if e else text
    return {"verdict": verdict, "explanation": explanation, "sources": hits}
# ---------- command line ----------
def show(result):
    print("\nANSWER:", result["answer"])
    if result["rewritten_query"]:
        print("(retried with query:", result["rewritten_query"] + ")")
    print("\nSOURCES:")
    for h in result["sources"]:
        score = "" if h["score"] is None else f"  score={h['score']:.2f}"
        print(f"  [{h['id']}] {h['source']} p.{h['page']}{score}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "check":
        r = check_claim(" ".join(sys.argv[2:]))
        print("\nVERDICT:", r["verdict"])
        print(r["explanation"])
        return
    if len(sys.argv) > 1:
        show(ask(" ".join(sys.argv[1:])))
        return
    print("Ask a question (empty line to quit).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        show(ask(q))


if __name__ == "__main__":
    main()
