"""app.py - upload documents, then ask questions about them.
Run from the project folder:  streamlit run app.py
"""
import io
import os
import re

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import rag
from build import CHUNK_WORDS, EMBED_MODEL, OVERLAP_WORDS, clean, split_words, tokenize

st.set_page_config(page_title="Ask your documents", layout="wide")


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_reranker():
    return CrossEncoder(rag.RERANK_MODEL)


def read_pages(uploaded):
    pages = []
    for f in uploaded:
        if f.name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(f.getvalue()))
            for page_no, page in enumerate(reader.pages, start=1):
                text = clean(page.extract_text() or "")
                if text:
                    pages.append({"source": f.name, "page": page_no, "text": text})
        else:
            text = clean(f.getvalue().decode("utf-8", errors="ignore"))
            if text:
                pages.append({"source": f.name, "page": 1, "text": text})
    return pages


def build_kb(uploaded):
    chunks = []
    for page in read_pages(uploaded):
        for piece in split_words(page["text"], CHUNK_WORDS, OVERLAP_WORDS):
            chunks.append({
                "id": f"c{len(chunks):04d}",
                "source": page["source"],
                "page": page["page"],
                "text": piece,
            })
    if not chunks:
        return None
    texts = [c["text"] for c in chunks]
    emb = get_embedder().encode(texts, normalize_embeddings=True, batch_size=32)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return {"chunks": chunks, "index": index, "bm25": bm25}


def retrieve(kb, query, k=rag.FINAL_K):
    chunks = kb["chunks"]
    q = get_embedder().encode([rag.QUERY_PREFIX + query], normalize_embeddings=True)
    _, ids = kb["index"].search(np.asarray(q, dtype="float32"), rag.CANDIDATES)
    vector_ids = [int(i) for i in ids[0] if i != -1]
    scores = kb["bm25"].get_scores(tokenize(query))
    bm25_ids = [int(i) for i in np.argsort(scores)[::-1][:rag.CANDIDATES]]
    fused = rag.rrf([vector_ids, bm25_ids])[:rag.CANDIDATES]
    pairs = [(query, chunks[i]["text"]) for i in fused]
    rerank_scores = get_reranker().predict(pairs)
    order = np.argsort(rerank_scores)[::-1][:k]
    return [dict(chunks[fused[j]], score=float(rerank_scores[j])) for j in order]


def answer(kb, question):
    def attempt(search_query):
        hits = retrieve(kb, search_query)
        context = rag.format_context(hits)
        text = rag.llm(rag.ANSWER_PROMPT.format(context=context, question=question))
        ok = "NOT_FOUND" not in text
        if ok:
            verdict = rag.llm(rag.VERIFY_PROMPT.format(context=context, answer=text), max_tokens=5)
            ok = verdict.upper().startswith("YES")
        return text, hits, ok

    text, hits, ok = attempt(question)
    rewritten = None
    if not ok:
        rewritten = rag.llm(rag.REWRITE_PROMPT.format(question=question), max_tokens=60)
        text, hits, ok = attempt(rewritten)
    if not ok:
        text = rag.NOT_FOUND_MSG
    return text, hits, rewritten, ok


# ---------------- interface ----------------
with st.sidebar:
    st.header("Setup")
    key = st.text_input("GEMINI API key", type="password")
    if key and key != st.session_state.get("key"):
        os.environ["GEMINI_API_KEY"] = key
        rag.client.cache_clear()
        st.session_state["key"] = key
    uploaded = st.file_uploader(
        "Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True
    )

st.title("Ask your documents")
if not uploaded:
    st.info("Upload one or more PDFs in the sidebar to begin.")
    st.stop()

signature = tuple((f.name, f.size) for f in uploaded)
if st.session_state.get("signature") != signature:
    with st.spinner("Reading and indexing your documents..."):
        st.session_state["kb"] = build_kb(uploaded)
        st.session_state["signature"] = signature
kb = st.session_state["kb"]

if kb is None:
    st.error("No readable text found. Scanned PDFs (photos of pages) can't be read.")
    st.stop()

st.success(f"Indexed {len(kb['chunks'])} chunks from {len(uploaded)} file(s).")

mode = st.radio(
    "What do you want to do?", ["Ask a question", "Check a statement"], horizontal=True
)
has_key = bool(os.getenv("GEMINI_API_KEY"))

if mode == "Ask a question":
    question = st.text_input("Ask a question about your documents")
    if question:
        if has_key:
            with st.spinner("Searching and writing the answer..."):
                text, hits, rewritten, ok = answer(kb, question)
            if ok:
                st.markdown("### Answer")
                st.write(text)
                cited = set(re.findall(r"c\d{4}", text))
                used = [h for h in hits if h["id"] in cited]
                if used:
                    with st.expander("Sources used"):
                        for h in used:
                            st.markdown(f"**[{h['id']}] {h['source']}, page {h['page']}**")
                            st.write(h["text"])
            else:
                st.warning(text)
        else:
            with st.spinner("Searching..."):
                hits = retrieve(kb, question)
            best = hits[0]
            if best["score"] > 0:
                st.markdown("### Best matching passage")
                st.caption(f"{best['source']}, page {best['page']}")
                st.write(best["text"])
                st.info("Add an API key in the sidebar to get a written answer instead of a raw passage.")
            else:
                st.warning("Nothing relevant found in the uploaded documents.")
else:
    statement = st.text_area("Paste a statement (or a few sentences) to check against your documents")
    if st.button("Check") and statement.strip():
        if not has_key:
            st.warning("Checking statements needs a Gemini API key. Add it in the sidebar.")
        else:
            parts = re.split(r"(?<=[.!?])\s+", statement.strip())
            sentences = [s.strip() for s in parts if len(s.strip()) > 5][:10]
            if not sentences:
                sentences = [statement.strip()]
            labels = {
                "TRUE": "✅ Correct",
                "FALSE": "❌ Incorrect",
                "NOT_ENOUGH_INFO": "❔ Not covered in the documents",
            }
            for s in sentences:
                with st.spinner("Checking..."):
                    r = rag.check_claim(s, hits=retrieve(kb, s))
                st.markdown(f"**{s}**")
                st.write(f"{labels[r['verdict']]}: {r['explanation']}")
                cited = set(re.findall(r"c\d{4}", r["explanation"]))
                used = [h for h in r["sources"] if h["id"] in cited]
                if used:
                    with st.expander("Evidence"):
                        for h in used:
                            st.markdown(f"**[{h['id']}] {h['source']}, page {h['page']}**")
                            st.write(h["text"])
                st.divider()
