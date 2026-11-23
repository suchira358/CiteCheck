# CiteCheck

Upload your PDFs, ask questions and get answers with citations, or paste a statement to check whether your documents support it.

## What it does

- **Ask a question:** finds the most relevant passages in your documents and writes a short answer that cites the passages it used. If the documents don't contain the answer, it says "Not found in the documents" instead of guessing.
- **Check a statement:** paste a sentence (or a few), and it replies *Correct*, *Incorrect*, or *Not covered in the documents*, with a short explanation and the supporting passage. "Correct" here means *consistent with your documents*, not true in the real world.
- **Upload your own PDFs** in the browser. Nothing is read from a fixed folder.

## How it works

1. **Chunking:** each page is split into overlapping chunks of about 300 words, and each chunk keeps its file name and page number.
2. **Hybrid retrieval:** every question searches two ways, by meaning (embeddings with `bge-small`, stored in FAISS) and by keywords (BM25). The two ranked lists are merged with reciprocal rank fusion.
3. **Reranking:** a cross-encoder re-scores the top 20 candidates and the best 4 go to the LLM.
4. **Grounded answer:** the LLM (Gemini) is told to use only the retrieved passages, cite chunk IDs, and reply NOT_FOUND if the answer isn't there. An optional second pass checks that the answer is supported, and retries once with a rewritten query.
5. **Statement checking:** the same retrieval, then the LLM judges the statement against the passages using a strict three-way verdict.

## Evaluation

Retrieval was tested on **25 questions** written from my OS lecture notes (55 chunks). Each question has a labelled correct chunk. *Hit@k* is how often the correct chunk appears in the top k results, and *MRR* measures how high it ranks on average.

| Method | Hit@1 | Hit@3 | MRR@10 |
|---|---|---|---|
| Vector only | 84% | 96% | 0.90 |
| BM25 only | 80% | 92% | 0.87 |
| Hybrid (vector + BM25) | 88% | 100% | 0.93 |
| Hybrid + reranker | 96% | 100% | 0.98 |

Adding keyword search and then a reranker improved how often the correct passage ranked first.

## Limitations

- Small test set (one question is worth 4 percentage points) and a single-subject corpus, so treat the numbers as indicative, not exact.
- Statement checking only sees the top retrieved passages. If the evidence is elsewhere, it may answer "not covered".
- It can only be as correct as the documents. If a document is wrong, the verdict will be too.
- Scanned PDFs (images of pages) are not supported, because there is no text to extract.
- The free Gemini API sometimes returns 503 errors under load, so the code retries with short waits and falls back to a second model.

## Tech stack

Python, sentence-transformers (`bge-small-en-v1.5`, `ms-marco-MiniLM` cross-encoder), FAISS, rank_bm25, pypdf, Streamlit, Google Gemini API (`google-genai`).

## Run it

```bash
pip install -r requirements.txt
```

Set your own Gemini API key (never put it in the code):

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_key"
# macOS / Linux
export GEMINI_API_KEY="your_key"
```

Start the app, then upload PDFs in the sidebar:

```bash
python -m streamlit run app.py
```

### Reproduce the evaluation

1. Put your own documents in `data/raw/` (not included in this repo).
2. `python build.py` builds the indexes.
3. Write your test questions in `data/eval/qa.json`. See the top of `eval.py` for the format.
4. `python eval.py` prints the table and saves it to `eval/results.md`.

## Project structure

```
app.py            Streamlit app: upload PDFs, ask questions, check statements
rag.py            retrieval, reranking, answer generation, statement checking
build.py          chunking and index building for the evaluation corpus
eval.py           retrieval evaluation (hit@k, MRR)
requirements.txt
```

## How this was built

I designed the pipeline and the evaluation, built it with the help of an AI coding assistant, ran the experiments myself, and tested the results on my own documents.

## Sources

Evaluation documents: [my own OS lecture notes / name your source]. Not included in the repo.
