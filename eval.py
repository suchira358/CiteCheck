"""eval.py - compare vector / BM25 / hybrid+rerank on your own test questions.

Your test file is data/eval/qa.json, a list like:
[
  {"question": "How do I read a file line by line?", "expected_chunk_ids": ["c0012"], "answerable": true},
  {"question": "Who won the 2018 World Cup?", "answerable": false}
]
Aim for ~30 questions, including ~5 unanswerable ones (to test the "not found" fallback).

Commands (run from the project root):
    python eval.py find <keyword>   # list chunk ids containing a word (helps you write expected_chunk_ids)
    python eval.py                  # retrieval table: hit@1, hit@3, MRR@10 for each mode
    python eval.py --fallback       # also test unanswerable questions with the full pipeline (uses the LLM API)
"""
import json
import sys
from pathlib import Path

QA_FILE = Path("data/eval/qa.json")
CHUNKS_FILE = Path("data/chunks.json")
RESULTS_FILE = Path("eval/results.md")


def find(keyword):
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    kw = keyword.lower()
    found = 0
    for c in chunks:
        pos = c["text"].lower().find(kw)
        if pos != -1:
            found += 1
            snippet = c["text"][max(0, pos - 60): pos + 100]
            print(f"[{c['id']}] {c['source']} p.{c['page']}: ...{snippet}...")
    print(f"\n{found} chunk(s) contain '{keyword}'")


def evaluate_retrieval(items):
    from rag import MODES, retrieve
    rows = []
    for mode in MODES:
        hit1 = hit3 = 0
        rr_total = 0.0
        for item in items:
            ids = [h["id"] for h in retrieve(item["question"], mode, final_k=10)]
            ranks = [ids.index(c) + 1 for c in item["expected_chunk_ids"] if c in ids]
            if ranks:
                best = min(ranks)
                rr_total += 1.0 / best
                hit1 += best == 1
                hit3 += best <= 3
        n = len(items)
        rows.append((mode, hit1 / n, hit3 / n, rr_total / n))
    return rows


def evaluate_fallback(items):
    from rag import NOT_FOUND_MSG, ask
    refused = 0
    for item in items:
        result = ask(item["question"])
        ok = result["answer"] == NOT_FOUND_MSG
        refused += ok
        print(f"  {'OK ' if ok else 'BAD'} {item['question']}")
    return refused, len(items)


def main():
    args = sys.argv[1:]
    if args and args[0] == "find":
        find(" ".join(args[1:]))
        return

    qa = json.loads(QA_FILE.read_text(encoding="utf-8"))
    answerable = [q for q in qa if q.get("answerable", True) and q.get("expected_chunk_ids")]
    unanswerable = [q for q in qa if not q.get("answerable", True)]
    if not answerable:
        raise SystemExit("No answerable questions with expected_chunk_ids in data/eval/qa.json")

    rows = evaluate_retrieval(answerable)
    lines = [
        f"Retrieval results on {len(answerable)} answerable questions",
        "",
        "| Method | Hit@1 | Hit@3 | MRR@10 |",
        "|---|---|---|---|",
    ]
    for mode, h1, h3, mrr in rows:
        lines.append(f"| {mode} | {h1:.0%} | {h3:.0%} | {mrr:.2f} |")

    if "--fallback" in args and unanswerable:
        print("\nTesting unanswerable questions:")
        refused, total = evaluate_fallback(unanswerable)
        lines += ["", f"Unanswerable questions correctly refused: {refused}/{total}"]

    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
    
