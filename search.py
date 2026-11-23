import sys
from rag import retrieve

for h in retrieve(" ".join(sys.argv[1:])):
    print(f"[{h['id']}] {h['source']} p.{h['page']} score={h['score']:.2f}")
    print(h["text"][:300], "\n")
