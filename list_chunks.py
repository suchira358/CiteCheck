import json

for c in json.load(open("data/chunks.json", encoding="utf-8")):
    print(f"[{c['id']}] p.{c['page']}: {c['text'][:100]}")
