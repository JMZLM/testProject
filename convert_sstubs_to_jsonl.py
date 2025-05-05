# convert_sstubs_to_jsonl.py

import json, os

INPUT = "data/sstubs.json"           # your downloaded file
OUTPUT = "data/sstubs/bugs.jsonl"    # where we’ll write the converted data

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(INPUT, "r", encoding="utf-8") as fin:
    # peek first non-whitespace character
    first_char = None
    while True:
        pos = fin.tell()
        ch  = fin.read(1)
        if not ch:
            break
        if not ch.isspace():
            first_char = ch
            fin.seek(pos)
            break

    # load accordingly
    if first_char == "[":
        # it's a JSON array
        all_data = json.load(fin)
    else:
        # assume JSON Lines
        all_data = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            all_data.append(json.loads(line))

# now convert to our (bug, func, label) format
with open(OUTPUT, "w", encoding="utf-8") as fout:
    for obj in all_data:
        # Build a simple bug report from the patch or category
        bug_text = obj.get("patch") or obj.get("category") or ""
        buggy    = obj.get("before", "").strip()
        fixed    = obj.get("after",  "").strip()
        if not bug_text or not buggy or not fixed:
            continue

        # Write the buggy version (label=1)
        fout.write(json.dumps({
            "bug":  bug_text,
            "func": buggy,
            "label": 1
        }, ensure_ascii=False) + "\n")

        # Write the fixed version (label=0)
        fout.write(json.dumps({
            "bug":  bug_text,
            "func": fixed,
            "label": 0
        }, ensure_ascii=False) + "\n")

print("✅ Wrote converted dataset to", OUTPUT)
