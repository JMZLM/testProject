import random

IN  = r"E:\'25 code\testProject\data\sstubs\bugs.jsonl"
OUT = r"E:\'25 code\testProject\data\sstubs\small.jsonl"
N   = 2000

with open(IN, 'r', encoding='utf-8') as f:
    lines = [l for l in f if l.strip()]

sample = random.sample(lines, min(N, len(lines)))

with open(OUT, 'w', encoding='utf-8') as f:
    f.writelines(sample)

print(f"Wrote {len(sample)} lines to {OUT}")
