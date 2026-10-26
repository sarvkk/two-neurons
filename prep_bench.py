#!/usr/bin/env python3
"""Prepare benchmark prompt sets for the steering eval.

Outputs (data/, relative to this repo):
  sst.jsonl         - SimpleSafetyTests, all 100 (5 harm areas x 20)
  sst_subset.jsonl  - 20 of them (4 per harm area), for single-neuron/ablation runs
  xstest_safe.jsonl - 80 benign XSTest prompts, stratified across types

Sources:
  SST:    https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests (sst_test_cases.csv)
  XSTest: https://huggingface.co/datasets/LLMSafety/XSTest (parquet, 250 safe / 200 unsafe)
"""
import csv
import json
import os
import urllib.request
from collections import defaultdict

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

SST_URL = ("https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests/"
           "resolve/main/sst_test_cases.csv")
XST_URL = ("https://huggingface.co/datasets/LLMSafety/XSTest/"
           "resolve/main/data/train-00000-of-00001.parquet")

sst_csv = os.path.join(DATA, "sst_test_cases.csv")
xst_pq = os.path.join(DATA, "xstest.parquet")
if not os.path.exists(sst_csv):
    urllib.request.urlretrieve(SST_URL, sst_csv)
if not os.path.exists(xst_pq):
    urllib.request.urlretrieve(XST_URL, xst_pq)

rows = list(csv.DictReader(open(sst_csv, encoding="utf-8")))
assert len(rows) == 100, len(rows)

with open(os.path.join(DATA, "sst.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps({"dataset": "SimpleSafetyTests", "id": r["id"],
                            "category": r["harm_area"], "prompt": r["prompt"]}) + "\n")

# 4 per harm area, evenly spaced: ids 1,6,11,16 within each block of 20
by_area = defaultdict(list)
for r in rows:
    by_area[r["harm_area"]].append(r)
sub = []
for area, rs in by_area.items():
    sub += [rs[i] for i in (0, 5, 10, 15)]
with open(os.path.join(DATA, "sst_subset.jsonl"), "w") as f:
    for r in sub:
        f.write(json.dumps({"dataset": "SimpleSafetyTests", "id": r["id"],
                            "category": r["harm_area"], "prompt": r["prompt"]}) + "\n")

x = pq.read_table(xst_pq).to_pylist()
safe = [r for r in x if r["label"] == "safe"]
by_type = defaultdict(list)
for r in safe:
    by_type[r["type"]].append(r)

# stratified: round-robin across benign types, 80 total
picked, i = [], 0
types = sorted(by_type)
while len(picked) < 80:
    for t in types:
        if i < len(by_type[t]) and len(picked) < 80:
            picked.append(by_type[t][i])
    i += 1
    if i > 50:
        break

with open(os.path.join(DATA, "xstest_safe.jsonl"), "w") as f:
    for j, r in enumerate(picked):
        f.write(json.dumps({"dataset": "XSTest", "id": f"xst_{j:03d}",
                            "category": r["type"], "prompt": r["prompt"]}) + "\n")

print(f"SST: {len(rows)} -> sst.jsonl | subset {len(sub)}")
print(f"XSTest safe: {len(safe)} total, picked {len(picked)} across {len(types)} types")
print("types:", {t: len(v) for t, v in sorted(by_type.items())})
