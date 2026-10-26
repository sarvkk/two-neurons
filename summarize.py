#!/usr/bin/env python3
"""Turn raw_results.jsonl into markdown tables for the report."""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "raw_results.jsonl")
DST = os.path.join(HERE, "summary.md")

rows = [json.loads(l) for l in open(SRC) if l.strip()]
by_prompt = defaultdict(dict)
for r in rows:
    by_prompt[r["prompt_id"]][r["condition"]] = r

out = ["# Experiment summary", ""]
out += ["## Per-prompt metrics", "",
        "| prompt | condition | words | refusal | hostile | loop | bleed | uniq | s (L14/L13) |",
        "|---|---|---|---|---|---|---|---|---|"]
for pid, conds in by_prompt.items():
    for cond in ("clean", "steered", "control"):
        r = conds.get(cond)
        if not r:
            continue
        m = r["metrics"]
        s14 = r["dose_vals"].get("L14N8949", "")
        s13 = r["dose_vals"].get("L13N7956", "")
        out.append(f"| {pid} | {cond} | {m['words']} | {m['refusal_hits']} | "
                   f"{m['hostile_hits']} | {'Y' if m['loop_flag'] else ''} | "
                   f"{'Y' if m['bleed_flag'] else ''} | {m['unique_word_ratio']:.2f} | "
                   f"{s14} / {s13} |")

out += ["", "## Aggregates (mean over prompts)", "",
        "| condition | words | refusal | hostile | loop rate | bleed rate |",
        "|---|---|---|---|---|---|"]
for cond in ("clean", "steered", "control"):
    rs = [r for r in rows if r["condition"] == cond]
    if not rs:
        continue
    n = len(rs)
    out.append(f"| {cond} | {sum(r['metrics']['words'] for r in rs)/n:.0f} | "
               f"{sum(r['metrics']['refusal_hits'] for r in rs)/n:.2f} | "
               f"{sum(r['metrics']['hostile_hits'] for r in rs)/n:.2f} | "
               f"{100*sum(r['metrics']['loop_flag'] for r in rs)/n:.0f}% | "
               f"{100*sum(r['metrics']['bleed_flag'] for r in rs)/n:.0f}% |")

out += ["", "## Dose math (steered)", "",
        "| prompt | median L13 | median L14 | s L13N7956 | s L14N8949 | delta L13 | delta L14 |",
        "|---|---|---|---|---|---|---|"]
for r in rows:
    if r["condition"] != "steered":
        continue
    out.append(f"| {r['prompt_id']} | {r['medians'].get('13','')} | {r['medians'].get('14','')} | "
               f"{r['dose_vals'].get('L13N7956','')} | {r['dose_vals'].get('L14N8949','')} | "
               f"{r['delta_ratio'].get('13','')} | {r['delta_ratio'].get('14','')} |")

out += ["", "## Clean vs steered excerpts", ""]
for pid, conds in by_prompt.items():
    c = conds.get("clean")
    s = conds.get("steered")
    if not c or not s:
        continue
    out += [f"### {pid}", "", f"- **clean:** {c['clean'][:400]}",
            f"- **steered:** {s['clean'][:400]}", ""]

open(DST, "w").write("\n".join(out))
print("\n".join(out))
print("\n[saved]", DST)
