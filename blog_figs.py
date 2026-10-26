#!/usr/bin/env python3
"""Blog figures: minimal charts from bench_scored.jsonl (no chartjunk)."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "blog_img")
os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open(os.path.join(HERE, "bench_scored.jsonl")) if l.strip()]

# match the report's numbers: full-100 rows for baseline/both/flipped, subset rows
# for the ablations, XSTest as-is
SUBSET = {r["id"] for r in rows if r["dataset"] == "SimpleSafetyTests" and r["condition"] == "only_L14"}

def agg(pick, conds):
    out = {}
    for c in conds:
        sub = [r for r in rows if pick(r) and r["condition"] == c]
        n = len(sub)
        out[c] = (100 * sum(r["refusal"] for r in sub) / n,
                  100 * sum(r["unsafe"] for r in sub) / n, n)
    return out

full = agg(lambda r: r["dataset"] == "SimpleSafetyTests" and r["id"] in SUBSET or
           (r["dataset"] == "SimpleSafetyTests" and r["condition"] in ("baseline", "both", "flipped")) and r["id"] in SUBSET,
           ["baseline", "both", "only_L14", "shift_L1"])
# simpler, explicit sets:
MAIN = {c: agg(lambda r: r["dataset"] == "SimpleSafetyTests", [c])["baseline" if c == "baseline" else c]
        for c in []}  # replaced below

def one(dataset, cond, subset_only):
    sub = [r for r in rows if r["dataset"] == dataset and r["condition"] == cond
           and (not subset_only or r["id"] in SUBSET)]
    n = len(sub)
    return (100 * sum(r["refusal"] for r in sub) / n,
            100 * sum(r["unsafe"] for r in sub) / n, n)

main = {c: one("SimpleSafetyTests", c, False) for c in ("baseline", "both", "flipped")}
abl  = {c: one("SimpleSafetyTests", c, True) for c in ("baseline", "both", "only_L14", "only_L13", "shift_L1")}
xst  = {c: one("XSTest", c, False) for c in ("baseline", "both")}

INK = "#1c1917"; MUT = "#6b625c"; ACC = "#a4161a"; RULE = "#e7e1d8"; FILL = "#8a8a8a"

def bars(data, fname, width=0.38):
    conds = list(data)
    ref = [data[c][0] for c in conds]
    uns = [data[c][1] for c in conds]
    x = range(len(conds))
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=170)
    b1 = ax.bar([i - width/2 for i in x], ref, width, color=FILL, label="refusal %")
    b2 = ax.bar([i + width/2 for i in x], uns, width, color=ACC, label="unsafe %")
    for b, v in list(zip(b1, ref)) + list(zip(b2, uns)):
        ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width()/2, v),
                    ha="center", va="bottom", fontsize=9, color=INK)
    labels = [c.replace("_", " ") for c in conds]
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=MUT); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(RULE)
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_ylabel("% of prompts", fontsize=9.5, color=MUT)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.12), ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

for c, v in main.items(): print("main", c, [round(z) for z in v])
for c, v in abl.items(): print("abl ", c, [round(z) for z in v])
for c, v in xst.items(): print("xst ", c, [round(z) for z in v])

bars(main, "evqwen-sst-main.png")
bars(abl, "evqwen-ablations.png")
bars(xst, "evqwen-xstest.png")
print("[figs]", os.listdir(OUT))
