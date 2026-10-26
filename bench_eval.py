#!/usr/bin/env python3
"""Benchmark steering eval: SimpleSafetyTests + XSTest (benign).

Conditions per prompt set:
  SST (all 100):       baseline, both, flipped
  SST subset (20):     baseline, both, flipped, only_L14, only_L13, shift_L1
  XSTest safe (80):    baseline, both

  both     = L14 N8949 +1, L13 N7956 -1   (the tweet's recipe, 0-indexed layers)
  flipped  = signs reversed
  only_L14 = L14 N8949 +1
  only_L13 = L13 N7956 -1
  shift_L1 = L15 N8949 +1, L14 N7956 -1   (tests the 1-indexed reading)

Recipe: clamp down_proj input neuron n at every token to
    s = dose * median(||resid leaving layer L, tokens 1:) / ||W_down[:, n]||

Outputs bench_runs.jsonl next to the script (resumable; key = dataset+id+condition).
"""
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-4B"
MAX_NEW = 72
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "bench_runs.jsonl")

CONDITIONS = {
    "baseline": [],
    "both": [(14, 8949, +1.0), (13, 7956, -1.0)],
    "flipped": [(14, 8949, -1.0), (13, 7956, +1.0)],
    "only_L14": [(14, 8949, +1.0)],
    "only_L13": [(13, 7956, -1.0)],
    "shift_L1": [(15, 8949, +1.0), (14, 7956, -1.0)],
}

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

sst = load_jsonl(os.path.join(DATA, "sst.jsonl"))
sst_sub_ids = {r["id"] for r in load_jsonl(os.path.join(DATA, "sst_subset.jsonl"))}
xst = load_jsonl(os.path.join(DATA, "xstest_safe.jsonl"))

PLAN = []  # (row, [conditions])
for r in sst:
    conds = ["baseline", "both", "flipped"]
    if r["id"] in sst_sub_ids:
        conds += ["only_L14", "only_L13", "shift_L1"]
    PLAN.append((r, conds))
for r in xst:
    PLAN.append((r, ["baseline", "both"]))

# ---------------------------------------------------------------- model
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16,
                         llm_int8_skip_modules=[])
print("[load] Qwen3.5-4B 4-bit ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
model.eval()

if model.lm_head.__class__.__name__ == "Linear4bit":
    emb = model.model.embed_tokens.weight
    lin = torch.nn.Linear(emb.shape[1], emb.shape[0], bias=False,
                          dtype=emb.dtype, device="meta")
    lin.weight = emb
    model.lm_head = lin

down, dnames = {}, {}
for name, mod in model.named_modules():
    m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
    if m:
        down[int(m.group(1))] = mod
        dnames[int(m.group(1))] = name
text_root = dnames[min(down)].split(".layers.")[0]
layers = {L: model.get_submodule(f"{text_root}.layers.{L}") for L in range(32)}

# column norms for every (layer, neuron) any condition touches
from bitsandbytes import functional as BF
needed = set()
for specs in CONDITIONS.values():
    for L, n, _d in specs:
        needed.add((L, n))
colnorm = {}
for L in sorted({L for L, _ in needed}):
    w = BF.dequantize_4bit(down[L].weight.data, down[L].weight.quant_state)
    for LL, n in needed:
        if LL == L:
            colnorm[(L, n)] = float(w[:, n].float().norm())
    del w
torch.cuda.empty_cache()
print(f"[load] {time.time()-t0:.1f}s | colnorms: "
      + ", ".join(f"L{L}N{n}={v:.4f}" for (L, n), v in sorted(colnorm.items())), flush=True)

# ---------------------------------------------------------------- hooks
DOSE = {}   # (layer, neuron) -> clamp value

def _pre_hook(L):
    def hook(module, args, kwargs):
        for (LL, n), s in DOSE.items():
            if LL == L:
                args[0][..., n] = s
        return args, kwargs
    return hook

for L in down:
    down[L].register_forward_pre_hook(_pre_hook(L), with_kwargs=True)


def render(text):
    out = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True, return_tensors="pt")
    ids = out.input_ids if hasattr(out, "input_ids") else out
    return ids.to("cuda")


def capture(ids):
    got = {}
    handles = []
    for L in (13, 14, 15):
        holder = {}
        def cap(_m, _a, o, _h=holder):
            _h["h"] = o[0] if isinstance(o, tuple) else o
        handles.append(layers[L].register_forward_hook(cap))
        got[L] = holder
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    for h in handles:
        h.remove()
    return {L: got[L]["h"] for L in got}


def med_for(meds, L):
    """Residual median for layer L; shifted layers fall back to the nearest measured one."""
    if L in meds:
        return meds[L]
    return meds[min(meds, key=lambda k: abs(k - L))]


os.makedirs(os.path.dirname(OUT), exist_ok=True)
done = set()
if os.path.exists(OUT):
    for r in load_jsonl(OUT):
        done.add((r["dataset"], r["id"], r["condition"]))
log = open(OUT, "a")

for row, conds in PLAN:
    todo = [c for c in conds if (row["dataset"], row["id"], c) not in done]
    if not todo:
        continue
    ids = render(row["prompt"])
    caps = capture(ids)
    meds = {L: float(h[0, 1:, :].float().norm(dim=-1).median()) for L, h in caps.items()}
    print(f"\n[{row['dataset']}/{row['id']}] {row['prompt'][:70]!r}", flush=True)
    for cond in todo:
        DOSE.clear()
        doses = {}
        for L, n, d in CONDITIONS[cond]:
            s = d * med_for(meds, L) / colnorm[(L, n)]
            DOSE[(L, n)] = s
            doses[f"L{L}N{n}"] = round(s, 2)
        attn = torch.ones_like(ids)
        t = time.time()
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                                 do_sample=False, use_cache=True)
        resp = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        dt = time.time() - t
        DOSE.clear()
        log.write(json.dumps({
            "dataset": row["dataset"], "id": row["id"], "category": row["category"],
            "prompt": row["prompt"], "condition": cond, "dose_vals": doses,
            "medians": {str(L): round(v, 2) for L, v in meds.items()},
            "response": resp, "n_tokens": int(out.shape[1] - ids.shape[1]),
            "seconds": round(dt, 1),
        }) + "\n")
        log.flush()
        print(f"  {cond:9} {dt:4.1f}s {int(out.shape[1]-ids.shape[1]):3d}tok "
              f"{doses} -> {resp[:90]!r}", flush=True)

log.close()
print("\n[done]", OUT)
