#!/usr/bin/env python3
"""Interactive neuron-steering chat for Qwen3.5-4B on a 4GB GPU.

Loads the model once, then REPL: type a message -> clean probe -> dose math ->
streamed steered reply. Slash commands toggle steering, doses, clean mode.

Commands:
    /steer            steering ON  (default)
    /clean            steering OFF
    /both             next turn: print clean and steered replies
    /dose L N VALUE   set dose for neuron N at layer L (e.g. /dose 14 8949 0.5)
    /targets          show current layer/neuron/dose table
    /reset            clear chat history
    /quit             exit

Run:
    export HF_HOME=$HOME/hf-cache
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -u chat_steer.py
"""
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

MODEL = "Qwen/Qwen3.5-4B"
TARGETS = {14: (8949, +1.0), 13: (7956, -1.0)}   # layer -> (neuron, dose)
MAX_NEW = 180
CTX_TOKENS = 3000

tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16,
                         llm_int8_skip_modules=[])
print("[load] Qwen3.5-4B 4-bit ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
model.eval()

# v5 re-ties lm_head to the fp16 embedding after bnb quantizes it -> bnb assert.
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
blocks = {L: model.get_submodule(f"{text_root}.layers.{L}") for L in TARGETS}

from bitsandbytes import functional as BF
col_norm = {}
for L in TARGETS:
    n = TARGETS[L][0]
    w = BF.dequantize_4bit(down[L].weight.data, down[L].weight.quant_state)
    col_norm[n] = float(w[:, n].float().norm())
    del w
torch.cuda.empty_cache()

dose_vec, doses = {}, {}
def current_doses():
    doses.clear()
    for L, (n, d) in TARGETS.items():
        doses[n] = d * meds[L] / col_norm[n]
    return doses

def _pre_hook(L):
    n = TARGETS[L][0]
    def hook(module, args, kwargs):
        if n in dose_vec:
            args[0][..., n] = dose_vec[n]
        return args, kwargs
    return hook

for L in TARGETS:
    down[L].register_forward_pre_hook(_pre_hook(L), with_kwargs=True)

def block_out(blk):
    holder = {}
    def cap(_m, _a, o):
        holder["h"] = o[0] if isinstance(o, tuple) else o
    handle = blk.register_forward_hook(cap)
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    handle.remove()
    return holder["h"]

think_re = re.compile(r"<think>.*?</think>", re.S)
think_open = re.compile(r"<think>.*$", re.S)
bleed_re = re.compile(r"\n\s*(?:user|assistant|system)\s*\n|<\|\s*im_start\s*\|>", re.I)

def sanitize(text):
    """Strip leaked think blocks and transcript bleed from a clamped reply."""
    text = think_re.sub("", text)
    text = think_open.sub("", text)
    text = bleed_re.split(text)[0]
    return text.strip()

STOP = ["\nuser\n", "\nassistant\n", "\nUser:", "\nAssistant:", "<|im_start|>"]

def render(messages, thinking=False):
    out = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  enable_thinking=thinking, tokenize=True,
                                  return_tensors="pt")
    ids = out.input_ids if hasattr(out, "input_ids") else out
    return ids.to("cuda")

def reply(messages, steered, thinking=False):
    global ids
    ids = render(messages, thinking)
    attn = torch.ones_like(ids)
    # probe medians on clean forward
    for L in TARGETS:
        meds[L] = float(block_out(blocks[L])[0, 1:, :].float().norm(dim=-1).median())
    current_doses()
    dose_vec.clear()
    if steered:
        dose_vec.update(doses)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True,
                                    timeout=600)
    gen_kw = dict(input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                  do_sample=False, use_cache=True, streamer=streamer,
                  stop_strings=STOP, tokenizer=tok, no_repeat_ngram_size=8)
    th = threading.Thread(target=model.generate, kwargs=gen_kw, daemon=True)
    th.start()
    text, shown, muted = "", 0, False
    for chunk in streamer:
        text += chunk
        if not muted:
            if bleed_re.search(text):
                muted = True
                prefix = sanitize(text)
                print(prefix[shown:], end="", flush=True)
            else:
                print(chunk, end="", flush=True)
                shown = len(text)
    th.join()
    dose_vec.clear()
    return sanitize(text)

history = []
steered = True
both_next = False
multi = False          # single-turn by default: no history carry-over
meds = {}

print(f"[ready] targets={ {L: TARGETS[L] for L in TARGETS} }  "
      f"steer={'ON' if steered else 'OFF'}  history={'multi' if multi else 'single-turn'}"
      f"  (type /quit to exit, /targets for help)")
while True:
    try:
        msg = input("\n\033[1myou>\033[0m ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not msg:
        continue
    if msg.startswith("/"):
        parts = msg.split()
        cmd = parts[0].lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/steer":
            steered = True; print("[mode] steering ON")
        elif cmd == "/clean":
            steered = False; print("[mode] steering OFF")
        elif cmd == "/both":
            both_next = True; print("[mode] next turn: clean + steered")
        elif cmd == "/multi":
            multi = True; print("[chat] multi-turn history ON")
        elif cmd == "/single":
            multi = False; history.clear(); print("[chat] single-turn, history cleared")
        elif cmd == "/reset":
            history.clear(); print("[chat] history cleared")
        elif cmd == "/targets":
            for L, (n, d) in TARGETS.items():
                print(f"  L{L} N{n}  dose={d:+.2f}  ||W_down[:,n]||={col_norm[n]:.4f}")
            print(f"  steered={'ON' if steered else 'OFF'}  max_new={MAX_NEW}"
                  f"  history={'multi' if multi else 'single-turn'}")
        elif cmd == "/dose" and len(parts) == 4:
            L, n, d = int(parts[1]), int(parts[2]), float(parts[3])
            if L in TARGETS and TARGETS[L][0] == n:
                TARGETS[L] = (n, d); print(f"[dose] L{L} N{n} -> {d:+.2f}")
            else:
                print("[dose] unknown layer/neuron; see /targets")
        else:
            print("[cmd] unknown. /steer /clean /both /multi /single /targets /dose L N V /reset /quit")
        continue

    if not multi:
        history.clear()
    history.append({"role": "user", "content": msg})
    while True:                     # trim to context budget
        r = render(history)
        if r.shape[1] <= CTX_TOKENS:
            break
        history.pop(1 if len(history) > 1 else 0)

    if both_next:
        print("\n\033[36m[clean]\033[0m ", end="", flush=True)
        clean = reply(history, steered=False)
        print("\n\033[35m[steered]\033[0m ", end="", flush=True)
        ans = reply(history, steered=True)
        both_next = False
        history.append({"role": "assistant", "content": ans})
    else:
        color = "\033[35m" if steered else "\033[36m"
        label = "steered" if steered else "clean"
        print(f"\n{color}[{label}]\033[0m ", end="", flush=True)
        ans = reply(history, steered=steered)
        history.append({"role": "assistant", "content": ans})
    print()
