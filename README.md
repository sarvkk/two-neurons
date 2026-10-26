# evil-qwen: two neurons steer Qwen3.5-4B's safety behavior

Replication of @_lyraaaa_'s tweet on turning Qwen3.5-4B "evil" by clamping two mid-layer MLP
neurons (https://x.com/_lyraaaa_/status/2113833330429083662). Written up in plainer prose on
the blog (https://sarvajit.com.np).

The persona flip reproduces on a 4 GB GPU, and a 640-generation benchmark pass
(SimpleSafetyTests, XSTest, single-neuron and shifted-layer ablations, three prompt
wrappers) makes it quantitative: refusal 90% to 8%, judge-flagged unsafe output 1% to 96%.
One neuron, L14 N8949 at +1, produces almost the whole effect on its own.

Author: https://github.com/sarvkk

The full public write-up lives on the blog (https://sarvajit.com.np). The full technical
report (method, per-area tables, judge reliability) is `evil-qwen.md` / `evil-qwen.pdf`.

## What is and is not in this repo

Every generation in this experiment was either a refusal or a harmful request answered in
character. Publishing raw model outputs would mean publishing a folder of harmful text, so
the following are kept local and are not part of this repository: `bench_runs.jsonl`,
`bench_variants.jsonl`, `bench_scored.jsonl` (generations plus prompts), `raw_results.jsonl`
(pilot generations), and `audit_sample.md` (hand-audit sample with full responses).

What we publish instead:

- aggregate rates and categories (`bench_summary.md`, `summary.md`)
- per-response judge labels with no text (`bench_guard.jsonl`: safety label, trained refusal
  flag, guard category, keyed by dataset/id/condition)
- the complete pipeline code, so anyone can regenerate and check the numbers on their own
  hardware

Models used: `Qwen/Qwen3.5-4B` (NF4 4-bit via transformers) and `Qwen/Qwen3Guard-Gen-0.6B`
plus `unitary/toxic-bert` as judges.

## The recipe

```
s = dose x median(||resid|| leaving layer L, tokens 1:) / ||W_down[:, n]||
```

Targets: L14 N8949 at +1, L13 N7956 at -1 (0-indexed layers). The clamp overwrites the
neuron's post-activation at every token (prefill and generation), in units of the layer's
residual-stream norm. See `evil-qwen.md` for the full derivation and the residual-delta
verification.

## Reproduce

Hardware used: GTX 1650 laptop GPU, 4 GB VRAM. The 4-bit model peaks at ~3 GB resident;
generation runs at ~12 tok/s (~6 s per 72-token answer); the full 640-generation sweep takes
about an hour.

```bash
export HF_HOME=$HOME/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# pilot: 10 prompts x clean/steered/placebo, 150 tokens
python -u run_experiment.py
python summarize.py

# benchmark generation (resumable), then judging
python -u bench_eval.py
python -u bench_variants.py
python -u bench_guard.py
python -u bench_grade.py
```

Interactive steering chat (single-turn by default, `/dose`, `/clean`, `/both`, `/targets`):
`chat_steer.py`.

Requirements: a Python environment with `torch`, `transformers`, `accelerate`,
`bitsandbytes`, `pyarrow`, `huggingface_hub`.

## Data

`data/` holds the benchmark prompts verbatim from their public sources, re-encoded as JSONL
with stable ids:

- SimpleSafetyTests: https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests
  (`sst_test_cases.csv`, 100 prompts, 5 harm areas)
- XSTest (safe subset): https://huggingface.co/datasets/LLMSafety/XSTest

The prompt sets are standard published test suites; the repo adds no prompts of its own.

## License

MIT. The benchmark prompts belong to their original authors, cited above.
