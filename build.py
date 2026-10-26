#!/usr/bin/env python3
"""Build evil-qwen.html (styled) and evil-qwen.pdf from evil-qwen.md."""
import os
import re
import subprocess

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "evil-qwen.md")
HTML = os.path.join(HERE, "evil-qwen.html")
PDF = os.path.join(HERE, "evil-qwen.pdf")

md = open(SRC, encoding="utf-8").read()
m = re.match(r"#\s+(.+)", md)
title = m.group(1).strip() if m else "Evil Qwen"
if m:
    md = md[m.end():]

body = markdown.markdown(
    md,
    extensions=["tables", "fenced_code", "sane_lists", "smarty", "attr_list"],
)

css = """
:root{--ink:#1c1917;--muted:#6b625c;--accent:#a4161a;--bg:#fdfcf9;
--rule:#e7e1d8;--code-bg:#f3efe8}
*{box-sizing:border-box}
@page{size:A4;margin:19mm 17mm 21mm}
body{font-family:Georgia,"Times New Roman",serif;background:var(--bg);
color:var(--ink);max-width:780px;margin:0 auto;padding:56px 30px 90px;
line-height:1.68;font-size:17px;-webkit-font-smoothing:antialiased}
h1,h2,h3,h4{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",
Helvetica,Arial,sans-serif;line-height:1.24;color:#111}
h1{font-size:2.15rem;letter-spacing:-.025em;margin:0 0 .35em;font-weight:750}
.meta{font-family:ui-sans-serif,system-ui,sans-serif;color:var(--muted);
font-size:.92rem;margin:0 0 2.4em;padding-bottom:1.2em;
border-bottom:1px solid var(--rule);line-height:1.6}
.meta b{color:var(--ink)}
h2{font-size:1.45rem;margin:2.5em 0 .6em;padding-bottom:.22em;
border-bottom:1px solid var(--rule);font-weight:700}
h3{font-size:1.1rem;margin:1.9em 0 .45em;font-weight:650}
p{margin:0 0 1.05em}
strong{color:#000;font-weight:700}
em{font-style:italic}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid #d9b3b3}
code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.86em;
background:var(--code-bg);padding:.12em .34em;border-radius:4px}
pre{background:#f6f2ea;border:1px solid var(--rule);border-radius:8px;
padding:14px 16px;overflow-x:auto;font-size:.83rem;line-height:1.55;
margin:1.3em 0}
pre code{background:none;padding:0;font-size:inherit}
table{border-collapse:collapse;width:100%;margin:1.5em 0;font-size:.84rem;
font-family:ui-sans-serif,system-ui,sans-serif;line-height:1.45}
th{text-align:left;border-bottom:2px solid #d6cfc3;padding:9px 10px;
background:#f5f1e9;font-weight:650;color:#111}
td{border-bottom:1px solid var(--rule);padding:8px 10px;vertical-align:top}
tbody tr:nth-child(even) td{background:#fbf8f3}
blockquote{margin:1.3em 0;padding:.2em 0 .2em 1.1em;border-left:3px solid var(--accent);
color:#3f3833;font-style:italic}
hr{border:none;border-top:1px solid var(--rule);margin:2.2em 0}
ul,ol{margin:0 0 1.1em;padding-left:1.4em}
li{margin:.28em 0}
.foot{margin-top:3em;padding-top:1.2em;border-top:1px solid var(--rule);
font-family:ui-sans-serif,system-ui,sans-serif;font-size:.8rem;color:var(--muted)}
@media print{
 body{max-width:none;padding:0;background:#fff;font-size:10.5pt;line-height:1.55}
 h1{font-size:23pt} h2{font-size:14pt} h3{font-size:11pt}
 pre,table,blockquote,tr{page-break-inside:avoid}
 h1,h2,h3{page-break-after:avoid}
 a{color:inherit;border-bottom:none}
 .foot{page-break-inside:avoid}
}
"""

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<article>
<h1>{title}</h1>
<p class="meta"><b>Qwen3.5-4B</b> · neuron steering experiment ·
targets <b>L14 N8949 (+1)</b> and <b>L13 N7956 (−1)</b> ·
670 logged generations scored by Qwen3Guard-Gen-0.6B · October 2026</p>
{body}
<p class="foot">Reproduce: <code>bench_eval.py</code> → <code>bench_guard.py</code> →
<code>bench_grade.py</code> · raw data: <code>bench_runs.jsonl</code>,
<code>raw_results.jsonl</code> · interactive: <code>chat_steer.py</code></p>
</article>
</body>
</html>
"""

open(HTML, "w", encoding="utf-8").write(html)
print("[html]", HTML)

cmd = [
    "chromium", "--headless", "--disable-gpu", "--no-sandbox",
    "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=4000", f"--print-to-pdf={PDF}",
    f"file://{HTML}",
]
try:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("[pdf] chromium stderr:", r.stderr[-800:])
except FileNotFoundError:
    print("[pdf] chromium not on PATH; trying chromium-browser")
    cmd[0] = "/usr/bin/chromium-browser"
    subprocess.run(cmd, capture_output=True, text=True, timeout=300)

if os.path.exists(PDF):
    print(f"[pdf] {PDF} ({os.path.getsize(PDF)/1024:.0f} KB)")
else:
    print("[pdf] FAILED")
