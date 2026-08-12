"""Rebuild the write-up CSVs from .out files I fetched off the cluster with sftp.

extract_logs.sh does the same thing on the login node. This one is for when the
login shell is hung and sftp is the only way in, so the .out files land locally
and get parsed here.

    python refresh_logs.py <dir with .out files> <writeup logs dir>
"""
import re
import sys
from pathlib import Path

EVAL = re.compile(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)", re.M)
LOGIT = re.compile(r"attn_logit_max ([\d.]+)\s+per-block L0=([\d.]+) L\d+=([\d.]+) L\d+=([\d.]+)")

# .out file, val-loss csv, attention-logit csv (None if the run has no probe)
JOBS = [
    ("d2z_plain_lr1e3_10140.out",      "gpt3large_linear_d2z_lr1e3.csv",   "logit_d2z_lr1e3.csv"),
    ("d2z_plain_lr2e3_10141.out",      "gpt3large_linear_d2z_lr2e3.csv",   "logit_d2z_lr2e3.csv"),
    ("p3s_d2z_lr9p6e3_60k_10134.out",  "gpt3small_d2z_lr9p6e3_60k.csv",    None),
    ("p3s_d2z_lr6e4_180k_10135.out",   "gpt3small_d2z_lr6e4_180k.csv",     None),
]

src, logs = Path(sys.argv[1]), Path(sys.argv[2])

for out, csv_name, logit_name in JOBS:
    if not (src / out).exists():
        print(f"skipping {out}, not fetched")
        continue
    raw = (src / out).read_text(errors="ignore")
    # dict first: a resumed run repeats steps, and the later line is the one I want
    rows = sorted({int(m[0]): (m[1], m[2]) for m in EVAL.findall(raw)}.items())
    with open(logs / csv_name, "w", newline="") as f:
        f.write("step,train_loss,val_loss\n")
        for step, (tr, va) in rows:
            f.write(f"{step},{tr},{va}\n")
    print(f"{csv_name:<36} {len(rows):>4} rows, last step {rows[-1][0]}")

    if logit_name:
        steps = [s for s, _ in rows]
        hits = LOGIT.findall(raw)
        n = min(len(steps), len(hits))
        with open(logs / logit_name, "w", newline="") as f:
            f.write("step,L0,L12,L23,max\n")
            for s, h in zip(steps[:n], hits[:n]):
                f.write(f"{s},{h[1]},{h[2]},{h[3]},{h[0]}\n")
        print(f"{logit_name:<36} {n:>4} rows")
