"""Run table, log loading and plot style. Shared by analysis.py and make_figures.py."""
from pathlib import Path
import csv
import math

LOGS = Path(__file__).resolve().parent.parent / "logs"
FIGS = Path(__file__).resolve().parent.parent / "figures"

# Settings below were read off each job's logged config dict, not its filename.
# s is the multiplier on the standard 1/sqrt(d_head) attention scale. s=0.5 is not a
# stray difference: the code computes q = x + LN(f(x)) with no explicit factor, so the
# 1/2 in Q = (1/2)(x + f(x)) lives in the scale. 9944 drops it and normalises per head.
RUNS = {
    # GPT-3 Large, linear query + QK-norm: the learning-rate sweep
    "qkn_7p5e4": dict(log="gpt3large_linear_qknorm_lr7p5e4.csv", job=9776,
                      model="GPT-3 Large", query="linear", mit="QK-norm",
                      lr=7.5e-4, min_lr=1e-5, s=1.0, seed=43, sched="cosine",
                      steps=60002, state="completed"),
    "qkn_1p5e3": dict(log="gpt3large_linear_qknorm_lr1p5e3_seed43.csv", job=9777,
                      model="GPT-3 Large", query="linear", mit="QK-norm",
                      lr=1.5e-3, min_lr=1e-5, s=1.0, seed=43, sched="cosine",
                      steps=60002, state="completed"),
    "qkn_1p5e3_s44": dict(log="gpt3large_linear_qknorm_lr1p5e3_seed44.csv", job=9924,
                      model="GPT-3 Large", query="linear", mit="QK-norm",
                      lr=1.5e-3, min_lr=1e-5, s=1.0, seed=44, sched="cosine",
                      steps=60002, state="completed"),
    "qkn_2p55e3": dict(log="gpt3large_linear_qknorm_lr2p55e3.csv", job=9778,
                      model="GPT-3 Large", query="linear", mit="QK-norm",
                      lr=2.55e-3, min_lr=1e-5, s=1.0, seed=43, sched="cosine",
                      steps=60002, state="completed"),
    # same arm, different LR floor, so off the sweep's protocol and out of the fit
    "qkn_anchor": dict(log="gpt3large_linear_qknorm_lr2p5e4_anchor.csv", job=9645,
                      model="GPT-3 Large", query="linear", mit="QK-norm",
                      lr=2.5e-4, min_lr=2.5e-5, s=1.0, seed=43, sched="cosine",
                      steps=60002, state="completed"),

    # GPT-3 Large, residual-GELU
    "resg_s1": dict(log="gpt3large_resgelu_knormph_s1_lr1p5e3.csv", job=9944,
                      model="GPT-3 Large", query="residual_gelu", mit="k-norm per-head",
                      lr=1.5e-3, min_lr=1e-5, s=1.0, seed=43, sched="cosine",
                      steps=60002, state="completed"),
    "resg_s05_43": dict(log="gpt3large_resgelu_s0p5_seed43.csv", job=9436,
                      model="GPT-3 Large", query="residual_gelu", mit="none",
                      lr=2.55e-3, min_lr=2e-5, s=0.5, seed=43, sched="cosine",
                      steps=60002, state="completed"),
    "resg_s05_44": dict(log="gpt3large_resgelu_s0p5_seed44.csv", job=9647,
                      model="GPT-3 Large", query="residual_gelu", mit="none",
                      lr=2.55e-3, min_lr=2e-5, s=0.5, seed=44, sched="cosine",
                      steps=60002, state="completed"),
    "resg_s05_45": dict(log="gpt3large_resgelu_s0p5_seed45.csv", job="9780+10127",
                      model="GPT-3 Large", query="residual_gelu", mit="none",
                      lr=2.55e-3, min_lr=2e-5, s=0.5, seed=45, sched="cosine",
                      steps=60002, state="completed (resumed at 47k)"),

    # GPT-3 Large, plain linear, no mitigation
    "plain_42": dict(log="gpt3large_linear_plain_seed42.csv", job=8952,
                      model="GPT-3 Large", query="linear", mit="none",
                      lr=2.5e-4, min_lr=2.5e-5, s=1.0, seed=42, sched="cosine",
                      steps=60002, state="completed"),
    "plain_43_div": dict(log="gpt3large_linear_plain_seed43_diverged.csv", job=9435,
                      model="GPT-3 Large", query="linear", mit="none",
                      lr=2.5e-4, min_lr=2.5e-5, s=1.0, seed=43, sched="cosine",
                      steps=60002, state="diverged"),
    "plain_44_div": dict(log="gpt3large_linear_plain_seed44_diverged.csv", job=9779,
                      model="GPT-3 Large", query="linear", mit="none",
                      lr=2.5e-4, min_lr=2.5e-5, s=1.0, seed=44, sched="cosine",
                      steps=60002, state="diverged"),

    # GPT-3 Large, plain linear under decay-to-zero
    "d2z_1e3": dict(log="gpt3large_linear_d2z_lr1e3.csv", job=10140,
                      model="GPT-3 Large", query="linear", mit="none",
                      lr=1e-3, min_lr=0.0, s=1.0, seed=43, sched="linear (D2Z)",
                      steps=60002, state="diverged, guard stopped at 25k"),
    "d2z_2e3": dict(log="gpt3large_linear_d2z_lr2e3.csv", job=10141,
                      model="GPT-3 Large", query="linear", mit="none",
                      lr=2e-3, min_lr=0.0, s=1.0, seed=43, sched="linear (D2Z)",
                      steps=60002, state="completed"),

    # GPT-3 small, decay-to-zero learning-rate sweep at 60k
    "sm_6e4":   dict(log="gpt3small_d2z_lr6e4_60k.csv", job=10130, model="GPT-3 small",
                     query="linear", mit="none", lr=6e-4, min_lr=0.0, s=1.0,
                     seed="default", sched="linear (D2Z)", steps=60002, state="completed"),
    "sm_1p2e3": dict(log="gpt3small_d2z_lr1p2e3_60k.csv", job=10131, model="GPT-3 small",
                     query="linear", mit="none", lr=1.2e-3, min_lr=0.0, s=1.0,
                     seed="default", sched="linear (D2Z)", steps=60002, state="completed"),
    "sm_2p4e3": dict(log="gpt3small_d2z_lr2p4e3_60k.csv", job=10132, model="GPT-3 small",
                     query="linear", mit="none", lr=2.4e-3, min_lr=0.0, s=1.0,
                     seed="default", sched="linear (D2Z)", steps=60002, state="completed"),
    "sm_4p8e3": dict(log="gpt3small_d2z_lr4p8e3_60k.csv", job=10133, model="GPT-3 small",
                     query="linear", mit="none", lr=4.8e-3, min_lr=0.0, s=1.0,
                     seed="default", sched="linear (D2Z)", steps=60002, state="completed"),
    "sm_9p6e3": dict(log="gpt3small_d2z_lr9p6e3_60k.csv", job=10134, model="GPT-3 small",
                     query="linear", mit="none", lr=9.6e-3, min_lr=0.0, s=1.0,
                     seed="default", sched="linear (D2Z)", steps=60002, state="completed"),

    # first of the 180k arm, still running
    "sm180_6e4": dict(log="gpt3small_d2z_lr6e4_180k.csv", job=10135, model="GPT-3 small",
                     query="linear", mit="none", lr=6e-4, min_lr=0.0, s=1.0,
                     seed="default", sched="linear (D2Z)", steps=180000, state="running"),
}

LOGIT = {"d2z_1e3": "logit_d2z_lr1e3.csv", "d2z_2e3": "logit_d2z_lr2e3.csv"}


def load(key):
    """steps, train, val for a RUNS key."""
    st, tr, va = [], [], []
    with open(LOGS / RUNS[key]["log"], newline="") as f:
        for row in csv.DictReader(f):
            st.append(int(row["step"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
    return st, tr, va


def load_logit(key):
    """steps, layer-12 max, overall max for a D2Z run's attention-logit probe."""
    st, l12, mx = [], [], []
    with open(LOGS / LOGIT[key], newline="") as f:
        for row in csv.DictReader(f):
            st.append(int(row["step"]))
            l12.append(float(row["L12"]))
            mx.append(float(row["max"]))
    return st, l12, mx


def val_at(key, step):
    st, _, va = load(key)
    return va[st.index(step)] if step in st else None


def final_val(key):
    st, _, va = load(key)
    return va[-1], st[-1]


def quad_fit(lrs, vals):
    """Quadratic through three (lr, loss) points in ln(lr). Returns A, B, C, vertex.

    A converts a factor-f error in the learning rate into a loss penalty A (ln f)^2.
    """
    x1, x2, x3 = (math.log(v) for v in lrs)
    y1, y2, y3 = vals
    A = (y1 / ((x1 - x2) * (x1 - x3)) + y2 / ((x2 - x1) * (x2 - x3))
         + y3 / ((x3 - x1) * (x3 - x2)))
    B = -(y1 * (x2 + x3) / ((x1 - x2) * (x1 - x3))
          + y2 * (x1 + x3) / ((x2 - x1) * (x2 - x3))
          + y3 * (x1 + x2) / ((x3 - x1) * (x3 - x2)))
    return A, B, y2 - A * x2 * x2 - B * x2, math.exp(-B / (2 * A))


# Plot style, matching Figure 1 of the foundational study: white ground, thin full box,
# dotted grid, serif type, legend below the axes.
FRAME = "#333333"
BASE = "#1a1a1a"
MUTED = "#8a8a8a"
CAT = ["#2a78d6", "#eb6834"]                                   # linear arm, residual arm
SEQ = {250: "#86b6ef", 350: "#5598e7", 450: "#2a78d6",         # learning-rate ladder
       550: "#1c5cab", 650: "#104281"}
CRITICAL = "#d03b3b"                                           # diverged


def style():
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.edgecolor": FRAME, "axes.linewidth": 0.6,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#b4b4b4", "grid.linestyle": ":", "grid.linewidth": 0.5,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5, "axes.labelcolor": BASE,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.color": FRAME, "ytick.color": FRAME,
        "xtick.labelcolor": BASE, "ytick.labelcolor": BASE,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.1, "lines.markersize": 3.6,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "legend.labelcolor": BASE, "legend.handlelength": 2.2,
        "legend.columnspacing": 1.4, "legend.handletextpad": 0.5,
        "pdf.fonttype": 42,
    })
