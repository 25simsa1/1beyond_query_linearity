"""Build the seven figures in figures/ from logs/. Run: python code/make_figures.py"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, FixedLocator, FuncFormatter
import numpy as np

from common import (RUNS, load, load_logit, final_val, quad_fit, style,
                    FIGS, CAT, SEQ, CRITICAL, BASE, FRAME, MUTED)

style()
FIGS.mkdir(exist_ok=True)
BLUE, ORANGE = CAT


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print(f"  {name}")


def below(fig, ncol, ax):
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
               ncol=ncol, frameon=False)


def step_axis(ax, label=True):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
    if label:
        ax.set_xlabel("Gradient steps ($\\times 10^{3}$)")


def lr_axis(ax, lrs):
    """Log axis ticked at the rates actually run. Minor-tick labels collide, so drop them."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(lrs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v * 1e3:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Peak learning rate ($\\times 10^{-3}$)")


def log_axis(ax, which, ticks, fmt="{:g}"):
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, p: fmt.format(v)))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def lr_bowl():
    sweep = ["qkn_7p5e4", "qkn_1p5e3", "qkn_2p55e3"]
    lrs = [RUNS[k]["lr"] for k in sweep]
    vals = [final_val(k)[0] for k in sweep]
    A, B, C, vx = quad_fit(lrs, vals)

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.45), layout="constrained")

    a.plot([RUNS["qkn_anchor"]["lr"]] + lrs, [final_val("qkn_anchor")[0]] + vals,
           ":", color=MUTED, lw=0.8, zorder=1)
    a.plot(lrs, vals, "o", color=BLUE, zorder=3, label="Swept, LR floor $10^{-5}$")
    a.plot([RUNS["qkn_anchor"]["lr"]], [final_val("qkn_anchor")[0]], "s", mfc="white",
           mec=BASE, mew=0.9, ms=4.2, zorder=3,
           label="Setting the comparison used, floor $2.5\\!\\times\\!10^{-5}$")
    lr_axis(a, [2.5e-4, 5e-4, 1e-3, 2e-3])
    a.set_ylabel("Final validation log-loss")
    a.set_ylim(2.532, 2.628)

    xs = np.logspace(math.log10(7.0e-4), math.log10(2.85e-3), 200)
    b.plot(xs, A * np.log(xs) ** 2 + B * np.log(xs) + C, "-", lw=0.9, color=BLUE,
           alpha=0.6, zorder=1, label="Quadratic in $\\ln$ LR")
    b.axvline(vx, ls="--", lw=0.7, color=MUTED, zorder=0)
    b.plot(lrs, vals, "o", color=BLUE, zorder=4)          # labelled in panel (a)
    b.plot([1.5e-3], [final_val("qkn_1p5e3_s44")[0]], "s", mfc="white", mec=BLUE,
           mew=0.9, ms=4.2, zorder=5, label="Same configuration, second ordering")
    b.plot([1.5e-3], [final_val("resg_s1")[0]], "D", color=ORANGE, ms=3.8, zorder=4,
           label="Residual-GELU + per-head, $s=1$")
    lr_axis(b, [0.75e-3, 1.5e-3, 2.55e-3])
    b.set_ylabel("Final validation log-loss")
    b.set_ylim(2.5335, 2.5605)

    ha, la = a.get_legend_handles_labels()
    hb, lb = b.get_legend_handles_labels()
    fig.legend(ha + hb, la + lb, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "lr_sweep_bowl.pdf")


def arm_matched():
    fig, (a, b) = plt.subplots(2, 1, figsize=(4.9, 3.7), sharex=True,
                               layout="constrained", height_ratios=[1.25, 1])

    # Full run in both panels. The final separation is 0.0020, unreadable on an
    # absolute axis, which is what the lower panel is for.
    for i, k in enumerate(["resg_s05_43", "resg_s05_44", "resg_s05_45"]):
        st, _, va = load(k)
        a.plot(st[1:], va[1:], "-", lw=0.6, color=MUTED, zorder=1,
               label="Residual-GELU, $s=0.5$, three orderings" if i == 0 else None)
    for k, col, ls, lab in [("qkn_1p5e3", BASE, "-", "Linear + QK-norm, $s=1$"),
                            ("resg_s1", ORANGE, "--", "Residual-GELU + per-head, $s=1$")]:
        st, _, va = load(k)
        a.plot(st[1:], va[1:], ls, color=col, lw=1.1, zorder=3, label=lab)
    a.set_ylabel("Validation log-loss")
    a.set_ylim(2.50, 3.42)

    sa, _, va = load("qkn_1p5e3")
    sb, _, vb = load("resg_s1")
    steps = [s for s in sa if s in sb and s > 0]
    diff = [(vb[sb.index(s)] - va[sa.index(s)]) * 1e3 for s in steps]
    b.axhline(0, lw=0.6, color=FRAME, zorder=2)
    b.plot(steps, diff, "-", lw=1.0, color=BASE, zorder=3)
    b.fill_between(steps, 0, diff, where=[d > 0 for d in diff], color=BASE,
                   alpha=0.10, lw=0, zorder=1, interpolate=True)
    b.fill_between(steps, 0, diff, where=[d <= 0 for d in diff], color=ORANGE,
                   alpha=0.16, lw=0, zorder=1, interpolate=True)
    b.set_ylabel("Residual $-$ linear ($\\times 10^{-3}$)")
    b.set_ylim(-10, 26)
    step_axis(b)

    below(fig, 3, a)
    save(fig, "arm_matched.pdf")


def training_curves():
    ladder = [("qkn_7p5e4", SEQ[250]), ("qkn_1p5e3", SEQ[450]), ("qkn_2p55e3", SEQ[650])]
    fig, (a, b) = plt.subplots(2, 1, figsize=(4.9, 3.7), sharex=True, layout="constrained")
    for k, col in ladder:
        st, tr, va = load(k)
        a.plot(st[1:], va[1:], "-", color=col, lw=1.1,
               label=f"$\\eta_{{\\max}} = {RUNS[k]['lr'] * 1e3:g}\\times 10^{{-3}}$")
        b.plot(st[1:], tr[1:], "-", color=col, lw=1.1)
    a.set_ylabel("Validation log-loss")
    b.set_ylabel("Train log-loss")
    for ax in (a, b):
        ax.set_ylim(2.45, 3.10)
    step_axis(b)
    below(fig, 3, a)
    save(fig, "training_curves.pdf")


def d2z_stability():
    fig, (a, b) = plt.subplots(2, 1, figsize=(4.9, 3.7), sharex=True, layout="constrained")
    for k, col, ls, lab in [
            ("d2z_2e3", BASE, "-", "$\\eta_{\\max} = 2\\times 10^{-3}$, ran to 60k"),
            ("d2z_1e3", CRITICAL, "--", "$\\eta_{\\max} = 1\\times 10^{-3}$, diverged")]:
        st, _, va = load(k)
        a.plot(st[1:], va[1:], ls, color=col, lw=1.1, label=lab)
        ls_, l12, _ = load_logit(k)
        b.plot(ls_[1:], l12[1:], ls, color=col, lw=1.1)
    st, _, va = load("d2z_1e3")
    a.plot([st[va.index(min(va))]], [min(va)], "v", color=CRITICAL, ms=3.6)
    log_axis(a, "y", [2.6, 3, 4, 5, 7])
    a.set_ylim(2.52, 9.2)
    a.set_ylabel("Validation log-loss")
    b.set_ylabel("Max attention logit, layer 12")
    b.set_ylim(0, 560)
    step_axis(b)
    below(fig, 2, a)
    save(fig, "d2z_stability.pdf")


def small_sweep():
    sm = [("sm_6e4", SEQ[250]), ("sm_1p2e3", SEQ[350]), ("sm_2p4e3", SEQ[450]),
          ("sm_4p8e3", SEQ[550]), ("sm_9p6e3", SEQ[650])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.45), layout="constrained")
    for k, col in sm:
        st, _, va = load(k)
        a.plot(st[1:], va[1:], "-", color=col, lw=1.0,
               label=f"${RUNS[k]['lr'] * 1e3:g}\\times 10^{{-3}}$")
    a.set_ylabel("Validation log-loss")
    a.set_ylim(2.88, 3.32)
    step_axis(a)

    lrs = [RUNS[k]["lr"] for k, _ in sm]
    fins = [final_val(k)[0] for k, _ in sm]

    # bracketed now, so fit the three around the minimum
    inner = ["sm_2p4e3", "sm_4p8e3", "sm_9p6e3"]
    A, B, C, vx = quad_fit([RUNS[k]["lr"] for k in inner],
                           [final_val(k)[0] for k in inner])
    xs = np.logspace(math.log10(2.1e-3), math.log10(1.1e-2), 200)
    b.plot(xs, A * np.log(xs) ** 2 + B * np.log(xs) + C, "-", lw=0.9, color=SEQ[450],
           alpha=0.6, zorder=1, label="Quadratic in $\\ln$ LR")
    b.axvline(vx, ls="--", lw=0.7, color=MUTED, zorder=0)
    for (k, col), lr, v in zip(sm, lrs, fins):
        b.plot([lr], [v], "o", ms=4.2, color=col, zorder=3)
    lr_axis(b, [0.6e-3, 1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3])
    b.set_ylabel("Final validation log-loss")
    b.set_ylim(2.8905, 2.9365)

    ha, la = a.get_legend_handles_labels()
    hb, lb = b.get_legend_handles_labels()
    fig.legend(ha + hb, la + lb, loc="outside lower center", ncol=6, frameon=False)
    save(fig, "small_sweep_60k.pdf")


def lr_vs_horizon():
    # Bjorck et al. (arXiv:2409.19913v3) Table 9: measured optimal LR for their 125m
    # model, which is our GPT-3 small geometry (12L/768/12 heads, base LR 6e-4).
    D = [25, 50, 100, 200, 400, 800]
    L = [1.34e-3, 1.02e-3, 6.60e-4, 4.12e-4, 2.51e-4, 1.98e-4]
    fig, ax = plt.subplots(figsize=(4.3, 2.7), layout="constrained")
    ax.plot(D, L, "o-", color=BASE, lw=1.1, ms=3.6,
            label="Measured optimum, their Table 9")

    # local slope over 25 to 100B, the segment holding our two budgets
    lx = [math.log(d) for d in D[:3]]
    ly = [math.log(v) for v in L[:3]]
    mx, my = sum(lx) / 3, sum(ly) / 3
    beta = -sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)
    icept = my + beta * mx
    xs = np.logspace(math.log10(22), math.log10(118), 50)
    ax.plot(xs, np.exp(icept - beta * np.log(xs)), "--", lw=1.0, color=BLUE,
            label=f"Local slope $\\beta = {beta:.2f}$ over 25 to 100B")
    for d in (29.5, 88.5):
        ax.axvline(d, ls=":", lw=0.8, color=CRITICAL, zorder=0)
    ax.plot([], [], ":", lw=0.8, color=CRITICAL, label="Our two budgets, 29.5B and 88.5B")

    log_axis(ax, "x", [25, 50, 100, 200, 400, 800])
    log_axis(ax, "y", [2e-4, 3e-4, 5e-4, 1e-3], fmt="{:.4g}")
    ax.set_ylim(1.7e-4, 1.75e-3)
    ax.set_xlabel("Token horizon (billions)")
    ax.set_ylabel("Optimal peak learning rate")
    below(fig, 2, ax)
    save(fig, "lr_vs_horizon.pdf")


def ordering():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5), layout="constrained",
                               width_ratios=[1.15, 1])

    for i, k in enumerate(["plain_43_div", "plain_44_div"]):
        st, _, va = load(k)
        a.plot(st[1:], va[1:], "--", color=CRITICAL, lw=1.0,
               label="Plain, orderings 43 and 44: stopped" if i == 0 else None)
    # these two land on top of each other, so the survivor goes down thick and pale
    st, _, va = load("plain_42")
    a.plot(st[1:], va[1:], "-", color=SEQ[250], lw=2.6, solid_capstyle="round",
           label="Plain, ordering 42: completed")
    st, _, va = load("qkn_anchor")
    a.plot(st[1:], va[1:], "-", color=BASE, lw=0.9,
           label="QK-norm, ordering 43: completed")
    log_axis(a, "y", [2.6, 3, 4, 5, 7])
    a.set_ylim(2.52, 9.2)
    a.set_ylabel("Validation log-loss")
    step_axis(a)

    # Spreads are 0.0002 and 0.0023 sitting on means 0.0070 apart, so centre each arm
    # on its own mean or the smaller spread is a single dot.
    for y, (lab, col, keys) in enumerate([
            ("Linear + QK-norm", BLUE, ["qkn_1p5e3", "qkn_1p5e3_s44"]),
            ("Residual-GELU", ORANGE, ["resg_s05_43", "resg_s05_44", "resg_s05_45"])]):
        vs = [final_val(k)[0] for k in keys]
        dev = [(v - sum(vs) / len(vs)) * 1e3 for v in vs]
        b.plot([min(dev), max(dev)], [y] * 2, "-", color=col, lw=1.0, zorder=2)
        b.plot(dev, [y] * len(dev), "o", color=col, ms=4.6, zorder=3)
        b.annotate(f"{lab}, spread {max(vs) - min(vs):.4f}", xy=(0, y + 0.26),
                   ha="center", color=BASE, fontsize=7.5)
    b.axvline(0, lw=0.6, color=FRAME, zorder=1)
    b.set_ylim(-0.55, 1.62)
    b.set_yticks([])
    b.set_xlim(-1.7, 1.7)
    b.set_xlabel("Deviation from that arm's mean ($\\times 10^{-3}$)")

    below(fig, 3, a)
    save(fig, "ordering_sensitivity.pdf")


if __name__ == "__main__":
    print("figures/")
    for f in (lr_bowl, arm_matched, training_curves, d2z_stability,
              small_sweep, lr_vs_horizon, ordering):
        f()
