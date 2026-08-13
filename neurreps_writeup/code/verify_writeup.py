"""Assert every number quoted in writeup_neurreps.tex against logs/.

Run:  python code/verify_writeup.py
Any mismatch between the prose and the data is a hard failure here, so this is the
check to run after editing either.
"""
import math, re, sys
from pathlib import Path
from common import RUNS, load, load_logit, final_val, val_at, quad_fit

TEX = Path(__file__).resolve().parent.parent / "writeup_neurreps.tex"
src = TEX.read_text(encoding="utf-8")
fails, checks = [], 0

def eq(label, got, want, tol=5e-5):
    global checks
    checks += 1
    if abs(got - want) > tol:
        fails.append(f"{label}: data says {got:.6g}, paper says {want:.6g}")

def in_tex(label, *needles):
    """The paper must literally contain each needle."""
    global checks
    for n in needles:
        checks += 1
        if n not in src:
            fails.append(f"{label}: paper does not contain {n!r}")

# section 2: the sweep
sweep = ["qkn_7p5e4", "qkn_1p5e3", "qkn_2p55e3"]
vals = [final_val(k)[0] for k in sweep]
A, B, C, vx = quad_fit([RUNS[k]["lr"] for k in sweep], vals)
eq("best swept point", min(vals), 2.5403)
eq("fitted vertex (e-3)", vx * 1e3, 1.56, tol=5e-3)
eq("curvature", A, 0.0276, tol=5e-5)
anchor = final_val("qkn_anchor")[0]
eq("anchor value", anchor, 2.6165)
eq("gap anchor to optimum", anchor - min(vals), 0.076, tol=5e-4)
in_tex("sweep prose", "2.5403", "0.076", "1.56\\times10^{-3}", "0.0276")

# the "N times the residual seed spread" multiple
res3 = [final_val(k)[0] for k in ("resg_s05_43", "resg_s05_44", "resg_s05_45")]
spread3 = max(res3) - min(res3)
mult = (anchor - min(vals)) / spread3
m = re.search(r"roughly \$(\d+)\$ times the entire spread", src)
if m:
    eq("multiple of the 3-seed spread", mult, float(m.group(1)), tol=1.0)
else:
    fails.append("could not find the 'roughly N times the entire spread' claim")

# residual runs range, and the matched-LR tie
eq("residual min", min(res3), 2.5464)
eq("residual max", max(res3), 2.5487)
eq("matched-LR tie", abs(final_val("qkn_2p55e3")[0] - final_val("resg_s05_43")[0]), 0.0003)
in_tex("tie prose", "2.5469", "2.5472", "0.0003", "2.5464", "2.5487")

# section 3: arm-dependent seed spread
d_lin = abs(final_val("qkn_1p5e3")[0] - final_val("qkn_1p5e3_s44")[0])
d_res = abs(final_val("resg_s05_43")[0] - final_val("resg_s05_44")[0])
eq("linear seed spread", d_lin, 0.0002)
eq("residual seed spread", d_res, 0.0015)
eq("ratio of spreads", d_res / d_lin, 7.5, tol=0.05)
eq("3-seed spread", spread3, 0.0023)
st1, _, v1 = load("qkn_1p5e3"); st2, _, v2 = load("qkn_1p5e3_s44")
last11 = [s for s in st1 if s in st2][-11:]
maxd = max(abs(v1[st1.index(s)] - v2[st2.index(s)]) for s in last11)
eq("max paired diff, last 11 evals", maxd, 0.0013)
in_tex("seed prose", "2.5405", "0.0002", "0.0015", "0.0023", "7.5", "0.0013")

# the Figure 3(b) caption claim: the gap between arm means dwarfs the smaller spread,
# which is why that panel is centred per arm rather than plotted on an absolute axis
mean_lin = (final_val("qkn_1p5e3")[0] + final_val("qkn_1p5e3_s44")[0]) / 2
mean_res = sum(res3) / 3
eq("gap between arm means over smaller spread", (mean_res - mean_lin) / d_lin, 35, tol=1.0)
in_tex("figure 3 caption", "$35$ times the smaller")

# section 4: the per-head variant
lin, non = final_val("qkn_1p5e3")[0], final_val("resg_s1")[0]
eq("variant value", non, 2.5383)
eq("variant margin", lin - non, 0.0020)
eq("margin in linear sigmas", (lin - non) / d_lin, 10, tol=0.6)
eq("margin in residual sigmas", (lin - non) / d_res, 1.3, tol=0.05)
sa, _, va = load("qkn_1p5e3"); sb, _, vb = load("resg_s1")
cross = [(s, vb[sb.index(s)] - va[sa.index(s)]) for s in sa if s in sb and s > 0]
behind = [s for s, d in cross if d > 0]
eq("evals residual is behind", len(behind), 18, tol=0.5)
eq("total evals compared", len(cross), 60, tol=0.5)
eq("first behind step", min(behind), 2000, tol=0.5)
eq("last behind step", max(behind), 21000, tol=0.5)
in_tex("variant prose", "2.5383", "0.0020", "$18$ of $60$", "step $2000$", "step $21000$")

# section 6: D2Z stability
st, _, va = load("d2z_1e3")
mn = min(va); at = st[va.index(mn)]
eq("d2z 1e-3 min", mn, 2.8393)
eq("d2z 1e-3 min step", at, 20000, tol=0.5)
eq("d2z 1e-3 guard step", st[-1], 25000, tol=0.5)
s2, _, v2b = load("d2z_2e3")
eq("d2z 2e-3 last step", s2[-1], 60000, tol=0.5)
eq("d2z 2e-3 final", v2b[-1], 2.5504)
_, l12a, _ = load_logit("d2z_1e3"); _, l12b, _ = load_logit("d2z_2e3")
eq("logit peak, diverged run", max(l12a), 274, tol=0.5)
eq("logit peak, survivor", max(l12b), 486, tol=0.5)
in_tex("d2z prose", "2.8393", "$20$k", "$25$k", "$2.5504$", "$486$", "$274$")

# the recovery claim: both deltas must be measured from the plain arm's own baseline,
# 8952, and NOT from the QK-norm anchor. mixing them was a real error in an earlier draft.
plain = final_val("plain_42")[0]
eq("plain baseline", plain, 2.6247)
eq("plain to swept optimum", plain - min(vals), 0.0844, tol=5e-5)
eq("plain to d2z survivor", plain - v2b[-1], 0.0743, tol=5e-5)
eq("fraction recovered (%)", (plain - v2b[-1]) / (plain - min(vals)) * 100, 88, tol=0.6)
checks += 1
if v2b[-1] >= final_val("qkn_7p5e4")[0]:
    fails.append("paper says the d2z survivor beats the 7.5e-4 sweep point; it does not")
in_tex("recovery prose", "2.6247", "0.0844", "0.0743", "88\\%", "2.5551")

# sections 5 and 7: budgets, epochs, horizon slope
N, CORPUS, TOK = 124_373_760, 9.036e9, 480 * 1024
eq("tokens per step", TOK, 491_520)
eq("60k tokens (B)", 60002 * TOK / 1e9, 29.5, tol=0.05)
eq("180k tokens (B)", 180000 * TOK / 1e9, 88.5, tol=0.05)
eq("60k TPP", 60002 * TOK / N, 237, tol=0.5)
eq("180k TPP", 180000 * TOK / N, 711, tol=0.5)
eq("60k epochs", 60002 * TOK / CORPUS, 3.3, tol=0.05)
eq("180k epochs", 180000 * TOK / CORPUS, 9.8, tol=0.05)
BJ = [(25, 1.34e-3), (50, 1.02e-3), (100, 6.60e-4)]
lx = [math.log(d) for d, _ in BJ]; ly = [math.log(v) for _, v in BJ]
mx, my = sum(lx) / 3, sum(ly) / 3
beta = -sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)
eq("local horizon slope", beta, 0.51, tol=5e-3)
eq("predicted shift", 3.0 ** beta, 1.75, tol=5e-3)
eq("cost of not moving", A * math.log(3.0 ** beta) ** 2, 0.0087, tol=5e-5)
eq("cost in linear sigmas", A * math.log(3.0 ** beta) ** 2 / d_lin, 43, tol=1.0)
in_tex("horizon prose", "0.51", "1.75", "0.0087", "$43", "0.6531", "$237$", "$711$",
       "9.036", "$3.3$", "$9.8$", "491{,}520")
# DeepSeek arithmetic quoted in section 5
eq("deepseek tpp exponent", 0.4757 - 0.5243, -0.049, tol=5e-4)
eq("deepseek eta drop", 10 ** (math.log10(2e19 / 1e17) * 0.125), 1.94, tol=5e-3)
in_tex("deepseek prose", "0.5243", "0.4757", "C^{-0.049}", "1.94")

# section 4: the win is on train loss too, not a generalization effect
tr_lin = load("qkn_1p5e3")[1][-1]
tr_non = load("resg_s1")[1][-1]
eq("train-loss lead of the variant", tr_lin - tr_non, 0.0028)
eq("linear train-to-val gap", lin - tr_lin, 0.0463)
eq("variant train-to-val gap", non - tr_non, 0.0471)
checks += 1
if not (non - tr_non) > (lin - tr_lin):
    fails.append("paper says the variant's train-to-val gap is the wider one; it is not")
in_tex("train-side prose", "0.0028", "0.0471", "0.0463")

# section 7: the small sweep, all five finished
sm = ["sm_6e4", "sm_1p2e3", "sm_2p4e3", "sm_4p8e3", "sm_9p6e3"]
for k in sm:
    checks += 1
    if load(k)[0][-1] < 60000:
        fails.append(f"{RUNS[k]['job']} has not reached 60000; section 7 assumes it has")
finals = {RUNS[k]["lr"]: final_val(k)[0] for k in sm}
eq("small 6e-4 final", finals[6e-4], 2.9311)
eq("small 1.2e-3 final", finals[1.2e-3], 2.9053)
eq("small 2.4e-3 final", finals[2.4e-3], 2.8994)
eq("small 4.8e-3 final", finals[4.8e-3], 2.8948)
eq("small 9.6e-3 final", finals[9.6e-3], 2.9103)
# the section's claim is now that the bowl is bracketed with an interior minimum
lrs = sorted(finals)
best = min(finals, key=finals.get)
checks += 1
if best in (min(lrs), max(lrs)):
    fails.append("paper claims an interior minimum; the best point is at a grid edge")
eq("9.6e-3 penalty vs 4.8e-3", finals[9.6e-3] - finals[4.8e-3], 0.0155, tol=5e-5)
As, Bs, Cs, vs_ = quad_fit([2.4e-3, 4.8e-3, 9.6e-3],
                           [finals[2.4e-3], finals[4.8e-3], finals[9.6e-3]])
eq("small fitted vertex (e-3)", vs_ * 1e3, 3.98, tol=5e-3)
eq("small curvature", As, 0.0209, tol=5e-5)
eq("small bowl vs large bowl", As / A, 0.76, tol=5e-3)
eq("grid centre low by", vs_ / 2.4e-3, 1.7, tol=0.05)
eq("180k predicted optimum (e-3)", vs_ / 1.75 * 1e3, 2.27, tol=0.02)
# the mid-anneal reversal the section leans on: at 36k the order is monotone the other
# way, the 36k leader finishes last, and the eventual winner is still 4th of 5 at 54k
at36 = {lr: val_at(k, 36000) for k, lr in ((k, RUNS[k]["lr"]) for k in sm)}
eq("36k 6e-4", at36[6e-4], 3.0348)
eq("36k 9.6e-3", at36[9.6e-3], 3.1948)
checks += 1
if not all(at36[lrs[i]] < at36[lrs[i + 1]] for i in range(len(lrs) - 1)):
    fails.append("paper says step 36k is monotone increasing in lr; it is not")
checks += 1
if min(at36, key=at36.get) != max(finals, key=finals.get):
    fails.append("paper says the 36k leader finishes last; it does not")
at54 = {lr: val_at(k, 54000) for k, lr in ((k, RUNS[k]["lr"]) for k in sm)}
eq("winner's rank at 54k", sorted(at54, key=at54.get).index(best) + 1, 4, tol=0.5)
# the 180k arm is under way and past 60k, which is what the section says
st180 = load("sm180_6e4")[0]
checks += 1
if st180[-1] < 60000:
    fails.append("section 7 says the 180k run is past step 60k; it is not")
checks += 1
if st180[-1] >= 180000:
    fails.append("section 7 calls the 180k run under way, but it has finished")
in_tex("small prose", "2.9311", "2.9053", "2.8994", "2.8948", "2.9103",
       "3.0348", "3.1948", "$36$k", "$54$k",
       "3.98\\times10^{-3}", "0.0209", "2.3\\times10^{-3}",
       "4.8\\times10^{-3}", "9.6\\times10^{-3}")

# figures referenced all exist
FIGS = Path(__file__).resolve().parent.parent / "figures"
for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src):
    checks += 1
    if not (FIGS / m.group(1)).exists():
        fails.append(f"figure referenced but missing: {m.group(1)}")

print(f"{checks} checks")
if fails:
    print(f"\n{len(fails)} FAILURES:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all quoted numbers match the logs")
