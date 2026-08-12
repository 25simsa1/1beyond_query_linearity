"""Print the numbers the write-up quotes, from logs/. Run: python code/analysis.py"""
import math
from common import RUNS, load, load_logit, final_val, val_at, quad_fit


def head(t):
    print("\n" + t + "\n" + "-" * len(t))


head("Final validation loss, GPT-3 Large")
order = ["plain_42", "qkn_anchor", "qkn_7p5e4", "qkn_1p5e3", "qkn_1p5e3_s44",
         "qkn_2p55e3", "resg_s1", "resg_s05_43", "resg_s05_44", "resg_s05_45"]
print(f"{'job':>11}  {'query':<14} {'norm':<16} {'s':>4} {'peak LR':>9} "
      f"{'floor':>8} {'order':>5}  {'final':>8}")
for k in order:
    r = RUNS[k]
    print(f"{str(r['job']):>11}  {r['query']:<14} {r['mit']:<16} {r['s']:>4} "
          f"{r['lr']:>9.2e} {r['min_lr']:>8.1e} {str(r['seed']):>5}  "
          f"{final_val(k)[0]:>8.4f}")


head("Learning-rate sweep, stabilised linear arm (QK-norm, floor 1e-5, ordering 43)")
sweep = ["qkn_7p5e4", "qkn_1p5e3", "qkn_2p55e3"]
vals = [final_val(k)[0] for k in sweep]
for k, v in zip(sweep, vals):
    print(f"   {RUNS[k]['lr']:.2e}  ->  {v:.4f}")
A, B, C, vertex = quad_fit([RUNS[k]["lr"] for k in sweep], vals)
anchor = final_val("qkn_anchor")[0]
print(f"\n   curvature {A:.4f} loss per (ln lr)^2, optimum {vertex:.3e}")
print(f"   anchor at 2.5e-4 is {anchor:.4f}, so {anchor - min(vals):.4f} above the optimum")
for f in (1.3, 1.75, 2.0):
    print(f"   a factor {f} error costs {A * math.log(f) ** 2:.4f}")


head("Seed spread depends on the arm")
d_lin = abs(final_val("qkn_1p5e3")[0] - final_val("qkn_1p5e3_s44")[0])
res3 = [final_val(k)[0] for k in ("resg_s05_43", "resg_s05_44", "resg_s05_45")]
d_res = abs(res3[0] - res3[1])
print(f"   linear+QK-norm at 1.5e-3, orderings 43/44 : {d_lin:.4f}")
print(f"   residual-GELU at 2.55e-3, orderings 43/44 : {d_res:.4f}")
print(f"   residual over three orderings             : {max(res3) - min(res3):.4f}")
print(f"   ratio {d_res / d_lin:.1f}x, so an 'n sigma' claim has to name its arm")
st1, _, v1 = load("qkn_1p5e3")
st2, _, v2 = load("qkn_1p5e3_s44")
last11 = [s for s in st1 if s in st2][-11:]
print(f"   paired over the last {len(last11)} evals, max |diff| "
      f"{max(abs(v1[st1.index(s)] - v2[st2.index(s)]) for s in last11):.4f}")


head("Attention scale: s=0.5 is the method")
print("   q = x + LN(f(x)) is computed with no explicit factor, so the 1/2 in")
print("   Q = (1/2)(x + f(x)) sits in the scale. 9944 drops it and normalises per head,")
print("   which makes it a variant rather than a corrected baseline.")
for k in ["resg_s05_43", "resg_s05_44", "resg_s05_45", "resg_s1", "qkn_1p5e3", "qkn_2p55e3"]:
    r = RUNS[k]
    print(f"   {str(r['job']):>11}  {r['query']:<14} {r['mit']:<16} s={r['s']}  "
          f"{r['lr']:.2e}  ->  {final_val(k)[0]:.4f}")
print(f"\n   at matched LR, each method as defined:")
print(f"     9778 linear+QK-norm s=1     2.55e-3 -> {final_val('qkn_2p55e3')[0]:.4f}")
print(f"     9436 residual-GELU  s=0.5   2.55e-3 -> {final_val('resg_s05_43')[0]:.4f}")
print(f"     {abs(final_val('qkn_2p55e3')[0] - final_val('resg_s05_43')[0]):.4f} apart, "
      f"so indistinguishable")
lin, non = final_val("qkn_1p5e3")[0], final_val("resg_s1")[0]
print(f"\n   per-head variant against the tuned linear arm, both s=1 at 1.5e-3:")
print(f"     9777 {lin:.4f}   9944 {non:.4f}   variant ahead by {lin - non:.4f}")
print(f"     {(lin - non) / d_lin:.0f}x the linear arm's spread, {(lin - non) / d_res:.1f}x "
      f"the residual arm's, so it needs a replicate")
tr_lin, tr_non = load("qkn_1p5e3")[1][-1], load("resg_s1")[1][-1]
print(f"     train loss {tr_lin:.4f} vs {tr_non:.4f}, lead {tr_lin - tr_non:.4f}")
print(f"     train-to-val gap {lin - tr_lin:.4f} vs {non - tr_non:.4f}, so it fits better")
print(f"     rather than generalising better")


head("Where the arms differ during training")
sa, _, va = load("qkn_1p5e3")
sb, _, vb = load("resg_s1")
cross = [(s, vb[sb.index(s)] - va[sa.index(s)]) for s in sa if s in sb and s > 0]
behind = [s for s, d in cross if d > 0]
print(f"   residual behind at {len(behind)} of {len(cross)} evals, "
      f"from step {min(behind)} to {max(behind)}")
sign = None
for s, d in cross:
    now = "residual ahead" if d < 0 else "linear ahead"
    if now != sign:
        print(f"   step {s:>6}: {now} by {abs(d):.4f}")
        sign = now


head("Decay-to-zero on the plain linear arm")
for k in ["d2z_1e3", "d2z_2e3"]:
    st, _, va = load(k)
    mn = min(va)
    _, l12, _ = load_logit(k)
    print(f"   {RUNS[k]['job']} at {RUNS[k]['lr']:.0e}: best {mn:.4f} at step "
          f"{st[va.index(mn)]}, last {va[-1]:.4f} at {st[-1]}")
    print(f"     layer-12 logit peaks at {max(l12):.0f}   [{RUNS[k]['state']}]")
print("   the guard stops after two evals more than 0.5 above the running minimum, so")
print("   10140's stop at 25k is an automatic abort. Divergence is not monotone in the")
print("   rate: 1e-3 died while 2e-3 ran the full 60k, on the same init and ordering.")

d2z = final_val("d2z_2e3")[0]
best_sweep = min(vals)
plain = final_val("plain_42")[0]      # 8952, the plain arm's own surviving cosine run
print(f"\n   what the surviving run costs the confound story: the plain arm, no QK-norm,")
print(f"   no z-loss, no per-head bound, reaches {d2z:.4f} on schedule and rate alone.")
print(f"   both deltas below are measured from 8952 at {plain:.4f}, the plain arm's own")
print(f"   baseline, not from the QK-norm anchor at {anchor:.4f}:")
print(f"     stabilise then sweep : {plain - best_sweep:.4f}   (to {best_sweep:.4f})")
print(f"     schedule and rate    : {plain - d2z:.4f}   (to {d2z:.4f}), "
      f"{(plain - d2z) / (plain - best_sweep) * 100:.0f}% of it")
print(f"   and it beats the 7.5e-4 sweep point ({final_val('qkn_7p5e4')[0]:.4f}) outright.")


head("GPT-3 small decay-to-zero sweep at 60k")
sm = ["sm_6e4", "sm_1p2e3", "sm_2p4e3", "sm_4p8e3", "sm_9p6e3"]
for k in sm:
    st, _, va = load(k)
    print(f"   {RUNS[k]['lr']:.2e}  ->  {va[-1]:.4f} at {st[-1]}   "
          f"(last 5 evals: {(va[-1] - va[-5]) / 4:+.4f} per 1k)")
inner = ["sm_2p4e3", "sm_4p8e3", "sm_9p6e3"]
As, Bs, Cs, vs_ = quad_fit([RUNS[k]["lr"] for k in inner],
                           [final_val(k)[0] for k in inner])
print(f"\n   bracketed: 9.6e-3 lands {final_val('sm_9p6e3')[0] - final_val('sm_4p8e3')[0]:+.4f}")
print(f"   against 4.8e-3, so the minimum is interior. quadratic through the inner three")
print(f"   puts it at {vs_:.3e}, curvature {As:.4f}, which is {As / A:.2f}x the GPT-3 Large")
print(f"   bowl, so this one is flatter. the grid centre of 2.4e-3 was low by "
      f"{vs_ / 2.4e-3:.1f}x.")

print("\n   why an early read would have inverted it:")
for step in (36000, 48000, 54000, 60000):
    row = [(RUNS[k]["lr"], val_at(k, step)) for k in sm]
    rank = sorted(row, key=lambda p: p[1])
    print(f"     step {step:>6}: best->worst " + "  ".join(f"{lr:.1e}" for lr, _ in rank))
print("   the 36k leader finishes last and 4.8e-3 is still fourth of five at 54k, so no")
print("   rate comparison here is read off a run that has not annealed.")

st180, _, v180 = load("sm180_6e4")
print(f"\n   the 180k arm has begun: {RUNS['sm180_6e4']['job']} at 6e-4, step "
      f"{st180[-1]} of 180000, val {v180[-1]:.4f}")
print(f"   at the local beta below, 3x the budget predicts the optimum near "
      f"{vs_ / 1.75:.2e}, which is where 2.4e-3 already sits.")


head("Token budget, TPP and epochs (GPT-3 small, 124.37M params, 9.036B corpus)")
N, CORPUS, TOK = 124_373_760, 9.036e9, 480 * 1024
for steps in (60002, 100000, 180000):
    t = steps * TOK
    print(f"   {steps:>6} steps -> {t / 1e9:5.1f}B tokens, {t / N:5.0f} TPP, "
          f"{t / CORPUS:4.1f} epochs, {t / N / 20:4.1f}x Chinchilla")

print("\n   Bjorck's 125m optima (his Table 9), which is our geometry:")
BJ = [(25e9, 1.34e-3), (50e9, 1.02e-3), (100e9, 6.60e-4),
      (200e9, 4.12e-4), (400e9, 2.51e-4), (800e9, 1.98e-4)]
for (d0, l0), (d1, l1) in zip(BJ, BJ[1:]):
    print(f"     {d0 / 1e9:5.0f}B -> {d1 / 1e9:5.0f}B : beta "
          f"{math.log(l0 / l1) / math.log(d1 / d0):.3f}")
seg = BJ[:3]                          # 25 to 100B holds our 29.5B and 88.5B
mx = sum(math.log(d) for d, _ in seg) / 3
my = sum(math.log(l) for _, l in seg) / 3
beta = (-sum((math.log(d) - mx) * (math.log(l) - my) for d, l in seg)
        / sum((math.log(d) - mx) ** 2 for d, _ in seg))
shift = 3.0 ** beta
print(f"   local beta over our window {beta:.3f}, so 3x the tokens moves the optimum "
      f"{shift:.2f}x")
print(f"   not moving it costs {A * math.log(shift) ** 2:.4f}, "
      f"{A * math.log(shift) ** 2 / d_lin:.0f}x the linear arm's spread")
