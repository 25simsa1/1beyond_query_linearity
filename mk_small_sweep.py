"""Generate the GPT-3 small D2Z LR sweep: 5 LRs x {60k, 180k} steps.

Derived from the p3 baseline recipe, changing only the LR, the budget, and the
schedule/floor that make it D2Z. Built in Python, then py_compiled and
re-imported with assertions, per project rule 3.
"""
import pprint, py_compile, importlib.util, importlib.machinery, sys, os

CFGDIR = "configs"
BASE = f"{CFGDIR}/config_p3_baseline_recipe_01wd_3_6e-3lr_2e-5.py"

LRS = [("6e4", 6.0e-4), ("1p2e3", 1.2e-3), ("2p4e3", 2.4e-3),
       ("4p8e3", 4.8e-3), ("9p6e3", 9.6e-3)]
BUDGETS = [("60k", 60002, [20000, 40000, 60000]),
           ("180k", 180000, [60000, 120000, 180000])]

def model_args_of(path):
    ns = {}
    exec(open(path).read(), ns)
    return ns["model_args"]

base = model_args_of(BASE)
# sanity on the base before deriving anything
assert base["batch_size"] * base["accumulation_size"] == 480, "effective batch must be 480 to match the 480-column index files"
assert base["n_layer"] == 12 and base["n_embd"] == 768 and base["query_mode"] == "original"

sys.path.insert(0, ".")
from model import GPTConfig

written = []
for lrtag, lr in LRS:
    for btag, iters, ckpts in BUDGETS:
        ma = dict(base)                     # preserves key order
        ma["learning_rate"] = lr
        ma["max_iters"] = iters
        ma["lr_decay_iters"] = iters
        ma["min_lr"] = 0.0                  # D2Z
        ma["lr_schedule"] = "linear"        # D2Z
        ma["save_checkpoint_steps"] = ckpts

        name = f"config_p3small_d2z_lr{lrtag}_{btag}.py"
        out = f"{CFGDIR}/{name}"
        header = (
            f"# generated from {os.path.basename(BASE)} by mk_small_sweep.py -- do not hand edit\n"
            f"# GPT-3 small D2Z LR sweep: lr={lr:g}, {iters} steps "
            f"({iters*480*1024/1e9:.1f}B tokens, {iters*480*1024/124373760:.0f} TPP)\n"
        )
        open(out, "w").write(header + "model_args = " + pprint.pformat(ma, indent=4, sort_dicts=False) + "\n")
        py_compile.compile(out, doraise=True)

        loader = importlib.machinery.SourceFileLoader(f"c_{lrtag}_{btag}", out)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        m = importlib.util.module_from_spec(spec)
        sys.modules[loader.name] = m
        loader.exec_module(m)
        got = m.model_args

        expect_changed = {"learning_rate", "max_iters", "lr_decay_iters", "min_lr",
                          "lr_schedule", "save_checkpoint_steps"}
        actually_changed = {k for k in set(got) | set(base) if got.get(k) != base.get(k)}
        assert actually_changed <= expect_changed, f"{name}: unexpected changes {actually_changed - expect_changed}"
        assert got["learning_rate"] == lr and got["max_iters"] == iters
        assert got["lr_decay_iters"] == iters and got["min_lr"] == 0.0
        assert got["lr_schedule"] == "linear"
        assert got["batch_size"] * got["accumulation_size"] == 480
        assert max(ckpts) <= iters

        c = GPTConfig(**got)
        assert c.lr_schedule == "linear" and c.min_lr == 0.0
        assert c.learning_rate == lr and c.max_iters == iters
        written.append((name, lr, iters, got["batch_size"], got["accumulation_size"]))

# cross-check: within a budget the configs differ ONLY in learning_rate
for btag, iters, _ in BUDGETS:
    mas = [model_args_of(f"{CFGDIR}/config_p3small_d2z_lr{t}_{btag}.py") for t, _ in LRS]
    for other in mas[1:]:
        d = {k for k in set(mas[0]) | set(other) if mas[0].get(k) != other.get(k)}
        assert d == {"learning_rate"}, f"{btag}: configs differ in {d}, expected only learning_rate"

print(f"{len(written)} configs written and verified\n")
print(f"{'config':44} {'lr':>9} {'iters':>7} {'micro':>6} {'accum':>6}")
for n, lr, it, b, a in written:
    print(f"{n:44} {lr:9.2e} {it:7d} {b:6d} {a:6d}")
print("\nwithin each budget the 10 configs differ only in learning_rate")
print("ASSERTIONS PASSED")
