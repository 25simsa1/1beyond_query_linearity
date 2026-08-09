"""Generate the qk_norm D2Z config at lr 2.55e-3, matching the existing 1.5e-3 one.

Mirrors what mk_d2z.py did: take the cosine qknorm config and change only the
schedule and the floor, so the run is a one-variable contrast against 9778.
Builds in Python, then py_compiles and re-imports to assert, per project rule 3.
"""
import pprint, py_compile, importlib.util, importlib.machinery, sys, os

CFGDIR = "configs/configs_tied"
SRC = f"{CFGDIR}/config_tiedw_original_large_gpt3_s43_qknorm_lr2p55e3.py"
SIB = f"{CFGDIR}/config_tiedw_original_large_gpt3_s43_qknorm_lr1p5e3_d2z.py"
OUT = f"{CFGDIR}/config_tiedw_original_large_gpt3_s43_qknorm_lr2p55e3_d2z.py"

def model_args_of(path):
    ns = {}
    exec(open(path).read(), ns)
    return ns["model_args"]

src = model_args_of(SRC)
sib = model_args_of(SIB)

ma = dict(src)                 # preserves key order
ma["min_lr"] = 0.0             # D2Z: decay to zero
ma["lr_schedule"] = "linear"   # D2Z: linear, not cosine

header = (
    f"# generated from {os.path.basename(SRC)} by mk_d2z_2p55e3.py -- do not hand edit\n"
    f"# one-variable D2Z test vs 9778 (2.5469): schedule + floor only\n"
)
body = "model_args = " + pprint.pformat(ma, indent=4, sort_dicts=False) + "\n"
open(OUT, "w").write(header + body)
py_compile.compile(OUT, doraise=True)

# re-import assertion
loader = importlib.machinery.SourceFileLoader("cfg_new", OUT)
spec = importlib.util.spec_from_loader("cfg_new", loader)
m = importlib.util.module_from_spec(spec)
sys.modules["cfg_new"] = m
loader.exec_module(m)
got = m.model_args

# 1. only the two intended keys differ from the cosine source
diff_src = {k for k in set(got) | set(src) if got.get(k) != src.get(k)}
assert diff_src == {"min_lr", "lr_schedule"}, f"unexpected diff vs source: {diff_src}"
assert got["min_lr"] == 0.0 and got["lr_schedule"] == "linear"

# 2. only learning_rate differs from the existing 1.5e-3 D2Z sibling
diff_sib = {k for k in set(got) | set(sib) if got.get(k) != sib.get(k)}
assert diff_sib == {"learning_rate"}, f"unexpected diff vs sibling: {diff_sib}"
assert got["learning_rate"] == 0.00255, got["learning_rate"]
assert sib["learning_rate"] == 0.0015

# 3. it actually builds a model config
sys.path.insert(0, ".")
from model import GPTConfig
c = GPTConfig(**got)
assert c.lr_schedule == "linear" and c.min_lr == 0.0 and c.qk_norm is True
assert (c.batch_size, c.accumulation_size, c.max_iters) == (40, 12, 60002)

print(f"wrote {OUT}")
print(f"  diff vs cosine source   : {sorted(diff_src)}")
print(f"  diff vs 1.5e-3 D2Z twin : {sorted(diff_sib)}")
print(f"  lr={got['learning_rate']} min_lr={got['min_lr']} sched={got['lr_schedule']} qk_norm={got['qk_norm']}")
print("ASSERTIONS PASSED")
