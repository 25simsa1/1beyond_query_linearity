import os, sys, numpy as np, torch
seed = int(sys.argv[1])
dst = f"data/openwebtext_s{seed}"
os.makedirs(dst, exist_ok=True)
# share the token bins AND the fixed seed-42 eval set (so only train ORDER varies across seeds)
for f in ["train.bin", "val.bin", "eval_train_indices.npy", "eval_val_indices.npy"]:
    lp = os.path.join(dst, f)
    if not os.path.exists(lp):
        os.symlink(os.path.join("..", "openwebtext", f), lp)
train = np.memmap(os.path.join(dst, "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(dst, "val.bin"), dtype=np.uint16, mode="r")
tr = os.path.join(dst, "train_indices.npy")
va = os.path.join(dst, "val_indices.npy")
if os.path.exists(tr) and os.path.exists(va):
    print(f"seed {seed}: train/val indices already exist, skip")
else:
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    ixt = torch.randint(train.size - 1024, (600000, 5 * 8 * 12), device="cpu", generator=g)
    ixv = torch.randint(val.size - 1024, (600000, 5 * 8 * 12), device="cpu", generator=g)
    np.save(tr, ixt.numpy()); np.save(va, ixv.numpy())
    print(f"seed {seed}: wrote train/val indices [600000, 480]")
print(f"seed {seed} contents:", sorted(os.listdir(dst)))
