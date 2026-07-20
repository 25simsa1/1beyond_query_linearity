#!/usr/bin/env python
"""Set up data/openwebtext_512 for the 512-effective-batch track.
- train/val indices regenerated at [600000, 512] (seed 42, train-then-val on one stream)
- token .bin files and the eval index set are symlinked from data/openwebtext (shared;
  microbatch 32 divides the existing 2400-col eval, so no eval regen needed)
This is a NEW data stream -> the 512 track is intentionally not comparable to the 480 runs.
"""
import os, numpy as np, torch

src, dst = "data/openwebtext", "data/openwebtext_512"
os.makedirs(dst, exist_ok=True)

for f in ["train.bin", "val.bin", "eval_train_indices.npy", "eval_val_indices.npy"]:
    lp = os.path.join(dst, f)
    if not os.path.exists(lp):
        os.symlink(os.path.join("..", "openwebtext", f), lp)   # relative symlink
        print(f"symlinked {f}")

train = np.memmap(os.path.join(dst, "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(dst, "val.bin"), dtype=np.uint16, mode="r")
tr = os.path.join(dst, "train_indices.npy")
va = os.path.join(dst, "val_indices.npy")

if os.path.exists(tr) and os.path.exists(va):
    print("512 train/val indices already exist -> skip")
else:
    g = torch.Generator(device="cpu"); g.manual_seed(42)
    ixt = torch.randint(train.size - 1024, (600000, 512), device="cpu", generator=g)
    ixv = torch.randint(val.size - 1024, (600000, 512), device="cpu", generator=g)
    np.save(tr, ixt.numpy()); np.save(va, ixv.numpy())
    print("wrote train_indices.npy & val_indices.npy [600000, 512]")

print("DONE contents:", sorted(os.listdir(dst)))
