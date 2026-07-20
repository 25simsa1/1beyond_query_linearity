#!/usr/bin/env python
"""
Headless data prep for beyond_query_linearity on the cluster.
Combines Data_Handling.ipynb (OpenWebText tokenization) + Generate_Indices.ipynb,
kept byte-faithful to those notebooks so the data matches the rest of the team.

- val split : seed 2357, test_size 0.0005, shuffle=True   (== Data_Handling.ipynb)
- indices   : torch.Generator seed 42, train then val drawn on the SAME stream per block
              (== Generate_Indices.ipynb -- do NOT reseed between train and val)
Idempotent: skips any step whose outputs already exist.
"""
import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

folder_path = 'data/openwebtext'
os.makedirs(folder_path, exist_ok=True)

# respect the cores SLURM actually gave us (not the whole node)
try:
    n_alloc = len(os.sched_getaffinity(0))
except AttributeError:
    n_alloc = os.cpu_count() or 1
num_proc = max(1, 1 + n_alloc // 2)
print(f"[prep] allocated cores={n_alloc}, num_proc={num_proc}", flush=True)

enc = tiktoken.get_encoding("gpt2")
train_bin = os.path.join(folder_path, 'train.bin')
val_bin = os.path.join(folder_path, 'val.bin')

# ---- Step 1: tokenize OpenWebText -> train.bin / val.bin ----
if os.path.exists(train_bin) and os.path.exists(val_bin):
    print("[prep] train.bin & val.bin already exist -> skipping tokenization", flush=True)
else:
    print("[prep] loading dataset Skylion007/openwebtext (~54GB cache) ...", flush=True)
    dataset = load_dataset("Skylion007/openwebtext", num_proc=num_proc, trust_remote_code=True)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    print("[prep] tokenizing splits ...", flush=True)
    tokenized = split_dataset.map(process, remove_columns=['text'],
                                  desc="tokenizing the splits", num_proc=num_proc)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(folder_path, f'{split}.bin')
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    print("[prep] tokenization done (train.bin ~17GB, val.bin ~8.5MB)", flush=True)

# ---- Step 2: shuffle indices (mirrors Generate_Indices.ipynb exactly) ----
import torch
train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
print(f"[prep] train tokens={train_data.size}, val tokens={val_data.size}", flush=True)

tr_idx = os.path.join(folder_path, 'train_indices.npy')
va_idx = os.path.join(folder_path, 'val_indices.npy')
etr_idx = os.path.join(folder_path, 'eval_train_indices.npy')
eva_idx = os.path.join(folder_path, 'eval_val_indices.npy')

# Block A: main train/val indices [600000, 5*8*12=480] -- one seed-42 stream, train THEN val
if os.path.exists(tr_idx) and os.path.exists(va_idx):
    print("[prep] main indices exist -> skip", flush=True)
else:
    gen = torch.Generator(device='cpu'); gen.manual_seed(42)
    ix_train = torch.randint(train_data.size - 1024, (600000, 5 * 8 * 12), device='cpu', generator=gen)
    ix_val = torch.randint(val_data.size - 1024, (600000, 5 * 8 * 12), device='cpu', generator=gen)
    np.save(tr_idx, ix_train.numpy()); np.save(va_idx, ix_val.numpy())
    print("[prep] wrote train_indices.npy & val_indices.npy [600000,480]", flush=True)

# Block B: eval indices [600000//1000=600, 200*12=2400] -- fresh seed-42 stream, train THEN val
if os.path.exists(etr_idx) and os.path.exists(eva_idx):
    print("[prep] eval indices exist -> skip", flush=True)
else:
    gen = torch.Generator(device='cpu'); gen.manual_seed(42)
    ix_train = torch.randint(train_data.size - 1024, (600000 // 1000, 200 * 12), device='cpu', generator=gen)
    ix_val = torch.randint(val_data.size - 1024, (600000 // 1000, 200 * 12), device='cpu', generator=gen)
    np.save(etr_idx, ix_train.numpy()); np.save(eva_idx, ix_val.numpy())
    print("[prep] wrote eval_train_indices.npy & eval_val_indices.npy [600,2400]", flush=True)

print("[prep] ALL DONE", flush=True)
