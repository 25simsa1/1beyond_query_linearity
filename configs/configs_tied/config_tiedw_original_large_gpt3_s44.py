"""
NanoGPT Configuration - GPT-3 Large geometry, "original" (linear-query) baseline
Owner: fastsen (slsang29)   |   Colby HPC

Pivot per Marko: match the publicly-shared GPT-3 sizes (Table 2.1, arXiv:2005.14165),
NOT GPT-2 Large. Key difference is d_head (96, not 64) and d_model (1536, not 1280).

GPT-3 Large (Table 2.1):  n_layers=24, d_model=1536, n_heads=16, d_head=96,
                          d_ff=4*d_model=6144, LR=2.5e-4.
Query mode : original  (Q = X @ W_Q)
Steps      : 60k.  block_size kept at 1024 (shared tokenized data + indices;
             GPT-3's n_ctx is 2048 but switching would break comparability / need re-indexing).

batch_size below is the microbatch (train.py reads it from here now). Effective batch is
pinned to 480 by train.py (accum = 480 // batch_size), and eval_iters = 2400 // batch_size,
so train/eval index coverage stays identical across the sweep. Tune batch_size to the GPU.
"""
import math

model_args = {
    # Model architecture - GPT-3 Large geometry
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 24,
    "num_heads": 16,
    "n_embd": 1536,
    "head_size": 96,               # d_head = 1536 // 16 = 96  (the GPT-3 vs GPT-2 difference)
    "mlp_hidden_size": 1536 * 4,   # d_ff = 4 * d_model = 6144

    # Weight configuration
    "tie_weights": True,

    # Query mode: linear baseline (gpt-2 / gpt-3 equivalent)
    "query_mode": "original",

    # Regularization
    "dropout": 0.0,
    "bias": False,

    # Training batch configuration (effective batch pinned to 480 by train.py)
    # probe_mem.py: GPT-3 Large ~1.0GB/sample (lighter than GPT-2 Large); 32 fits A100-80GB
    # with wide margin (~57GB est w/ compile). Could push to 40 (~69GB) after a clean full-card probe.
    "batch_size": 40,              # microbatch (RTX PRO 6000, 102GB): accum 12, eval_iters 60
    "accumulation_size": 12,       # 480 // 40

    # Attention scale (standard scaled dot-product for head_size=96)
    "scale": 1 / (math.sqrt(1536 // 16)),   # 1/sqrt(96)

    # Optimizer settings
    "learning_rate": 2.5e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Learning rate schedule (cosine to min over the full 60k)
    "decay_lr": True,
    "warmup_iters": 2000,
    "lr_decay_iters": 60002,
    "min_lr": 2.5e-5,

    # Checkpoint and stopping
    "save_checkpoint_steps": [20000, 40000, 60000],
    "max_iters": 60002,
}
