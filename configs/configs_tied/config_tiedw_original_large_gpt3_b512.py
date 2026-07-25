"""
GPT-3 Large "original" (linear-query) baseline -- 512 EFFECTIVE-BATCH TRACK.

SEPARATE TRACK, intentionally NOT comparable to the 480 runs: different data stream
(indices regenerated at [600000, 512]) and different token budget (~524k tokens/step,
~31.5B over 60k, vs ~492k / ~29.5B for the 480 runs).

Runs with train_512.py (NOT train.py): dataset=openwebtext_512, and effective batch is
pinned to 512 (accum = 512 // batch_size). eval_iters = 2400 // batch_size (eval set reused).
Microbatch 32 -> accum 16, eval_iters 75; fits A100 / RTX PRO 6000 / H200.
"""
import math

model_args = {
    # Model architecture - GPT-3 Large geometry (Table 2.1)
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 24,
    "num_heads": 16,
    "n_embd": 1536,
    "head_size": 96,
    "mlp_hidden_size": 1536 * 4,

    "tie_weights": True,
    "query_mode": "original",
    "dropout": 0.0,
    "bias": False,

    # 512 effective batch: microbatch 32 x accum 16
    "batch_size": 32,
    "accumulation_size": 16,

    "scale": 1 / (math.sqrt(1536 // 16)),   # 1/sqrt(96)

    "learning_rate": 2.5e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    "decay_lr": True,
    "warmup_iters": 2000,
    "lr_decay_iters": 60002,
    "min_lr": 2.5e-5,

    "save_checkpoint_steps": [20000, 40000, 60000],
    "max_iters": 60002,
    "init_from": "resume",   # added: continue on H200 from RTX ckpt
}
