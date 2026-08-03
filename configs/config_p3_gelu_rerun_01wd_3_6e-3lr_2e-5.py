"""
Phase 3 config: gelu_rerun

The headline residual_gelu re-run on THIS machine so its eval is directly comparable (paired) to the control. Machine-clean Delta (Arm 1).
"""

import math

model_args = {
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 12,
    "num_heads": 12,
    "n_embd": 768,
    "head_size": 64,
    "mlp_hidden_size": 3072,
    "tie_weights": True,
    "query_mode": 'residual_gelu',
    "dropout": 0.0,
    "bias": False,
    "batch_size": 12,
    "accumulation_size": 40,
    "scale": 1/(2*math.sqrt(768//12)),   # halved (NRQ temperature)
    "learning_rate": 0.0036,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "decay_lr": True,
    "warmup_iters": 2000,
    "lr_decay_iters": 60002,
    "min_lr": 2e-05,
    "save_checkpoint_steps": [10000, 20000, 40000, 60000],
    "max_iters": 60002,
}
