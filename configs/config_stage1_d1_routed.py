"""
Phase 3 config: control

Param-identical GELU->identity ablation. Only query_mode differs from the headline. Same lr/wd/min_lr/scale/data. This isolates the nonlinearity (Arm 0).
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
    "query_mode": 'routed',
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
    "save_checkpoint_steps": [10000, 20000],
    "max_iters": 20002,
    "n_experts": 8,
    "expert_rank": 48,
}
