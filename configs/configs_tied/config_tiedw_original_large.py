"""
NanoGPT Configuration - GPT-2 Large, "Original" (gpt-2-equivalent / linear-query) baseline
Owner: fastsen (slsang29)   |   Colby HPC, SLURM job 8501, node n10 (A100 80GB)
Purpose: linear-query control for the GPT-2 Large 60k sweep (to compare against the
         residual_gelu / nonlinear-query runs, e.g. Jiwon's).

Query mode : original  (Q = X @ W_Q)
Model size : GPT-2 Large  (n_embd=1280, n_layer=36, num_heads=20, head_size=64)  ~774M params
LR         : OpenAI 760M-row default  (max 2.5e-4, min 2.5e-5)  -> ~2x Chinchilla at 60k steps
Microbatch : batch_size=24, accumulation_size=20  (product 480 == default effective batch)
             tokens/step = 480 * 1024 = 491,520;  60,002 steps ~= 29.5B tokens.

NOTE: train.py hardcodes the training-loop batch_size and eval_iters (it does NOT read them
      from this dict). This run uses the train.py literals batch_size=24 and eval_iters=100
      (= 2400/24), which make train (480 cols) and eval (2400 cols) index coverage
      byte-identical to the default 12/200 layout -> comparable to the rest of the sweep.
      (batch 24 chosen to fit the A100-80GB; 40/60 OOM on the full-vocab logits + 36-layer
      activations. Effective batch is unchanged at 480, so results are unaffected.)
"""
import math

model_args = {
    # Model architecture - GPT-2 Large
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 36,
    "num_heads": 20,
    "n_embd": 1280,
    "head_size": 64,
    "mlp_hidden_size": 1280 * 4,

    # Weight configuration
    "tie_weights": True,

    # Query mode: linear baseline (gpt-2 equivalent)
    "query_mode": "original",

    # Regularization
    "dropout": 0.0,
    "bias": False,

    # Training batch configuration (effective batch = 24*20 = 480)
    "batch_size": 24,
    "accumulation_size": 20,

    # Attention scale (standard scaled dot-product for head_size = 1280//20 = 64)
    "scale": 1 / (math.sqrt(1280 // 20)),

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
