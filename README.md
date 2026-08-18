# Beyond Linearity in Attention Projections
### The Case for Nonlinear Queries
Official implementation and model weights for the paper: **"Beyond Linearity in Attention Projections: The Case for Nonlinear Queries"** ([arXiv:2603.13381](https://arxiv.org/abs/2603.13381)).
This repository implements nonlinear query projections of the form $Q(X) = (X + f_\theta(X))/2$, where $f_\theta$ is a bottleneck MLP, replacing the standard linear $W_Q$ at the same parameter budget. Building on the algebraic redundancy of $W_Q$ established by [Karbevski and Mijoski (2025)](https://arxiv.org/abs/2510.23912)), we show that nonlinear queries consistently improve validation loss over baseline, comfortably outperforming a model with 12.5% more non-embedding parameters.

To be presented at the ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM)).

> **Note on this fork.** Upstream is [MarkoKarbevski/beyond_query_linearity](https://github.com/MarkoKarbevski/beyond_query_linearity). This fork adds a stability and learning-rate audit of the GPT-3 Large comparison, together with the architecture variants and tooling it required. See [Fork Additions](#-fork-additions-simon-sang).

---
## 🚀 Quick Start
### 1. Model Checkpoints
Pre-trained checkpoints and training losses from our runs are available for download:
* **[Download from Google Drive](https://drive.google.com/drive/folders/1JNlDCGk1Rw-kfsgmDBtLOkjl9iyGXCtt?usp=drive_link)**

You can explore the losses using `explore losses.ipynb`
### 2. Data Preparation

We utilize the **OpenWebText** dataset. Follow these steps after preparing the `uv` environment:
1. **Dataset Acquisition:** Run `Data_Handling.ipynb` to download and preprocess the raw data.
2. **Reproducibility:** Run `Generate_Indices.ipynb` to ensure consistent data shuffling and splitting.
3. **Configuration:** Plenty configurations can be found in the configs folder. Creating a new one is relatively simple following the examples.

You might want to modify `Generate_Indices.ipynb` if you want to create differently sized batches or a train run that runs for longer than 600k steps.
### 3. Training

To initiate training on a specific GPU (e.g., GPU 0), use the following command:
`python train.py _a_config_file_ --gpu {gpus_to_use}`

For example:
`python train.py configs/configs_tied/config_tiedw_original.py --gpu 0`

Note: The repo has been created for a single GPU training, but it should not be difficult to modify it for DDP training as well, following Karpathy's example.
---
## 🛠 Architecture

The attention mechanism has been modified to support nonlinear query projections: the standard linear $W_Q$ is replaced with a residual bottleneck MLP $Q(X) = (X + f_\theta(X))/2$, where $f_\theta(X) = \text{LN}(\text{GELU}(\text{RMSNorm}(X)W_1)W_2)$ with $W_1 \in \mathbb{R}^{d \times r}$, $W_2 \in \mathbb{R}^{r \times d}$, and $r = d/2$. 

Keys and values remain standard linear projections.
---
## 🔬 Fork Additions (Simon Sang)

Commits from 20 July 2026 onward are mine (20 commits, 84 files changed, 7,248 insertions). Only `model.py` and `train.py` are modified; everything else is new.

### Architecture (`model.py`)

Three query modes beyond the four upstream:

* **`residual_linear`** replaces the bottleneck MLP correction with a single linear map.
* **`routed`** makes the correction a token-routed mixture of `n_experts` low-rank query experts, sized so that `K * 2 * d * r_e = d^2`, giving 8 experts at rank 48 for `d = 768` and matching the parameter count of a full `W_Q`.
* **`split_depth`** gives the full-rank nonlinear query only to layers from `split_layer` onward, leaving earlier layers on the identity.

Three normalization paths, each aimed at the attention-logit growth behind the divergences:

* **`qk_norm`** applies RMSNorm to Q and K per head, bounding the logits directly.
* **`k_norm`** normalizes K alone. The query `q = x + LN(f(x))` is already bounded as a sum of two normed terms, and `||q||` acts as a per-token attention temperature that norming would discard.
* **`per_head_norm`** normalizes the anchor and the correction within each head, bounding `||q_h||` pointwise rather than in expectation.

Also `z_loss` (the PaLM/T5 softmax log-partition penalty), `lr_schedule` selecting cosine or linear decay-to-zero (D2Z, [Bergsma et al. 2025](https://arxiv.org/abs/2502.15938)), and an attention-logit probe for testing divergence against a `1e4` threshold.

### Training (`train.py`, `train_512.py`)

* An in-process divergence guard that exits on a non-finite train or validation loss, or on validation exceeding the best by 0.5 across two consecutive evaluations, so a SLURM `--mail-type` notification fires instead of a node burning its walltime.
* Batch size, seed, dataset and init source threaded through the config, with `eval_iters` scaled by batch size.
* `train_512.py` for the 512-batch track, plus eight sbatch scripts covering A100 and H200 nodes in plain and QK-normalized variants.
* Per-seed index generation (`gen_indices_seed.py`, `gen_indices_512.py`), so data order varies independently of initialization.
* 25 new configs, generated by `mk_d2z_2p55e3.py` and `mk_small_sweep.py` rather than written by hand.

### Analysis and write-up (`neurreps_writeup/`)

`extract_logs.sh` and `refresh_logs.py` pull per-step losses out of SLURM `.out` files into the 22 CSVs under `neurreps_writeup/logs/`. In `neurreps_writeup/code/`, `common.py` builds the run table from each job's logged config rather than from its filename, `make_figures.py` regenerates the seven figures, `analysis.py` prints every number the write-up quotes, and `verify_writeup.py` asserts each of those numbers against the logs at `5e-5` tolerance, so any disagreement between prose and data fails hard.

```
python neurreps_writeup/code/make_figures.py     # rebuild figures/
python neurreps_writeup/code/analysis.py         # print every quoted number
python neurreps_writeup/code/verify_writeup.py   # assert the .tex against logs/
```

### Findings

* The GPT-3 Large comparison ran its linear baseline at the GPT-3 default `2.5e-4`, close to the highest rate the un-normalized linear model survives. Seeds 43 and 44 diverge there.
* z-loss only delays the divergence. QK-normalization removes it, which is what makes a learning-rate sweep of the linear arm possible.
* The stabilized linear arm reaches **2.5403** at `1.5e-3`, which is 0.076 below the same arm at the default rate and below all three residual-GELU runs (2.5464 to 2.5487). The headline architectural gap does not survive giving the linear arm its own tuned rate, so I read the earlier deltas as recipe-level rather than architectural.
* Run-to-run spread is arm-dependent. Two seeds of the stabilized linear arm land 0.0002 apart, while the residual arm spreads 0.0015 over the same change, so how many sigma a result is worth depends on which arm supplies the sigma.
* A residual variant that normalizes the query per head and drops the customary halving of the attention scale reaches **2.5383**, the best number in the project, 0.0020 ahead of the tuned linear arm at matched rate. That margin is 10x the linear arm's seed spread but only 1.3x the residual arm's, so it is promising and not yet a win.
* Decay-to-zero costs stability at this scale. The plain arm diverged at `1e-3` yet ran to completion at `2e-3`, reaching 2.5504 with no normalization of any kind, closing 88% of the distance between the plain arm's default-rate run and the swept optimum.

The full argument and figures are in [`neurreps_writeup/writeup_neurreps.pdf`](neurreps_writeup/writeup_neurreps.pdf).

---
## 📝 Citation
If you find this work useful in your research, please cite:
```bibtex
@article{karbevski2026beyond,
  title={Beyond Linearity in Attention Projections: The Case for Nonlinear Queries},
  author={Karbevski, Marko},
  journal={arXiv preprint arXiv:2603.13381},
  year={2026},
  note={Presented at the ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM)}
}
```
---
## 🙏 Acknowledgments
I am grateful to the anonymous reviewers for their constructive feedback, and to Nils Graef, Yiping Ji, Haris Mandal, and Antonij Mijoski for valuable discussions. This codebase builds on the [nanoGPT](https://github.com/karpathy/nanoGPT) repository by Andrej Karpathy.

---
## 🤝 Collaboration & Contributing (Open-Source & Commercial)

This repository represents an independent research initiative focused on establishing the mathematical foundations and structural validity of Nonlinear Residual Queries (NRQ). 

To isolate the core architectural mechanics cleanly, development has been focused entirely on delivering a verified implementation, rather than an industrialized production framework. As a solo researcher, I highly welcome both open-source and commercial collaborations to improve the work.

I am very interested in collaboration focused on a multitude of axes, including, but not limited to:

* **Validation at Scale:** Evaluating the structural stability, performance deltas, and scaling curves of the NRQ block at the larger scales.
* **Cross-Domain Evaluation:** Extending and testing the non-linear query architecture across diverse modalities beyond standard autoregressive language modeling, including vision, audio, and multimodal generative tasks.
* **Optimization:** Engineering high-performance distributed training integration, hardware-specific acceleration, and custom CUDA kernels to optimize throughput.
* **Generalization & Theoretical Developpment:** Generalizing the work to multiple projections, including the K, V and O, as well as further developping a theory in order to explain the beyond-scaling-laws performance.

To discuss this further, please contact me via any of the emails outlined in the paper, or via linkedin.
---
The code has been tested on Python version `3.12.11` using a single Nvidia 5090 RTX GPU.
