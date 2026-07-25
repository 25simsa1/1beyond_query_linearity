#!/bin/bash
#SBATCH --job-name=gpt3L_orig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=slsang29@colby.edu

set -euo pipefail
cd "$HOME/beyond_query_linearity"
module load uv/0.11.26
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host=$(hostname)  date=$(date)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# batch_size / eval_iters are read from the config by train.py; effective batch stays 480
uv run python -u train.py configs/configs_tied/config_tiedw_original_large_gpt3.py --gpu all
echo "TRAIN_EXIT=$?"
