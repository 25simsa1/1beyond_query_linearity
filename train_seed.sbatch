#!/bin/bash
#SBATCH --job-name=bql_seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=slsang29@colby.edu
set -euo pipefail
cd "$HOME/beyond_query_linearity"
module load uv/0.11.26
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host=$(hostname) date=$(date) BQL_DATASET=${BQL_DATASET:-openwebtext} CFG=${CFG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
uv run python -u train.py "${CFG}" --gpu all
echo "EXIT=$?"
