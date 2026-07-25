#!/bin/bash
#SBATCH --job-name=owt_prep
#SBATCH --partition=normal
#SBATCH --exclude=n2,n7,n8,n10,n14,n15,n16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=slsang29@colby.edu

set -euo pipefail
cd "$HOME/beyond_query_linearity"
module load uv/0.11.26
echo "host=$(hostname)  date=$(date)  cores=$(nproc)  mem=$(free -g | awk '/Mem:/{print $2}')G"
# uv run will sync the env first if needed, then run
uv run python prepare_data.py
echo "=== data dir after prep ==="
ls -lh data/openwebtext/
echo "PREP_DONE"
