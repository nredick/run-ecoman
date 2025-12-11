#!/bin/bash
#SBATCH --job-name=aspect2ecoman
#SBATCH --output=./out_%j/aspect2ecoman_%j.out
#SBATCH --error=./out_%j/aspect2ecoman_%j.err
#SBATCH --time=01:00:00                     # 1 hr max
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=high
#SBATCH --mem=512G                          # to use magali's nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

set -euo pipefail

# Check for positional argument $1, and if it's empty, use the default 28.37
MODEL=${1:-28.37}
OUTDIR=${OUTDIR:-./out_$SLURM_JOB_ID}

# convert MODEL to path PVD_FILE
PVD_FILE="/home/nredick/nredick-pfs/csz-models/$MODEL/solution.pvd"

# check that the PVD file exists
if ! [[ -f "$PVD_FILE" ]]; then
    echo "PVD file \"$PVD_FILE\" does not exist."
    exit 1
fi

# create output directory if it doesn't exist
if ! [[ -d "$OUTDIR" ]]; then
    mkdir -p "$OUTDIR"
fi 

module load conda
mamba activate $HOME/.conda/envs/ecoman

python3 aspect2ecoman.py -f "$PVD_FILE" -o "$OUTDIR"
