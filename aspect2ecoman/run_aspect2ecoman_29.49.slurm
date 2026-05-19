#!/bin/bash
#SBATCH --job-name=a2e_29.49
#SBATCH --output=/home/nredick/nredick-pfs/csz-models/29.49/ecoman/aspect2ecoman/out_%j/aspect2ecoman_%j.out
#SBATCH --error=/home/nredick/nredick-pfs/csz-models/29.49/ecoman/aspect2ecoman/out_%j/aspect2ecoman_%j.err
#SBATCH --time=01:00:00                     # 1 hr max
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=high
#SBATCH --mem=128G
#SBATCH --constraint='zen|zen2|zen3'        # do not run or compile on zen4 nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

set -euo pipefail

MODEL=29.49 # defines the model that will be preprocessed
OUT_DIR=/home/nredick/nredick-pfs/csz-models/$MODEL/ecoman/aspect2ecoman/out_$SLURM_JOB_ID

# convert MODEL to path PVD_FILE
PVD_FILE="/home/nredick/nredick-pfs/csz-models/$MODEL/solution.pvd"

# check that the PVD file exists
if ! [[ -f "$PVD_FILE" ]]; then
    echo "PVD file \"$PVD_FILE\" does not exist."
    exit 1
fi

# create output directory if it doesn't exist
if ! [[ -d "$OUT_DIR" ]]; then
    mkdir -p "$OUT_DIR"
fi

module load conda
mamba activate $HOME/.conda/envs/ecoman

# 5 km EULERIAN resolution, 25 km LAGRANGIAN spacing, 10 Ma of advection
python3 aspect2ecoman.py -f "$PVD_FILE" -o "$OUT_DIR" --resolution 5 --lagrangian-spacing 25 --timemax 10e6 

# cp the python file to the output directory for record-keeping
cp aspect2ecoman.py "$OUT_DIR"/aspect2ecoman.py

# cp this file to out dir for record keeping
current_path=$(realpath "$0")
cp $current_path "$OUT_DIR/"
