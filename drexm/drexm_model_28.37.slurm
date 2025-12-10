#!/bin/bash
#SBATCH --job-name=drexm_model_28.37
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/28.37/ecoman/drexm/out_%j/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/28.37/ecoman/drexm/out_%j/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=125                          # must be equal to product of MPI proc distribution along axis from DAT file
#SBATCH --partition=high
#SBATCH --mem=512G                          # to use magali's nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
#SBATCH --chdir=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics/D-REX_M

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=28.37

MODEL_DIR="$BASEDIR/$MODEL" # model directory
WORKING_DIR="$MODEL_DIR/ecoman/drexm/" # model inputs (drexm_input.dat) & slurm outputs
OUT_DIR="$MODEL_DIR/ecoman/drexm/out_$SLURM_JOB_ID/" # will contain drex_m outputs

echo "Running D-REX_M on model \"$MODEL\"" && echo
echo "Model directory: $MODEL_DIR/"
# echo "Working directory: $WORKING_DIR"
echo "Output directory: $OUT_DIR" && echo

# if [[ -d $OUT_DIR ]]; then
#     echo "Emptying $OUT_DIR contents..." && echo
#     rm -rf $OUT_DIR*
# 

# make outdir if it dne
if ! [[ -d $OUT_DIR ]]; then
    echo "Creating output directory: $OUT_DIR"
    mkdir -p $OUT_DIR
fi

# check if the input file exists
if ! [[ -f $WORKING_DIR/drexm_input.dat ]]; then
    echo "D-REX_M input file \"drexm_input.dat\" does not exist or is improperly named."
    exit 1
fi

# load necessary modules
echo "Loading modules..."
module load hdf5/1.14.5%oneapi@2025.0.0
module load openmpi/5.0.5%oneapi@2025.0.0 # it will auto unload the other mpi version

# run drexm
echo "Running ECOMAN/D-REX_M on $MODEL"
srun ./drex_m $WORKING_DIR/drexm_input.dat
