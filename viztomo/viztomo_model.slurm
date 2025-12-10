#!/bin/bash
#SBATCH --job-name=viztomo_model
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/a03.02.28_Wn1.0e-20_Wc1.0e-21_Ws2.5e-22_wet0010_a03.38-EFns-short-tipN050_C-00_S-50-wkz75-nosplit-Gw0.00/ecoman/viztomo/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/a03.02.28_Wn1.0e-20_Wc1.0e-21_Ws2.5e-22_wet0010_a03.38-EFns-short-tipN050_C-00_S-50-wkz75-nosplit-Gw0.00/ecoman/viztomo/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=high
#SBATCH --mem=512G      # to use magali's nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
#SBATCH --chdir=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics/VIZTOMO

echo "Loading modules for job"
module load hdf5/1.14.5%oneapi@2025.0.0
module load openmpi/5.0.5%oneapi@2025.0.0 # it will auto unload the other mpi version

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=a03.02.28_Wn1.0e-20_Wc1.0e-21_Ws2.5e-22_wet0010_a03.38-EFns-short-tipN050_C-00_S-50-wkz75-nosplit-Gw0.00

MODEL_DIR="$BASEDIR/$MODEL" # model directory
WORKING_DIR="$MODEL_DIR/ecoman/viztomo" # model inputs (viztomo_input.dat) & slurm outputs
OUTPUT_DIR="$WORKING_DIR/out" # will contain viztomo outputs

mkdir -p $OUTPUT_DIR 2> /dev/null

# # cp cijkl file back to grpsoftware
# cp $OUTDIR/Cijkl0001.h5 $OUTPUT_DIR

echo "Running ECOMAN/VIZTOMO on $MODEL"
srun ./viztomo $WORKING_DIR/viztomo_input.dat

# move the outputs to pfs
# mv $COOKBOOKDIR/out/* $OUTDIR
