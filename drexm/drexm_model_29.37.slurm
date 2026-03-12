#!/bin/bash
#SBATCH --job-name=drexm_model_29.37
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/29.37/ecoman/drexm/out_%j/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/29.37/ecoman/drexm/out_%j/%x_%j.err
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=125                          # must be equal to product of MPI proc distribution along axis from DAT file
# SBATCH --partition=high
#SBATCH --mem=512G                          # to use magali's nodes
#SBATCH --constraint='zen|zen2|zen3'        # do not run or compile on zen4 nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
#SBATCH --chdir=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics/D-REX_M

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=29.37
SJI=$1

# if sji not set, fail 
if [[ -z "$SJI" ]]; then
    echo "Error: SJI (preprocessing job ID) not set. Please provide it as the first argument."
    exit 1
fi

PP_DIR=/quobyte/billengrp/nredick/ecoman/preprocessing/out_$SJI # where to copy dat and vtp from (preprocessing)
MODEL_DIR="$BASEDIR/$MODEL" # model directory
OUT_DIR="$MODEL_DIR/ecoman/drexm/out_$SLURM_JOB_ID" # model inputs (drexm_input.dat) & slurm outputs

echo "Running D-REX_M on model \"$MODEL\"" && echo
echo "Model directory: $MODEL_DIR/"
echo "Preprocessed data directory: $PP_DIR"
echo "Output directory: $OUT_DIR" && echo

# if [[ -d $OUT_DIR ]]; then
#     echo "Emptying $OUT_DIR contents..." && echo
#     rm -rf $OUT_DIR*
# 

# copy vtp0001.h5 and drexm_input.dat from preprocessing output to working directory
echo "Copying vtp0001.h5 and drexm_input.dat from preprocessing output to $OUT_DIR ..."
cp $PP_DIR/vtp0001.h5 $OUT_DIR/
cp $PP_DIR/drexm_input.dat $OUT_DIR/

# make outdir if it dne
if ! [[ -d $OUT_DIR ]]; then
    echo "Creating output directory: $OUT_DIR"
    mkdir -p $OUT_DIR
fi

# check if the input file exists
if ! [[ -f $OUT_DIR/drexm_input.dat ]]; then
    echo "D-REX_M input file \"drexm_input.dat\" does not exist or is improperly named."
    exit 1
fi

# load necessary modules
echo "Loading modules..."
module load hdf5/1.14.5%oneapi@2025.0.0
module load openmpi/5.0.5%oneapi@2025.0.0 # it will auto unload the other mpi version

# update the input/output paths in the dat file
echo "Updating dat file with correct paths..."
sed -i "/^# input_dir: path to input directory/{ n; s|.*|${OUT_DIR}/|; }" $OUT_DIR/drexm_input.dat
sed -i "/^# output_dir: path to output directory/{ n; s|.*|${OUT_DIR}/|; }" $OUT_DIR/drexm_input.dat

# run drexm
echo "Running ECOMAN/D-REX_M on $MODEL"
srun ./drex_m $OUT_DIR/drexm_input.dat

# mv outputs to correct directory
# mv $BASEDIR/$MODEL/out $OUT_DIR
