#!/bin/bash
#SBATCH --job-name=viztomo_model_28.35
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/28.35/ecoman/viztomo/out_%j/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/28.35/ecoman/viztomo/out_%j/%x_%j.err

#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1 # needs to be 1 for viztomo
#SBATCH --mem=512G                          # to use magali's nodes
#SBATCH --constraint='zen|zen2|zen3'        # do not run or compile on zen4 nodes
#SBATCH --partition=high
#SBATCH --hint=nomultithread

#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
#SBATCH --chdir=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics/VIZTOMO


BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=28.35
SJI=$1
DJI=$2

if [[ -z "$SJI" ]]; then
    echo "Error: SJI (preprocessing job ID) not set. Please provide it as the 1st argument."
    exit 1
fi

if [[ -z "$DJI" ]]; then
    echo "Error: DJI (D-REX_M job ID) not set. Please provide it as the 2nd argument."
    exit 1
fi

echo "SCRIPT PARAMS: SJI=$SJI DJI=$DJI"

PP_DIR=/quobyte/billengrp/nredick/ecoman/preprocessing/out_$SJI # where to copy dat and vtp from (preprocessing)
DREXM_DIR="$BASEDIR/$MODEL/ecoman/drexm/out_$DJI" # where to read Cijkl from (drexm output)
MODEL_DIR="$BASEDIR/$MODEL" # model directory
WORKING_DIR="$MODEL_DIR/ecoman/viztomo" # model inputs (viztomo_input.dat)
OUT_DIR="$WORKING_DIR/out_$SLURM_JOB_ID" # viztomo output directory

echo "Running VIZTOMO on model \"$MODEL\"" && echo
echo "Model directory: $MODEL_DIR/"
echo "D-REX_M output directory: $DREXM_DIR"
echo "Output directory: $OUT_DIR" && echo

# make outdir if it dne
if ! [[ -d $OUT_DIR ]]; then
    echo "Creating output directory: $OUT_DIR"
    mkdir -p $OUT_DIR
fi

# copy viztomo_input.dat from working directory to output directory
echo "Copying viztomo_input.dat to $OUT_DIR ..."
cp $PP_DIR/viztomo_input.dat $OUT_DIR/

# check if the input file exists
if ! [[ -f $OUT_DIR/viztomo_input.dat ]]; then
    echo "VIZTOMO input file \"viztomo_input.dat\" does not exist or is improperly named."
    exit 1
fi

# load necessary modules
module load oneapi/2025.0.0 hdf5/1.14.5%oneapi@2025.0.0

# update the input/output paths in the dat file
echo "Updating dat file with correct paths..."
sed -i "/^# cijkl_dir: path to directory where to read Cijkl\* files/{ n; s|.*|${DREXM_DIR}/|; }" $OUT_DIR/viztomo_input.dat
sed -i "/^# output_dir: path to directory where to save output files/{ n; s|.*|${OUT_DIR}/|; }" $OUT_DIR/viztomo_input.dat

# run viztomo
echo "Running ECOMAN/VIZTOMO on $MODEL"
srun ./viztomo $OUT_DIR/viztomo_input.dat

# cp this file to out dir for record keeping 
current_path=$(realpath "$0")
cp $current_path "$OUT_DIR/" 