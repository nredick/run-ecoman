#!/bin/bash
#SBATCH --job-name=viztomo_29.37
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/29.37/ecoman/viztomo/out_%j/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/29.37/ecoman/viztomo/out_%j/%x_%j.err

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1 # needs to be 1 for viztomo
#SBATCH --mem=512G                          # to use magali's nodes
#SBATCH --constraint='zen|zen2|zen3'        # do not run or compile on zen4 nodes
#SBATCH --partition=high
# SBATCH --hint=nomultithread

#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
# SBATCH --chdir=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics/VIZTOMO

SRC_DIR=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=29.37
AEID=$1
DJI=$2

if [[ -z "$AEID" ]]; then
    echo "[$(basename $0)] Error: AEID (aspect2ecoman job ID) not set. Please provide it as the 1st argument."
    exit 1
fi

if [[ -z "$DJI" ]]; then
    echo "[$(basename $0)] Error: DJI (D-REX_M job ID) not set. Please provide it as the 2nd argument."
    exit 1
fi

echo "[$(basename $0)] SCRIPT PARAMS: AEID=$AEID DJI=$DJI"

DREXM_DIR="$BASEDIR/$MODEL/ecoman/drexm/out_$DJI" # where to read Cijkl from (drexm output)
PP_DIR=/quobyte/billengrp/nredick/csz-models/$MODEL/ecoman/aspect2ecoman/out_$AEID # where to copy dat from (aspect2ecoman)
MODEL_DIR="$BASEDIR/$MODEL" # model directory
WORKING_DIR="$MODEL_DIR/ecoman/viztomo"
OUT_DIR="$WORKING_DIR/out_$SLURM_JOB_ID" # viztomo output directory

# set up WORK_DIR
WORK_DIR="/nfs/hive/scratch-mpi-io/$SLURM_JOB_ID"
mkdir -p "$WORK_DIR" && cd "$WORK_DIR" || exit 1

# rsync to make it easier to exclude unnecessary subdirs
rsync -avr --exclude '.git' --exclude 'cookbooks' --exclude 'D-REX_S' --exclude 'EXEV' --exclude 'D-REX_M' --exclude 'VIZVISC' --exclude 'compile_ECOMAN2.0-geodynamics' "$SRC_DIR/" "$WORK_DIR/src/"

echo "[$(basename $0)] Running VIZTOMO on model \"$MODEL\"" && echo
echo "[$(basename $0)] Model directory: $MODEL_DIR/"
echo "[$(basename $0)] Preprocessed data directory: $PP_DIR"
echo "[$(basename $0)] D-REX_M output directory: $DREXM_DIR"
# echo "[$(basename $0)] Working directory: $WORK_DIR"
echo "[$(basename $0)] Output directory: $OUT_DIR" && echo

# make outdir if it dne
if ! [[ -d $OUT_DIR ]]; then
    echo "[$(basename $0)] Creating output directory: $OUT_DIR"
    mkdir -p $OUT_DIR
fi

# copy viztomo_input.dat from working directory to output directory
echo "[$(basename $0)] Copying viztomo_input.dat to $WORK_DIR ..."
cp $PP_DIR/viztomo_input.dat $WORK_DIR/

# check if the input file exists
if ! [[ -f $WORK_DIR/viztomo_input.dat ]]; then
    echo "[$(basename $0)] VIZTOMO input file \"viztomo_input.dat\" does not exist or is improperly named."
    exit 1
fi

# load necessary modules
echo "[$(basename $0)] Loading modules..."
module load oneapi/2025.0.0 hdf5/1.14.5%oneapi@2025.0.0 pmix/5.0.3%oneapi@2025.0.0

# update the input/output paths in the dat file
echo "[$(basename $0)] Updating dat file with correct paths..."
sed -i "/^# cijkl_dir: path to directory where to read Cijkl\* files/{ n; s|.*|${DREXM_DIR}/|; }" $WORK_DIR/viztomo_input.dat
sed -i "/^# output_dir: path to directory where to save output files/{ n; s|.*|${WORK_DIR}/|; }" $WORK_DIR/viztomo_input.dat

pushd "$WORK_DIR/src/VIZTOMO"

# run viztomo
echo "[$(basename $0)] Running ECOMAN/VIZTOMO on $MODEL"
srun ./viztomo $WORK_DIR/viztomo_input.dat

popd

# mv "$WORK_DIR" "$OUT_DIR/"

rsync -avr --exclude 'src' "$WORK_DIR/" "$OUT_DIR/" 

echo "[$(basename $0)] Done. Executed as: sbatch viztomo_29.37.slurm $AEID $DJI"

# cp this file to out dir for record keeping 
current_path=$(realpath "$0")
cp $current_path "$OUT_DIR/" 