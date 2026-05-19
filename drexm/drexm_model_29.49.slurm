#!/bin/bash
#SBATCH --job-name=drexm_29.49

#SBATCH --output=/quobyte/billengrp/nredick/csz-models/29.49/ecoman/drexm/out_%j/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/29.49/ecoman/drexm/out_%j/%x_%j.err

#SBATCH --time=24:00:00
#SBATCH --ntasks=16 # must == product of MPI proc distribution along axis from DAT file
#SBATCH --cpus-per-task=6 
#SBATCH --mem-per-cpu=6G

#SBATCH --constraint='zen|zen2|zen3'        # do not run or compile on zen4 nodes
#SBATCH --partition=high

#SBATCH --account=billengrp
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=nrredick@ucdavis.edu

SRC_DIR=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=29.49
AEID="$1" # id of the preproccessing (aspect2ecoman job)
# echo "[$(basename $0)] AEID = $AEID" && echo

# if AEID not set, fail 
if [[ -z "$AEID" ]]; then
    echo "[$(basename $0)] Error: AEID (aspect2ecoman job ID) not set. Please provide it as the first argument."
    exit 1
fi

PP_DIR=/quobyte/billengrp/nredick/csz-models/$MODEL/ecoman/aspect2ecoman/out_$AEID # where to copy dat and vtp from (aspect2ecoman)
MODEL_DIR="$BASEDIR/$MODEL" # model directory
OUT_DIR="$MODEL_DIR/ecoman/drexm/out_$SLURM_JOB_ID" # model inputs (drexm_input.dat) & slurm outputs

# set up WORK_DIR
WORK_DIR="/nfs/hive/scratch-mpi-io/$SLURM_JOB_ID"
mkdir -p "$WORK_DIR" && cd "$WORK_DIR" || exit 1

# rsync to make it easier to exclude unnecessary subdirs
rsync -avr --exclude '.git' --exclude 'cookbooks' --exclude 'D-REX_S' --exclude 'EXEV' --exclude 'VIZTOMO' --exclude 'VIZVISC' --exclude 'compile_ECOMAN2.0-geodynamics' "$SRC_DIR/" "$WORK_DIR/src/"

echo "[$(basename $0)] Running D-REX_M on model \"$MODEL\"" && echo
echo "[$(basename $0)] Model directory: $MODEL_DIR/"
echo "[$(basename $0)] Preprocessed data directory: $PP_DIR"
echo "[$(basename $0)] Working directory: $WORK_DIR"
echo "[$(basename $0)] Output directory: $OUT_DIR" && echo

# copy vtp0001.h5 and drexm_input.dat from aspect2ecoman output to working directory
echo "[$(basename $0)] Copying vtp0001.h5 and drexm_input.dat from aspect2ecoman output to $WORK_DIR ..."
cp $PP_DIR/vtp0001.h5 $WORK_DIR/
cp $PP_DIR/drexm_input.dat $WORK_DIR/

# make outdir if it dne
if ! [[ -d $OUT_DIR ]]; then
    echo "[$(basename $0)] Creating output directory: $OUT_DIR"
    mkdir -p $OUT_DIR
fi

# check if the input file exists
if ! [[ -f $WORK_DIR/drexm_input.dat ]]; then
    echo "[$(basename $0)] D-REX_M input file \"drexm_input.dat\" does not exist or is improperly named."
    exit 1
fi

# load necessary modules
echo "[$(basename $0)] Loading modules..."
module load oneapi/2025.0.0 hdf5/1.14.5%oneapi@2025.0.0 pmix/5.0.3%oneapi@2025.0.0

# update the input/output paths in the dat file
echo "[$(basename $0)] Updating dat file with correct paths..."
sed -i "/^# input_dir: path to input directory/{ n; s|.*|${WORK_DIR}/|; }" $WORK_DIR/drexm_input.dat
sed -i "/^# output_dir: path to output directory/{ n; s|.*|${WORK_DIR}/|; }" $WORK_DIR/drexm_input.dat

pushd "$WORK_DIR/src/D-REX_M"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# run drexm
echo "[$(basename $0)] Running ECOMAN/D-REX_M on $MODEL"
srun --cpu-bind=cores ./drex_m $WORK_DIR/drexm_input.dat

popd

# cp this file to out dir for record keeping 
current_path=$(realpath "$0")
cp $current_path "$WORK_DIR/" 

# copy outputs back to quobyte 
rsync -avr --exclude 'src' "$WORK_DIR/" "$OUT_DIR/" 

pushd $OUT_DIR

/home/nredick/nredick-pfs/mysoftware/scripts/extract_stats.sh drexm*.out

popd 

echo "[$(basename $0)] Done. Executed as: sbatch drexm_model_29.49.slurm $AEID"