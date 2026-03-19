#!/bin/bash
#SBATCH --job-name=sks_split_model_28.37
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/28.37/ecoman/sks-split/out_%j/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/28.37/ecoman/sks-split/out_%j/%x_%j.err

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=25 # cores 
#SBATCH --mem=512G
#SBATCH --constraint='zen|zen2|zen3'
#SBATCH --partition=high
#SBATCH --hint=nomultithread

#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu

SRC_DIR=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-seismology.SKS-SPLIT

SJI=$1
DJI=$2

if [[ -z "$SJI" ]]; then
    echo "Error: SJI (aspect2ecoman job ID) not set. Please provide it as the first argument."
    exit 1
fi

if [[ -z "$DJI" ]]; then
    echo "Error: DJI (D-REX_M job ID) not set. Please provide it as the second argument."
    exit 1
fi

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=28.37
MODEL_DIR="$BASEDIR/$MODEL"
PP_DIR="/quobyte/billengrp/nredick/ecoman/aspect2ecoman/out_$SJI"
OUTPUT_DIR="$MODEL_DIR/ecoman/sks-split/out_$SLURM_JOB_ID"
cijkl_dir="$MODEL_DIR/ecoman/drexm/out_$DJI"

echo "[$(basename $0)] Running SKS-SPLIT on model \"$MODEL\""
echo "[$(basename $0)] Model directory:            $MODEL_DIR"
echo "[$(basename $0)] Preprocessed data directory: $PP_DIR"
echo "[$(basename $0)] Cijkl directory:             $cijkl_dir"
echo "[$(basename $0)] Output directory:            $OUTPUT_DIR"

# validate inputs
if ! [[ -f "$PP_DIR/stack_input.dat" ]]; then
    echo "Error: stack_input.dat not found in $PP_DIR"
    exit 1
fi

if ! [[ -d "$cijkl_dir" ]]; then
    echo "Error: cijkl_dir does not exist: $cijkl_dir"
    exit 1
fi

# load modules
echo "[$(basename $0)] Loading modules..."
module load oneapi/2025.0.0 hdf5/1.14.5%oneapi@2025.0.0
echo

# set up WORK_DIR
WORK_DIR="/nfs/hive/scratch-mpi-io/$SLURM_JOB_ID"
mkdir -p "$WORK_DIR" && cd "$WORK_DIR" || exit 1

# cp instead of symlink to avoid stale file handling errors across diff filesystems
cp -r "$SRC_DIR/." "$WORK_DIR/run/"

# copy stack_input.dat and Cijkl data into WORK_DIR so all I/O stays local to scratch-mpi-io
cp "$PP_DIR/stack_input.dat" "$WORK_DIR/"
cp -r "$cijkl_dir" "$WORK_DIR/cijkl"
LOCAL_CIJKL="$WORK_DIR/cijkl" # cijkl dir on the nfs (scratch)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

############################# Control parameters #############################

timestep=0001                   # Cijkl*.h5 file number with 4 digits: e.g. 40 must be 0040

mode=0                          # 0: all layers 
                                # 1: remove lithosphere 
                                # 2: remove deep layers

split_dir=splitting             # Define sub-directory of cijkl_dir where splitting parameters will be saved

avg=0                           # 0: use true seismograms from reflectivity
                                # 1: based on averaged tensors

verbose=1                       # verbosity level

save_bascan=1                   # save the actual back-azimuth dependence?

idisp=25                        # way to compute splitting,
                                # 15: Levin and Park 1999
                                # 25: Menke

varpar=1                        # depth variable parameters
density=3.353                   # reference density
incidence=5                     # incidence angle for SKS
bottom=660                      # Set depth of stack of layers in km
lith_bot=0                      # calc splitting from layers below "bottom" everywhere if mode = 1
execute=1                       # 0 = dry run (no outputs in splitting/), 1 = exec as normal 

##############################################################################

pushd "$WORK_DIR/run"

# run stack_calc
echo "[$(basename $0)] Running stack_calc..."
srun --ntasks=1 stack_calc "$LOCAL_CIJKL" "$WORK_DIR/stack_input.dat" "$WORK_DIR" $timestep $mode
echo "[$(basename $0)] stack_calc finished with exit code $?"

# run split_calc
echo "[$(basename $0)] Running split_calc..."
mpirun -np $SLURM_NTASKS split_calc "$LOCAL_CIJKL" $timestep $mode $split_dir $avg $verbose $save_bascan $idisp $varpar $density $incidence $bottom $lith_bot $execute
echo "[$(basename $0)] split_calc finished with exit code $?"

popd

# move outputs from scratch to OUTPUT_DIR
echo "[$(basename $0)] Collecting outputs to $OUTPUT_DIR..."
mkdir -p "$OUTPUT_DIR"

mv "$LOCAL_CIJKL/splitting" "$LOCAL_CIJKL/seismic_stations" "$OUTPUT_DIR/"
mv "$WORK_DIR/run/cpu*" "$OUTPUT_DIR/"
mv "$WORK_DIR"/stack_input.dat "$OUTPUT_DIR/" 2>/dev/null

# save a copy of this script alongside the outputs
cp "$(realpath "$0")" "$OUTPUT_DIR/"

echo "[$(basename $0)] Done. Executed as: sbatch sks_split_model_28.37.slurm $SJI $DJI"
>&2 echo "$(pwd)"