#!/bin/bash
#SBATCH --job-name=sks_split_29.53
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/29.53/ecoman/sks-split/out_%j/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/29.53/ecoman/sks-split/out_%j/%x_%j.err

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32 # cores 
#SBATCH --mem=512G
#SBATCH --constraint='zen|zen2|zen3'
#SBATCH --partition=high

#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu

SRC_DIR=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-seismology.SKS-SPLIT

AEID=$1
DJI=$2
BOT=$3

if [[ -z "$AEID" ]]; then
    echo "Error: AEID (aspect2ecoman job ID) not set. Please provide it as the first argument."
    exit 1
fi

if [[ -z "$DJI" ]]; then
    echo "Error: DJI (D-REX_M job ID) not set. Please provide it as the second argument."
    exit 1
fi

if [[ -z "$BOT" ]]; then
    echo "Error: BOT (bottom depth in km) not set. Please provide it as the third argument."
    exit 1
fi

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=29.53
MODEL_DIR="$BASEDIR/$MODEL"
PP_DIR=/quobyte/billengrp/nredick/csz-models/$MODEL/ecoman/aspect2ecoman/out_$AEID
OUT_DIR="$MODEL_DIR/ecoman/sks-split/out_$SLURM_JOB_ID"
cijkl_dir="$MODEL_DIR/ecoman/drexm/out_$DJI"

echo "[$(basename $0)] Running SKS-SPLIT on model \"$MODEL\""
echo "[$(basename $0)] Model directory:            $MODEL_DIR"
echo "[$(basename $0)] Preprocessed data directory: $PP_DIR"
echo "[$(basename $0)] Cijkl directory:             $cijkl_dir"
echo "[$(basename $0)] Output directory:            $OUT_DIR"

# Validate inputs
if ! [[ -f "$PP_DIR/stack_input.dat" ]]; then
    echo "Error: stack_input.dat not found in $PP_DIR"
    exit 1
fi

if ! [[ -d "$cijkl_dir" ]]; then
    echo "Error: cijkl_dir does not exist: $cijkl_dir"
    exit 1
fi

# Load modules
echo "[$(basename $0)] Loading modules..."
module load oneapi/2025.0.0 hdf5/1.14.5%oneapi@2025.0.0 pmix/5.0.3%oneapi@2025.0.0
echo

# --- ISOLATED WORK DIR IN SCRATCH ---
WORK_DIR="/nfs/hive/scratch-mpi-io/j_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR" || exit 1
cd "$WORK_DIR" || exit 1

# Copy source code to isolated workspace
rsync -avr --exclude ".git" --exclude "viz" "$SRC_DIR/" "$WORK_DIR/src/"
rsync -avr "$PP_DIR/stack_input.dat" "$WORK_DIR/"

# --- ISOLATED CIJKL WORKSPACE ---
# Copy Cijkl input files locally to prevent multiple jobs from stepping on each other's files in Quobyte
mkdir -p "$WORK_DIR/c"
rsync -avr --exclude="splitting/" --exclude="seismic_stations/" "$cijkl_dir/" "$WORK_DIR/c/"
LOCAL_CIJKL="$WORK_DIR/c"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

############################# Control parameters #############################

timestep=0001                   # Cijkl*.h5 file number with 4 digits: e.g. 40 must be 0040

mode=1                          # 0: all layers 
                                # 1: remove lithosphere 
                                # 2: remove deep layers

split_dir="splitting"           # Safe to keep as standard name inside isolated WORK_DIR

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
echo "[$(basename $0)] using bottom = $BOT"
bottom=$BOT                      # Set depth of stack of layers in km
lith_bot=100                    # calc splitting from layers below "bottom" everywhere if mode = 1
execute=1                       # 0 = dry run (no outputs in splitting/), 1 = exec as normal 
truncate=1                      # 1 = ignore data below bottom, 0 = operate normally (all layers below lithbot)


##############################################################################

pushd "$WORK_DIR/src" > /dev/null

# Run stack_calc
echo "[$(basename $0)] Running stack_calc..."
srun --ntasks=1 ./stack_calc "$LOCAL_CIJKL" "$WORK_DIR/stack_input.dat" "$WORK_DIR" $timestep $mode
echo "[$(basename $0)] stack_calc finished with exit code $?"

# Run split_calc
echo "[$(basename $0)] Running split_calc..."
mpirun -np $SLURM_NTASKS ./split_calc "$LOCAL_CIJKL" $timestep $mode $split_dir $avg $verbose $save_bascan $idisp $varpar $density $incidence $bottom $lith_bot $execute $truncate 
echo "[$(basename $0)] split_calc finished with exit code $?"

popd > /dev/null

# Collect outputs to OUT_DIR
echo "[$(basename $0)] Collecting outputs to $OUT_DIR..."
mkdir -p "$OUT_DIR"

if [[ -d "$LOCAL_CIJKL/splitting" ]]; then
    rsync -avr "$LOCAL_CIJKL/splitting/" "$OUT_DIR/splitting/"
fi

if [[ -d "$LOCAL_CIJKL/seismic_stations" ]]; then
    rsync -avr "$LOCAL_CIJKL/seismic_stations/" "$OUT_DIR/seismic_stations/"
fi

# Copy station .sav files if created in LOCAL_CIJKL root
rsync -avr --include="*.sav" --exclude="*" "$LOCAL_CIJKL/" "$OUT_DIR/" 2>/dev/null || true

# Copy cpu logs & inputs
rsync -avr --include="cpu*/" --include="cpu*/**" --exclude="*" "$WORK_DIR/src/" "$OUT_DIR/cpus/"
rsync -avr "$WORK_DIR/stack_input.dat" "$OUT_DIR/"

# Save job script
cp "$(realpath "$0")" "$OUT_DIR/"

# Clean up temporary scratch directory
rm -rf "$WORK_DIR"

echo "[$(basename $0)] Done. Executed as: sbatch sks_split_29.53.slurm $AEID $DJI with bottom=$BOT"