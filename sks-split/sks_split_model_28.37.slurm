#!/bin/bash
#SBATCH --job-name=sks_split_model_28.37
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/28.37/ecoman/sks-split/out_%j/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/28.37/ecoman/sks-split/out_%j/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=high
#SBATCH --mem=512G                  # to use magali's nodes
#SBATCH --constraint='zen|zen2|zen3'        # do not run or compile on zen4 nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
#SBATCH --chdir=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-seismology.SKS-SPLIT

# load necessary modules
echo "[$(basename $0)] Loading modules..."
module load oneapi/2025.0.0 hdf5/1.14.5%oneapi@2025.0.0
# module load openmpi/5.0.5%oneapi@2025.0.0 # it will auto unload the other mpi version
echo

# set number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=28.37
SJI=$1
DJI=$2

# if sji not set, fail 
if [[ -z "$SJI" ]]; then
    echo "Error: SJI (preprocessing job ID) not set. Please provide it as the first argument."
    exit 1
fi

if [[ -z "$DJI" ]]; then
    echo "Error: DJI (D-REX_M job ID) not set. Please provide it as the second argument."
    exit 1
fi

PP_DIR="/quobyte/billengrp/nredick/ecoman/preprocessing/out_$SJI" # where to copy dat and vtp from (preprocessing)
MODEL_DIR="$BASEDIR/$MODEL" # model directory
# OUTPUT_DIR="$MODEL_DIR/ecoman/sks-split" # model inputs (stack_input.dat) & slurm outputs
OUTPUT_DIR="$MODEL_DIR/ecoman/sks-split/out_$SLURM_JOB_ID" # will contain sks-split outputs

echo "[$(basename $0)] Running SKS-SPLIT on model \"$MODEL\""
echo "[$(basename $0)] Model directory: $MODEL_DIR"
# echo "[$(basename $0)] Working directory: $OUTPUT_DIR"
echo "[$(basename $0)] Preprocessed data directory: $PP_DIR"
echo "[$(basename $0)] Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/" 2> /dev/null

# copy stack_input.dat from PP_DIR to OUTPUT_DIR
echo "[$(basename $0)] Copying stack_input.dat to $PP_DIR/stack_input.dat ..."
cp "$PP_DIR/stack_input.dat" "$OUTPUT_DIR/"

# check if the input file exists
if ! [[ -f "$OUTPUT_DIR/stack_input.dat" ]]; then
    echo "[$(basename $0)] SKS-SPLIT input file \"stack_input.dat\" does not exist or is improperly named."
    exit 1
fi

############################# Define control parameters #############################

cijkl_dir="$MODEL_DIR/ecoman/drexm/out_$DJI" # loc of Cijkl*.h5 file to be loaded
echo "[$(basename $0)] Cijkl dir: $cijkl_dir"

timestep=0001                   # Cijkl*.h5 file number with 4 digits: e.g. 40 must be 0040

mode=0                          # 0: all layers 1: remove lithosphere 2: remove deep layers

split_dir=splitting	        # Define sub-directory of cijkl_dir where splitting parameters will be saved

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

bottom=6371.5                      # Set depth of stack of layers in km

lith_bot=0                    # calculate splitting due to layers below this depth everywhere if mode = 1

execute=1                       # set to zero for debugging

############################# Define control parameters #############################

# Generate stack of horizontal layers with different elastic tensors (scaled by density) beneath each seismic station
srun ./stack_calc $cijkl_dir "$OUTPUT_DIR/stack_input.dat" $OUTPUT_DIR $timestep $mode
# print completed this
echo "[$(basename $0)] stack_calc finished running with exit code $?"

wait

# Compute splitting parameters averaged over back-azimuth
srun ./split_calc $cijkl_dir $timestep $mode $split_dir $avg $verbose $save_bascan $idisp $varpar $density $incidence $bottom $lith_bot $execute
echo "[$(basename $0)] split_calc finished running with exit code $?"


# # todo clean up
# rm -rf $OUTPUT_DIR/cpu*
# # mv -f "$cijkl_dir/seismic_stations/" "$OUTPUT_DIR/"
# mv -f cpu* "$OUTPUT_DIR/"
# mv -f "$cijkl_dir/seismic_stations/" "$cijkl_dir/splitting/" 

# "$cijkl_dir/stack_input.dat" "$OUTPUT_DIR/"
# rm "$cijkl_dir/savstack"
# rm "$cijkl_dir/stack_input.dat"

mv "$cijkl_dir/splitting/" "$cijkl_dir/seismic_stations/" "$cijkl_dir/savstack" "$OUTPUT_DIR/" 
rm "$cijkl_dir/stack_input.dat"

# Get the full path of the current script
current_path=$(realpath "$0")
cp $current_path "$OUTPUT_DIR/" 