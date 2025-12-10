#!/bin/bash
#SBATCH --job-name=sks_split_model
#SBATCH --output=/quobyte/billengrp/nredick/csz-models/a03.02.28_Wn1.0e-20_Wc1.0e-21_Ws2.5e-22_wet0010_a03.38-EFns-short-tipN050_C-00_S-50-wkz75-nosplit-Gw0.00/ecoman/sks-split/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/csz-models/a03.02.28_Wn1.0e-20_Wc1.0e-21_Ws2.5e-22_wet0010_a03.38-EFns-short-tipN050_C-00_S-50-wkz75-nosplit-Gw0.00/ecoman/sks-split/%x_%j.err
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=high
#SBATCH --mem=512G                  # to use magali's nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
#SBATCH --chdir=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-seismology.SKS-SPLIT

# load necessary modules
echo "[$(basename $0)] Loading modules..."
module load hdf5/1.14.5%oneapi@2025.0.0
module load openmpi/5.0.5%oneapi@2025.0.0 # it will auto unload the other mpi version
echo

# set number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASEDIR=/quobyte/billengrp/nredick/csz-models
MODEL=a03.02.28_Wn1.0e-20_Wc1.0e-21_Ws2.5e-22_wet0010_a03.38-EFns-short-tipN050_C-00_S-50-wkz75-nosplit-Gw0.00

MODEL_DIR="$BASEDIR/$MODEL" # model directory
WORKING_DIR="$MODEL_DIR/ecoman/sks-split" # model inputs (stack_input.dat) & slurm outputs
OUTPUT_DIR="$WORKING_DIR/out" # will contain sks-split outputs

echo "[$(basename $0)] Running SKS-SPLIT on model \"$MODEL\""
echo "[$(basename $0)] Model directory: $MODEL_DIR"
# echo "[$(basename $0)] Working directory: $WORKING_DIR"
echo "[$(basename $0)] Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/" 2> /dev/null

# empty the dir if it exists
if [[ -d $OUTPUT_DIR ]]; then
    echo "[$(basename $0)] Emptying $OUTPUT_DIR contents..."
    rm -rf "$OUTPUT_DIR/*"
fi

# check if the input file exists
if ! [[ -f "$WORKING_DIR/stack_input.dat" ]]; then
    echo "[$(basename $0)] SKS-SPLIT input file \"stack_input.dat\" does not exist or is improperly named."
    exit 1
fi

############################# Define control parameters #############################

cijkl_dir="$MODEL_DIR/ecoman/drexm/out" # loc of Cijkl*.h5 file to be loaded
echo "[$(basename $0)] Cijkl dir: $cijkl_dir"

timestep=0001                   # Cijkl*.h5 file number with 4 digits: e.g. 40 must be 0040

mode=0                          # 0: all layers 1: remove lithosphere 2: remove deep layers

split_dir=splitting	        # Define sub-directory of cijkl_dir where splitting parameters will be saved

avg=1                           # 0: use true seismograms from reflectivity
                                # 1: based on averaged tensors

verbose=2                       # verbosity level

save_bascan=1                   # save the actual back-azimuth dependence?

idisp=15                        # way to compute splitting,
                                # 15: Levin and Park 1999
                                # 25: Menke

varpar=1                        # depth variable parameters

density=3.353                   # reference density

incidence=5                     # incidence angle for SKS

bottom=6371.5                      # Set depth of stack of layers in km

lith_bot=0                    # calculate splitting due to layers below this depth everywhere if mode = 1

execute=0                       # set to zero for debugging

############################# Define control parameters #############################

# Generate stack of horizontal layers with different elastic tensors (scaled by density) beneath each seismic station
srun ./stack_calc $cijkl_dir "$WORKING_DIR/stack_input.dat" $OUTPUT_DIR $timestep $mode
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
mv -f "$cijkl_dir/seismic_stations" "$cijkl_dir/splitting" "$cijkl_dir/stack_input.dat" "$cijkl_dir/savstack" "$OUTPUT_DIR/"
# rm "$cijkl_dir/savstack"
# rm "$cijkl_dir/stack_input.dat"