#!/bin/bash
#SBATCH --job-name=sks_split_cookbook
#SBATCH --output=/quobyte/billengrp/nredick/ecoman/out/2Dcartesian_convection/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/ecoman/out/2Dcartesian_convection/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=high
#SBATCH --mem=512G      # to use magali's nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
#SBATCH --chdir=/quobyte/billengrp/Software/ECOMAN/ECOMAN2.0-seismology.SKS-SPLIT

module load hdf5/1.14.5%oneapi@2025.0.0

# set number of threads
export OMP_NUM_THREADS=1

COOKBOOK=2Dcartesian_convection
OUTDIR="/quobyte/billengrp/nredick/ecoman/out/$COOKBOOK"

# Define control parameters
cijkl_dir="/quobyte/billengrp/nredick/ecoman/out/2Dcartesian_convection" # loc of Cijkl*.h5 file to be loaded

# need the input dat file from the cookbooks dir
cp /quobyte/billengrp/Software/ECOMAN/ECOMAN2.0-geodynamics/cookbooks/2Dcartesian_convection/stack_input.dat $cijkl_dir

timestep=0001                   # Cijkl*.h5 file number with 4 digits: e.g. 40 must be 0040

mode=0                          # 0: all layers 1: remove lithosphere 2: remove deep layers

split_dir=splitting	        # Define sub-directory of cijkl_dir where splitting parameters will be saved

avg=0                           # 0: use true seismograms from reflectivity
                                # 1: based on averaged tensors

verbose=0                       # verbosity level

save_bascan=1                   # save the actual back-azimuth dependence?

idisp=15                        # way to compute splitting,
                                # 15: Levin and Park 1999
                                # 25: Menke

varpar=1                        # depth variable parameters

density=3.353                   # reference density

incidence=5                     # incidence angle for SKS

bottom=400                      # Set depth of stack of layers in km

lith_bot=100                    # calculate splitting due to layers below this depth everywhere if mode = 1

execute=1                       # set to zero for debugging

#Generate stack of horizontal layers with different elastic tensors (scaled by density) beneath each seismic station
./stack_calc $cijkl_dir $timestep $mode

wait

#Compute splitting parameters averaged over back-azimuth
srun ./split_calc $cijkl_dir $timestep $mode $split_dir $avg $verbose $save_bascan $idisp $varpar $density $incidence $bottom $lith_bot $execute

rm -rf $OUTDIR/cpu*
mv -f cpu* $OUTDIR