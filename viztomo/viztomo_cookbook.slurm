#!/bin/bash
#SBATCH --job-name=viztomo_cookbook
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
#SBATCH --chdir=/quobyte/billengrp/Software/ECOMAN/ECOMAN2.0-geodynamics/VIZTOMO

BASEDIR="/quobyte/billengrp/Software/ECOMAN/ECOMAN2.0-geodynamics"
COOKBOOK=2Dcartesian_convection

COOKBOOKDIR="$BASEDIR/cookbooks/$COOKBOOK"
RUNDIR="$BASEDIR/VIZTOMO"
OUTDIR="/quobyte/billengrp/nredick/ecoman/out/$COOKBOOK"

# if [[ ! -d "$RUNDIR/out/" ]]; then
#     mkdir "$RUNDIR/out/"
# fi

mkdir -p $COOKBOOKDIR/out/
# echo $COOKBOOKDIR/out/

# if [[ -d "$COOKBOOKDIR/out/" ]]; then
#     rm -rf "$COOKBOOKDIR/out/"
# fi

# if [[ -s "$OUTDIR" ]]; then
#     rm -rf "$OUTDIR/*"
# fi

# todo see which of these actually need to be loaded (if any)
echo "Loading modules for job"
module load hdf5/1.14.5%oneapi@2025.0.0
module load openmpi/5.0.5%oneapi@2025.0.0 # it will auto unload the other mpi version

# cp cijkl file back to grpsoftware
cp $OUTDIR/Cijkl0001.h5 $COOKBOOKDIR/out/

echo "Running ECOMAN/VIZTOMO on $COOKBOOK"

srun ./viztomo $COOKBOOKDIR/viztomo_input.dat

# # move the outputs to pfs
mv $COOKBOOKDIR/out/* $OUTDIR
