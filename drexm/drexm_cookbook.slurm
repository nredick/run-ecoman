#!/bin/bash
#SBATCH --job-name=drexm_cookbook
#SBATCH --output=/quobyte/billengrp/nredick/ecoman/out/2Dcartesian_convection/%x_%j.out
#SBATCH --error=/quobyte/billengrp/nredick/ecoman/out/2Dcartesian_convection/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8                          # must be equal to product of MPI proc distribution along axis from DAT file
#SBATCH --partition=high
#SBATCH --mem=512G                          # to use magali's nodes
#SBATCH --constraint='zen|zen2|zen3'        # do not run or compile on zen4 nodes
#SBATCH --account=billengrp
#SBATCH --mail-type=end
#SBATCH --mail-user=nrredick@ucdavis.edu
#SBATCH --chdir=/quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics/D-REX_M

# /quobyte/billengrp/Software/ECOMAN/ECOMAN2.0-geodynamics/D-REX_M
# /quobyte/billengrp/nredick/mysoftware/ECOMAN2.0-geodynamics/D-REX_M

BASEDIR="/quobyte/billengrp/Software/ECOMAN/ECOMAN2.0-geodynamics"
COOKBOOK=2Dcartesian_convection

COOKBOOKDIR="$BASEDIR/cookbooks/$COOKBOOK"
RUNDIR="$BASEDIR/D-REX_M"
OUTDIR="/quobyte/billengrp/nredick/ecoman/out/$COOKBOOK"

if [[ ! -d "$RUNDIR/out/" ]]; then
    mkdir "$RUNDIR/out/"
fi

if [[ -d "$COOKBOOKDIR/out/" ]]; then
    rm -rf "$COOKBOOKDIR/out/"
fi

if [[ -d "$OUTDIR" ]]; then
    rm -rf "$OUTDIR/*"
fi

# todo see which of these actually need to be loaded (if any)
echo "Loading modules for job"
module load hdf5/1.14.5%oneapi@2025.0.0
# module load intel-oneapi-mkl/2024.0.0%oneapi@2025.0.0
module load openmpi/5.0.5%oneapi@2025.0.0 # it will auto unload the other mpi version

echo "Running ECOMAN/D-REX_M on $COOKBOOK"

srun ./drex_m $COOKBOOKDIR/drexm_input.dat
# srun -n 1 gdb -batch -ex run -ex bt -ex quit ./drex_m $COOKBOOK/drexm_input.dat > gdb.log 2>&1

# move the outputs to pfs
mv $COOKBOOKDIR/out/* $OUTDIR
