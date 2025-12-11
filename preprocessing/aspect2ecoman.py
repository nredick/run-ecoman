#!/usr/bin/env python3
import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pyvista as pv
from numba import njit, prange

# Static DREX template with placeholder for the Eulerian/Lagrangian grid block
DREXM_TEMPLATE = """# MPI proc distribution along axis
  5  # nproc1
  5  # nproc2
  5  # nproc3

# A) INPUT AND OUTPUT DIRECTORIES/FILES

# input_dir: path to input directory !!! Remember to put slash at the end
/quobyte/billengrp/nredick/csz-models/{MODEL_ID}/ecoman/drexm/

# output_dir: path to output directory !!! Remember to put slash at the end
/quobyte/billengrp/nredick/csz-models/{MODEL_ID}/ecoman/drexm/

     1 # Tinit
     1 # Tstp
     1 # Tend
 1000 # OutputStep
  3.155760e+13 # timemax: max timespan in sec (or adimensional time) for steady-state conditions 1 Ma

# B) DEFINE THE COMPUTATIONAL DOMAIN

     3 #  dimensions (2 = 2D; 3 = 3D)

     2 #  cartspher (coordinate system: 1 = cartesian; 2 = polar/spherical)

     1 #  basicstag (position of velocity nodes: 1 = basic nodes; 2 = shifted nodes)

# Eulerian Grid
{GRID_BLOCK}
# C) LPO PARAMETERS

        3 # size3: cubic root of total number of grains = size3^3

# Upper mantle: Ol + Ens

     7d1 # Xol(1) : fraction of olivine (%)
 5961d3 # minx2(1) : Min vertical distribution
 6371d3 # maxx2(1) : Max vertical distribution
  3.5d0 # stressexp(1)
    125d0 # Mob(1)
  0.3d0 # chi(1)
     5d0 # lambda(1)
 1.00d0 # fractdislrock(1) : fraction of def. accommodated by anis. phases
     1d0 # tau(1,1) [100](010) Olivine
     2d0 # tau(1,2) [100](001) Olivine
     3d0 # tau(1,3) [001](010) Olivine
    7d60 # tau(1,4) [001](100) Olivine
     1d0 # tau(1,5) [001](100) Enstatite
        1 # single_crystal_elastic_db(1,1) : Olivine single crystal elastic tensor as in elastic_database.f90
        2 # single_crystal_elastic_db(1,2) : Enstatite single crystal elastic tensor as in elastic_database.f90

# Upper transition zone: Wd + Grt

     7d1 # Xol(2) : fraction of wadsleyite (%)
 5851d3 # minx2(2) : Min vertical distribution
 5961d3 # maxx2(2) : Max vertical distribution
  3.5d0 # stressexp(2)
    125d0 # Mob(2)
  0.3d0 # chi(2)
     5d0 # lambda(2)
 1.00d0 # fractdislrock(2) : fraction of def. accommodated by anis. phases
     5d0 # tau(2,1) [100](001)   Wadsleyite
     1d0 # tau(2,2) [100](010)   Wadsleyite
     5d0 # tau(2,3) [100](011)   Wadsleyite
     5d0 # tau(2,4) [100](021)   Wadsleyite
     5d0 # tau(2,5) [111](10-1)  Wadsleyite
     5d0 # tau(2,6) [11_1](101)  Wadsleyite
     5d0 # tau(2,7) [1_11](10-1) Wadsleyite
     5d0 # tau(2,8) [1_1_1](101) Wadsleyite
     5d0 # tau(2,9) [001](010)   Wadsleyite
        4 # single_crystal_elastic_db(2,1) : Wadsleyite single crystal elastic tensor as in elastic_database.f90
        6 # single_crystal_elastic_db(2,2) : Garnet single crystal elastic tensor as in elastic_database.f90

# Lower transition zone: Rw + Grt

     7d1 # Xol(3) : fraction of ringwoodite (%)
 5711d3 # minx2(3) : Min vertical distribution
 5851d3 # maxx2(3) : Max vertical distribution
        5 # single_crystal_elastic_db(3,1) : Wadsleyite single crystal elastic tensor as in elastic_database.f90
        6 # single_crystal_elastic_db(3,2) : Garnet single crystal elastic tensor as in elastic_database.f90

# Lower mantle: Brd + MgO

     8d1 # Xol(4) : fraction of bridgmanite (%)
 3500d3 # minx2(4) : Min vertical distribution
 5711d3 # maxx2(4) : Max vertical distribution
  3.0d0 # stressexp(4)
    125d0 # Mob(4)
  0.3d0 # chi(4)
     5d0 # lambda(4)
 1.00d0 # fractdislrock(4) : fraction of def. accommodated by anis. phases
     5d0 # tau(4,1) [100](010)  Bridgmanite
     5d0 # tau(4,2) [100](001)  Bridgmanite
     5d0 # tau(4,3) [010](100)  Bridgmanite
     5d0 # tau(4,4) [010](001)  Bridgmanite
     1d0 # tau(4,5) [001](100)  Bridgmanite
     5d0 # tau(4,6) [001](010)  Bridgmanite
     5d0 # tau(4,7) [001](110)  Bridgmanite
     5d0 # tau(4,8) [001](-110) Bridgmanite
     5d0 # tau(4,9) [110](001)  Bridgmanite
     5d0 # tau(4,10)[-110](001) Bridgmanite
     5d0 # tau(4,11)[110](-110) Bridgmanite
     5d0 # tau(4,12)[-110](110) Bridgmanite
        8 # single_crystal_elastic_db(4,1) : Bridgmanite single crystal elastic tensor as in elastic_database.f90
      10 # single_crystal_elastic_db(4,2) : MgO single crystal elastic tensor as in elastic_database.f90

# Lower mantle: Brd + MgO

    8d1 # Xol(5) : fraction of PPv (%)
 3.0d0 # stressexp(5)
  125d0 # Mob(5)
 0.3d0 # chi(5)
    5d0 # lambda(5)
1.00d0 # fractdislrock(5) : fraction of def. accommodated by anis. phases
    5d0 # tau(5,1) [100](010)  PPv
    1d0 # tau(5,2) [100](001)  PPv
    5d0 # tau(5,3) [010](100)  PPv
    5d0 # tau(5,4) [010](001)  PPv
    5d0 # tau(5,5) [001](100)  PPv
    5d0 # tau(5,6) [001](010)  PPv
    5d0 # tau(5,7) [001](110)  PPv
    5d0 # tau(5,8) [001](-110) PPv
    5d0 # tau(5,9)[110](1-10) PPv
    5d0 # tau(5,10)[-110](110) PPv
     11 # single_crystal_elastic_db(5,1) : PPv single crystal elastic tensor as in elastic_database.f90
     10 # single_crystal_elastic_db(5,2) : MgO single crystal elastic tensor as in elastic_database.f90

# D) SET PRE-EXISTING FABRIC

        0 # fossilfabric (0: no pre-existing fabric; 1,2 = pre-existing fabric from pre-computed LPO file)
     0d0 # mx1minfab: (mX,mPhi)min where to set pre-existing fabric
     0d0 # mx1maxfab: (mX,mPhi)max where to set pre-existing fabric
     0d0 # mx2minfab: (mY,mR)min where to set pre-existing fabric
     0d0 # mx2maxfab: (mY,mR)max where to set pre-existing fabric
     0d0 # mx3minfab: (mZ,mColat)min where to set pre-existing fabric
     0d0 # mx3maxfab: (mZ,mColat)max where to set pre-existing fabric

# E) SET OPERATING MODES

        0 # fsemod ( 0 = compute FSE + LPO ; 1 = compute only FSE)
        1 # uppermantlemod ( 0 = LPO for the whole mantle ; 1 = LPO only for the upper mantle)
        1 # fractdislmod ( 0 = 100% disl. creep; 1 = combined diff./disl. creep)
        2 # fabrictransformmod ( 0 = no phase transformation; 1 = retain LPO after phase transformation ; 2 = reset LPO --> isotropic)
        2 # ptmod (0 = room P-T ; 1 = scale elastic properties f(P,T) but phase transitions at minx2/maxx2 depths; 2 = scale elastic properties f(P,T) and phase transitions at density crossovers))
        3 # eosmod (1 = Dunite; 2 = Hartzburgite; 3 = Pyrolite; 4 = Basalt ; 5 = Pyroxenite)
100.0d0 # fractvoigt (fraction of Voigt average, from 0 % to 100 %. The rest is Reuss average)

##############################################################################
"""

# Static STACK template with placeholders for spatial bounds
STACK_TEMPLATE = """ 10    !! nsx1 : number of seismic stations equally spaced along axis 1 direction
 10    !! nsx3 : number of seismic stations equally spaced along axis 3 direction

 0    !! depthaxis = define whether depth axis is positive downward (0) or upward (1)

{LON_MIN} !! i1first : (X,Phi)min (m,deg)
{LON_MAX} !! i1last  : (X,Phi)max (m,deg)
{R_MIN} !! i2first : (Y,R)min (m)
{R_MAX} !! i2last  : (Y,R)max (m)
{COLAT_MIN} !! i3first : (Z,Theta)min (m,deg)
{COLAT_MAX} !! i3last  : (Z,Theta)max (m,deg)

 100.0d3 !! maxdist : maximum horizontal distance for interpolation of markers to the given vertical profile (m)

 25.0d3 !! minlayer: minimum layer thickness (m)

  1d-3  !! Xscale : scaling factor to convert depths of the geodynamic model into km
"""

# Static VIZTOMO template with placeholders for Eulerian grid bounds
VIZTOMO_TEMPLATE = """# A) INPUT AND OUTPUT DIRECTORIES/FILES

# cijkl_dir: path to directory where to read Cijkl* files !!! remember to put slash at the end

# output_dir: path to directory where to save output files !!! remember to put slash at the end

   1  # Tinit: initial number of the source Cijkl*.h5 files to be processed
   1  # Tstep: increment number of the source Cijkl*.h5 files to be processed
   1  # Tend : final number of the source Cijkl*.h5 files to be processed

# B) VISUALIZE PROPERTIES OF LAGRANGIAN AGGREGATES

0  # Lagrangian

0.0  # ln_fse_min:: minimum threshold of ln(fse_max/fse_min) to visualize the following properties

0  # uppermantlemod (when active displays only upper mantle aggregates with ln_fse >= ln_fse_min)
0  # rocktypemod
0  # fse3Dmod (when active, allows for plotting the 3D FSE)
0  # fseminmod
0  # fsemaxmod
0  # TIaxismod
0  # vpmaxmod
0  # dvsmaxmod

# C) SPO: EXTRINSIC ELASTIC ANISOTROPY

0  # spomod (when active, the LPO elastic tensors are reset to 0). The Effective Medium is chosen as: 1 = STILWE (Backus, 1962, JGR); 2/3 = DEM (Mainprice, 1997, EPSL). Define SPO model parameters in spo_input.dat

# D) EULERIAN GRIDDING: INTERPOLATE THE AGGREGATES PROPERTIES TO A GRID

1  # Eulerian

# Axis 1 (X-cart or Long)
{LON_MIN}  # n1first: (X,Phi)min
{LON_MAX}  # n1last : (X,Phi)max
{NX1}  # nx11: number of nodes

# Axis 2 (Y-cart or Radial)
{R_MIN}  # n2first: (Y,R)min
{R_MAX}  # n2last : (Y,R)max
{NX2}  # nx21: number of nodes

# Axis 3 (Z-cart or Colat)
{COLAT_MIN}  # n3first: (Z,Colat)min
{COLAT_MAX}  # n3last : (Z,Colat)max
{NX3}  # nx31: number of nodes

      1  # vpvsmod:      Vp, Vs (1) isotropic or (2) along the cosx1,cosx2,cosx3 direction
      1  # dvpvsmod:     dVp,dVs with respect to (1) horizontally averaged Vp,Vs or (2) vertical reference profile
      0  # zoeppritzmod: Vp, Vs (1) isotropic or (2) along the cosx1,cosx2,cosx3 direction

    90d0 # cosx1: angle (in degrees) from axis 1 of incoming seismic wave when directionmod active
    00d0 # cosx2: angle (in degrees) from axis 2 of incoming seismic wave when directionmod active
    90d0 # cosx3: angle (in degrees) from axis 3 of incoming seismic wave when directionmod active

       1 # nx1ref: axis 1 node where to take the reference Vp, Vs profile to compute dVp and dVs when dvpvsmod == 2
       2 # nx3ref: axis 3 node where to take the reference Vp, Vs profile to compute dVp and dVs when dvpvsmod == 2

      1  # radialmod = Vsh^2/Vsv^2

      1  # azimod : azimuthal anisotropy
    0d0  # aziscalex1
    0d0  # aziscalex2
    0d0  # az1scalex3

      0 # reflectmod (when active, reflect the aggregates with respect to the indicated axis; e.g., when  = 1, then reflect with respect to X axis; if eulerianmod is active, then need to double the domain)
      0  # replicateZmod (when active, replicate the 2D tomographic model along the Z (Colat) direction)
      0  # specfem3Dmod (when active, print elastic tensors to txt file to be used as an input by Specfem3D)
      1  # psimod (when active, print elastic tensors to txt file to be used as an input by PSI)
"""

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# CLI
parser = argparse.ArgumentParser(description="Convert ASPECT output to DREX input")
parser.add_argument(
    "-f",
    "--file",
    required=True,
    help="Path to ASPECT .pvd solution file",
)
parser.add_argument(
    "-o",
    "--outdir",
    default=None,
    help="Optional output directory for generated drexm_input_*.dat (default: script directory)",
)
parser.add_argument(
    "--no-crop-660",
    dest="crop_660",
    action="store_false",
    default=True,
    help="Disable cropping at 660 km depth (default: crop at 660 km)",
)
args = parser.parse_args()

pvd_file = args.file
outdir = args.outdir

# get the full path of the pvd file
pvd_file = os.path.abspath(pvd_file)

# extract the model ID from the path (parent directory of pvd file)
MODEL_ID = os.path.basename(os.path.dirname(pvd_file))
logger.info(f"Using MODEL_ID: {MODEL_ID}")

# Set default output directory and ensure it exists
if outdir is None:
    outdir = os.path.dirname(__file__) or os.getcwd()
outdir = os.path.abspath(outdir)
os.makedirs(outdir, exist_ok=True)
logger.info(f"Output directory: {outdir}")


def fmt_d(val: float) -> str:
    """Format a float in Fortran-style double precision (d-notation)."""
    return f"{val:.6e}".replace("e", "d")

# n_long, n_rad, n_colat are target number of Lagrangian aggregates along each axis
def generate_grid_blocks(bounds, nx=50, ny=50, nz=50, n_long=20, n_rad=30, n_colat=50):
    """Return text block for Eulerian and Lagrangian grids using bounds and target counts."""
    lon_min, lon_max, colat_min, colat_max, r_min, r_max = bounds

    # Compute arc lengths (meters)
    Rmean = 0.5 * (r_min + r_max)
    dlam = (lon_max - lon_min) * math.pi / 180.0
    dth = (colat_max - colat_min) * math.pi / 180.0
    th_mean = 0.5 * (colat_min + colat_max) * math.pi / 180.0

    long_len = max(0.0, dlam) * Rmean * max(1e-9, math.sin(th_mean))
    colat_len = max(0.0, dth) * Rmean
    rad_len = max(0.0, r_max - r_min)

    mx1stp = long_len / max(1, n_long) if long_len > 0 else 1.0
    mx2stp = rad_len / max(1, n_rad) if rad_len > 0 else 1.0
    mx3stp = colat_len / max(1, n_colat) if colat_len > 0 else 1.0

    eps = 100.0
    mx1stp = max(mx1stp, eps)
    mx2stp = max(mx2stp, eps)
    mx3stp = max(mx3stp, eps)

    grid_text = f"""# Axis 1 (X-cart or Long)
    {fmt_d(lon_min)} # x1min: (X,Phi)min
    {fmt_d(lon_max)} # x1max: (X,Phi)max
      {nx} # nx1: number of grid nodes
      0 # x1periodic: periodic boundary (no = 0, yes = else)

# Axis 2 (Y-cart or Radial)
    {fmt_d(r_min)} # x2min: (Y,R)min
    {fmt_d(r_max)} # x2max: (Y,R)max
      {ny} # nx2: number of grid nodes
      0 # x2periodic: periodic boundary (no = 0, yes = else)

# Axis 3 (Z-cart or Colat)
    {fmt_d(colat_min)} # x3min: (Z,Colat)min
    {fmt_d(colat_max)} # x3max: (Z,Colat)max
      {nz} # nx3: number of grid nodes
      0 # x3periodic: periodic boundary (no = 0, yes = else)

# Lagrangian Grid

# Axis 1 (X-cart or Long)
    {fmt_d(lon_min)} # mx1min: (mX,mPhi)min
    {fmt_d(lon_max)} # mx1max: (mX,mPhi)max
    {mx1stp:.0f}d0 # mx1stp: spacing of aggregates (in meters)

# Axis 2 (Y-cart or Radial)
    {fmt_d(r_min)} # mx2min: (mY,mR)min
    {fmt_d(r_max)} # mx2max: (mY,mR)max
    {mx2stp:.0f}d0 # mx2stp: spacing of aggregates (in meters)

# Axis 3 (Z-cart or Colat)
    {fmt_d(colat_min)} # mx3min: (mZ,mColat)min
    {fmt_d(colat_max)} # mx3max: (mZ,mColat)max
    {mx3stp:.0f}d0 # mx3stp: spacing of aggregates (in meters)
"""

    return grid_text


def write_drexm_with_grid(output_path: str, grid_block: str):
    """Render the embedded template with the provided grid block and write it out."""
    body = DREXM_TEMPLATE.replace("{GRID_BLOCK}", grid_block.rstrip() + "\n")
    body = body.replace("{MODEL_ID}", MODEL_ID)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    logger.info(f"Wrote new grid-injected file: {output_path}")


def write_stack_input(output_path: str, bounds):
    """Render the embedded stack template with bounds and write it out."""
    lon_min, lon_max, colat_min, colat_max, r_min, r_max = bounds
    body = (
        STACK_TEMPLATE.replace("{LON_MIN}", f"  {fmt_d(lon_min)}")
        .replace("{LON_MAX}", f"  {fmt_d(lon_max)}")
        .replace("{R_MIN}", f"  {fmt_d(r_min)}")
        .replace("{R_MAX}", f"  {fmt_d(r_max)}")
        .replace("{COLAT_MIN}", f"   {fmt_d(colat_min)}")
        .replace("{COLAT_MAX}", f"   {fmt_d(colat_max)}")
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    logger.info(f"Wrote stack input file: {output_path}")


def write_viztomo_input(output_path: str, bounds, nx, ny, nz):
    """Render the embedded viztomo template with bounds and grid dimensions."""
    lon_min, lon_max, colat_min, colat_max, r_min, r_max = bounds
    body = (
        VIZTOMO_TEMPLATE.replace("{LON_MIN}", f"  {fmt_d(lon_min)}")
        .replace("{LON_MAX}", f"  {fmt_d(lon_max)}")
        .replace("{R_MIN}", f"  {fmt_d(r_min)}")
        .replace("{R_MAX}", f"  {fmt_d(r_max)}")
        .replace("{COLAT_MIN}", f"  {fmt_d(colat_min)}")
        .replace("{COLAT_MAX}", f"  {fmt_d(colat_max)}")
        .replace("{NX1}", f"      {nx}")
        .replace("{NX2}", f"      {ny}")
        .replace("{NX3}", f"      {nz}")
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    logger.info(f"Wrote viztomo input file: {output_path}")


# ============================================================
# LOAD ASPECT OUTPUT
# ============================================================

logger.info(f"Reading mesh from {pvd_file}")
try:
    t0 = time.time()
    mesh = pv.read(pvd_file)  # takes about a min for the csz models
    logger.info(f"Finished reading mesh ({time.time() - t0:.1f}s)")
except Exception:
    logger.exception(f"Failed to read mesh from {pvd_file}")
    raise

# If mesh is a MultiBlock (common from .pvd), get the first block
if isinstance(mesh, pv.MultiBlock):
    mesh = mesh[0]
    logger.info(f"MultiBlock detected — using first block (points={mesh.n_points})")

# Report mesh info
try:
    logger.info(f"Loaded mesh: points={mesh.n_points}, bounds={mesh.bounds}")
except Exception:
    logger.debug("Loaded mesh but couldn't read n_points/bounds for logging")

# ============================================================
# CARTESIAN → SPHERICAL
# ============================================================


@njit(parallel=True, cache=True, fastmath=True)
def cartesian_to_spherical(x, y, z):
    """
    Returns:
      r     [m]
      lon   [deg]
      colat [deg]
    """
    n = len(x)
    r = np.empty(n)
    lon = np.empty(n)
    colat = np.empty(n)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        zi = z[i]

        ri = np.sqrt(xi * xi + yi * yi + zi * zi)
        r[i] = ri

        ph = np.arctan2(yi, xi)
        if ph < 0:
            ph += 2 * np.pi
        lon[i] = ph * 180 / np.pi

        # colatitude in radians
        if ri > 1e-15:
            colat[i] = np.arccos(zi / ri) * 180 / np.pi
        else:
            colat[i] = 0.0

    return r, lon, colat


xyz = mesh.points
x, y, z = (
    xyz[:, 0].astype(np.float32),
    xyz[:, 1].astype(np.float32),
    xyz[:, 2].astype(np.float32),
)

logger.info(f"Converting {len(x)} points from cartesian to spherical coordinates")
r, lon, colat = cartesian_to_spherical(x, y, z)

# Replace mesh coordinates with (lon, colat, r)
sph_mesh = mesh.copy()
sph_mesh.points = np.column_stack([lon, colat, r])
# try:
#     sph_path = os.path.join(outdir, "mesh.vtu")
#     sph_mesh.save(sph_path, binary=True)
#     logger.info(f"Saved spherical mesh to {sph_path} (bounds={sph_mesh.bounds})")
# except Exception:
#     logger.exception(f"Failed to save spherical mesh to {sph_path}")
#     raise

# ============================================================
# BUILD GRID + RESAMPLE
# ============================================================

# xmin, xmax, ymin, ymax, zmin, zmax = sph_mesh.bounds
lon_min, lon_max, colat_min, colat_max, r_min, r_max = sph_mesh.bounds
nx = ny = nz = 150  # user-controlled

# Crop the domain: trim 3 degrees longitude and 1 degree colatitude on each side
lon_crop = 3.0  # degrees
colat_crop = 1.0  # degrees

# xmin += lon_crop
# xmax -= lon_crop
# ymin += colat_crop
# ymax -= colat_crop

lon_min += lon_crop
lon_max -= lon_crop
colat_min += colat_crop
colat_max -= colat_crop

# logger.info(
#     f"Cropped bounds: lon=[{xmin:.2f}, {xmax:.2f}], colat=[{ymin:.2f}, {ymax:.2f}], r=[{zmin:.0f}, {zmax:.0f}]"
# )

logger.info(
    f"Cropped bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], colat=[{colat_min:.2f}, {colat_max:.2f}], r=[{r_min:.0f}, {r_max:.0f}]"
)

# Crop the depth to 660 km if the mesh extends deeper
# Optionally crop at 660 km depth (only model upper mantle)
if args.crop_660:
    depth_660km = 6371e3 - 660e3  # radius at 660 km depth
    if r_min < depth_660km:
        logger.info(f"Cropping domain at 660 km depth (r={depth_660km:.0f} m)")
        r_min = depth_660km
else:
    logger.info("Modeling full depth (660 km cropping disabled)")

# ===========================================================
#  RESAMPLE SPHERICAL MESH ONTO UNIFORM GRID
# ============================================================

grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
# grid.origin = (xmin, ymin, zmin)
grid.origin = (lon_min, colat_min, r_min)

# grid.spacing = (
#     (xmax - xmin) / (nx - 1),
#     (ymax - ymin) / (ny - 1),
#     (zmax - zmin) / (nz - 1),
# )

grid.spacing = (
    (lon_max - lon_min) / (nx - 1),
    (colat_max - colat_min) / (ny - 1),
    (r_max - r_min) / (nz - 1),
)

try:
    t0 = time.time()
    resampled = grid.sample(sph_mesh)
    res_path = os.path.join(outdir, "resampled.vtk")
    # resampled.save(res_path, binary=True)
    # logger.info(f"Resampled grid saved to {res_path} ({time.time() - t0:.1f}s)")
    logger.info(f"Resampled dims: {resampled.dimensions}")
except Exception:
    logger.exception("Failed to resample or save resampled grid")
    raise

try:
    grid_block = generate_grid_blocks(
        resampled.bounds, nx=nx, ny=ny, nz=nz, n_long=20, n_rad=30, n_colat=50
    )

    drexm_output_path = os.path.join(outdir, f"drexm_input.dat")
    write_drexm_with_grid(drexm_output_path, grid_block)

    stack_output_path = os.path.join(outdir, f"stack_input.dat")
    write_stack_input(stack_output_path, resampled.bounds)

    viztomo_output_path = os.path.join(outdir, f"viztomo_input.dat")
    write_viztomo_input(viztomo_output_path, resampled.bounds, nx, ny, nz)
except Exception:
    logger.exception("Failed to generate or inject grid block into drexm_input.dat")
    raise  


# ============================================================
# REORDER VTK -> ECOMAN/DREX_M
# ============================================================


@njit
def reorder_to_drex(arr, nx, ny, nz):
    out = np.empty_like(arr)

    for k in range(nz):  # depth (fastest in DREX)
        for j in range(ny):  # colat
            for i in range(nx):  # lon
                vtk_idx = i + nx * (j + ny * k)
                drex_idx = k + nz * (i + nx * j)
                out[drex_idx] = arr[vtk_idx]

    return out


nx, ny, nz = resampled.dimensions
N = nx * ny * nz

s_per_yr = 60 * 60 * 24 * 365.25

T = reorder_to_drex(resampled.point_data["T"], *resampled.dimensions)
P = reorder_to_drex(resampled.point_data["p"], *resampled.dimensions)

vel = resampled.point_data.get("velocity")
if vel is None:
    logger.error("Resampled data has no 'velocity' point_data field")
    raise KeyError("velocity not found in resampled.point_data")

V1 = reorder_to_drex(vel[:, 0], nx, ny, nz) / s_per_yr
V2 = reorder_to_drex(vel[:, 1], nx, ny, nz) / s_per_yr
V3 = reorder_to_drex(vel[:, 2], nx, ny, nz) / s_per_yr

Fd = np.ones(N, dtype=np.float32)

logger.debug(
    f"Velocity ranges: V1=({V1.min()}, {V1.max()}), V2=({V2.min()}, {V2.max()}), V3=({V3.min()}, {V3.max()})"
)

# ============================================================
# WRITE HDF5 FILE FOR DREX
# ============================================================

fname = os.path.join(outdir, "vtp0001.h5")
# fname = os.path.join(outdir, fname)
# if os.path.isfile(fname):
#     logger.info(f"Removing existing file {fname}")
#     os.remove(fname)

time_val = 0.0
dt = 3.155760e12  # must be >0

try:
    with h5py.File(fname, "w") as f:
        f.attrs.create("Time", data=[dt, time_val])
        g = f.create_group("Nodes")
        g["Tk"] = T
        g["P"] = P
        g["V1"] = V1
        g["V2"] = V2
        g["V3"] = V3
        g["Fd"] = Fd
    logger.info(f"Wrote HDF5 file: {fname} (Time attr dt={dt}, time={time_val})")
except Exception:
    logger.exception(f"Failed to write HDF5 file {fname}")
    raise

logger.info("~fin~")
