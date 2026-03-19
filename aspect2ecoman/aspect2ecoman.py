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

# ============================================================
# SET UP LOGGING
# ============================================================

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
    "--keep-660",
    dest="crop_660",
    action="store_false",
    default=True,
    help="Disable cropping at 660 km depth (default: crop at 660 km)",
)

parser.add_argument(
    "--resolution",
    type=int,
    default=10,
    help=f"Target resmapled model resolution in km (default: 10)",
)

# ============================================================
# LOAD/PARSE ARGS
# ============================================================

args = parser.parse_args()

pvd_file = args.file
outdir = args.outdir
resolution = args.resolution

# get the full path of the pvd file
pvd_file = os.path.abspath(pvd_file)

# ============================================================
# CONFIGURATION
# ============================================================

# extract the model ID from the path (parent directory of pvd file)
MODEL_ID = os.path.basename(os.path.dirname(pvd_file))
logger.info(f"Using MODEL_ID: {MODEL_ID}")

# Set default output directory and ensure it exists
if outdir is None:
    outdir = os.path.dirname(__file__) or os.getcwd()
outdir = os.path.abspath(outdir)
os.makedirs(outdir, exist_ok=True)
logger.info(f"Output directory: {outdir}")

# ============================================================
# LOAD *.DAT TEMPLATES
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

drexm_template_path = os.path.join(script_dir, "DREXM_TEMPLATE.txt")
# Static DREX template with placeholder for the Eulerian/Lagrangian grid block
with open(drexm_template_path, "r") as f:
    DREXM_TEMPLATE = f.read()

viztomo_template_path = os.path.join(script_dir, "VIZTOMO_TEMPLATE.txt")
# Static VIZTOMO template with placeholders for Eulerian grid bounds
with open(viztomo_template_path, "r") as f:
    VIZTOMO_TEMPLATE = f.read()

stack_template_path = os.path.join(script_dir, "STACK_TEMPLATE.txt")
# Static STACK template with placeholders for spatial bounds
with open(stack_template_path, "r") as f:
    STACK_TEMPLATE = f.read()

# ============================================================
# DEFINE HELPER FUNCTIONS FOR TEMPLATED OUTPUTS
# ============================================================


def fmt_d(val: float) -> str:
    """Format a float in Fortran-style double precision (d-notation)."""
    return f"{val:.6e}".replace("e", "d")


# n_long, n_rad, n_colat are target number of Lagrangian aggregates along each axis
def generate_grid_blocks(bounds, nx, ny, nz, n_long, n_rad, n_colat):
    """Return text block for Eulerian and Lagrangian grids using bounds and target counts."""
    azi_min, azi_max, colat_min, colat_max, r_min, r_max = bounds

    # Compute arc lengths (meters)
    Rmean = 0.5 * (r_min + r_max)
    dlam = (azi_max - azi_min) * math.pi / 180.0
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
    {fmt_d(azi_min)} # x1min: (X,Phi)min
    {fmt_d(azi_max)} # x1max: (X,Phi)max
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
    {fmt_d(azi_min)} # mx1min: (mX,mPhi)min
    {fmt_d(azi_max)} # mx1max: (mX,mPhi)max
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
    logger.info(f"Wrote D-REX_M input file: {output_path}")


def write_stack_input(output_path: str, bounds):
    """Render the embedded stack template with bounds and write it out."""
    azi_min, azi_max, colat_min, colat_max, r_min, r_max = bounds
    body = (
        STACK_TEMPLATE.replace("{AZI_MIN}", f"  {fmt_d(azi_min)}")
        .replace("{AZI_MAX}", f"  {fmt_d(azi_max)}")
        .replace("{R_MIN}", f"  {fmt_d(r_min)}")
        .replace("{R_MAX}", f"  {fmt_d(r_max)}")
        .replace("{COLAT_MIN}", f"   {fmt_d(colat_min)}")
        .replace("{COLAT_MAX}", f"   {fmt_d(colat_max)}")
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    logger.info(f"Wrote STACK input file: {output_path}")


def write_viztomo_input(output_path: str, bounds, nx, ny, nz):
    """Render the embedded viztomo template with bounds and grid dimensions."""
    azi_min, azi_max, colat_min, colat_max, r_min, r_max = bounds
    body = (
        VIZTOMO_TEMPLATE.replace("{AZI_MIN}", f"  {fmt_d(azi_min)}")
        .replace("{AZI_MAX}", f"  {fmt_d(azi_max)}")
        .replace("{R_MIN}", f"  {fmt_d(r_min)}")
        .replace("{R_MAX}", f"  {fmt_d(r_max)}")
        .replace("{COLAT_MIN}", f"  {fmt_d(colat_min)}")
        .replace("{COLAT_MAX}", f"  {fmt_d(colat_max)}")
        .replace("{NX1}", f"      {nx}")
        .replace("{NX2}", f"      {ny}")
        .replace("{NX3}", f"      {nz}")
        .replace("{MODEL_ID}", MODEL_ID)
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    logger.info(f"Wrote VIZTOMO input file: {output_path}")


# ============================================================
# LOAD ASPECT *.PVD
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
# CONVERT MESH FROM CARTESIAN → SPHERICAL COORDS
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

xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
logger.info(
    f"Original bounds in cartesian coordinates: lon=[{xmin:.2f}, {xmax:.2f}], lat=[{ymin:.2f}, {ymax:.2f}], depth=[{zmin:.0f}, {zmax:.0f}]"
)

logger.info(f"Converting {len(x):,} points from cartesian to spherical coordinates")
r, lon, colat = cartesian_to_spherical(x, y, z)

# Replace mesh coordinates with (lon, colat, r)
sph_mesh = mesh.copy()
sph_mesh.points = np.column_stack([lon, colat, r])

# ! below block of code can be used for testing
# try:
#     sph_path = os.path.join(outdir, "mesh.vtu")
#     sph_mesh.save(sph_path, binary=True)
#     logger.info(f"Saved spherical mesh to {sph_path} (bounds={sph_mesh.bounds})")
# except Exception:
#     logger.exception(f"Failed to save spherical mesh to {sph_path}")
#     raise

# ============================================================
# CROP DOMAIN AZIMUTH, COLATITUDE, RADIUS (@660, OPTIONAL)
# ============================================================

# xmin, xmax, ymin, ymax, zmin, zmax = sph_mesh.bounds
azi_min, azi_max, colat_min, colat_max, r_min, r_max = sph_mesh.bounds

logger.info(
    f"Original bounds in spherical coordinates: azimuth=[{azi_min:.2f}, {azi_max:.2f}], colatitude=[{colat_min:.2f}, {colat_max:.2f}], radius=[{r_min:.0f}, {r_max:.0f}]"
)

# Crop the domain: trim 3 degrees azimuth and 1 degree colatitude on each side
# azi_crop = 3.0  # degrees
colat_crop = 1.0  # degrees

# azi_min += azi_crop
# azi_max -= azi_crop

# set absolute longitudinal min/max
# this is specific to the model domain of menno's models, and its selection is based on where we expect to see interesting flow
# no relevant flow near the thick continental lithosphere
# add 360 because we are working in the range [0, 2π], not [-180°, 180°]
azi_max = -135 + 360 
azi_min = -112 + 360

# set colatitidue values
colat_min += colat_crop
colat_max -= colat_crop

colat_min = int(np.floor(colat_min))
colat_max = int(np.floor(colat_max))

logger.info(
    f"Cropped bounds: azimuth=[{azi_min:.2f}, {azi_max:.2f}], colatitude=[{colat_min:.2f}, {colat_max:.2f}], radius=[{r_min:.0f}, {r_max:.0f}]"
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

# ============================================================
# CALCULATE DOMAIN PARAMETERS TO ACHIEVE DESIRED RESOLUTION
# ============================================================

# define earth's radius in km
R = 6371

# 1. convert angular spans to radians
d_colat_rad = math.radians(abs(colat_max - colat_min))
d_azi_rad = math.radians(abs(azi_max - azi_min))

# 2. calculate ny
# ny = total arc length / target resolution
ny = int((R * d_colat_rad) // resolution)

# 3. calculate nx (horizontal depends on theta)
# use the max sin(theta) to ensure that there is no undersampling of the widest part
# In colatitude, 90 is the equator, so use max of colat 
max_theta_rad = math.radians(max(colat_min, colat_max))
widest_arc_km = R * math.sin(max_theta_rad) * d_azi_rad

# 4. calculate nx
nx = int(widest_arc_km // resolution)

# 5. calculate nz (Radial/Depth), radius is defined in m => need to convert
nz = int(abs(r_max - r_min) // int(resolution*10**3))

# log nx, ny, nz
logger.info(
    f"Target grid dimensions: nx={nx}, ny={ny}, nz={nz} samples per dimension to achieve target resolution of {resolution} km"
)

# ===========================================================
#  RESAMPLE SPHERICAL MESH ONTO UNIFORM GRID
# ===========================================================

grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
grid.origin = (azi_min, colat_min, r_min)

grid.spacing = (
    abs(azi_max - azi_min) / (nx - 1),
    abs(colat_max - colat_min) / (ny - 1),
    abs(r_max - r_min) / (nz - 1),
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

# ===========================================================
# COMPUTE IDEAL # OF LAGRANGIAN AGGREGATES BASED ON RESOLUTION
# ===========================================================

# a minimum of 3-5 aggregates per cell gets rid of the errors in SKS-SPLIT related to no convergence and empty nodes

target_aggregrates = 3  # number of target aggregates defined per grid cell

# n_long, n_rad, n_colat are target number of Lagrangian aggregates along each axis
n_long = nx * target_aggregrates
n_rad = nz * target_aggregrates
n_colat = ny * target_aggregrates

# log values
logger.info(
    f"Configuring Lagrangian aggregates per dimension: n_long={n_long}, n_rad={n_rad}, n_colat={n_colat}"
)

# ===========================================================
# GENERATE *.DAT OUTPUTS (THE CONFIG FILES FOR ECOMAN STEPS)
# ===========================================================

try:
    grid_block = generate_grid_blocks(
        resampled.bounds,
        nx=nx,
        ny=ny,
        nz=nz,
        n_long=n_long,
        n_rad=n_rad,
        n_colat=n_colat,
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
# REORDER VTK DATA -> ECOMAN/DREX_M
# ============================================================


@njit
def reorder_to_drex(arr, nx, ny, nz):
    out = np.empty_like(arr)

    for k in range(nz):  # depth (fastest in DREXM)
        for j in range(ny):  # colat
            for i in range(nx):  # lon
                vtk_idx = i + nx * (j + ny * k)
                drex_idx = k + nz * (i + nx * j)
                out[drex_idx] = arr[vtk_idx]

    return out


# get absolute nx, ny, nz
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
# WRITE HDF5 FILE (INPUT FOR DREXM)
# ============================================================

fname = os.path.join(outdir, "vtp0001.h5")
# fname = os.path.join(outdir, fname)
# if os.path.isfile(fname):
#     logger.info(f"Removing existing file {fname}")
#     os.remove(fname)

time_val = 0.0
dt = 3.155760e12  # must be >0; 3.155760e12 = 1 Ma

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
