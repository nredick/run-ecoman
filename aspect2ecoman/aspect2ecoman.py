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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

SCRIPT_START = time.time()


def log_section(title: str):
    """Print a clearly delimited section header."""
    logger.info("")
    logger.info("--- %s", title)


def log_kv(rows: list[tuple[str, str]]):
    """
    Print aligned key-value pairs.

    Pass (None, None) to insert a blank line between groups.
    """
    valid_labels = [r[0] for r in rows if r[0] is not None]
    col_w = max((len(l) for l in valid_labels), default=16)

    for label, value in rows:
        if label is None:
            logger.info("")
        else:
            logger.info("  %-*s : %s", col_w, label, value)


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(description="Convert ASPECT output to DREX input")
parser.add_argument("-f", "--file", required=True, help="Path to ASPECT .pvd solution file")
parser.add_argument(
    "-o", "--outdir", default=None,
    help="Output directory for generated files (default: script directory)",
)
parser.add_argument(
    "--keep-660", dest="crop_660", action="store_false", default=True,
    help="Disable cropping at 660 km depth (default: crop at 660 km)",
)
parser.add_argument(
    "--resolution", type=int, default=50,
    help="Target resampled model resolution in km (default: 50)",
)
parser.add_argument(
    "--timemax", type=float, default=3e6,
    help="Advection duration in years (default: 3e6)",
)

# ============================================================
# LOAD / PARSE ARGS
# ============================================================

args = parser.parse_args()

pvd_file   = os.path.abspath(args.file)
outdir     = args.outdir
resolution = args.resolution
TIMEMAX    = args.timemax

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = os.path.basename(os.path.dirname(pvd_file))

if outdir is None:
    outdir = os.path.dirname(__file__) or os.getcwd()
outdir = os.path.abspath(outdir)
os.makedirs(outdir, exist_ok=True)

log_section("CONFIGURATION")
log_kv([
    ("Model ID",           MODEL_ID),
    ("PVD file",           pvd_file),
    ("Output dir",         outdir),
    (None, None),
    ("Resolution",         f"{resolution} km"),
    ("Advection duration", f"{TIMEMAX:.2e} yr"),
    ("Crop at 660 km",     str(args.crop_660)),
    ("Timestamp",          datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
])

# ============================================================
# LOAD *.DAT TEMPLATES
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))


def _load_template(name: str) -> str:
    path = os.path.join(script_dir, name)
    with open(path, "r") as f:
        return f.read()


DREXM_TEMPLATE   = _load_template("DREXM_TEMPLATE.txt")
VIZTOMO_TEMPLATE = _load_template("VIZTOMO_TEMPLATE.txt")
STACK_TEMPLATE   = _load_template("STACK_TEMPLATE.txt")

logger.info("Loaded DREXM / VIZTOMO / STACK templates from %s", script_dir)

# ============================================================
# CONSTANTS & HELPER FUNCTIONS
# ============================================================

# Single seconds-per-year constant (Gregorian calendar: 365.2425 days)
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

log_kv(
    [("SECONDS_PER_YEAR", SECONDS_PER_YEAR),]
)

def year2sec(years: float) -> float:
    """Convert years to seconds."""
    return years * SECONDS_PER_YEAR

def sec2year(seconds: float) -> float:
    """Convert seconds to years."""
    return seconds / SECONDS_PER_YEAR


def fmt_d(val: float) -> str:
    """Format a float in Fortran-style double precision (d-notation)."""
    return f"{val:.6e}".replace("e", "d")


def generate_grid_blocks(bounds, nx, ny, nz, n_lon, n_rad, n_colat):
    """Return text block for Eulerian and Lagrangian grids."""
    lon_min, lon_max, colat_min, colat_max, r_min, r_max = bounds

    Rmean   = 0.5 * (r_min + r_max)
    dlam    = (lon_max - lon_min) * math.pi / 180.0
    dth     = (colat_max - colat_min) * math.pi / 180.0
    th_mean = 0.5 * (colat_min + colat_max) * math.pi / 180.0

    lon_len   = max(0.0, dlam) * Rmean * max(1e-9, math.sin(th_mean))
    colat_len = max(0.0, dth) * Rmean
    rad_len   = max(0.0, r_max - r_min)

    eps    = 100.0
    mx1stp = max(lon_len   / max(1, n_lon)   if lon_len   > 0 else 1.0, eps)
    mx2stp = max(rad_len   / max(1, n_rad)   if rad_len   > 0 else 1.0, eps)
    mx3stp = max(colat_len / max(1, n_colat) if colat_len > 0 else 1.0, eps)

    return f"""# Axis 1 (X-cart or Long)
    {fmt_d(lon_min)} # x1min: (X,Phi)min
    {fmt_d(lon_max)} # x1max: (X,Phi)max
      {nx} # nx1: number of grid nodes
      0 # x1periodic: periodic boundary (no = 0, yes = else)

# Axis 2 (Y-cart or Radial)
    {fmt_d(r_min)} # x2min: (Y,R)min
    {fmt_d(r_max)} # x2max: (Y,R)max
      {nz} # nx2: number of grid nodes
      0 # x2periodic: periodic boundary (no = 0, yes = else)

# Axis 3 (Z-cart or Colat)
    {fmt_d(colat_min)} # x3min: (Z,Colat)min
    {fmt_d(colat_max)} # x3max: (Z,Colat)max
      {ny} # nx3: number of grid nodes
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


def write_drexm_with_grid(output_path: str, grid_block: str):
    body = DREXM_TEMPLATE.replace("{GRID_BLOCK}", grid_block.rstrip() + "\n")
    body = body.replace("{MODEL_ID}", MODEL_ID)
    body = body.replace("{TIMEMAX}", str(year2sec(TIMEMAX)))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    logger.info("  Written: %s", output_path)


def write_stack_input(output_path: str, bounds):
    lon_min, lon_max, colat_min, colat_max, r_min, r_max = bounds
    body = (
        STACK_TEMPLATE
        .replace("{LON_MIN}",   f"  {fmt_d(lon_min)}")
        .replace("{LON_MAX}",   f"  {fmt_d(lon_max)}")
        .replace("{R_MIN}",     f"  {fmt_d(r_min)}")
        .replace("{R_MAX}",     f"  {fmt_d(r_max)}")
        .replace("{COLAT_MIN}", f"   {fmt_d(colat_min)}")
        .replace("{COLAT_MAX}", f"   {fmt_d(colat_max)}")
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    logger.info("  Written: %s", output_path)


def write_viztomo_input(output_path: str, bounds, nx, ny, nz):
    lon_min, lon_max, colat_min, colat_max, r_min, r_max = bounds
    body = (
        VIZTOMO_TEMPLATE
        .replace("{LON_MIN}",   f"  {fmt_d(lon_min)}")
        .replace("{LON_MAX}",   f"  {fmt_d(lon_max)}")
        .replace("{R_MIN}",     f"  {fmt_d(r_min)}")
        .replace("{R_MAX}",     f"  {fmt_d(r_max)}")
        .replace("{COLAT_MIN}", f"  {fmt_d(colat_min)}")
        .replace("{COLAT_MAX}", f"  {fmt_d(colat_max)}")
        .replace("{NX1}",       f"      {nx}")
        .replace("{NX2}",       f"      {ny}")
        .replace("{NX3}",       f"      {nz}")
        .replace("{MODEL_ID}",  MODEL_ID)
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)
    logger.info("  Written: %s", output_path)


# ============================================================
# LOAD ASPECT *.PVD
# ============================================================

log_section("LOADING MESH")
logger.info("Reading mesh from: %s", pvd_file)

try:
    t0   = time.time()
    mesh = pv.read(pvd_file)
    logger.info("Mesh loaded in %.1f s", time.time() - t0)
except Exception:
    logger.exception("Failed to read mesh from %s", pvd_file)
    raise

if isinstance(mesh, pv.MultiBlock):
    mesh = mesh[0]
    logger.info("MultiBlock detected — using first block")

log_kv([
    ("Points",      f"{mesh.n_points:,}"),
    ("Cells",       f"{mesh.n_cells:,}"),
    ("Fields",      ", ".join(mesh.point_data.keys())),
    (None, None),
    ("X range (m)", f"[{mesh.bounds[0]:.2f}, {mesh.bounds[1]:.2f}]"),
    ("Y range (m)", f"[{mesh.bounds[2]:.2f}, {mesh.bounds[3]:.2f}]"),
    ("Z range (m)", f"[{mesh.bounds[4]:.0f}, {mesh.bounds[5]:.0f}]"),
])

log_section("FIELD STATISTICS (raw, pre-conversion)")

for name in mesh.point_data.keys():
    arr = mesh.point_data[name]
    if arr.ndim == 1:
        log_kv([
            (f"{name} min",  f"{arr.min():.4e}"),
            (f"{name} max",  f"{arr.max():.4e}"),
            (f"{name} mean", f"{arr.mean():.4e}"),
            (None, None),
        ])
    else:
        for col in range(arr.shape[1]):
            log_kv([
                (f"{name}[:,{col}] min",  f"{arr[:, col].min():.4e}"),
                (f"{name}[:,{col}] max",  f"{arr[:, col].max():.4e}"),
                (f"{name}[:,{col}] mean", f"{arr[:, col].mean():.4e}"),
                (None, None),
            ])

# ============================================================
# CONVERT MESH FROM CARTESIAN → SPHERICAL COORDS
# ============================================================

log_section("COORDINATE CONVERSION  (Cartesian → Spherical)")


@njit(parallel=True, cache=True, fastmath=True)
def cartesian_to_spherical(x, y, z):
    """
    Returns:
      r     [m]
      lon   [deg]
      colat [deg]
    """
    n     = len(x)
    r     = np.empty(n)
    lon   = np.empty(n)
    colat = np.empty(n)

    for i in prange(n):
        xi, yi, zi = x[i], y[i], z[i]
        ri = np.sqrt(xi*xi + yi*yi + zi*zi)
        r[i] = ri

        ph = np.arctan2(yi, xi)
        if ph < 0:
            ph += 2 * np.pi
        lon[i] = ph * 180 / np.pi

        colat[i] = np.arccos(zi / ri) * 180 / np.pi if ri > 1e-15 else 0.0

    return r, lon, colat


xyz     = mesh.points
x, y, z = (xyz[:, 0].astype(np.float32),
            xyz[:, 1].astype(np.float32),
            xyz[:, 2].astype(np.float32))

logger.info("Converting %s points to spherical coordinates ...", f"{len(x):,}")
t0 = time.time()
r, lon, colat = cartesian_to_spherical(x, y, z)
logger.info("Conversion complete in %.1f s", time.time() - t0)

sph_mesh        = mesh.copy()
sph_mesh.points = np.column_stack([lon, colat, r])

log_kv([
    ("Longitude (deg)",    f"[{sph_mesh.bounds[0]:.2f}, {sph_mesh.bounds[1]:.2f}]"),
    ("Colatitude (deg)", f"[{sph_mesh.bounds[2]:.2f}, {sph_mesh.bounds[3]:.2f}]"),
    ("Radius (m)",       f"[{sph_mesh.bounds[4]:.0f}, {sph_mesh.bounds[5]:.0f}]"),
])

# ============================================================
# CROP DOMAIN
# ============================================================

log_section("DOMAIN CROPPING")

_, _, _, _, r_min, r_max = sph_mesh.bounds

# colat_crop = 5.0  # degrees — trim boundary artefacts

# Hardcoded longitudinal window (specific to CSZ model domain; [0, 360] convention)
lon = [-132 + 360, -112 + 360]
lon_min = min(lon)
lon_max = max(lon)

# Hardcoded colatitudinal window (specific to CSZ model domain; [0, 360] convention)
colat = [34, 55]
colat_min = min(colat)
colat_max = max(colat)
 
if args.crop_660:
    r_min_660 = 6371e3 - 660e3
    if r_min < r_min_660:
        r_min = r_min_660
    depth_label = "surface → 660 km  (cropped)"
else:
    depth_label = f"surface → {(6371e3 - r_min)/1e3:.0f} km  (full depth)"
 
bounds = (lon_min, lon_max, colat_min, colat_max, r_min, r_max)
 
log_kv([
    ("Longitude (deg)",    f"[{lon_min:.1f}, {lon_max:.1f}]  (span {abs(lon_max - lon_min):.1f}°)"),
    ("Colatitude (deg)",   f"[{colat_min:.1f}, {colat_max:.1f}]  (span {colat_max - colat_min:.1f}°)"),
    ("Radius (m)",         f"[{r_min:.0f}, {r_max:.0f}]"),
    ("Depth range",        depth_label),
])

# ============================================================
# GRID NODE CALCULATION
# ============================================================

log_section("GRID NODE CALCULATION")


def calculate_nodes_from_spacing(bounds, target_spacing_km=resolution):
    """Calculate number of Eulerian grid nodes to achieve target physical spacing."""
    lon_min, lon_max, colat_min, colat_max, r_min, r_max = bounds

    target_spacing_m = target_spacing_km * 1000   # km → m
    Rmean            = 0.5 * (r_min + r_max)      # mean radius [m]

    d_colat_rad = abs(colat_max - colat_min) * math.pi / 180.0
    d_lon_rad   = abs(lon_max   - lon_min)   * math.pi / 180.0

    # Radial: linear distance
    rad_length       = abs(r_max - r_min)
    nz               = int(rad_length / target_spacing_m) + 1

    # Colatitude: arc length = Rmean * dθ
    colat_arc_length = Rmean * d_colat_rad
    ny               = int(colat_arc_length / target_spacing_m) + 1

    # Longitude: arc length = Rmean * sin(θ_mean) * dφ
    mean_theta_rad   = 0.5 * (colat_min + colat_max) * math.pi / 180.0
    lon_arc_length   = Rmean * math.sin(mean_theta_rad) * d_lon_rad
    nx               = int(lon_arc_length / target_spacing_m) + 1

    return nx, ny, nz, lon_arc_length, colat_arc_length, rad_length


nx_target, ny_target, nz_target, lon_arc_m, colat_arc_m, rad_m = \
    calculate_nodes_from_spacing(bounds, target_spacing_km=resolution)

Rmean_domain = 0.5 * (r_min + r_max)

log_kv([
    ("Target spacing",   f"{resolution} km"),
    ("Mean radius",      f"{Rmean_domain/1e3:.1f} km"),
    (None, None),
    ("Longitude arc",      f"{lon_arc_m/1e3:.1f} km"),
    ("Colatitude arc",   f"{colat_arc_m/1e3:.1f} km"),
    ("Radial extent",    f"{rad_m/1e3:.1f} km"),
    (None, None),
    ("nx (longitude)",     nx_target),
    ("ny (colatitude)",  ny_target),
    ("nz (radial)",      nz_target),
    ("Total nodes",      f"{nx_target * ny_target * nz_target:,}"),
])

# ============================================================
# RESAMPLE ONTO UNIFORM GRID
# ============================================================

log_section("RESAMPLING")

grid            = pv.ImageData()
grid.dimensions = (nx_target, ny_target, nz_target)
grid.origin     = (lon_min, colat_min, r_min)
grid.spacing    = (
    abs(lon_max   - lon_min)   / (nx_target - 1),
    abs(colat_max - colat_min) / (ny_target - 1),
    abs(r_max     - r_min)     / (nz_target - 1),
)

log_kv([
    ("dlon",   f"{grid.spacing[0]:.4f} °/node"),
    ("dcolat", f"{grid.spacing[1]:.4f} °/node"),
    ("dr",     f"{grid.spacing[2]/1e3:.2f} km/node"),
])

logger.info("Resampling %s-point mesh onto %d × %d × %d grid ...",
            f"{mesh.n_points:,}", nx_target, ny_target, nz_target)

try:
    t0        = time.time()
    resampled = grid.sample(sph_mesh)
    resample_time = time.time() - t0
    logger.info("Resampling complete in %.1f s", resample_time)
except Exception:
    logger.exception("Failed to resample grid")
    raise

nx_actual, ny_actual, nz_actual = resampled.dimensions

log_kv([
    ("nx (actual)", nx_actual),
    ("ny (actual)", ny_actual),
    ("nz (actual)", nz_actual),
])

# ============================================================
# LAGRANGIAN AGGREGATE COUNTS
# ============================================================

log_section("LAGRANGIAN AGGREGATES")

# ! if target_aggregates < 3, VIZTOMO and SKS-SPLIT may not be able to find enough aggregates
target_aggregates  = 3  # aggregates per Eulerian cell per axis

n_lon              = nx_actual * target_aggregates
n_rad              = nz_actual * target_aggregates
n_colat            = ny_actual * target_aggregates
n_total_aggregates = n_lon * n_colat * n_rad

log_kv([
    ("Per cell (per axis)", target_aggregates),
    ("n_lon",               n_lon),
    ("n_colat",             n_colat),
    ("n_rad",               n_rad),
    ("Total aggregates",    f"{n_total_aggregates:,}"),
])

# ============================================================
# GENERATE *.DAT CONFIG FILES
# ============================================================

log_section("WRITING CONFIG FILES")

try:
    grid_block = generate_grid_blocks(
        resampled.bounds,
        nx=nx_actual, ny=ny_actual, nz=nz_actual,
        n_lon=n_lon, n_rad=n_rad, n_colat=n_colat,
    )
    write_drexm_with_grid(os.path.join(outdir, "drexm_input.dat"), grid_block)
    write_stack_input(os.path.join(outdir, "stack_input.dat"), resampled.bounds)
    write_viztomo_input(
        os.path.join(outdir, "viztomo_input.dat"),
        resampled.bounds, nx_actual, ny_actual, nz_actual,
    )
except Exception:
    logger.exception("Failed to write config files")
    raise

# ============================================================
# REORDER VTK DATA → ECOMAN/DREX_M LAYOUT
# ============================================================

log_section("REORDERING FIELD DATA  (VTK → D-REX_M)")


@njit
def reorder_to_drex(arr, nx, ny, nz):
    """Remap flat VTK index (x fastest) to D-REX_M index (z/depth fastest)."""
    out = np.empty_like(arr)
    for k in range(nz):          # radial / depth (fastest in D-REX_M)
        for j in range(ny):      # colatitude
            for i in range(nx):  # longitude
                vtk_idx  = i + nx * (j + ny * k)
                drex_idx = k + nz * (i + nx * j)
                out[drex_idx] = arr[vtk_idx]
    return out


logger.info("Reordering T, P, velocity fields ...")

T = reorder_to_drex(resampled.point_data["T"], *resampled.dimensions)
P = reorder_to_drex(resampled.point_data["p"], *resampled.dimensions)

vel = resampled.point_data.get("velocity") / SECONDS_PER_YEAR

if vel is None:
    logger.error("Field 'velocity' not found in resampled point data")
    raise KeyError("velocity not found in resampled.point_data")

# ASPECT outputs velocity in m/yr; using SI units for DREX-M (m/s)
V1 = reorder_to_drex(vel[:, 0], nx_actual, ny_actual, nz_actual)
V2 = reorder_to_drex(vel[:, 1], nx_actual, ny_actual, nz_actual)
V3 = reorder_to_drex(vel[:, 2], nx_actual, ny_actual, nz_actual)
Fd = np.zeros(nx_actual * ny_actual * nz_actual, dtype=np.float32)

log_kv([
    ("T  (K)",   f"[{T.min():.1f}, {T.max():.1f}]"),
    ("P  (Pa)",  f"[{P.min():.3e}, {P.max():.3e}]"),
    ("V1 (m/s)", f"[{V1.min():.3e}, {V1.max():.3e}]"),
    ("V2 (m/s)", f"[{V2.min():.3e}, {V2.max():.3e}]"),
    ("V3 (m/s)", f"[{V3.min():.3e}, {V3.max():.3e}]"),
])

# ============================================================
# WRITE HDF5 OUTPUT
# ============================================================

log_section("WRITING HDF5 OUTPUT")

fname = os.path.join(outdir, "vtp0001.h5")
time_val = 0.0
# dt0 = time interval over which the fields are representative
# for steady state, set arbitrarily large an let CFL condition determine appropriate step (and therefore numcycles)
dt0 = year2sec(1e6) # 1,000,000 years

try:
    with h5py.File(fname, "w") as f:
        f.attrs.create("Time", data=[dt0, time_val])
        g       = f.create_group("Nodes")
        g["Tk"] = T
        g["P"]  = P
        g["V1"] = V1
        g["V2"] = V2
        g["V3"] = V3
        g["Fd"] = Fd
    logger.info("  Written: %s", fname)
except Exception:
    logger.exception("Failed to write HDF5 file %s", fname)
    raise

log_kv([
    ("dt0",      f"{dt0:.4e} s  (~{dt0/SECONDS_PER_YEAR/1e6:.2f} Myr)"),
    ("t0",       f"{time_val:.1f} s"),
    ("Datasets", "Tk, P, V1, V2, V3, Fd"),
    ("N nodes",  f"{len(T):,}"),
])

# ============================================================
# RUN SUMMARY  —  appendix-ready
# ============================================================

log_section("RUN SUMMARY")

depth_min_km = (6371e3 - r_max) / 1e3
depth_max_km = (6371e3 - r_min) / 1e3

log_kv([
    # Input
    ("Model ID",              MODEL_ID),
    ("Source file",           os.path.basename(pvd_file)),
    ("Input mesh points",     f"{mesh.n_points:,}"),
    (None, None),
    # Domain
    ("Longitude (°E)",          f"{lon_min:.1f} → {lon_max:.1f}  (span {abs(lon_max - lon_min):.1f}°)"),
    ("Colatitude (°)",        f"{colat_min:.1f} → {colat_max:.1f}  (span {colat_max - colat_min:.1f}°)"),
    ("Depth (km)",            f"{depth_min_km:.0f} → {depth_max_km:.0f}"),
    ("Colat edge trim",       f"{colat_crop}° each side"),
    ("Crop at 660 km",        str(args.crop_660)),
    (None, None),
    # Grid
    ("Target spacing",        f"{resolution} km"),
    ("Mean radius",           f"{Rmean_domain/1e3:.1f} km"),
    ("Longitude arc",           f"{lon_arc_m/1e3:.1f} km"),
    ("Colatitude arc",        f"{colat_arc_m/1e3:.1f} km"),
    ("Radial extent",         f"{rad_m/1e3:.1f} km"),
    ("nx / ny / nz (target)", f"{nx_target} / {ny_target} / {nz_target}"),
    ("nx / ny / nz (actual)", f"{nx_actual} / {ny_actual} / {nz_actual}"),
    ("Total Eulerian nodes",  f"{nx_actual * ny_actual * nz_actual:,}"),
    ("dlon",                  f"{grid.spacing[0]:.4f} °/node"),
    ("dcolat",                f"{grid.spacing[1]:.4f} °/node"),
    ("dr",                    f"{grid.spacing[2]/1e3:.2f} km/node"),
    (None, None),
    # Lagrangian
    ("Agg. per cell",         target_aggregates),
    ("n_lon / n_colat / n_rad", f"{n_lon} / {n_colat} / {n_rad}"),
    ("Total aggregates",      f"{n_total_aggregates:,}"),
    (None, None),
    # Advection
    ("Advection duration",    f"{TIMEMAX:.2e} yr  ({TIMEMAX/1e6:.1f} Myr)"),
    ("Advection duration (s)", f"{year2sec(TIMEMAX):.4e}"),
    ("Timestep dt0",          f"{dt0:.4e} s  (~{dt0/SECONDS_PER_YEAR/1e6:.2f} Myr)"),
    (None, None),
    # Outputs
    ("drexm_input.dat",       os.path.join(outdir, "drexm_input.dat")),
    ("stack_input.dat",       os.path.join(outdir, "stack_input.dat")),
    ("viztomo_input.dat",     os.path.join(outdir, "viztomo_input.dat")),
    ("vtp0001.h5",            fname),
    (None, None),
    ("Total run time",        f"{time.time() - SCRIPT_START:.1f} s"),
])