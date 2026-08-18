# 12_add_soil_texture.py
#
# Extracts PCT_SAND and PCT_CLAY from the CLM5 surface NetCDF file and
# appends four new columns to spatial_flash_drought_properties.csv:
#
#   PCT_SAND_L1     : percent sand, top soil layer only (layer 1, 0-2 cm)
#   PCT_CLAY_L1     : percent clay, top soil layer only
#   PCT_SAND_WTMEAN : percent sand, thickness-weighted mean across all 10 layers
#   PCT_CLAY_WTMEAN : percent clay, thickness-weighted mean across all 10 layers
#
# CLM5 soil layer geometry (CLM5 Tech Note Table 2.1.1):
#   Layer  Node depth (m)  Thickness dz (m)  Interface depth (m)
#     1        0.010            0.020              0.020
#     2        0.040            0.040              0.060
#     3        0.090            0.060              0.120
#     4        0.160            0.080              0.200
#     5        0.260            0.120              0.320
#     6        0.400            0.160              0.480
#     7        0.580            0.200              0.680
#     8        0.800            0.240              0.920
#     9        1.060            0.280              1.200
#    10        1.360            0.320              1.520   (total 1.52 m)
#
# Weighted mean = sum(PCT * dz) / sum(dz)
# An unweighted mean is wrong: layer thicknesses range from 0.02 m to 0.32 m.
#
# Coordinate notes:
#   - LATIXY / LONGXY are 2-D arrays (lsmlat x lsmlon), not 1-D vectors.
#   - LONGXY is in 0-360 deg E convention; the CSV uses +/-180, so we convert.
#   - Matching uses nearest-neighbour KD-tree with a 0.0625 deg tolerance.

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# -- Paths ------------------------------------------------------------------
NC_FILE = (
    "/global/cfs/cdirs/m2702/liliyao/inputdata/cesm_inputdata/lnd/clm2/"
    "surfdata_map/"
    "surfdata_0.125nldas2_SSP5-8.5_78pfts_CMIP6_1980_c231121_nlevurb5_PFTDATAMASK.nc"
)
CSV_FILE = (
    "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/"
    "flash_drought_spatial_pattern_analysis/"
    "spatial_flash_drought_properties.csv"
)

HALF_CELL = 0.0625   # half of 0.125 deg grid spacing -- match tolerance
NLEV      = 10       # number of soil layers in CLM5

# CLM5 layer thicknesses dz (m), layers 1-10, from CLM5 Tech Note Table 2.1.1
# Total column = 1.52 m; thicknesses grow from 0.02 m (layer 1) to 0.32 m (layer 10)
DZ = np.array([0.020, 0.040, 0.060, 0.080, 0.120,
               0.160, 0.200, 0.240, 0.280, 0.320])
assert len(DZ) == NLEV and np.isclose(DZ.sum(), 1.52), "Layer thickness mismatch!"

# -- 1. Load NetCDF ---------------------------------------------------------
print("Reading CLM5 surface file ...")
with nc.Dataset(NC_FILE, "r") as ds:

    # 2-D coordinate arrays (lsmlat, lsmlon)
    latixy = ds.variables["LATIXY"][:]   # shape (224, 464)
    longxy = ds.variables["LONGXY"][:]   # shape (224, 464), 0-360 deg E

    # Convert longitude to +/-180 deg
    longxy_180 = ((longxy + 180.0) % 360.0) - 180.0

    # Soil texture: (nlevsoi=10, lsmlat=224, lsmlon=464)
    pct_sand = ds.variables["PCT_SAND"][:].data.astype(np.float64)
    pct_clay = ds.variables["PCT_CLAY"][:].data.astype(np.float64)

# -- 2. Flatten the NC grid into a lookup table ----------------------------
lat_flat = latixy.ravel()        # (224*464,)
lon_flat = longxy_180.ravel()

sand_L1_flat = pct_sand[0].ravel()    # top layer only
clay_L1_flat = pct_clay[0].ravel()

# Thickness-weighted mean: sum(PCT * dz) / sum(dz)
# DZ shape (10,); pct_sand shape (10, 224, 464)
# Broadcast DZ along spatial axes -> (10, 1, 1)
dz_3d = DZ[:, np.newaxis, np.newaxis]
sand_wtmean_flat = (np.sum(pct_sand * dz_3d, axis=0) / DZ.sum()).ravel()
clay_wtmean_flat = (np.sum(pct_clay * dz_3d, axis=0) / DZ.sum()).ravel()

# Build KD-tree for fast nearest-neighbour matching
print("Building KD-tree over %d CLM5 grid cells ..." % len(lat_flat))
nc_coords = np.column_stack([lat_flat, lon_flat])   # (N, 2)
tree = cKDTree(nc_coords)

# -- 3. Load the flash-drought CSV -----------------------------------------
print("Reading CSV ...")
props = pd.read_csv(CSV_FILE)
props["lat"] = props["lat"].round(6)
props["lon"] = props["lon"].round(6)

csv_coords = props[["lat", "lon"]].values   # (M, 2)

# -- 4. Nearest-neighbour query --------------------------------------------
print("Matching %d CSV rows to CLM5 grid ..." % len(props))
dist, idx = tree.query(csv_coords, k=1, workers=-1)

# Flag any match farther than half a grid cell
bad = dist > HALF_CELL
if bad.any():
    print("  WARNING: %d rows exceed the %.4f deg distance threshold "
          "(max dist = %.4f deg). Those rows will get NaN."
          % (bad.sum(), HALF_CELL, dist.max()))
    idx[bad] = -1   # sentinel -- will produce NaN below

# -- 5. Assign extracted values --------------------------------------------
def safe_lookup(flat_arr, indices):
    """Return values from flat_arr at indices; return NaN where index == -1."""
    out = np.where(indices >= 0,
                   flat_arr[np.clip(indices, 0, None)],
                   np.nan)
    return out

props["PCT_SAND_L1"]     = safe_lookup(sand_L1_flat,     idx)
props["PCT_CLAY_L1"]     = safe_lookup(clay_L1_flat,     idx)
props["PCT_SAND_WTMEAN"] = safe_lookup(sand_wtmean_flat, idx)
props["PCT_CLAY_WTMEAN"] = safe_lookup(clay_wtmean_flat, idx)

# -- 6. Sanity check -------------------------------------------------------
print("\n[Sanity check -- new columns]")
for col in ["PCT_SAND_L1", "PCT_CLAY_L1", "PCT_SAND_WTMEAN", "PCT_CLAY_WTMEAN"]:
    s = props[col]
    print("  %-20s  NaN=%4d  min=%.2f  mean=%.2f  max=%.2f"
          % (col, s.isna().sum(), s.min(), s.mean(), s.max()))

# Sand + clay should never exceed ~100% (silt makes up the rest)
sc_sum = props["PCT_SAND_L1"] + props["PCT_CLAY_L1"]
if (sc_sum > 105).any():
    print("  WARNING: %d rows have sand+clay (L1) > 105%%" % (sc_sum > 105).sum())
else:
    print("  Sand+Clay L1 sum check: OK (all <= 105%%)")

# Cross-check: weighted mean should differ from L1 if there is vertical variation
diff_sand = (props["PCT_SAND_WTMEAN"] - props["PCT_SAND_L1"]).abs().mean()
diff_clay = (props["PCT_CLAY_WTMEAN"] - props["PCT_CLAY_L1"]).abs().mean()
print("  Mean |WTMEAN - L1|: sand=%.2f%%  clay=%.2f%%  "
      "(non-zero confirms layered variation)" % (diff_sand, diff_clay))

# -- 7. Save ---------------------------------------------------------------
props.to_csv(CSV_FILE, index=False)
print("\nDone. Columns added: PCT_SAND_L1, PCT_CLAY_L1, "
      "PCT_SAND_WTMEAN, PCT_CLAY_WTMEAN")
print(props[["lat", "lon",
             "PCT_SAND_L1", "PCT_CLAY_L1",
             "PCT_SAND_WTMEAN", "PCT_CLAY_WTMEAN"]].head(10).to_string(index=False))