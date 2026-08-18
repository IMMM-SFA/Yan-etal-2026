"""
extract_pft_cft_fractions.py

Uses Compiled_LULC_Fractions_CONUS.csv as the land-cell mask.
For each of the 52,682 CONUS grid cells (with their cluster label),
extracts per-cell area fractions for:
  - Natural PFTs 0-14  (PCT_NAT_PFT within the natural vegetation landunit)
  - Crop CFTs  15-78   (PCT_CFT within the crop landunit)
Both expressed as fractions of the total grid cell area.

Output directory: ./pft_cft_lulc/
  - lulc_{YEAR}.csv   one per year 1980-2015  (36 files)
  - lulc_mean.csv     36-year temporal mean

Columns in every CSV:
  lat, lon, cluster,
  natveg_frac, crop_frac,
  pft_00 ... pft_14,
  cft_15 ... cft_78

Usage:
  python extract_pft_cft_fractions.py
"""

import xarray as xr
import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
input_nc = (
    '/global/cfs/cdirs/m2702/liliyao/inputdata/cesm_inputdata/lnd/clm2/'
    'surfdata_map/'
    'landuse.timeseries_0.125nldas2_SSP5-8.5_78_CMIP6_1980-2019_c231122.nc'
)
mask_csv   = 'Compiled_LULC_Fractions_CONUS.csv'   # put next to this script
output_dir = './pft_cft_lulc'

N_NAT_PFT = 15    # natpft indices 0-14
N_CFT     = 64    # file cft indices 0-63 = global CFT 15-78

ROUND_DECIMALS = 4   # rounding for lat/lon matching

# ---------------------------------------------------------------------------
# 1. Load mask (CONUS land cells + cluster labels)
# ---------------------------------------------------------------------------
print("Loading land-cell mask from %s ..." % mask_csv)
mask_df = pd.read_csv(mask_csv, usecols=['lat', 'lon', 'cluster'])
mask_df['lat_r'] = mask_df['lat'].round(ROUND_DECIMALS)
mask_df['lon_360'] = mask_df['lon'] % 360
mask_df['lon_r'] = mask_df['lon_360'].round(ROUND_DECIMALS)
n_cells = len(mask_df)
print("  %d CONUS land cells, clusters: %s" % (n_cells, sorted(mask_df['cluster'].unique())))

# ---------------------------------------------------------------------------
# 2. Open NetCDF (lazy)
# ---------------------------------------------------------------------------
print("Opening %s ..." % os.path.basename(input_nc))
ds = xr.open_dataset(input_nc, chunks={'time': 1})

# ---------------------------------------------------------------------------
# 3. Subset time 1980-2015
# ---------------------------------------------------------------------------
print("Subsetting time 1980-2015 ...")
try:
    ds = ds.sel(time=slice(1980, 2015))
except KeyError:
    ds = ds.sel(time=slice('1980', '2015'))

time_vals = ds['time'].values
n_years   = len(time_vals)
print("  %d time steps found" % n_years)

# ---------------------------------------------------------------------------
# 4. Build rounded lat/lon 2-D lookup arrays from the NetCDF
# ---------------------------------------------------------------------------
print("Building spatial lookup index ...")

if 'LATIXY' not in ds:
    raise RuntimeError("LATIXY not found - check variable names with ncdump -h")
if 'LONGXY' not in ds:
    raise RuntimeError("LONGXY not found")

lat2d = ds['LATIXY'].isel(time=0).values if 'time' in ds['LATIXY'].dims else ds['LATIXY'].values
lon2d = ds['LONGXY'].isel(time=0).values if 'time' in ds['LONGXY'].dims else ds['LONGXY'].values

lat_flat_nc = lat2d.flatten().round(ROUND_DECIMALS)
lon_flat_nc = lon2d.flatten().round(ROUND_DECIMALS)
n_nc        = lat_flat_nc.size

# Build a dict (lat_r, lon_r) -> flat index in the NC array
print("  Building (lat, lon) -> flat-index map ...")
coord_to_idx = {}
for idx in range(n_nc):
    key = (lat_flat_nc[idx], lon_flat_nc[idx])
    coord_to_idx[key] = idx

# Map each mask row to its flat NC index
print("  Matching mask cells to NC grid ...")
nc_indices = []
missing    = 0
for _, row in mask_df.iterrows():
    key = (row['lat_r'], row['lon_r'])
    if key in coord_to_idx:
        nc_indices.append(coord_to_idx[key])
    else:
        nc_indices.append(-1)
        missing += 1

nc_indices = np.array(nc_indices)
if missing > 0:
    print("  WARNING: %d mask cells had no matching NC grid cell!" % missing)
    # drop unmatched rows
    valid      = nc_indices >= 0
    mask_df    = mask_df[valid].reset_index(drop=True)
    nc_indices = nc_indices[valid]
    n_cells    = len(mask_df)
    print("  Proceeding with %d matched cells." % n_cells)

lat_out     = mask_df['lat'].values
lon_out     = mask_df['lon'].values
cluster_out = mask_df['cluster'].values

# ---------------------------------------------------------------------------
# 5. Pre-compute lazy landunit DataArrays
# ---------------------------------------------------------------------------
print("Setting up landunit fraction arrays ...")
pct_crop        = ds['PCT_CROP']
pct_lake        = ds['PCT_LAKE']
pct_urban_total = ds['PCT_URBAN'].sum(dim='numurbl')
pct_glacier     = ds['PCT_GLACIER'] if 'PCT_GLACIER' in ds else xr.zeros_like(pct_crop)

natveg_frac = (100.0 - pct_crop - pct_lake - pct_urban_total - pct_glacier) / 100.0
crop_frac   = pct_crop / 100.0

nat_pft_da  = ds['PCT_NAT_PFT']   # (time, natpft, ...)
cft_da      = ds['PCT_CFT']       # (time, cft, ...)

# Output column names
pft_cols  = ['pft_%02d' % i for i in range(N_NAT_PFT)]
cft_cols  = ['cft_%02d' % (j + 15) for j in range(N_CFT)]
all_cols  = ['lat', 'lon', 'cluster', 'natveg_frac', 'crop_frac'] + pft_cols + cft_cols

# ---------------------------------------------------------------------------
# 6. Helper: extract values for matched cells at one time step
# ---------------------------------------------------------------------------
def extract_timestep(t_idx):
    """
    Returns numpy array (n_cells, n_cols) for the matched CONUS cells only.
    nc_indices selects the right flat positions from the full NC grid.
    """
    nv_all = natveg_frac.isel(time=t_idx).values.flatten()   # (n_nc,)
    cf_all = crop_frac.isel(time=t_idx).values.flatten()

    nv = nv_all[nc_indices]   # (n_cells,)
    cf = cf_all[nc_indices]

    # PCT_NAT_PFT: (natpft, spatial_flat) -> select nc_indices -> (15, n_cells)
    nat_raw = nat_pft_da.isel(time=t_idx).values.reshape(N_NAT_PFT, n_nc)
    nat_blk = (nat_raw[:, nc_indices] / 100.0) * nv[np.newaxis, :]   # (15, n_cells)

    # PCT_CFT: (cft, spatial_flat) -> select nc_indices -> (64, n_cells)
    cft_raw = cft_da.isel(time=t_idx).values.reshape(N_CFT, n_nc)
    cft_blk = (cft_raw[:, nc_indices] / 100.0) * cf[np.newaxis, :]   # (64, n_cells)

    # Stack columns: (n_cells, 2+1+2+15+64) = (n_cells, 84)
    return np.column_stack([
        lat_out, lon_out, cluster_out,
        nv, cf,
        nat_blk.T,   # (n_cells, 15)
        cft_blk.T    # (n_cells, 64)
    ])

# ---------------------------------------------------------------------------
# 7. Loop years: write per-year CSV, accumulate for mean
# ---------------------------------------------------------------------------
os.makedirs(output_dir, exist_ok=True)

mean_accum = None
mean_count = 0   # counts years included in mean (1981-2015 only)

for t_idx, t_val in enumerate(time_vals):
    year = int(str(t_val)[:4]) if not isinstance(t_val, (int, np.integer)) else int(t_val)
    print("  Year %d (%d/%d) ..." % (year, t_idx + 1, n_years))

    arr    = extract_timestep(t_idx)
    df_yr  = pd.DataFrame(arr, columns=all_cols)
    # cluster should be int
    df_yr['cluster'] = df_yr['cluster'].astype(int)

    out_path = os.path.join(output_dir, 'lulc_%d.csv' % year)
    df_yr.to_csv(out_path, index=False, float_format='%.6f')

    # Accumulate mean for 1981-2015 only (skip 1980)
    if year >= 1981:
        data = arr[:, 3:].astype(np.float64)
        if mean_accum is None:
            mean_accum = data
        else:
            mean_accum += data
        mean_count += 1

# ---------------------------------------------------------------------------
# 8. Write mean CSV
# ---------------------------------------------------------------------------
mean_accum /= mean_count   # mean over 1981-2015 = 35 years
meta = np.column_stack([lat_out, lon_out, cluster_out])
mean_arr = np.column_stack([meta, mean_accum])
df_mean  = pd.DataFrame(mean_arr, columns=all_cols)
df_mean['cluster'] = df_mean['cluster'].astype(int)

mean_path = os.path.join(output_dir, 'lulc_mean.csv')
df_mean.to_csv(mean_path, index=False, float_format='%.6f')
print("Mean CSV -> %s  (averaged over %d years: 1981-2015)" % (mean_path, mean_count))

ds.close()
print("Done! %d annual CSVs + 1 mean CSV in %s/" % (n_years, output_dir))
print("Each file: %d rows (CONUS land cells) x %d columns" % (n_cells, len(all_cols)))