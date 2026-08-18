# -*- coding: utf-8 -*-
"""
Z7_plot_seasonal_GPP_S1S3_S2S4.py
==================================
Mean seasonal (Apr-Oct) GPP time series for the SAME grid cells used in
Paper_Figure_5_LULC_GPP_v4.py panels (e)/(f):

  S1 (RCP4.5, hist LULC)      -- cells that are >=60% Grass under S1 AND
  S3 (SSP3-RCP4.5, SSP LULC)  -- >=60% Crop under S3 at the IDENTICAL cell
                                  (grass -> cropland conversion under SSP3)

  S2 (RCP8.5, hist LULC)      -- cells that are >=60% Grass under S2 AND
  S4 (SSP5-RCP8.5, SSP LULC)  -- >=60% Crop under S4 at the IDENTICAL cell
                                  (grass -> cropland conversion under SSP5)

For each of the 4 (scenario, land-cover) groups, this averages daily
gridcell-mean GPP (cooler + hotter ensemble members pooled, all simulation
years pooled) into an April-October day-of-year climatology, to show
whether crop cells drop to ~0 GPP after early-summer harvest while the
same cells under grass (S1/S2) stay productive.

*** ACTION NEEDED ***
GPP netCDF filenames are NOT consistent across scenario folders. Two
patterns are confirmed from user-provided examples; the rest are TODO
placeholders (folder entries set to None below) -- please confirm the
exact filenames (or paste `ls` output of each processed_gpp/gridcell_avg/
directory) and fill in GPP_FILENAME_PATTERNS.
"""
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus'

LULC_FILE = os.path.join(BASE_DIR, 'hist/pft_cft_lulc/lulc_mean.csv')
SSP3_LULC_FILES = [
    os.path.join(BASE_DIR, 'ssp3_rcp45_cooler_near', 'pft_cft_lulc', 'lulc_mean.csv'),
    os.path.join(BASE_DIR, 'ssp3_rcp45_hotter_near', 'pft_cft_lulc', 'lulc_mean.csv'),
]
SSP5_LULC_FILES = [
    os.path.join(BASE_DIR, 'ssp5_rcp85_cooler_near', 'pft_cft_lulc', 'lulc_mean.csv'),
    os.path.join(BASE_DIR, 'ssp5_rcp85_hotter_near', 'pft_cft_lulc', 'lulc_mean.csv'),
]

GPP_SUBDIR = 'processed_gpp/gridcell_avg'

# ------------------------------------------------------------------
# CONFIRMED for all 8 folders (user-provided C1 filenames):
# ------------------------------------------------------------------
GPP_FILENAME_PATTERNS = {
    'ssp3_rcp45_cooler_near': ('C{c}_ssp3_rcp45_cooler_near_Gridcell_True_Avg_GPP.nc', True),
    'ssp3_rcp45_hotter_near': ('C{c}_ssp3_rcp45_hotter_near_Gridcell_True_Avg_GPP.nc', True),
    'rcp45_cooler_near':      ('C{c}_nf_45_c_h1_dailyGPP_Gridcell_True_Avg.nc',         True),
    'rcp45_hotter_near':      ('C{c}_nf_45_h_h1_dailyGPP_Gridcell_True_Avg.nc',         True),
    'rcp85_cooler_near':      ('C{c}_nf_85_c_h1_dailyGPP_Gridcell_True_Avg.nc',         True),
    'rcp85_hotter_near':      ('C{c}_nf_85_h_h1_dailyGPP_Gridcell_True_Avg.nc',         True),
    'ssp5_rcp85_cooler_near': ('C{c}_ssp5_rcp85_cooler_near_Gridcell_True_Avg_GPP.nc',  True),
    'ssp5_rcp85_hotter_near': ('C{c}_ssp5_rcp85_hotter_near_Gridcell_True_Avg_GPP.nc',  True),
}

CLUSTERS = range(1, 8)

FOREST_PFTS = [f'pft_{i:02d}' for i in range(1, 9)]
GRASS_PFTS  = ['pft_12', 'pft_13', 'pft_14']
CROP_CFTS   = [f'cft_{i}' for i in range(15, 79)]
LULC_THRESHOLD = 0.6

MATCH_TOL_DEG = 0.01   # tolerance for matching nc cell lat/lon to LULC lat/lon

DOY_START = 91    # Apr 1 (non-leap)
DOY_END   = 304   # Oct 31 (non-leap)

C_GRASS = '#2D6A4F'
C_CROP  = '#C77800'
PALETTE_LULC = {'Grass': C_GRASS, 'Crop': C_CROP}
LINESTYLE_PAIR = {'S1': '-', 'S3': '-', 'S2': '--', 'S4': '--'}

SCENARIO_FOLDERS = {
    'RCP4.5':      ['rcp45_cooler_near', 'rcp45_hotter_near'],
    'RCP8.5':      ['rcp85_cooler_near', 'rcp85_hotter_near'],
    'SSP3-RCP4.5': ['ssp3_rcp45_cooler_near', 'ssp3_rcp45_hotter_near'],
    'SSP5-RCP8.5': ['ssp5_rcp85_cooler_near', 'ssp5_rcp85_hotter_near'],
}

# (leg_name, land-cover category, scenario key, cell-set key)
LULC_PAIR_SPECS = [
    ('S1', 'Grass', 'RCP4.5',      'S1S3'),
    ('S3', 'Crop',  'SSP3-RCP4.5', 'S1S3'),
    ('S2', 'Grass', 'RCP8.5',      'S2S4'),
    ('S4', 'Crop',  'SSP5-RCP8.5', 'S2S4'),
]

# ==============================================================================
# 1. LULC CLASSIFICATION + SAME-CELL CONVERSION TEST (mirrors panel e/f logic)
# ==============================================================================
print("1. Loading LULC and building same-cell grass->crop selections...")

def load_lulc_fracs(file_paths):
    per_file = []
    for fp in file_paths:
        d = pd.read_csv(fp)
        d['lat'] = d['lat'].round(4)
        d['lon'] = d['lon'].round(4)
        d['frac_forest'] = d[FOREST_PFTS].sum(axis=1)
        d['frac_grass']  = d[GRASS_PFTS].sum(axis=1)
        d['frac_crop']   = d[CROP_CFTS].sum(axis=1)
        per_file.append(d[['lat', 'lon', 'frac_forest', 'frac_grass', 'frac_crop']])

    merged = per_file[0]
    for d in per_file[1:]:
        merged = merged.merge(d, on=['lat', 'lon'], how='inner', suffixes=('', '_b'))
        for col in ['frac_forest', 'frac_grass', 'frac_crop']:
            merged[col] = merged[[col, col + '_b']].mean(axis=1)
            merged.drop(columns=[col + '_b'], inplace=True)

    merged['dominant'] = 'mixed'
    merged.loc[merged['frac_forest'] >= LULC_THRESHOLD, 'dominant'] = 'Forest'
    merged.loc[merged['frac_grass']  >= LULC_THRESHOLD, 'dominant'] = 'Grass'
    merged.loc[merged['frac_crop']   >= LULC_THRESHOLD, 'dominant'] = 'Crop'
    for idx, row in merged[merged['dominant'] == 'mixed'].iterrows():
        fracs = {'Forest': row['frac_forest'], 'Grass': row['frac_grass'], 'Crop': row['frac_crop']}
        best = max(fracs, key=fracs.get)
        if fracs[best] >= LULC_THRESHOLD:
            merged.at[idx, 'dominant'] = best

    merged['lon'] = merged['lon'].apply(lambda x: x + 360 if x < 0 else x)
    return merged[['lat', 'lon', 'dominant']]

lulc_hist = load_lulc_fracs([LULC_FILE])
lulc_ssp3 = load_lulc_fracs(SSP3_LULC_FILES)
lulc_ssp5 = load_lulc_fracs(SSP5_LULC_FILES)

cells_grass_hist = lulc_hist[lulc_hist['dominant'] == 'Grass'][['lat', 'lon']]
cells_crop_ssp3  = lulc_ssp3[lulc_ssp3['dominant'] == 'Crop'][['lat', 'lon']]
cells_crop_ssp5  = lulc_ssp5[lulc_ssp5['dominant'] == 'Crop'][['lat', 'lon']]

common_S1S3 = cells_grass_hist.merge(cells_crop_ssp3, on=['lat', 'lon'], how='inner')
common_S2S4 = cells_grass_hist.merge(cells_crop_ssp5, on=['lat', 'lon'], how='inner')
print(f"   S1(Grass)&S3(Crop) common cells: {len(common_S1S3):,}")
print(f"   S2(Grass)&S4(Crop) common cells: {len(common_S2S4):,}")

CELL_SETS = {'S1S3': common_S1S3, 'S2S4': common_S2S4}
CELL_TREES = {
    key: cKDTree(df[['lat', 'lon']].values) if len(df) else None
    for key, df in CELL_SETS.items()
}


def match_mask(nc_lat, nc_lon, cell_set_key):
    tree = CELL_TREES[cell_set_key]
    if tree is None:
        return np.zeros(len(nc_lat), dtype=bool)
    lon_conv = np.where(nc_lon < 0, nc_lon + 360, nc_lon)
    pts = np.column_stack([nc_lat, lon_conv])
    dist, _ = tree.query(pts)
    return dist <= MATCH_TOL_DEG


def get_gpp_var(ds):
    """Auto-detect the GPP data variable, since variable names differ
    between the two known netCDF schemas (Gridcell_Avg_GPP vs others)."""
    candidates = [v for v in ds.data_vars if 'gpp' in v.lower()]
    if not candidates:
        raise ValueError(f"No GPP-like variable found; available: {list(ds.data_vars)}")
    preferred = [v for v in candidates if 'avg' in v.lower() or 'gridcell' in v.lower()]
    return preferred[0] if preferred else candidates[0]

# ==============================================================================
# 2. LOAD GPP, BUILD APR-OCT DAY-OF-YEAR CLIMATOLOGY PER GROUP
# ==============================================================================
print("\n2. Loading GPP netCDFs and building April-Oct seasonal means...")

doy_axis = np.arange(DOY_START, DOY_END + 1)
group_series = {}   # (leg, cat) -> mean GPP array aligned with doy_axis
group_n_cells = {}

for leg, cat, scen_key, cell_key in LULC_PAIR_SPECS:
    vals_by_doy = {d: [] for d in doy_axis}
    n_cells_total = 0

    for folder in SCENARIO_FOLDERS[scen_key]:
        pattern, confirmed = GPP_FILENAME_PATTERNS.get(folder, (None, False))
        if pattern is None:
            print(f"   TODO: no filename pattern set for folder '{folder}' -- skipping")
            continue
        if not confirmed:
            print(f"   NOTE: filename pattern for '{folder}' is an unverified guess "
                  f"-- confirm before trusting this group's result")

        for c in CLUSTERS:
            fpath = os.path.join(BASE_DIR, folder, GPP_SUBDIR, pattern.format(c=c))
            if not os.path.exists(fpath):
                print(f"   WARNING: not found -- {fpath}")
                continue

            ds = xr.open_dataset(fpath)
            nc_lat = ds['lat'].values
            nc_lon = ds['lon'].values
            mask = match_mask(nc_lat, nc_lon, cell_key)
            if mask.sum() == 0:
                ds.close()
                continue

            gpp_var = get_gpp_var(ds)
            gpp = ds[gpp_var].values[:, mask]     # (time, n_matched_cells)
            doy = ds['time'].dt.dayofyear.values  # noleap calendar -> 1-365

            in_season = (doy >= DOY_START) & (doy <= DOY_END)
            gpp_season = gpp[in_season, :]
            doy_season = doy[in_season]

            cell_mean_ts = np.nanmean(gpp_season, axis=1)
            for d, v in zip(doy_season, cell_mean_ts):
                if not np.isnan(v):
                    vals_by_doy[d].append(v)

            n_cells_total += int(mask.sum())
            ds.close()

    mean_by_doy = np.array([
        np.mean(vals_by_doy[d]) if len(vals_by_doy[d]) > 0 else np.nan
        for d in doy_axis
    ])
    group_series[(leg, cat)] = mean_by_doy
    group_n_cells[(leg, cat)] = n_cells_total
    print(f"   {leg} ({scen_key}) {cat}: {n_cells_total} matched cell-instances "
          f"across clusters/ensemble members")

# ==============================================================================
# 3. PLOT
# ==============================================================================
print("\n3. Rendering figure...")

fig, ax = plt.subplots(figsize=(10, 6))

for leg, cat, scen_key, cell_key in LULC_PAIR_SPECS:
    ax.plot(doy_axis, group_series[(leg, cat)],
            color=PALETTE_LULC[cat],
            linestyle=LINESTYLE_PAIR[leg],
            linewidth=2.5,
            label=f'{leg}: {cat}')

month_starts = {'Apr': 91, 'May': 121, 'Jun': 152, 'Jul': 182, 'Aug': 213, 'Sep': 244, 'Oct': 274}
ax.set_xticks(list(month_starts.values()))
ax.set_xticklabels(list(month_starts.keys()))

ax.set_xlabel('Month', fontsize=13, weight='bold')
ax.set_ylabel('Mean Gridcell GPP (gC m-2 d-1)', fontsize=13, weight='bold')

# Floor the y-axis at the minimum value reached by the crop curves (S3/S4)
# rather than 0. Crop GPP never actually reaches 0 in Apr because these
# "Crop" cells are only >=60% crop -- the remainder is grass/shrub, so a
# residual signal persists. Anchoring the axis floor to that residual
# crop minimum (instead of 0) visually emphasizes how far grass (S1/S2)
# stays ABOVE that floor through Aug-Oct, i.e. crop's effectively shorter
# growing season, without implying crop GPP truly hits zero.
crop_min = np.nanmin(np.concatenate([
    group_series[('S3', 'Crop')],
    group_series[('S4', 'Crop')],
]))

ax.tick_params(labelsize=12)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10, framealpha=0.9, loc='best')

_, ymax = ax.get_ylim()
ax.set_ylim(crop_min, ymax)

plt.tight_layout()

out_file = os.path.join(BASE_DIR, 'ssp3_rcp45_cooler_near', 'Z7_S1S3_S2S4_seasonal_GPP.png')
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figure saved: {out_file}")