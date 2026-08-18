# -*- coding: utf-8 -*-
"""
Z8_plot_monthly_VPD_S1S3_S2S4.py
=================================
Fig. S11 companion to Fig. S10 (Z7): mean monthly VPD, restricted to the
SAME grassland-to-cropland conversion grid cells used in Fig. S10
(common_S1S3 / common_S2S4), under:

  S1/S3 pair -> RCP4.5 near-term climate forcing (rcp45_cooler_near +
                rcp45_hotter_near WRF VPD files)
  S2/S4 pair -> RCP8.5 near-term climate forcing (rcp85_cooler_near +
                rcp85_hotter_near WRF VPD files)

Since S1 and S3 (and S2 and S4) share the same atmospheric forcing and
differ only in land-cover, this reads VPD once per RCP pair rather than
separately for the LULC-only scenario folders (temperature is intentionally
NOT plotted per user request).
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

VPD_SUBDIR = 'wrf_daily_precip_vpd'
VPD_FILE_PATTERN = 'wrf_daily_vpd_precip_2021_2055_cluster_{c}.nc'
VPD_VAR = 'VPD'

CLUSTERS = range(1, 8)

FOREST_PFTS = [f'pft_{i:02d}' for i in range(1, 9)]
GRASS_PFTS  = ['pft_12', 'pft_13', 'pft_14']
CROP_CFTS   = [f'cft_{i}' for i in range(15, 79)]
LULC_THRESHOLD = 0.6

MATCH_TOL_DEG = 0.01

# RCP pair -> WRF forcing folders. S1/S3 share RCP4.5 forcing;
# S2/S4 share RCP8.5 forcing.
FORCING_FOLDERS = {
    'S1S3': ['rcp45_cooler_near', 'rcp45_hotter_near'],
    'S2S4': ['rcp85_cooler_near', 'rcp85_hotter_near'],
}
BAR_LABELS = {'S1S3': 'RCP4.5 near-term (S1/S3 cells)',
              'S2S4': 'RCP8.5 near-term (S2/S4 cells)'}
BAR_COLORS = {'S1S3': '#2E86AB', 'S2S4': '#E66101'}

# ==============================================================================
# 1. LULC CLASSIFICATION + SAME-CELL CONVERSION TEST (mirrors Fig 5e/f, S10)
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

    # keep lon in 0-360 convention (matches drought/GPP file convention)
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


def match_mask(nc_lat, nc_lon, pair_key):
    """nc_lon here is in -180/180 convention (WRF files); convert to
    0-360 to match the LULC-derived cell sets before nearest-neighbor lookup."""
    tree = CELL_TREES[pair_key]
    if tree is None:
        return np.zeros(len(nc_lat), dtype=bool)
    lon_conv = np.where(nc_lon < 0, nc_lon + 360, nc_lon)
    pts = np.column_stack([nc_lat, lon_conv])
    dist, _ = tree.query(pts)
    return dist <= MATCH_TOL_DEG

# ==============================================================================
# 2. LOAD VPD, BUILD MONTHLY MEANS PER RCP PAIR (over conversion cells only)
# ==============================================================================
print("\n2. Loading WRF daily VPD and building monthly means...")

months = np.arange(1, 13)
pair_monthly_vpd = {}   # pair_key -> array of 12 monthly means

for pair_key, folders in FORCING_FOLDERS.items():
    vals_by_month = {m: [] for m in months}

    for folder in folders:
        for c in CLUSTERS:
            fpath = os.path.join(BASE_DIR, folder, VPD_SUBDIR,
                                  VPD_FILE_PATTERN.format(c=c))
            if not os.path.exists(fpath):
                print(f"   WARNING: not found -- {fpath}")
                continue

            ds = xr.open_dataset(fpath)
            nc_lat = ds['lat'].values
            nc_lon = ds['lon'].values
            mask = match_mask(nc_lat, nc_lon, pair_key)
            if mask.sum() == 0:
                ds.close()
                continue

            vpd = ds[VPD_VAR].values[:, mask]     # (time, n_matched_cells)
            month = ds['time'].dt.month.values

            cell_mean_ts = np.nanmean(vpd, axis=1)
            for m, v in zip(month, cell_mean_ts):
                if not np.isnan(v):
                    vals_by_month[m].append(v)

            ds.close()

    monthly_mean = np.array([
        np.mean(vals_by_month[m]) if len(vals_by_month[m]) > 0 else np.nan
        for m in months
    ])
    pair_monthly_vpd[pair_key] = monthly_mean
    print(f"   {pair_key} ({BAR_LABELS[pair_key]}): done")

# ==============================================================================
# 3. PLOT
# ==============================================================================
print("\n3. Rendering figure...")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(months))
bw = 0.35

ax.bar(x - bw / 2, pair_monthly_vpd['S1S3'], width=bw,
       color=BAR_COLORS['S1S3'], edgecolor='white', linewidth=0.4,
       label=BAR_LABELS['S1S3'])
ax.bar(x + bw / 2, pair_monthly_vpd['S2S4'], width=bw,
       color=BAR_COLORS['S2S4'], edgecolor='white', linewidth=0.4,
       label=BAR_LABELS['S2S4'])

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(x)
ax.set_xticklabels(month_labels)

ax.set_xlabel('Month', fontsize=13, weight='bold')
ax.set_ylabel('Mean VPD (kPa)', fontsize=13, weight='bold')

ax.tick_params(labelsize=12)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10, framealpha=0.9, loc='upper left')

plt.tight_layout()

out_file = os.path.join(BASE_DIR, 'ssp3_rcp45_cooler_near', 'Z8_S1S3_S2S4_monthly_VPD.png')
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figure saved: {out_file}")