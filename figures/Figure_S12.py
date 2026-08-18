# -*- coding: utf-8 -*-
"""
Figure_5_GrassForest_a.py
============================
Standalone 1x2 figure: CDF panels for grid cells that are Grass under the
historical/RCP LULC and convert to FOREST under the SSP LULC pathway.

Panel layout:
  (a) CDF: GPP decline rate,  S1 Grass vs S3 Forest / S2 Grass vs S4 Forest
  (b) CDF: Onset date,        S1 Grass vs S3 Forest / S2 Grass vs S4 Forest

Cell selection: a grid cell is included in the S1-vs-S3 pair only if it is
>=60% grass under S1 (RCP4.5, historical LULC) AND >=60% forest under S3
(SSP3-RCP4.5 LULC) -- i.e. the same cell converts from grass to forest
under the SSP3 land-use pathway. The S2-vs-S4 pair uses the analogous
>=60% grass (S2/hist) AND >=60% forest (S4/SSP5) test.

Also prints, per cluster/region (1-7), how many common cells fall in the
S1(Grass)&S3(Forest) and S2(Grass)&S4(Forest) sets.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# =============================================================================
# GLOBAL FONT SIZE CONSTANTS
# =============================================================================
FS_PANEL_LABEL = 14
FS_AXIS_LABEL  = 11
FS_TICK        = 10
FS_LEGEND      = 10

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus'

VALID_CELLS_CSV = os.path.join(BASE_DIR, 'hist', 'valid_conus_flash_drought_cells.csv')
LULC_FILE       = os.path.join(BASE_DIR, 'hist/pft_cft_lulc/lulc_mean.csv')

SSP3_LULC_FILES = [
    os.path.join(BASE_DIR, 'ssp3_rcp45_cooler_near', 'pft_cft_lulc', 'lulc_mean.csv'),
    os.path.join(BASE_DIR, 'ssp3_rcp45_hotter_near', 'pft_cft_lulc', 'lulc_mean.csv'),
]
SSP5_LULC_FILES = [
    os.path.join(BASE_DIR, 'ssp5_rcp85_cooler_near', 'pft_cft_lulc', 'lulc_mean.csv'),
    os.path.join(BASE_DIR, 'ssp5_rcp85_hotter_near', 'pft_cft_lulc', 'lulc_mean.csv'),
]

DROUGHT_FILE_PATTERN = 'drought_events_fixed_season_cluster_{c}_with_GPP_Zhang2025.csv'
CLUSTERS = range(1, 8)

OUT_FILE = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/ssp3_rcp45_cooler_near/Figure_GrassForest_CDF_1x2.png'

# =============================================================================
# REGION METADATA
# =============================================================================
REGION_NAMES = {1: 'NW', 2: 'SW', 3: 'NGP', 4: 'SGP', 5: 'MW', 6: 'SE', 7: 'NE'}

# =============================================================================
# SCENARIO METADATA
# =============================================================================
SCENARIO_FOLDERS = {
    'RCP4.5':      [os.path.join(BASE_DIR, 'rcp45_cooler_near'),
                     os.path.join(BASE_DIR, 'rcp45_hotter_near')],
    'RCP8.5':      [os.path.join(BASE_DIR, 'rcp85_cooler_near'),
                     os.path.join(BASE_DIR, 'rcp85_hotter_near')],
    'SSP3-RCP4.5': [os.path.join(BASE_DIR, 'ssp3_rcp45_cooler_near'),
                     os.path.join(BASE_DIR, 'ssp3_rcp45_hotter_near')],
    'SSP5-RCP8.5': [os.path.join(BASE_DIR, 'ssp5_rcp85_cooler_near'),
                     os.path.join(BASE_DIR, 'ssp5_rcp85_hotter_near')],
}

# Colours: Grass keeps the paper convention; Forest gets a new, distinct hue.
C_GRASS  = '#2D6A4F'   # deep emerald (unchanged)
C_FOREST = '#6B4226'   # deep brown (distinct from grass green)

PALETTE_LULC = {'Grass': C_GRASS, 'Forest': C_FOREST}

# Pair 1 (S1 grass / S3 forest) drawn solid; pair 2 (S2 grass / S4 forest) dashed
LULC_PAIR_LINESTYLE = {'S1': '-', 'S3': '-', 'S2': '--', 'S4': '--'}
LULC_LEG_NAMES = {
    ('S1', 'Grass'):  'S1: Grass',
    ('S3', 'Forest'): 'S3: Forest',
    ('S2', 'Grass'):  'S2: Grass',
    ('S4', 'Forest'): 'S4: Forest',
}

FOREST_PFTS = [f'pft_{i:02d}' for i in range(1, 9)]
GRASS_PFTS  = ['pft_12', 'pft_13', 'pft_14']
CROP_CFTS   = [f'cft_{i}' for i in range(15, 79)]
LULC_THRESHOLD      = 0.6
MAX_DURATION_PENTADS = 6

MONTH_JDAYS  = [91, 121, 152, 182, 213, 244, 274]
MONTH_LABELS = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']

# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_valid_cells():
    print('Loading valid cells...')
    vc = pd.read_csv(VALID_CELLS_CSV)
    vc['lon'] = np.where(vc['lon'] > 180, vc['lon'] - 360.0, vc['lon'])
    vc['lat'] = vc['lat'].round(4)
    vc['lon'] = vc['lon'].round(4)
    valid_set = set(zip(vc['lat'], vc['lon']))
    print(f'  Valid cells: {len(valid_set):,}')
    return valid_set


def load_lulc_fracs(file_paths):
    """Load LULC fraction csv(s) and return per-cell frac_grass/frac_forest/
    dominant/cluster. The cluster number is read directly from the 3rd
    column of the LULC file (same file, same grid -- no separate join)."""
    per_file = []
    for fp in file_paths:
        d = pd.read_csv(fp)
        cluster_col = d.columns[2]   # 3rd column = cluster number
        d = d.rename(columns={cluster_col: 'cluster'})
        d['lat'] = d['lat'].round(4)
        d['lon'] = d['lon'].round(4)
        d['frac_forest'] = d[FOREST_PFTS].sum(axis=1)
        d['frac_grass']  = d[GRASS_PFTS].sum(axis=1)
        d['frac_crop']   = d[CROP_CFTS].sum(axis=1)
        per_file.append(d[['lat', 'lon', 'cluster', 'frac_forest', 'frac_grass', 'frac_crop']])

    merged = per_file[0]
    for d in per_file[1:]:
        merged = merged.merge(d.drop(columns=['cluster']), on=['lat', 'lon'],
                               how='inner', suffixes=('', '_b'))
        for col in ['frac_forest', 'frac_grass', 'frac_crop']:
            merged[col] = merged[[col, col + '_b']].mean(axis=1)
            merged.drop(columns=[col + '_b'], inplace=True)

    merged['dominant'] = 'mixed'
    merged.loc[merged['frac_forest'] >= LULC_THRESHOLD, 'dominant'] = 'Forest'
    merged.loc[merged['frac_grass']  >= LULC_THRESHOLD, 'dominant'] = 'Grass'
    merged.loc[merged['frac_crop']   >= LULC_THRESHOLD, 'dominant'] = 'Crop'
    for idx, row in merged[merged['dominant'] == 'mixed'].iterrows():
        fracs = {'Forest': row['frac_forest'],
                 'Grass':  row['frac_grass'],
                 'Crop':   row['frac_crop']}
        best = max(fracs, key=fracs.get)
        if fracs[best] >= LULC_THRESHOLD:
            merged.at[idx, 'dominant'] = best

    # keep lon in positive convention to match drought files
    merged['lon'] = merged['lon'].apply(lambda x: x + 360 if x < 0 else x)
    return merged[['lat', 'lon', 'cluster', 'frac_grass', 'frac_forest', 'dominant']]


def load_drought_events_lulc(folder_list):
    dfs = []
    for folder in folder_list:
        for c in CLUSTERS:
            fpath = os.path.join(folder, DROUGHT_FILE_PATTERN.format(c=c))
            if not os.path.exists(fpath):
                continue
            d = pd.read_csv(fpath)
            d['lat'] = d['lat'].round(4)
            d['lon'] = d['lon'].round(4)
            dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_kde_cdf(vals, x_grid):
    kde = gaussian_kde(vals, bw_method='scott')
    pdf = kde(x_grid)
    cdf = cumulative_trapezoid(pdf, x_grid, initial=0)
    cdf = cdf / cdf[-1]
    return cdf


# =============================================================================
# LOAD DATA
# =============================================================================
print('=' * 60)
print('LOADING DATA')
print('=' * 60)

valid_set = load_valid_cells()

print('\n-- Grass -> Forest CDF data --')
lulc_hist = load_lulc_fracs([LULC_FILE])
lulc_ssp3 = load_lulc_fracs(SSP3_LULC_FILES)
lulc_ssp5 = load_lulc_fracs(SSP5_LULC_FILES)

cells_grass_hist  = lulc_hist[lulc_hist['dominant'] == 'Grass'][['lat', 'lon', 'cluster']]
cells_forest_ssp3 = lulc_ssp3[lulc_ssp3['dominant'] == 'Forest'][['lat', 'lon']]
cells_forest_ssp5 = lulc_ssp5[lulc_ssp5['dominant'] == 'Forest'][['lat', 'lon']]

# cluster is a fixed spatial attribute of the grid cell, so it's carried
# through from the grass (hist) side of the merge -- same grid as the
# forest (SSP) side, just read directly from the LULC file's own cluster column.
common_S1S3 = cells_grass_hist.merge(cells_forest_ssp3, on=['lat', 'lon'], how='inner')
common_S2S4 = cells_grass_hist.merge(cells_forest_ssp5, on=['lat', 'lon'], how='inner')
print(f'  S1(Grass)&S3(Forest) common cells: {len(common_S1S3):,}')
print(f'  S2(Grass)&S4(Forest) common cells: {len(common_S2S4):,}')

# --- Per-cluster (region) breakdown of the common cells ---
print('\nS1(Grass)&S3(Forest) common cells by region:')
counts_13 = common_S1S3['cluster'].value_counts().sort_index()
for cid in CLUSTERS:
    n = int(counts_13.get(cid, 0))
    print(f'  Cluster {cid} ({REGION_NAMES[cid]}): {n:,}')

print('\nS2(Grass)&S4(Forest) common cells by region:')
counts_24 = common_S2S4['cluster'].value_counts().sort_index()
for cid in CLUSTERS:
    n = int(counts_24.get(cid, 0))
    print(f'  Cluster {cid} ({REGION_NAMES[cid]}): {n:,}')

# For the drought-event merge below we only need lat/lon
common_S1S3 = common_S1S3[['lat', 'lon']]
common_S2S4 = common_S2S4[['lat', 'lon']]

# (leg_name, land-cover category, scenario key, cell set to restrict to)
LULC_PAIR_SPECS = [
    ('S1', 'Grass',  'RCP4.5',      common_S1S3),
    ('S3', 'Forest', 'SSP3-RCP4.5', common_S1S3),
    ('S2', 'Grass',  'RCP8.5',      common_S2S4),
    ('S4', 'Forest', 'SSP5-RCP8.5', common_S2S4),
]

x_grid_slope = np.linspace(0, 17, 1000)
x_grid_onset = np.linspace(91, 281, 1000)
cdfs_lulc_slope = {}
cdfs_lulc_onset = {}
n_lulc          = {}
mean_lulc_slope = {}
mean_lulc_onset = {}

for leg, cat, scen_key, cell_set in LULC_PAIR_SPECS:
    df_raw = load_drought_events_lulc(SCENARIO_FOLDERS[scen_key])
    df_flash = df_raw[
        (df_raw['drought_type'] == 'Flash') &
        (df_raw['intensification_pentads'] <= MAX_DURATION_PENTADS)
    ].copy()
    df_flash['onset_julian'] = (df_flash['S0_index'] - 1) * 5 + 1
    df_merged = df_flash.merge(cell_set, on=['lat', 'lon'], how='inner')
    df_merged = df_merged.dropna(subset=['GPP_slope', 'onset_julian'])
    df_merged['abs_slope'] = df_merged['GPP_slope'].abs()

    vals_s = df_merged['abs_slope'].values
    vals_o = df_merged['onset_julian'].values
    n_lulc[(leg, cat)] = len(vals_s)
    cdfs_lulc_slope[(leg, cat)] = compute_kde_cdf(vals_s, x_grid_slope)
    cdfs_lulc_onset[(leg, cat)] = compute_kde_cdf(vals_o, x_grid_onset)
    mean_lulc_slope[(leg, cat)] = float(np.mean(vals_s))
    mean_lulc_onset[(leg, cat)] = float(np.mean(vals_o))
    print(f'  {leg} ({scen_key}) {cat}: n={len(vals_s):,}')

print('\nMEAN VALUES: panel (a) GPP decline rate (abs, percentile/pentad)')
for leg, cat, scen_key, cell_set in LULC_PAIR_SPECS:
    print(f'  {leg}: {cat:6s} ({scen_key:12s})  mean = {mean_lulc_slope[(leg, cat)]:.3f}  (n={n_lulc[(leg, cat)]:,})')

print('\nMEAN VALUES: panel (b) flash drought onset date (Julian day)')
for leg, cat, scen_key, cell_set in LULC_PAIR_SPECS:
    print(f'  {leg}: {cat:6s} ({scen_key:12s})  mean = {mean_lulc_onset[(leg, cat)]:.2f}  (n={n_lulc[(leg, cat)]:,})')

# =============================================================================
# BUILD FIGURE (1x2)
# =============================================================================
print('\n' + '=' * 60)
print('RENDERING FIGURE')
print('=' * 60)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.6))
fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.15, wspace=0.28)

# ------------------------------------------------------------------
# (a) CDF GPP slope by LULC x scenario  (was panel e)
# ------------------------------------------------------------------
print('  Panel (a): CDF GPP slope, S1 Grass vs S3 Forest / S2 Grass vs S4 Forest...')
for leg, cat in [('S1', 'Grass'), ('S3', 'Forest'), ('S2', 'Grass'), ('S4', 'Forest')]:
    ax_a.plot(x_grid_slope, cdfs_lulc_slope[(leg, cat)],
              color=PALETTE_LULC[cat],
              linestyle=LULC_PAIR_LINESTYLE[leg],
              linewidth=2.2,
              label=LULC_LEG_NAMES[(leg, cat)])

ax_a.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.55)
ax_a.set_xlim([0, 17])
ax_a.set_ylim([0, 1])
ax_a.set_xlabel(r'GPP decline rate (percentile/pentad)',
                fontsize=FS_AXIS_LABEL, labelpad=3)
ax_a.set_ylabel('Cumulative Probability', fontsize=FS_AXIS_LABEL, labelpad=4)
ax_a.tick_params(labelsize=FS_TICK)
ax_a.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.legend(fontsize=FS_LEGEND - 1, frameon=False, loc='lower right',
            handlelength=1.8, labelspacing=0.3)
ax_a.text(0.02, 0.97, '(a)', transform=ax_a.transAxes,
          fontsize=FS_PANEL_LABEL, fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------
# (b) CDF onset date by LULC  (was panel f)
# ------------------------------------------------------------------
print('  Panel (b): CDF onset date, S1 Grass vs S3 Forest / S2 Grass vs S4 Forest...')
for leg, cat in [('S1', 'Grass'), ('S3', 'Forest'), ('S2', 'Grass'), ('S4', 'Forest')]:
    ax_b.plot(x_grid_onset, cdfs_lulc_onset[(leg, cat)],
              color=PALETTE_LULC[cat],
              linestyle=LULC_PAIR_LINESTYLE[leg],
              linewidth=2.2)

for jd in MONTH_JDAYS:
    ax_b.axvline(jd, color='gray', linewidth=0.5, linestyle=':', alpha=0.45)

ax_b.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.55)
ax_b.set_xticks(MONTH_JDAYS)
ax_b.set_xticklabels(MONTH_LABELS, fontsize=FS_TICK)
ax_b.set_xlim([91, 281])
ax_b.set_ylim([0, 1])
ax_b.set_xlabel('Flash drought onset date', fontsize=FS_AXIS_LABEL, labelpad=3)
ax_b.set_ylabel('Cumulative Probability', fontsize=FS_AXIS_LABEL, labelpad=4)
ax_b.tick_params(axis='y', labelsize=FS_TICK)
ax_b.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(0.02, 0.97, '(b)', transform=ax_b.transAxes,
          fontsize=FS_PANEL_LABEL, fontweight='bold', va='top', ha='left')

# =============================================================================
# SAVE
# =============================================================================
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nFigure saved: {OUT_FILE}')
plt.show()