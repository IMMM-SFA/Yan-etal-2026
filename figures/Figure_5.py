# -*- coding: utf-8 -*-
"""
Z_combined_3x2_figure.py
========================
PNAS-quality 3x2 multi-panel figure combining Z1-Z6 analyses.

Panel layout:
  (a) Bar: Flash Drought GPP Slope by region/scenario      [Z1]
  (b) Bar: Flash Drought Min GPP at G2 by region/scenario  [Z4]
  (c) CDF: GPP Slope by scenario (4 scenarios)             [Z2]
  (d) CONUS spatial map: delta GPP (SSP3 - RCP4.5)         [Z3]
  (e) CDF: GPP Slope, S1 Grass vs S3 Crop / S2 Grass vs S4 Crop  [Z5]
  (f) CDF: Onset date,  S1 Grass vs S3 Crop / S2 Grass vs S4 Crop  [Z6]

  (e)/(f) cell selection: a grid cell is included in the S1-vs-S3 pair only
  if it is >=60% grass under S1 (RCP4.5, using historical LULC) AND >=60%
  crop under S3 (SSP3-RCP4.5 LULC) -- i.e. the same cell converts from
  grass to cropland under the SSP3 land-use pathway. The S2-vs-S4 pair
  uses the analogous >=60% grass (S2/hist) AND >=60% crop (S4/SSP5) test.

Color conventions:
  RCP4.5        -> #2E86AB  (dark blue)
  SSP3-RCP4.5   -> #66C2E8  (light blue)
  RCP8.5        -> #E66101  (dark orange)
  SSP5-RCP8.5   -> #F5A962  (light orange)
  Grass         -> #2D6A4F  (deep emerald green)
  Crop          -> #C77800  (warm amber/ochre)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde, mannwhitneyu
from scipy.integrate import cumulative_trapezoid

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# =============================================================================
# GLOBAL FONT / SIZE CONSTANTS  (tune once -> propagates everywhere)
# =============================================================================
FS_PANEL_LABEL = 14   # (a) (b) ... bold letter
FS_AXIS_LABEL  = 11   # x/y axis label
FS_TICK        = 10   # tick labels
FS_LEGEND      = 10   # legend text
FS_LEGEND_TTL  = 10   # legend title
FS_CBAR        = 10   # colour-bar label / ticks

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus'

MASTER_CSV      = os.path.join(BASE_DIR, 'Compiled_Master_GPP_Metrics_ClimateOnly_with_AbsG0_G2_Severity_v3.csv')
VALID_CELLS_CSV = os.path.join(BASE_DIR, 'hist', 'valid_conus_flash_drought_cells.csv')
LULC_FILE       = os.path.join(BASE_DIR, 'hist/pft_cft_lulc/lulc_mean.csv')

# Scenario-specific LULC (used for S3/S4 -- grass->crop conversion under SSPs).
# S1 (RCP4.5) and S2 (RCP8.5) use the same historical LULC as LULC_FILE above.
SSP3_LULC_FILES = [
    os.path.join(BASE_DIR, 'ssp3_rcp45_cooler_near', 'pft_cft_lulc', 'lulc_mean.csv'),
    os.path.join(BASE_DIR, 'ssp3_rcp45_hotter_near', 'pft_cft_lulc', 'lulc_mean.csv'),
]
SSP5_LULC_FILES = [
    os.path.join(BASE_DIR, 'ssp5_rcp85_cooler_near', 'pft_cft_lulc', 'lulc_mean.csv'),
    os.path.join(BASE_DIR, 'ssp5_rcp85_hotter_near', 'pft_cft_lulc', 'lulc_mean.csv'),
]
US_BND_FILE     = os.path.join(BASE_DIR, 'hist', 'us_coor.txt')
BOUNDARY_FILES  = [
    os.path.join(BASE_DIR, 'hist', f'R{i}_{name}.txt')
    for i, name in zip(range(1, 8), ['nw', 'sw', 'ngp', 'sgp', 'mw', 'se', 'ne'])
]

DROUGHT_FILE_PATTERN = 'drought_events_fixed_season_cluster_{c}_with_GPP_Zhang2025.csv'
CLUSTERS = range(1, 8)

OUT_FILE = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/ssp3_rcp45_cooler_near/Paper_Figure_5_v4.png'

# =============================================================================
# SCENARIO / REGION METADATA
# =============================================================================
REGION_NAMES  = {1: 'NW', 2: 'SW', 3: 'NGP', 4: 'SGP', 5: 'MW', 6: 'SE', 7: 'NE'}
GROUP_LABELS  = ['CONUS'] + [REGION_NAMES[c] for c in CLUSTERS]

HIST_LABEL    = 'Historical'
RCP45_HOTTER  = 'RCP4.5 Hotter Near'
RCP45_COOLER  = 'RCP4.5 Cooler Near'
RCP85_HOTTER  = 'RCP8.5 Hotter Near'
RCP85_COOLER  = 'RCP8.5 Cooler Near'
SSP345_HOTTER = 'SSP3-RCP4.5 Hotter Near'
SSP345_COOLER = 'SSP3-RCP4.5 Cooler Near'
SSP585_HOTTER = 'SSP5-RCP8.5 Hotter Near'
SSP585_COOLER = 'SSP5-RCP8.5 Cooler Near'

SCENARIO_ORDER = ['RCP4.5 (Near)', 'SSP3-RCP4.5 (Near)', 'RCP8.5 (Near)', 'SSP5-RCP8.5 (Near)']
SCENARIO_SHORT = ['RCP4.5', 'SSP3-RCP4.5', 'RCP8.5', 'SSP5-RCP8.5']   # for CDF legend

# Scenario colours (consistent across ALL panels)
C_RCP45  = '#2E86AB'
C_SSP345 = '#66C2E8'
C_RCP85  = '#E66101'
C_SSP585 = '#F5A962'

PALETTE_SCEN = {
    'RCP4.5 (Near)':      C_RCP45,
    'SSP3-RCP4.5 (Near)': C_SSP345,
    'RCP8.5 (Near)':      C_RCP85,
    'SSP5-RCP8.5 (Near)': C_SSP585,
}

# CDF scenario map (Z2): label -> folder list
SCENARIO_MAPPING_CDF = {
    'RCP4.5':      ['rcp45_hotter_near',      'rcp45_cooler_near'],
    'SSP3-RCP4.5': ['ssp3_rcp45_hotter_near', 'ssp3_rcp45_cooler_near'],
    'RCP8.5':      ['rcp85_hotter_near',      'rcp85_cooler_near'],
    'SSP5-RCP8.5': ['ssp5_rcp85_hotter_near', 'ssp5_rcp85_cooler_near'],
}
PALETTE_CDF = {
    'RCP4.5':      C_RCP45,
    'SSP3-RCP4.5': C_SSP345,
    'RCP8.5':      C_RCP85,
    'SSP5-RCP8.5': C_SSP585,
}
LINESTYLE_CDF = {
    'RCP4.5':      '-',
    'SSP3-RCP4.5': '--',
    'RCP8.5':      '-',
    'SSP5-RCP8.5': '--',
}

# RCP scenarios for LULC CDFs (Z5, Z6)
RCP_SCENARIOS_LULC = {
    'RCP4.5': [os.path.join(BASE_DIR, 'rcp45_cooler_near'),
               os.path.join(BASE_DIR, 'rcp45_hotter_near')],
    'RCP8.5': [os.path.join(BASE_DIR, 'rcp85_cooler_near'),
               os.path.join(BASE_DIR, 'rcp85_hotter_near')],
}

# LULC colours: print-safe, high-contrast pair
C_GRASS = '#2D6A4F'   # deep emerald
C_CROP  = '#C77800'   # warm amber/ochre

PALETTE_LULC  = {'Grass': C_GRASS, 'Crop': C_CROP}
LINESTYLE_RCP = {'RCP4.5': '-', 'RCP8.5': '--'}

FOREST_PFTS = [f'pft_{i:02d}' for i in range(1, 9)]
GRASS_PFTS  = ['pft_12', 'pft_13', 'pft_14']
CROP_CFTS   = [f'cft_{i}' for i in range(15, 79)]
LULC_THRESHOLD      = 0.6
MAX_DURATION_PENTADS = 6

# Onset date month ticks
MONTH_JDAYS  = [91, 121, 152, 182, 213, 244, 274]
MONTH_LABELS = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']

# =============================================================================
# ---- DATA LOADING HELPERS ---------------------------------------------------
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


def load_master_csv(valid_set):
    print('Loading master CSV...')
    df = pd.read_csv(MASTER_CSV)
    df['lon'] = np.where(df['lon'] > 180, df['lon'] - 360.0, df['lon'])
    df['lat'] = df['lat'].round(4)
    df['lon'] = df['lon'].round(4)
    df = df[df.apply(lambda r: (r['lat'], r['lon']) in valid_set, axis=1)].copy()
    print(f'  Rows after valid filter: {len(df):,}')
    return df


def nine_way_merge(df, slope_col):
    """Inner-merge all 9 scenarios on lat/lon, return merged DataFrame."""

    def get_vals(scenario_label, alias):
        sub = (df[df['scenario'] == scenario_label]
               [['lat', 'lon', 'cluster', slope_col]]
               .rename(columns={slope_col: alias})
               .copy())
        return sub

    df_hist    = get_vals(HIST_LABEL,    'Hist')
    df_45h     = get_vals(RCP45_HOTTER,  'RCP45_H')
    df_45c     = get_vals(RCP45_COOLER,  'RCP45_C')
    df_85h     = get_vals(RCP85_HOTTER,  'RCP85_H')
    df_85c     = get_vals(RCP85_COOLER,  'RCP85_C')
    df_ssp345h = get_vals(SSP345_HOTTER, 'SSP345_H')
    df_ssp345c = get_vals(SSP345_COOLER, 'SSP345_C')
    df_ssp585h = get_vals(SSP585_HOTTER, 'SSP585_H')
    df_ssp585c = get_vals(SSP585_COOLER, 'SSP585_C')

    merged = df_hist.copy()
    for df_tmp, col in [
        (df_45h, 'RCP45_H'), (df_45c, 'RCP45_C'),
        (df_85h, 'RCP85_H'), (df_85c, 'RCP85_C'),
        (df_ssp345h, 'SSP345_H'), (df_ssp345c, 'SSP345_C'),
        (df_ssp585h, 'SSP585_H'), (df_ssp585c, 'SSP585_C'),
    ]:
        merged = merged.merge(df_tmp[['lat', 'lon', col]], on=['lat', 'lon'], how='inner')

    merged['RCP45_Near']  = (merged['RCP45_H']  + merged['RCP45_C'])  / 2.0
    merged['RCP85_Near']  = (merged['RCP85_H']  + merged['RCP85_C'])  / 2.0
    merged['SSP345_Near'] = (merged['SSP345_H'] + merged['SSP345_C']) / 2.0
    merged['SSP585_Near'] = (merged['SSP585_H'] + merged['SSP585_C']) / 2.0
    merged = merged.dropna(subset=['Hist', 'RCP45_Near', 'RCP85_Near',
                                   'SSP345_Near', 'SSP585_Near'])
    print(f'  9-way merge -> {len(merged):,} cells')
    return merged


def regional_stats(merged):
    col_map = {
        'RCP4.5 (Near)':      'RCP45_Near',
        'SSP3-RCP4.5 (Near)': 'SSP345_Near',
        'RCP8.5 (Near)':      'RCP85_Near',
        'SSP5-RCP8.5 (Near)': 'SSP585_Near',
    }
    stats = {}
    for label, col in col_map.items():
        vals = [merged[col].mean()]
        for cid in CLUSTERS:
            sub = merged[merged['cluster'] == cid]
            vals.append(sub[col].mean())
        stats[label] = vals
    return stats


def load_lulc_fracs(file_paths):
    """Load LULC fraction csv(s) and return per-cell frac_grass/frac_crop/dominant.

    If more than one file is given (e.g. cooler + hotter variants of the same
    scenario), the PFT/CFT fractions are averaged across files on the common
    lat/lon grid before the dominant land-cover classification is applied.
    """
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
        fracs = {'Forest': row['frac_forest'],
                 'Grass':  row['frac_grass'],
                 'Crop':   row['frac_crop']}
        best = max(fracs, key=fracs.get)
        if fracs[best] >= LULC_THRESHOLD:
            merged.at[idx, 'dominant'] = best

    # keep lon in positive convention to match drought files
    merged['lon'] = merged['lon'].apply(lambda x: x + 360 if x < 0 else x)
    return merged[['lat', 'lon', 'frac_grass', 'frac_crop', 'dominant']]


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
# ---- LOAD ALL DATA ----------------------------------------------------------
# =============================================================================
print('=' * 60)
print('LOADING DATA')
print('=' * 60)

valid_set = load_valid_cells()
df_master = load_master_csv(valid_set)

# --- panels (a) and (b): bar data ---
print('\n-- Bar panel data (slope) --')
df_slope = df_master.copy()
df_slope['abs_slope'] = df_slope['flash_mean_GPP_slope'].abs()

# temporary col rename so nine_way_merge works for slope
df_slope_renamed = df_slope.rename(columns={'abs_slope': 'flash_mean_GPP_slope_use'})
# Use a copy of master with abs_slope in place for slope bars
merged_slope = nine_way_merge(
    df_master.assign(**{'flash_mean_GPP_slope':
                        df_master['flash_mean_GPP_slope'].abs()}),
    'flash_mean_GPP_slope'
)
stats_slope = regional_stats(merged_slope)

print('\n-- Bar panel data (min GPP G2) --')
merged_g2   = nine_way_merge(df_master, 'flash_meanGPP_G2_abs_value')
stats_g2    = regional_stats(merged_g2)

# --- panel (c): CDF by scenario (Z2-style, per-event from folder files) ---
print('\n-- CDF scenario data (Z2) --')
df_valid_coords = pd.DataFrame(list(valid_set), columns=['lat', 'lon'])
cdfs_scen    = {}
n_scen       = {}
x_grid_slope = np.linspace(0, 17, 1000)

for label, folders in SCENARIO_MAPPING_CDF.items():
    all_dfs = []
    for folder in folders:
        scenario_dir = os.path.join(BASE_DIR, folder)
        for cid in CLUSTERS:
            fpath = os.path.join(
                scenario_dir,
                f'drought_events_fixed_season_cluster_{cid}_with_GPP_Zhang2025.csv'
            )
            if not os.path.exists(fpath):
                continue
            d = pd.read_csv(fpath,
                            usecols=lambda c: c in
                            ['lat', 'lon', 'drought_type',
                             'intensification_pentads', 'GPP_slope'])
            if d.empty:
                continue
            d['lon_std']  = np.where(d['lon'] > 180, d['lon'] - 360.0, d['lon'])
            d['lat_join'] = d['lat'].round(4)
            d['lon_join'] = d['lon_std'].round(4)
            d = d.merge(df_valid_coords.rename(columns={'lat': 'lat_join', 'lon': 'lon_join'}),
                        on=['lat_join', 'lon_join'], how='inner')
            all_dfs.append(d)
    if not all_dfs:
        print(f'  WARNING: no data for {label}')
        continue
    dfc = pd.concat(all_dfs, ignore_index=True)
    dfc = dfc.dropna(subset=['GPP_slope'])
    dfc = dfc[(dfc['GPP_slope'] <= 0) & (dfc['GPP_slope'] >= -50)]
    dfc = dfc[dfc['drought_type'] == 'Flash']
    dfc = dfc[dfc['intensification_pentads'] <= MAX_DURATION_PENTADS]
    dfc['abs_slope'] = dfc['GPP_slope'].abs()
    vals = dfc['abs_slope'].values
    n_scen[label] = len(vals)
    cdfs_scen[label] = compute_kde_cdf(vals, x_grid_slope)
    print(f'  {label}: n={len(vals):,}, median={np.median(vals):.2f}')

# --- panel (d): spatial delta map (Z3) ---
print('\n-- Spatial map data (Z3) --')

def get_g2_vals(scenario_label, alias):
    out = (df_master[df_master['scenario'] == scenario_label]
           [['lat', 'lon', 'cluster', 'flash_meanGPP_G2_abs_value']]
           .rename(columns={'flash_meanGPP_G2_abs_value': alias})
           .copy())
    out['lat'] = out['lat'].round(4)
    out['lon'] = out['lon'].round(4)
    return out

df_45c_sp   = get_g2_vals(RCP45_COOLER,           'r45c').drop(columns=['cluster'], errors='ignore')
df_45h_sp   = get_g2_vals(RCP45_HOTTER,           'r45h').drop(columns=['cluster'], errors='ignore')
df_ssp3c_sp = get_g2_vals(SSP345_COOLER,          'ssp3c').drop(columns=['cluster'], errors='ignore')
df_ssp3h_sp = get_g2_vals(SSP345_HOTTER,          'ssp3h').drop(columns=['cluster'], errors='ignore')

# Relaxed merge for (c): outer-join so a cell survives if it has EITHER
# hotter or cooler (or both) for each scenario, instead of requiring both.
# RCP45_mean / SSP3_mean are then computed as the mean of whichever
# sub-runs are actually present (skipna), rather than needing both.
merged_sp = (df_45c_sp
             .merge(df_45h_sp,   on=['lat', 'lon'], how='outer')
             .merge(df_ssp3c_sp, on=['lat', 'lon'], how='outer')
             .merge(df_ssp3h_sp, on=['lat', 'lon'], how='outer'))
merged_sp['RCP45_mean'] = merged_sp[['r45c', 'r45h']].mean(axis=1, skipna=True)
merged_sp['SSP3_mean']  = merged_sp[['ssp3c', 'ssp3h']].mean(axis=1, skipna=True)
merged_sp['delta']      = merged_sp['SSP3_mean'] - merged_sp['RCP45_mean']
merged_sp['lon_plot']   = np.where(merged_sp['lon'] > 180,
                                   merged_sp['lon'] - 360,
                                   merged_sp['lon'])
# A cell is only usable once it has at least one value on each side of
# the delta -- drop on 'delta' itself, not the raw per-run columns.
merged_sp = merged_sp.dropna(subset=['delta'])
print(f'  Spatial grid cells: {len(merged_sp):,}')

# --- panels (e) and (f): S1-Grass vs S3-Crop / S2-Grass vs S4-Crop CDFs ---
print('\n-- LULC-change CDF data (Z5, Z6) --')

# S1 (RCP4.5) and S2 (RCP8.5) both use the historical LULC.
lulc_hist = load_lulc_fracs([LULC_FILE])
# S3 (SSP3-RCP4.5) and S4 (SSP5-RCP8.5) use their own scenario LULC
# (cooler+hotter fractions averaged before classification).
lulc_ssp3 = load_lulc_fracs(SSP3_LULC_FILES)
lulc_ssp5 = load_lulc_fracs(SSP5_LULC_FILES)

cells_grass_hist = lulc_hist[lulc_hist['dominant'] == 'Grass'][['lat', 'lon']]
cells_crop_ssp3  = lulc_ssp3[lulc_ssp3['dominant'] == 'Crop'][['lat', 'lon']]
cells_crop_ssp5  = lulc_ssp5[lulc_ssp5['dominant'] == 'Crop'][['lat', 'lon']]

# Same-cell requirement: >=60% grass under S1/S2 (hist) AND >=60% crop
# under S3/S4 (SSP LULC), at the identical lat/lon.
common_S1S3 = cells_grass_hist.merge(cells_crop_ssp3, on=['lat', 'lon'], how='inner')
common_S2S4 = cells_grass_hist.merge(cells_crop_ssp5, on=['lat', 'lon'], how='inner')
print(f'  S1(Grass)&S3(Crop) common cells: {len(common_S1S3):,}')
print(f'  S2(Grass)&S4(Crop) common cells: {len(common_S2S4):,}')

# Scenario -> folder list (cooler + hotter pooled together, matches Z2 convention)
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

# (leg_name, land-cover category, scenario key, cell set to restrict to)
LULC_PAIR_SPECS = [
    ('S1', 'Grass', 'RCP4.5',      common_S1S3),
    ('S3', 'Crop',  'SSP3-RCP4.5', common_S1S3),
    ('S2', 'Grass', 'RCP8.5',      common_S2S4),
    ('S4', 'Crop',  'SSP5-RCP8.5', common_S2S4),
]

x_grid_onset = np.linspace(91, 281, 1000)
cdfs_lulc_slope = {}
cdfs_lulc_onset = {}
n_lulc          = {}

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
    print(f'  {leg} ({scen_key}) {cat}: n={len(vals_s):,}')

# =============================================================================
# ---- BUILD FIGURE -----------------------------------------------------------
# =============================================================================
print('\n' + '=' * 60)
print('RENDERING FIGURE')
print('=' * 60)

# Figure size
FIG_W, FIG_H = 15.0, 13.0

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Equal height rows, tighter vertical spacing
gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[1.0, 1.0, 1.0],
    hspace=0.32,
    wspace=0.28,
    left=0.07,
    right=0.97,
    top=0.96,
    bottom=0.06,
)

ax_a = fig.add_subplot(gs[0, 0])   # (a) Z1: bar GPP slope
ax_b = fig.add_subplot(gs[0, 1])   # (b) Z2: CDF GPP slope by scenario
ax_c = fig.add_subplot(gs[1, 0])   # (c) Z3: CONUS spatial delta map
ax_d = fig.add_subplot(gs[1, 1])   # (d) Z4: bar min GPP G2
ax_e = fig.add_subplot(gs[2, 0])   # (e) Z5: CDF GPP slope by LULC
ax_f = fig.add_subplot(gs[2, 1])   # (f) Z6: CDF onset date by LULC

# Legend name mapping
LEG_NAMES = {
    'RCP4.5 (Near)':      'S1',
    'SSP3-RCP4.5 (Near)': 'S3',
    'RCP8.5 (Near)':      'S2',
    'SSP5-RCP8.5 (Near)': 'S4',
}

# ------------------------------------------------------------------
# Shared bar-plot helper (no legend drawn inside -- handled per panel)
# ------------------------------------------------------------------
def draw_bar_panel(ax, stats, ylabel, panel_label, ylim_pad=0.10):
    n_groups = len(GROUP_LABELS)
    x   = np.arange(n_groups)
    bw  = 0.185
    off = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw]

    bars = {}
    for i, label in enumerate(SCENARIO_ORDER):
        b = ax.bar(x + off[i], stats[label], width=bw,
                   color=PALETTE_SCEN[label], alpha=0.88,
                   edgecolor='white', linewidth=0.4,
                   label=LEG_NAMES[label], zorder=3)
        bars[label] = b

    ax.axvline(x=0.5, color='gray', linewidth=0.8,
               linestyle='--', alpha=0.5, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL, labelpad=4)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.45, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_vals = [v for vals in stats.values() for v in vals]
    ymax = max(all_vals) * (1 + ylim_pad)
    ax.set_ylim(0, ymax)

    ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
            fontsize=FS_PANEL_LABEL, fontweight='bold', va='top', ha='left')

    return ax


# ------------------------------------------------------------------
# (a) Z1: GPP slope bars
# ------------------------------------------------------------------
print('  Panel (a) = Z1: bar GPP slope...')
draw_bar_panel(
    ax_a, stats_slope,
    ylabel='GPP decline rate\n' + r'(percentile/pentad)',
    panel_label='(a)'
)
# 2x2 legend in upper centre, no box
handles_a, labels_a = ax_a.get_legend_handles_labels()
ax_a.legend(
    handles_a, labels_a,
    ncol=2,
    fontsize=FS_LEGEND - 1,
    loc='upper center',
    frameon=False,
    handlelength=1.2,
    borderpad=0.4,
    labelspacing=0.3,
    columnspacing=0.8,
)

# ------------------------------------------------------------------
# (b) Z2: CDF of GPP slope by scenario
# ------------------------------------------------------------------
print('  Panel (b) = Z2: CDF GPP slope by scenario...')
CDF_LEG_NAMES = {
    'RCP4.5':      'S1',
    'SSP3-RCP4.5': 'S3',
    'RCP8.5':      'S2',
    'SSP5-RCP8.5': 'S4',
}
for label in SCENARIO_MAPPING_CDF.keys():
    if label not in cdfs_scen:
        continue
    ax_b.plot(x_grid_slope, cdfs_scen[label],
              color=PALETTE_CDF[label],
              linestyle=LINESTYLE_CDF[label],
              linewidth=2.2,
              label=CDF_LEG_NAMES[label])

ax_b.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.55)
ax_b.set_xlim([0, 17])
ax_b.set_ylim([0, 1])
ax_b.set_xlabel(r'GPP decline rate (percentile/pentad)',
                fontsize=FS_AXIS_LABEL, labelpad=3)
ax_b.set_ylabel('Cumulative Probability', fontsize=FS_AXIS_LABEL, labelpad=4)
ax_b.tick_params(labelsize=FS_TICK)
ax_b.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.legend(fontsize=FS_LEGEND - 1, frameon=False, loc='lower right',
            handlelength=1.8, labelspacing=0.3)
ax_b.text(0.02, 0.97, '(b)', transform=ax_b.transAxes,
          fontsize=FS_PANEL_LABEL, fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------
# (c) Z3: Spatial CONUS delta map
# ------------------------------------------------------------------
print('  Panel (c) = Z3: spatial delta map...')
ax_c.set_facecolor('#f2f2f2')
# DIAGNOSTIC ONLY: uncomment to visually separate "no data" from "delta ~ 0"
# (PiYG never renders pure dark gray, so any dark patches showing through
# are missing cells, not near-zero values)
# ax_c.set_facecolor('#404040')

if os.path.exists(US_BND_FILE):
    df_us = pd.read_csv(US_BND_FILE, sep=r'\s+', header=None, on_bad_lines='skip')
    ax_c.scatter(df_us[0][::10], df_us[1][::10],
                 color='#aaaaaa', s=0.3, alpha=0.4, zorder=2)

grid_sp = merged_sp.pivot(index='lat', columns='lon_plot', values='delta')
sc = ax_c.pcolormesh(grid_sp.columns, grid_sp.index, grid_sp.values,
                     cmap='PiYG', vmin=-1, vmax=1,
                     zorder=1, shading='nearest')

for cid, bnd_file in zip(CLUSTERS, BOUNDARY_FILES):
    if os.path.exists(bnd_file):
        df_reg = pd.read_csv(bnd_file, sep=r'\s+', header=None, on_bad_lines='skip')
        ax_c.scatter(df_reg[1], df_reg[0],
                     color='#333333', s=0.6, alpha=0.8, zorder=3)
        # Region label at the boundary centroid
        ax_c.text(df_reg[1].mean(), df_reg[0].mean(), REGION_NAMES[cid],
                  fontsize=FS_TICK, fontweight='bold', color='#333333',
                  ha='center', va='center', zorder=4,
                  path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

# Fill the subplot fully -- let the map use all available height
ax_c.set_xlim([-126, -66])
ax_c.set_ylim([24, 50])
ax_c.set_aspect('auto')   # fill subplot box; map proportions still look correct
ax_c.set_axis_off()

cbar = plt.colorbar(sc, ax=ax_c,
                    orientation='horizontal',
                    pad=0.02, shrink=0.85, aspect=32, extend='both')
cbar.set_label(
    r'Min. GPP diff. under flash drought (gC m$^{-2}$ day$^{-1}$): S3 - S1',
    fontsize=FS_CBAR, labelpad=3
)
cbar.ax.tick_params(labelsize=FS_CBAR)

ax_c.text(0.02, 0.03, '(c)', transform=ax_c.transAxes,
          fontsize=FS_PANEL_LABEL, fontweight='bold', va='bottom', ha='left', color='black')

# ------------------------------------------------------------------
# (d) Z4: Min GPP G2 bars -- legend (same style as (a))
# ------------------------------------------------------------------
print('  Panel (d) = Z4: bar min GPP G2...')
draw_bar_panel(
    ax_d, stats_g2,
    ylabel=r'Min. GPP under flash drought (gC m$^{-2}$ day$^{-1}$)',
    panel_label='(d)'
)
# 2x2 legend in upper centre, no box (same as (a))
handles_d, labels_d = ax_d.get_legend_handles_labels()
ax_d.legend(
    handles_d, labels_d,
    ncol=2,
    fontsize=FS_LEGEND - 1,
    loc='upper center',
    frameon=False,
    handlelength=1.2,
    borderpad=0.4,
    labelspacing=0.3,
    columnspacing=0.8,
)

# ------------------------------------------------------------------
# (e) Z5: CDF GPP slope by LULC x scenario
# ------------------------------------------------------------------
print('  Panel (e) = Z5: CDF GPP slope, S1 Grass vs S3 Crop / S2 Grass vs S4 Crop...')
LULC_LEG_NAMES = {
    ('S1', 'Grass'): 'S1: Grass',
    ('S3', 'Crop'):  'S3: Crop',
    ('S2', 'Grass'): 'S2: Grass',
    ('S4', 'Crop'):  'S4: Crop',
}
# Pair 1 (S1 grass / S3 crop) drawn solid; pair 2 (S2 grass / S4 crop) dashed
LULC_PAIR_LINESTYLE = {'S1': '-', 'S3': '-', 'S2': '--', 'S4': '--'}
for leg, cat in [('S1', 'Grass'), ('S3', 'Crop'), ('S2', 'Grass'), ('S4', 'Crop')]:
    ax_e.plot(x_grid_slope, cdfs_lulc_slope[(leg, cat)],
              color=PALETTE_LULC[cat],
              linestyle=LULC_PAIR_LINESTYLE[leg],
              linewidth=2.2,
              label=LULC_LEG_NAMES[(leg, cat)])

ax_e.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.55)
ax_e.set_xlim([0, 17])
ax_e.set_ylim([0, 1])
ax_e.set_xlabel(r'GPP decline rate (percentile/pentad)',
                fontsize=FS_AXIS_LABEL, labelpad=3)
ax_e.set_ylabel('Cumulative Probability', fontsize=FS_AXIS_LABEL, labelpad=4)
ax_e.tick_params(labelsize=FS_TICK)
ax_e.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
ax_e.legend(fontsize=FS_LEGEND - 1, frameon=False, loc='lower right',
            handlelength=1.8, labelspacing=0.3)
ax_e.text(0.02, 0.97, '(e)', transform=ax_e.transAxes,
          fontsize=FS_PANEL_LABEL, fontweight='bold', va='top', ha='left')

# ------------------------------------------------------------------
# (f) Z6: CDF onset date by LULC -- legend (same as (e))
# ------------------------------------------------------------------
print('  Panel (f) = Z6: CDF onset date, S1 Grass vs S3 Crop / S2 Grass vs S4 Crop...')
for leg, cat in [('S1', 'Grass'), ('S3', 'Crop'), ('S2', 'Grass'), ('S4', 'Crop')]:
    ax_f.plot(x_grid_onset, cdfs_lulc_onset[(leg, cat)],
              color=PALETTE_LULC[cat],
              linestyle=LULC_PAIR_LINESTYLE[leg],
              linewidth=2.2,
              label=LULC_LEG_NAMES[(leg, cat)])

for jd in MONTH_JDAYS:
    ax_f.axvline(jd, color='gray', linewidth=0.5, linestyle=':', alpha=0.45)

ax_f.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.55)
ax_f.set_xticks(MONTH_JDAYS)
ax_f.set_xticklabels(MONTH_LABELS, fontsize=FS_TICK)
ax_f.set_xlim([91, 281])
ax_f.set_ylim([0, 1])
ax_f.set_xlabel('Flash drought onset date', fontsize=FS_AXIS_LABEL, labelpad=3)
ax_f.set_ylabel('Cumulative Probability', fontsize=FS_AXIS_LABEL, labelpad=4)
ax_f.tick_params(axis='y', labelsize=FS_TICK)
ax_f.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
ax_f.spines['top'].set_visible(False)
ax_f.spines['right'].set_visible(False)
ax_f.legend(fontsize=FS_LEGEND - 1, frameon=False, loc='lower right',
            handlelength=1.8, labelspacing=0.3)
ax_f.text(0.02, 0.97, '(f)', transform=ax_f.transAxes,
          fontsize=FS_PANEL_LABEL, fontweight='bold', va='top', ha='left')

# =============================================================================
# SAVE
# =============================================================================
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nFigure saved: {OUT_FILE}')
plt.show()