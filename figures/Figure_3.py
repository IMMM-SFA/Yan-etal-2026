# -*- coding: utf-8 -*-
"""
Combined 2x3 PNAS figure: subplots (a)-(f) correspond to Y4-Y9.
Layout:
  Row 0: (a) Y4 bar GPP slope regional  |  (b) Y5 delta CDF GPP slope  |  (c) Y6 bar GPP slope vs onset
  Row 1: (d) Y7 spatial RCP85-hist G2   |  (e) Y8 bar G2 abs GPP       |  (f) Y9 delta CDF iWUE
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR   = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus'
MASTER_CSV = os.path.join(BASE_DIR,
    'Compiled_Master_GPP_Metrics_ClimateOnly_with_AbsG0_G2_Severity_v3.csv')
VALID_CELLS_CSV = os.path.join(BASE_DIR, 'hist',
    'valid_conus_flash_drought_cells.csv')
US_BND_FILE = os.path.join(BASE_DIR, 'hist', 'us_coor.txt')
BOUNDARY_FILES = [
    os.path.join(BASE_DIR, 'hist', f'R{i}_{name}.txt')
    for i, name in zip(range(1, 8), ['nw', 'sw', 'ngp', 'sgp', 'mw', 'se', 'ne'])
]

WUE_SCENARIO_MAPPING = {
    'Historical':    (os.path.join(BASE_DIR, 'hist/WUE'),
                      'Historical_Baseline_iWUE_1981_2015.csv'),
    'RCP4.5 Cooler': (os.path.join(BASE_DIR, 'rcp45_cooler_near/WUE'), 'iWUE_future.csv'),
    'RCP4.5 Hotter': (os.path.join(BASE_DIR, 'rcp45_hotter_near/WUE'), 'iWUE_future.csv'),
    'RCP8.5 Cooler': (os.path.join(BASE_DIR, 'rcp85_cooler_near/WUE'), 'iWUE_future.csv'),
    'RCP8.5 Hotter': (os.path.join(BASE_DIR, 'rcp85_hotter_near/WUE'), 'iWUE_future.csv'),
}

CLUSTERS       = range(1, 8)
CLUSTER_NAMES  = ['CONUS', 'NW', 'SW', 'NGP', 'SGP', 'MW', 'SE', 'NE']
REGION_LABELS  = ['NW', 'SW', 'NGP', 'SGP', 'MW', 'SE', 'NE']

# ==============================================================================
# CONSISTENT COLOR PALETTE
# Bar charts (3-scenario): Hist / RCP4.5 mean / RCP8.5 mean
# CDF charts (4-scenario individual runs)
# ==============================================================================
C_HIST   = '#8b8b8b'
C_RCP45  = '#2E86AB'   # RCP4.5 bar/CDF color (consistent across all panels)
C_RCP85  = '#E66101'   # RCP8.5 bar/CDF color (consistent across all panels)

# Individual scenario CDF colors -- consistent across Y5 and Y9
C_R45C   = '#1a6faf'   # RCP4.5 Cooler  solid
C_R45H   = '#74b3e0'   # RCP4.5 Hotter  dashed
C_R85C   = '#c94b00'   # RCP8.5 Cooler  solid
C_R85H   = '#f4a46a'   # RCP8.5 Hotter  dashed

LS_COOLER = (0, ())        # solid
LS_HOTTER = (0, (5, 2))    # dashed

# ==============================================================================
# FONT SIZES  -- generous for PNAS Word embedding
# ==============================================================================
FS_TICK   = 11
FS_LABEL  = 12
FS_LEGEND = 11
FS_PANEL  = 15

# ==============================================================================
# HELPER: load valid cells
# ==============================================================================
def load_valid_cells():
    vc = pd.read_csv(VALID_CELLS_CSV)
    vc['lon'] = np.where(vc['lon'] > 180, vc['lon'] - 360.0, vc['lon'])
    vc['lat'] = vc['lat'].round(4)
    vc['lon'] = vc['lon'].round(4)
    return set(zip(vc['lat'], vc['lon']))

# ==============================================================================
# HELPER: load event CSVs for a scenario group
# ==============================================================================
CSV_TMPL = 'drought_events_fixed_season_cluster_{}_with_GPP_Zhang2025.csv'

def load_event_slopes(folder_list, valid_cells,
                      extra_cols=None, flash_only=True,
                      onset_max=6, slope_col='GPP_slope'):
    needed = ['lat', 'lon', 'drought_type', 'drought_phase_pentads',
              'intensification_pentads', slope_col]
    if extra_cols:
        needed += extra_cols
    frames = []
    for folder in folder_list:
        for cid in CLUSTERS:
            fpath = os.path.join(BASE_DIR, folder, CSV_TMPL.format(cid))
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath, usecols=lambda c: c in needed)
            df['lon'] = np.where(df['lon'] > 180, df['lon'] - 360.0, df['lon'])
            df['lat'] = df['lat'].round(4)
            df['lon'] = df['lon'].round(4)
            if flash_only:
                df = df[df['drought_type'] == 'Flash']
            df = df[df.apply(lambda r: (r['lat'], r['lon']) in valid_cells, axis=1)]
            df = df.dropna(subset=[slope_col])
            df = df[(df[slope_col] <= 0) & (df[slope_col] >= -50)]
            if 'intensification_pentads' in df.columns:
                df = df[df['intensification_pentads'] <= onset_max]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ==============================================================================
# HELPER: CDF function
# ==============================================================================
def make_cdf_fn(values):
    sv  = np.sort(values)
    cdf = np.arange(1, len(sv) + 1) / len(sv)
    return lambda x: np.interp(x, sv, cdf, left=0.0, right=1.0)

# ==============================================================================
# PRE-LOAD SHARED DATA
# ==============================================================================
print("Loading data...")

# Master CSV
master_df = pd.read_csv(MASTER_CSV)
master_df['lon'] = np.where(master_df['lon'] > 180,
                             master_df['lon'] - 360.0, master_df['lon'])
master_df['lat'] = master_df['lat'].round(4)
master_df['lon'] = master_df['lon'].round(4)

# Valid cells
valid_cells = load_valid_cells()
print(f"  Valid cells: {len(valid_cells):,}")

# ---- Y4 / Y8 shared inner merge on master CSV ----
def get_master_col(scenario_label, alias, col):
    out = (master_df[master_df['scenario'] == scenario_label]
           [['lat', 'lon', 'cluster', col]]
           .rename(columns={col: alias}).copy())
    return out

SLOPE_COL = 'flash_mean_GPP_slope'
G2_COL    = 'flash_meanGPP_G2_abs_value'

def inner_merge_5way(col, aliases=('hist','r45c','r45h','r85c','r85h')):
    scenario_labels = ['Historical', 'RCP4.5 Cooler Near', 'RCP4.5 Hotter Near',
                       'RCP8.5 Cooler Near', 'RCP8.5 Hotter Near']
    dfs = []
    for lbl, alias in zip(scenario_labels, aliases):
        d = master_df[master_df['scenario'] == lbl][['lat','lon','cluster',col]].copy()
        d = d.rename(columns={col: alias})
        dfs.append(d)
    merged = dfs[0]
    for d, alias in zip(dfs[1:], aliases[1:]):
        merged = merged.merge(d[['lat','lon', alias]], on=['lat','lon'], how='inner')
    merged = merged.dropna()
    # valid cell filter
    merged = merged[merged.apply(
        lambda r: (r['lat'], r['lon']) in valid_cells, axis=1)].copy()
    return merged

print("  Building Y4 merge (GPP slope)...")
m_slope = inner_merge_5way(SLOPE_COL)
m_slope['abs_hist'] = m_slope['hist'].abs()
m_slope['abs_r45c'] = m_slope['r45c'].abs()
m_slope['abs_r45h'] = m_slope['r45h'].abs()
m_slope['abs_r85c'] = m_slope['r85c'].abs()
m_slope['abs_r85h'] = m_slope['r85h'].abs()
m_slope['RCP45_Near'] = (m_slope['abs_r45c'] + m_slope['abs_r45h']) / 2.0
m_slope['RCP85_Near'] = (m_slope['abs_r85c'] + m_slope['abs_r85h']) / 2.0
m_slope['Hist_abs']   = m_slope['abs_hist']
print(f"  Y4 cells: {len(m_slope):,}")

print("  Building Y8 merge (G2 abs GPP)...")
m_g2 = inner_merge_5way(G2_COL)
m_g2['RCP45_mean'] = (m_g2['r45c'] + m_g2['r45h']) / 2.0
m_g2['RCP85_mean'] = (m_g2['r85c'] + m_g2['r85h']) / 2.0
print(f"  Y8 cells: {len(m_g2):,}")

# Y7: RCP8.5 mean - hist spatial
m_g2['delta'] = m_g2['RCP85_mean'] - m_g2['hist']
m_g2['lon_plot'] = np.where(m_g2['lon'] > 180, m_g2['lon'] - 360, m_g2['lon'])

# ---- Panel (d) ONLY: relaxed spatial merge ----
# Instead of requiring all 5 scenarios (strict inner merge used for m_g2 / panel e),
# keep a cell as long as it has Historical G2 AND at least one of RCP8.5
# Cooler/Hotter G2 (if only Hotter has a value, keep it and use that value
# as the "RCP85 mean" for that cell). This fills in more grid cells for the
# spatial map only; panel (e) and all other bar/CDF plots keep using m_g2.
def get_master_col2(scenario_label, col, alias):
    return (master_df[master_df['scenario'] == scenario_label]
            [['lat', 'lon', col]].rename(columns={col: alias}).copy())

d_hist = get_master_col2('Historical',         G2_COL, 'hist')
d_r85c = get_master_col2('RCP8.5 Cooler Near', G2_COL, 'r85c')
d_r85h = get_master_col2('RCP8.5 Hotter Near', G2_COL, 'r85h')

m_g2_d = d_hist.merge(d_r85c, on=['lat', 'lon'], how='left')
m_g2_d = m_g2_d.merge(d_r85h, on=['lat', 'lon'], how='left')
m_g2_d = m_g2_d.dropna(subset=['hist'])                        # hist required
m_g2_d = m_g2_d[m_g2_d[['r85c', 'r85h']].notna().any(axis=1)]  # >=1 RCP8.5 run required
m_g2_d['RCP85_mean_d'] = m_g2_d[['r85c', 'r85h']].mean(axis=1, skipna=True)
m_g2_d['delta_d']      = m_g2_d['RCP85_mean_d'] - m_g2_d['hist']
m_g2_d['lon_plot']      = np.where(m_g2_d['lon'] > 180, m_g2_d['lon'] - 360, m_g2_d['lon'])
m_g2_d = m_g2_d[m_g2_d.apply(
    lambda r: (r['lat'], r['lon']) in valid_cells, axis=1)].copy()
print(f"  Panel (d) spatial cells -- relaxed: {len(m_g2_d):,}  "
      f"(vs strict 5-way used elsewhere: {len(m_g2):,})")

# ---- Y5: event-level GPP slope CDFs (cooler+hotter events pooled, not meaned) ----
print("  Loading Y5 event slopes...")
hist_slopes = load_event_slopes(['hist'], valid_cells)['GPP_slope'].abs().values
r45_slopes  = load_event_slopes(['rcp45_cooler_near', 'rcp45_hotter_near'], valid_cells)['GPP_slope'].abs().values
r85_slopes  = load_event_slopes(['rcp85_cooler_near', 'rcp85_hotter_near'], valid_cells)['GPP_slope'].abs().values

# ---- Y6: GPP slope vs onset pentad ----
print("  Loading Y6 onset-binned slopes...")
def load_onset_scenario(label, folder_list):
    frames = []
    for folder in folder_list:
        for cid in CLUSTERS:
            fpath = os.path.join(BASE_DIR, folder, CSV_TMPL.format(cid))
            if not os.path.exists(fpath):
                continue
            cols_needed = ['lat','lon','drought_type',
                           'intensification_pentads','GPP_slope']
            df = pd.read_csv(fpath, usecols=lambda c: c in cols_needed)
            if df.empty:
                continue
            df['lon'] = np.where(df['lon'] > 180, df['lon'] - 360.0, df['lon'])
            df['lat'] = df['lat'].round(4)
            df['lon'] = df['lon'].round(4)
            df = df[df.apply(lambda r: (r['lat'], r['lon']) in valid_cells, axis=1)]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['Scenario'] = label
    return out

df_y6 = pd.concat([
    load_onset_scenario('Historical',  ['hist']),
    load_onset_scenario('RCP4.5 Near', ['rcp45_cooler_near', 'rcp45_hotter_near']),
    load_onset_scenario('RCP8.5 Near', ['rcp85_cooler_near', 'rcp85_hotter_near']),
], ignore_index=True)
df_y6 = df_y6[df_y6['drought_type'] == 'Flash'].copy()
df_y6 = df_y6.dropna(subset=['GPP_slope','intensification_pentads'])
df_y6 = df_y6[(df_y6['GPP_slope'] <= 0) & (df_y6['GPP_slope'] >= -50)]
df_y6 = df_y6[df_y6['intensification_pentads'] <= 6].copy()
df_y6['intensification_pentads'] = df_y6['intensification_pentads'].astype(int)
df_y6['GPP_Collapse_Rate'] = df_y6['GPP_slope'].abs()

# ---- Y9: WUE delta CDF ----
print("  Building Y9 WUE shared mask...")
REF_COL = G2_COL

def get_master_wue(scenario_label, alias):
    out = (master_df[master_df['scenario'] == scenario_label][['lat','lon',REF_COL]]
           .rename(columns={REF_COL: alias}).copy())
    return out

m_hist_w = get_master_wue('Historical',         'v_hist')
m_45c_w  = get_master_wue('RCP4.5 Cooler Near', 'v_45c')
m_45h_w  = get_master_wue('RCP4.5 Hotter Near', 'v_45h')
m_85c_w  = get_master_wue('RCP8.5 Cooler Near', 'v_85c')
m_85h_w  = get_master_wue('RCP8.5 Hotter Near', 'v_85h')

wue_merged = (m_hist_w
              .merge(m_45c_w, on=['lat','lon'], how='inner')
              .merge(m_45h_w, on=['lat','lon'], how='inner')
              .merge(m_85c_w, on=['lat','lon'], how='inner')
              .merge(m_85h_w, on=['lat','lon'], how='inner')
              .dropna())
shared_latlon_wue = set(zip(wue_merged['lat'].round(4),
                             wue_merged['lon'].round(4)))

wue_data = {}
for label, (folder, fname) in WUE_SCENARIO_MAPPING.items():
    fpath = os.path.join(folder, fname)
    if not os.path.exists(fpath):
        print(f"  WARNING WUE not found: {fpath}")
        continue
    df_w = pd.read_csv(fpath, usecols=['lat','lon','cluster','mean_iwue'])
    df_w['lat'] = df_w['lat'].round(4)
    df_w['lon'] = np.where(df_w['lon'] > 180, df_w['lon'] - 360, df_w['lon']).round(4)
    df_w = df_w.rename(columns={'mean_iwue': 'WUE_mean'}).dropna(subset=['WUE_mean'])
    df_w = df_w[df_w['WUE_mean'] > 0]
    df_w['_key'] = list(zip(df_w['lat'], df_w['lon']))
    df_w = df_w[df_w['_key'].isin(shared_latlon_wue)].drop(columns='_key').reset_index(drop=True)
    p99 = df_w['WUE_mean'].quantile(0.99)
    df_w = df_w[df_w['WUE_mean'] <= p99].reset_index(drop=True)
    wue_data[label] = df_w

print("Data loading complete.\n")

# ==============================================================================
# REGIONAL SUMMARY HELPERS
# ==============================================================================
def regional_summary(merged, hist_col, rcp45_col, rcp85_col):
    conus = {
        'cluster_name': 'CONUS',
        'Hist':  merged[hist_col].mean(),
        'RCP45': merged[rcp45_col].mean(),
        'RCP85': merged[rcp85_col].mean(),
    }
    rows = [conus]
    for cid, rname in zip(sorted(merged['cluster'].dropna().unique()), REGION_LABELS):
        sub = merged[merged['cluster'] == cid]
        rows.append({
            'cluster_name': rname,
            'Hist':  sub[hist_col].mean(),
            'RCP45': sub[rcp45_col].mean(),
            'RCP85': sub[rcp85_col].mean(),
        })
    return pd.DataFrame(rows)

# ==============================================================================
# BAR Y-LIM HELPER -- non-zero baseline to make differences easier to see.
# ==============================================================================
def set_nonzero_ylim(ax, values, min_frac=0.7, top_pad=1.25):
    vals = np.asarray(values, dtype=float)
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    step = 0.5 if vmax < 10 else (1.0 if vmax < 50 else 5.0)
    ymin = np.floor((vmin * min_frac) / step) * step
    ymin = max(0.0, ymin)
    ymax = vmax * top_pad
    ax.set_ylim(ymin, ymax)
    return ymin, ymax

# ==============================================================================
# BUILD FIGURE
# ==============================================================================
print("Rendering 3x2 figure...")

fig = plt.figure(figsize=(12, 11.5))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.25, wspace=0.32,
                       left=0.08, right=0.97,
                       top=0.96, bottom=0.06)

axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# -----------------------------------------------------------------------
# (a) Y4 -- Bar: GPP decline rate by region, 3 scenarios
# -----------------------------------------------------------------------
ax = ax_a
reg_slope = regional_summary(m_slope, 'Hist_abs', 'RCP45_Near', 'RCP85_Near')
x = np.arange(len(CLUSTER_NAMES))
w = 0.25
ax.bar(x - w, reg_slope['Hist'],  w, label='Hist.',
       color=C_HIST,  zorder=3)
ax.bar(x,      reg_slope['RCP45'], w, label='S1',
       color=C_RCP45, zorder=3)
ax.bar(x + w,  reg_slope['RCP85'], w, label='S2',
       color=C_RCP85, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(CLUSTER_NAMES, fontsize=FS_TICK)
ax.yaxis.set_tick_params(labelsize=FS_TICK)
ax.set_ylabel('GPP decline rate\n(percentile/pentad)',
              fontsize=FS_LABEL)
set_nonzero_ylim(ax, reg_slope[['Hist','RCP45','RCP85']].values, top_pad=1.3)
ax.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, frameon=False,
          handlelength=1.2, labelspacing=0.3, columnspacing=1.0)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.02, 0.97, panel_labels[0], transform=ax.transAxes,
        fontsize=FS_PANEL, fontweight='bold', ha='left', va='top')

# -----------------------------------------------------------------------
# (b) Y5 -- Full CDF: event GPP decline rate, hist / rcp45mean / rcp85mean
# -----------------------------------------------------------------------
ax = ax_b
from scipy.stats import gaussian_kde as _kde
from scipy.integrate import cumulative_trapezoid as _cumtrap

x_gpp = np.linspace(0, 14, 1000)

def smooth_cdf(vals, xg):
    k   = _kde(vals, bw_method='scott')
    pdf = k(xg)
    c   = _cumtrap(pdf, xg, initial=0)
    return c / c[-1]

cdf_b_hist = smooth_cdf(hist_slopes, x_gpp)
cdf_b_r45  = smooth_cdf(r45_slopes,  x_gpp)
cdf_b_r85  = smooth_cdf(r85_slopes,  x_gpp)

ax.plot(x_gpp, cdf_b_hist, color=C_HIST,  linewidth=1.8, label='Hist.')
ax.plot(x_gpp, cdf_b_r45,  color=C_RCP45, linewidth=1.8, label='S1')
ax.plot(x_gpp, cdf_b_r85,  color=C_RCP85, linewidth=1.8, label='S2')
ax.axhline(0.5, color='gray', linewidth=0.6, linestyle=':', alpha=0.6)

ax.set_xlim([1, 13])
ax.set_ylim([0, 1])
ax.set_xlabel('GPP decline rate (percentile/pentad)',
              fontsize=FS_LABEL)
ax.set_ylabel('Cumulative probability', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=FS_LEGEND, loc='lower right', frameon=False,
          handlelength=1.5, labelspacing=0.3)
ax.text(0.02, 0.97, panel_labels[1], transform=ax.transAxes,
        fontsize=FS_PANEL, fontweight='bold', ha='left', va='top')

# -----------------------------------------------------------------------
# (c) Y6 -- Bar: GPP slope vs onset pentad, 3 scenarios
# -----------------------------------------------------------------------
ax = ax_c
PENTAD_BINS = [1, 2, 3, 4, 5, 6]
MIN_N = 15
stats_y6 = (df_y6.groupby(['intensification_pentads', 'Scenario'])
            .agg(mean_rate=('GPP_Collapse_Rate', 'mean'),
                 n=('GPP_Collapse_Rate', 'count'))
            .reset_index())
stats_y6 = stats_y6[stats_y6['n'] >= MIN_N]

scenarios_y6 = ['Historical', 'RCP4.5 Near', 'RCP8.5 Near']
colors_y6    = {'Historical': C_HIST, 'RCP4.5 Near': C_RCP45, 'RCP8.5 Near': C_RCP85}
legend_y6    = {'Historical': 'Hist.', 'RCP4.5 Near': 'S1', 'RCP8.5 Near': 'S2'}
x6 = np.arange(len(PENTAD_BINS))
w6 = 0.22
n_sc = len(scenarios_y6)

for i, scen in enumerate(scenarios_y6):
    sub = stats_y6[stats_y6['Scenario'] == scen].set_index('intensification_pentads')
    means = [sub.loc[p, 'mean_rate'] if p in sub.index else np.nan
             for p in PENTAD_BINS]
    offset = (i - (n_sc - 1) / 2.0) * w6
    ax.bar(x6 + offset, means, w6,
           label=legend_y6[scen], color=colors_y6[scen],
           zorder=3)

ax.set_xticks(x6)
ax.set_xticklabels(
    [str(p) for p in PENTAD_BINS],
    fontsize=FS_TICK)
ax.yaxis.set_tick_params(labelsize=FS_TICK)
ax.set_xlabel('Flash drought onset duration (pentads)', fontsize=FS_LABEL)
ax.set_ylabel('GPP decline rate\n(percentile/pentad)',
              fontsize=FS_LABEL)
set_nonzero_ylim(ax, stats_y6['mean_rate'].values, top_pad=1.40)
ax.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, frameon=False,
          handlelength=1.2, labelspacing=0.3, columnspacing=1.0)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.02, 0.97, panel_labels[2], transform=ax.transAxes,
        fontsize=FS_PANEL, fontweight='bold', ha='left', va='top')

# -----------------------------------------------------------------------
# (d) Y7 -- Spatial: RCP8.5 mean - Historical G2 abs GPP
# -----------------------------------------------------------------------
ax = ax_d
ax.set_facecolor('#f4f4f4')

if os.path.exists(US_BND_FILE):
    df_us = pd.read_csv(US_BND_FILE, sep=r'\s+', header=None, on_bad_lines='skip')
    ax.scatter(df_us[0][::10], df_us[1][::10],
               color='grey', s=0.3, alpha=0.4, zorder=2)

grid_d = m_g2_d.pivot(index='lat', columns='lon_plot', values='delta_d')
sc = ax.pcolormesh(grid_d.columns, grid_d.index, grid_d.values,
                   cmap='PiYG', vmin=-1, vmax=1,
                   zorder=1, shading='nearest')

for bnd_file, reg_name in zip(BOUNDARY_FILES, REGION_LABELS):
    if os.path.exists(bnd_file):
        df_reg = pd.read_csv(bnd_file, sep=r'\s+', header=None, on_bad_lines='skip')
        ax.scatter(df_reg[1], df_reg[0], color='black', s=0.5, zorder=3)
        label_y_offset = 1.6 if reg_name in ('MW', 'NE') else 0.0
        ax.text(df_reg[1].mean(), df_reg[0].mean() + label_y_offset, reg_name,
                fontsize=FS_LABEL, fontweight='bold', color='black',
                ha='center', va='center', zorder=4,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

ax.set_aspect('auto')
ax.set_xlim([-126, -66])
ax.set_ylim([24, 50])
ax.set_axis_off()

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                    pad=0.01, shrink=0.95, aspect=28, extend='both')
cbar.set_label('Min. GPP diff. under flash drought (gC m$^{-2}$ day$^{-1}$): S2 - hist.',
               fontsize=FS_LABEL)
cbar.ax.tick_params(labelsize=FS_TICK - 1)

ax.text(0.02, 0.03, panel_labels[3], transform=ax.transAxes,
        fontsize=FS_PANEL, fontweight='bold', ha='left', va='bottom',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

# -----------------------------------------------------------------------
# (e) Y8 -- Bar: G2 abs GPP by region, 3 scenarios (NO value labels)
# -----------------------------------------------------------------------
ax = ax_e
reg_g2 = regional_summary(m_g2, 'hist', 'RCP45_mean', 'RCP85_mean')
x8 = np.arange(len(CLUSTER_NAMES))
w8 = 0.25
ax.bar(x8 - w8, reg_g2['Hist'],  w8, label='Hist.',
       color=C_HIST,  zorder=3)
ax.bar(x8,       reg_g2['RCP45'], w8, label='S1',
       color=C_RCP45, zorder=3)
ax.bar(x8 + w8,  reg_g2['RCP85'], w8, label='S2',
       color=C_RCP85, zorder=3)
ax.set_xticks(x8)
ax.set_xticklabels(CLUSTER_NAMES, fontsize=FS_TICK)
ax.yaxis.set_tick_params(labelsize=FS_TICK)
ax.set_ylabel('Min. GPP under flash drought\n(gC m$^{-2}$ day$^{-1}$)',
              fontsize=FS_LABEL)
set_nonzero_ylim(ax, reg_g2[['Hist','RCP45','RCP85']].values, top_pad=1.25)
ax.legend(fontsize=FS_LEGEND, loc='upper right', ncol=3, frameon=False,
          handlelength=1.2, labelspacing=0.3, columnspacing=1.0)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.02, 0.97, panel_labels[4], transform=ax.transAxes,
        fontsize=FS_PANEL, fontweight='bold', ha='left', va='top')

# -----------------------------------------------------------------------
# (f) Y9 -- Full CDF: iWUE, hist / rcp45mean / rcp85mean
# -----------------------------------------------------------------------
ax = ax_f
if len(wue_data) >= 5:
    hist_wue = wue_data['Historical']['WUE_mean'].values
    r45_wue  = np.concatenate([wue_data['RCP4.5 Cooler']['WUE_mean'].values,
                                wue_data['RCP4.5 Hotter']['WUE_mean'].values])
    r85_wue  = np.concatenate([wue_data['RCP8.5 Cooler']['WUE_mean'].values,
                                wue_data['RCP8.5 Hotter']['WUE_mean'].values])

    all_wue = np.concatenate([hist_wue, r45_wue, r85_wue])
    x_wue   = np.linspace(np.percentile(all_wue, 0.5),
                           np.percentile(all_wue, 99.5), 1000)

    cdf_f_hist = smooth_cdf(hist_wue, x_wue)
    cdf_f_r45  = smooth_cdf(r45_wue,  x_wue)
    cdf_f_r85  = smooth_cdf(r85_wue,  x_wue)

    ax.plot(x_wue, cdf_f_hist, color=C_HIST,  linewidth=1.8, label='Hist.')
    ax.plot(x_wue, cdf_f_r45,  color=C_RCP45, linewidth=1.8, label='S1')
    ax.plot(x_wue, cdf_f_r85,  color=C_RCP85, linewidth=1.8, label='S2')
    ax.axhline(0.5, color='gray', linewidth=0.6, linestyle=':', alpha=0.6)

    ax.set_xlim([1, 5])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean inherent water-use efficiency (gC kPa kg$^{-1}$)',
                  fontsize=FS_LABEL)
    ax.set_ylabel('Cumulative probability', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=FS_LEGEND, loc='lower right', frameon=False,
              handlelength=1.5, labelspacing=0.3)
else:
    ax.text(0.5, 0.5, 'WUE data not found\n(check paths)',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_axis_off()

ax.text(0.02, 0.97, panel_labels[5], transform=ax.transAxes,
        fontsize=FS_PANEL, fontweight='bold', ha='left', va='top')

# -----------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------
OUT_DIR = os.path.join(BASE_DIR, 'rcp45_cooler_near')
os.makedirs(OUT_DIR, exist_ok=True)
out_png = os.path.join(OUT_DIR, 'Pager_Figure_3_v4.png')
out_pdf = os.path.join(OUT_DIR, 'Pager_Figure_3_v4.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved:\n  {out_png}\n  {out_pdf}")
plt.show()
print("Done.")