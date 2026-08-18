import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DATA = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus'

SCENARIO_DIRS = {
    'rcp45_hotter':  os.path.join(BASE_DATA, 'rcp45_hotter_near'),
    'rcp45_cooler':  os.path.join(BASE_DATA, 'rcp45_cooler_near'),
    'rcp85_hotter':  os.path.join(BASE_DATA, 'rcp85_hotter_near'),
    'rcp85_cooler':  os.path.join(BASE_DATA, 'rcp85_cooler_near'),
    'ssp345_hotter': os.path.join(BASE_DATA, 'ssp3_rcp45_hotter_near'),
    'ssp345_cooler': os.path.join(BASE_DATA, 'ssp3_rcp45_cooler_near'),
    'ssp585_hotter': os.path.join(BASE_DATA, 'ssp5_rcp85_hotter_near'),
    'ssp585_cooler': os.path.join(BASE_DATA, 'ssp5_rcp85_cooler_near'),
}

LULC_FILES = {
    'hist':          os.path.join(BASE_DATA, 'hist/pft_cft_lulc/lulc_mean.csv'),
    'rcp45_cooler':  os.path.join(BASE_DATA, 'ssp3_rcp45_cooler_near/pft_cft_lulc/lulc_mean.csv'),
    'rcp45_hotter':  os.path.join(BASE_DATA, 'ssp3_rcp45_hotter_near/pft_cft_lulc/lulc_mean.csv'),
    'rcp85_cooler':  os.path.join(BASE_DATA, 'ssp5_rcp85_cooler_near/pft_cft_lulc/lulc_mean.csv'),
    'rcp85_hotter':  os.path.join(BASE_DATA, 'ssp5_rcp85_hotter_near/pft_cft_lulc/lulc_mean.csv'),
}

CSV_TEMPLATE = 'drought_events_fixed_season_cluster_{}.csv'
N_CLUSTERS   = 7

OUTPUT_DIR = os.path.join(BASE_DATA, 'ssp3_rcp45_cooler_near')
os.makedirs(OUTPUT_DIR, exist_ok=True)

US_BND_FILE    = os.path.join(BASE_DATA, 'hist', 'us_coor.txt')
BOUNDARY_FILES = [
    os.path.join(BASE_DATA, 'hist', f)
    for f in ['R1_nw.txt', 'R2_sw.txt', 'R3_ngp.txt',
              'R4_sgp.txt', 'R5_mw.txt', 'R6_se.txt', 'R7_ne.txt']
]

REGION_NAMES = {1: 'NW', 2: 'SW', 3: 'NGP', 4: 'SGP',
                5: 'MW', 6: 'SE', 7: 'NE'}

# PFT/CFT column groups for LULC
GRASS_COLS  = ['pft_12', 'pft_13', 'pft_14']
SHRUB_COLS  = ['pft_09', 'pft_10', 'pft_11']
FOREST_COLS = ['pft_{:02d}'.format(i) for i in range(1, 9)]
CROP_COLS   = ['cft_{}'.format(i) for i in range(15, 79)]

LULC_CATEGORIES = {
    'Grassland': GRASS_COLS,
    'Shrub':     SHRUB_COLS,
    'Forest':    FOREST_COLS,
    'Crop':      CROP_COLS,
}

# Shared colors -- consistent across all bar panels
COLOR_RCP45  = '#2E86AB'
COLOR_SSP345 = '#66C2E8'
COLOR_RCP85  = '#E66101'
COLOR_SSP585 = '#F5A962'

# LULC map colormap
CMAP_LULC = 'PiYG'
LULC_VMAX = 50   # percentage points (+/-50 pp)

# ==============================================================================
# FONT SIZES
# ==============================================================================
FS       = 15
FS_LABEL = 16
FS_TITLE = 16
FS_PANEL = 20
FS_CBAR  = 16

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         FS,
    'axes.labelsize':    FS_LABEL,
    'axes.titlesize':    FS_TITLE,
    'xtick.labelsize':   FS,
    'ytick.labelsize':   FS,
    'legend.fontsize':   FS,
    'axes.linewidth':    1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

# ==============================================================================
# LULC DATA LOADING
# ==============================================================================

def load_lulc(fpath):
    df = pd.read_csv(fpath)
    df['lat'] = df['lat'].round(4)
    df['lon'] = df['lon'].apply(lambda x: x - 360.0 if x > 180.0 else x).round(4)
    if 'cluster' in df.columns:
        df = df.drop(columns=['cluster'])
    return df


def sum_category(df, cols):
    present = [c for c in cols if c in df.columns]
    return df[present].sum(axis=1)


def build_lulc_data():
    """Load LULC files; compute per-cell crop fraction and category means.

    Returns:
        df_map_rcp45 : lat, lon, diff_crop  (RCP4.5 mean - hist, fraction)
        df_map_rcp85 : lat, lon, diff_crop  (RCP8.5 mean - hist, fraction)
        bar_means    : dict scenario -> dict category -> mean pct
    """
    print('Loading LULC files...')
    df_hist      = load_lulc(LULC_FILES['hist'])
    df_r45c      = load_lulc(LULC_FILES['rcp45_cooler'])
    df_r45h      = load_lulc(LULC_FILES['rcp45_hotter'])
    df_r85c      = load_lulc(LULC_FILES['rcp85_cooler'])
    df_r85h      = load_lulc(LULC_FILES['rcp85_hotter'])

    # RCP4.5 mean and RCP8.5 mean (average cooler+hotter)
    def avg_two(dfa, dfb):
        m = dfa.merge(dfb, on=['lat', 'lon'], suffixes=('_a', '_b'))
        result = m[['lat', 'lon']].copy()
        all_cols = (GRASS_COLS + SHRUB_COLS + FOREST_COLS +
                    [c for c in CROP_COLS if c in dfa.columns])
        for c in all_cols:
            ca, cb = c + '_a', c + '_b'
            if ca in m.columns and cb in m.columns:
                result[c] = (m[ca] + m[cb]) / 2.0
            elif ca in m.columns:
                result[c] = m[ca]
        return result

    df_rcp45 = avg_two(df_r45c, df_r45h)
    df_rcp85 = avg_two(df_r85c, df_r85h)

    # Spatial map: cropland fraction change
    def crop_diff(df_fut, df_h):
        dc = pd.DataFrame({'lat': df_h['lat'], 'lon': df_h['lon']})
        dc['crop_hist'] = sum_category(df_h,   CROP_COLS)
        m = dc.merge(
            pd.DataFrame({
                'lat': df_fut['lat'], 'lon': df_fut['lon'],
                'crop_fut': sum_category(df_fut, CROP_COLS)
            }), on=['lat', 'lon'], how='inner')
        m['diff_crop'] = (m['crop_fut'] - m['crop_hist']) * 100.0
        return m[['lat', 'lon', 'diff_crop']]

    df_map_rcp45 = crop_diff(df_rcp45, df_hist)
    df_map_rcp85 = crop_diff(df_rcp85, df_hist)

    # Bar chart: mean category fractions (%) for 4 scenarios + hist
    def cat_means(df):
        return {cat: sum_category(df, cols).mean() * 100.0
                for cat, cols in LULC_CATEGORIES.items()}

    bar_means = {
        'Hist.':     cat_means(df_hist),
        'rcp45mean': cat_means(df_rcp45),
        'rcp85mean': cat_means(df_rcp85),
    }

    print('  LULC map cells rcp45={}, rcp85={}'.format(
        len(df_map_rcp45), len(df_map_rcp85)))
    return df_map_rcp45, df_map_rcp85, bar_means


# ==============================================================================
# DROUGHT DATA LOADING
# ==============================================================================

def load_event_metric(scenario_key, metric_col):
    scenario_dir = SCENARIO_DIRS[scenario_key]
    chunks = []
    for i in range(1, N_CLUSTERS + 1):
        fp = os.path.join(scenario_dir, CSV_TEMPLATE.format(i))
        if not os.path.exists(fp):
            print('  WARNING: missing {}'.format(fp))
            continue
        df = pd.read_csv(fp, usecols=[
            'cell_id_sm', 'lat', 'lon', 'drought_type',
            'intensification_pentads', metric_col
        ])
        df['cluster'] = i
        chunks.append(df)
    if not chunks:
        raise FileNotFoundError(
            'No cluster CSVs found for scenario: {}'.format(scenario_key))
    df_all = pd.concat(chunks, ignore_index=True)
    df_all['lon'] = np.where(df_all['lon'] > 180,
                             df_all['lon'] - 360.0, df_all['lon'])
    df_all = df_all[df_all['drought_type'] == 'Flash'].copy()
    df_all = df_all[df_all['intensification_pentads'] < 6].copy()
    cell_mean = (df_all.groupby(['lat', 'lon', 'cluster'], as_index=False)
                 [metric_col].mean()
                 .rename(columns={metric_col: 'metric'}))
    return cell_mean


def load_flash_ratio_from_clusters(scenario_key):
    scenario_dir = SCENARIO_DIRS[scenario_key]
    chunks = []
    for i in range(1, N_CLUSTERS + 1):
        fp = os.path.join(scenario_dir, CSV_TEMPLATE.format(i))
        if not os.path.exists(fp):
            print('  WARNING: missing {}'.format(fp))
            continue
        df = pd.read_csv(fp, usecols=[
            'cell_id_sm', 'lat', 'lon', 'drought_type',
            'intensification_pentads'
        ])
        df['cluster'] = i
        chunks.append(df)
    if not chunks:
        raise FileNotFoundError(
            'No cluster CSVs found for scenario: {}'.format(scenario_key))
    df_all = pd.concat(chunks, ignore_index=True)
    df_all['lon'] = np.where(df_all['lon'] > 180,
                             df_all['lon'] - 360.0, df_all['lon'])
    df_all['is_flash'] = (
        (df_all['drought_type'] == 'Flash') &
        (df_all['intensification_pentads'] < 6)
    ).astype(int)
    df_all['is_drought'] = 1
    cell = df_all.groupby(['lat', 'lon', 'cluster'], as_index=False).agg(
        n_flash=('is_flash', 'sum'),
        n_total=('is_drought', 'sum')
    )
    cell['metric'] = np.where(
        cell['n_total'] > 0,
        cell['n_flash'] / cell['n_total'] * 100,
        np.nan
    )
    return cell[['lat', 'lon', 'cluster', 'metric']]


def build_drought_row(loader_fn, label):
    print('Loading {}...'.format(label))
    r45h = loader_fn('rcp45_hotter')
    r45c = loader_fn('rcp45_cooler')
    r85h = loader_fn('rcp85_hotter')
    r85c = loader_fn('rcp85_cooler')
    s45h = loader_fn('ssp345_hotter')
    s45c = loader_fn('ssp345_cooler')
    s85h = loader_fn('ssp585_hotter')
    s85c = loader_fn('ssp585_cooler')

    merged = r45h.rename(columns={'metric': 'val_r45h'})

    def add(df, name):
        return merged.merge(
            df[['lat', 'lon', 'cluster', 'metric']].rename(
                columns={'metric': name}),
            on=['lat', 'lon', 'cluster'], how='inner')

    merged = add(r45c, 'val_r45c')
    merged = add(r85h, 'val_r85h')
    merged = add(r85c, 'val_r85c')
    merged = add(s45h, 'val_s45h')
    merged = add(s45c, 'val_s45c')
    merged = add(s85h, 'val_s85h')
    merged = add(s85c, 'val_s85c')

    merged['val_rcp45']  = merged[['val_r45h', 'val_r45c']].mean(axis=1)
    merged['val_ssp345'] = merged[['val_s45h', 'val_s45c']].mean(axis=1)
    merged['val_rcp85']  = merged[['val_r85h', 'val_r85c']].mean(axis=1)
    merged['val_ssp585'] = merged[['val_s85h', 'val_s85c']].mean(axis=1)
    merged['delta_345']  = merged['val_ssp345'] - merged['val_rcp45']
    merged['delta_585']  = merged['val_ssp585'] - merged['val_rcp85']

    print('  Cells: {}  delta_345 [{:.2f}, {:.2f}]  delta_585 [{:.2f}, {:.2f}]'.format(
        len(merged),
        merged['delta_345'].min(), merged['delta_345'].max(),
        merged['delta_585'].min(), merged['delta_585'].max()))
    return merged


# ==============================================================================
# SHARED DRAWING HELPERS
# ==============================================================================

REGION_LABELS = [
    (-120.0, 44.5, 'NW'), (-117.0, 34.5, 'SW'), (-101.0, 46.0, 'NGP'),
    (-100.0, 35.5, 'SGP'), (-90.0, 42.5, 'MW'), (-85.0, 33.0, 'SE'),
    (-74.0,  43.5, 'NE'),
]


def add_panel_label(ax, letter, x=0.02, y=0.03, ha='left', va='bottom'):
    ax.text(x, y, '(' + letter + ')',
            transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold',
            ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='none',
                      edgecolor='none', alpha=0.75),
            zorder=10)


def draw_map_base(ax):
    ax.set_facecolor('#e8e8e8')
    if os.path.exists(US_BND_FILE):
        df_us = pd.read_csv(US_BND_FILE, sep=r'\s+', header=None,
                            on_bad_lines='skip')
        ax.scatter(df_us[0][::10], df_us[1][::10],
                   color='white', s=0.5, alpha=0.9, zorder=2, rasterized=True)
    for bnd_file in BOUNDARY_FILES:
        if os.path.exists(bnd_file):
            df_reg = pd.read_csv(bnd_file, sep=r'\s+', header=None,
                                 on_bad_lines='skip')
            ax.scatter(df_reg[1], df_reg[0],
                       color='black', s=4.5, zorder=3, rasterized=True)
    ax.set_aspect(1.3)
    ax.set_xlim([-126, -66])
    ax.set_ylim([24, 50])
    ax.set_xlabel('Longitude (deg)', fontsize=FS_LABEL)
    ax.set_ylabel('Latitude (deg)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS)
    ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.4)


def draw_delta_map(ax, df, col, title, vmax, cmap='RdBu_r',
                   show_region_labels=False):
    draw_map_base(ax)
    sc = ax.scatter(
        df['lon'], df['lat'],
        c=df[col],
        cmap=cmap, s=18, marker='s',
        vmin=-vmax, vmax=vmax,
        alpha=0.95, zorder=1, rasterized=True
    )
    ax.set_title(title, fontsize=FS_TITLE, pad=5)
    if show_region_labels:
        for lon_c, lat_c, label in REGION_LABELS:
            ax.text(lon_c, lat_c, label,
                    fontsize=13, fontweight='bold', color='black',
                    ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='none', alpha=0.55))
    return sc


def attach_colorbar(fig, sc, ax_left, ax_right, label, vmax=None,
                    y_offset=0.038):
    fig.canvas.draw()
    p0 = ax_left.get_position()
    p1 = ax_right.get_position()
    full_w = p1.x1 - p0.x0
    cbar_w = full_w * 0.60
    cbar_x = p0.x0 + (full_w - cbar_w) / 2.0
    cax = fig.add_axes([cbar_x, p0.y0 - y_offset, cbar_w, 0.009])
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='both')
    cbar.set_label(label, fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_CBAR - 1)
    if vmax is not None:
        tick_step = max(2, int(np.ceil(vmax / 4 / 2)) * 2)
        cbar.set_ticks(np.arange(-vmax, vmax + 1, tick_step))
    return cbar


def set_bar_ylim(ax, val_arrays, pad_frac=0.12):
    """Zoom the y-axis to the data range instead of forcing ymin=0,
    so differences between bars are easier to see."""
    all_vals = np.concatenate([np.asarray(v).ravel() for v in val_arrays])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin, vmax = all_vals.min(), all_vals.max()
    rng = vmax - vmin
    if rng == 0:
        rng = abs(vmax) if vmax != 0 else 1.0
    pad = rng * pad_frac
    ax.set_ylim(vmin - pad, vmax + pad)


def draw_drought_bar(ax, df, ylabel, show_legend=False, ymax=None):
    """Grouped bar for drought rows: 4 scenarios with triangle markers
    showing SSP - RCP delta above each RCP/SSP pair."""
    regions  = sorted(df['cluster'].unique())
    n_groups = 1 + len(regions)
    x        = np.arange(n_groups)
    bw       = 0.18
    offsets  = [-1.5*bw, -0.5*bw, 0.5*bw, 1.5*bw]

    bar_specs = [
        ('val_rcp45',  COLOR_RCP45,  'S1'),
        ('val_ssp345', COLOR_SSP345, 'S3'),
        ('val_rcp85',  COLOR_RCP85,  'S2'),
        ('val_ssp585', COLOR_SSP585, 'S4'),
    ]

    all_vals = []
    for i, (col, color, label) in enumerate(bar_specs):
        conus_mean   = df[col].mean()
        region_means = [df[df['cluster'] == r][col].mean() for r in regions]
        vals = np.array([conus_mean] + region_means)
        all_vals.append(vals)
        ax.bar(x + offsets[i], vals, width=bw,
               color=color, alpha=0.88,
               edgecolor='white', linewidth=0.5,
               label=label, zorder=3)

    ax.axvline(x=0.5, color='gray', linewidth=0.8,
               linestyle='--', alpha=0.6, zorder=2)

    xlabels = ['CONUS'] + [REGION_NAMES[r] for r in regions]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    set_bar_ylim(ax, all_vals)

    if ymax is not None:
        ymin_cur = ax.get_ylim()[0]
        ax.set_ylim(ymin_cur, ymax)

    if show_legend:
        ax.legend(frameon=False, fontsize=FS - 1,
                  loc='upper center', bbox_to_anchor=(0.65, 1.0),
                  handlelength=1.2, handleheight=0.9, ncol=2)


def draw_lulc_bar(ax, bar_means, show_legend=False):
    """Grouped bar for LULC row: Hist / RCP4.5 / RCP8.5 with
    triangle markers (same style as drought bars)."""
    categories = list(LULC_CATEGORIES.keys())
    scenarios  = ['Hist.', 'rcp45mean', 'rcp85mean']
    colors     = [COLOR_RCP45,  COLOR_RCP45,  COLOR_RCP85]
    # Use hist as light blue (same as SSP345 in panel f), RCP45 dark blue, RCP85 dark orange
    colors     = ['#888780', COLOR_SSP345, COLOR_SSP585]
    labels     = ['Hist/S1/S2', 'S3', 'S4']

    x   = np.arange(len(categories))
    bw  = 0.22
    offsets = [-bw, 0.0, bw]

    bar_tops = {}
    for i, (scen, color, lbl) in enumerate(zip(scenarios, colors, labels)):
        vals = np.array([bar_means[scen][cat] for cat in categories])
        ax.bar(x + offsets[i], vals, width=bw,
               color=color, alpha=0.88,
               edgecolor='white', linewidth=0.5,
               label=lbl, zorder=3)
        bar_tops[scen] = vals

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=FS)
    ax.set_ylabel('Mean CONUS land cover fraction (%)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    set_bar_ylim(ax, list(bar_tops.values()))

    if show_legend:
        ax.legend(frameon=False, fontsize=FS - 1,
                  loc='upper center', handlelength=1.2,
                  handleheight=0.9, ncol=1)


def vmax_from(df, c1='delta_345', c2='delta_585'):
    vals = pd.concat([df[c1].abs(), df[c2].abs()]).dropna()
    return np.ceil(np.percentile(vals, 98) / 5) * 5


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # ---------- Load all data ----------
    df_lulc_rcp45, df_lulc_rcp85, lulc_bar = build_lulc_data()
    df_ratio   = build_drought_row(load_flash_ratio_from_clusters,
                                   'flash drought frequency ratio')
    df_sev     = build_drought_row(lambda k: load_event_metric(k, 'severity'),
                                   'flash drought severity')
    df_onset   = build_drought_row(lambda k: load_event_metric(k, 'decline_rate'),
                                   'flash drought onset speed')

    # Colorbar limits
    vmax_lulc  = LULC_VMAX
    vmax_ratio = vmax_from(df_ratio)
    vmax_sev   = vmax_from(df_sev)
    vmax_onset = 8
    print('vmaxes: lulc={:.2f}  ratio={:.0f}  sev={:.0f}  onset={:.0f}'.format(
        vmax_lulc, vmax_ratio, vmax_sev, vmax_onset))

    # ===========================================================================
    # Figure: 4 rows x 3 cols
    #   col 0: map A (RCP4.5 or SSP3-4.5 delta)
    #   col 1: map B (RCP8.5 or SSP5-8.5 delta)
    #   col 2: bar chart
    # ===========================================================================
    fig = plt.figure(figsize=(26, 28))
    gs  = gridspec.GridSpec(
        4, 3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.85],
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.44,
        wspace=0.28,
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(3)]
    (ax_a, ax_b, ax_c,
     ax_d, ax_e, ax_f,
     ax_g, ax_h, ax_i,
     ax_j, ax_k, ax_l) = axes

    bar_axes = {ax_c, ax_f, ax_i, ax_l}
    for ax, letter in zip(axes, 'abcdefghijkl'):
        if ax in bar_axes:
            add_panel_label(ax, letter, x=0.02, y=0.97, ha='left', va='top')
        else:
            add_panel_label(ax, letter, x=0.02, y=0.03, ha='left', va='bottom')

    # ---- Row 1: LULC cropland change ----------------------------------------
    sc_lulc1 = draw_delta_map(
        ax_a, df_lulc_rcp45, 'diff_crop',
        'S3 - Hist/S1',
        vmax_lulc, cmap=CMAP_LULC, show_region_labels=True)
    sc_lulc2 = draw_delta_map(
        ax_b, df_lulc_rcp85, 'diff_crop',
        'S4 - Hist/S2',
        vmax_lulc, cmap=CMAP_LULC)
    attach_colorbar(fig, sc_lulc1, ax_a, ax_b,
                    'Change in cropland fraction (future - hist., %)',
                    vmax=None)   # symmetric but no fixed step needed
    draw_lulc_bar(ax_c, lulc_bar, show_legend=True)

    # ---- Row 2: flash drought frequency ratio --------------------------------
    sc_r = draw_delta_map(
        ax_d, df_ratio, 'delta_345',
        'S3 - S1',
        vmax_ratio)
    draw_delta_map(
        ax_e, df_ratio, 'delta_585',
        'S4 - S2',
        vmax_ratio)
    attach_colorbar(fig, sc_r, ax_d, ax_e,
                    'Change in flash drought frequency ratio (%)',
                    vmax=vmax_ratio)
    draw_drought_bar(ax_f, df_ratio,
                     ylabel='Flash drought ratio (%)',
                     show_legend=True, ymax=85)

    # ---- Row 3: flash drought severity ---------------------------------------
    sc_s = draw_delta_map(
        ax_g, df_sev, 'delta_345',
        'S3 - S1',
        vmax_sev)
    draw_delta_map(
        ax_h, df_sev, 'delta_585',
        'S4 - S2',
        vmax_sev)
    attach_colorbar(fig, sc_s, ax_g, ax_h,
                    'Change in mean flash drought severity'
                    ' (percentile deficit)',
                    vmax=vmax_sev)
    draw_drought_bar(ax_i, df_sev,
                     ylabel='Mean flash drought severity\n(percentile deficit)',
                     show_legend=True)

    # ---- Row 4: flash drought onset speed ------------------------------------
    sc_o = draw_delta_map(
        ax_j, df_onset, 'delta_345',
        'S3 - S1',
        vmax_onset)
    draw_delta_map(
        ax_k, df_onset, 'delta_585',
        'S4 - S2',
        vmax_onset)
    attach_colorbar(fig, sc_o, ax_j, ax_k,
                    'Change in mean flash drought onset speed'
                    ' (percentile/pentad)',
                    vmax=vmax_onset)
    draw_drought_bar(ax_l, df_onset,
                     ylabel='Mean flash drought onset speed\n'
                            '(percentile/pentad)',
                     show_legend=True)

    # Save
    out_pdf = os.path.join(OUTPUT_DIR, 'paper_figure_4.pdf')
    out_png = os.path.join(OUTPUT_DIR, 'paper_figure_4.png')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    print('Saved:\n  ' + out_pdf + '\n  ' + out_png)
    plt.show()


if __name__ == '__main__':
    main()