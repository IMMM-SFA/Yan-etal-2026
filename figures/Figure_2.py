import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DATA = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus'

SCENARIO_DIRS = {
    'hist':         os.path.join(BASE_DATA, 'hist'),
    'rcp45_hotter': os.path.join(BASE_DATA, 'rcp45_hotter_near'),
    'rcp45_cooler': os.path.join(BASE_DATA, 'rcp45_cooler_near'),
    'rcp85_hotter': os.path.join(BASE_DATA, 'rcp85_hotter_near'),
    'rcp85_cooler': os.path.join(BASE_DATA, 'rcp85_cooler_near'),
}

CSV_FILE     = 'all_drought_frequencies_CONUS.csv'
CSV_TEMPLATE = 'drought_events_fixed_season_cluster_{}.csv'
N_CLUSTERS   = 7

OUTPUT_DIR = os.path.join(BASE_DATA, 'rcp45_cooler_near')
os.makedirs(OUTPUT_DIR, exist_ok=True)

US_BND_FILE    = os.path.join(BASE_DATA, 'hist', 'us_coor.txt')
BOUNDARY_FILES = [
    os.path.join(BASE_DATA, 'hist', f)
    for f in ['R1_nw.txt', 'R2_sw.txt', 'R3_ngp.txt',
              'R4_sgp.txt', 'R5_mw.txt', 'R6_se.txt', 'R7_ne.txt']
]

REGION_NAMES = {1: 'NW', 2: 'SW', 3: 'NGP', 4: 'SGP',
                5: 'MW', 6: 'SE', 7: 'NE'}

COLOR_HIST  = '#888780'
COLOR_RCP45 = '#2E86AB'
COLOR_RCP85 = '#E66101'

# ==============================================================================
# FONT SIZES  -- all unified, larger
# ==============================================================================
FS       = 15   # tick labels, legend
FS_LABEL = 16   # axis labels
FS_TITLE = 16   # panel titles
FS_PANEL = 20   # (a)(b)... labels
FS_CBAR  = 16   # colorbar label, same as axis labels

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
# DATA LOADING
# ==============================================================================

def load_ratio(scenario_key):
    """Row 1: flash drought frequency ratio from all_drought_frequencies_CONUS.csv."""
    fp = os.path.join(SCENARIO_DIRS[scenario_key], CSV_FILE)
    df = pd.read_csv(fp)
    df['lon'] = np.where(df['lon'] > 180, df['lon'] - 360.0, df['lon'])
    df['flash_ratio'] = np.where(
        df['total_drought_frequency'] > 0,
        df['flash_drought_frequency'] / df['total_drought_frequency'] * 100,
        np.nan
    )
    return df[['lat', 'lon', 'cluster', 'flash_ratio']]


def load_event_metric(scenario_key, metric_col):
    """Rows 2-3: per-event metric from cluster CSVs.
    Keeps Flash drought only, removes onset > 30 days (pentads >= 6).
    Returns mean metric per cell."""
    scenario_dir = SCENARIO_DIRS[scenario_key]
    chunks = []
    for i in range(1, N_CLUSTERS + 1):
        fp = os.path.join(scenario_dir, CSV_TEMPLATE.format(i))
        if not os.path.exists(fp):
            print("  WARNING: missing {}".format(fp))
            continue
        df = pd.read_csv(fp, usecols=[
            'cell_id_sm', 'lat', 'lon', 'drought_type',
            'intensification_pentads', metric_col
        ])
        df['cluster'] = i
        chunks.append(df)
    if not chunks:
        raise FileNotFoundError(
            "No cluster CSVs found for scenario: {}".format(scenario_key))
    df_all = pd.concat(chunks, ignore_index=True)
    df_all['lon'] = np.where(df_all['lon'] > 180,
                             df_all['lon'] - 360.0, df_all['lon'])
    # Flash only, onset <= 30 days
    df_all = df_all[df_all['drought_type'] == 'Flash'].copy()
    df_all = df_all[df_all['intensification_pentads'] < 6].copy()
    cell_mean = (df_all.groupby(['lat', 'lon', 'cluster'], as_index=False)
                 [metric_col].mean()
                 .rename(columns={metric_col: 'metric'}))
    return cell_mean


def build_row1():
    """Flash drought frequency ratio (%)."""
    print("Row 1: loading flash drought frequency ratio...")
    hist         = load_ratio('hist')
    rcp45_hotter = load_ratio('rcp45_hotter')
    rcp45_cooler = load_ratio('rcp45_cooler')
    rcp85_hotter = load_ratio('rcp85_hotter')
    rcp85_cooler = load_ratio('rcp85_cooler')

    merged = hist.rename(columns={'flash_ratio': 'val_hist'})

    def add(df, name):
        return merged.merge(
            df[['lat', 'lon', 'cluster', 'flash_ratio']].rename(
                columns={'flash_ratio': name}),
            on=['lat', 'lon', 'cluster'], how='left')

    merged = add(rcp45_hotter, 'val_r45h')
    merged = add(rcp45_cooler, 'val_r45c')
    merged = add(rcp85_hotter, 'val_r85h')
    merged = add(rcp85_cooler, 'val_r85c')
    merged['val_rcp45'] = merged[['val_r45h', 'val_r45c']].mean(axis=1)
    merged['val_rcp85'] = merged[['val_r85h', 'val_r85c']].mean(axis=1)
    merged['delta_rcp45'] = merged['val_rcp45'] - merged['val_hist']
    merged['delta_rcp85'] = merged['val_rcp85'] - merged['val_hist']
    return merged


def build_row_event(metric_col, label):
    """Generic loader for per-event metrics (severity, decline_rate)."""
    print("Loading {}...".format(label))

    def load(key):
        return load_event_metric(key, metric_col)

    hist         = load('hist')
    rcp45_hotter = load('rcp45_hotter')
    rcp45_cooler = load('rcp45_cooler')
    rcp85_hotter = load('rcp85_hotter')
    rcp85_cooler = load('rcp85_cooler')

    merged = hist.rename(columns={'metric': 'val_hist'})

    def add(df, name):
        return merged.merge(
            df[['lat', 'lon', 'cluster', 'metric']].rename(
                columns={'metric': name}),
            on=['lat', 'lon', 'cluster'], how='left')

    merged = add(rcp45_hotter, 'val_r45h')
    merged = add(rcp45_cooler, 'val_r45c')
    merged = add(rcp85_hotter, 'val_r85h')
    merged = add(rcp85_cooler, 'val_r85c')
    merged['val_rcp45'] = merged[['val_r45h', 'val_r45c']].mean(axis=1)
    merged['val_rcp85'] = merged[['val_r85h', 'val_r85c']].mean(axis=1)
    merged['delta_rcp45'] = merged['val_rcp45'] - merged['val_hist']
    merged['delta_rcp85'] = merged['val_rcp85'] - merged['val_hist']
    return merged


# ==============================================================================
# HELPERS
# ==============================================================================

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


REGION_LABELS = [
    (-120.0, 44.5, 'NW'),
    (-117.0, 34.5, 'SW'),
    (-101.0, 46.0, 'NGP'),
    (-100.0, 35.5, 'SGP'),
    ( -90.0, 42.5, 'MW'),
    ( -85.0, 33.0, 'SE'),
    ( -74.0, 43.5, 'NE'),
]


def draw_delta_map(ax, df, col, title, vmax, show_region_labels=False):
    draw_map_base(ax)
    sc = ax.scatter(
        df['lon'], df['lat'],
        c=df[col],
        cmap='RdBu_r', s=18, marker='s',
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


def attach_colorbar(fig, sc, ax_left, ax_right, label, vmax=None):
    """Attach a compact horizontal colorbar beneath two map axes."""
    fig.canvas.draw()
    p0 = ax_left.get_position()
    p1 = ax_right.get_position()
    # shrink to 60% of the combined width, centered
    full_w = p1.x1 - p0.x0
    cbar_w = full_w * 0.60
    cbar_x = p0.x0 + (full_w - cbar_w) / 2.0
    cax = fig.add_axes([cbar_x, p0.y0 - 0.038, cbar_w, 0.009])
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='both')
    cbar.set_label(label, fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_CBAR - 1)
    if vmax is not None:
        tick_step = max(2, int(np.ceil(vmax / 4 / 2)) * 2)
        cbar.set_ticks(np.arange(-vmax, vmax + 1, tick_step))
    return cbar


def draw_bar_panel(ax, df, bar_cols, bar_labels, ylabel,
                   show_legend=False):
    """Grouped bar: CONUS + 7 regions, 3 scenarios."""
    regions  = sorted(df['cluster'].unique())
    x        = np.arange(1 + len(regions))
    bw       = 0.24
    offsets  = [-bw, 0.0, bw]
    colors   = [COLOR_HIST, COLOR_RCP45, COLOR_RCP85]

    all_vals = []
    for i, (col, color, label) in enumerate(
            zip(bar_cols, colors, bar_labels)):
        conus_mean   = df[col].mean()
        region_means = [df[df['cluster'] == r][col].mean() for r in regions]
        vals = [conus_mean] + region_means
        all_vals.extend(vals)
        ax.bar(x + offsets[i], vals, width=bw,
               color=color, alpha=0.88,
               edgecolor='white', linewidth=0.5,
               label=label, zorder=3)

    # Zoom the y-axis to the data range instead of forcing a 0 baseline,
    # so bar-to-bar differences are easier to see. Bottom is clipped to
    # the true minimum of the bars (bars still start visually at the
    # axis floor). Extra headroom is left above the tallest bar so the
    # in-axes legend (below) has clear space and never overlaps a bar.
    data_min, data_max = min(all_vals), max(all_vals)
    data_range = data_max - data_min
    pad_bottom = data_range * 0.12 if data_range > 0 else data_max * 0.1
    pad_top    = data_range * 0.22 if data_range > 0 else data_max * 0.15
    y_bottom = max(0.0, data_min - pad_bottom)
    y_top    = data_max + pad_top
    ax.set_ylim(y_bottom, y_top)

    ax.axvline(x=0.5, color='gray', linewidth=0.8,
               linestyle='--', alpha=0.6, zorder=2)

    xlabels = ['CONUS'] + [REGION_NAMES[r] for r in regions]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=FS)
    ax.set_xlabel('', fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_legend:
        # Sit just inside the top of the axes (roughly level with the
        # (c)/(f)/(i) panel-letter labels) rather than floating above
        # the axes, using the extra top headroom reserved above.
        ax.legend(frameon=False, fontsize=FS, loc='upper center',
                  bbox_to_anchor=(0.5, 1.0), ncol=3,
                  handlelength=1.2, handleheight=0.9,
                  columnspacing=1.2, borderaxespad=0.3)


# ==============================================================================
# MAIN
# ==============================================================================

def vmax_from(df):
    """Shared symmetric colorbar limit: 98th pct of |delta|, rounded to 5."""
    vals = pd.concat([df['delta_rcp45'].abs(),
                      df['delta_rcp85'].abs()]).dropna()
    return np.ceil(np.percentile(vals, 98) / 5) * 5


def main():
    # Load all three row datasets
    df1 = build_row1()
    df2 = build_row_event('severity',    'flash drought severity')
    df3 = build_row_event('decline_rate','flash drought SM decline rate')

    vmax1 = 30   # fixed +/-30 for flash drought frequency ratio
    vmax2 = vmax_from(df2)
    vmax3 = 8    # fixed +/-8 for SM decline rate
    print("  vmax row1={:.0f}  row2={:.0f}  row3={:.0f}".format(
        vmax1, vmax2, vmax3))

    # ===========================================================================
    # Figure: 3 rows x 3 columns
    #   col 0: RCP4.5 delta map
    #   col 1: RCP8.5 delta map
    #   col 2: regional bar chart
    # ===========================================================================
    fig = plt.figure(figsize=(26, 21))
    gs  = gridspec.GridSpec(
        3, 3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.85],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.42,   # extra vertical space for colorbars between rows
        wspace=0.28,
    )

    panels = []
    for row in range(3):
        for col in range(3):
            panels.append(fig.add_subplot(gs[row, col]))

    # Unpack axes  (a)..(i)
    (ax_a, ax_b, ax_c,
     ax_d, ax_e, ax_f,
     ax_g, ax_h, ax_i) = panels

    # Map panels: bottom-left; bar panels (c,f,i): top-left
    panel_letters = ['a','b','c','d','e','f','g','h','i']
    bar_panel_set = {ax_c, ax_f, ax_i}
    for ax, letter in zip(panels, panel_letters):
        if ax in bar_panel_set:
            add_panel_label(ax, letter,
                            x=0.02, y=0.90, ha='left', va='top')
        else:
            add_panel_label(ax, letter,
                            x=0.02, y=0.03, ha='left', va='bottom')

    # ---- Row 1: flash drought frequency ratio --------------------------------
    sc1 = draw_delta_map(ax_a, df1, 'delta_rcp45',
                         'S1 - Hist.', vmax1,
                         show_region_labels=True)
    draw_delta_map(ax_b, df1, 'delta_rcp85',
                   'S2 - Hist.', vmax1)
    attach_colorbar(fig, sc1, ax_a, ax_b,
                    'Change in flash drought frequency ratio (future - historical, %)',
                    vmax=vmax1)
    draw_bar_panel(ax_c, df1,
                   ['val_hist',  'val_rcp45', 'val_rcp85'],
                   ['Hist.', 'S1', 'S2'],
                   ylabel='Flash drought ratio (%)',
                   show_legend=True)

    # ---- Row 2: flash drought severity ---------------------------------------
    sc2 = draw_delta_map(ax_d, df2, 'delta_rcp45',
                         'S1 - Hist.', vmax2)
    draw_delta_map(ax_e, df2, 'delta_rcp85',
                   'S2 - Hist.', vmax2)
    attach_colorbar(fig, sc2, ax_d, ax_e,
                    'Change in mean flash drought severity'
                    ' (future - historical, percentile deficit)',
                    vmax=vmax2)
    draw_bar_panel(ax_f, df2,
                   ['val_hist',  'val_rcp45', 'val_rcp85'],
                   ['Hist.', 'S1', 'S2'],
                   ylabel='Mean flash drought severity\n(percentile deficit)',
                   show_legend=True)

    # ---- Row 3: SM decline rate ----------------------------------------------
    sc3 = draw_delta_map(ax_g, df3, 'delta_rcp45',
                         'S1 - Hist.', vmax3)
    draw_delta_map(ax_h, df3, 'delta_rcp85',
                   'S2 - Hist.', vmax3)
    attach_colorbar(fig, sc3, ax_g, ax_h,
                    'Change in mean flash drought onset speed'
                    ' (future - historical, percentile/pentad)',
                    vmax=vmax3)
    draw_bar_panel(ax_i, df3,
                   ['val_hist',  'val_rcp45', 'val_rcp85'],
                   ['Hist.', 'S1', 'S2'],
                   ylabel='Mean flash drought onset speed\n(percentile/pentad)',
                   show_legend=True)

    # Save
    out_pdf = os.path.join(OUTPUT_DIR, 'Figure_flash_drought_3x3.pdf')
    out_png = os.path.join(OUTPUT_DIR, 'Figure_flash_drought_3x3.png')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    print("Saved:\n  " + out_pdf + "\n  " + out_png)
    plt.show()


if __name__ == '__main__':
    main()