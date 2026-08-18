import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import glob

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = './'
OUTPUT_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_TEMPLATE     = 'drought_events_fixed_season_cluster_{}.csv'
US_BND_FILE      = 'us_coor.txt'
BOUNDARY_FILES   = ['R1_nw.txt', 'R2_sw.txt', 'R3_ngp.txt',
                    'R4_sgp.txt', 'R5_mw.txt', 'R6_se.txt', 'R7_ne.txt']

GPP_VALID_CELLS  = 'valid_conus_flash_drought_cells.csv'
GPP_FILE_PATTERN = 'drought_events_fixed_season_cluster_*_with_GPP_Zhang2025.csv'
GPP_COLS         = ['cell_id_sm', 'drought_type', 'GPP_slope', 'intensification_pentads']

COLOR_FLASH = '#E66101'
COLOR_SLOW  = '#5E3C99'

EBM_DATA = [
    ('GS Precip',     8.797610, 'clim'),
    ('GS Solar',      5.993224, 'clim'),
    ('Aridity',       5.743810, 'clim'),
    ('Soil Sand %',   5.113836, 'soil'),
    ('Forest Cover',  4.489470, 'land'),
    ('T/ET Ratio',    4.285789, 'bio'),
    ('GPP 95th',      4.051611, 'bio'),
    ('Crop Cover',    4.049665, 'land'),
    ('VPD 95th',      3.938898, 'atmos'),
    ('Soil Clay %',   3.832922, 'soil'),
]
EBM_CATEGORY_COLORS = {
    'clim':  '#7B68B5',  # Climatology
    'atmos': '#2E86AB',  # Atmospheric Extremes
    'land':  '#C0392B',  # Land Cover
    'bio':   '#1D9E75',  # Vegetation Biophysics
    'soil':  '#C16A2F',  # Soil Physics
}

# ==============================================================================
# FONT SIZES
# ==============================================================================
FS       = 13
FS_LABEL = 14
FS_PANEL = 15

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         FS,
    'axes.labelsize':    FS_LABEL,
    'axes.titlesize':    FS_LABEL,
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

def load_spatial_data():
    print("Loading spatial cluster CSV files...")
    dfs = []
    for i in range(1, 8):
        fp = os.path.join(BASE_DIR, CSV_TEMPLATE.format(i))
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df['cluster'] = i
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No spatial CSV files found.")
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['lon'] = np.where(df_all['lon'] > 180, df_all['lon'] - 360.0, df_all['lon'])
    grouped = df_all.groupby(['lat', 'lon', 'cluster'])
    counts  = grouped['drought_type'].value_counts().unstack(fill_value=0)
    if 'Slow'  not in counts.columns: counts['Slow']  = 0
    if 'Flash' not in counts.columns: counts['Flash'] = 0
    counts['Total']           = counts['Flash'] + counts['Slow']
    counts['Flash_Ratio_Pct'] = np.where(
        counts['Total'] > 0, counts['Flash'] / counts['Total'] * 100, np.nan
    )
    return counts.reset_index()[['lat', 'lon', 'cluster', 'Flash_Ratio_Pct']]


def load_gpp_data():
    print("Loading GPP data...")
    try:
        valid_cells = pd.read_csv(GPP_VALID_CELLS)
        valid_ids   = set(valid_cells['cell'].unique())
        all_files   = glob.glob(GPP_FILE_PATTERN)
        chunks = []
        for f in all_files:
            tmp = pd.read_csv(f, usecols=GPP_COLS)
            tmp = tmp[tmp['cell_id_sm'].isin(valid_ids)]
            tmp = tmp[tmp['GPP_slope'] < 0].copy()
            tmp = tmp[~((tmp['drought_type'] == 'Flash') &
                        (tmp['intensification_pentads'] >= 6))]
            tmp['GPP_Collapse_Rate'] = tmp['GPP_slope'].abs()
            chunks.append(tmp)
        df = pd.concat(chunks, ignore_index=True)
        print(f"  GPP valid events: {len(df)}")
        return df
    except Exception as e:
        print(f"  GPP loading error: {e}")
        return None


# ==============================================================================
# HELPER
# ==============================================================================

def add_panel_label(ax, letter, x, y, ha='left', va='bottom'):
    ax.text(x, y, '(' + letter + ')',
            transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold',
            ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='none',
                      edgecolor='none', alpha=0.75),
            zorder=10)


# ==============================================================================
# PANEL DRAWING
# ==============================================================================

def draw_panel_a(ax, spatial_df):
    """CONUS map. Colorbar attached directly below the map axes via
    make_axes_locatable so no dead whitespace is created."""
    ax.set_facecolor('#e8e8e8')

    # Data dots first (zorder=1)
    sc = ax.scatter(
        spatial_df['lon'], spatial_df['lat'],
        c=spatial_df['Flash_Ratio_Pct'],
        cmap='RdBu_r', s=18, marker='s',
        vmin=0, vmax=100, alpha=0.95,
        zorder=1, rasterized=True
    )

    # State boundaries: white dots (contrast on RdBu_r) - Y1 column order
    us_bnd_path = os.path.join(BASE_DIR, US_BND_FILE)
    if os.path.exists(us_bnd_path):
        df_us = pd.read_csv(us_bnd_path, sep=r'\s+', header=None, on_bad_lines='skip')
        ax.scatter(df_us[0][::10], df_us[1][::10],
                   color='white', s=0.5, alpha=0.9, zorder=2, rasterized=True)

    # Regional boundaries: black, larger - Y1 column order
    for bnd_file in BOUNDARY_FILES:
        bnd_path = os.path.join(BASE_DIR, bnd_file)
        if os.path.exists(bnd_path):
            df_reg = pd.read_csv(bnd_path, sep=r'\s+', header=None, on_bad_lines='skip')
            ax.scatter(df_reg[1], df_reg[0],
                       color='black', s=4.5, zorder=3, rasterized=True)

    ax.set_aspect(1.3)
    ax.set_xlim([-126, -66])
    ax.set_ylim([24, 50])
    ax.set_xlabel('Longitude (deg)', fontsize=FS_LABEL)
    ax.set_ylabel('Latitude (deg)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS)
    ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.4)

    # Colorbar: short horizontal, centered under the map, not full width
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                        pad=0.12, shrink=0.45, aspect=25, extend='neither')
    cbar.set_label('Flash drought frequency ratio (%)', fontsize=FS_LABEL - 1)
    cbar.ax.tick_params(labelsize=FS - 2)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.set_xticklabels(['0 (Slow only)', '25', '50 (Equal)', '75', '100 (Flash only)'],
                             fontsize=FS - 2)

    # Region labels - approximate center of each climate region
    # (lon, lat, label)
    region_labels = [
        (-120.0, 44.5, 'NW'),   # R1 Northwest
        (-117.0, 34.5, 'SW'),   # R2 Southwest
        (-101.0, 46.0, 'NGP'),  # R3 Northern Great Plains
        (-100.0, 35.5, 'SGP'),  # R4 Southern Great Plains
        ( -90.0, 42.5, 'MW'),   # R5 Midwest
        ( -85.0, 33.0, 'SE'),   # R6 Southeast
        ( -74.0, 43.5, 'NE'),   # R7 Northeast
    ]
    for lon_c, lat_c, label in region_labels:
        ax.text(lon_c, lat_c, label,
                fontsize=13, fontweight='bold', color='black',
                ha='center', va='center', zorder=5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='none', alpha=0.55))

    add_panel_label(ax, 'a', x=0.02, y=0.03, ha='left', va='bottom')
    return sc


def draw_panel_b(ax):
    terms  = [d[0] for d in EBM_DATA]
    values = [d[1] for d in EBM_DATA]
    cats   = [d[2] for d in EBM_DATA]
    colors = [EBM_CATEGORY_COLORS[c] for c in cats]

    y_pos = np.arange(len(terms))
    bars  = ax.barh(y_pos, values, color=colors, edgecolor='white',
                    linewidth=0.5, height=0.68, alpha=0.88)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left', fontsize=FS,
                color='#222222')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=FS)
    ax.invert_yaxis()
    ax.set_xlabel('Relative importance (%)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS)
    ax.set_xlim(0, 12)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor=EBM_CATEGORY_COLORS['clim'],  label='Climatology',           alpha=0.88),
        mpatches.Patch(facecolor=EBM_CATEGORY_COLORS['atmos'], label='Atmospheric extremes',  alpha=0.88),
        mpatches.Patch(facecolor=EBM_CATEGORY_COLORS['land'],  label='Land cover',            alpha=0.88),
        mpatches.Patch(facecolor=EBM_CATEGORY_COLORS['bio'],   label='Vegetation biophysics', alpha=0.88),
        mpatches.Patch(facecolor=EBM_CATEGORY_COLORS['soil'],  label='Soil physics',          alpha=0.88),
    ]
    ax.legend(handles=legend_elements, fontsize=FS, frameon=False,
              loc='lower right', handlelength=1.2, handleheight=0.9)

    add_panel_label(ax, 'b', x=0.97, y=0.97, ha='right', va='top')


def draw_panel_c(ax, gpp_df):
    if gpp_df is None:
        ax.text(0.5, 0.5, 'Data not available', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return

    for dtype, color in [('Flash', COLOR_FLASH), ('Slow', COLOR_SLOW)]:
        subset = gpp_df[gpp_df['drought_type'] == dtype]['GPP_Collapse_Rate']
        sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.35,
                    linewidth=0, clip=(0, 20), label=dtype, common_norm=False)

    ax.set_xlim(0, 20)
    ax.set_xlabel('GPP decline rate (percentile/pentad)', fontsize=FS_LABEL)
    ax.set_ylabel('Probability density', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=FS, loc='upper right')

    add_panel_label(ax, 'c', x=0.03, y=0.97, ha='left', va='top')


def draw_panel_d(ax, gpp_df):
    if gpp_df is None:
        ax.text(0.5, 0.5, 'Data not available', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return

    bins   = [1, 3, 5, 7, 9, 11, np.inf]
    labels = ['1-2', '3-4', '5-6', '7-8', '9-10', '11+']
    gpp_df = gpp_df.copy()
    gpp_df['Onset_Bin'] = pd.cut(gpp_df['intensification_pentads'],
                                  bins=bins, labels=labels, right=False)
    stats = (gpp_df.groupby(['Onset_Bin', 'drought_type'], observed=True)['GPP_Collapse_Rate']
             .mean().reset_index())

    bar_width = 0.35
    x         = np.arange(len(labels))

    for i, (dtype, color) in enumerate(zip(['Flash', 'Slow'], [COLOR_FLASH, COLOR_SLOW])):
        subset = stats[stats['drought_type'] == dtype]
        vals = []
        for b in labels:
            row = subset[subset['Onset_Bin'] == b]
            vals.append(row['GPP_Collapse_Rate'].values[0] if len(row) > 0 else np.nan)
        offset = (i - 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width, color=color,
               alpha=0.88, edgecolor='white', linewidth=0.5,
               label=dtype, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS)
    ax.set_xlabel('Onset duration (pentads)', fontsize=FS_LABEL)
    ax.set_ylabel('Mean GPP decline rate\n(percentile/pentad)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS)
    ax.set_ylim(4, 9)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=FS, loc='upper right')

    add_panel_label(ax, 'd', x=0.50, y=0.97, ha='center', va='top')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    spatial_df = load_spatial_data()
    gpp_df     = load_gpp_data()

    # Layout:
    #   Row 0 (tall): (a) CONUS map [wide] | (b) EBM bar
    #   Row 1 (short): (c) KDE             | (d) onset bar
    # Row 1 is shorter because the KDE/bar plots don't need as much height.
    # The map fills its full allocated space because make_axes_locatable
    # attaches the colorbar without consuming axes height.
    fig = plt.figure(figsize=(24, 13))
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.6, 1.0],   # row 0 significantly taller
        width_ratios=[1.8, 1.0],    # map column much wider
        hspace=0.30,
        wspace=0.22,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a, spatial_df)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c, gpp_df)
    draw_panel_d(ax_d, gpp_df)

    # After draw_panel_a, draw_panel_b, etc. -- add these lines before savefig:
    fig.canvas.draw()  # force layout calculation
    pos_a = ax_a.get_position()
    pos_c = ax_c.get_position()
    # Keep (c)'s left edge and height, but set its width to match (a)
    ax_c.set_position([pos_a.x0, pos_c.y0, pos_a.width, pos_c.height])

    pos_b = ax_b.get_position()
    pos_d = ax_d.get_position()
    ax_d.set_position([pos_b.x0, pos_d.y0, pos_b.width, pos_d.height])

    out_path = os.path.join(OUTPUT_DIR, 'Figure_PNAS_multipanel.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'Figure_PNAS_multipanel.png')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print("Saved:\n  " + out_path + "\n  " + png_path)
    plt.show()


if __name__ == '__main__':
    main()