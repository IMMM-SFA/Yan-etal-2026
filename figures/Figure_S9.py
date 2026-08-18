import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# DATA (pasted from Figure_4.py --print-only output)
# ==============================================================================
REGIONS = ['CONUS', 'NW', 'SW', 'NGP', 'SGP', 'MW', 'SE', 'NE']

DATA = {
    'Grassland': {
        'NW': (38.0, 12.4, 14.0), 'SW': (38.8, 25.1, 27.8), 'NGP': (55.8, 22.4, 28.6),
        'SGP': (54.5, 16.8, 21.6), 'MW': (29.1, 8.2, 10.0), 'SE': (31.3, 8.9, 10.2),
        'NE': (20.0, 3.9, 4.2), 'CONUS': (39.4, 15.7, 18.6),
    },
    'Shrub': {
        'NW': (0.1, 7.0, 7.3), 'SW': (11.1, 24.3, 25.3), 'NGP': (0.0, 0.9, 1.1),
        'SGP': (3.6, 5.7, 6.0), 'MW': (0.0, 0.0, 0.0), 'SE': (0.0, 0.0, 0.0),
        'NE': (0.0, 0.0, 0.0), 'CONUS': (3.0, 6.9, 7.2),
    },
    'Forest': {
        'NW': (42.5, 49.1, 54.8), 'SW': (11.8, 28.4, 28.6), 'NGP': (9.3, 12.6, 14.8),
        'SGP': (6.3, 10.6, 12.6), 'MW': (20.4, 28.7, 30.7), 'SE': (43.5, 50.8, 55.3),
        'NE': (60.5, 66.5, 68.7), 'CONUS': (23.2, 31.5, 34.0),
    },
    'Crop': {
        'NW': (11.6, 29.1, 21.0), 'SW': (7.5, 19.6, 15.1), 'NGP': (28.9, 63.5, 54.7),
        'SGP': (29.0, 64.7, 56.7), 'MW': (45.2, 59.0, 53.9), 'SE': (20.7, 34.5, 27.0),
        'NE': (12.5, 21.6, 17.1), 'CONUS': (22.7, 42.5, 35.9),
    },
}

CATEGORIES = ['Grassland', 'Shrub', 'Forest', 'Crop']

# Colors/labels matching draw_lulc_bar() in Figure_4.py
COLOR_HIST  = '#888780'
COLOR_SSP345 = '#66C2E8'
COLOR_SSP585 = '#F5A962'
COLORS = [COLOR_HIST, COLOR_SSP345, COLOR_SSP585]
LABELS = ['Hist./S1/S2', 'S3', 'S4']

FS = 13
FS_LABEL = 14
FS_TITLE = 15
FS_PANEL = 17

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.size':       FS,
    'axes.labelsize':  FS_LABEL,
    'axes.titlesize':  FS_TITLE,
    'xtick.labelsize': FS,
    'ytick.labelsize': FS,
    'legend.fontsize': FS,
    'axes.linewidth':  1.0,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

fig, axes = plt.subplots(4, 1, figsize=(11, 16), sharex=True)

x  = np.arange(len(REGIONS))
bw = 0.24
offsets = [-bw, 0.0, bw]

for ax, cat, letter in zip(axes, CATEGORIES, 'cdef'):
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        vals = [DATA[cat][r][i] for r in REGIONS]
        ax.bar(x + offsets[i], vals, width=bw, color=color, alpha=0.9,
               edgecolor='white', linewidth=0.5, label=label, zorder=3)

    ax.axvline(x=0.5, color='gray', linewidth=0.8, linestyle='--',
               alpha=0.6, zorder=2)
    ax.set_ylabel('{} fraction (%)'.format(cat), fontsize=FS_LABEL)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.01, 0.96, '(' + letter + ')', transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', ha='left', va='top')

axes[0].legend(frameon=False, fontsize=FS - 1, loc='upper right', ncol=3,
               handlelength=1.2, handleheight=0.9)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(REGIONS, fontsize=FS)
axes[-1].set_xlabel('Region', fontsize=FS_LABEL)

fig.suptitle('LULC category fractions by CONUS region and scenario',
             fontsize=FS_TITLE + 1, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])

out_png = '/mnt/user-data/outputs/lulc_regional_4panel.png'
out_pdf = '/mnt/user-data/outputs/lulc_regional_4panel.pdf'
fig.savefig(out_png)
fig.savefig(out_pdf)
print('Saved:\n  ' + out_png + '\n  ' + out_pdf)