import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('spatial_flash_drought_properties.csv')

# cluster -> region mapping (based on centroid lat/lon)
cluster_to_region = {
    1: 'NW',
    2: 'SW',
    3: 'NGP',
    4: 'SGP',
    5: 'MW',
    6: 'SE',
    7: 'NE',
}
df['region'] = df['cluster'].map(cluster_to_region)

region_order = ['NW', 'SW', 'NGP', 'SGP', 'MW', 'SE', 'NE']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#DD8452']
color_map = dict(zip(region_order, colors))

y_col = 'ratio_flash'

# rows = categories, each row is a list of (column, xlabel)
rows = [
    ('Climatology', [
        ('aridity_index', 'Aridity Index (PET / P)'),
        ('precip_gs_mean', 'Growing Season (GS) Precip (mm)'),
        ('solar_mean', r'Growing Season (GS) Solar (W m$^{-2}$)'),
    ]),
    ('Atmospheric Extremes', [
        ('VPD95', 'Growing Season (GS) VPD 95th (kPa)'),
        ('heatwave_days', 'Heatwave Days (days)'),
        ('max_dry_days', 'Max Dry Days (days)'),
    ]),
    ('Vegetation Biophysics', [
        ('GPP95', r'Growing Season (GS) GPP 95th (gC m$^{-2}$ day$^{-1}$)'),
        ('iWUE_mean', r'Inherent Water-Use Efficiency (gC kPa kg$^{-1}$)'),
        ('TET_mean', 'T/ET Ratio [0, 1]'),
    ]),
]

# cluster means and grid-cell counts (used to size the markers)
counts = df.groupby('region').size().reindex(region_order)
sizes = 150 + 450 * (counts - counts.min()) / (counts.max() - counts.min())

fig, axes = plt.subplots(3, 3, figsize=(15, 13), layout='compressed')

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
panel_idx = 0

for row_idx, (cat_name, vars_list) in enumerate(rows):
    for col_idx, (x_col, xlabel) in enumerate(vars_list):
        ax = axes[row_idx, col_idx]
        means = df.groupby('region')[[x_col, y_col]].mean().reindex(region_order)

        for region in region_order:
            ax.scatter(means.loc[region, x_col], means.loc[region, y_col],
                       s=sizes[region], color=color_map[region],
                       edgecolors='black', linewidths=1, zorder=3)
            ax.annotate(region, (means.loc[region, x_col], means.loc[region, y_col]),
                        xytext=(6, 6), textcoords='offset points', fontsize=9, fontweight='bold')

        ax.set_xlabel(xlabel, fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(f'{cat_name}\nFlash Drought Ratio', fontsize=10.5, fontweight='bold')
        else:
            ax.set_ylabel('Flash Drought Ratio', fontsize=10)

        ax.text(-0.12, 1.06, panel_labels[panel_idx], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle='--')
        panel_idx += 1

plt.savefig('cluster_mean_scatter_3x3.png', dpi=150)
print('saved')