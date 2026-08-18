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

# 2 rows x 3 columns: Land Cover (4) + Soil Physics: Sand, Clay (2)
panels = [
    ('forest_fraction', 'Forest Cover [0, 1]'),
    ('crop_fraction', 'Crop Cover [0, 1]'),
    ('shrub_fraction', 'Shrub Cover [0, 1]'),
    ('grass_fraction', 'Grass Cover [0, 1]'),
    ('PCT_SAND_WTMEAN', 'Soil Sand % (%)'),
    ('PCT_CLAY_WTMEAN', 'Soil Clay % (%)'),
]

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# cluster means and grid-cell counts (used to size the markers)
counts = df.groupby('region').size().reindex(region_order)
sizes = 150 + 450 * (counts - counts.min()) / (counts.max() - counts.min())

fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='compressed')
axes = axes.flatten()

for ax, (x_col, xlabel), panel in zip(axes, panels, panel_labels):
    means = df.groupby('region')[[x_col, y_col]].mean().reindex(region_order)

    for region in region_order:
        ax.scatter(means.loc[region, x_col], means.loc[region, y_col],
                   s=sizes[region], color=color_map[region],
                   edgecolors='black', linewidths=1, zorder=3)
        ax.annotate(region, (means.loc[region, x_col], means.loc[region, y_col]),
                    xytext=(6, 6), textcoords='offset points', fontsize=9, fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel('Flash Drought Ratio', fontsize=10.5)
    ax.text(-0.12, 1.06, panel, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')

plt.savefig('landcover_soilphysics_scatter_2x3.png', dpi=150)
print('saved')