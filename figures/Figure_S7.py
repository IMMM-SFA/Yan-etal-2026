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

vars_info = [
    ('PCT_SAND_WTMEAN', 'Soil Sand %', '(%)'),
    ('PCT_CLAY_WTMEAN', 'Soil Clay %', '(%)'),
]

panel_labels = ['(a)', '(b)']

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#DD8452']

fig, axes = plt.subplots(2, 1, figsize=(7, 7.5), layout='compressed')

for ax, (col, label, unit), panel in zip(axes, vars_info, panel_labels):
    grouped = df.groupby('region')[col]
    medians = grouped.median().reindex(region_order)
    p05 = grouped.quantile(0.05).reindex(region_order)
    p95 = grouped.quantile(0.95).reindex(region_order)

    lower_err = (medians - p05).clip(lower=0)
    upper_err = (p95 - medians).clip(lower=0)
    yerr = np.vstack([lower_err.values, upper_err.values])

    ax.bar(region_order, medians.values, yerr=yerr,
           capsize=4, color=colors, edgecolor='black', linewidth=0.6,
           error_kw={'elinewidth': 1, 'ecolor': 'black'})

    ax.set_ylabel(f'{label}\n{unit}', fontsize=11)
    ax.text(-0.08, 1.05, panel, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

axes[-1].set_xlabel('Region', fontsize=11)

plt.savefig('soil_physics_by_region.png', dpi=150)
print('saved')