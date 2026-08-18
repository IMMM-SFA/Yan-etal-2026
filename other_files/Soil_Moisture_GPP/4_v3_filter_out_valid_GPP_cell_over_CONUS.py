import xarray as xr
import numpy as np
import glob
from scipy.spatial import cKDTree

# ---------------------------------------------------------
# 1. Process the 2D Vegetation Fraction Mask
# ---------------------------------------------------------
print("Loading Vegetation Fraction Data...")
veg_ds = xr.open_dataset('/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/total_vegetated_fraction_1980-2015.nc')

# Calculate the long-term mean vegetation fraction
veg_mean = veg_ds['Total_Vegetated_Fraction'].mean(dim='time')

# Extract 2D coordinate arrays and standardize longitudes to -180 to 180
veg_lat = veg_ds['lat'].values
veg_lon = veg_ds['lon'].values
veg_lon = np.where(veg_lon > 180, veg_lon - 360, veg_lon)

# ---------------------------------------------------------
# 2. Process the 1D GPP Data & Apply the Xue et al. Logic
# ---------------------------------------------------------
file_pattern = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/gpp_cluster_*_cell_avg_with_coords.nc'
print(f"Loading GPP Cluster files from: {file_pattern}")
gpp_ds = xr.open_mfdataset(file_pattern, combine='nested', concat_dim='cell', parallel=True)

# Standardize GPP Longitude to -180 to 180
gpp_lon = xr.where(gpp_ds.lon > 180, gpp_ds.lon - 360, gpp_ds.lon)
gpp_ds = gpp_ds.assign_coords(lon=gpp_lon)

# Filter for the Growing Season: April 1 to October 31
gs_mask = gpp_ds['time'].dt.month.isin([4, 5, 6, 7, 8, 9, 10])
gpp_gs = gpp_ds.sel(time=gs_mask)

print("Calculating Historical Growing Season Mean GPP...")
gpp_mean_gs = gpp_gs['Gridcell_Avg_GPP'].mean(dim='time').compute()

# ---------------------------------------------------------
# 3. Spatial Mapping: KDTree (2D Veg Grid -> 1D GPP Cells)
# ---------------------------------------------------------
print("Building KDTree for fast spatial mapping...")
# Flatten the 2D vegetation coordinates into a 1D list of (lat, lon) pairs
veg_coords = np.column_stack((veg_lat.ravel(), veg_lon.ravel()))
tree = cKDTree(veg_coords)

# Extract target 1D coordinates from the GPP dataset
target_lats = gpp_ds.lat.values
target_lons = gpp_ds.lon.values
target_coords = np.column_stack((target_lats, target_lons))

# Query the KDTree to find the nearest vegetation grid index for each GPP cell
print("Querying KDTree nearest neighbors...")
distances, nearest_indices = tree.query(target_coords)

# Flatten the vegetation mean array and extract the mapped values
veg_mean_values = veg_mean.values.ravel()
matched_veg_fractions = veg_mean_values[nearest_indices]

# Convert the NumPy array back to an xarray DataArray tied to your specific cells
cell_veg_fraction = xr.DataArray(
    matched_veg_fractions, 
    dims=['cell'], 
    coords={'cell': gpp_ds.cell.values}
)

# ---------------------------------------------------------
# 4. Apply the 3-Tiered Masking Criteria
# ---------------------------------------------------------
print("Applying 3-Tiered Masking Criteria...")
# Criterion 1: GPP > 0
mask_c1 = gpp_mean_gs > 0

# Criterion 2: Remove bottom 10th percentile (calculated strictly on cells > 0)
percentile_10th = gpp_mean_gs.where(mask_c1).quantile(0.10)
mask_c2 = gpp_mean_gs >= percentile_10th

# Criterion 3: Total Vegetated Area >= 10%
mask_c3 = cell_veg_fraction >= 0.10

# Combine all masks (Logical AND)
final_valid_mask = mask_c1 & mask_c2 & mask_c3

# Print retention statistics
total_cells = len(final_valid_mask)
retained_cells = int(final_valid_mask.sum())
print("\n--- Masking Summary ---")
print(f"Total Initial Cells: {total_cells}")
print(f"Cells passing GPP > 0: {int(mask_c1.sum())}")
print(f"10th Percentile GPP Threshold: {float(percentile_10th.values):.4f} gC/m2/day")
print(f"Cells passing GPP >= 10th Pct: {int(mask_c2.sum())}")
print(f"Cells passing Veg >= 10%: {int(mask_c3.sum())}")
print(f"FINAL RETAINED CELLS: {retained_cells} ({(retained_cells/total_cells)*100:.1f}%)")

# ---------------------------------------------------------
# 5. Extract Valid Cells and Save Target Indices
# ---------------------------------------------------------
print("Exporting valid coordinates to CSV...")

# Extract just the lat/lon variables and apply the mask, dropping invalid cells
valid_ds = gpp_ds[['lat', 'lon']].where(final_valid_mask, drop=True)

# Convert to DataFrame. The index will be 'cell', and columns will be 'lat', 'lon'
valid_cells_df = valid_ds.to_dataframe().reset_index()

# Ensure integer typing for the cell IDs (xarray sometimes casts to float when masking)
valid_cells_df['cell'] = valid_cells_df['cell'].astype(int)

# Save to CSV
valid_cells_df[['cell', 'lat', 'lon']].to_csv('valid_conus_flash_drought_cells.csv', index=False)
print("Saved valid cell list to 'valid_conus_flash_drought_cells.csv'")