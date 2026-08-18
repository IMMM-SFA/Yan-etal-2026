import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import gc

# ==============================================================================
# CONFIGURATION
# ==============================================================================
base_dir = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist'
# The directory with the original files (to get the lat/lon arrays)
original_gpp_dir = os.path.join(base_dir, 'processed_gpp', 'gridcell_avg')
# The directory with your currently processed cluster files (missing lat/lon)
processed_dir = os.path.join(base_dir, 'reordered_cluster_gpp')
cluster_map_file = os.path.join(base_dir, 'conus_lat_lon_cluster_US_climate_change.txt')

# Helper: 0-360 Longitude Conversion
def to_360(lon_array):
    lon_array = np.array(lon_array)
    return np.where(lon_array < 0, lon_array + 360, lon_array)

# ==============================================================================
# 1. LOAD CLUSTER MAP
# ==============================================================================
print("1. Loading Cluster Map...")
df_map = pd.read_csv(cluster_map_file, sep='\s+', names=['lat', 'lon', 'cluster'], header=None)
df_map['lon'] = to_360(df_map['lon'].values).round(4)
df_map['lat'] = df_map['lat'].round(4)
df_map = df_map[df_map['cluster'].isin(range(1, 8))]
df_map = df_map.set_index(['lat', 'lon'])

# ==============================================================================
# 2. RECONSTRUCT THE EXACT LAT/LON ARRAYS (METADATA ONLY)
# ==============================================================================
print("2. Scanning original files to extract ordered lat/lon arrays...")
orig_files = sorted(glob.glob(os.path.join(original_gpp_dir, "C*_hist_h1_TLAI_GPP_Gridcell_True_Avg.nc")))

# Dictionaries to hold the sequential list of lat and lon arrays for each cluster
cluster_coords = {c: {'lat': [], 'lon': []} for c in range(1, 8)}

for f_path in orig_files:
    # Open lazily just for coords
    with xr.open_dataset(f_path, decode_times=False) as ds:
        lats = ds['lat'].values.round(4)
        lons = to_360(ds['lon'].values).round(4)
        
        df_curr = pd.DataFrame({'lat': lats, 'lon': lons, 'file_idx': range(len(lats))})
        merged = df_curr.join(df_map, on=['lat', 'lon'], how='inner')
        
        if merged.empty:
            continue
            
        # Extract the matching lat/lon in the exact order they were processed previously
        for c_id, group in merged.groupby('cluster'):
            indices = group['file_idx'].values
            cluster_coords[c_id]['lat'].append(lats[indices])
            cluster_coords[c_id]['lon'].append(lons[indices])

# ==============================================================================
# 3. INJECT COORDINATES INTO PROCESSED FILES
# ==============================================================================
print("3. Injecting coordinates into processed cluster files...")

for c_id in range(1, 8):
    input_file = os.path.join(processed_dir, f'gpp_cluster_{c_id}_cell_avg.nc')
    output_file = os.path.join(processed_dir, f'gpp_cluster_{c_id}_cell_avg_with_coords.nc')
    
    if not os.path.exists(input_file):
        print(f"   -> Skipping Cluster {c_id}: {input_file} not found.")
        continue
    
    # Concatenate the lists of arrays into final 1D numpy arrays for this cluster
    final_lat = np.concatenate(cluster_coords[c_id]['lat'])
    final_lon = np.concatenate(cluster_coords[c_id]['lon'])
    
    print(f"\nProcessing Cluster {c_id}...")
    
    # Open the processed dataset (chunked to avoid RAM overload on the time dimension)
    ds = xr.open_dataset(input_file, chunks={'time': 1000})
    
    # Sanity check: Ensure the rebuilt lat/lon size matches the cell dimension
    num_cells_in_ds = ds.sizes['cell']
    if len(final_lat) != num_cells_in_ds:
        print(f"      ERROR: Mismatch! File has {num_cells_in_ds} cells, but rebuilt coords have {len(final_lat)} cells.")
        continue
        
    print(f"   -> Match confirmed: {num_cells_in_ds} cells. Assigning coordinates...")
    
    # Assign the arrays to the 'cell' dimension
    ds['lat'] = (('cell',), final_lat)
    ds['lon'] = (('cell',), final_lon)
    
    # Explicitly upgrade these variables to be structural coordinates
    ds = ds.set_coords(['lat', 'lon'])
    
    # Save to a NEW file (this streams the chunks nicely)
    print(f"   -> Writing updated dataset to: {os.path.basename(output_file)}")
    ds.to_netcdf(output_file)
    
    # Cleanup
    del ds
    gc.collect()

print("\nFinished! All files now contain proper spatial coordinates.")