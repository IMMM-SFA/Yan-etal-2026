import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import gc  # Garbage Collector

# ==============================================================================
# CONFIGURATION
# ==============================================================================
base_dir = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist'
input_gpp_dir = os.path.join(base_dir, 'processed_gpp', 'gridcell_avg')
output_dir = os.path.join(base_dir, 'reordered_cluster_gpp')
cluster_map_file = os.path.join(base_dir, 'conus_lat_lon_cluster_US_climate_change.txt')

os.makedirs(output_dir, exist_ok=True)

# Helper: 0-360 Longitude Conversion for CLM5 compatibility
def to_360(lon_array):
    lon_array = np.array(lon_array)
    return np.where(lon_array < 0, lon_array + 360, lon_array)

# ==============================================================================
# 1. LOAD CLUSTER MAP (FAST LOOKUP)
# ==============================================================================
print(f"1. Loading Cluster Map...")
df_map = pd.read_csv(cluster_map_file, sep='\s+', names=['lat', 'lon', 'cluster'], header=None)
df_map['lon'] = to_360(df_map['lon'].values).round(4)
df_map['lat'] = df_map['lat'].round(4)

# Filter for Clusters 1-7 only
df_map = df_map[df_map['cluster'].isin(range(1, 8))]

# Set MultiIndex for O(1) spatial joins
df_map = df_map.set_index(['lat', 'lon'])

print(f"   -> Map ready. {len(df_map)} cells targeted.")

# ==============================================================================
# 2. BUILD THE SPATIAL INDEX MAP (METADATA SCAN)
# ==============================================================================
print(f"2. Scanning input files to build Index Map (Metadata only)...")

gpp_files = sorted(glob.glob(os.path.join(input_gpp_dir, "C*_hist_h1_TLAI_GPP_Gridcell_True_Avg.nc")))
cluster_indices = {c: [] for c in range(1, 8)}

for f_path in gpp_files:
    # Open ONLY coordinates to prevent loading the 13,140 time steps into RAM
    with xr.open_dataset(f_path, decode_times=False) as ds:
        lats = ds['lat'].values.round(4)
        lons = to_360(ds['lon'].values).round(4)
        n_cells = len(lats)
        
        df_curr = pd.DataFrame({'lat': lats, 'lon': lons, 'file_idx': range(n_cells)})
        
        # Vectorized inner join to identify intersecting grid cells
        merged = df_curr.join(df_map, on=['lat', 'lon'], how='inner')
        
        if merged.empty:
            continue
            
        for c_id, group in merged.groupby('cluster'):
            if c_id in cluster_indices:
                indices = group['file_idx'].values
                cluster_indices[c_id].append( (f_path, indices) )

print("   -> Index Map built. Commencing out-of-core data extraction...")

# ==============================================================================
# 3. PROCESS CLUSTERS VIA DASK LAZY EVALUATION
# ==============================================================================
for c_id in range(1, 8):
    file_list = cluster_indices[c_id]
    
    if not file_list:
        print(f"\nCluster {c_id}: NO DATA FOUND. Skipping.")
        continue
        
    print(f"\nProcessing Cluster {c_id}...")
    print(f"   -> Aggregating {len(file_list)} file chunks...")
    
    datasets_to_merge = []
    
    for f_path, indices in file_list:
        # Lazy load: chunking over the 'time' dimension (13140 steps) is critical here
        ds = xr.open_dataset(f_path, chunks={'time': 1000}) 
        
        # UPDATED: Target the exact variable name in your NetCDF schema
        if 'Gridcell_Avg_GPP' not in ds.data_vars:
            print(f"      WARNING: 'Gridcell_Avg_GPP' missing in {os.path.basename(f_path)}")
            continue
            
        # Isolate the variable and slice spatially
        ds_subset = ds[['Gridcell_Avg_GPP']].isel(cell=indices)
        datasets_to_merge.append(ds_subset)
    
    if not datasets_to_merge:
        continue
        
    # Lazy concatenation along the 'cell' dimension
    ds_final = xr.concat(datasets_to_merge, dim='cell')
    
    # Inject metadata for downstream clarity
    ds_final.attrs['cluster_id'] = c_id
    ds_final.attrs['description'] = f'Gridcell Average GPP - Reordered to Target Cluster {c_id}'
    ds_final.attrs['units'] = 'gC/m2/s' # Standard CLM5 flux units
    
    out_name = f'gpp_cluster_{c_id}_cell_avg.nc'
    out_path = os.path.join(output_dir, out_name)
    
    print(f"   -> Writing {out_path} ... (Triggering Dask computation)")
    
    # Execute the Dask graph and stream to disk
    ds_final.to_netcdf(out_path)
    
    # Force memory flush
    print(f"   -> Finished Cluster {c_id}.")
    del ds_final
    del datasets_to_merge
    gc.collect() 

print("\nData reordering and extraction complete.")