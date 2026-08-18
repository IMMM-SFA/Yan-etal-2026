import xarray as xr
import pandas as pd
import numpy as np
import os
import gc

# --- Configuration ---
base_dir = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist'
gpp_input_dir = os.path.join(base_dir, 'reordered_cluster_gpp')
area_file = os.path.join(base_dir, 'crop_area_fractions_1980-2015.nc')
output_dir = base_dir 

crops = {
    'Corn':    [('corn_pft17', 17),    ('corn_pft75', 75)],
    'Soybean': [('soybean_pft23', 23), ('soybean_pft77', 77)],
    'Wheat':   [('wheat_pft19', 19),   ('wheat_pft21', 21)]
}

# Strict Filter: Cell must have >1% TOTAL area in ALL 36 years
MIN_AREA_THRESHOLD = 0.01 

# Compression Settings (CRITICAL for file size)
COMPRESSION = dict(zlib=True, complevel=5, shuffle=True, dtype='float32', _FillValue=None)

# Time Setup (1980-2015)
years = range(1980, 2016)
days_per_year = 365
total_days = len(years) * days_per_year # 13140

# Helper: Force 0-360 Longitude
def to_360(lon_array):
    lon_array = np.array(lon_array)
    return np.where(lon_array < 0, lon_array + 360, lon_array)

# ==============================================================================
# 1. PROCESS AREA FILE (FLATTEN 2D GRID)
# ==============================================================================
print(f"1. Loading Area File: {area_file}")
ds_area = xr.open_dataset(area_file)
print("   -> Converting Area to DataFrame...")
df_area_raw = ds_area.to_dataframe().reset_index()

# Handle Lat/Lon naming
possible_lats = [c for c in df_area_raw.columns if 'lat' in c.lower() and 'lsm' not in c.lower()]
possible_lons = [c for c in df_area_raw.columns if 'lon' in c.lower() and 'lsm' not in c.lower()]
lat_col = possible_lats[0] if possible_lats else 'lat'
lon_col = possible_lons[0] if possible_lons else 'lon'

df_area_raw[lon_col] = to_360(df_area_raw[lon_col].values).round(4)
df_area_raw[lat_col] = df_area_raw[lat_col].round(4)
df_area_raw = df_area_raw.rename(columns={lat_col: 'lat', lon_col: 'lon'})

# ==============================================================================
# 2. IDENTIFY VALID CELLS AND STORE EXACT FRACTIONS
# ==============================================================================
print("2. Pre-calculating Valid Cells and Fractions...")
valid_mask_map = {} 
area_value_map = {} # New: Dictionary to hold the exact area fractions

for crop_name, components in crops.items():
    total_area_series = 0
    for pft_name, _ in components:
        if pft_name in df_area_raw.columns:
            total_area_series += df_area_raw[pft_name].fillna(0)
    
    df_temp = df_area_raw[['lat', 'lon', 'time']].copy()
    df_temp['total_area'] = total_area_series
    
    # Calculate minimum area over the 36 years
    min_area = df_temp.groupby(['lat', 'lon'])['total_area'].min()
    
    # Filter by the 1% threshold
    valid_series = min_area[min_area > MIN_AREA_THRESHOLD]
    
    valid_mask_map[crop_name] = set(valid_series.index)
    area_value_map[crop_name] = valid_series.to_dict() # Store the exact values mapped to (lat, lon)
    
    print(f"   {crop_name}: {len(valid_series)} cells passed 1% base filter.")

# ==============================================================================
# 3. PROCESS CLUSTERS (PER CELL CALCULATION)
# ==============================================================================
print("\n3. Processing Clusters (Generating Gridded Output)...")
dates_noleap = pd.date_range("1980-01-01", periods=total_days + 20)
dates_noleap = dates_noleap[~((dates_noleap.month == 2) & (dates_noleap.day == 29))][:total_days]

for c_id in range(1, 8):
    gpp_file = os.path.join(gpp_input_dir, f'New_Cluster_{c_id}_raw_GPP.nc')
    if not os.path.exists(gpp_file): continue
        
    print(f"\n  Processing Cluster {c_id}...")
    ds_gpp = xr.open_dataset(gpp_file, chunks={'time': 'auto'})
    
    # Get Grid Info
    gpp_lats = ds_gpp['lat'].values.round(4)
    gpp_lons = to_360(ds_gpp['lon'].values).round(4)
    n_cells = len(gpp_lats)
    
    # Prepare Output Dataset
    ds_out = xr.Dataset(
        coords={
            'time': dates_noleap,
            'cell': np.arange(n_cells),
            'lat': (('cell'), gpp_lats),
            'lon': (('cell'), gpp_lons)
        }
    )
    
    # Prepare Area DataFrame for this cluster
    df_gpp_geo = pd.DataFrame({'lat': gpp_lats, 'lon': gpp_lons, 'cell_idx': range(n_cells)})
    df_cluster_area = pd.merge(df_gpp_geo, df_area_raw, on=['lat', 'lon'], how='left')
    
    # Variables to Encode later
    encoding_dict = {}

    for crop_name, components in crops.items():
        print(f"    Calculating {crop_name}...")
        
        # 1. Init Result (Using float32 to save RAM/Disk)
        result_gpp = np.zeros((n_cells, total_days), dtype=np.float32)
        fraction_arr = np.zeros(n_cells, dtype=np.float32) # New: Array for area fractions
        
        valid_set = valid_mask_map[crop_name]
        crop_area_dict = area_value_map[crop_name]
        
        is_valid = [ (lat, lon) in valid_set for lat, lon in zip(gpp_lats, gpp_lons) ]
        valid_indices = np.where(is_valid)[0]
        
        # Populate the static fraction array for the NetCDF output
        for i, (lat, lon) in enumerate(zip(gpp_lats, gpp_lons)):
            if (lat, lon) in valid_set:
                fraction_arr[i] = crop_area_dict[(lat, lon)]
        
        if len(valid_indices) == 0:
            ds_out[f'{crop_name}_GPP'] = (('cell', 'time'), result_gpp)
            ds_out[f'{crop_name}_fraction'] = (('cell',), fraction_arr)
            encoding_dict[f'{crop_name}_GPP'] = COMPRESSION
            encoding_dict[f'{crop_name}_fraction'] = dict(zlib=True, complevel=5, dtype='float32')
            continue
            
        subset_num = np.zeros((len(valid_indices), total_days), dtype=np.float32)
        subset_den = np.zeros((len(valid_indices), total_days), dtype=np.float32)
        
        for pft_area_name, pft_id in components:
            gpp_var = f"{crop_name.lower()}_pft{pft_id}_GPP"
            if gpp_var not in ds_gpp: continue
            
            # Slice time to 13140
            raw_gpp = ds_gpp[gpp_var].isel(cell=valid_indices, time=slice(0, total_days)).values
            raw_gpp = np.nan_to_num(raw_gpp, nan=0.0)
            
            pivot_area = df_cluster_area[df_cluster_area['cell_idx'].isin(valid_indices)]
            pivot_area = pivot_area.pivot(index='cell_idx', columns='time', values=pft_area_name)
            pivot_area = pivot_area.reindex(valid_indices).fillna(0.0)
            
            area_daily = np.repeat(pivot_area.values, days_per_year, axis=1)
            
            if raw_gpp.shape != area_daily.shape and raw_gpp.shape == area_daily.T.shape:
                area_daily = area_daily.T
            
            subset_num += (raw_gpp * area_daily)
            subset_den += area_daily
            
        with np.errstate(divide='ignore', invalid='ignore'):
            weighted_vals = subset_num / subset_den
        
        weighted_vals = np.nan_to_num(weighted_vals, nan=0.0)
        result_gpp[valid_indices, :] = weighted_vals.astype(np.float32)
        
        # Add GPP to Dataset
        ds_out[f'{crop_name}_GPP'] = (('cell', 'time'), result_gpp)
        ds_out[f'{crop_name}_GPP'].attrs = {'units': 'gC/m2/day'}
        
        # Add Fraction to Dataset
        ds_out[f'{crop_name}_fraction'] = (('cell',), fraction_arr)
        ds_out[f'{crop_name}_fraction'].attrs = {'units': 'fraction (0-1)', 'description': 'Minimum crop area fraction across 1980-2015'}
        
        # Add Compression settings
        encoding_dict[f'{crop_name}_GPP'] = COMPRESSION
        encoding_dict[f'{crop_name}_fraction'] = dict(zlib=True, complevel=5, dtype='float32')

    # Save File with Compression
    out_path = os.path.join(output_dir, f'gpp_cluster_{c_id}.nc')
    ds_out.attrs['cluster_id'] = c_id
    
    print(f"    Saving {out_path} (Compressed)...")
    ds_out.to_netcdf(out_path, encoding=encoding_dict)
    
    del ds_gpp
    del ds_out
    gc.collect()

print("\nAll Done!")