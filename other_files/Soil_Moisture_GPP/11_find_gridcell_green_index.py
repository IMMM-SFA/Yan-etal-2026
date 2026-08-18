import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = './' # Update to your path if needed
GPP_FILE_TEMPLATE = 'gpp_cluster_{}_cell_avg_with_coords.nc'

YEARS_TOTAL = 36
YEARS_TO_KEEP = 35
DAYS_IN_YEAR = 365
PENTADS_IN_YEAR = 73

# Growing Season Pentad Indices (0-indexed)
# Pentad 18: April 1 - April 5
# Pentad 60: Oct 28 - Nov 1
GS_PENTAD_START = 18  
GS_PENTAD_END = 61    # Exclusive bound in Python, so it slices up to 60

# ==============================================================================
# 1. PROCESS GRIDCELL AVERAGE GPP FOR ALL CLUSTERS
# ==============================================================================
print("--- Building Pentad-Based Ecosystem Greening Feature ---")

all_cluster_data = []

for cluster_id in range(1, 8):
    file_path = os.path.join(BASE_DIR, GPP_FILE_TEMPLATE.format(cluster_id))
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        continue
        
    print(f"Processing Cluster {cluster_id}...")
    
    # Load dataset
    ds = xr.open_dataset(file_path)
    
    # Extract arrays. Check if shape is (time, cell) or (cell, time)
    # User specified Gridcell_Avg_GPP(time=13140, cell=4723)
    gpp_raw = ds['Gridcell_Avg_GPP'].values
    
    if gpp_raw.shape[0] == 13140:
        # Transpose to (cell, time) for easier slicing
        gpp_raw = gpp_raw.T 
        
    lats = ds['lat'].values
    lons = ds['lon'].values
    
    # Normalize Longitude to standard -180 to 180 (if it's in 0-360)
    lons = np.where(lons > 180, lons - 360.0, lons)
    
    n_cells = gpp_raw.shape[0]
    
    # --- The Temporal Tensor Math ---
    
    # Step A: Drop the 1st year (Spin-up discard)
    # 13140 total days - 365 days = 12775 days
    gpp_35yr = gpp_raw[:, DAYS_IN_YEAR:]  # Shape: (cells, 12775)
    
    # Step B: Reshape into (cells, years, pentads, days_per_pentad)
    # 12775 = 35 years * 73 pentads * 5 days
    gpp_reshaped = gpp_35yr.reshape(n_cells, YEARS_TO_KEEP, PENTADS_IN_YEAR, 5)
    
    # Step C: Average the 5 days to get the Pentad GPP
    # Shape becomes: (cells, 35 years, 73 pentads)
    gpp_pentads = np.nanmean(gpp_reshaped, axis=3)
    
    # Step D: Extract ONLY the growing season (April 1 - Oct 31)
    # Shape becomes: (cells, 35 years, 43 pentads)
    gpp_gs_pentads = gpp_pentads[:, :, GS_PENTAD_START:GS_PENTAD_END]
    
    # Step E: Calculate the 95th Percentile of Pentads for EACH year
    # Shape becomes: (cells, 35 years)
    gpp_annual_95th = np.nanpercentile(gpp_gs_pentads, 95, axis=2)
    
    # Step F: Take the Mean across the 35 years
    # Shape becomes: (cells,)
    gpp_greening_feature = np.nanmean(gpp_annual_95th, axis=1)
    
    # Store results for this cluster
    df_cluster = pd.DataFrame({
        'lat': lats.round(4),
        'lon': lons.round(4),
        'cluster': cluster_id,
        'gpp_greening_index': gpp_greening_feature
    })
    
    all_cluster_data.append(df_cluster)

# ==============================================================================
# 2. COMBINE AND SAVE MASTER DATASET
# ==============================================================================
df_master = pd.concat(all_cluster_data, ignore_index=True)

# Drop any cells that were entirely NaN (ocean/water cells)
df_master = df_master.dropna(subset=['gpp_greening_index'])

# Save the final Features Dataset
OUTPUT_CSV = os.path.join(BASE_DIR, 'Gridcell_GPP_Greening_Feature.csv')
df_master.to_csv(OUTPUT_CSV, index=False)

print(f"\n[SUCCESS] Extracted Greening Feature for {len(df_master)} grid cells.")
print(f"Saved dataset to: {OUTPUT_CSV}")
print("\nSample Output:")
print(df_master.head())