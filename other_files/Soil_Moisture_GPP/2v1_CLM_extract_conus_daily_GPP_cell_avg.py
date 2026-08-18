import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import time
from multiprocessing import Pool

# --- Configuration ---
INPUT_DIR = '/global/cfs/cdirs/m2702/liliyao/CLMBGC_postprocessing/QAQC/historical_with_LUH2_1980-2015/pft_level_vars/pft_GPP_LAI_daily_hist'
OUTPUT_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/processed_gpp/gridcell_avg'
LU_FILE = '/global/cfs/cdirs/m2702/liliyao/inputdata/cesm_inputdata/lnd/clm2/surfdata_map/landuse.timeseries_0.125nldas2_SSP5-8.5_78_CMIP6_1980-2019_c231122.nc'

CHUNK_SIZE = 500  # Days per RAM chunk
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_file(filepath):
    start_time = time.time()
    filename_only = os.path.basename(filepath)
    print(f"[{filename_only}] Starting dynamic LULC processing...")

    # ====================================================================
    # 1. LOAD LAND USE TIME SERIES (DYNAMIC WEIGHTS BASE)
    # ====================================================================
    ds_lu = xr.open_dataset(LU_FILE)
    try:
        ds_lu = ds_lu.sel(time=slice(1980, 2015))
    except KeyError:
        ds_lu = ds_lu.sel(time=slice('1980', '2015'))
        
    # --- FIX: Handle Integer vs. Datetime Encodings ---
    raw_time = ds_lu['time'].values
    if np.issubdtype(raw_time.dtype, np.integer) or np.issubdtype(raw_time.dtype, np.floating):
        # The file is just raw numbers [1980, 1981...]
        lu_years = raw_time.astype(int)
    else:
        # The file is cftime/datetime objects
        lu_years = np.array([t.year for t in raw_time])
    # --------------------------------------------------    
    # Extract arrays into memory for ultra-fast lookup (divide by 100 for fractions)
    pct_crop = ds_lu['PCT_CROP'].values / 100.0
    pct_lake = ds_lu['PCT_LAKE'].values / 100.0
    pct_urb = ds_lu['PCT_URBAN'].sum(dim='numurbl').values / 100.0
    
    # Derive PCT_NATVEG using Conservation of Land Area
    pct_natveg = np.clip(1.0 - pct_crop - pct_lake - pct_urb, 0.0, 1.0)
    
    pct_nat_pft = ds_lu['PCT_NAT_PFT'].values / 100.0
    pct_cft = ds_lu['PCT_CFT'].values / 100.0
    ds_lu.close()

    # ====================================================================
    # 2. OPEN HISTORY FILE & MAP PFTs TO GRIDCELLS
    # ====================================================================
    ds = xr.open_dataset(filepath)
    
    lats = ds['pfts1d_lat'].values
    lons = ds['pfts1d_lon'].values
    itypes = ds['pfts1d_itype_veg'].values
    
    # Fortran 1-based indices to Python 0-based indices for array slicing
    jxy = ds['pfts1d_jxy'].values - 1  # lat index
    ixy = ds['pfts1d_ixy'].values - 1  # lon index
    
    # Map back to unique gridcell IDs
    df_geo = pd.DataFrame({'lat': lats, 'lon': lons})
    unique_cells = df_geo.drop_duplicates().reset_index(drop=True)
    n_cells = len(unique_cells)
    
    coord_strings = df_geo['lat'].astype(str) + "_" + df_geo['lon'].astype(str)
    pft_to_gridcell_map, _ = pd.factorize(coord_strings)
    pft_to_gridcell_map = pft_to_gridcell_map.astype(np.int64)
    
    # ====================================================================
    # 3. BUILD THE DYNAMIC WEIGHT MATRIX (36 Years x N_PFTs)
    # ====================================================================
    n_years = len(lu_years)
    n_pfts = len(itypes)
    dynamic_weights = np.zeros((n_years, n_pfts), dtype=np.float32)
    
    print(f"[{filename_only}] Calculating year-to-year dynamic area weights...")
    for p in range(n_pfts):
        j = int(jxy[p])
        i = int(ixy[p])
        veg_type = int(itypes[p])
        
        if veg_type <= 14:  # Natural Vegetation & Bareground
            dynamic_weights[:, p] = pct_natveg[:, j, i] * pct_nat_pft[:, veg_type, j, i]
        else:               # Agricultural & Bioenergy Crops
            cft_idx = veg_type - 15
            dynamic_weights[:, p] = pct_crop[:, j, i] * pct_cft[:, cft_idx, j, i]

    # ====================================================================
    # 4. CHUNKED TIME PROCESSING WITH DYNAMIC WEIGHTS
    # ====================================================================
    n_time = 13140
    gridcell_gpp = np.zeros((n_time, n_cells), dtype='float32')
    
    print(f"[{filename_only}] Aggregating fluxes using dynamic LUH2 weights...")
    for start_idx in range(0, n_time, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, n_time)
        
        # Pull GPP chunk and fix units/NaNs
        gpp_chunk_daily = ds['GPP'].isel(time=slice(start_idx, end_idx)).values * 86400.0
        gpp_chunk_daily = np.nan_to_num(gpp_chunk_daily, nan=0.0)
        
# Match each day to its exact calendar year to grab the correct LUH2 weights
        time_chunk = ds['time'].isel(time=slice(start_idx, end_idx)).values
        
        # --- FIX: Safely extract year from the history file ---
        if np.issubdtype(time_chunk.dtype, np.integer) or np.issubdtype(time_chunk.dtype, np.floating):
            # In case the history file also uses raw numbers (rare, but safe)
            chunk_years = time_chunk.astype(int)
        else:
            try:
                # Standard Pandas datetime or cftime
                chunk_years = np.array([t.year for t in time_chunk])
            except AttributeError:
                # Failsafe for weird xarray object arrays
                chunk_years = pd.DatetimeIndex(time_chunk).year.values
        # ------------------------------------------------------        
        # Map year (e.g., 2005) to the dynamic_weights index
        year_idx = chunk_years - lu_years[0] 
        chunk_weights = dynamic_weights[year_idx, :]  # Shape: (CHUNK_SIZE, N_PFTs)
        
        # Apply transient weights
        weighted_gpp = gpp_chunk_daily * chunk_weights
        
        # Aggregate securely to the gridcell
        for idx in range(end_idx - start_idx):
            global_t = start_idx + idx
            gridcell_gpp[global_t, :] = np.bincount(
                pft_to_gridcell_map, 
                weights=weighted_gpp[idx, :], 
                minlength=n_cells
            )

    # ====================================================================
    # 5. SAVE DATASET
    # ====================================================================
    cell_idx = np.arange(n_cells)
    clipped_time = ds['time'].values[:n_time]
    out_ds = xr.Dataset(coords={'cell': cell_idx, 'time': clipped_time})
    out_ds['lat'] = (('cell'), unique_cells['lat'].values)
    out_ds['lon'] = (('cell'), unique_cells['lon'].values)
    
    out_ds['Gridcell_Avg_GPP'] = (('time', 'cell'), gridcell_gpp)
    out_ds['Gridcell_Avg_GPP'].attrs = {
        'units': 'gC/m2/day',
        'description': 'True area-weighted gridcell average GPP. Weighting calculated dynamically year-over-year from LUH2 Landuse Timeseries to capture transient LULC changes.'
    }
    
    comp = dict(zlib=True, complevel=5, shuffle=True)
    encoding = {'Gridcell_Avg_GPP': comp}
    
    nc_out_name = filename_only.replace(".nc", "_Gridcell_True_Avg.nc")
    nc_path = os.path.join(OUTPUT_DIR, nc_out_name)
    out_ds.to_netcdf(nc_path, encoding=encoding)
    
    ds.close()
    out_ds.close()
    
    elapsed = (time.time() - start_time) / 60
    print(f"[{filename_only}] Successfully saved in {elapsed:.1f} minutes.")
    return nc_path

if __name__ == '__main__':
    search_pattern = os.path.join(INPUT_DIR, 'C*_hist_h1_TLAI_GPP.nc')
    files_to_process = sorted(glob.glob(search_pattern))
    
    if files_to_process:
        print(f"Initiating multiprocessor pool for {len(files_to_process)} files...")
        num_workers = min(len(files_to_process), 7) 
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_file, files_to_process)
            
        print("\n? All dynamically-weighted Gridcell GPP files generated successfully!")