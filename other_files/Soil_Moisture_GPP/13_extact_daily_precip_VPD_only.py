import os
# Force internal C-libraries to be strictly single-threaded inside each worker.
# This prevents CPU thrashing when using multiprocessing.Pool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
import multiprocessing as mp

# ==========================================
# 1. FILE PATHS & PARAMETERS
# ==========================================
WRF_DIR = '/global/cfs/cdirs/m2702/gsharing/WRF_CLM5/historical_1980_2019'
CSV_PATH = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/Gridcell_GPP_Greening_Feature.csv'

START_YEAR = 1981
END_YEAR = 2015

# Number of parallel workers (4 to 8 is safe for Perlmutter)
NUM_WORKERS = 8 

def normalize_longitude(lon):
    return lon % 360.0

def build_spatial_mapper_arrays(sample_nc_path, csv_path):
    """Builds the KDTree once and returns lightweight 1D integer arrays to pass to workers."""
    print("?? Building Global Spatial KDTree Mapper...", flush=True)
    df = pd.read_csv(csv_path)
    target_lats = df['lat'].values
    target_lons = normalize_longitude(df['lon'].values)
    
    ds_grid = xr.open_dataset(sample_nc_path)
    wrf_lats = ds_grid['LATIXY'].values
    wrf_lons = normalize_longitude(ds_grid['LONGXY'].values)
    
    wrf_shape = wrf_lats.shape 
    wrf_coords = np.column_stack((wrf_lats.flatten(), wrf_lons.flatten()))
    target_coords = np.column_stack((target_lats, target_lons))
    
    tree = cKDTree(wrf_coords)
    distances, flat_indices = tree.query(target_coords, k=1)
    
    lat_indices, lon_indices = np.unravel_index(flat_indices, wrf_shape)
    
    return lat_indices, lon_indices, df['lat'].values, df['lon'].values, df['cluster'].values

# ==========================================
# 2. THE "MAP" FUNCTION (Runs in Parallel)
# ==========================================
def process_single_year(args):
    """Worker function: Processes exactly 1 year and saves as a final NC file."""
    year, target_files, lat_indices, lon_indices, final_lats, final_lons, final_clusters = args
    
    # Final output name for the year
    output_name = f"temp_wrf_conus_{year}.nc"
    
    # --- ROBUST FILE VERIFICATION ---
    if os.path.exists(output_name):
        is_valid = False
        # 1. Check if file size is > 100 MB (100,000,000 bytes)
        # Completed files should be ~151 MB. Broken stubs are ~1.6 MB.
        if os.path.getsize(output_name) > 100000000:
            try:
                # 2. Try opening it to ensure it isn't corrupted and has exactly 365 days
                with xr.open_dataset(output_name) as check_ds:
                    if check_ds['VPD'].shape[0] == 365:
                        is_valid = True
            except Exception:
                pass # Fails to open -> corrupted
        
        if is_valid:
            return f"? Year {year} already verified and fully complete (Size: {os.path.getsize(output_name)/1e6:.1f} MB). Skipping."
        else:
            print(f"?? Year {year} file is incomplete ({os.path.getsize(output_name)/1024:.0f} KB). Deleting and re-processing...", flush=True)
            os.remove(output_name) # Delete the broken 1.6MB stub
            
    print(f"   -> Worker starting Year {year}...", flush=True)
    
    cell_dim = xr.DataArray(np.arange(len(lat_indices)), dims='cell')
    da_lat_idx = xr.DataArray(lat_indices, dims='cell', coords={'cell': cell_dim})
    da_lon_idx = xr.DataArray(lon_indices, dims='cell', coords={'cell': cell_dim})
    
    monthly_datasets = []
    
    # --- File-by-File Streaming ---
    for idx, filepath in enumerate(target_files, 1):
        filename = os.path.basename(filepath)
        print(f"      [{year}] Loading {idx}/12: {filename}", flush=True)
        
        ds_month = xr.open_dataset(filepath, engine='netcdf4')
        ds_cells = ds_month.isel(lat=da_lat_idx, lon=da_lon_idx)
        
        ds_daily = ds_cells[['TBOT', 'QBOT', 'PSRF', 'PRECTmms']].resample(time='1D').mean()
        
        # ?? BULLETPROOF LEAP DAY FILTER ??
        is_leap_day = (ds_daily.time.dt.month == 2) & (ds_daily.time.dt.day == 29)
        ds_daily = ds_daily.sel(time=~is_leap_day)
        
        monthly_datasets.append(ds_daily)
        ds_month.close() 
        
    print(f"   -> [{year}] Concatenating and computing thermodynamics...", flush=True)
    
    ds_year = xr.concat(monthly_datasets, dim='time')
    
    # Thermodynamics
    tbot_c = ds_year['TBOT'] - 273.15
    es = 0.6112 * np.exp((17.67 * tbot_c) / (tbot_c + 243.5))
    psrf_kpa = ds_year['PSRF'] / 1000.0
    ea = (ds_year['QBOT'] * psrf_kpa) / (0.622 + (1 - 0.622) * ds_year['QBOT'])
    
    vpd = es - ea
    vpd = xr.where(vpd < 0, 0.001, vpd)
    precip_mm_day = ds_year['PRECTmms'] * 86400.0
    
    ds_out = xr.Dataset({
        'VPD': (['time', 'cell'], vpd.data),
        'PRECIP': (['time', 'cell'], precip_mm_day.data),
        'lat': (['cell'], final_lats),
        'lon': (['cell'], final_lons),
        'cluster': (['cell'], final_clusters)
    }, coords={
        'time': ds_year.time,
        'cell': np.arange(len(final_lats))
    })
    
    ds_out['VPD'].attrs = {'units': 'kPa', 'long_name': 'Daily Mean Vapor Pressure Deficit'}
    ds_out['PRECIP'].attrs = {'units': 'mm/day', 'long_name': 'Daily Total Precipitation'}
    
    # Save the year file directly
    ds_out.to_netcdf(output_name, compute=True)
    
    return f"? Year {year} complete!"

# ==========================================
# 3. MASTER PIPELINE
# ==========================================
def master_pipeline():
    all_files = sorted(glob.glob(os.path.join(WRF_DIR, '*.nc')))
    
    files_by_year = {}
    for f in all_files:
        year = int(os.path.basename(f).split('-')[0])
        if START_YEAR <= year <= END_YEAR:
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(f)
            
    if not files_by_year:
        raise ValueError("No WRF files found for the specified years.")
        
    sample_file = files_by_year[START_YEAR][0]
    lat_idx, lon_idx, lats, lons, clusters = build_spatial_mapper_arrays(sample_file, CSV_PATH)
    
    pool_args = []
    for year, files in files_by_year.items():
        pool_args.append((year, files, lat_idx, lon_idx, lats, lons, clusters))
        
    print(f"\n?? Spawning {NUM_WORKERS} parallel workers to process {len(pool_args)} years...", flush=True)
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(process_single_year, pool_args), 1):
            print(f"[{i}/{len(pool_args)} Years Completed] {res}", flush=True)
            
    print("\n? All years processed successfully! Check your directory for the wrf_conus_daily_vpd_precip_YYYY.nc files.")

if __name__ == "__main__":
    master_pipeline()