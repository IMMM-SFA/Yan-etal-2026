import os
import glob
import numpy as np
import xarray as xr

def merge_and_split():
    print("============================================================")
    print("?? STEP 1: LOCATING YEARLY WRF FILES")
    print("============================================================")
    
    # Grab all 35 files and sort them chronologically
    files = sorted(glob.glob('temp_wrf_conus_*.nc'))
    
    if len(files) == 0:
        raise FileNotFoundError("No 'temp_wrf_conus_YYYY.nc' files found in this directory.")
        
    print(f"?? Found {len(files)} yearly files.")
        
    print("?? Building Xarray Dataset...")
    # FIX APPLIED: data_vars='minimal' prevents xarray from incorrectly 
    # stacking 1D variables (lat, lon, cluster) along the time dimension.
    ds_master = xr.open_mfdataset(
        files, 
        combine='nested', 
        concat_dim='time', 
        data_vars='minimal',
        coords='minimal',
        compat='override',
        parallel=False,
        engine='netcdf4'
    )
    
    print("\n============================================================")
    print("?? STEP 2: LOADING INTO RAM (THE 'REDUCE' PHASE)")
    print("============================================================")
    print("?? Pulling the entire 35-year array into memory (~5.5 GB)...")
    ds_master = ds_master.load()
    
    # ?? BULLETPROOF SAFEGUARD ??
    # Force lat, lon, and cluster back to 1D just in case the NetCDF headers forced a stack
    for var in ['lat', 'lon', 'cluster']:
        if 'time' in ds_master[var].dims:
            ds_master[var] = ds_master[var].isel(time=0)
    
    # Verification checks
    total_days = ds_master.sizes['time']
    expected_days = len(files) * 365
    print(f"?? Total Days Loaded: {total_days} (Expected: {expected_days})")

    print("\n============================================================")
    print("?? STEP 3: SPLITTING BY CLUSTER")
    print("============================================================")
    
    # cluster_array is now guaranteed to be a 1D vector (52682,)
    cluster_array = ds_master['cluster'].values
    
    for cluster_id in range(1, 8):
        mask = (cluster_array == cluster_id)
        num_cells = mask.sum()
        
        if num_cells > 0:
            print(f"?? Processing Cluster {cluster_id} ({num_cells} grid cells)...")
            
            ds_cluster = ds_master.isel(cell=mask)
            ds_cluster = ds_cluster.assign_coords(cell=np.arange(num_cells))
            
            output_name = f'./wrf_daily_precip_vpd/wrf_daily_vpd_precip_1981_2015_cluster_{cluster_id}.nc'
            ds_cluster.to_netcdf(output_name)
            print(f"   ? Saved successfully: {output_name}")
        else:
            print(f"?? Cluster {cluster_id} has 0 cells. Skipping.")

    print("\n============================================================")
    print("?? ALL PROCESSING COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    merge_and_split()