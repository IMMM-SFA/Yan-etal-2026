# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

# -- PATHS --------------------------------------------------------------------
CLUSTER_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/wrf_daily_precip_vpd_tair_solar'
CSV_PATH    = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties.csv'
# -----------------------------------------------------------------------------

def main():
    # -- Load CSV ----------------------------------------------------------
    print("Loading CSV...", flush=True)
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows loaded.")

    # Build a lookup key from rounded lat/lon for safe matching
    df['_key'] = list(zip(df['lat'].round(6), df['lon'].round(6)))

    solar_dict = {}   # key -> solar_mean value

    # -- Process each cluster file -----------------------------------------
    cluster_files = sorted(glob.glob(os.path.join(CLUSTER_DIR, 'wrf_daily_solar_tair_1981_2015_cluster_*.nc')))

    if not cluster_files:
        raise FileNotFoundError(f"No cluster files found in:\n  {CLUSTER_DIR}")

    print(f"\nFound {len(cluster_files)} cluster files.", flush=True)

    for fpath in cluster_files:
        cname = os.path.basename(fpath)
        print(f"\nProcessing {cname}...", flush=True)

        ds = xr.open_dataset(fpath)

        # -- Step 1: Filter April–October ---------------------------------
        ds_gs = ds.sel(time=ds.time.dt.month.isin(range(4, 11)))
        n_days = ds_gs.sizes['time']
        print(f"  April-Oct days selected: {n_days}  (expected ~214 x 35 = {214*35})", flush=True)

        # -- Step 2: Mean over all selected days ? (cell,) ----------------
        solar_mean = ds_gs['SOLAR'].mean(dim='time').values   # shape: (cell,)

        lats = ds['lat'].values
        lons = ds['lon'].values

        # -- Step 3: Store in dict keyed by (lat, lon) --------------------
        for i in range(len(solar_mean)):
            key = (round(float(lats[i]), 6), round(float(lons[i]), 6))
            solar_dict[key] = float(solar_mean[i])

        ds.close()
        print(f"  Cells processed: {len(solar_mean)}", flush=True)

    # -- Map solar_mean back to CSV rows -----------------------------------
    print("\nMerging solar_mean into CSV...", flush=True)
    df['solar_mean'] = df['_key'].map(solar_dict)

    n_matched  = df['solar_mean'].notna().sum()
    n_missing  = df['solar_mean'].isna().sum()
    print(f"  Matched : {n_matched} rows")
    print(f"  Missing : {n_missing} rows")

    if n_missing > 0:
        print("  WARNING: some rows had no matching cell. Check lat/lon precision.")

    df = df.drop(columns=['_key'])

    # -- Save back to CSV --------------------------------------------------
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved updated CSV to:\n  {CSV_PATH}")
    print(f"solar_mean stats:")
    print(df['solar_mean'].describe().to_string())
    print("\nDone!")

if __name__ == "__main__":
    main()