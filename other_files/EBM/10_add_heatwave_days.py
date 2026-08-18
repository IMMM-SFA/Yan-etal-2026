# -*- coding: utf-8 -*-
"""
compute_heatwave_days_append_csv.py
=====================================
Heatwave day metric using Moving Climatological Baseline Window (MCB).

Method (per cell, per calendar day D in April-Oct):
  1. Gather Tair values from a 15-day window centered on D
     (+-7 days) across ALL 35 baseline years -> ~15 x 35 = 525 values
  2. Compute 95th percentile -> threshold(D), spatially and temporally varying
  3. Flag a day as exceeding threshold if Tair(day) > threshold(D)
  4. Only count flagged days that belong to a streak of >= 3 consecutive
     exceedance days (following HWMI definition)
  5. Sum qualifying heatwave days per cell across all years, divide by 35
     -> mean annual heatwave days

Output column added to CSV: 'heatwave_days'  [days/year]
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

CLUSTER_DIR  = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/wrf_daily_precip_vpd_tair_solar'
CSV_PATH     = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties.csv'

MONTHS_GS    = list(range(4, 11))   # April-October
WINDOW_HALF  = 7                     # +-7 days -> 15-day window
PERCENTILE   = 95
MIN_CONSEC   = 3                     # minimum consecutive exceedance days to qualify
N_YEARS      = 35


def count_consecutive_heatwave_days(exceedance_matrix):
    """
    Given a boolean matrix of shape (n_days, n_cells),
    return an integer array of shape (n_cells,) counting only the days
    that belong to a streak of >= MIN_CONSEC consecutive True values
    along the time axis (axis=0).

    Works column-by-column using run-length logic via numpy diff.
    """
    n_days, n_cells = exceedance_matrix.shape
    hw_count = np.zeros(n_cells, dtype=np.float32)

    for c in range(n_cells):
        col = exceedance_matrix[:, c].astype(np.int8)

        # find starts and ends of True runs
        padded = np.concatenate(([0], col, [0]))
        diff   = np.diff(padded)
        starts = np.where(diff == 1)[0]   # index where run begins
        ends   = np.where(diff == -1)[0]  # index where run ends (exclusive)

        for s, e in zip(starts, ends):
            run_len = e - s
            if run_len >= MIN_CONSEC:
                hw_count[c] += run_len

    return hw_count


def compute_heatwave_days_one_cluster(fpath: str):
    """
    Returns (lats, lons, heatwave_days_per_year) arrays, each shape (n_cells,).
    """
    print(f"  Opening {os.path.basename(fpath)}...", flush=True)
    ds = xr.open_dataset(fpath)

    # -- Filter April-October ----------------------------------------------
    ds_gs = ds.sel(time=ds.time.dt.month.isin(MONTHS_GS))

    tair = ds_gs['TAIR'].values          # (n_days_gs, n_cells)  [degC]
    lats = ds['lat'].values              # (n_cells,)
    lons = ds['lon'].values              # (n_cells,)

    # FIX: use xarray .dt directly — works with cftime NoLeap calendar
    doys = ds_gs.time.dt.dayofyear.values    # (n_days_gs,)

    n_days, n_cells = tair.shape
    unique_doys = np.unique(doys)

    print(f"  April-Oct days: {n_days}  |  unique DOYs: {len(unique_doys)}", flush=True)
    print(f"  Computing MCB {PERCENTILE}th percentile thresholds + "
          f">={MIN_CONSEC}-day consecutive filter...", flush=True)

    # -- Build full exceedance boolean matrix (n_days, n_cells) -----------
    exceedance = np.zeros((n_days, n_cells), dtype=bool)

    for doy in unique_doys:
        # 15-day window across all years
        window_doys  = np.arange(doy - WINDOW_HALF, doy + WINDOW_HALF + 1)
        window_mask  = np.isin(doys, window_doys)
        window_tair  = tair[window_mask, :]                  # (~525, n_cells)

        # 95th percentile threshold per cell
        threshold = np.percentile(window_tair, PERCENTILE, axis=0)  # (n_cells,)

        # flag exact days matching this DOY
        this_doy_mask = (doys == doy)
        exceedance[this_doy_mask, :] = (
            tair[this_doy_mask, :] > threshold[np.newaxis, :]
        )

    # -- Apply consecutive-days filter (>= MIN_CONSEC) --------------------
    # NOTE: we process the full April-Oct time series as one block per year
    # so streaks are correctly identified within each year but not across
    # the Apr-Oct boundary (which is physically correct).
    years      = ds_gs.time.dt.year.values   # works with cftime too
    unique_yrs = np.unique(years)

    total_hw_count = np.zeros(n_cells, dtype=np.float32)

    for yr in unique_yrs:
        yr_mask  = (years == yr)
        yr_exceedance = exceedance[yr_mask, :]               # (days_in_yr, n_cells)
        total_hw_count += count_consecutive_heatwave_days(yr_exceedance)

    # -- Mean annual heatwave days -----------------------------------------
    heatwave_days_per_year = total_hw_count / N_YEARS

    ds.close()
    print(f"  Done. Heatwave days/year range: "
          f"{heatwave_days_per_year.min():.1f} - {heatwave_days_per_year.max():.1f}", flush=True)

    return lats, lons, heatwave_days_per_year


def main():
    # -- Load CSV ----------------------------------------------------------
    print("Loading CSV...", flush=True)
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows loaded.")
    df['_key'] = list(zip(df['lat'].round(6), df['lon'].round(6)))

    hw_dict = {}

    # -- Process each cluster file -----------------------------------------
    cluster_files = sorted(glob.glob(
        os.path.join(CLUSTER_DIR, 'wrf_daily_solar_tair_1981_2015_cluster_*.nc')
    ))
    if not cluster_files:
        raise FileNotFoundError(f"No cluster files found in:\n  {CLUSTER_DIR}")

    print(f"\nFound {len(cluster_files)} cluster files.\n", flush=True)

    for fpath in cluster_files:
        t0 = datetime.now()
        lats, lons, hw_days = compute_heatwave_days_one_cluster(fpath)

        for i in range(len(hw_days)):
            key = (round(float(lats[i]), 6), round(float(lons[i]), 6))
            hw_dict[key] = float(hw_days[i])

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  Cluster done in {elapsed:.1f}s\n", flush=True)

    # -- Merge into CSV ----------------------------------------------------
    print("Merging heatwave_days into CSV...", flush=True)
    df['heatwave_days'] = df['_key'].map(hw_dict)

    n_matched = df['heatwave_days'].notna().sum()
    n_missing = df['heatwave_days'].isna().sum()
    print(f"  Matched : {n_matched} rows")
    print(f"  Missing : {n_missing} rows")
    if n_missing > 0:
        print("  WARNING: some rows had no match. Check lat/lon precision.")

    df = df.drop(columns=['_key'])
    df.to_csv(CSV_PATH, index=False)

    print(f"\nSaved updated CSV to:\n  {CSV_PATH}")
    print("\nheatwave_days stats (days/year):")
    print(df['heatwave_days'].describe().to_string())
    print("\nDone!")


if __name__ == "__main__":
    main()