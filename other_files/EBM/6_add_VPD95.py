# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR     = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist'
WRF_DIR      = os.path.join(BASE_DIR, 'wrf_daily_precip_vpd')
WRF_TEMPLATE = 'wrf_daily_vpd_precip_1981_2015_cluster_{}.nc'

CSV_IN  = os.path.join(BASE_DIR, 'flash_drought_spatial_pattern_analysis',
                       'spatial_flash_drought_properties.csv')
CSV_OUT = os.path.join(BASE_DIR, 'flash_drought_spatial_pattern_analysis',
                       'spatial_flash_drought_properties_with_VPD95.csv')

# WRF already starts at 1981-01-01, 12775 days = 35 years * 365 days (no-leap)
YEARS_TO_KEEP    = 35
DAYS_IN_YEAR     = 365
PENTADS_IN_YEAR  = 73   # 73 * 5 = 365

# Growing Season Pentad Indices (0-indexed), same as GPP script:
# Pentad 18 = April 1-5, Pentad 60 = Oct 27-31
GS_PENTAD_START = 18
GS_PENTAD_END   = 61   # exclusive ? pentads 18..60

# ==============================================================================
# MAIN
# ==============================================================================
print("--- Building Pentad-Based VPD95 Feature ---")

all_cluster_data = []

for cluster_id in range(1, 8):
    print(f"\nProcessing Cluster {cluster_id}...")

    wrf_file = os.path.join(WRF_DIR, WRF_TEMPLATE.format(cluster_id))
    ds = xr.open_dataset(wrf_file)

    # Extract VPD as numpy array; shape is (time=12775, cell=N)
    vpd_raw = ds['VPD'].values   # (12775, N)
    lats    = ds['lat'].values
    lons    = ds['lon'].values
    lons    = np.where(lons > 180, lons - 360.0, lons)

    ds.close()

    # Transpose to (cell, time) to match GPP script convention
    vpd_raw = vpd_raw.T          # (N, 12775)
    n_cells = vpd_raw.shape[0]

    # Verify time length
    expected_days = YEARS_TO_KEEP * DAYS_IN_YEAR
    assert vpd_raw.shape[1] == expected_days, (
        f"Cluster {cluster_id}: expected {expected_days} days, "
        f"got {vpd_raw.shape[1]}. Check if WRF has a spin-up year.")

    # ------------------------------------------------------------------
    # Step A: Reshape into (cells, years, pentads, days_per_pentad)
    # ------------------------------------------------------------------
    vpd_reshaped = vpd_raw.reshape(n_cells, YEARS_TO_KEEP, PENTADS_IN_YEAR, 5)

    # ------------------------------------------------------------------
    # Step B: Average 5 days -> pentad mean
    # Shape: (cells, 35, 73)
    # ------------------------------------------------------------------
    vpd_pentads = np.nanmean(vpd_reshaped, axis=3)

    # ------------------------------------------------------------------
    # Step C: Extract growing season pentads (Apr 1 - Oct 31)
    # Shape: (cells, 35, 43)
    # ------------------------------------------------------------------
    vpd_gs = vpd_pentads[:, :, GS_PENTAD_START:GS_PENTAD_END]

    # ------------------------------------------------------------------
    # Step D: 95th percentile across growing-season pentads per year
    # Shape: (cells, 35)
    # ------------------------------------------------------------------
    vpd_annual_95th = np.nanpercentile(vpd_gs, 95, axis=2)

    # ------------------------------------------------------------------
    # Step E: 35-year mean
    # Shape: (cells,)
    # ------------------------------------------------------------------
    vpd95_feature = np.nanmean(vpd_annual_95th, axis=1)

    df_cluster = pd.DataFrame({
        'lat':   lats.round(4),
        'lon':   lons.round(4),
        'VPD95': vpd95_feature
    })

    all_cluster_data.append(df_cluster)
    print(f"  Done: {n_cells} cells, VPD95 range "
          f"[{vpd95_feature.min():.3f}, {vpd95_feature.max():.3f}] kPa")

# ==============================================================================
# MERGE INTO EXISTING CSV
# ==============================================================================
print("\nMerging into spatial_flash_drought_properties.csv ...")
df_vpd = pd.concat(all_cluster_data, ignore_index=True)
df_vpd = df_vpd.dropna(subset=['VPD95'])

df_main = pd.read_csv(CSV_IN)

for col in ['lat', 'lon']:
    df_main[col] = df_main[col].round(4)
    df_vpd[col]  = df_vpd[col].round(4)

df_out = df_main.merge(df_vpd[['lat', 'lon', 'VPD95']], on=['lat', 'lon'], how='left')

n_missing = df_out['VPD95'].isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} rows had no VPD95 match.")

df_out.to_csv(CSV_OUT, index=False, float_format='%.6f')
print(f"\nCOMPLETE: saved {len(df_out)} rows to:\n  {CSV_OUT}")
print(df_out[['lat', 'lon', 'VPD95']].head())