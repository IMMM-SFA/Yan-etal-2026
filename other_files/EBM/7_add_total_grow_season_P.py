# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import netCDF4 as nc
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
                       'spatial_flash_drought_properties_with_precip.csv')

# WRF: 12775 days = 35 years x 365 days (no-leap), starting 1981-01-01
YEARS_TO_KEEP = 35
DAYS_IN_YEAR  = 365

# Build day-of-year month labels for a no-leap 365-day year
# Used to identify April (month 4) through October (month 7) days
_doy = pd.date_range('2001-01-01', periods=365, freq='D')  # any non-leap year
MONTH_OF_DAY = _doy.month.values  # shape (365,) — month index for each doy

# Growing season mask: April=4 through October=10
GS_DAY_MASK = (MONTH_OF_DAY >= 4) & (MONTH_OF_DAY <= 10)  # 214 days

# ==============================================================================
# MAIN
# ==============================================================================
print("--- Building Growing Season Total Precipitation Feature ---")
print(f"Growing season days per year: {GS_DAY_MASK.sum()} (Apr-Oct, no-leap)")

all_cluster_data = []

for cluster_id in range(1, 8):
    print(f"\nProcessing Cluster {cluster_id}...")

    wrf_file = os.path.join(WRF_DIR, WRF_TEMPLATE.format(cluster_id))
    ds = nc.Dataset(wrf_file)

    # Read PRECIP: shape (time=12775, cell=N)
    precip_raw = ds.variables['PRECIP'][:]   # masked array
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    ds.close()

    # Convert to plain numpy float32, fill NaN
    precip_raw = np.where(np.ma.getmaskarray(precip_raw),
                          np.nan, np.array(precip_raw, dtype=np.float32))

    # Normalize longitude
    lons = np.where(lons > 180, lons - 360.0, lons)

    # Transpose to (cell, time)
    precip = precip_raw.T   # (N, 12775)
    n_cells = precip.shape[0]

    expected_days = YEARS_TO_KEEP * DAYS_IN_YEAR
    assert precip.shape[1] == expected_days, (
        f"Cluster {cluster_id}: expected {expected_days} days, "
        f"got {precip.shape[1]}.")

    # ------------------------------------------------------------------
    # Step A: Reshape into (cells, years, 365)
    # ------------------------------------------------------------------
    precip_3d = precip.reshape(n_cells, YEARS_TO_KEEP, DAYS_IN_YEAR)
    # Shape: (N, 35, 365)

    # ------------------------------------------------------------------
    # Step B: Select growing season days only (Apr-Oct = 214 days/yr)
    # ------------------------------------------------------------------
    precip_gs = precip_3d[:, :, GS_DAY_MASK]
    # Shape: (N, 35, 214)

    # ------------------------------------------------------------------
    # Step C: Sum over growing season days for each year
    # ------------------------------------------------------------------
    precip_annual_sum = np.nansum(precip_gs, axis=2)
    # Shape: (N, 35)

    # If ALL days in a year were NaN, nansum returns 0 — set those back to NaN
    all_nan_mask = np.all(np.isnan(precip_gs), axis=2)
    precip_annual_sum[all_nan_mask] = np.nan

    # ------------------------------------------------------------------
    # Step D: 35-year mean of annual growing season totals
    # ------------------------------------------------------------------
    precip_gs_mean = np.nanmean(precip_annual_sum, axis=1)
    # Shape: (N,)

    df_cluster = pd.DataFrame({
        'lat':          lats.round(4),
        'lon':          lons.round(4),
        'precip_gs_mean': precip_gs_mean
    })

    all_cluster_data.append(df_cluster)
    print(f"  Done: {n_cells} cells, precip_gs_mean range "
          f"[{np.nanmin(precip_gs_mean):.1f}, {np.nanmax(precip_gs_mean):.1f}] mm")

# ==============================================================================
# MERGE INTO EXISTING CSV
# ==============================================================================
print("\nMerging into spatial_flash_drought_properties.csv ...")
df_precip = pd.concat(all_cluster_data, ignore_index=True)
df_precip = df_precip.dropna(subset=['precip_gs_mean'])

df_main = pd.read_csv(CSV_IN)

for col in ['lat', 'lon']:
    df_main[col]   = df_main[col].round(4)
    df_precip[col] = df_precip[col].round(4)

df_out = df_main.merge(df_precip[['lat', 'lon', 'precip_gs_mean']],
                       on=['lat', 'lon'], how='left')

n_missing = df_out['precip_gs_mean'].isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} rows had no precip match")

df_out.to_csv(CSV_OUT, index=False, float_format='%.6f')
print(f"\nCOMPLETE: saved {len(df_out)} rows to:\n  {CSV_OUT}")
print(df_out[['lat', 'lon', 'precip_gs_mean']].head())