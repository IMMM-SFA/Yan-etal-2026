# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
import warnings
from numpy.lib.stride_tricks import as_strided

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
                       'spatial_flash_drought_properties_with_maxdrydays.csv')

# WRF: 12775 days = 35 years x 365 days (no-leap), starting 1981-01-01
YEARS_TO_KEEP = 35
DAYS_IN_YEAR  = 365

# Dry day threshold: standard climatological definition (WMO)
DRY_THRESHOLD_MM = 1.0  # day is "dry" if precip < 1.0 mm

# Build day-of-year month labels for a no-leap 365-day year
_doy = pd.date_range('2001-01-01', periods=365, freq='D')
MONTH_OF_DAY = _doy.month.values  # (365,)
GS_DAY_MASK  = (MONTH_OF_DAY >= 4) & (MONTH_OF_DAY <= 10)  # 214 days, Apr-Oct
GS_N_DAYS    = int(GS_DAY_MASK.sum())  # 214

# ==============================================================================
# HELPER: vectorised max consecutive dry days across all cells for one year
# ==============================================================================
def max_consecutive_ones(bool_array):
    """
    Find the longest run of True values in each row of a 2-D boolean array.

    Parameters
    ----------
    bool_array : np.ndarray, shape (n_cells, n_days), dtype bool
        True  = dry day
        False = wet day or NaN day

    Returns
    -------
    np.ndarray, shape (n_cells,), dtype int
        Maximum consecutive dry days per cell.
    """
    n_cells, n_days = bool_array.shape
    max_runs = np.zeros(n_cells, dtype=np.int32)

    # Pad with a False column on each side so run-end detection works at edges
    padded = np.concatenate(
        [np.zeros((n_cells, 1), dtype=bool),
         bool_array,
         np.zeros((n_cells, 1), dtype=bool)],
        axis=1
    )  # (n_cells, n_days+2)

    # Detect transitions: +1 = run starts, -1 = run ends
    diff = np.diff(padded.astype(np.int8), axis=1)  # (n_cells, n_days+1)

    for i in range(n_cells):
        starts = np.where(diff[i] == 1)[0]
        ends   = np.where(diff[i] == -1)[0]
        if len(starts) == 0:
            max_runs[i] = 0
        else:
            max_runs[i] = int(np.max(ends - starts))

    return max_runs


# ==============================================================================
# MAIN
# ==============================================================================
#print("--- Building Max Consecutive Dry Days (Growing Season) Feature ---")
#print(f"Dry day threshold : precip < {DRY_THRESHOLD_MM} mm/day")
#print(f"Growing season    : April 1 – October 31 ({GS_N_DAYS} days/year, no-leap)")

all_cluster_data = []

for cluster_id in range(1, 8):
    print(f"\nProcessing Cluster {cluster_id}...")

    wrf_file = os.path.join(WRF_DIR, WRF_TEMPLATE.format(cluster_id))
    ds = nc.Dataset(wrf_file)

    precip_raw = ds.variables['PRECIP'][:]
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    ds.close()

    # Fill masked values with NaN
    precip_raw = np.where(np.ma.getmaskarray(precip_raw),
                          np.nan, np.array(precip_raw, dtype=np.float32))

    lons = np.where(lons > 180, lons - 360.0, lons)

    # Transpose to (cell, time)
    precip = precip_raw.T          # (N, 12775)
    n_cells = precip.shape[0]

    expected_days = YEARS_TO_KEEP * DAYS_IN_YEAR
    assert precip.shape[1] == expected_days, (
        f"Cluster {cluster_id}: expected {expected_days} days, got {precip.shape[1]}.")

    # ------------------------------------------------------------------
    # Step A: Reshape to (cells, years, 365)
    # ------------------------------------------------------------------
    precip_3d = precip.reshape(n_cells, YEARS_TO_KEEP, DAYS_IN_YEAR)

    # ------------------------------------------------------------------
    # Step B: Extract growing season days ? (cells, years, 214)
    # ------------------------------------------------------------------
    precip_gs = precip_3d[:, :, GS_DAY_MASK]

    # ------------------------------------------------------------------
    # Step C: Convert to dry-day boolean
    #   True  = dry  (precip < threshold, and not NaN)
    #   False = wet or missing
    # ------------------------------------------------------------------
    dry_bool = (precip_gs < DRY_THRESHOLD_MM) & (~np.isnan(precip_gs))
    # Shape: (n_cells, 35, 214)

    # ------------------------------------------------------------------
    # Step D: For each year, find max consecutive dry days per cell
    # ------------------------------------------------------------------
    annual_max_dry = np.zeros((n_cells, YEARS_TO_KEEP), dtype=np.int32)

    for yr in range(YEARS_TO_KEEP):
        # bool_array shape: (n_cells, 214)
        annual_max_dry[:, yr] = max_consecutive_ones(dry_bool[:, yr, :])

    # ------------------------------------------------------------------
    # Step E: 35-year mean
    # ------------------------------------------------------------------
    max_dry_days_mean = np.mean(annual_max_dry, axis=1).astype(np.float32)
    # Shape: (n_cells,)

    df_cluster = pd.DataFrame({
        'lat':          lats.round(4),
        'lon':          lons.round(4),
        'max_dry_days': max_dry_days_mean
    })

    all_cluster_data.append(df_cluster)
    print(f"  Done: {n_cells} cells, max_dry_days range "
          f"[{max_dry_days_mean.min():.1f}, {max_dry_days_mean.max():.1f}] days")

# ==============================================================================
# MERGE INTO EXISTING CSV
# ==============================================================================
print("\nMerging into spatial_flash_drought_properties.csv ...")
df_dry = pd.concat(all_cluster_data, ignore_index=True)

df_main = pd.read_csv(CSV_IN)

for col in ['lat', 'lon']:
    df_main[col] = df_main[col].round(4)
    df_dry[col]  = df_dry[col].round(4)

df_out = df_main.merge(df_dry[['lat', 'lon', 'max_dry_days']],
                       on=['lat', 'lon'], how='left')

n_missing = df_out['max_dry_days'].isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} rows had no match ")

df_out.to_csv(CSV_OUT, index=False, float_format='%.6f')
print(f"\nCOMPLETE: saved {len(df_out)} rows to:\n  {CSV_OUT}")
print(df_out[['lat', 'lon', 'max_dry_days']].head())