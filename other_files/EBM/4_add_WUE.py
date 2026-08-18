# -*- coding: utf-8 -*-
import pandas as pd

WUE_PATH = "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/WUE/Historical_Baseline_iWUE_1981_2015.csv"
CSV_IN   = "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties.csv"
CSV_OUT  = "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties_with_wue.csv"

# Load WUE — keep only lat, lon, and the mean column; rename it
wue = pd.read_csv(WUE_PATH, usecols=[0, 1, 3], header=0)
wue.columns = ["lat", "lon", "iWUE_mean"]

# Load main CSV
df = pd.read_csv(CSV_IN)

# Round both to 4 decimal places to avoid floating-point key mismatches
for col in ["lat", "lon"]:
    df[col]  = df[col].round(4)
    wue[col] = wue[col].round(4)

# Merge on lat/lon — left join keeps all rows in df
df = df.merge(wue[["lat", "lon", "iWUE_mean"]], on=["lat", "lon"], how="left")

n_missing = df["iWUE_mean"].isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} rows had no WUE match")

df.to_csv(CSV_OUT, index=False, float_format="%.6f")
print(f"Saved {len(df)} rows to:\n  {CSV_OUT}")
print(df[["lat", "lon", "iWUE_mean"]].head())