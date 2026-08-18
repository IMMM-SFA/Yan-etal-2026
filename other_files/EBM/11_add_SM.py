# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# -- Paths ------------------------------------------------------------------
SM_FILE  = "/global/cfs/cdirs/m2702/hongxiang/drought_propa_conus/hist/monthly_sm.txt"
CSV_FILE = ("/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/"
            "flash_drought_spatial_pattern_analysis/"
            "spatial_flash_drought_properties.csv")

# -- 1. Load soil-moisture data ---------------------------------------------
sm = pd.read_csv(SM_FILE, sep="\t")

# Build column index: all columns after lat/lon are year/month strings
time_cols = [c for c in sm.columns if c not in ("lat", "lon")]

# Parse each column name into (year, month)
def parse_ym(col):          # e.g. "1981/4" ? (1981, 4)
    y, m = col.split("/")
    return int(y), int(m)

parsed = {col: parse_ym(col) for col in time_cols}

# -- 2. Filter to 1981–2015, select Apr–Jun and Apr–Oct columns -----------
apr_jun_cols = [c for c in time_cols
                if 1981 <= parsed[c][0] <= 2015
                and 4 <= parsed[c][1] <= 6]

mar_cols = [c for c in time_cols
            if 1981 <= parsed[c][0] <= 2015
            and parsed[c][1] == 3]


apr_oct_cols = [c for c in time_cols
                if 1981 <= parsed[c][0] <= 2015
                and 4 <= parsed[c][1] <= 10]

# -- 3. Compute annual means per grid cell, then average across years ------
# Group by year so we get a proper annual mean first, then average those

def annual_seasonal_mean(df, cols, months):
    """
    For each year, average the selected months, then return the
    grand mean across all years (one value per row / grid cell).
    """
    years = sorted({parsed[c][0] for c in cols})
    annual_vals = []
    for yr in years:
        yr_cols = [c for c in cols if parsed[c][0] == yr]
        annual_vals.append(df[yr_cols].mean(axis=1))
    return pd.concat(annual_vals, axis=1).mean(axis=1)

sm["SM4_6"]  = annual_seasonal_mean(sm, apr_jun_cols,  months=range(4, 7))
sm["SM4_10"] = annual_seasonal_mean(sm, apr_oct_cols,  months=range(4, 11))
sm["SM3"]    = annual_seasonal_mean(sm, mar_cols,      months=range(3, 4))

sm_means = sm[["lat", "lon", "SM3", "SM4_6", "SM4_10"]]

# -- 4. Merge into the properties CSV --------------------------------------
props = pd.read_csv(CSV_FILE)

# Round to avoid floating-point join mismatches
for df in (props, sm_means):
    df["lat"] = df["lat"].round(6)
    df["lon"] = df["lon"].round(6)

# The CSV uses negative longitudes; monthly_sm.txt uses 0–360 — normalise if needed
if sm_means["lon"].max() > 180:
    sm_means = sm_means.copy()
    sm_means["lon"] = ((sm_means["lon"] + 180) % 360) - 180
    sm_means["lon"] = sm_means["lon"].round(6)

merged = props.merge(sm_means, on=["lat", "lon"], how="left")

missing = merged["SM4_6"].isna().sum()
if missing:
    print(f"Warning: {missing} rows in the CSV had no matching SM grid cell.")

# -- 5. Save ----------------------------------------------------------------
merged.to_csv(CSV_FILE, index=False)
print(f"Done. Columns added: SM3, SM4_6, SM4_10")
print(merged[["lat", "lon", "SM3", "SM4_6", "SM4_10"]].head())