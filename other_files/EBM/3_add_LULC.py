# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import netCDF4 as nc

# Paths
NC_PATH = "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/veg_classes_fraction_1980-2015.nc"
CSV_IN  = "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties.csv"
CSV_OUT = "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties_with_veg.csv"

# --- Load NetCDF and compute 35-year mean (skip year index 0) ---
ds = nc.Dataset(NC_PATH)

lat2d = ds.variables["lat"][:]
lon2d = ds.variables["lon"][:]

# Normalize longitude to [-180, 180] in case the NetCDF uses [0, 360]
lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

# time axis has 36 years; skip index 0 ? use indices 1–35
crop   = np.nanmean(ds.variables["Crop_Fraction"  ][1:, :, :], axis=0)
forest = np.nanmean(ds.variables["Forest_Fraction"][1:, :, :], axis=0)
grass  = np.nanmean(ds.variables["Grass_Fraction" ][1:, :, :], axis=0)
shrub  = np.nanmean(ds.variables["Shrub_Fraction" ][1:, :, :], axis=0)

ds.close()

# --- Flatten 2-D grids to 1-D for fast nearest-neighbour lookup ---
lat_flat    = lat2d.ravel()
lon_flat    = lon2d.ravel()   # now guaranteed [-180, 180]
crop_flat   = crop.ravel()
forest_flat = forest.ravel()
grass_flat  = grass.ravel()
shrub_flat  = shrub.ravel()

# --- Load CSV and match each point to nearest grid cell ---
df = pd.read_csv(CSV_IN)

def find_nearest(target_lat, target_lon):
    """Return flat index of the grid cell closest to (target_lat, target_lon)."""
    dist = (lat_flat - target_lat) ** 2 + (lon_flat - target_lon) ** 2
    return np.argmin(dist)

crop_vals, forest_vals, grass_vals, shrub_vals = [], [], [], []

for _, row in df.iterrows():
    idx = find_nearest(row["lat"], row["lon"])
    crop_vals.append(  float(np.clip(crop_flat[idx],   0, 1)))
    forest_vals.append(float(np.clip(forest_flat[idx], 0, 1)))
    grass_vals.append( float(np.clip(grass_flat[idx],  0, 1)))
    shrub_vals.append( float(np.clip(shrub_flat[idx],  0, 1)))

df["crop_fraction"]   = crop_vals
df["forest_fraction"] = forest_vals
df["grass_fraction"]  = grass_vals
df["shrub_fraction"]  = shrub_vals

# --- Write output ---
df.to_csv(CSV_OUT, index=False, float_format="%.6f")
print(f"Saved {len(df)} rows to:\n  {CSV_OUT}")
print(df[["lat","lon","crop_fraction","forest_fraction","grass_fraction","shrub_fraction"]].head())