import xarray as xr
import numpy as np
import pandas as pd

file1 = '/global/cfs/cdirs/m2702/liliyao/CLMBGC_postprocessing/QAQC/historical_with_LUH2_1980-2015/SOILLIQ_198001-201512_daily.nc'
ds = xr.open_dataset(file1)

lat_array = ds['lat'].values
lon_array = ds['lon'].values
SOILLIQ = ds['SOILLIQ']
n_time = SOILLIQ.shape[0]

# Load your cluster file
df = pd.read_csv("conus_lat_lon_cluster.txt",
                 sep=r"\s+", header=None,
                 names=["lat", "lon", "cluster"])

# Convert 0–360 to -180–180 if needed
# if df['lon'].max() > 180:
#     df['lon'] -= 360.0

def find_index(arr, x):
    return np.abs(arr - x).argmin()

df["lat_idx"] = df["lat"].apply(lambda x: find_index(lat_array, x))
df["lon_idx"] = df["lon"].apply(lambda x: find_index(lon_array, x))

clusters = sorted(df.cluster.unique())

# Precompute layer-sum once (lazy)
SOILLIQ_sum = SOILLIQ[:, :9, :, :].sum(dim="levsoi")

for cid in clusters:
    print(f"\nProcessing cluster {cid} ...")

    sub = df[df.cluster == cid]
    lat_idx = xr.DataArray(sub["lat_idx"].values, dims="cell")
    lon_idx = xr.DataArray(sub["lon_idx"].values, dims="cell")

    print("  cells =", len(sub))

    # Extract directly → shape (time, cell)
    da = SOILLIQ_sum.isel(lat=lat_idx, lon=lon_idx)

    # Transpose to (cell, time) but still Dask-backed
    da = da.transpose("cell", "time")

    # Build output dataset WITHOUT loading to numpy
    out = xr.Dataset(
        {"SOILLIQ_sum_0_9": da.astype("float32")},
        coords={
            "cell": np.arange(len(sub)),
            "lat": ("cell", sub["lat"].values),
            "lon": ("cell", sub["lon"].values),
            "time": ds["time"]
        }
    )

    # Write directly (streaming)
    out.to_netcdf(
        f"sm_cluster_{cid}.nc",
        compute=True,
        encoding={"SOILLIQ_sum_0_9": {"zlib": True, "complevel": 4}}
    )

    print(f"  Saved sm_cluster_{cid}.nc")
