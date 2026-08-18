import xarray as xr
import os

# --- Configuration ---
input_file = '/global/cfs/cdirs/m2702/liliyao/inputdata/cesm_inputdata/lnd/clm2/surfdata_map/landuse.timeseries_0.125nldas2_SSP5-8.5_78_CMIP6_1980-2019_c231122.nc'
output_dir = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/'
output_file = os.path.join(output_dir, 'veg_classes_fraction_1980-2015.nc')

print(f"Opening {os.path.basename(input_file)}...")
# Lazy loading to prevent NERSC memory overload
ds = xr.open_dataset(input_file, chunks={'time': 1})

# --- 1. Subset Time (1980-2015) ---
print("Subsetting time for 1980 to 2015...")
try:
    ds = ds.sel(time=slice(1980, 2015))
except KeyError:
    ds = ds.sel(time=slice('1980', '2015'))

# --- 2. Extract Base Landunits ---
print("Extracting landunit variables...")
pct_crop = ds['PCT_CROP']
pct_lake = ds['PCT_LAKE']
pct_urban_total = ds['PCT_URBAN'].sum(dim='numurbl')
pct_glacier = ds['PCT_GLACIER'] if 'PCT_GLACIER' in ds else 0.0

# Total Natural Vegetation fraction of the grid cell
natveg_frac = (100.0 - pct_crop - pct_lake - pct_urban_total - pct_glacier) / 100.0

# --- 3. Extract PFT Specific Fractions ---
print("Extracting specific PFTs...")

# A. Crop Fraction (Already total grid cell fraction)
crop_frac = pct_crop / 100.0

# B. Forest Fraction (PFT indices 1 through 8 in CLM5)
# Note: natpft=0 is bare soil, 1-8 are trees
tree_pft_sum = ds['PCT_NAT_PFT'].isel(natpft=slice(1, 9)).sum(dim='natpft')
forest_frac = natveg_frac * (tree_pft_sum / 100.0)

# C. Shrub Fraction (PFT indices 9 through 11 in CLM5)
shrub_pft_sum = ds['PCT_NAT_PFT'].isel(natpft=slice(9, 12)).sum(dim='natpft')
shrub_frac = natveg_frac * (shrub_pft_sum / 100.0)

# D. Grass Fraction (PFT indices 12 through 14 in CLM5)
grass_pft_sum = ds['PCT_NAT_PFT'].isel(natpft=slice(12, 15)).sum(dim='natpft')
grass_frac = natveg_frac * (grass_pft_sum / 100.0)

# --- 4. Construct Output Dataset ---
print("Building output NetCDF...")
ds_out = xr.Dataset()
ds_out['time'] = ds['time']

if 'LATIXY' in ds:
    ds_out['lat'] = ds['LATIXY'].isel(time=0) if 'time' in ds['LATIXY'].dims else ds['LATIXY']
if 'LONGXY' in ds:
    ds_out['lon'] = ds['LONGXY'].isel(time=0) if 'time' in ds['LONGXY'].dims else ds['LONGXY']

# Assign Data Variables
ds_out['Crop_Fraction'] = crop_frac
ds_out['Crop_Fraction'].attrs = {'units': 'fraction', 'long_name': 'Total Grid Cell Fraction of Crops'}

ds_out['Forest_Fraction'] = forest_frac
ds_out['Forest_Fraction'].attrs = {'units': 'fraction', 'long_name': 'Total Grid Cell Fraction of Trees (PFTs 1-8)'}

ds_out['Shrub_Fraction'] = shrub_frac
ds_out['Shrub_Fraction'].attrs = {'units': 'fraction', 'long_name': 'Total Grid Cell Fraction of Shrubs (PFTs 9-11)'}

ds_out['Grass_Fraction'] = grass_frac
ds_out['Grass_Fraction'].attrs = {'units': 'fraction', 'long_name': 'Total Grid Cell Fraction of Grasses (PFTs 12-14)'}

# --- 5. Save and Close ---
print(f"Saving to {output_file}...")
ds_out.to_netcdf(output_file)
ds.close()
print("Done!")