import pandas as pd
import numpy as np

DROUGHT_CSV = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties.csv'
GPP_CSV = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/Gridcell_GPP_Greening_Feature.csv'
OUTPUT_CSV = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties.csv'

df_drought = pd.read_csv(DROUGHT_CSV)
df_gpp = pd.read_csv(GPP_CSV)

# Drop GPP95 column if it already exists
if 'GPP95' in df_drought.columns:
    df_drought = df_drought.drop(columns=['GPP95'])
    print("Existing GPP95 column removed.")

df_gpp = df_gpp.rename(columns={'gpp_greening_index': 'GPP95'})
df_gpp['lat'] = df_gpp['lat'].round(4)
df_gpp['lon'] = df_gpp['lon'].round(4)
df_drought['lat'] = df_drought['lat'].round(4)
df_drought['lon'] = df_drought['lon'].round(4)

df_merged = df_drought.merge(df_gpp[['lat', 'lon', 'GPP95']], on=['lat', 'lon'], how='left')

n_matched = df_merged['GPP95'].notna().sum()
n_missing = df_merged['GPP95'].isna().sum()
print("Matched: " + str(n_matched))
print("Unmatched: " + str(n_missing))

# Round decimal columns to 2 digits
cols_to_round = ['ratio_flash', 'ratio_slow', 'aridity_index', 'GPP95']
for col in cols_to_round:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].round(2)

df_merged.to_csv(OUTPUT_CSV, index=False)
print("Done! Saved to: " + OUTPUT_CSV)
print(df_merged.head())