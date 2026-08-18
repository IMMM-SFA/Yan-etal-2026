# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
DATA_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_propa_conus/hist'
CSV_PATH = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties.csv'
OUTPUT_PATH = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist/flash_drought_spatial_pattern_analysis/spatial_flash_drought_properties_with_aridity.csv'

PET_FILE = os.path.join(DATA_DIR, 'monthly_PET.txt')
PPT_FILE = os.path.join(DATA_DIR, 'monthly_precip.txt')

YEAR_START = 1981
YEAR_END = 2015

# LOAD MONTHLY DATA FILES
def load_monthly_file(filepath):
    print("Loading: " + filepath)
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]: 'lat', df.columns[1]: 'lon'}, inplace=True)
    df['lon'] = np.where(df['lon'] > 180, df['lon'] - 360.0, df['lon'])
    df['lat'] = df['lat'].round(6)
    df['lon'] = df['lon'].round(6)
    return df

print("Loading PET and Precip files...")
df_pet = load_monthly_file(PET_FILE)
df_ppt = load_monthly_file(PPT_FILE)

# GET COLUMNS FOR TARGET YEARS
def get_year_month_cols(df, year_start, year_end):
    cols = []
    for col in df.columns:
        if '/' in col:
            try:
                yr = int(col.split('/')[0])
                if year_start <= yr <= year_end:
                    cols.append(col)
            except ValueError:
                continue
    return cols

pet_cols = get_year_month_cols(df_pet, YEAR_START, YEAR_END)
ppt_cols = get_year_month_cols(df_ppt, YEAR_START, YEAR_END)

print("PET columns found: " + str(len(pet_cols)))
print("PPT columns found: " + str(len(ppt_cols)))

assert set(pet_cols) == set(ppt_cols), "PET and Precip time columns do not match!"

# COMPUTE ANNUAL SUMS AND MEAN ARIDITY INDEX
print("Computing aridity index per grid cell...")

years = list(range(YEAR_START, YEAR_END + 1))

year_to_cols = {}
for yr in years:
    year_to_cols[yr] = [c for c in pet_cols if int(c.split('/')[0]) == yr]

pet_annual = pd.DataFrame(index=df_pet.index)
ppt_annual = pd.DataFrame(index=df_ppt.index)

for yr, cols in year_to_cols.items():
    pet_annual[yr] = df_pet[cols].sum(axis=1)
    ppt_annual[yr] = df_ppt[cols].sum(axis=1)

ai_annual = pet_annual.div(ppt_annual.replace(0, np.nan))

ai_mean = ai_annual.mean(axis=1)

df_pet['aridity_index'] = ai_mean.values
ai_df = df_pet[['lat', 'lon', 'aridity_index']].copy()

print("Aridity index computed for " + str(len(ai_df)) + " grid cells.")
print("AI min: " + str(round(ai_df['aridity_index'].min(), 3)))
print("AI max: " + str(round(ai_df['aridity_index'].max(), 3)))

# LOAD DROUGHT CSV AND MERGE
print("Loading spatial_flash_drought_properties.csv...")
df_drought = pd.read_csv(CSV_PATH)
df_drought['lat'] = df_drought['lat'].round(6)
df_drought['lon'] = df_drought['lon'].round(6)
print("Drought CSV rows: " + str(len(df_drought)))

df_merged = df_drought.merge(ai_df, on=['lat', 'lon'], how='left')

n_matched = df_merged['aridity_index'].notna().sum()
n_missing = df_merged['aridity_index'].isna().sum()
print("Matched: " + str(n_matched) + "  Unmatched: " + str(n_missing))

# SAVE OUTPUT
df_merged.to_csv(OUTPUT_PATH, index=False)
print("Done! Output saved to: " + OUTPUT_PATH)
print(df_merged.head())