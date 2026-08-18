# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import pandas as pd
import os

def standardize_lon(lon_array):
    return (lon_array + 180) % 360 - 180

def calculate_t_tet():
    print("Initializing T and T/ET Calculation...")

    base_dir = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist'
    et_dir   = os.path.join(base_dir, 'daily_ET_data')
    wrf_dir  = os.path.join(base_dir, 'wrf_daily_precip_vpd')

    csv_in  = os.path.join(base_dir, 'flash_drought_spatial_pattern_analysis',
                           'spatial_flash_drought_properties.csv')
    csv_out = os.path.join(base_dir, 'flash_drought_spatial_pattern_analysis',
                           'spatial_flash_drought_properties_with_T_TET.csv')

    master_df_list = []

    for c in range(1, 8):
        print(f"\n--- Processing Cluster {c} ---")

        vegt_file = os.path.join(et_dir, f'daily_QVEGT_cluster_{c}_1981_2015.nc')
        soil_file = os.path.join(et_dir, f'daily_QSOIL_cluster_{c}_1981_2015.nc')
        wrf_file  = os.path.join(wrf_dir, f'wrf_daily_vpd_precip_1981_2015_cluster_{c}.nc')

        ds_vegt = xr.open_dataset(vegt_file)
        ds_soil = xr.open_dataset(soil_file)
        ds_wrf  = xr.open_dataset(wrf_file)

        # ---------------------------------------------------------
        # STEP A: SPATIAL ALIGNMENT (same approach as cal_WUE.py)
        # Use WRF as master grid (it has precip for the dry-day mask)
        # ---------------------------------------------------------
        wrf_lat  = np.round(ds_wrf['lat'].values, 4)
        wrf_lon  = np.round(standardize_lon(ds_wrf['lon'].values), 4)
        vegt_lat = np.round(ds_vegt['lat'].values, 4)
        vegt_lon = np.round(standardize_lon(ds_vegt['lon'].values), 4)
        soil_lat = np.round(ds_soil['lat'].values, 4)
        soil_lon = np.round(standardize_lon(ds_soil['lon'].values), 4)

        df_wrf  = pd.DataFrame({'lat': wrf_lat,  'lon': wrf_lon,
                                 'wrf_idx': np.arange(len(wrf_lat)),
                                 'cluster': ds_wrf['cluster'].values})
        df_vegt = pd.DataFrame({'lat': vegt_lat, 'lon': vegt_lon,
                                 'vegt_idx': np.arange(len(vegt_lat))})
        df_soil = pd.DataFrame({'lat': soil_lat, 'lon': soil_lon,
                                 'soil_idx': np.arange(len(soil_lat))})

        df_master = df_wrf.merge(df_vegt, on=['lat', 'lon'], how='inner')
        df_master = df_master.merge(df_soil, on=['lat', 'lon'], how='inner')

        print(f"  Raw cells -> WRF: {len(df_wrf)} | VEGT: {len(df_vegt)} | SOIL: {len(df_soil)}")
        print(f"  Matched cells: {len(df_master)}")

        if len(df_master) == 0:
            raise ValueError(f"Cluster {c}: No spatial overlap found.")

        ds_wrf_a  = ds_wrf.isel(cell=df_master['wrf_idx'].values)
        ds_vegt_a = ds_vegt.isel(cell=df_master['vegt_idx'].values)
        ds_soil_a = ds_soil.isel(cell=df_master['soil_idx'].values)

        # Unify cell coordinates to avoid xarray alignment issues
        unified_cell = ds_wrf_a['cell'].values
        unified_lat  = ds_wrf_a['lat'].values
        unified_lon  = ds_wrf_a['lon'].values

        ds_vegt_a = ds_vegt_a.assign_coords(
            cell=unified_cell, lat=('cell', unified_lat), lon=('cell', unified_lon))
        ds_soil_a = ds_soil_a.assign_coords(
            cell=unified_cell, lat=('cell', unified_lat), lon=('cell', unified_lon))

        # ---------------------------------------------------------
        # STEP B: TEMPORAL ALIGNMENT
        # QVEGT/QSOIL start from 1980 (with spin-up); skip first 365 days ? 1981-01-01
        # WRF already starts at 1981-01-01
        # ---------------------------------------------------------
        print("  -> Truncating spin-up and aligning time...")

        precip = ds_wrf_a['PRECIP']
        # QVEGT and QSOIL already start at 1981-01-01 (no spin-up year), unlike GPP
        fctr  = ds_vegt_a['FCTR']   # T  (transpiration)
        qsoil = ds_soil_a['QSOIL']  # soil evaporation

        n_times = precip.sizes['time']
        assert fctr.sizes['time'] == n_times, (
            f"Cluster {c}: VEGT has {fctr.sizes['time']} steps, "
            f"WRF has {n_times}. Time mismatch.")
        assert qsoil.sizes['time'] == n_times, (
            f"Cluster {c}: QSOIL has {qsoil.sizes['time']} steps, "
            f"WRF has {n_times}. Time mismatch.")

        time_index = xr.date_range(
            start='1981-01-01', periods=n_times, calendar='noleap', use_cftime=True)

        precip = precip.assign_coords(time=time_index)
        fctr   = fctr.assign_coords(time=time_index)
        qsoil  = qsoil.assign_coords(time=time_index)

        # Load into memory
        precip = precip.compute()
        fctr   = fctr.compute()
        qsoil  = qsoil.compute()

        # ---------------------------------------------------------
        # STEP C: COMPUTE T AND T/ET
        # Dry days: precip <= 1.0 mm  (same threshold as cal_WUE.py)
        # Growing season: April-October
        # ET = QVEGT (fctr) + QSOIL
        # ---------------------------------------------------------
        print("  -> Applying masks and computing T and T/ET...")

        et = fctr + qsoil  # total evapotranspiration

        # Mask: dry days only; ET must exceed T (physically required); T must be non-negative
        dry_mask = (precip <= 1.0) & (et > 1e-6) & (fctr <= et) & (fctr >= 0)

        t_masked  = fctr.where(dry_mask)
        et_masked = et.where(dry_mask)

        # Growing season filter
        gs_mask = t_masked['time'].dt.month.isin([4, 5, 6, 7, 8, 9, 10])
        t_gs  = t_masked.sel(time=gs_mask)
        et_gs = et_masked.sel(time=gs_mask)

        # Annual mean T across valid dry growing-season days
        annual_T = t_gs.groupby('time.year').mean(dim='time', skipna=True)

        # Ratio clipped to [0, 1] as a safety net for residual floating-point noise
        tet_ratio  = (t_gs / et_gs).clip(0, 1)
        annual_TET = tet_ratio.groupby('time.year').mean(dim='time', skipna=True)

        # 35-year climatological mean
        mean_T   = annual_T.mean(dim='year',   skipna=True)
        mean_TET = annual_TET.mean(dim='year', skipna=True)

        # ---------------------------------------------------------
        # STEP D: BUILD OUTPUT DATAFRAME
        # ---------------------------------------------------------
        df_cluster = pd.DataFrame({
            'lat':     df_master['lat'].values,
            'lon':     df_master['lon'].values,
            'cluster': df_master['cluster'].values,
            'T_mean':  mean_T.values,
            'TET_mean': mean_TET.values,
        })

        master_df_list.append(df_cluster)

        ds_vegt.close()
        ds_soil.close()
        ds_wrf.close()

    # ---------------------------------------------------------
    # STEP E: MERGE INTO EXISTING CSV
    # ---------------------------------------------------------
    print("\nMerging into spatial_flash_drought_properties.csv ...")
    all_clusters = pd.concat(master_df_list, ignore_index=True)

    df_main = pd.read_csv(csv_in)

    # Round lat/lon to 4 dp to avoid float key mismatches
    for col in ['lat', 'lon']:
        df_main[col]      = df_main[col].round(4)
        all_clusters[col] = all_clusters[col].round(4)

    df_out = df_main.merge(
        all_clusters[['lat', 'lon', 'T_mean', 'TET_mean']],
        on=['lat', 'lon'], how='left')

    n_missing = df_out['T_mean'].isna().sum()
    if n_missing:
        print(f'WARNING: {n_missing} rows had no T/TET match; check cluster coverage.')

    df_out.to_csv(csv_out, index=False, float_format='%.6f')
    print(f"\nCOMPLETE: saved {len(df_out)} rows to:\n  {csv_out}")
    print(df_out[['lat', 'lon', 'T_mean', 'TET_mean']].head())


if __name__ == "__main__":
    calculate_t_tet()