# 1. Strict Onset: The event actually started from a wet state (>40%) without hitting a previous drought (<20% for 3 pentads) in between.
# 2. Growing Season Continuity: The search for the onset (S0) stays entirely within the active growing season (where GPP exists)
# 3. G0 Validity: The onset (S0) is at least the 2nd pentad of the growing season so that G0 refers to a valid time step with GPP data.
# 4. The drought start (S1) must be inside the growing season. If the soil moisture drops below 20% outside the growing season, we ignore it.
# 5. Truncate S2: If the drought starts in the season but extends beyond it, we forcefully set S2 to the index where the growing season ends.
#                 After truncating, we check if the remaining valid portion is still >=3 pentads. 
#                 If the season ends 1 pentad after the drought starts, it's too short to analyze and should be dropped.

import xarray as xr
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist'

# SM Data (Percentiles from GPA fit)
SM_GPA_FILE_TEMPLATE     = os.path.join(BASE_DIR, 'sm_gpa_percentiles_cluster_{}.nc')
# SM Coordinates (Lat/Lon source for SM)
SM_COORD_FILE_TEMPLATE   = os.path.join(BASE_DIR, 'sm_cluster_{}.nc') 

# GPP Data (New Percentiles, >= 1% Area Threshold Enforced)
GPP_PCTL_FILE_TEMPLATE   = os.path.join(BASE_DIR, 'gpp_percentile_cluster_{}.nc')
# GPP Coordinates & Raw Data (Lat/Lon source and raw GPP for phenology)
GPP_COORD_FILE_TEMPLATE  = os.path.join(BASE_DIR, 'gpp_cluster_{}.nc')

OUTPUT_TEMPLATE          = os.path.join(BASE_DIR, 'drought_events_hist_cellbycell_{}_cluster_{}.csv')

# Drought Parameters
PCTL_START_THRESH = 40.0   
PCTL_DROUGHT_THRESH = 20.0 
MIN_DURATION_PENTADS = 3   
FLASH_RATE_THRESH = 5.0    

def classify_drought_events(sm_series, crop_mask, year, gpp_idx, sm_idx, lat, lon, crop_name):
    """
    Identifies drought events, ignoring short transient dips (<3 pentads) during S0 search.
    Stores both GPP and SM indices (0-based).
    """
    is_drought = sm_series < PCTL_DROUGHT_THRESH
    
    padded = np.concatenate(([False], is_drought, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    
    major_drought_mask = np.zeros(len(sm_series), dtype=bool)
    for s, e in zip(starts, ends):
        if (e - s) >= MIN_DURATION_PENTADS:
            major_drought_mask[s:e] = True

    events = []
    
    for s1, s2 in zip(starts, ends):
        
        # RULE: S1 MUST be in the growing season
        if not crop_mask[s1]:
            continue

        # RULE: Truncate S2 to end of growing season
        season_end_idx = s2 
        for t in range(s1, s2):
            if not crop_mask[t]:
                season_end_idx = t
                break
        current_s2 = season_end_idx
        
        # Check Duration AFTER Truncation
        drought_phase_duration = current_s2 - s1
        if drought_phase_duration < MIN_DURATION_PENTADS:
            continue

        # FIND S0 (Onset)
        s0 = -1
        valid_onset = True
        
        for t in range(s1 - 1, -1, -1):
            if not crop_mask[t]: 
                valid_onset = False
                break
            
            val = sm_series[t]
            if val > PCTL_START_THRESH:
                if t < 1 or not crop_mask[t-1]:
                    valid_onset = False
                    break
                s0 = t
                break
            
            if major_drought_mask[t]:
                valid_onset = False
                break
        
        if s0 == -1 or not valid_onset:
            continue

        # METRICS & SAVE
        total_event_duration = current_s2 - s0
        onset_time = s1 - s0
        sm_s0 = sm_series[s0]
        sm_s1 = sm_series[s1]
        
        delta_t = max(onset_time, 1)
        rate = (sm_s0 - sm_s1) / delta_t
        
        d_type = 'Flash' if rate >= FLASH_RATE_THRESH else 'Slow'
        min_pctl = np.min(sm_series[s1:current_s2])
        
        events.append({
            'cell_id_gpp': int(gpp_idx),
            'cell_id_sm': int(sm_idx),
            'lat': round(float(lat), 4),
            'lon': round(float(lon), 4),
            'year': int(year),
            'crop': crop_name,
            'drought_type': d_type,
            'decline_rate': round(rate, 4),
            'S0_index': s0,
            'S1_index': s1,
            'S2_index': current_s2,
            'drought_phase_pentads': drought_phase_duration,
            'total_event_pentads': total_event_duration,
            'intensification_pentads': onset_time,
            'min_percentile': round(min_pctl, 2),
            'SM_at_S0': round(sm_s0, 2),
            'SM_at_S1': round(sm_s1, 2)
        })
        
    return events

def main():
    print("=== Flash Drought Detection (Matched Lat/Lon + Percentile 1% Filter) ===")
    
    for cluster_id in range(1, 8): 
        print(f"\n" + "="*40)
        print(f"STARTING CLUSTER {cluster_id}")
        print("="*40)

        # 1. Format File Paths
        sm_gpa_file = SM_GPA_FILE_TEMPLATE.format(cluster_id)
        sm_coord_file = SM_COORD_FILE_TEMPLATE.format(cluster_id)
        gpp_pctl_file = GPP_PCTL_FILE_TEMPLATE.format(cluster_id)
        gpp_coord_file = GPP_COORD_FILE_TEMPLATE.format(cluster_id)

        try:
            for f in [sm_gpa_file, sm_coord_file, gpp_pctl_file, gpp_coord_file]:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Missing file: {f}")

            # --- Load Coordinates ---
            ds_gpp_coords = xr.open_dataset(gpp_coord_file)
            gpp_lats = ds_gpp_coords['lat'].values
            gpp_lons = ds_gpp_coords['lon'].values
            
            ds_sm_coords = xr.open_dataset(sm_coord_file)
            sm_lats = ds_sm_coords['lat'].values
            sm_lons = ds_sm_coords['lon'].values
            
            # --- Build SM Lookup Dictionary ---
            print("  Building Coordinate Map...")
            sm_lookup = {}
            for idx, (lat, lon) in enumerate(zip(sm_lats, sm_lons)):
                key = (round(float(lat), 4), round(float(lon), 4))
                sm_lookup[key] = idx
            
            # --- Load Data Variables ---
            ds_sm_data = xr.open_dataset(sm_gpa_file)
            ds_gpp_pctl = xr.open_dataset(gpp_pctl_file)
            
            print(f"  Loaded all data for Cluster {cluster_id}")

        except Exception as e:
            print(f"Error loading files for Cluster {cluster_id}: {e}")
            continue
        
        # 3. PREPARE ARRAYS
        sm_values_array = ds_sm_data['sm_percentiles'].transpose('year', 'pentad', 'cell').values
        sm_years = ds_sm_data['year'].values
        n_years = sm_values_array.shape[0]
        
        # Find all crop percentile variables in the unified NetCDF
        crop_vars = [v for v in ds_gpp_pctl.data_vars if '_percentile' in v]
        all_events_in_cluster = []
        
        # 4. PROCESSING LOOP
        for var_name in crop_vars:
            crop_name = var_name.replace('_percentile', '')
            print(f"  Processing Crop: {crop_name}...")
            
            # A. Load Percentiles (NaNs natively represent the < 1% filtered cells)
            gpp_percentiles = ds_gpp_pctl[var_name].transpose('year', 'pentad', 'cell').values
            valid_cells_mask = ~np.isnan(gpp_percentiles)
            
            # B. Recreate Phenology Growing Season Mask (Climatology > 0)
            raw_gpp_var = f"{crop_name}_GPP"
            raw_gpp = ds_gpp_coords[raw_gpp_var].transpose('time', 'cell').values
            n_cells = raw_gpp.shape[1]
            
            # 1981-2015 slice to match 35 years
            raw_gpp_35y = raw_gpp[365:13140, :] 
            gpp_pentad = raw_gpp_35y.reshape(35, 73, 5, n_cells).mean(axis=2)
            gpp_clim = np.nanmean(gpp_pentad, axis=0)
            
            # Pentad is active if historical climatology > 0
            active_pentads = gpp_clim > 0
            gs_mask_3d = np.broadcast_to(active_pentads[np.newaxis, :, :], (35, 73, n_cells))
            
            # Final Mask: Must be a valid 1% cell AND in the growing season
            final_crop_mask = valid_cells_mask & gs_mask_3d
            
            count_events_crop = 0
            
            for y_idx in range(n_years):
                year_val = sm_years[y_idx]
                
                # Active cells: has at least one valid growing season pentad this year
                active_gpp_indices = np.where(np.any(final_crop_mask[y_idx], axis=0))[0]
                
                if len(active_gpp_indices) == 0:
                    continue
                
                for gpp_idx in active_gpp_indices:
                    lat_val = gpp_lats[gpp_idx]
                    lon_val = gpp_lons[gpp_idx]
                    
                    coord_key = (round(float(lat_val), 4), round(float(lon_val), 4))
                    sm_idx = sm_lookup.get(coord_key)
                    if sm_idx is None:
                        continue
                        
                    sm_ts = sm_values_array[y_idx, :, sm_idx]
                    mask_ts = final_crop_mask[y_idx, :, gpp_idx]
                    
                    events = classify_drought_events(
                        sm_ts, mask_ts, year_val, 
                        gpp_idx, sm_idx,  
                        lat_val, lon_val, 
                        crop_name
                    )
                    
                    if events:
                        all_events_in_cluster.extend(events)
                        count_events_crop += len(events)
            
            print(f"    -> Finished {crop_name}. Total Events: {count_events_crop}")

        # 5. SAVE RESULTS
        if all_events_in_cluster:
            df_all = pd.DataFrame(all_events_in_cluster)
            
            cols = [
                'cell_id_gpp', 'cell_id_sm', 
                'lat', 'lon', 'year', 'crop', 
                'drought_type', 'decline_rate', 
                'S0_index', 'S1_index', 'S2_index', 
                'drought_phase_pentads', 'total_event_pentads', 
                'intensification_pentads', 'min_percentile',
                'SM_at_S0', 'SM_at_S1'
            ]
            final_cols = [c for c in cols if c in df_all.columns]
            df_all = df_all[final_cols]
            
            for crop in df_all['crop'].unique():
                df_crop = df_all[df_all['crop'] == crop]
                outfile = OUTPUT_TEMPLATE.format(crop, cluster_id)
                df_crop.to_csv(outfile, index=False)
                print(f"  [SAVED] {len(df_crop)} events for {crop} -> {outfile}")
        else:
            print(f"  No events found for Cluster {cluster_id}.")
            
        ds_sm_data.close()
        ds_gpp_pctl.close()
        ds_gpp_coords.close()
        ds_sm_coords.close()

if __name__ == "__main__":
    main()