import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = '/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist'

# Input Templates
DROUGHT_CSV_TEMPLATE = 'drought_events_fixed_season_cluster_{}.csv'
GPP_NC_TEMPLATE      = 'gpp_percentiles_cluster_{}_gridcell_mean.nc'

OUTPUT_CSV_TEMPLATE  = 'drought_events_fixed_season_cluster_{}_with_GPP_Zhang2025.csv'

# Zhang et al. (2025) Parameters
G0_THRESHOLD = 40.0   # Pre-drought GPP must be > 40th percentile
START_YEAR = 1981     # EXACT ANCHOR: Maps index 0 of the array to 1981

def process_cluster(cluster_id):
    csv_path = os.path.join(BASE_DIR, DROUGHT_CSV_TEMPLATE.format(cluster_id))
    gpp_path = os.path.join(BASE_DIR, GPP_NC_TEMPLATE.format(cluster_id))
    
    if not os.path.exists(csv_path) or not os.path.exists(gpp_path):
        print(f"[{cluster_id}] Missing input files, skipping.")
        return

    print(f"\n=================================================")
    print(f" Processing Cluster {cluster_id}")
    print(f"=================================================")
    
    # 1. Load Data
    df_events = pd.read_csv(csv_path)
    ds_gpp = xr.open_dataset(gpp_path)
    
    gpp_var = 'Gridcell_Avg_GPP_percentile'
    
    # -------------------------------------------------------------
    # DEFENSE 1: DIMENSION LOCKING
    # -------------------------------------------------------------
    try:
        ds_gpp = ds_gpp.transpose('year', 'pentad', 'cell')
    except ValueError as e:
        print(f"  -> WARNING: Dimension names mismatch in xarray. Check your NetCDF dims. Error: {e}")
        return
        
    gpp_data = ds_gpp[gpp_var].values  
    
    # -------------------------------------------------------------
    # DEFENSE 2: FIX FRACTIONAL SCALE (0-1 -> 0-100)
    # -------------------------------------------------------------
    gpp_max_val = np.nanmax(gpp_data)
    if gpp_max_val <= 1.5:  
        print(f"  -> [AUTO-FIX] GPP data max is {gpp_max_val:.3f}. Converting 0-1 fraction to 0-100 percentage scale.")
        gpp_data = gpp_data * 100.0  
    else:
        print(f"  -> GPP data max is {gpp_max_val:.1f}. 0-100 percentage scale verified.")
        
    # -------------------------------------------------------------
    # DEFENSE 3: TEMPORAL ALIGNMENT
    # -------------------------------------------------------------
    if 'year' in ds_gpp.coords:
        raw_years = ds_gpp['year'].values
        if np.nanmin(raw_years) < 1000:
            print(f"  -> 'year' coord detected as indices (min={np.nanmin(raw_years)}). Remapping to calendar years.")
            years_array = np.arange(START_YEAR, START_YEAR + len(raw_years))
        else:
            print(f"  -> 'year' coord detected as calendar years.")
            years_array = raw_years
    else:
        print("  -> No 'year' coordinate found. Generating from START_YEAR.")
        years_array = np.arange(START_YEAR, START_YEAR + gpp_data.shape[0])
    
    year_to_idx = {yr: idx for idx, yr in enumerate(years_array)}
    
    # 3. Build KDTree to map SM lat/lon to GPP lat/lon
    print("  -> Building spatial KDTree for coordinate matching...")
    gpp_lats = ds_gpp['lat'].values
    gpp_lons = ds_gpp['lon'].values
    
    gpp_lons_360 = np.where(gpp_lons < 0, gpp_lons + 360, gpp_lons)
    gpp_coords = np.column_stack((gpp_lats, gpp_lons_360))
    kdtree = cKDTree(gpp_coords)
    
    sm_coords_unique = df_events[['lat', 'lon']].drop_duplicates().values
    sm_lons_360 = np.where(sm_coords_unique[:, 1] < 0, sm_coords_unique[:, 1] + 360, sm_coords_unique[:, 1])
    sm_coords_query = np.column_stack((sm_coords_unique[:, 0], sm_lons_360))
    
    distances, gpp_indices = kdtree.query(sm_coords_query)
    
    coord_to_gpp_idx = {}
    for i, row in enumerate(sm_coords_unique):
        if distances[i] < 0.05:
            coord_to_gpp_idx[(row[0], row[1])] = gpp_indices[i]
            
    print(f"  -> Matched {len(coord_to_gpp_idx)} unique grid cells.")

    # 4. Process Events (Zhang et al. 2025 Methodology)
    results = []
    dropped_for_G0 = 0
    missing_year_count = 0
    missing_spatial_count = 0
    
    for _, event in df_events.iterrows():
        sm_lat, sm_lon = event['lat'], event['lon']
        year = int(event['year'])
        s0_idx = int(event['S0_index'])
        s2_idx = int(event['S2_index'])
        
        # Look up matched spatial index
        gpp_cell_idx = coord_to_gpp_idx.get((sm_lat, sm_lon))
        if gpp_cell_idx is None:
            missing_spatial_count += 1
            continue
            
        # Look up matched temporal index
        yr_idx = year_to_idx.get(year)
        if yr_idx is None:
            missing_year_count += 1
            continue 
            
        # Extract the 73-pentad array 
        yearly_gpp_pctl = gpp_data[yr_idx, :, gpp_cell_idx]
            
        # 1. Define G0: 1 Pentad BEFORE drought onset
        g0_idx = s0_idx - 1
        if g0_idx < 0:
            continue
            
        g0_pctl = yearly_gpp_pctl[g0_idx]
        
        # 2. Enforce Zhang's Pre-Drought Health Rule
        if np.isnan(g0_pctl) or g0_pctl <= G0_THRESHOLD:
            dropped_for_G0 += 1
            continue
            
        # 3. Find G2: Minimum GPP percentile DURING the drought
        search_window = yearly_gpp_pctl[s0_idx : s2_idx + 1]
        
        if len(search_window) == 0 or np.all(np.isnan(search_window)):
            continue
            
        g2_pctl = np.nanmin(search_window)
        g2_idx = s0_idx + np.nanargmin(search_window)
        
        # 4. Calculate Total Ecosystem Resistance (GPP Slope)
        response_duration = g2_idx - g0_idx
        
        if response_duration > 0:
            gpp_slope = (g2_pctl - g0_pctl) / response_duration
        else:
            gpp_slope = np.nan
            
        # =========================================================
        # ADDED DIAGNOSTIC COLUMNS FOR AUDITING
        # =========================================================
        matched_gpp_lat = float(gpp_lats[gpp_cell_idx])
        matched_gpp_lon = float(gpp_lons[gpp_cell_idx])
        
        row_dict = event.to_dict()
        row_dict.update({
            'GPP_cell_idx': gpp_cell_idx,
            'GPP_lat': round(matched_gpp_lat, 4),
            'GPP_lon': round(matched_gpp_lon, 4),
            'G0_index': g0_idx,
            'G0_pctl': round(float(g0_pctl), 4),
            'G2_index': g2_idx,
            'G2_pctl': round(float(g2_pctl), 4),
            'GPP_response_duration': response_duration,
            'GPP_slope': round(float(gpp_slope), 4)
        })
        results.append(row_dict)
        
    # 5. Save Final Output
    df_final = pd.DataFrame(results)
    
    print(f"  -> Total Events before G0 Filter: {len(df_events)}")
    if missing_year_count > 0:
        print(f"  -> WARNING: {missing_year_count} events skipped due to missing year mapping.")
    if missing_spatial_count > 0:
        print(f"  -> WARNING: {missing_spatial_count} events skipped due to unmatched spatial KDTree.")
        
    print(f"  -> Dropped (Pre-drought G0 <= {G0_THRESHOLD}%): {dropped_for_G0}")
    
    if not df_final.empty:
        print(f"  -> Validated Events Saved: {len(df_final)}")
        
        # Ensure column order is clean, placing the diagnostic columns next to original lat/lon
        base_cols = list(df_events.columns)
        new_cols = ['GPP_cell_idx', 'GPP_lat', 'GPP_lon', 'G0_index', 'G0_pctl', 'G2_index', 'G2_pctl', 'GPP_response_duration', 'GPP_slope']
        
        # Insert the GPP tracking columns right after the SM lon column for easy side-by-side reading
        lon_idx = base_cols.index('lon') + 1
        final_col_order = base_cols[:lon_idx] + ['GPP_cell_idx', 'GPP_lat', 'GPP_lon'] + base_cols[lon_idx:] + new_cols[3:]
        
        df_final = df_final[[c for c in final_col_order if c in df_final.columns]]
        
        out_file = os.path.join(BASE_DIR, OUTPUT_CSV_TEMPLATE.format(cluster_id))
        df_final.to_csv(out_file, index=False)
        print(f"  -> [SAVED] {out_file}")
    else:
        print(f"  -> Validated Events Saved: 0")

if __name__ == "__main__":
    for i in range(1, 8):
        process_cluster(i)