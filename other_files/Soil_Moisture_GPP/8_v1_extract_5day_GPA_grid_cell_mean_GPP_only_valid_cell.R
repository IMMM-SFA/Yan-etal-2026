# ==============================================================================
# Script 19: Compute Gridcell Mean GPP (5-Day Pentad Averages)
# Ecosystem: Earth System Modeling (CLM5) Flash Drought Analysis
# Features: Lat/Lon Spatial Hashing, 3D NetCDF Output [cell, pentad, year]
# ==============================================================================
library(ncdf4)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CLUSTERS <- 1:7
INPUT_DIR <- "/global/cfs/cdirs/m2702/hongxiang/drought_flash_conus/hist"
VALID_CELLS_FILE <- "valid_conus_flash_drought_cells.csv"

# Temporal Setup
DAYS_PER_YEAR <- 365
PENTADS_PER_YEAR <- 73
TOTAL_YEARS <- 36
YEARS_TO_KEEP <- 35

# Drop the 1st year (Spin-up / Initialization)
DAY_START_IDX <- 366
DAY_END_IDX <- TOTAL_YEARS * DAYS_PER_YEAR

# ==============================================================================
# 2. MAIN PROCESSING LOOP
# ==============================================================================
cat("Loading master valid cell mask...\n")
valid_cells_df <- read.csv(VALID_CELLS_FILE)

# Spatial Hash Keys (Rounded to 4 decimals for float-precision matching)
valid_keys <- paste(sprintf("%.4f", valid_cells_df$lat), 
                    sprintf("%.4f", valid_cells_df$lon), sep="_")

for (cid in CLUSTERS) {
  fname <- sprintf("%s/gpp_cluster_%d_cell_avg_with_coords.nc", INPUT_DIR, cid)
  cat(sprintf("\n========================================\nProcessing Cluster %d\nFile: %s\n", cid, fname))
  
  # --- Open NetCDF ---
  nc <- nc_open(fname)
  lat <- ncvar_get(nc, "lat")
  lon <- ncvar_get(nc, "lon")
  gpp_daily <- ncvar_get(nc, "Gridcell_Avg_GPP") # Shape: [cell, time]
  nc_close(nc)
  
  n_cells <- length(lat)
  
  # --- Standardize Longitude & Apply Mask ---
  lon_std <- ifelse(lon > 180, lon - 360, lon)
  nc_keys <- paste(sprintf("%.4f", lat), sprintf("%.4f", lon_std), sep="_")
  is_valid_cell <- (nc_keys %in% valid_keys)
  
  cat(sprintf(" -> Retained %d / %d cells based on Lat/Lon matching.\n", sum(is_valid_cell), n_cells))
  
  gpp_daily[!is_valid_cell, ] <- NA
  gpp_daily <- gpp_daily[, DAY_START_IDX:DAY_END_IDX] 
  
  # --- Aggregate Daily to Pentad ---
  cat(" -> Aggregating daily GPP into pentads...\n")
  gpp_pentad <- array(NA, dim=c(n_cells, YEARS_TO_KEEP, PENTADS_PER_YEAR))
  
  doy_index <- rep(1:DAYS_PER_YEAR, YEARS_TO_KEEP)
  pentad_index <- ceiling(doy_index / 5)
  pentad_index[pentad_index > 73] <- 73 
  
  for (y in 1:YEARS_TO_KEEP) {
    start_col <- (y - 1) * DAYS_PER_YEAR + 1
    end_col <- y * DAYS_PER_YEAR
    gpp_year_daily <- gpp_daily[, start_col:end_col]
    
    for (p in 1:PENTADS_PER_YEAR) {
      cols_in_pentad <- which(pentad_index[start_col:end_col] == p)
      if (length(cols_in_pentad) > 1) {
        gpp_pentad[, y, p] <- rowMeans(gpp_year_daily[, cols_in_pentad], na.rm=TRUE)
      } else {
        gpp_pentad[, y, p] <- gpp_year_daily[, cols_in_pentad]
      }
    }
  }
  
  # --- Reshape for Export ---
  # Transpose [n_cells, year, pentad] to [n_cells, pentad, year]
  out_gpp_3d <- array(NA, dim=c(n_cells, PENTADS_PER_YEAR, YEARS_TO_KEEP)) 
  for (i in 1:n_cells) {
    if (!is_valid_cell[i]) next 
    out_gpp_3d[i, , ] <- t(gpp_pentad[i, , ])
  }
  
  # --- Export to 3D NetCDF ---
  out_fname <- sprintf("gpp_5day_cluster_%d_gridcell_mean.nc", cid)
  cat(sprintf(" -> Saving 3D Data to: %s\n", out_fname))
  
  # Define the 3 Dimensions
  dim_cell   <- ncdim_def("cell", "index", 1:n_cells)
  dim_pentad <- ncdim_def("pentad", "index", 1:PENTADS_PER_YEAR)
  dim_year   <- ncdim_def("year", "index", 1:YEARS_TO_KEEP)
  
  fill_val <- -9999.0
  var_lat <- ncvar_def("lat", "degrees_north", list(dim_cell), fill_val)
  var_lon <- ncvar_def("lon", "degrees_east", list(dim_cell), fill_val)
  
  # Define GPP Output Variable
  var_gpp <- ncvar_def("Gridcell_Avg_GPP_5day", "gC/m2/s", 
                        list(dim_cell, dim_pentad, dim_year), fill_val, compression=5)
  
  # Write
  nc_out <- nc_create(out_fname, list(var_lat, var_lon, var_gpp))
  
  ncvar_put(nc_out, var_lat, lat)
  ncvar_put(nc_out, var_lon, lon_std) 
  ncvar_put(nc_out, var_gpp, out_gpp_3d)
  
  nc_close(nc_out)
}

cat("\n*** Historical 3D GPP Pentad Extraction Successfully Completed! ***\n")