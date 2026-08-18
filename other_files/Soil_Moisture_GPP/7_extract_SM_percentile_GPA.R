library(ncdf4)
library(lmom)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CLUSTERS <- 1:7
START_YEAR <- 1981
END_YEAR   <- 2015

# ==============================================================================
# HELPER: ZERO-INFLATED GPA FIT
# ==============================================================================
fit_gpa_zeros <- function(x) {
  # x is a vector of 35 years of data for one pentad/cell
  n <- length(x)
  
  # 1. Identify Zeros (using small threshold for float precision)
  is_zero <- x <= 1e-6
  n_zeros <- sum(is_zero)
  
  # 2. Calculate q (Probability of Zero)
  # Logic: count / (n + 1)
  q <- n_zeros / (n + 1)
  
  # Prepare Outputs
  pctls <- rep(NA, n)
  params <- c(NA, NA, NA) # xi, alpha, k
  
  # 3. Case A: All Zeros
  if (n_zeros == n) {
    pctls[] <- q * 100
    return(list(p=pctls, par=c(0,0,0), q=q))
  }
  
  # 4. Case B: Mixed Data - Fit GPA to Non-Zeros
  non_zeros <- x[!is_zero]
  
  # Need at least 3 points to fit GPA safely
  if (length(non_zeros) < 3) {
    pctls[!is_zero] <- NA # Not enough data to fit
    pctls[is_zero]  <- q * 100
    return(list(p=pctls, par=c(NA,NA,NA), q=q))
  }
  
  # L-Moments
  lmom_vals <- samlmu(non_zeros)
  
  # Check validity (L2 must be positive)
  if (is.na(lmom_vals[2]) || lmom_vals[2] <= 0) {
    pctls[is_zero] <- q * 100
    return(list(p=pctls, par=c(NA,NA,NA), q=q))
  }
  
  # Fit Parameters (tryCatch for robustness)
  par <- tryCatch(pelgpa(lmom_vals), error=function(e) NULL)
  
  if (is.null(par)) {
    # Fit failed
    pctls[is_zero] <- q * 100
    return(list(p=pctls, par=c(NA,NA,NA), q=q))
  }
  
  # 5. Calculate CDF
  # Formula: CDF_final = CDF_gpa * (1 - q) + q
  cdf_gpa <- cdfgpa(non_zeros, par)
  cdf_final <- cdf_gpa * (1 - q) + q
  
  # Store
  pctls[!is_zero] <- cdf_final * 100
  pctls[is_zero]  <- q * 100
  
  params <- par # xi, alpha, k
  
  return(list(p=pctls, par=params, q=q))
}

# ==============================================================================
# MAIN LOOP
# ==============================================================================

for (cid in CLUSTERS) {
  fname <- sprintf("sm_cluster_%d.nc", cid)
  
  if (!file.exists(fname)) {
    next
  }
  
  cat(sprintf("\nProcessing Cluster %d...\n", cid))
  
  # --- 1. Load Data ---
  nc <- nc_open(fname)
  sm_data <- ncvar_get(nc, "SOILLIQ_sum_0_9") 
  nc_close(nc)
  
  # Ensure (Time, Cell)
  dims <- dim(sm_data)
  if (dims[1] != 13140 && dims[2] == 13140) sm_data <- t(sm_data)
  
  # Slice 1981-2015
  # Assuming NoLeap (365 days) starts 1980-01-01
  # 1981 starts at index 366
  start_idx <- 366 
  n_days_req <- 35 * 365
  sm_hist <- sm_data[start_idx:(start_idx + n_days_req - 1), ]
  
  n_cells <- ncol(sm_hist)
  
  # --- 2. Resample to Pentads (Fast) ---
  cat("  Resampling to Pentads...\n")
  # Grouping ID: 1 to (35*73)
  abs_pentad_id <- rep(1:(35*73), each=5)
  # Rowsum is much faster than aggregate
  sm_pentad <- rowsum(sm_hist, abs_pentad_id, reorder=FALSE, na.rm=TRUE) / 5
  
  # --- 3. Compute Percentiles & Params ---
  cat("  Fitting Zero-Inflated GPA...\n")
  
  # Output Arrays
  # Percentiles: (35 Years, 73 Pentads, N Cells) -> We'll flatten to (Pentad, Cell, Year) for R matrix
  # Actually, easier to store as (Total_Pentads, Cell) first
  
  out_pctls <- matrix(NA, nrow=nrow(sm_pentad), ncol=n_cells)
  
  # Parameter Arrays (73 Pentads, N Cells)
  # We only need 1 set of params per pentad (seasonality)
  out_par_xi    <- matrix(NA, nrow=73, ncol=n_cells)
  out_par_alpha <- matrix(NA, nrow=73, ncol=n_cells)
  out_par_k     <- matrix(NA, nrow=73, ncol=n_cells)
  out_par_q     <- matrix(NA, nrow=73, ncol=n_cells)
  
  # Loop over 73 Pentads (Seasonality)
  for (p in 1:73) {
    # Progress
    if (p %% 10 == 0) cat(sprintf("    Pentad %d/73\n", p))
    
    # Indices for this pentad across all years
    # e.g., Pentad 1, Pentad 74, Pentad 147...
    p_indices <- seq(from=p, by=73, length.out=35)
    
    # Extract Data (35 Years x N Cells)
    data_slice <- sm_pentad[p_indices, ]
    
    # Apply Function to each Cell
    # Returns a list of lists. We need to unpack.
    results <- apply(data_slice, 2, fit_gpa_zeros)
    
    # Unpack Results
    # 1. Percentiles (Back into the time series)
    # lapply extracts 'p' from the list
    pctls_vec <- unlist(lapply(results, function(x) x$p))
    # Reshape back to (35, N_Cells) and assign
    out_pctls[p_indices, ] <- matrix(pctls_vec, nrow=35, byrow=FALSE)
    
    # 2. Parameters (Store for this Pentad)
    pars <- do.call(rbind, lapply(results, function(x) x$par))
    qs   <- unlist(lapply(results, function(x) x$q))
    
    out_par_xi[p, ]    <- pars[, 1]
    out_par_alpha[p, ] <- pars[, 2]
    out_par_k[p, ]     <- pars[, 3]
    out_par_q[p, ]     <- qs
  }
  
  # --- 4. Save to NetCDF ---
  cat("  Saving NetCDF...\n")
  
  # Dimensions
  dim_year   <- ncdim_def("year", "years", 1981:2015)
  dim_pentad <- ncdim_def("pentad", "index", 1:73)
  dim_cell   <- ncdim_def("cell", "index", 1:n_cells)
  
  # Reshape Percentiles for NC: (Year, Pentad, Cell)
  # Current `out_pctls` is (2555, n_cells) ordered by (Year1-P1..P73, Year2...)
  # We need to map this carefully.
  # Array logic: Data fills first dim first.
  # We want array[year, pentad, cell].
  # R fills column-major.
  
  # Let's reshape `out_pctls` -> array(73, 35, n_cells) first (Pentad, Year, Cell)
  # Because out_pctls rows are P1, P2...P73 (Year 1), P1... (Year 2)
  # Actually, `rowsum` ordered it as P1-Y1, P2-Y1 ... P73-Y1, P1-Y2...
  # So vector is: Y1P1, Y1P2... Y1P73, Y2P1...
  # If we make array(73, 35, n_cells), it fills 73 pentads, then next year. Correct.
  
  pctl_array <- array(as.vector(out_pctls), dim=c(73, 35, n_cells))
  # Transpose to (Year, Pentad, Cell) if desired, or keep (Pentad, Year, Cell)
  # Standard usually (Year, Pentad, Cell). 
  pctl_final <- aperm(pctl_array, c(2, 1, 3)) # Swap 1 and 2
  
  # Variables
  var_pctl  <- ncvar_def("sm_percentiles", "percent", list(dim_year, dim_pentad, dim_cell), -999, compression=5)
  var_xi    <- ncvar_def("gpa_xi", "param", list(dim_pentad, dim_cell), -999, compression=5)
  var_alpha <- ncvar_def("gpa_alpha", "param", list(dim_pentad, dim_cell), -999, compression=5)
  var_k     <- ncvar_def("gpa_k", "param", list(dim_pentad, dim_cell), -999, compression=5)
  var_q     <- ncvar_def("gpa_q", "prob", list(dim_pentad, dim_cell), -999, compression=5)
  
  out_fname <- sprintf("sm_gpa_percentiles_cluster_%d.nc", cid)
  nc_out <- nc_create(out_fname, list(var_pctl, var_xi, var_alpha, var_k, var_q))
  
  ncvar_put(nc_out, var_pctl, pctl_final)
  ncvar_put(nc_out, var_xi, out_par_xi)
  ncvar_put(nc_out, var_alpha, out_par_alpha)
  ncvar_put(nc_out, var_k, out_par_k)
  ncvar_put(nc_out, var_q, out_par_q)
  
  ncatt_put(nc_out, "sm_percentiles", "description", "SSI using Zero-Inflated GPA (lmom)")
  ncatt_put(nc_out, "gpa_q", "description", "Probability of Zero (q)")
  
  nc_close(nc_out)
  cat(sprintf("  Saved %s\n", out_fname))
}