library(ncdf4)
library(lmom)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CLUSTERS <- 1:7
START_YEAR <- 1981
END_YEAR   <- 2015

# Distribution Names Map (matches your function order)
DIST_NAMES <- c("EXP", "GAM", "GEV", "GLO", "GPA", 
                "GNO", "GUM", "LN3", "NOR", "PE3", "WEI")

# ==============================================================================
# 1. YOUR FITTING FUNCTION
# ==============================================================================
fit_distributions <- function(data) {
  # Clean data
  data <- data[!is.na(data)]
  if (length(data) < 10) return(NA) 
  
  l_moments <- samlmu(data)
  if (any(is.na(l_moments))) return(NA)
  
  d_matrix <- matrix(data = 9999, nrow = 11, ncol = 1)
  
  # 1-EXP
  d_matrix[1,1] <- tryCatch({
    paras <- pelexp(l_moments)
    theo <- lmrexp(paras, nmom=4)
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)
  
  # 2-GAM
  d_matrix[2,1] <- tryCatch({
    paras <- pelgam(l_moments)
    theo <- lmrgam(paras, nmom=4)
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)
  
  # 3-GEV
  d_matrix[3,1] <- tryCatch({
    paras <- pelgev(l_moments)
    theo <- lmrgev(paras, nmom=4)    
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)  
  
  # 4-GLO
  d_matrix[4,1] <- tryCatch({
    paras <- pelglo(l_moments)
    theo <- lmrglo(paras, nmom=4)    
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)    
  
  # 5-GPA
  d_matrix[5,1] <- tryCatch({
    paras <- pelgpa(l_moments)
    theo <- lmrgpa(paras, nmom=4)    
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)    
  
  # 6-GNO
  d_matrix[6,1] <- tryCatch({
    paras <- pelgno(l_moments)
    theo <- lmrgno(paras, nmom=4)    
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)      
  
  # 7-GUM
  d_matrix[7,1] <- tryCatch({
    paras <- pelgum(l_moments)
    theo <- lmrgum(paras, nmom=4)    
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)   
  
  # 8-LN3
  d_matrix[8,1] <- tryCatch({
    paras <- pelln3(l_moments)
    theo <- lmrgum(paras, nmom=4)    
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)   
  
  # 9-NOR
  d_matrix[9,1] <- tryCatch({
    paras <- pelnor(l_moments)
    theo <- lmrnor(paras, nmom=4)  
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)
  }, error = function(e) 9999)   

  # 10-PE3
  d_matrix[10,1] <- tryCatch({  
    paras <- pelpe3(l_moments)
    theo <- lmrpe3(paras, nmom=4)    
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)    
  }, error = function(e) 9999)    
    
  # 11-WEI
  d_matrix[11,1] <- tryCatch({  
    paras <- pelwei(l_moments)
    theo <- lmrpe3(paras, nmom=4)    
    sqrt((l_moments[3]-theo[3])^2 + (l_moments[4]-theo[4])^2)    
  }, error = function(e) 9999)     
  
  return(which.min(d_matrix))
}

# ==============================================================================
# 2. MAIN PROCESSING LOOP
# ==============================================================================

# Global counter to track overall stats across all clusters
global_counts <- rep(0, 11)

for (cid in CLUSTERS) {
  fname <- sprintf("sm_cluster_%d.nc", cid)
  
  if (!file.exists(fname)) {
    cat(sprintf("Skipping %s (Not found)\n", fname))
    next
  }
  
  cat(sprintf("\n--- Processing Cluster %d ---\n", cid))
  
  # A. Load NetCDF
  nc <- nc_open(fname)
  sm_data <- ncvar_get(nc, "SOILLIQ_sum_0_9") 
  nc_close(nc)
  
  # Ensure shape is (Time, Cell)
  dims <- dim(sm_data)
  if (dims[1] != 13140 && dims[2] == 13140) { 
    sm_data <- t(sm_data)
  }
  
  # B. Slice 1981-2015 (35 Years)
  # Assuming 365-day calendar (NoLeap) starting 1980-01-01
  start_idx <- 366 
  end_idx <- 366 + (35 * 365) - 1
  sm_hist <- sm_data[start_idx:end_idx, ]
  
  # C. Resample to Pentads
  cat("  Resampling to Pentads...\n")
  
  # Grouping ID: 1 to (35*73)
  abs_pentad_id <- rep(1:(35*73), each=5)
  
  # Calculate Means
  sm_pentad_sums <- rowsum(sm_hist, abs_pentad_id, reorder=FALSE, na.rm=TRUE)
  sm_pentad_means <- sm_pentad_sums / 5
  
  # D. Fit Distributions & Count
  cat("  Fitting distributions (sampling checks)...\n")
  
  cluster_counts <- rep(0, 11)
  
  # Loop Pentads
  for (p in 1:73) {
    if (p %% 10 == 0) cat(sprintf("    Pentad %d/73...\n", p))
    
    # Get indices for this pentad across all 35 years
    indices <- seq(from=p, by=73, length.out=35)
    data_slice <- sm_pentad_means[indices, ]
    
    # Fit every cell
    winners <- apply(data_slice, 2, fit_distributions)
    
    # Update Counts
    # table() creates a frequency table of the winning indices (1-11)
    freqs <- table(factor(winners, levels=1:11))
    cluster_counts <- cluster_counts + as.numeric(freqs)
  }
  
  # E. Print Cluster Summary
  total_fits <- sum(cluster_counts)
  cat(sprintf("\n  Summary for Cluster %d:\n", cid))
  
  # Sort and print
  results <- data.frame(Dist=DIST_NAMES, Count=cluster_counts)
  results$Pct <- (results$Count / total_fits) * 100
  results <- results[order(-results$Pct), ]
  
  print(results, row.names=FALSE)
  
  # Accumulate global
  global_counts <- global_counts + cluster_counts
}

# ==============================================================================
# 3. OVERALL SUMMARY
# ==============================================================================
cat("\n=========================================\n")
cat("      OVERALL SUMMARY (ALL CLUSTERS)     \n")
cat("=========================================\n")

total_global <- sum(global_counts)
final_res <- data.frame(Dist=DIST_NAMES, Count=global_counts)
final_res$Pct <- (final_res$Count / total_global) * 100
final_res <- final_res[order(-final_res$Pct), ]

print(final_res, row.names=FALSE)