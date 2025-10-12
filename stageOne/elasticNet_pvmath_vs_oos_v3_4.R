# ---- III. Predictive Modelling: Version 3.4 ----
#
# Nested Cross-Validation (glmnet):
#   inner  - cv.glmnet with MANUAL K-folds (grouped = TRUE, keep = TRUE), select lambda.min AND lambda.1se across α-grid
#   outer  - glmnet exact refit on OUTER-TRAIN at the inner winner’s (alpha, lambda.*); evaluate on OUTER-HOLDOUT
#
# This script runs both rules side-by-side and reports per-fold and aggregated metrics
# for:   Rule ∈ { "lambda.min", "lambda.1se" }.
#
# Remark: glmnet cannot handle NAs?

## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)        # Read SPSS .sav files
library(tidyverse)    # Includes dplyr, tidyr, purrr, ggplot2, tibble, etc.
library(broom)        # For tidying model output
library(tictoc)       # For timing code execution
library(caret)        # varImp
library(glmnet)       # Elastic net / Lasso / Ridge
library(doParallel)
# library(Matrix)     # (optional) if use sparse model matrices

# Load data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE)
dim(pisa_2022_student_canada)   # 23073 x 1278

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") 

# Constants
M <- 10                  # Number of plausible values
G <- 80                  # Number of BRR replicate weights
k <- 0.5                 # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)   # 95% CI z-critical value; Two-sided 95%: ≈ 1.96

# Target varaible
pvmaths  <- paste0("PV", 1:M, "MATH")   # PV1MATH to PV10MATH

# Predictors
oos <- c("EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME")  

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)    # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                  # Final student weight

# Prepare modeling data
temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                  # IDs
         all_of(final_wt), all_of(rep_wts),   # Weights
         all_of(pvmaths), all_of(oos)) %>%    # PVs + predictors
  filter(if_all(all_of(oos), ~ !is.na(.)))    # Listwise deletion for predictors

dim(temp_data)  # 20003 x 97

# Check summaries
summary(temp_data[[final_wt]])
summary(temp_data[, pvmaths])
sapply(temp_data[, pvmaths], sd)
summary(temp_data[, oos])
sapply(temp_data[, oos], sd)

### ---- Helper: Compute weighted performance metrics ----
compute_metrics <- function(y_true, y_pred, w) {
  resid <- y_true - y_pred 
  mse <- sum(w * (-resid)^2) / sum(w)
  rmse <- sqrt(mse)
  mae <- sum(w * abs(-resid)) / sum(w)
  bias <- sum(w * (-resid)) / sum(w)              
  bias_pct <- 100 * sum(w * (-resid / y_true)) / sum(w)   # Attend to division-by-zero problem (not an issue here though)
  y_bar <- sum(w * y_true) / sum(w)
  sse <- sum(w * resid^2)
  sst <- sum(w * (y_true - y_bar)^2)       
  r2 <- 1 - (sse / sst)
  return(c(mse = mse, rmse = rmse, mae = mae, bias = bias, bias_pct = bias_pct, r2 = r2))
}

# Small helper: weighted standard deviation
wsd <- function(x, w) {
  mu <- sum(w * x) / sum(w)
  sqrt( sum(w * (x - mu)^2) / sum(w) )
}

# Small helper: calculate weighted RMSE
w_rmse <- function(y_true, y_pred, w) sqrt(sum(w * (y_pred - y_true)^2) / sum(w))

### ---- Random Train/Validation/Test (70/15/15) split ----
set.seed(123)                      # Ensure reproducibility
n <- nrow(temp_data)               # 20003
indices <- sample(n)               # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.70 * n)         # 14002
n_valid <- floor(0.15 * n)         # 3000
n_test  <- n - n_train - n_valid   # 3001
# Ensure sum = n

# Assign indices
train_idx <- indices[1:n_train]
valid_idx <- indices[(n_train + 1):(n_train + n_valid)]
test_idx  <- indices[(n_train + n_valid + 1):n]

# Subset the data
train_data <- temp_data[train_idx, ]
valid_data <- temp_data[valid_idx, ]
test_data  <- temp_data[test_idx, ]

# Check statistics 
round(as.data.frame(sapply(
  list(train=train_data, valid=valid_data, test=test_data),
  \(data) {
    c(
      # Split sizes
      n              = nrow(data),                           # number of students in split
      
      # Weight distribution 
      w_mean         = mean(data[[final_wt]]),               # mean of weights
      w_sum          = sum(data[[final_wt]]),                # total population weight
      w_min          = min(data[[final_wt]]),                # smallest weight
      w_max          = max(data[[final_wt]]),                # largest weight
      w_sd           = sd(data[[final_wt]]),                 # standard deviation of weights
      
      # School representation
      n_schools      = length(unique(data$CNTSCHID)),        # number of distinct schools
      
      # Plausible values (weighted stats)
      pv1math_mean   = sum(data[[final_wt]] * data[[pvmaths[1]]]) / sum(data[[final_wt]]), # <=> weighted.mean(data[[pvmaths[1]]], data[[final_wt]])
      pv1math_sd     = wsd(data[[pvmaths[1]]], data[[final_wt]]),
      
      # Predictor example (weighted stats)
      exerprac_mean  = sum(data[[final_wt]] * data$EXERPRAC) / sum(data[[final_wt]]),      # <=> weighted.mean(data$EXERPRAC, data[[final_wt]])
      exerprac_sd    = wsd(data$EXERPRAC, data[[final_wt]])
    )
  }
)), 2)
# >                   train    valid     test
# > n              14002.00  3000.00  3001.00
# > w_mean            15.83    15.62    15.89
# > w_sum         221628.31 46871.57 47689.14
# > w_min              1.05     1.05     1.05
# > w_max            833.47   833.47    88.54
# > w_sd              15.42    20.24    13.90
# > n_schools        831.00   761.00   764.00
# > pv1math_mean     502.36   504.08   502.59
# > pv1math_sd        93.61    90.42    94.62
# > exerprac_mean      4.34     4.52     4.43
# > exerprac_sd        3.36     3.35     3.39

## ---- Main model using final student weights (W_FSTUWT) ----

### ---- Nested Cross-Validation for PV1MATH (grouped=TRUE, keep=TRUE; lambda.min & lambda.1se) ----

# Target plausible value
pv1math <- pvmaths[1]   # "PV1MATH"

# TRAIN/VALID/TEST matrices (VALID/TEST kept but not used for nested-CV choices)
X_train <- as.matrix(train_data[, oos])
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

X_valid <- as.matrix(valid_data[, oos])
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

X_test  <- as.matrix(test_data[,  oos])
y_test  <- test_data[[pv1math]]
w_test  <- test_data[[final_wt]]

# --- Helpers: folds and diagnostics ---

# Build K manual folds (indices) with a fixed seed over N rows
make_folds <- function(n_cv, num_folds = 5L, seed = 123L) {
  set.seed(seed)
  cv_order <- sample.int(n_cv)
  bounds <- floor(seq(0, n_cv, length.out = num_folds + 1))
  stopifnot(all(diff(bounds) > 0))
  folds <- vector("list", num_folds)
  for (k in seq_len(num_folds)) folds[[k]] <- cv_order[(bounds[k] + 1):bounds[k + 1]]
  stopifnot(identical(sort(unlist(folds)), seq_len(n_cv)))
  folds
}

# Weight diagnostics per fold
print_weight_balance <- function(w, folds, title = "Weight balance") {
  s <- tibble::tibble(
    fold   = seq_along(folds),
    n      = vapply(folds, length, integer(1)),
    w_sum  = vapply(folds, function(idx) sum(w[idx]), numeric(1)),
    w_mean = vapply(folds, function(idx) mean(w[idx]), numeric(1)),
    w_med  = vapply(folds, function(idx) median(w[idx]), numeric(1)),
    w_effn = vapply(folds, function(idx) { wi <- w[idx]; (sum(wi)^2) / sum(wi^2) }, numeric(1))
  ) |>
    dplyr::mutate(w_share = w_sum / sum(w_sum))
  message(title)
  print(s, n = Inf, width = Inf)
  cat(sprintf("Weight share range: [%.3f, %.3f]\n", min(s$w_share), max(s$w_share)))
  cat(sprintf("Max/Min share ratio: %.3f\n", max(s$w_share) / min(s$w_share)))
  cat(sprintf("Coeff. of variation of shares: %.3f\n\n", stats::sd(s$w_share) / mean(s$w_share)))
}

# α-grid
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso

# --- Outer folds over TRAIN ---
num_folds_outer <- 5L
n_cv_outer <- nrow(X_train)
outer_folds <- make_folds(n_cv_outer, num_folds_outer, seed = 123L)

print_weight_balance(w_train, outer_folds, title = "Outer CV: weight balance over TRAIN")
# > Outer CV: weight balance over TRAIN
# > # A tibble: 5 × 7
# > fold     n  w_sum w_mean w_med w_effn w_share
# > <int> <int>  <dbl>  <dbl> <dbl>  <dbl>   <dbl>
# >    1     1  2800 44065.   15.7  8.93  1569.   0.199
# >    2     2  2800 45527.   16.3  9.60  1067.   0.205
# >    3     3  2801 43273.   15.4  8.86  1571.   0.195
# >    4     4  2800 44610.   15.9  9.49  1615.   0.201
# >    5     5  2801 44154.   15.8  9.37  1581.   0.199
# > Weight share range: [0.195, 0.205]
# > Max/Min share ratio: 1.052
# > Coeff. of variation of shares: 0.019

# Storage (separate containers for each rule)
outer_oof_pred_min      <- rep(NA_real_, n_cv_outer)
outer_oof_pred_1se      <- rep(NA_real_, n_cv_outer)

outer_metrics_min       <- vector("list", num_folds_outer)
outer_metrics_1se       <- vector("list", num_folds_outer)

inner_winners_rows_min  <- vector("list", num_folds_outer)
inner_winners_rows_1se  <- vector("list", num_folds_outer)

# Parallel backend for cv.glmnet (inner loop)
doParallel::registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Nested Cross-Validation (glmnet; grouped=TRUE, keep=TRUE; lambda.min & lambda.1se)")
for (o in seq_len(num_folds_outer)) {
  message(sprintf("\n==== Outer fold %d/%d ====", o, num_folds_outer))
  outer_hold_idx  <- outer_folds[[o]]
  outer_train_idx <- setdiff(seq_len(n_cv_outer), outer_hold_idx)
  
  # OUTER-TRAIN / OUTER-HOLDOUT splits
  X_train_outer <- X_train[outer_train_idx, , drop = FALSE]
  y_train_outer <- y_train[outer_train_idx]
  w_train_outer <- w_train[outer_train_idx]
  
  X_hold_outer  <- X_train[outer_hold_idx, , drop = FALSE]
  y_hold_outer  <- y_train[outer_hold_idx]
  w_hold_outer  <- w_train[outer_hold_idx]
  
  # Inner folds (on OUTER-TRAIN)
  n_cv_inner      <- length(outer_train_idx)
  num_folds_inner <- 5L
  inner_folds     <- make_folds(n_cv_inner, num_folds_inner, seed = 123L)
  
  # foldid (1..K for each OUTER-TRAIN row)
  foldid_inner <- integer(n_cv_inner)
  for (k in seq_along(inner_folds)) foldid_inner[inner_folds[[k]]] <- k
  stopifnot(all(foldid_inner %in% seq_len(num_folds_inner)), length(foldid_inner) == n_cv_inner)
  
  print_weight_balance(w_train_outer, inner_folds,
                       title = sprintf("Inner CV (outer fold %d): weight balance over OUTER-TRAIN", o))
  
  # --- Inner tuner over α-grid using cv.glmnet (grouped=TRUE, keep=TRUE) ---
  per_alpha_list_min <- vector("list", length(alpha_grid))   # one row per α at lambda.min
  per_alpha_list_1se <- vector("list", length(alpha_grid))   # one row per α at lambda.1se
  
  tic(sprintf("Inner tuning (outer fold %d)", o))
  for (i in seq_along(alpha_grid)) {
    alpha <- alpha_grid[i]
    message(sprintf("  [Inner α %d/%d] alpha = %.1f  (manual 5-fold on OUTER-TRAIN)", i, length(alpha_grid), alpha))
    
    set.seed(123)
    cvmod <- cv.glmnet(
      x = X_train_outer,
      y = y_train_outer,
      weights = w_train_outer,
      type.measure = "mse",
      foldid = foldid_inner,      # manual inner folds
      grouped = TRUE,             # as requested
      keep = TRUE,                # keep OOF preds (cvmod$fit.preval)
      parallel = TRUE,
      trace.it = 0,
      alpha = alpha,
      family = "gaussian",
      standardize = TRUE,
      intercept = TRUE
    )
    
    # ---- lambda.min row (per α) ----
    idx_min     <- cvmod$index["min", 1]
    per_alpha_list_min[[i]] <- tibble::tibble(
      alpha       = alpha,
      alpha_idx   = i,
      s           = "lambda.min",
      lambda      = cvmod$lambda[idx_min],
      lambda_idx  = idx_min,
      nzero       = cvmod$nzero[idx_min],
      dev_ratio   = as.numeric(cvmod$glmnet.fit$dev.ratio[idx_min]),
      cvm         = as.numeric(cvmod$cvm[idx_min]),
      cvsd        = as.numeric(cvmod$cvsd[idx_min]),
      cvlo        = as.numeric(cvmod$cvlo[idx_min]),
      cvup        = as.numeric(cvmod$cvup[idx_min])
    )
    
    # ---- lambda.1se row (per α) ----
    idx_1se     <- cvmod$index["1se", 1]
    per_alpha_list_1se[[i]] <- tibble::tibble(
      alpha       = alpha,
      alpha_idx   = i,
      s           = "lambda.1se",
      lambda      = cvmod$lambda[idx_1se],
      lambda_idx  = idx_1se,
      nzero       = cvmod$nzero[idx_1se],
      dev_ratio   = as.numeric(cvmod$glmnet.fit$dev.ratio[idx_1se]),
      cvm         = as.numeric(cvmod$cvm[idx_1se]),
      cvsd        = as.numeric(cvmod$cvsd[idx_1se]),
      cvlo        = as.numeric(cvmod$cvlo[idx_1se]),
      cvup        = as.numeric(cvmod$cvup[idx_1se])
    )
  } # α loop
  toc()
  
  # Collate inner tuning results (one row per α per rule)
  tuning_results_min <- dplyr::bind_rows(per_alpha_list_min)
  tuning_results_1se <- dplyr::bind_rows(per_alpha_list_1se)
  
  # Winners across α (tie-breaks: fewer nzero → larger λ → smaller α)
  best_min <- tuning_results_min %>%
    dplyr::arrange(cvm, nzero, dplyr::desc(lambda), alpha) %>%
    dplyr::slice(1)
  
  best_1se <- tuning_results_1se %>%
    dplyr::arrange(cvm, nzero, dplyr::desc(lambda), alpha) %>%
    dplyr::slice(1)
  
  print(as.data.frame(best_min),  row.names = FALSE)
  print(as.data.frame(best_1se),  row.names = FALSE)
  message(sprintf("Inner winner @ lambda.min (outer %d): alpha=%.2f | lambda=%.6f | nzero=%d | CVM=%.5f (± %.5f)",
                  o, best_min$alpha, best_min$lambda, best_min$nzero, best_min$cvm, best_min$cvsd))
  message(sprintf("Inner winner @ lambda.1se (outer %d): alpha=%.2f | lambda=%.6f | nzero=%d | CVM=%.5f (± %.5f)",
                  o, best_1se$alpha, best_1se$lambda, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
  
  # Record winners (for audit)
  inner_winners_rows_min[[o]] <- tibble::tibble(
    outer_fold   = o,
    alpha        = best_min$alpha,
    alpha_idx    = best_min$alpha_idx,
    s            = best_min$s,
    lambda       = best_min$lambda,
    lambda_idx   = best_min$lambda_idx,
    nzero        = best_min$nzero,
    dev_ratio    = best_min$dev_ratio,
    cvm          = best_min$cvm,
    cvsd         = best_min$cvsd
  )
  inner_winners_rows_1se[[o]] <- tibble::tibble(
    outer_fold   = o,
    alpha        = best_1se$alpha,
    alpha_idx    = best_1se$alpha_idx,
    s            = best_1se$s,
    lambda       = best_1se$lambda,
    lambda_idx   = best_1se$lambda_idx,
    nzero        = best_1se$nzero,
    dev_ratio    = best_1se$dev_ratio,
    cvm          = best_1se$cvm,
    cvsd         = best_1se$cvsd
  )
  
  # --- Outer refit at the inner winners; evaluate on OUTER‑HOLDOUT ---
  
  # lambda.min
  fit_outer_min <- glmnet(
    x = X_train_outer, y = y_train_outer, weights = w_train_outer,
    family = "gaussian",
    alpha = best_min$alpha,
    lambda = best_min$lambda,
    standardize = TRUE, intercept = TRUE
  )
  pred_outer_hold_min <- as.numeric(predict(fit_outer_min, newx = X_hold_outer))
  fold_metrics_min <- compute_metrics(y_true = y_hold_outer, y_pred = pred_outer_hold_min, w = w_hold_outer)
  outer_metrics_min[[o]] <- fold_metrics_min
  outer_oof_pred_min[outer_hold_idx] <- pred_outer_hold_min
  message(sprintf("Outer %d @ lambda.min | RMSE=%.5f | MAE=%.5f | Bias=%.5f | Bias%%=%.3f | R2=%.5f",
                  o, fold_metrics_min["rmse"], fold_metrics_min["mae"], fold_metrics_min["bias"],
                  fold_metrics_min["bias_pct"], fold_metrics_min["r2"]))
  
  # lambda.1se
  fit_outer_1se <- glmnet(
    x = X_train_outer, y = y_train_outer, weights = w_train_outer,
    family = "gaussian",
    alpha = best_1se$alpha,
    lambda = best_1se$lambda,
    standardize = TRUE, intercept = TRUE
  )
  pred_outer_hold_1se <- as.numeric(predict(fit_outer_1se, newx = X_hold_outer))
  fold_metrics_1se <- compute_metrics(y_true = y_hold_outer, y_pred = pred_outer_hold_1se, w = w_hold_outer)
  outer_metrics_1se[[o]] <- fold_metrics_1se
  outer_oof_pred_1se[outer_hold_idx] <- pred_outer_hold_1se
  message(sprintf("Outer %d @ lambda.1se | RMSE=%.5f | MAE=%.5f | Bias=%.5f | Bias%%=%.3f | R2=%.5f",
                  o, fold_metrics_1se["rmse"], fold_metrics_1se["mae"], fold_metrics_1se["bias"],
                  fold_metrics_1se["bias_pct"], fold_metrics_1se["r2"]))
}
toc()
# > Nested Cross-Validation (glmnet; grouped=TRUE, keep=TRUE; lambda.min & lambda.1se): 7.888 sec elapsed

# Stop the parallel backend:
# doParallel::stopImplicitCluster()

#### ---- Aggregations over outer holds (both rules) ----

# Helpers to aggregate & print for a given rule
aggregate_rule <- function(outer_metrics_list, outer_oof_pred_vec, rule_label) {
  outer_metrics_matrix <- do.call(rbind, outer_metrics_list)
  metrics_outer_mean <- tibble::tibble(
    Rule = rule_label,
    Aggregation = "Simple-mean (outer folds)",
    RMSE = mean(outer_metrics_matrix[, "rmse"]),
    MAE  = mean(outer_metrics_matrix[, "mae"]),
    Bias = mean(outer_metrics_matrix[, "bias"]),
    `Bias%` = mean(outer_metrics_matrix[, "bias_pct"]),
    R2   = mean(outer_metrics_matrix[, "r2"])
  )
  stopifnot(all(!is.na(outer_oof_pred_vec)))
  pooled_vec <- compute_metrics(y_true = y_train, y_pred = outer_oof_pred_vec, w = w_train)
  metrics_outer_pooled <- tibble::tibble(
    Rule = rule_label,
    Aggregation = "Pooled-weighted OOF",
    RMSE = pooled_vec["rmse"],
    MAE  = pooled_vec["mae"],
    Bias = pooled_vec["bias"],
    `Bias%` = pooled_vec["bias_pct"],
    R2   = pooled_vec["r2"]
  )
  list(mean = metrics_outer_mean, pooled = metrics_outer_pooled,
       fold_matrix = outer_metrics_matrix)
}

agg_min  <- aggregate_rule(outer_metrics_min, outer_oof_pred_min, "lambda.min")
agg_1se  <- aggregate_rule(outer_metrics_1se, outer_oof_pred_1se, "lambda.1se")

# Report both rules side-by-side
message("\n==== Nested-CV aggregated performance (both rules) ====")
print(as.data.frame(dplyr::bind_rows(agg_min$mean, agg_min$pooled,
                                     agg_1se$mean, agg_1se$pooled)),
      row.names = FALSE)
# >       Rule               Aggregation     RMSE      MAE         Bias    Bias%         R2
# > lambda.min Simple-mean (outer folds) 91.14713 73.26421  0.003382402 3.609923 0.05141579
# > lambda.min       Pooled-weighted OOF 91.15820 73.27609 -0.003418394 3.609600 0.05165984
# > lambda.1se Simple-mean (outer folds) 91.63234 73.79330  0.038618824 3.697239 0.04129571
# > lambda.1se       Pooled-weighted OOF 91.64270 73.80510  0.032468960 3.697013 0.04155230

# Inner winners tables (for audit)
inner_winners_table_min <- dplyr::bind_rows(inner_winners_rows_min)
inner_winners_table_1se <- dplyr::bind_rows(inner_winners_rows_1se)

message("\n==== Inner winners by outer fold — lambda.min ====")
print(as.data.frame(inner_winners_table_min), row.names = FALSE)
# > outer_fold alpha alpha_idx          s   lambda lambda_idx nzero  dev_ratio      cvm      cvsd
# >          1     0         1 lambda.min 1.946972        100     4 0.05400632 8339.792 127.24488
# >          2     0         1 lambda.min 1.932126        100     4 0.05287971 8276.363 114.20922
# >          3     0         1 lambda.min 1.942695        100     4 0.05210320 8353.164  99.42507
# >          4     0         1 lambda.min 3.092842         95     4 0.05357119 8282.665  94.44261
# >          5     0         1 lambda.min 2.675982         97     4 0.05600287 8300.907 120.67357
message("\n==== Inner winners by outer fold — lambda.1se ====")
print(as.data.frame(inner_winners_table_1se), row.names = FALSE)
# > outer_fold alpha alpha_idx          s    lambda lambda_idx nzero  dev_ratio      cvm     cvsd
# >          1   0.4         5 lambda.1se 14.522694         14     3 0.04077433 8456.983 102.0847
# >          2   1.0        11 lambda.1se  6.326842         13     3 0.04125652 8376.297 114.0700
# >          3   0.6         7 lambda.1se  9.660525         14     2 0.04111135 8441.610 114.4581
# >          4   0.8         9 lambda.1se  6.600715         15     3 0.04353156 8365.974 114.4426
# >          5   0.1         2 lambda.1se 37.931360         19     4 0.04281270 8409.177 126.5648

# Per-fold outer holdout metrics tables
outer_folds_table_min <- tibble::tibble(
  outer_fold = seq_len(num_folds_outer),
  RMSE   = agg_min$fold_matrix[, "rmse"],
  MAE    = agg_min$fold_matrix[, "mae"],
  Bias   = agg_min$fold_matrix[, "bias"],
  `Bias%`= agg_min$fold_matrix[, "bias_pct"],
  R2     = agg_min$fold_matrix[, "r2"]
)
outer_folds_table_1se <- tibble::tibble(
  outer_fold = seq_len(num_folds_outer),
  RMSE   = agg_1se$fold_matrix[, "rmse"],
  MAE    = agg_1se$fold_matrix[, "mae"],
  Bias   = agg_1se$fold_matrix[, "bias"],
  `Bias%`= agg_1se$fold_matrix[, "bias_pct"],
  R2     = agg_1se$fold_matrix[, "r2"]
)

message("\n==== Per-fold outer holdout metrics — lambda.min ====")
print(as.data.frame(outer_folds_table_min), row.names = FALSE)
# > outer_fold     RMSE      MAE       Bias    Bias%         R2
# >          1 90.51214 71.97217 -2.1352939 3.109354 0.05072207
# >          2 91.74644 74.11427 -0.1800436 3.672001 0.05446979
# >          3 90.25410 72.17687  2.7640966 4.090455 0.05756811
# >          4 91.78449 74.12674  2.1676703 4.096750 0.05259027
# >          5 91.43848 73.93099 -2.5995174 3.081056 0.04172871
message("\n==== Per-fold outer holdout metrics — lambda.1se ====")
print(as.data.frame(outer_folds_table_1se), row.names = FALSE)
# > outer_fold     RMSE      MAE       Bias    Bias%         R2
# >          1 90.93323 72.81726 -2.5039033 3.111984 0.04186873
# >          2 92.35731 74.67685 -0.1143671 3.766336 0.04183664
# >          3 91.02259 72.68704  2.7529403 4.174469 0.04145070
# >          4 92.23652 74.54060  2.2593688 4.187478 0.04323567
# >          5 91.61207 74.24477 -2.2009446 3.245926 0.03808682

# Consolidated: inner winners + outer holdout performance — both rules
message("\n==== Consolidated (winners + outer performance) — both rules ====")
consolidated_min <- inner_winners_table_min %>%
  dplyr::left_join(outer_folds_table_min, by = "outer_fold") %>%
  dplyr::mutate(Rule = "lambda.min")

consolidated_1se <- inner_winners_table_1se %>%
  dplyr::left_join(outer_folds_table_1se, by = "outer_fold") %>%
  dplyr::mutate(Rule = "lambda.1se")

consolidated_both <- dplyr::bind_rows(consolidated_min, consolidated_1se) %>%
  dplyr::relocate(Rule, outer_fold)

print(as.data.frame(consolidated_both), row.names = FALSE)
# >       Rule outer_fold alpha alpha_idx          s    lambda lambda_idx nzero  dev_ratio      cvm      cvsd     RMSE      MAE       Bias    Bias%         R2
# > lambda.min          1   0.0         1 lambda.min  1.946972        100     4 0.05400632 8339.792 127.24488 90.51214 71.97217 -2.1352939 3.109354 0.05072207
# > lambda.min          2   0.0         1 lambda.min  1.932126        100     4 0.05287971 8276.363 114.20922 91.74644 74.11427 -0.1800436 3.672001 0.05446979
# > lambda.min          3   0.0         1 lambda.min  1.942695        100     4 0.05210320 8353.164  99.42507 90.25410 72.17687  2.7640966 4.090455 0.05756811
# > lambda.min          4   0.0         1 lambda.min  3.092842         95     4 0.05357119 8282.665  94.44261 91.78449 74.12674  2.1676703 4.096750 0.05259027
# > lambda.min          5   0.0         1 lambda.min  2.675982         97     4 0.05600287 8300.907 120.67357 91.43848 73.93099 -2.5995174 3.081056 0.04172871
# > lambda.1se          1   0.4         5 lambda.1se 14.522694         14     3 0.04077433 8456.983 102.08473 90.93323 72.81726 -2.5039033 3.111984 0.04186873
# > lambda.1se          2   1.0        11 lambda.1se  6.326842         13     3 0.04125652 8376.297 114.07000 92.35731 74.67685 -0.1143671 3.766336 0.04183664
# > lambda.1se          3   0.6         7 lambda.1se  9.660525         14     2 0.04111135 8441.610 114.45809 91.02259 72.68704  2.7529403 4.174469 0.04145070
# > lambda.1se          4   0.8         9 lambda.1se  6.600715         15     3 0.04353156 8365.974 114.44257 92.23652 74.54060  2.2593688 4.187478 0.04323567
# > lambda.1se          5   0.1         2 lambda.1se 37.931360         19     4 0.04281270 8409.177 126.56476 91.61207 74.24477 -2.2009446 3.245926 0.03808682
