# ---- III. Predictive Modelling: Version 3.4 ----
#
# Nested Cross-Validation: 
#  inner - xgb.cv with manual K-folds + OOF reconstruction (tuning hyperparameters)
#  outer - xgb.train (evaluating performance)

## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)        # Read SPSS .sav files
library(tidyverse)    # Includes dplyr, tidyr, purrr, ggplot2, tibble, etc.
library(broom)        # For tidying model output
library(tictoc)       # For timing code execution
library(xgboost)      # For implementing XGBoost model 
library(DiagrammeR)

# Load data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE)
dim(pisa_2022_student_canada)   # 23073 x 1278

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") 

# Constants
M <- 10                  # Number of plausible values
G <- 80                  # Number of BRR replicate weights
k <- 0.5                 # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)   # 95% CI z-critical value

# Target varaible
pvmaths  <- paste0("PV", 1:M, "MATH")   # PV1MATH to PV10MATH

# Predictors
oos <- c("EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME")  

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)   # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                 # Final student weight

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

### ---- Random Train/Validation/Test (70/15/15) split ----
set.seed(123)          # Ensure reproducibility
n <- nrow(temp_data)   # 20003
indices <- sample(n)   # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.70 * n)         # 14002
n_valid <- floor(0.15 * n)         # 3000
n_test  <- n - n_train - n_valid   # 3001

# Assign indices
train_idx <- indices[1:n_train]
valid_idx <- indices[(n_train + 1):(n_train + n_valid)]
test_idx  <- indices[(n_train + n_valid + 1):n]

# Subset the data
train_data <- temp_data[train_idx, ]
valid_data <- temp_data[valid_idx, ]
test_data  <- temp_data[test_idx, ]

## ---- Main model using final student weights (W_FSTUWT) ----
### ---- Nested Cross-Validation for PV1MATH only ----

# Define target plausible value
pv1math <- pvmaths[1]   # "PV1MATH"

# Prepare training/validation/test split columns
X_train <- train_data[, oos]
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

X_valid <- valid_data[, oos]
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

X_test  <- test_data[,  oos]
y_test  <- test_data[[pv1math]]
w_test  <- test_data[[final_wt]]

# Create DMatrix (note: dvalid/dtest not used in nested CV; kept for symmetry)
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train, weight = w_train)
dvalid <- xgb.DMatrix(data = as.matrix(X_valid), label = y_valid, weight = w_valid)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test,  weight = w_test)

# --- Helpers: folds and weights ---

# Build K manual folds (indices) with a fixed seed; over N rows
make_folds <- function(n_cv, num_folds = 5L, seed = 123L) {
  set.seed(seed)
  cv_order <- sample.int(n_cv)
  bounds <- floor(seq(0, n_cv, length.out = num_folds + 1))
  cv_folds <- vector("list", num_folds)
  for (k in seq_len(num_folds)) cv_folds[[k]] <- cv_order[(bounds[k] + 1):bounds[k + 1]]
  stopifnot(identical(sort(unlist(cv_folds)), seq_len(n_cv)))
  cv_folds
}

# Print weight diagnostics 
print_weight_balance <- function(w, folds, title = "Weight balance") {
  s <- tibble::tibble(
    fold   = seq_along(folds),
    n      = vapply(folds, length, integer(1)),
    w_sum  = vapply(folds, function(idx) sum(w[idx]), numeric(1)),
    w_mean = vapply(folds, function(idx) mean(w[idx]), numeric(1)),
    w_med  = vapply(folds, function(idx) median(w[idx]), numeric(1)),
    w_effn = vapply(folds, function(idx) {
      wi <- w[idx]; (sum(wi)^2) / sum(wi^2)
    }, numeric(1))
  ) |>
    dplyr::mutate(w_share = w_sum / sum(w_sum))
  message(title)
  print(s, n = Inf, width = Inf)
  cat(sprintf("Weight share range: [%.3f, %.3f]\n", min(s$w_share), max(s$w_share)))
  cat(sprintf("Max/Min share ratio: %.3f\n", max(s$w_share) / min(s$w_share)))
  cat(sprintf("Coeff. of variation of shares: %.3f\n\n", stats::sd(s$w_share) / mean(s$w_share)))
}

# Assert xgboost callback exists 
if (is.null(tryCatch(getFromNamespace("cb.cv.predict", "xgboost"), error = function(e) NULL))) {
  stop("xgboost build lacks cb.cv.predict(save_models=TRUE). Update/repair your xgboost installation.")
}

# --- Define the hyperparameter grid  ---
grid <- expand.grid(
  nrounds   = c(100, 200, 300),
  max_depth = c(4, 6, 8),
  eta       = c(0.01, 0.05, 0.10, 0.30),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)
stopifnot(is.data.frame(grid), nrow(grid) >= 1)

# --- Outer folds over TRAIN (X_train / y_train / w_train) ---
num_folds_outer <- 5L
n_cv_outer <- nrow(X_train)
outer_folds <- make_folds(n_cv_outer, num_folds_outer)

# Show outer-fold weight diagnostics
print_weight_balance(w_train, outer_folds, title = "Outer CV: weight balance over TRAIN")
# > Outer CV: weight balance over TRAIN
# > # A tibble: 5 × 7
# >    fold     n  w_sum w_mean w_med w_effn w_share
# >   <int> <int>  <dbl>  <dbl> <dbl>  <dbl>   <dbl>
# > 1     1  2800 44065.   15.7  8.93  1569.   0.199
# > 2     2  2800 45527.   16.3  9.60  1067.   0.205
# > 3     3  2801 43273.   15.4  8.86  1571.   0.195
# > 4     4  2800 44610.   15.9  9.49  1615.   0.201
# > 5     5  2801 44154.   15.8  9.37  1581.   0.199
# > Weight share range: [0.195, 0.205]
# > Max/Min share ratio: 1.052
# > Coeff. of variation of shares: 0.019

# Storage for outer predictions/metrics (@ OOF-best only)
outer_oof_pred      <- rep(NA_real_, n_cv_outer)
outer_metrics       <- vector("list", num_folds_outer)
inner_winners_rows  <- vector("list", num_folds_outer)

# --- Nested loop: For each outer fold, run inner tuner on OUTER-TRAIN ---
tictoc::tic("Nested Cross-Validation")
for (o in seq_len(num_folds_outer)) {
  message(sprintf("\n==== Outer fold %d/%d ====", o, num_folds_outer))
  outer_hold_idx  <- outer_folds[[o]]
  outer_train_idx <- setdiff(seq_len(n_cv_outer), outer_hold_idx)
  
  # OUTER-TRAIN / OUTER-HOLDOUT matrices
  dtrain_outer   <- xgb.DMatrix(
    data  = as.matrix(X_train[outer_train_idx, , drop = FALSE]),
    label = y_train[outer_train_idx],
    weight= w_train[outer_train_idx]
  )
  dholdout_outer <- xgb.DMatrix(
    data  = as.matrix(X_train[outer_hold_idx, , drop = FALSE]),
    label = y_train[outer_hold_idx],
    weight= w_train[outer_hold_idx]
  )
  
  # --- Inner folds (on OUTER-TRAIN) ---
  n_cv_inner <- length(outer_train_idx)
  num_folds_inner <- 5L
  inner_folds_local <- make_folds(n_cv_inner, num_folds_inner)
  
  # Map inner folds (local -> absolute indices in full TRAIN)
  inner_folds_abs <- lapply(inner_folds_local, function(idxs) outer_train_idx[idxs])
  stopifnot(all(vapply(inner_folds_abs, function(v) all(v %in% outer_train_idx), logical(1))))
  
  # Inner weight diagnostics 
  print_weight_balance(w_train[outer_train_idx], inner_folds_local,
                       title = sprintf("Inner CV (outer fold %d): weight balance over OUTER-TRAIN", o))
  
  # Held-out DMatrices per inner fold (reused for OOF predictions)
  dvalid_fold_inner <- lapply(seq_len(num_folds_inner), function(k) {
    idx_abs <- inner_folds_abs[[k]]
    xgb.DMatrix(
      as.matrix(X_train[idx_abs, , drop = FALSE]),
      label = y_train[idx_abs], weight = w_train[idx_abs]
    )
  })
  
  # --- Inner tuner (CV+OOF tracking) ---
  best_rmse_by_cv_inner  <- Inf
  best_rmse_by_oof_inner <- Inf
  best_cv_info  <- NULL
  best_oof_info <- NULL
  
  # Containers (per inner run, for inspection)
  tuning_results <- tibble::tibble(
    grid_id = integer(),
    param_name = character(),
    nrounds = integer(),
    max_depth = integer(),
    eta = double(),
    best_iter_by_cv = integer(),
    best_rmse_by_cv = double(),
    test_rmse_std_at_cvbest   = double(),
    train_rmse_mean_at_cvbest = double(),
    train_rmse_std_at_cvbest  = double(),
    best_iter_by_oof = integer(),
    best_rmse_by_oof = double(),
    rmse_oof_at_cvbest_iter   = double(),
    rmse_oof_at_final_iter    = double()
  )
  cv_eval_list   <- vector("list", nrow(grid))
  oof_curve_list <- vector("list", nrow(grid))
  
  tictoc::tic(sprintf("Inner tuning (outer fold %d)", o))
  for (i in seq_len(nrow(grid))) {
    row <- grid[i, ]
    message(sprintf("  [Inner %d/%d] nrounds=%d, max_depth=%d, eta=%.3f",
                    i, nrow(grid), row$nrounds, row$max_depth, row$eta))
    
    params <- list(
      objective   = "reg:squarederror",
      max_depth   = row$max_depth,
      eta         = row$eta,
      eval_metric = "rmse",
      nthread     = max(1, parallel::detectCores() - 1)
    )
    
    # 1) xgb.cv on OUTER-TRAIN with manual inner folds
    cv_mod <- xgb.cv(
      params  = params,
      data    = dtrain_outer,
      nrounds = row$nrounds,
      folds   = inner_folds_local,    # local (relative) folds for dtrain_outer
      showsd  = TRUE,
      verbose = FALSE,
      early_stopping_rounds = NULL,
      prediction = TRUE,
      stratified = FALSE,
      callbacks = list(getFromNamespace("cb.cv.predict", "xgboost")(save_models = TRUE))
    )
    
    # CV rule (fold-mean)
    best_iter_by_cv_i <- which.min(cv_mod$evaluation_log$test_rmse_mean)
    best_rmse_by_cv_i <- cv_mod$evaluation_log$test_rmse_mean[best_iter_by_cv_i]
    
    # Sanity: ensure fold models exist
    stopifnot(!is.null(cv_mod$models), length(cv_mod$models) == num_folds_inner)
    
    # 2) Manual OOF reconstruction across inner folds, pooled weighted
    oof_pred_matrix_i  <- matrix(NA_real_, nrow = n_cv_inner, ncol = row$nrounds)
    rmse_oof_by_iter_i <- numeric(row$nrounds)
    for (iter in seq_len(row$nrounds)) {
      for (k in seq_len(num_folds_inner)) {
        idx_abs <- inner_folds_abs[[k]]
        idx_loc <- match(idx_abs, outer_train_idx)  # absolute -> local positions (1..n_cv_inner)
        stopifnot(!any(is.na(idx_loc)))
        oof_pred_matrix_i[idx_loc, iter] <- predict(
          cv_mod$models[[k]],
          dvalid_fold_inner[[k]],
          iterationrange = c(1, iter + 1)
        )
      }
      rmse_oof_by_iter_i[iter] <- sqrt(
        sum(w_train[outer_train_idx] * (y_train[outer_train_idx] - oof_pred_matrix_i[, iter])^2) /
          sum(w_train[outer_train_idx])
      )
    }
    best_iter_by_oof_i        <- which.min(rmse_oof_by_iter_i)
    best_rmse_by_oof_i        <- rmse_oof_by_iter_i[best_iter_by_oof_i]
    rmse_oof_at_cvbest_iter_i <- rmse_oof_by_iter_i[best_iter_by_cv_i]
    rmse_oof_at_final_iter_i  <- sqrt(
      sum(w_train[outer_train_idx] * (y_train[outer_train_idx] - cv_mod$pred)^2) /
        sum(w_train[outer_train_idx])
    )
    # At last iter, reconstructed OOF equals cv_mod$pred
    stopifnot(isTRUE(all.equal(oof_pred_matrix_i[, row$nrounds], cv_mod$pred)))
    
    # Save logs for this config
    cv_eval_list[[i]]   <- dplyr::mutate(as.data.frame(cv_mod$evaluation_log), grid_id = i)
    oof_curve_list[[i]] <- rmse_oof_by_iter_i
    tuning_results <- dplyr::add_row(
      tuning_results,
      grid_id = i,
      param_name = sprintf("nrounds=%d, max_depth=%d, eta=%.3f", row$nrounds, row$max_depth, row$eta),
      nrounds = row$nrounds,
      max_depth = row$max_depth,
      eta = row$eta,
      best_iter_by_cv = best_iter_by_cv_i,
      best_rmse_by_cv = best_rmse_by_cv_i,
      test_rmse_std_at_cvbest   = cv_mod$evaluation_log$test_rmse_std[best_iter_by_cv_i],
      train_rmse_mean_at_cvbest = cv_mod$evaluation_log$train_rmse_mean[best_iter_by_cv_i],
      train_rmse_std_at_cvbest  = cv_mod$evaluation_log$train_rmse_std[best_iter_by_cv_i],
      best_iter_by_oof = best_iter_by_oof_i,
      best_rmse_by_oof = best_rmse_by_oof_i,
      rmse_oof_at_cvbest_iter   = rmse_oof_at_cvbest_iter_i,
      rmse_oof_at_final_iter    = rmse_oof_at_final_iter_i
    )
    
    # Track winners (CV rule and OOF rule)
    if (best_rmse_by_cv_i < best_rmse_by_cv_inner) {
      best_rmse_by_cv_inner <- best_rmse_by_cv_i
      best_cv_info <- list(
        grid_id = i, params = params, row = row,
        best_iter_by_cv  = best_iter_by_cv_i,
        best_iter_by_oof = best_iter_by_oof_i,
        cv_eval  = cv_eval_list[[i]],
        oof_curve= oof_curve_list[[i]]
      )
    }
    if (best_rmse_by_oof_i < best_rmse_by_oof_inner) {
      best_rmse_by_oof_inner <- best_rmse_by_oof_i
      best_oof_info <- list(
        grid_id = i, params = params, row = row,
        best_iter_by_cv  = best_iter_by_cv_i,
        best_iter_by_oof = best_iter_by_oof_i,
        cv_eval  = cv_eval_list[[i]],
        oof_curve= oof_curve_list[[i]]
      )
    }
  } # grid loop
  tictoc::toc()
  
  # Inspect winners (optional summaries)
  message("Top configs by CV fold-mean (inner):")
  tuning_results %>%
    dplyr::arrange(best_rmse_by_cv) %>% head(5) %>% as.data.frame() %>% print(row.names = FALSE)
  
  message("Top configs by pooled OOF RMSE (inner):")
  tuning_results %>%
    dplyr::arrange(best_rmse_by_oof) %>% head(5) %>% as.data.frame() %>% print(row.names = FALSE)
  
  # Assert CV winner and OOF winner are the same config
  stopifnot(best_cv_info$grid_id == best_oof_info$grid_id)
  
  # Record the inner winners (one row per outer fold)
  inner_winners_rows[[o]] <- tibble::tibble(
    outer_fold       = o,
    grid_id          = best_oof_info$grid_id,
    n_cap            = best_oof_info$row$nrounds,
    max_depth        = best_oof_info$row$max_depth,
    eta              = best_oof_info$row$eta,
    best_iter_by_cv  = best_oof_info$best_iter_by_cv,
    best_rmse_by_cv  = min(best_oof_info$cv_eval$test_rmse_mean),
    best_iter_by_oof = best_oof_info$best_iter_by_oof,
    best_rmse_by_oof = min(best_oof_info$oof_curve)
  )
  
  message(sprintf("Inner winners agree (grid %d): n_cap=%d, max_depth=%d, eta=%.3f | CV-best iter=%d | OOF-best iter=%d",
                  best_oof_info$grid_id, best_oof_info$row$nrounds,
                  best_oof_info$row$max_depth, best_oof_info$row$eta,
                  best_oof_info$best_iter_by_cv, best_oof_info$best_iter_by_oof))
  
  # --- Outer refit: once to n_cap (no early stopping; no watchlist logging of holdout) ---
  main_model_outer <- xgb.train(
    params  = list(                                  # <=> params = best_oof_info$params
      objective = "reg:squarederror",
      max_depth = best_oof_info$params$max_depth,
      eta       = best_oof_info$params$eta,
      eval_metric = "rmse",
      nthread   = max(1, parallel::detectCores() - 1)
    ),
    data    = dtrain_outer,
    nrounds = best_oof_info$row$nrounds,             # n_cap
    verbose = 1,
    early_stopping_rounds = NULL
  )
  
  # Predict OUTER-HOLDOUT at inner OOF-best
  pred_outer_hold <- predict(
    main_model_outer, dholdout_outer,
    iterationrange = c(1, best_oof_info$best_iter_by_oof + 1)
  )
  
  # Metrics on OUTER-HOLDOUT @ OOF-best only
  fold_metrics <- compute_metrics(
    y_true = y_train[outer_hold_idx],
    y_pred = pred_outer_hold,
    w      = w_train[outer_hold_idx]
  )
  outer_metrics[[o]] <- fold_metrics
  
  # Save into global outer OOF vector
  outer_oof_pred[outer_hold_idx] <- pred_outer_hold
  
  message(sprintf("Outer fold %d metrics @ OOF-best | RMSE=%.5f | MAE=%.5f | Bias=%.5f | Bias%%=%.3f | R2=%.5f",
                  o, fold_metrics["rmse"], fold_metrics["mae"], fold_metrics["bias"],
                  fold_metrics["bias_pct"], fold_metrics["r2"]))
}
tictoc::toc()
# > Nested Cross-Validation: 555.264 sec elapsed

#### ---- Aggregation over outer holds ----

# 1) Simple average of the 5 per-fold metrics
outer_metrics_matrix <- do.call(rbind, outer_metrics)
metrics_outer_mean <- tibble::tibble(
  Aggregation = "Simple-mean (outer folds)",
  RMSE   = mean(outer_metrics_matrix[, "rmse"]),
  MAE    = mean(outer_metrics_matrix[, "mae"]),
  Bias   = mean(outer_metrics_matrix[, "bias"]),
  `Bias%`= mean(outer_metrics_matrix[, "bias_pct"]),
  R2     = mean(outer_metrics_matrix[, "r2"])
)

# 2) Pooled weighted OOF across all outer holdouts
stopifnot(all(!is.na(outer_oof_pred)))
metrics_outer_pooled_vec <- compute_metrics(
  y_true = y_train,
  y_pred = outer_oof_pred,
  w      = w_train
)
metrics_outer_pooled <- tibble::tibble(
  Aggregation = "Pooled-weighted OOF",
  RMSE   = metrics_outer_pooled_vec["rmse"],
  MAE    = metrics_outer_pooled_vec["mae"],
  Bias   = metrics_outer_pooled_vec["bias"],
  `Bias%`= metrics_outer_pooled_vec["bias_pct"],
  R2     = metrics_outer_pooled_vec["r2"]
)

# Report both
message("\n==== Nested-CV aggregated performance (@ inner OOF-best only) ====")
print(as.data.frame(dplyr::bind_rows(metrics_outer_mean, metrics_outer_pooled)), row.names = FALSE)
# >               Aggregation     RMSE      MAE      Bias    Bias%         R2
# > Simple-mean (outer folds) 89.48667 71.82019 -2.424188 2.955675 0.08567885
# >       Pooled-weighted OOF 89.51628 71.83841 -2.417535 2.958584 0.08551480

# Inner winners table (one row per outer fold) 
inner_winners_table <- dplyr::bind_rows(inner_winners_rows)
message("\n==== Inner winners (by outer fold) ====")
print(as.data.frame(inner_winners_table), row.names = FALSE)
# > outer_fold grid_id n_cap max_depth  eta best_iter_by_cv best_rmse_by_cv best_iter_by_oof best_rmse_by_oof
# >          1      19   100         4 0.10              49        89.80442               49         89.78712
# >          2      28   100         4 0.30              14        88.98448               14         88.99512
# >          3      10   100         4 0.05             100        90.04623              100         90.04102
# >          4      11   200         4 0.05             117        89.24695              117         89.25851
# >          5      19   100         4 0.10              50        89.43701               50         89.44419

# Return per-fold table for audit
outer_folds_table <- tibble::tibble(
  outer_fold = seq_len(num_folds_outer),
  RMSE   = outer_metrics_matrix[, "rmse"],
  MAE    = outer_metrics_matrix[, "mae"],
  Bias   = outer_metrics_matrix[, "bias"],
  `Bias%`= outer_metrics_matrix[, "bias_pct"],
  R2     = outer_metrics_matrix[, "r2"]
)
print(as.data.frame(outer_folds_table), row.names = FALSE)
# > outer_fold     RMSE      MAE      Bias    Bias%         R2
# >          1 88.84984 70.58234 -4.654410 2.434079 0.08526975
# >          2 91.14974 73.38226 -2.008353 3.181205 0.06672880
# >          3 87.46807 70.39541 -1.437836 3.025419 0.11485340
# >          4 90.43158 72.97910  1.027013 3.723020 0.08031438
# >          5 89.53414 71.76183 -5.047354 2.414652 0.08122791

# Consolidated table: join inner winners with outer-fold performance
message("\n==== Consolidated table: inner winners + outer holdout performance ====")
inner_winners_table %>%
  dplyr::left_join(outer_folds_table, by = "outer_fold") %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > outer_fold grid_id n_cap max_depth  eta best_iter_by_cv best_rmse_by_cv best_iter_by_oof best_rmse_by_oof     RMSE      MAE      Bias    Bias%         R2
# >          1      19   100         4 0.10              49        89.80442               49         89.78712 88.84984 70.58234 -4.654410 2.434079 0.08526975
# >          2      28   100         4 0.30              14        88.98448               14         88.99512 91.14974 73.38226 -2.008353 3.181205 0.06672880
# >          3      10   100         4 0.05             100        90.04623              100         90.04102 87.46807 70.39541 -1.437836 3.025419 0.11485340
# >          4      11   200         4 0.05             117        89.24695              117         89.25851 90.43158 72.97910  1.027013 3.723020 0.08031438
# >          5      19   100         4 0.10              50        89.43701               50         89.44419 89.53414 71.76183 -5.047354 2.414652 0.08122791