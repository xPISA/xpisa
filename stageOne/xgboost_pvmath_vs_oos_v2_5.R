# ---- II. Predictive Modelling: Version 2.5 ----

# xgb.train with manual K-folds + OOF reconstruction (tuning hyperparameters)

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

### ---- Tune XGBoost model for PV1MATH only: xgb.train + Manual Folds + OOF ----

# Define target plausible value
pv1math <- pvmaths[1]   # "PV1MATH"

# Prepare training/validation/test data
X_train <- train_data[, oos]
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

X_valid <- valid_data[, oos]
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

X_test  <- test_data[,  oos]
y_test  <- test_data[[pv1math]]
w_test  <- test_data[[final_wt]]

# Create DMatrix 
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train, weight = w_train)
dvalid <- xgb.DMatrix(data = as.matrix(X_valid), label = y_valid, weight = w_valid)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test,  weight = w_test)

# Manual CV folds (fixed for reproducibility; over TRAIN only) 
set.seed(123)
num_folds <- 5L
n_cv <- nrow(X_train)
cv_order <- sample.int(n_cv)  # permutation of train rows
bounds <- floor(seq(0, n_cv, length.out = num_folds + 1))
cv_folds <- vector("list", num_folds)
for (k in seq_len(num_folds)) {
  cv_folds[[k]] <- cv_order[(bounds[k] + 1):bounds[k + 1]]
}
stopifnot(identical(sort(unlist(cv_folds)), seq_len(n_cv)))

# Check CV-fold weight balance (size, weight share, effective n)
local({
  s <- tibble::tibble(
    fold   = seq_along(cv_folds),
    n      = vapply(cv_folds, length, integer(1)),
    w_sum  = vapply(cv_folds, function(idx) sum(w_train[idx],   na.rm = TRUE), numeric(1)),
    w_mean = vapply(cv_folds, function(idx) mean(w_train[idx],  na.rm = TRUE), numeric(1)),
    w_med  = vapply(cv_folds, function(idx) median(w_train[idx],na.rm = TRUE), numeric(1)),
    w_effn = vapply(cv_folds, function(idx) {
      w <- w_train[idx]; (sum(w, na.rm = TRUE)^2) / sum(w^2, na.rm = TRUE)
    }, numeric(1))
  ) |>
    dplyr::mutate(w_share = w_sum / sum(w_sum, na.rm = TRUE))
  print(s, n = Inf, width = Inf)
  cat(sprintf("Weight share range: [%.3f, %.3f]\n",
              min(s$w_share, na.rm = TRUE), max(s$w_share, na.rm = TRUE)))
  cat(sprintf("Max/Min share ratio: %.3f\n",
              max(s$w_share, na.rm = TRUE) / min(s$w_share, na.rm = TRUE)))
  cat(sprintf("Coeff. of variation of shares: %.3f\n",
              stats::sd(s$w_share, na.rm = TRUE) / mean(s$w_share, na.rm = TRUE)))
})

# Held-out DMatrices per fold (reused for OOF predictions)
dvalid_fold <- lapply(seq_len(num_folds), function(k) {
  idx <- cv_folds[[k]]
  xgb.DMatrix(as.matrix(X_train[idx, , drop = FALSE]),
              label = y_train[idx], weight = w_train[idx])
})

# Hyperparameter grid (3 × 3 × 4 = 36) 
grid <- expand.grid(
  nrounds   = c(100, 200, 300),
  max_depth = c(4, 6, 8),
  eta       = c(0.01, 0.05, 0.10, 0.30),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

# Containers across grid (include BOTH CV- and OOF-based summaries per config)
tuning_results <- tibble::tibble(
  grid_id = integer(),
  param_name = character(),
  nrounds = integer(),
  max_depth = integer(),
  eta = double(),
  # CV-based (fold-mean)
  best_iter_by_cv = integer(),
  best_rmse_by_cv = double(),
  test_rmse_std_at_cvbest   = double(),
  train_rmse_mean_at_cvbest = double(),
  train_rmse_std_at_cvbest  = double(),
  # OOF-based (pooled, weighted)
  best_iter_by_oof = integer(),
  best_rmse_by_oof = double(),
  rmse_oof_at_cvbest_iter   = double(),
  rmse_oof_at_final_iter    = double()
)

cv_eval_list   <- vector("list", nrow(grid))   # per-config CV evaluation log (means & sds)
oof_curve_list <- vector("list", nrow(grid))   # per-config OOF RMSE curve (per iteration)

# Track BOTH winners across the grid
best_rmse_by_cv  <- Inf
best_rmse_by_oof <- Inf
best_cv_info  <- NULL   # list(grid_id, params, row, best_iter_by_cv, best_iter_by_oof, cv_eval, oof_curve)
best_oof_info <- NULL

# Grid search (manual CV via xgb.train + pooled OOF)
set.seed(123)
tictoc::tic("Tuning (xgb.train + Manual OOF)")
for (i in seq_len(nrow(grid))) {
  row <- grid[i, ]
  message(sprintf("Grid %d/%d: nrounds=%d, max_depth=%d, eta=%.3f",
                  i, nrow(grid), row$nrounds, row$max_depth, row$eta))
  
  params <- list(
    objective   = "reg:squarederror",
    max_depth   = row$max_depth,
    eta         = row$eta,
    eval_metric = "rmse",
    nthread     = max(1, parallel::detectCores() - 1)
  )
  
  # Train fold boosters to full cap via xgb.train
  dtrain_fold <- vector("list", num_folds)
  mod_fold    <- vector("list", num_folds)
  eval_fold   <- vector("list", num_folds)
  
  for (k in seq_len(num_folds)) {
    valid_idx_k <- cv_folds[[k]]
    train_idx_k <- setdiff(seq_len(n_cv), valid_idx_k)
    
    dtrain_fold[[k]] <- xgb.DMatrix(as.matrix(X_train[train_idx_k, , drop=FALSE]),
                                    label = y_train[train_idx_k],
                                    weight = w_train[train_idx_k])
    # dvalid_fold[[k]] already prepared above
    
    mod_fold[[k]] <- xgb.train(
      params  = params,
      data    = dtrain_fold[[k]],
      nrounds = row$nrounds,
      watchlist = list(train = dtrain_fold[[k]], valid = dvalid_fold[[k]]),
      verbose = 1,
      early_stopping_rounds = NULL
    )
    eval_fold[[k]] <- as.data.frame(mod_fold[[k]]$evaluation_log)  # iter, train_rmse, valid_rmse
    stopifnot(nrow(eval_fold[[k]]) == row$nrounds)
  }
  
  # Aggregate fold logs to mimic xgb.cv$evaluation_log
  train_rmse_matrix <- do.call(cbind, lapply(eval_fold, function(df) df$train_rmse))
  valid_rmse_matrix <- do.call(cbind, lapply(eval_fold, function(df) df$valid_rmse))
  
  cv_eval_log <- tibble::tibble(
    iter             = seq_len(row$nrounds),
    train_rmse_mean  = rowMeans(train_rmse_matrix),
    train_rmse_std   = sqrt(pmax(0, rowMeans(train_rmse_matrix^2) - rowMeans(train_rmse_matrix)^2)),
    test_rmse_mean   = rowMeans(valid_rmse_matrix),
    test_rmse_std    = sqrt(pmax(0, rowMeans(valid_rmse_matrix^2) - rowMeans(valid_rmse_matrix)^2))
  )
  
  # CV-best iteration (fold-mean)
  best_iter_by_cv_i <- which.min(cv_eval_log$test_rmse_mean)
  best_rmse_by_cv_i <- cv_eval_log$test_rmse_mean[best_iter_by_cv_i]
  
  # Reconstruct pooled OOF curve across iterations
  oof_pred_matrix_i <- matrix(NA_real_, nrow = n_cv, ncol = row$nrounds)
  rmse_oof_by_iter_i <- numeric(row$nrounds)
  for (iter in seq_len(row$nrounds)) {
    for (k in seq_len(num_folds)) {
      idx <- cv_folds[[k]]
      oof_pred_matrix_i[idx, iter] <- predict(
        mod_fold[[k]],
        dvalid_fold[[k]],
        iterationrange = c(1, iter + 1)
      )
    }
    rmse_oof_by_iter_i[iter] <- sqrt(
      sum(w_train * (y_train - oof_pred_matrix_i[, iter])^2) / sum(w_train)
    )
  }
  best_iter_by_oof_i <- which.min(rmse_oof_by_iter_i)
  best_rmse_by_oof_i <- rmse_oof_by_iter_i[best_iter_by_oof_i]
  rmse_oof_at_cvbest_iter_i <- rmse_oof_by_iter_i[best_iter_by_cv_i]
  rmse_oof_at_final_iter_i  <- rmse_oof_by_iter_i[row$nrounds]
  
  # Save per-config artifacts
  cv_eval_list[[i]]   <- dplyr::mutate(cv_eval_log, grid_id = i)
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
    test_rmse_std_at_cvbest   = cv_eval_log$test_rmse_std[best_iter_by_cv_i],
    train_rmse_mean_at_cvbest = cv_eval_log$train_rmse_mean[best_iter_by_cv_i],
    train_rmse_std_at_cvbest  = cv_eval_log$train_rmse_std[best_iter_by_cv_i],
    best_iter_by_oof = best_iter_by_oof_i,
    best_rmse_by_oof = best_rmse_by_oof_i,
    rmse_oof_at_cvbest_iter   = rmse_oof_at_cvbest_iter_i,
    rmse_oof_at_final_iter    = rmse_oof_at_final_iter_i
  )
  
  # Track best-by-CV across grid
  if (best_rmse_by_cv_i < best_rmse_by_cv) {
    best_rmse_by_cv <- best_rmse_by_cv_i
    best_cv_info <- list(
      grid_id = i, params = params, row = row,
      best_iter_by_cv  = best_iter_by_cv_i,
      best_iter_by_oof = best_iter_by_oof_i,
      cv_eval  = cv_eval_list[[i]],
      oof_curve= oof_curve_list[[i]]
    )
  }
  # Track best-by-OOF across grid
  if (best_rmse_by_oof_i < best_rmse_by_oof) {
    best_rmse_by_oof <- best_rmse_by_oof_i
    best_oof_info <- list(
      grid_id = i, params = params, row = row,
      best_iter_by_cv  = best_iter_by_cv_i,
      best_iter_by_oof = best_iter_by_oof_i,
      cv_eval  = cv_eval_list[[i]],
      oof_curve= oof_curve_list[[i]]
    )
  }
}
tictoc::toc()
# > Tuning (xgb.train + Manual OOF): 145.816 sec elapsed

#### ---- Explore tuning output (both rules) ----
message("Top configs by CV fold-mean (test_rmse_mean @ CV-best):")
tuning_results %>% dplyr::arrange(best_rmse_by_cv) %>% head(10) %>% as.data.frame() %>% print(row.names = FALSE)
# > grid_id                          param_name nrounds max_depth  eta best_iter_by_cv best_rmse_by_cv test_rmse_std_at_cvbest train_rmse_mean_at_cvbest train_rmse_std_at_cvbest best_iter_by_oof best_rmse_by_oof rmse_oof_at_cvbest_iter rmse_oof_at_final_iter
# >      19 nrounds=100, max_depth=4, eta=0.100     100         4 0.10              50        89.44827                1.217818                  87.27138                0.3012648               50         89.47617                89.47617               89.77462
# >      20 nrounds=200, max_depth=4, eta=0.100     200         4 0.10              50        89.44827                1.217818                  87.27138                0.3012648               50         89.47617                89.47617               90.35422
# >      21 nrounds=300, max_depth=4, eta=0.100     300         4 0.10              50        89.44827                1.217818                  87.27138                0.3012648               50         89.47617                89.47617               90.79848
# >      11 nrounds=200, max_depth=4, eta=0.050     200         4 0.05             102        89.45365                1.220706                  87.29514                0.2836398              102         89.48140                89.48140               89.68866
# >      12 nrounds=300, max_depth=4, eta=0.050     300         4 0.05             102        89.45365                1.220706                  87.29514                0.2836398              102         89.48140                89.48140               89.91420
# >      10 nrounds=100, max_depth=4, eta=0.050     100         4 0.05              99        89.45510                1.220395                  87.34661                0.2816726               99         89.48281                89.48281               89.48382
# >      28 nrounds=100, max_depth=4, eta=0.300     100         4 0.30              15        89.51472                1.288254                  87.37651                0.2707659               15         89.54464                89.54464               90.87895
# >      29 nrounds=200, max_depth=4, eta=0.300     200         4 0.30              15        89.51472                1.288254                  87.37651                0.2707659               15         89.54464                89.54464               91.96992
# >      30 nrounds=300, max_depth=4, eta=0.300     300         4 0.30              15        89.51472                1.288254                  87.37651                0.2707659               15         89.54464                89.54464               92.84912
# >      22 nrounds=100, max_depth=6, eta=0.100     100         6 0.10              44        90.30762                1.440544                  84.13944                0.3316562               44         90.34120                90.34120               91.66760

message("Top configs by pooled OOF RMSE (min across iterations):")
# > tuning_results %>% dplyr::arrange(best_rmse_by_oof) %>% head(10) %>% as.data.frame() %>% print(row.names = FALSE)
# > grid_id                          param_name nrounds max_depth  eta best_iter_by_cv best_rmse_by_cv test_rmse_std_at_cvbest train_rmse_mean_at_cvbest train_rmse_std_at_cvbest best_iter_by_oof best_rmse_by_oof rmse_oof_at_cvbest_iter rmse_oof_at_final_iter
# >      19 nrounds=100, max_depth=4, eta=0.100     100         4 0.10              50        89.44827                1.217818                  87.27138                0.3012648               50         89.47617                89.47617               89.77462
# >      20 nrounds=200, max_depth=4, eta=0.100     200         4 0.10              50        89.44827                1.217818                  87.27138                0.3012648               50         89.47617                89.47617               90.35422
# >      21 nrounds=300, max_depth=4, eta=0.100     300         4 0.10              50        89.44827                1.217818                  87.27138                0.3012648               50         89.47617                89.47617               90.79848
# >      11 nrounds=200, max_depth=4, eta=0.050     200         4 0.05             102        89.45365                1.220706                  87.29514                0.2836398              102         89.48140                89.48140               89.68866
# >      12 nrounds=300, max_depth=4, eta=0.050     300         4 0.05             102        89.45365                1.220706                  87.29514                0.2836398              102         89.48140                89.48140               89.91420
# >      10 nrounds=100, max_depth=4, eta=0.050     100         4 0.05              99        89.45510                1.220395                  87.34661                0.2816726               99         89.48281                89.48281               89.48382
# >      28 nrounds=100, max_depth=4, eta=0.300     100         4 0.30              15        89.51472                1.288254                  87.37651                0.2707659               15         89.54464                89.54464               90.87895
# >      29 nrounds=200, max_depth=4, eta=0.300     200         4 0.30              15        89.51472                1.288254                  87.37651                0.2707659               15         89.54464                89.54464               91.96992
# >      30 nrounds=300, max_depth=4, eta=0.300     300         4 0.30              15        89.51472                1.288254                  87.37651                0.2707659               15         89.54464                89.54464               92.84912
2# >      2 nrounds=100, max_depth=6, eta=0.100     100         6 0.10              44        90.30762                1.440544                  84.13944                0.3316562               44         90.34120                90.34120               91.66760

stopifnot(!is.null(best_cv_info), !is.null(best_oof_info))

message(sprintf(
  "CV winner -> grid_id=%d | cap=%d | max_depth=%d | eta=%.3f | CV-best iter=%d | OOF-best iter=%d | CV-min=%.5f | OOF-min=%.5f",
  best_cv_info$grid_id, best_cv_info$row$nrounds, best_cv_info$row$max_depth, best_cv_info$row$eta,
  best_cv_info$best_iter_by_cv, best_cv_info$best_iter_by_oof,
  best_rmse_by_cv, min(best_cv_info$oof_curve)
))
# > CV winner -> grid_id=19 | cap=100 | max_depth=4 | eta=0.100 | CV-best iter=50 | OOF-best iter=50 | CV-min=89.44827 | OOF-min=89.47617
message(sprintf(
  "OOF winner -> grid_id=%d | cap=%d | max_depth=%d | eta=%.3f | CV-best iter=%d | OOF-best iter=%d | OOF-min=%.5f | CV-min@that=%.5f",
  best_oof_info$grid_id, best_oof_info$row$nrounds, best_oof_info$row$max_depth, best_oof_info$row$eta,
  best_oof_info$best_iter_by_cv, best_oof_info$best_iter_by_oof,
  best_rmse_by_oof, min(best_oof_info$cv_eval$test_rmse_mean)
))
# > OOF winner -> grid_id=19 | cap=100 | max_depth=4 | eta=0.100 | CV-best iter=50 | OOF-best iter=50 | OOF-min=89.47617 | CV-min@that=89.44827

### ---- Refit with OOF winner config ----
# Sanity: confirm CV-winner and OOF-winner are the SAME config (as assumed here).
stopifnot(best_cv_info$grid_id == best_oof_info$grid_id)

# Use the OOF-winner config to refit once to its n_cap
set.seed(123)
main_model <- xgb.train(
  params  = best_oof_info$params,
  data    = dtrain,
  nrounds = best_oof_info$row$nrounds,
  watchlist = list(train = dtrain, valid = dvalid),
  verbose = 1,
  early_stopping_rounds = NULL
)

# VALID-selected cutoff from refit (v1.4 naming)
best_iter <- which.min(main_model$evaluation_log$valid_rmse)  # 86
best_rmse <- min(main_model$evaluation_log$valid_rmse)        # 85.97932

#### ---- Explore refit, importance, and quick tree ----
main_model
print(as.data.frame(main_model$evaluation_log), row.names = FALSE)

xgb.importance(model = main_model)  
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.4045125 0.1980696 0.1860000
# > 2: STUDYHMW 0.2390889 0.2680625 0.3046667
# > 3: EXERPRAC 0.2171170 0.2911851 0.2646667
# > 4: WORKHOME 0.1392815 0.2426828 0.2446667
xgb.importance(model = main_model, trees = 0:(best_oof_info$best_iter_by_oof - 1))
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.4386612 0.2268731 0.2160000
# > 2: STUDYHMW 0.2287507 0.2669174 0.3013333
# > 3: EXERPRAC 0.2118834 0.3125198 0.2653333
# > 4: WORKHOME 0.1207047 0.1936897 0.2173333

xgb.plot.importance(
  importance_matrix = xgb.importance(model = main_model),
  top_n = NULL, measure = "Gain", rel_to_first = TRUE,
  xlab = "Relative Importance"
)

# Learning curves on TRAIN/VALID during refit
main_model$evaluation_log |>
  tidyr::pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot2::ggplot(ggplot2::aes(x = iter, y = RMSE, color = Dataset)) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::labs(
    title = "XGBoost RMSE over Boosting Rounds (refit to n_cap)",
    x = "Boosting Round", y = "RMSE", color = "Dataset"
  ) +
  ggplot2::theme_minimal()

# Optional tree snapshots
xgb.plot.tree(model = main_model, trees = 0)
xgb.plot.tree(model = main_model, trees = max(0, best_cv_info$best_iter_by_cv - 1))

### ---- Predict and evaluate performance on TRAIN/VALID/TEST ----
# CV-selected cutoff
pred_cv_train <- predict(main_model, dtrain, iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))
pred_cv_valid <- predict(main_model, dvalid, iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))
pred_cv_test  <- predict(main_model, dtest,  iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))

# OOF-selected cutoff
pred_oof_train <- predict(main_model, dtrain, iterationrange = c(1, best_oof_info$best_iter_by_oof + 1))
pred_oof_valid <- predict(main_model, dvalid, iterationrange = c(1, best_oof_info$best_iter_by_oof + 1))
pred_oof_test  <- predict(main_model, dtest,  iterationrange = c(1, best_oof_info$best_iter_by_oof + 1))

# VALID-selected cutoff (from refit)
pred_train <- predict(main_model, dtrain, iterationrange = c(1, best_iter + 1))
pred_valid <- predict(main_model, dvalid, iterationrange = c(1, best_iter + 1))
pred_test  <- predict(main_model, dtest,  iterationrange = c(1, best_iter + 1))

# Consistency check: VALID RMSE equals watchlist at best_iter
stopifnot(isTRUE(all.equal(
  unname(compute_metrics(y_valid, pred_valid, w_valid)["rmse"]),
  main_model$evaluation_log$valid_rmse[best_iter]
)))

# Metrics: CV-best
metrics_cv <- tibble::tibble(
  Model   = "CV-best",
  Dataset = c("Training","Validation","Test"),
  RMSE    = c(compute_metrics(y_train, pred_cv_train,  w_train)["rmse"],
              compute_metrics(y_valid, pred_cv_valid,  w_valid)["rmse"],
              compute_metrics(y_test,  pred_cv_test,   w_test )["rmse"]),
  MAE     = c(compute_metrics(y_train, pred_cv_train,  w_train)["mae"],
              compute_metrics(y_valid, pred_cv_valid,  w_valid)["mae"],
              compute_metrics(y_test,  pred_cv_test,   w_test )["mae"]),
  Bias    = c(compute_metrics(y_train, pred_cv_train,  w_train)["bias"],
              compute_metrics(y_valid, pred_cv_valid,  w_valid)["bias"],
              compute_metrics(y_test,  pred_cv_test,   w_test )["bias"]),
  `Bias%` = c(compute_metrics(y_train, pred_cv_train,  w_train)["bias_pct"],
              compute_metrics(y_valid, pred_cv_valid,  w_valid)["bias_pct"],
              compute_metrics(y_test,  pred_cv_test,   w_test )["bias_pct"]),
  R2      = c(compute_metrics(y_train, pred_cv_train,  w_train)["r2"],
              compute_metrics(y_valid, pred_cv_valid,  w_valid)["r2"],
              compute_metrics(y_test,  pred_cv_test,   w_test )["r2"])
)

# Metrics: OOF-best
metrics_oof <- tibble::tibble(
  Model   = "OOF-best",
  Dataset = c("Training","Validation","Test"),
  RMSE    = c(compute_metrics(y_train, pred_oof_train, w_train)["rmse"],
              compute_metrics(y_valid, pred_oof_valid, w_valid)["rmse"],
              compute_metrics(y_test,  pred_oof_test,  w_test )["rmse"]),
  MAE     = c(compute_metrics(y_train, pred_oof_train, w_train)["mae"],
              compute_metrics(y_valid, pred_oof_valid, w_valid)["mae"],
              compute_metrics(y_test,  pred_oof_test,  w_test )["mae"]),
  Bias    = c(compute_metrics(y_train, pred_oof_train, w_train)["bias"],
              compute_metrics(y_valid, pred_oof_valid,  w_valid)["bias"],
              compute_metrics(y_test,  pred_oof_test,   w_test )["bias"]),
  `Bias%` = c(compute_metrics(y_train, pred_oof_train, w_train)["bias_pct"],
              compute_metrics(y_valid, pred_oof_valid,  w_valid)["bias_pct"],
              compute_metrics(y_test,  pred_oof_test,   w_test )["bias_pct"]),
  R2      = c(compute_metrics(y_train, pred_oof_train,  w_train)["r2"],
              compute_metrics(y_valid, pred_oof_valid,  w_valid)["r2"],
              compute_metrics(y_test,  pred_oof_test,   w_test )["r2"])
)

# Metrics: VALID-best (from refit)
metrics <- tibble::tibble(
  Model   = "VALID-best",
  Dataset = c("Training","Validation","Test"),
  RMSE    = c(compute_metrics(y_train, pred_train, w_train)["rmse"],
              compute_metrics(y_valid, pred_valid, w_valid)["rmse"],
              compute_metrics(y_test,  pred_test,  w_test )["rmse"]),
  MAE     = c(compute_metrics(y_train, pred_train, w_train)["mae"],
              compute_metrics(y_valid, pred_valid,  w_valid)["mae"],
              compute_metrics(y_test,  pred_test,   w_test )["mae"]),
  Bias    = c(compute_metrics(y_train, pred_train, w_train)["bias"],
              compute_metrics(y_valid, pred_valid,  w_valid)["bias"],
              compute_metrics(y_test,  pred_test,   w_test )["bias"]),
  `Bias%` = c(compute_metrics(y_train, pred_train,  w_train)["bias_pct"],
              compute_metrics(y_valid, pred_valid,   w_valid)["bias_pct"],
              compute_metrics(y_test,  pred_test,    w_test )["bias_pct"]),
  R2      = c(compute_metrics(y_train, pred_train,  w_train)["r2"],
              compute_metrics(y_valid, pred_valid,   w_valid)["r2"],
              compute_metrics(y_test,  pred_test,    w_test )["r2"])
)

# Side-by-side comparison 
print(as.data.frame(dplyr::bind_rows(metrics_cv, metrics_oof, metrics)), row.names = FALSE)
# >      Model    Dataset     RMSE      MAE       Bias    Bias%         R2
# >    CV-best   Training 87.58393 70.23074 -2.5873895 2.846265 0.12456984
# >    CV-best Validation 86.12982 68.85730 -6.0443116 1.960081 0.09264719
# >    CV-best       Test 89.51025 71.48393 -4.2692352 2.597415 0.10513692      # <-
# >   OOF-best   Training 87.58393 70.23074 -2.5873895 2.846265 0.12456984      
# >   OOF-best Validation 86.12982 68.85730 -6.0443116 1.960081 0.09264719
# >   OOF-best       Test 89.51025 71.48393 -4.2692352 2.597415 0.10513692      # <-
# > VALID-best   Training 86.90028 69.63192 -0.0583894 3.322464 0.13818318
# > VALID-best Validation 85.97932 68.61388 -3.6624783 2.427803 0.09581539
# > VALID-best       Test 89.53938 71.54735 -1.7380570 3.101568 0.10455429      # <-

# Summary 
message(sprintf(
  "Summary | CV-best iter = %d | OOF-best iter = %d | VALID-best iter = %d | n_cap (refit) = %d",
  best_oof_info$best_iter_by_cv, best_oof_info$best_iter_by_oof, best_iter, best_oof_info$row$nrounds
))
# > Summary | CV-best iter = 50 | OOF-best iter = 50 | VALID-best iter = 86 | n_cap (refit) = 100
message(sprintf(
  "OOF RMSE @ CV-best = %.5f | @ OOF-best = %.5f | @ VALID-best = %.5f",
  best_oof_info$oof_curve[best_oof_info$best_iter_by_cv],
  best_oof_info$oof_curve[best_oof_info$best_iter_by_oof],
  best_oof_info$oof_curve[best_iter]
))
# > OOF RMSE @ CV-best = 89.47617 | @ OOF-best = 89.47617 | @ VALID-best = 89.68640
