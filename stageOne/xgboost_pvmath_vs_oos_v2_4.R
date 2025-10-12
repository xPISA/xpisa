# ---- II. Predictive Modelling: Version 2.4 ----

# xgb.cv with manual K-folds + OOF reconstruction (tuning hyperparameters)

## ---- 1. Setup ----

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

# To compare performance without listwise deletion 
# temp_data <- pisa_2022_student_canada %>%
#   select(CNTSCHID, CNTSTUID,                  # IDs
#          all_of(final_wt), all_of(rep_wts),   # Weights
#          all_of(pvmaths), all_of(oos))        # PVs + predictors

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

## ---- 2. PV1MATH only ----

# --- Remark ---
# 1) Repeat the same process for PV2MATH - PV10MATH.
# 2) Apply best results from PV1MATH to all plausible values in mathematics. 

### ---- Fit main model using final student weights (W_FSTUWT) on the training data----

#### ---- Tune XGBoost model for PV1MATH only: xgb.cv + Manual Folds + OOF----

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

# Check CV-fold weight balance
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

# Held-out DMatrices per fold (reused for OOF predictions)
dvalid_fold <- lapply(seq_len(num_folds), function(k) {
  idx <- cv_folds[[k]]
  xgb.DMatrix(as.matrix(X_train[idx, , drop = FALSE]),
              label = y_train[idx], weight = w_train[idx])
})

# Ensure callback exists; then call directly inside xgb.cv()
if (is.null(tryCatch(getFromNamespace("cb.cv.predict", "xgboost"), error = function(e) NULL))) {
  stop("xgboost build lacks cb.cv.predict(save_models=TRUE). Update/repair your xgboost installation.")
}

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

cv_eval_list   <- vector("list", nrow(grid))   # per-config evaluation_log (means & sds)
oof_curve_list <- vector("list", nrow(grid))   # per-config OOF RMSE curve (per iteration)

# Track BOTH winners across the grid (renamed per request; no "_overall")
best_rmse_by_cv  <- Inf
best_rmse_by_oof <- Inf
best_cv_info  <- NULL   # list(grid_id, params, row, best_iter_by_cv, best_iter_by_oof, cv_eval, oof_curve)
best_oof_info <- NULL


# Grid search 
set.seed(123)
tictoc::tic("Tuning (xgb.cv + Manual OOF)")
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
  
  # 1) Run xgb.cv to get CV curves (means ± sd) and OOF snapshot @ final iter + fold models
  cv_mod <- xgb.cv(
    params  = params,
    data    = dtrain,
    nrounds = row$nrounds,
    folds   = cv_folds,
    showsd  = TRUE,
    verbose = TRUE,   # requested logical TRUE
    early_stopping_rounds = NULL,
    prediction = TRUE,
    stratified = FALSE,
    callbacks = list(getFromNamespace("cb.cv.predict", "xgboost")(save_models = TRUE))
  )
  
  # CV-best iteration by validation RMSE (fold-mean) — avoid temp 'ev'
  best_iter_by_cv_i <- which.min(cv_mod$evaluation_log$test_rmse_mean)
  best_rmse_by_cv_i <- cv_mod$evaluation_log$test_rmse_mean[best_iter_by_cv_i]
  
  # 2) OOF curve reconstruction using the saved fold boosters 
  oof_pred_matrix_i <- matrix(NA_real_, nrow = n_cv, ncol = row$nrounds)
  rmse_oof_by_iter_i <- numeric(row$nrounds)
  for (iter in seq_len(row$nrounds)) {
    for (k in seq_len(num_folds)) {
      idx <- cv_folds[[k]]
      oof_pred_matrix_i[idx, iter] <- predict(
        cv_mod$models[[k]], dvalid_fold[[k]],
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
  rmse_oof_at_final_iter_i  <- sqrt(sum(w_train * (y_train - cv_mod$pred)^2) / sum(w_train))
  stopifnot(isTRUE(all.equal(oof_pred_matrix_i[, row$nrounds], cv_mod$pred)))
  
  # Save per-config artifacts
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
# > Tuning (xgb.cv + Manual OOF): 138.471 sec elapsed

##### ---- Explore tuning output (both rules) ----
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
tuning_results %>% dplyr::arrange(best_rmse_by_oof) %>% head(10) %>% as.data.frame() %>% print(row.names = FALSE)
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

stopifnot(!is.null(best_cv_info), !is.null(best_oof_info))

message(sprintf("CV winner -> grid_id=%d | cap=%d | max_depth=%d | eta=%.3f | CV-best iter=%d | OOF-best iter=%d | CV-min=%.5f | OOF-min=%.5f",
                best_cv_info$grid_id, best_cv_info$row$nrounds, best_cv_info$row$max_depth, best_cv_info$row$eta,
                best_cv_info$best_iter_by_cv, best_cv_info$best_iter_by_oof,
                best_rmse_by_cv, min(best_cv_info$oof_curve)))
# > CV winner -> grid_id=19 | cap=100 | max_depth=4 | eta=0.100 | CV-best iter=50 | OOF-best iter=50 | CV-min=89.44827 | OOF-min=89.47617

message(sprintf("OOF winner -> grid_id=%d | cap=%d | max_depth=%d | eta=%.3f | CV-best iter=%d | OOF-best iter=%d | OOF-min=%.5f | CV-min@that=%.5f",
                best_oof_info$grid_id, best_oof_info$row$nrounds, best_oof_info$row$max_depth, best_oof_info$row$eta,
                best_oof_info$best_iter_by_cv, best_oof_info$best_iter_by_oof,
                best_rmse_by_oof, min(best_oof_info$cv_eval$test_rmse_mean)))
# > OOF winner -> grid_id=19 | cap=100 | max_depth=4 | eta=0.100 | CV-best iter=50 | OOF-best iter=50 | OOF-min=89.47617 | CV-min@that=89.44827

#### ---- Refit with OOF winner config ----

# Sanity: confirm CV-winner and OOF-winner are the SAME config (as assumed here).
stopifnot(best_cv_info$grid_id == best_oof_info$grid_id)

# Use the OOF-winner config to refit 
set.seed(123)
main_model <- xgb.train(
  params = list(                                            # <=> params = best_oof_info$params
    objective = "reg:squarederror",                         # Specify the learning task and the corresponding learning objective: reg:squarederror Regression with squared loss (Default)
    max_depth = best_oof_info$params$max_depth,                          
    eta = best_oof_info$params$eta,                                      
    eval_metric = "rmse",                                   # Default: metric will be assigned according to objective(rmse for regression) 
    nthread = max(1, parallel::detectCores() - 1)           # Manually specify number of thread
    #seed = 123                                             # Random number seed for reproducibility (e.g. tune subsample, colsample_bytree for regularization)
  ),
  data    = dtrain,
  nrounds = best_oof_info$row$nrounds,
  watchlist = list(train = dtrain, valid = dvalid),
  verbose = 1,
  early_stopping_rounds = NULL
)

# VALID-selected cutoff from refit 
best_iter <- which.min(main_model$evaluation_log$valid_rmse)  # 86
best_rmse <- min(main_model$evaluation_log$valid_rmse)        # 85.97932

##### ---- Explore refit, importance, and quick tree ----
main_model
print(as.data.frame(main_model$evaluation_log), row.names = FALSE)
# > iter train_rmse valid_rmse
# >    1  461.11274  462.34838
# >    2  416.82702  418.17171
# >  ...
# >   99   86.71605   86.02139
# >  100   86.70045   86.04323

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
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round", y = "RMSE", color = "Dataset"
  ) +
  ggplot2::theme_minimal()

# Tree snapshots
xgb.plot.tree(model = main_model, trees = 0)
xgb.plot.tree(model = main_model, trees = max(0, best_cv_info$best_iter_by_cv - 1))


#### ---- Predict and evaluate performance on TRAIN/VALID/TEST ----

# CV-selected cutoff
pred_cv_train <- predict(main_model, dtrain, iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))
pred_cv_valid <- predict(main_model, dvalid, iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))
pred_cv_test  <- predict(main_model, dtest,  iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))

# OOF-selected cutoff
pred_oof_train <- predict(main_model, dtrain, iterationrange = c(1, best_oof_info$best_iter_by_oof+ 1))
pred_oof_valid <- predict(main_model, dvalid, iterationrange = c(1, best_oof_info$best_iter_by_oof+ 1))
pred_oof_test  <- predict(main_model, dtest,  iterationrange = c(1, best_oof_info$best_iter_by_oof+ 1))

# VALID-selected cutoff (from refit)
pred_train <- predict(main_model, dtrain, iterationrange = c(1, best_iter + 1))
pred_valid <- predict(main_model, dvalid, iterationrange = c(1, best_iter + 1))
pred_test  <- predict(main_model, dtest,  iterationrange = c(1, best_iter + 1))

# Consistency check: VALID RMSE equals watchlist at best_iter
stopifnot(isTRUE(all.equal(
  unname(compute_metrics(y_valid, pred_valid, w_valid)["rmse"]),
  main_model$evaluation_log$valid_rmse[best_iter]
)))

# Metrics 
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

metrics <- tibble::tibble(  # VALID-best (v1.4 naming)
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
  best_oof_info$oof_curve[best_oof_info$best_iter_by_cv], best_oof_info$oof_curve[best_oof_info$best_iter_by_oof], best_oof_info$oof_curve[best_iter]
))
# > OOF RMSE @ CV-best = 89.47617 | @ OOF-best = 89.47617 | @ VALID-best = 89.68640

## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH to all plausible values in mathematics. 

### ---- Fit main models using final weight (W_FSTUWT) ----

set.seed(123)

# Fit one XGBoost model per plausible value
tic("Fitting main models")
main_models <- lapply(pvmaths, function(pv) {
  
  X_train <- train_data[, oos]
  y_train <- train_data[[pv]]
  weight_train <- train_data[[final_wt]]
  
  X_valid <- valid_data[, oos]
  y_valid <- valid_data[[pv]]
  weight_valid <- valid_data[[final_wt]]
  
  dtrain <- xgb.DMatrix(
    data = as.matrix(X_train),
    label = y_train,
    weight = weight_train
  )
  
  dvalid <- xgb.DMatrix(
    data = as.matrix(X_valid),
    label = y_valid,
    weight = weight_valid
  )
  
  mod <- xgb.train(
    params = list(
      objective = "reg:squarederror",                  
      max_depth = best_oof_info$params$max_depth,               # best_oof_info$params$max_depth = 4
      eta = best_oof_info$params$eta,                           # best_oof_info$params$eta = 0.1
      eval_metric = "rmse",                            
      nthread = max(1, parallel::detectCores() - 1) 
    ),
    data = dtrain,
    nrounds = best_oof_info$best_iter_by_oof,                   # best_oof_info$best_iter_by_oof = 50
    watchlist = list(train = dtrain, eval = dvalid),
    verbose = 1,                               
    early_stopping_rounds=NULL               
  )
  
  list(
    mod = mod,
    formula = as.formula(paste(pv, "~", paste(oos, collapse = " + "))),
    importance = xgb.importance(model = mod)
  )
})
toc()
# > Fitting main models: 2.046 sec elapsed

main_models[[1]]             # Inspect first model: mod object, r2, and importance
main_models[[1]]$formula
main_models[[1]]$importance
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.4386612 0.2268731 0.2160000
# > 2: STUDYHMW 0.2287507 0.2669174 0.3013333
# > 3: EXERPRAC 0.2118834 0.3125198 0.2653333
# > 4: WORKHOME 0.1207047 0.1936897 0.2173333
main_models[[1]]$importance$Feature
main_models[[2]]$importance$Feature

main_models[[1]]$mod$evaluation_log
which.min(main_models[[1]]$mod$evaluation_log$eval_rmse)
# > [1] 50
min(main_models[[1]]$mod$evaluation_log$eval_rmse)
# > 86.12982
which.min(main_models[[2]]$mod$evaluation_log$eval_rmse)
# > [1] 50
min(main_models[[2]]$mod$evaluation_log$eval_rmse)
# > 86.23621
which.min(main_models[[3]]$mod$evaluation_log$eval_rmse)
# > [1] 49
min(main_models[[3]]$mod$evaluation_log$eval_rmse)
# > 88.37616
# ......

main_models[[1]]$mod$evaluation_log |>
  pivot_longer(cols = c(train_rmse, eval_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
  labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal()

tibble::tibble(
  pv = pvmaths,
  best_nround = sapply(main_models, function(m) {
    which.min(m$mod$evaluation_log$eval_rmse)
  }),
  best_eval_rmse = sapply(main_models, function(m) {
    min(m$mod$evaluation_log$eval_rmse)
  })
)

# --- Variable Importance ---

# Gain-based Variable Importance Matrix (10 PVs × p predictors) 
main_importance_matrix <- do.call(rbind, lapply(main_models, function(m) {
  # Extract Gain column from xgb.importance()
  main_importance <- m$importance
  setNames(main_importance$Gain, main_importance$Feature)[oos]  # Ensure correct column order and all variables present
}))
dim(main_importance_matrix)  # 10x4
main_importance_matrix
# >        EXERPRAC  STUDYHMW   WORKPAY  WORKHOME
# >  [1,] 0.2118834 0.2287507 0.4386612 0.1207047
# >  [2,] 0.2373384 0.2236851 0.4218412 0.1171352
# >  [3,] 0.2244615 0.2200791 0.4200739 0.1353855
# >  [4,] 0.2451418 0.1982921 0.4338216 0.1227446
# >  [5,] 0.2367563 0.2121871 0.4277520 0.1233046
# >  [6,] 0.2197643 0.2242791 0.4238547 0.1321019
# >  [7,] 0.2406219 0.2007926 0.4377968 0.1207887
# >  [8,] 0.2164281 0.2147762 0.4453651 0.1234305
# >  [9,] 0.2265361 0.2009755 0.4519473 0.1205411
# > [10,] 0.2200812 0.2189153 0.4489004 0.1121030

# Mean of gain importance across PVs
main_importance <- colMeans(main_importance_matrix)
main_importance
# >  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME 
# > 0.2279013 0.2142733 0.4350014 0.1228240 

# Display ranked importance
tibble(
  Variable = names(main_importance),
  Importance = main_importance
) |> arrange(desc(Importance))
# > # A tibble: 4 × 2
# > Variable Importance
# > <chr>         <dbl>
# > 1 WORKPAY       0.435
# > 2 EXERPRAC      0.228
# > 3 STUDYHMW      0.214
# > WORKHOME      0.123

# --- Estimates of Manual weighted R² (XGBoost models on training data) ---

main_r2s_weighted <- sapply(1:M, function(i) {
  pv <- pvmaths[i]
  model <- main_models[[i]]$mod
  
  # Extract weighted true values and predictions on training data
  y_true <- train_data[[pv]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, oos]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  
  y_pred <- predict(model, dtrain)  # prediction from fitted xgb model
  
  # Weighted mean and sums
  y_bar <- sum(w * y_true) / sum(w)
  sse <- sum(w * (y_true - y_pred)^2)
  sst <- sum(w * (y_true - y_bar)^2)
  
  # Weighted R²
  r2 <- 1 - sse / sst
  return(r2)
})

main_r2s_weighted
# > [1] 0.1245698 0.1249450 0.1254169 0.1297159 0.1262895 0.1281054 0.1254628 0.1261363 0.1242307 0.1299650

# Final Rubin's Step 2: mean of R² across plausible values
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.1264837

# --- Use helper function for all five metrics ---
main_metrics <- sapply(1:M, function(i) {
  pv <- pvmaths[i]
  model <- main_models[[i]]$mod
  
  y_true <- train_data[[pv]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, oos]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  y_pred <- predict(model, dtrain)
  
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
main_metrics
# >         mse     rmse      mae      bias bias_pct        r2
# > 1  7670.945 87.58393 70.23074 -2.587390 2.846265 0.1245698
# > 2  7574.569 87.03200 69.70512 -2.590101 2.792019 0.1249450
# > 3  7787.733 88.24813 70.58085 -2.585766 2.929485 0.1254169
# > 4  7657.372 87.50641 70.00630 -2.589091 2.832513 0.1297159
# > 5  7715.576 87.83835 70.24901 -2.587316 2.860673 0.1262895
# > 6  7684.539 87.66151 70.06856 -2.581862 2.881085 0.1281054
# > 7  7603.805 87.19980 69.88523 -2.587670 2.837522 0.1254628
# > 8  7684.407 87.66075 70.23024 -2.584258 2.865028 0.1261363
# > 9  7764.996 88.11922 70.49299 -2.585178 2.914878 0.1242307
# > 10 7506.756 86.64154 69.66155 -2.588451 2.767635 0.1299650

### ---- Replicate models using BRR replicate weights (XGBoost, fixed hyperparameters) ----

set.seed(123)

tic("Fitting replicate models")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    X_train <- train_data[, oos]
    y_train <- train_data[[pv]]
    weight_train <- train_data[[w]]
    
    X_valid <- valid_data[, oos]
    y_valid <- valid_data[[pv]]
    weight_valid <- valid_data[[w]]
    
    dtrain <- xgb.DMatrix(
      data = as.matrix(X_train),
      label = y_train,
      weight = weight_train
    )
    
    dvalid <- xgb.DMatrix(
      data = as.matrix(X_valid),
      label = y_valid,
      weight = weight_valid
    )
    
    mod <- xgb.train(
      params = list(
        objective = "reg:squarederror",
        max_depth = best_oof_info$params$max_depth,             # best_oof_info$params$max_depth = 4
        eta = best_oof_info$params$eta,                         # best_oof_info$params$eta = 0.1
        eval_metric = "rmse",
        nthread = max(1, parallel::detectCores() - 1)
      ),
      data = dtrain,
      nrounds = best_oof_info$best_iter_by_oof,                 # best_oof_info$best_iter_by_oof = 50   
      watchlist = list(train = dtrain, eval = dvalid),
      verbose = 1,
      early_stopping_rounds = NULL
    )
    
    list(
      mod = mod,
      formula = as.formula(paste(pv, "~", paste(oos, collapse = " + "))),
      importance = xgb.importance(model = mod)
    )
  })
})
toc()
# > Fitting replicate models: 129.717 sec elapsed

replicate_models[[1]][[1]]$mod        
replicate_models[[1]][[1]]$formula    
replicate_models[[1]][[1]]$importance 

# --- Gain-Based Variable Importance Matrix for Replicates: (M × G × p) ---
rep_importance_array <- array(NA, dim = c(M, G, length(oos)),
                              dimnames = list(NULL, NULL, oos))

for (m in 1:M) {
  for (g in 1:G) {
    imp <- replicate_models[[m]][[g]]$importance
    rep_importance_array[m, g, ] <- setNames(imp$Gain, imp$Feature)[oos]
  }
}

# Check structure
dim(rep_importance_array)  # 10 x 80 x 4
rep_importance_array[1, 1, ]  # e.g., PV1, BRR replicate 1
# >  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME 
# > 0.2046227 0.2392140 0.4170130 0.1391503 

# --- Weighted R² across (M × G) ---
rep_r2_weighted  <- matrix(NA, nrow = G, ncol = M)

for (m in 1:M) {
  pv <- pvmaths[m]
  y_true <- train_data[[pv]]
  
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w <- train_data[[rep_wts[g]]]  # Replicate weight g
    
    X_train <- train_data[, oos]
    dtrain <- xgb.DMatrix(data = as.matrix(X_train))
    
    y_pred <- predict(model, dtrain)
    
    y_bar <- sum(w * y_true) / sum(w)
    sse <- sum(w * (y_true - y_pred)^2)
    sst <- sum(w * (y_true - y_bar)^2)
    
    rep_r2_weighted [g, m] <- 1 - sse / sst
  }
}

# Check output
dim(rep_r2_weighted )       # 80 x 10
rep_r2_weighted 

### ---- Rubin + BRR for Standard Errors (SEs) ----

# --- Rubin + BRR: Gain-Based Variable Importance ---
sampling_var_importance   <- setNames(numeric(length(oos)), oos)
imputation_var_importance <- setNames(numeric(length(oos)), oos)
var_final_importance      <- setNames(numeric(length(oos)), oos)
se_final_importance       <- setNames(numeric(length(oos)), oos)
cv_final_importance       <- setNames(numeric(length(oos)), oos)

for (var in oos) {
  rep_vals <- rep_importance_array[, , var]  # dim: M × G
  rep_vals <- t(rep_vals)                          # dim: G × M → M × G
  
  sampling_var_importance[var] <- mean(sapply(1:M, function(m) {
    sum((rep_vals[, m] - main_importance_matrix[m, var])^2) / (G * (1 - k)^2)
  }))
  
  imputation_var_importance[var] <- sum((main_importance_matrix[, var] - main_importance[var])^2) / (M - 1)
  
  var_final_importance[var] <- sampling_var_importance[var] + (1 + 1 / M) * imputation_var_importance[var]
  se_final_importance[var] <- sqrt(var_final_importance[var])
  cv_final_importance[var] <- se_final_importance[var] / main_importance[var]
}

# --- Rubin + BRR: Weighted R² ---
sampling_var_r2s_weighted <- sapply(1:M, function(m) {
  sum((rep_r2_weighted [, m] - main_r2s_weighted[m])^2) / (G * (1 - k)^2)
})

sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
sampling_var_r2_weighted
# > [1] 3.138125e-06

imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
imputation_var_r2_weighted
# > [1] 4.293523e-06

var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
var_final_r2_weighted
# > [1] 7.861001e-06

se_final_r2_weighted <- sqrt(var_final_r2_weighted)
se_final_r2_weighted
# > [1] 0.002803748

#### ---- Final Output Tables ----

# --- Variable Importance Table (mean ± SE) ---
importance_table <- tibble(
  Variable = oos,  # e.g., "EXERPRAC", ...
  Importance = main_importance[oos],
  `Std. Error` = se_final_importance[oos],
  `CV` = cv_final_importance[oos]
) |> 
  arrange(desc(Importance))

# --- R-squared Output Table (Weighted only) ---
r2_weighted_table <- tibble(
  Metric = "R-squared (Weighted)",
  Estimate = main_r2_weighted,
  `Std. Error` = se_final_r2_weighted
)

# --- Print Output Tables ---
print(as.data.frame(r2_weighted_table), row.names = FALSE)
# >               Metric  Estimate Std. Error
# > R-squared (Weighted) 0.1264837 0.002803748
print(as.data.frame(importance_table), row.names = FALSE)
# > Variable Importance  Std. Error         CV
# >  WORKPAY  0.4350014 0.013148305 0.03022589
# > EXERPRAC  0.2279013 0.012381872 0.05432998
# > STUDYHMW  0.2142733 0.011922638 0.05564220
# > WORKHOME  0.1228240 0.007999668 0.06513116

# --- Plot: Variable Importance with SE ---
ggplot(importance_table, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(aes(ymin = Importance - `Std. Error`, ymax = Importance + `Std. Error`), width = 0.2) +
  coord_flip() +
  labs(title = "XGBoost Variable Importance (Gain) with Standard Errors",
       y = "Importance (Mean ± SE)", x = "Variable") +
  theme_minimal()

# --- Plot: Variable Importance with CV Gradient ---
ggplot(importance_table, aes(x = reorder(Variable, Importance), y = Importance, fill = CV)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  labs(
    title = "XGBoost Variable Importance (Gain) with Coefficient of Variation (CV)",
    x = "Variable",
    y = "Mean Importance",
    fill = "CV"
  ) +
  theme_minimal()


### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_true <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, oos]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  y_pred <- predict(model, dtrain)
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()

train_metrics_main
# >         mse     rmse      mae      bias bias_pct        r2
# > 1  7670.945 87.58393 70.23074 -2.587390 2.846265 0.1245698
# > 2  7574.569 87.03200 69.70512 -2.590101 2.792019 0.1249450
# > 3  7787.733 88.24813 70.58085 -2.585766 2.929485 0.1254169
# > 4  7657.372 87.50641 70.00630 -2.589091 2.832513 0.1297159
# > 5  7715.576 87.83835 70.24901 -2.587316 2.860673 0.1262895
# > 6  7684.539 87.66151 70.06856 -2.581862 2.881085 0.1281054
# > 7  7603.805 87.19980 69.88523 -2.587670 2.837522 0.1254628
# > 8  7684.407 87.66075 70.23024 -2.584258 2.865028 0.1261363
# > 9  7764.996 88.11922 70.49299 -2.585178 2.914878 0.1242307
# > 10 7506.756 86.64154 69.66155 -2.588451 2.767635 0.1299650
train_metrics_main$r2   # = main_r2s_weighted
# > [1] 0.1245698 0.1249450 0.1254169 0.1297159 0.1262895 0.1281054 0.1254628 0.1261363 0.1242307 0.1299650

train_metric_main <- colMeans(train_metrics_main)
train_metric_main 
# >          mse         rmse          mae         bias     bias_pct           r2
# > 7665.0697593   87.5491628   70.1110600   -2.5867083    2.8527103    0.1264837 
train_metric_main["r2"]  # = main_r2_weighted
# >        r2 
# > 0.1264837

# --- Replicate predictions for training data ---
tic("Computing train_metrics_replicates")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_true <- train_data[[pvmaths[m]]]
    w <- train_data[[rep_wts[g]]]
    X_train <- train_data[, oos]
    dtrain <- xgb.DMatrix(data = as.matrix(X_train))
    y_pred <- predict(model, dtrain)
    compute_metrics(y_true, y_pred, w)
  }) |> t()
}) # a list of M=10 matrices, each of shape 80x5
toc()
# > Computing train_metrics_replicates: 4.709 sec elapsed
class(train_metrics_replicates[[1]]); dim(train_metrics_replicates[[1]])
train_metrics_replicates[[1]] # 80x6 "matrix" "array"

# Sanity check for consistency
train_metrics_replicates[[1]][, "r2"]
rep_r2_weighted [, 1]
rep_r2_weighted [, 1] == train_metrics_replicates[[1]][, "r2"]
isTRUE(all.equal(rep_r2_weighted [, 1], train_metrics_replicates[[1]][, "r2"])) # TRUE

# Combine BRR + Rubin
sampling_var_matrix_train <- sapply(1:M, function(m) {
  sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>  # class(train_metrics_main[1, ]), "data.frame";  class(unlist(train_metrics_main[1, ])), "numeric"
    colSums() / (G * (1 - k)^2)
})
sampling_var_matrix_train
# >                  [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > mse      4.052932e+02 4.029095e+02 4.544469e+02 3.665574e+02 4.869393e+02 4.281608e+02 4.258930e+02 4.635832e+02 5.314529e+02 4.377457e+02
# > rmse     1.331638e-02 1.339605e-02 1.471718e-02 1.204815e-02 1.591283e-02 1.403548e-02 1.412177e-02 1.522317e-02 1.726416e-02 1.470528e-02
# > mae      9.398112e-03 1.023276e-02 1.078377e-02 8.701485e-03 1.186750e-02 1.271820e-02 1.022351e-02 1.162537e-02 1.202876e-02 1.024393e-02
# > bias     1.073397e-06 1.045499e-06 1.157594e-06 1.004753e-06 1.113314e-06 1.070513e-06 1.120985e-06 1.105995e-06 1.086515e-06 1.069693e-06
# > bias_pct 6.953721e-05 6.984218e-05 1.035441e-04 6.405103e-05 8.030218e-05 8.629697e-05 1.143199e-04 8.437749e-05 1.066203e-04 7.527460e-05
# > r2       2.762450e-06 2.529594e-06 3.400751e-06 3.037601e-06 3.305503e-06 2.637620e-06 3.398393e-06 3.128012e-06 3.633324e-06 3.548002e-06

# <=> Equivalent codes
# sampling_var_matrix_train <- sapply(1:M, function(m) {
#   sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
#     colMeans() / (1 - k)^2
# })
# sampling_var_matrix_train

# Debugged and check consistency
sampling_var_matrix_train["r2", ]
# > [1] 2.762450e-06 2.529594e-06 3.400751e-06 3.037601e-06 3.305503e-06 2.637620e-06 3.398393e-06 3.128012e-06 3.633324e-06 3.548002e-06
sampling_var_r2s_weighted
# > [1] 2.762450e-06 2.529594e-06 3.400751e-06 3.037601e-06 3.305503e-06 2.637620e-06 3.398393e-06 3.128012e-06 3.633324e-06 3.548002e-06
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
sampling_var_r2_weighted
# > [1] 3.138125e-06

sampling_var_train <- rowMeans(sampling_var_matrix_train)
sampling_var_train                                           # sampling_var_train['r2'] = sampling_var_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 4.402982e+02 1.447405e-02 1.078234e-02 1.084826e-06 8.541659e-05 3.138125e-06 

imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
imputation_var_train                                              # imputation_var_train ['r2'] = imputation_var_r2_weighted 
# >         mse         rmse          mae         bias     bias_pct           r2 
# > 7.270693e+03 2.376173e-01 9.386193e-02 6.077829e-06 2.486778e-03 4.293523e-06 

var_final_train <- sampling_var_train + (1 + 1/M) * imputation_var_train
var_final_train                                              # var_final_train ['r2'] = var_final_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 8.438060e+03 2.758531e-01 1.140305e-01 7.770437e-06 2.820872e-03 7.861001e-06

se_final_train <- sqrt(var_final_train)
se_final_train                                         # se_final_train ['r2'] = se_final_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 91.858914906  0.525217170  0.337683972  0.002787550  0.053111883  0.002803748 

# Confidence intervals
ci_lower <- train_metric_main - z_crit * se_final_train
ci_upper <- train_metric_main + z_crit * se_final_train
ci_length <- ci_upper - ci_lower

train_eval <- tibble(
  Metric = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train, scientific = FALSE),
  CI_lower = format(ci_lower, scientific = FALSE),
  CI_upper = format(ci_upper, scientific = FALSE),
  CI_length = format(ci_length, scientific = FALSE)
)

print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate  Standard_error    CI_lower    CI_upper   CI_length
# >       MSE   7665.0697593   91.858914906 7485.0295944 7845.109924 360.08032975
# >      RMSE     87.5491628    0.525217170   86.5197560   88.578570   2.05881347
# >       MAE     70.1110600    0.337683972   69.4492115   70.772908   1.32369685
# >      Bias     -2.5867083    0.002787550   -2.5921718   -2.581245   0.01092700
# >     Bias%      2.8527103    0.053111883    2.7486129    2.956808   0.20819475
# > R-squared      0.1264837    0.002803748    0.1209885    0.131979   0.01099049

### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_true <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- valid_data[, oos]
  dvalid <- xgb.DMatrix(data = as.matrix(X_valid))
  y_pred <- predict(model, dvalid)
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
valid_metrics_main

# Combine across plausible values
valid_metric_main <- colMeans(valid_metrics_main)
valid_metric_main

# Replicate predictions on validation set
tic("Computing valid_metrics_replicates")
valid_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_true <- valid_data[[pvmaths[m]]]
    w <- valid_data[[rep_wts[g]]]
    X_valid <- valid_data[, oos]
    dvalid <- xgb.DMatrix(data = as.matrix(X_valid))
    y_pred <- predict(model, dvalid)
    compute_metrics(y_true, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates: 3.635 sec elapsed

# Combine BRR + Rubin's Rules
sampling_var_matrix_valid <- sapply(1:M, function(m) {
  sweep(valid_metrics_replicates[[m]], 2, unlist(valid_metrics_main[m, ]))^2 |> 
    colSums() / (G * (1 - k)^2)
})
sampling_var_valid <- rowMeans(sampling_var_matrix_valid)
imputation_var_valid <- colSums((valid_metrics_main - matrix(valid_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
var_final_valid <- sampling_var_valid + (1 + 1/M) * imputation_var_valid
se_final_valid <- sqrt(var_final_valid)

# Confidence intervals
ci_lower_valid <- valid_metric_main - z_crit * se_final_valid
ci_upper_valid <- valid_metric_main + z_crit * se_final_valid
ci_length_valid <- ci_upper_valid - ci_lower_valid

# Format results
valid_eval <- tibble(
  Metric = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(valid_metric_main, scientific = FALSE),
  Standard_error = format(se_final_valid, scientific = FALSE),
  CI_lower = format(ci_lower_valid, scientific = FALSE),
  CI_upper = format(ci_upper_valid, scientific = FALSE),
  CI_length = format(ci_length_valid, scientific = FALSE)
)

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric  Point_estimate Standard_error      CI_lower     CI_upper    CI_length
# >       MSE   7391.50966429  194.895481207 7009.52154037 7773.4977882 763.97624783
# >      RMSE     85.96805704    1.126968485   83.75923940   88.1768747   4.41763529
# >       MAE     68.80261917    0.943035908   66.95430275   70.6509356   3.69663283
# >      Bias    - 4.85319953    1.376620788   -7.55132669   -2.1550724   5.39625433
# >     Bias%      2.20200596    0.306007217    1.60224284    2.8017691   1.19952625
# > R-squared      0.09619875    0.006492783    0.08347313    0.1089244   0.02545124

### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_true <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- test_data[, oos]
  dtest <- xgb.DMatrix(data = as.matrix(X_test))
  y_pred <- predict(model, dtest)
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
test_metrics_main

# Combine across plausible values
test_metric_main <- colMeans(test_metrics_main)
test_metric_main

# Replicate predictions on test set
tic("Computing test_metrics_replicates")
test_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_true <- test_data[[pvmaths[m]]]
    w <- test_data[[rep_wts[g]]]
    X_test <- test_data[, oos]
    dtest <- xgb.DMatrix(data = as.matrix(X_test))
    y_pred <- predict(model, dtest)
    compute_metrics(y_true, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates: 2.144 sec elapsed

# Combine BRR + Rubin's Rules
sampling_var_matrix_test <- sapply(1:M, function(m) {
  sweep(test_metrics_replicates[[m]], 2, unlist(test_metrics_main[m, ]))^2 |> 
    colSums() / (G * (1 - k)^2)
})

sampling_var_test <- rowMeans(sampling_var_matrix_test)
imputation_var_test <- colSums((test_metrics_main - matrix(test_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
var_final_test <- sampling_var_test + (1 + 1/M) * imputation_var_test
se_final_test <- sqrt(var_final_test)

# Confidence intervals
ci_lower_test <- test_metric_main - z_crit * se_final_test
ci_upper_test <- test_metric_main + z_crit * se_final_test
ci_length_test <- ci_upper_test - ci_lower_test

# Format results
test_eval <- tibble(
  Metric = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(test_metric_main, scientific = FALSE),
  Standard_error = format(se_final_test, scientific = FALSE),
  CI_lower = format(ci_lower_test, scientific = FALSE),
  CI_upper = format(ci_upper_test, scientific = FALSE),
  CI_length = format(ci_length_test, scientific = FALSE)
)

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper   CI_length
# >       MSE    7836.452996  135.587828313 7570.7057353 8102.2002558 531.4945205
# >      RMSE      88.521177    0.764866319   87.0220662   90.0202870   2.9982209
# >       MAE      71.165893    0.722933793   69.7489685   72.5828169   2.8338484
# >      Bias      -4.442872    0.840635362   -6.0904873   -2.7952573   3.2952301
# >     Bias%       2.453151    0.176163563    2.1078767    2.7984252   0.6905485
# > R-squared       0.102056    0.006651499    0.0890193    0.1150927   0.0260734

### ---- ** Predictive Performance on Training/Validation/Test Data (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process to avoid redundancy.

# Evaluation function 
evaluate_split <- function(split_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, oos, pvmaths) {
  
  # Main plausible values loop
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    y_true <- split_data[[pvmaths[i]]]
    w <- split_data[[final_wt]]
    features <- split_data[, oos]
    dmat <- xgb.DMatrix(data = as.matrix(features))
    y_pred <- predict(model, dmat)
    compute_metrics(y_true, y_pred, w)
  }) |> t() |> as.data.frame()
  
  main_point <- colMeans(main_metrics_df)
  
  # Replicate loop
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      y_true <- split_data[[pvmaths[m]]]
      w <- split_data[[rep_wts[g]]]
      features <- split_data[, oos]
      dmat <- xgb.DMatrix(data = as.matrix(features))
      y_pred <- predict(model, dmat)
      compute_metrics(y_true, y_pred, w)
    }) |> t()
  })
  
  # BRR sampling variance
  sampling_var_matrix <- sapply(1:M, function(m) {
    sweep(replicate_metrics[[m]], 2, unlist(main_metrics_df[m, ]))^2 |>
      colSums() / (G * (1 - k)^2)
  })
  sampling_var <- rowMeans(sampling_var_matrix)
  
  # Rubin's imputation variance
  imputation_var <- colSums((main_metrics_df - matrix(main_point, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
  
  # Total variance and confidence intervals
  var_final <- sampling_var + (1 + 1/M) * imputation_var
  se_final <- sqrt(var_final)
  ci_lower <- main_point - z_crit * se_final
  ci_upper <- main_point + z_crit * se_final
  ci_length <- ci_upper - ci_lower
  
  tibble::tibble(
    Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
    Point_estimate = format(main_point, scientific = FALSE),
    Standard_error = format(se_final, scientific = FALSE),
    CI_lower       = format(ci_lower, scientific = FALSE),
    CI_upper       = format(ci_upper, scientific = FALSE),
    CI_length      = format(ci_length, scientific = FALSE)
  )
}

# Evaluate for each data split
train_eval <- evaluate_split(train_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, oos, pvmaths)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, oos, pvmaths)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, oos, pvmaths)

# Display
print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

### ---- Summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate  Standard_error    CI_lower    CI_upper   CI_length
# >       MSE   7665.0697593   91.858914906 7485.0295944 7845.109924 360.08032975
# >      RMSE     87.5491628    0.525217170   86.5197560   88.578570   2.05881347
# >       MAE     70.1110600    0.337683972   69.4492115   70.772908   1.32369685
# >      Bias     -2.5867083    0.002787550   -2.5921718   -2.581245   0.01092700
# >     Bias%      2.8527103    0.053111883    2.7486129    2.956808   0.20819475
# > R-squared      0.1264837    0.002803748    0.1209885    0.131979   0.01099049

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric  Point_estimate Standard_error      CI_lower     CI_upper    CI_length
# >       MSE   7391.50966429  194.895481207 7009.52154037 7773.4977882 763.97624783
# >      RMSE     85.96805704    1.126968485   83.75923940   88.1768747   4.41763529
# >       MAE     68.80261917    0.943035908   66.95430275   70.6509356   3.69663283
# >      Bias    - 4.85319953    1.376620788   -7.55132669   -2.1550724   5.39625433
# >     Bias%      2.20200596    0.306007217    1.60224284    2.8017691   1.19952625
# > R-squared      0.09619875    0.006492783    0.08347313    0.1089244   0.02545124

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper   CI_length
# >       MSE    7836.452996  135.587828313 7570.7057353 8102.2002558 531.4945205
# >      RMSE      88.521177    0.764866319   87.0220662   90.0202870   2.9982209
# >       MAE      71.165893    0.722933793   69.7489685   72.5828169   2.8338484
# >      Bias      -4.442872    0.840635362   -6.0904873   -2.7952573   3.2952301
# >     Bias%       2.453151    0.176163563    2.1078767    2.7984252   0.6905485
# > R-squared       0.102056    0.006651499    0.0890193    0.1150927   0.0260734
