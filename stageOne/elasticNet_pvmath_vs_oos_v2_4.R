# ---- II. Predictive Modelling: Version 2.4 ----

# Tune hyperparameters (alpha α and lambda λ) using cv.glmnet, with manual K-folds + tracks out-of-fold (OOF) predictions/metrics.

## ---- 1. Setup ----

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
set.seed(123)          # Ensure reproducibility
n <- nrow(temp_data)   # 20003
indices <- sample(n)   # Randomly shuffle row indices

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

## ---- 2. PV1MATH only ----

# --- Remark ---
# 1) Repeat the same process for PV2MATH - PV10MATH.
# 2) Apply best results from PV1MATH to all plausible values in mathematics. 

## ---- Main model using final student weights (W_FSTUWT) ---- 

### ---- Tuning Elastic Net for PV1MATH only: cv.glmnet ----

# Target plausible value
pv1math <- pvmaths[1]

# Training data
X_train <- as.matrix(train_data[, oos])
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

# Validation data 
X_valid <- as.matrix(valid_data[, oos])
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

# Test data 
X_test <- as.matrix(test_data[, oos])
y_test <- test_data[[pv1math]]
w_test <- test_data[[final_wt]]

# --- Fixed Manual CV folds on TRAIN (reproducible) ---
set.seed(123)
num_folds <- 5L
n_cv      <- nrow(X_train)
stopifnot(n_cv >= num_folds)                                                             # basic guard
cv_order  <- sample.int(n_cv)                                                            # random permutation
bounds    <- floor(seq(0, n_cv, length.out = num_folds + 1))
stopifnot(all(diff(bounds) > 0))                                                         # no empty folds
cv_folds  <- vector("list", num_folds)
for (k in seq_len(num_folds)) cv_folds[[k]] <- cv_order[(bounds[k] + 1):bounds[k + 1]]
stopifnot(identical(sort(unlist(cv_folds)), seq_len(n_cv)))

# Convert cv_folds (list of indices) -> foldid (vector 1..K for each row)
foldid <- integer(n_cv)
for (k in seq_along(cv_folds)) foldid[cv_folds[[k]]] <- k
stopifnot(all(foldid %in% seq_len(num_folds)), length(foldid) == n_cv)

# Weight balance across folds
tapply(w_train, foldid, sum)
# >        1        2        3        4        5 
# > 44064.98 45527.15 43272.60 44609.90 44153.68 
table(foldid)
# > foldid
# >    1    2    3    4    5 
# > 2800 2800 2801 2800 2801 

# Further check CV-fold weight balance 
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

# Outcome distribution by fold
print(do.call(rbind, tapply(y_train, foldid, function(v) c(mean = mean(v), sd = sd(v)))))
# >       mean       sd
# > 1 489.4925 92.81556
# > 2 492.0924 93.06908
# > 3 489.7786 90.84370
# > 4 492.2019 92.46838
# > 5 490.3393 92.04746

#### ---- 1) Tuning: grouped = TRUE (Default), keep = TRUE ----

# Grid over elastic‑net mixing parameter
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso
# options: alpha_grid <- seq(0, 1, by = 0.05), alpha_grid <- sort(unique(c(seq(0, 1, by = 0.1), 0.001, 0.005, 0.01, 0.05))), etc. 

# Storage
cv_list        <- vector("list", length(alpha_grid))  # cv.glmnet fits per alpha
per_alpha_list <- vector("list", length(alpha_grid))  # two rows per alpha: lambda.min & lambda.1se

# Parallel backend (for cv.glmnet parallelization)
registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Grid over alpha (cv.glmnet, manual 5-fold)")
for (i in seq_along(alpha_grid)) {
  alpha <- alpha_grid[i]
  message(sprintf("Fitting cv.glmnet for alpha = %.1f (manual 5-fold CV on TRAIN)", alpha))
  
  # Fix fold randomness 
  set.seed(123)
  
  cvmod <- cv.glmnet(
    x = X_train,
    y = y_train,
    weights = w_train,
    type.measure = "mse", 
    foldid = foldid,           # <-
    grouped = TRUE,            # <-                              
    keep = TRUE,               # <-                               
    parallel = TRUE,          
    trace.it = 0,
    alpha = alpha,
    family = "gaussian",
    standardize = TRUE,
    intercept = TRUE
  )
  cv_list[[i]] <- cvmod
  
  # Indices
  idx_min <- cvmod$index["min", 1]
  idx_1se <- cvmod$index["1se", 1]
  
  # Lambdas and path metadata at those indices
  lambda_min <- cvmod$lambda[idx_min]      # == cvmod$lambda.min
  lambda_1se <- cvmod$lambda[idx_1se]      # == cvmod$lambda.1se
  
  nzero_min  <- cvmod$nzero[idx_min]
  nzero_1se  <- cvmod$nzero[idx_1se]
  
  dev_min    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_min])
  dev_1se    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_1se])
  
  # CV summaries (in-built)
  cvm_min    <- as.numeric(cvmod$cvm[idx_min])
  cvsd_min   <- as.numeric(cvmod$cvsd[idx_min])
  cvup_min   <- as.numeric(cvmod$cvup[idx_min])
  cvlo_min   <- as.numeric(cvmod$cvlo[idx_min])
  
  cvm_1se    <- as.numeric(cvmod$cvm[idx_1se])
  cvsd_1se   <- as.numeric(cvmod$cvsd[idx_1se])
  cvup_1se   <- as.numeric(cvmod$cvup[idx_1se])
  cvlo_1se   <- as.numeric(cvmod$cvlo[idx_1se])
  
  # Two rows per α: one for lambda.min, one for lambda.1se
  per_alpha_list[[i]] <- tibble::tibble(
    alpha       = alpha,
    alpha_idx   = i,
    s           = c("lambda.min", "lambda.1se"),
    lambda      = c(lambda_min,   lambda_1se),
    lambda_idx  = c(idx_min,      idx_1se),
    nzero       = c(nzero_min,    nzero_1se),
    dev_ratio   = c(dev_min,      dev_1se),
    cvm         = c(cvm_min,      cvm_1se),
    cvsd        = c(cvsd_min,     cvsd_1se),
    cvlo        = c(cvlo_min,     cvlo_1se),
    cvup        = c(cvup_min,     cvup_1se)
  )
}
toc()
# > Grid over alpha (cv.glmnet, manual 5-fold): 1.589 sec elapsed

##### ---- Explore the first tuning results ----
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by CV MSE (lower is better) irrespective of rule
tuning_results %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  #head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx          s      lambda lambda_idx nzero  dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.0         1 lambda.min  1.95726075        100     4 0.05352143 8310.048 58.19281 8251.855 8368.241
# >   0.1         2 lambda.min  0.73689611         61     4 0.05353111 8310.258 58.24930 8252.009 8368.507
# >   0.2         3 lambda.min  0.25395695         65     4 0.05353768 8310.297 58.28652 8252.011 8368.584
# >   0.3         4 lambda.min  0.18581163         64     4 0.05353779 8310.312 58.28966 8252.023 8368.602
# >   0.5         6 lambda.min  0.12235682         63     4 0.05353783 8310.317 58.29708 8252.020 8368.614
# >   0.4         5 lambda.min  0.15294602         63     4 0.05353765 8310.321 58.29005 8252.031 8368.611
# >   0.7         8 lambda.min  0.09591890         62     4 0.05353760 8310.335 58.29203 8252.043 8368.627
# >   0.8         9 lambda.min  0.08392904         62     4 0.05353767 8310.337 58.29291 8252.044 8368.630
# >   0.9        10 lambda.min  0.07460359         62     4 0.05353773 8310.339 58.29360 8252.046 8368.633
# >   0.6         7 lambda.min  0.11190538         62     4 0.05353751 8310.339 58.29257 8252.047 8368.632
# >   1.0        11 lambda.min  0.06714323         62     4 0.05353777 8310.341 58.29415 8252.046 8368.635
# >   0.8         9 lambda.1se  4.58443661         19     3 0.04684211 8362.782 58.05622 8304.726 8420.838
# >   0.2         3 lambda.1se 15.22431926         21     4 0.04702745 8363.283 59.17683 8304.106 8422.459
# >   0.7         8 lambda.1se  5.23935613         19     3 0.04675936 8363.425 58.07059 8305.355 8421.496
# >   0.6         7 lambda.1se  6.11258215         19     3 0.04664677 8364.305 58.09043 8306.214 8422.395
# >   0.3         4 lambda.1se 11.13911439         20     4 0.04676437 8364.329 58.65193 8305.677 8422.981
# >   0.0         1 lambda.1se 61.17840939         63     4 0.04648180 8364.969 59.27289 8305.696 8424.242
# >   0.1         2 lambda.1se 25.27899462         23     4 0.04679828 8365.288 59.82835 8305.460 8425.117
# >   0.5         6 lambda.1se  7.33509858         19     3 0.04648516 8365.576 58.11956 8307.456 8423.695
# >   0.4         5 lambda.1se  9.16887322         19     3 0.04623502 8367.558 58.16429 8309.393 8425.722
# >   1.0        11 lambda.1se  4.02513082         18     3 0.04631327 8368.111 57.47850 8310.633 8425.590
# >   0.9        10 lambda.1se  4.47236757         18     3 0.04625534 8368.550 57.49203 8311.058 8426.042

# Choose winners under each rule separately 
best_min  <- tuning_results %>%
  filter(s == "lambda.min") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%   # tie-breaks: fewer nonzeros (nzero), larger λ, smaller α
  slice(1)

best_1se  <- tuning_results %>%
  filter(s == "lambda.1se") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  slice(1)

print(as.data.frame(best_min),  row.names = FALSE)
# > alpha alpha_idx          s   lambda lambda_idx nzero  dev_ratio      cvm     cvsd     cvlo     cvup
# >      0         1 lambda.min 1.957261        100     4 0.05352143 8310.048 58.19281 8251.855 8368.241
print(as.data.frame(best_1se), row.names = FALSE)
# > alpha alpha_idx          s   lambda lambda_idx nzero  dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.8         9 lambda.1se 4.584437         19     3 0.04684211 8362.782 58.05622 8304.726 8420.838

best_alpha_min     <- best_min$alpha
best_lambda_min    <- best_min$lambda
best_alpha_idx_min <- best_min$alpha_idx

best_alpha_1se     <- best_1se$alpha
best_lambda_1se    <- best_1se$lambda
best_alpha_idx_1se <- best_1se$alpha_idx

message(sprintf("Winner @ lambda.min : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_min, best_lambda_min, best_min$nzero, best_min$cvm, best_min$cvsd))
# > Winner @ lambda.min : alpha = 0.00 | lambda = 1.957261 | nzero = 4 | CVM = 8310.04809 (± 58.19281)
message(sprintf("Winner @ lambda.1se : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_1se, best_lambda_1se, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
# > Winner @ lambda.1se : alpha = 0.80 | lambda = 4.584437 | nzero = 3 | CVM = 8362.78219 (± 58.05622)

##### ---- Predict & evaluate on TRAIN / VALID / TEST for both winners ----
cv_best_min  <- cv_list[[best_alpha_idx_min]]
cv_best_min
cv_best_1se  <- cv_list[[best_alpha_idx_1se]]
cv_best_1se

# Coefficients & importances
coef(cv_best_min,  s = best_lambda_min)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             s=1.957261
# > (Intercept) 521.236974
# > EXERPRAC     -1.855744
# > STUDYHMW      1.891380
# > WORKPAY      -6.361346
# > WORKHOME     -1.812967
coef(cv_best_1se,  s = best_lambda_1se)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >              s=4.584437
# > (Intercept) 518.4288701
# > EXERPRAC     -0.8246734
# > STUDYHMW      .        
# > WORKPAY      -5.5091581
# > WORKHOME     -0.6367917

varImp(cv_best_min$glmnet.fit,  lambda = best_lambda_min)
# >           Overall
# > EXERPRAC 1.855744
# > STUDYHMW 1.891380
# > WORKPAY  6.361346
# > WORKHOME 1.812967
varImp(cv_best_1se$glmnet.fit,  lambda = best_lambda_1se)
# >            Overall
# > EXERPRAC 0.8246734
# > STUDYHMW 0.0000000
# > WORKPAY  5.5091581
# > WORKHOME 0.6367917

# Predictions
pred_train_min <- as.numeric(predict(cv_best_min, newx = X_train, s = best_lambda_min))
pred_valid_min <- as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min))
pred_test_min  <- as.numeric(predict(cv_best_min,  newx = X_test,  s = best_lambda_min))

pred_train_1se <- as.numeric(predict(cv_best_1se, newx = X_train, s = best_lambda_1se))
pred_valid_1se <- as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se))
pred_test_1se  <- as.numeric(predict(cv_best_1se,  newx = X_test,  s = best_lambda_1se))

# Performance Metrics
metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Results tables
metric_results_min <- tibble::tibble(
  Rule    = "lambda.min",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"], metrics_valid_min["rmse"], metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],  metrics_valid_min["mae"],  metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"], metrics_valid_min["bias"], metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],   metrics_valid_min["r2"],   metrics_test_min["r2"])
)

metric_results_1se <- tibble::tibble(
  Rule    = "lambda.1se",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"], metrics_valid_1se["rmse"], metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],  metrics_valid_1se["mae"],  metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"], metrics_valid_1se["bias"], metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],   metrics_valid_1se["r2"],   metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.min   Training 91.06868 73.19982 -3.758605e-13 3.606879 0.05352143
# > lambda.min Validation 88.00754 70.62964 -3.015818e+00 2.744678 0.05265334
# > lambda.min       Test 91.32308 73.70615 -1.073580e+00 3.420891 0.06852296

print(as.data.frame(metric_results_1se),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.1se   Training 91.38946 73.52732 -3.623010e-13 3.661776 0.04684211
# > lambda.1se Validation 88.21869 70.73364 -2.553865e+00 2.892379 0.04810221
# > lambda.1se       Test 92.00023 74.25070 -6.904099e-01 3.571384 0.05465809

print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.min   Training 91.06868 73.19982 -3.758605e-13 3.606879 0.05352143
# > lambda.min Validation 88.00754 70.62964 -3.015818e+00 2.744678 0.05265334
# > lambda.min       Test 91.32308 73.70615 -1.073580e+00 3.420891 0.06852296
# > lambda.1se   Training 91.38946 73.52732 -3.623010e-13 3.661776 0.04684211
# > lambda.1se Validation 88.21869 70.73364 -2.553865e+00 2.892379 0.04810221
# > lambda.1se       Test 92.00023 74.25070 -6.904099e-01 3.571384 0.05465809

# Sanity check: wrapper vs underlying at the same λ
# all.equal(as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min)),
#           as.numeric(predict(cv_best_min$glmnet.fit, newx = X_valid, s = best_lambda_min)))
# # > [1] TRUE
# all.equal(as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se)),
#           as.numeric(predict(cv_best_1se$glmnet.fit, newx = X_valid, s = best_lambda_1se)))
# # > [1] TRUE

#### ---- 2) Tuning: grouped = FALSE, keep = TRUE ----

# Grid over elastic‑net mixing parameter
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso
# options: alpha_grid <- seq(0, 1, by = 0.05), alpha_grid <- sort(unique(c(seq(0, 1, by = 0.1), 0.001, 0.005, 0.01, 0.05))), etc. 

# Storage
cv_list        <- vector("list", length(alpha_grid))  # cv.glmnet fits per alpha
per_alpha_list <- vector("list", length(alpha_grid))  # two rows per alpha: lambda.min & lambda.1se

# Parallel backend (for cv.glmnet parallelization)
registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Grid over alpha (cv.glmnet, manual 5-fold)")
for (i in seq_along(alpha_grid)) {
  alpha <- alpha_grid[i]
  message(sprintf("Fitting cv.glmnet for alpha = %.1f (manual 5-fold CV on TRAIN)", alpha))
  
  # Fix fold randomness 
  set.seed(123)
  
  cvmod <- cv.glmnet(
    x = X_train,
    y = y_train,
    weights = w_train,
    type.measure = "mse", 
    foldid = foldid,            # <-
    grouped = FALSE,            # <-                              
    keep = TRUE,                # <-                               
    parallel = TRUE,          
    trace.it = 0,
    alpha = alpha,
    family = "gaussian",
    standardize = TRUE,
    intercept = TRUE
  )
  cv_list[[i]] <- cvmod
  
  # Indices
  idx_min <- cvmod$index["min", 1]
  idx_1se <- cvmod$index["1se", 1]
  
  # Lambdas and path metadata at those indices
  lambda_min <- cvmod$lambda[idx_min]      # == cvmod$lambda.min
  lambda_1se <- cvmod$lambda[idx_1se]      # == cvmod$lambda.1se
  
  nzero_min  <- cvmod$nzero[idx_min]
  nzero_1se  <- cvmod$nzero[idx_1se]
  
  dev_min    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_min])
  dev_1se    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_1se])
  
  # CV summaries (in-built)
  cvm_min    <- as.numeric(cvmod$cvm[idx_min])
  cvsd_min   <- as.numeric(cvmod$cvsd[idx_min])
  cvup_min   <- as.numeric(cvmod$cvup[idx_min])
  cvlo_min   <- as.numeric(cvmod$cvlo[idx_min])
  
  cvm_1se    <- as.numeric(cvmod$cvm[idx_1se])
  cvsd_1se   <- as.numeric(cvmod$cvsd[idx_1se])
  cvup_1se   <- as.numeric(cvmod$cvup[idx_1se])
  cvlo_1se   <- as.numeric(cvmod$cvlo[idx_1se])
  
  # Two rows per α: one for lambda.min, one for lambda.1se
  per_alpha_list[[i]] <- tibble::tibble(
    alpha       = alpha,
    alpha_idx   = i,
    s           = c("lambda.min", "lambda.1se"),
    lambda      = c(lambda_min,   lambda_1se),
    lambda_idx  = c(idx_min,      idx_1se),
    nzero       = c(nzero_min,    nzero_1se),
    dev_ratio   = c(dev_min,      dev_1se),
    cvm         = c(cvm_min,      cvm_1se),
    cvsd        = c(cvsd_min,     cvsd_1se),
    cvlo        = c(cvlo_min,     cvlo_1se),
    cvup        = c(cvup_min,     cvup_1se)
  )
}
toc()
# > Grid over alpha (cv.glmnet, manual 5-fold): 1.95 sec elapsed

##### ---- Explore the first tuning results ----
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by CV MSE (lower is better) irrespective of rule
tuning_results %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  #head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx          s      lambda lambda_idx nzero  dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.0         1 lambda.min  1.95726075        100     4 0.05352143 8310.048 94.66112 8215.387 8404.709
# >   0.1         2 lambda.min  0.73689611         61     4 0.05353111 8310.258 94.66966 8215.588 8404.928
# >   0.2         3 lambda.min  0.25395695         65     4 0.05353768 8310.297 94.67795 8215.619 8404.975
# >   0.3         4 lambda.min  0.18581163         64     4 0.05353779 8310.312 94.67837 8215.634 8404.991
# >   0.5         6 lambda.min  0.12235682         63     4 0.05353783 8310.317 94.67812 8215.639 8404.995
# >   0.4         5 lambda.min  0.15294602         63     4 0.05353765 8310.321 94.67826 8215.643 8405.000
# >   0.7         8 lambda.min  0.09591890         62     4 0.05353760 8310.335 94.67842 8215.657 8405.014
# >   0.8         9 lambda.min  0.08392904         62     4 0.05353767 8310.337 94.67859 8215.659 8405.016
# >   0.9        10 lambda.min  0.07460359         62     4 0.05353773 8310.339 94.67871 8215.660 8405.018
# >   0.6         7 lambda.min  0.11190538         62     4 0.05353751 8310.339 94.67836 8215.661 8405.018
# >   1.0        11 lambda.min  0.06714323         62     4 0.05353777 8310.341 94.67881 8215.662 8405.019
# >   0.7         8 lambda.1se  7.60140865         15     3 0.04314506 8395.434 94.91873 8300.515 8490.353
# >   0.2         3 lambda.1se 22.08788051         17     3 0.04284309 8396.383 94.94124 8301.442 8491.325
# >   0.0         1 lambda.1se 88.75939696         59     4 0.04270578 8396.604 94.92529 8301.679 8491.529
# >   0.3         4 lambda.1se 16.16094773         16     3 0.04287335 8396.763 94.93250 8301.830 8491.695
# >   0.6         7 lambda.1se  8.86831009         15     3 0.04295050 8396.964 94.92544 8302.039 8491.890
# >   0.5         6 lambda.1se 10.64197211         15     3 0.04267323 8399.161 94.93561 8304.225 8494.096
# >   0.1         2 lambda.1se 36.67549289         19     3 0.04203576 8402.123 94.99349 8307.129 8497.116
# >   0.4         5 lambda.1se 13.30246514         15     3 0.04224893 8402.547 94.95232 8307.595 8497.500
# >   1.0        11 lambda.1se  5.83977563         14     3 0.04213472 8402.898 94.94526 8307.953 8497.843
# >   0.9        10 lambda.1se  6.48863958         14     3 0.04203642 8403.758 94.94940 8308.808 8498.707
# >   0.8         9 lambda.1se  7.29971953         14     3 0.04191219 8404.844 94.95474 8309.889 8499.798

# Choose winners under each rule separately
best_min  <- tuning_results %>%
  filter(s == "lambda.min") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%   # tie-breaks: fewer nonzeros (nzero), larger λ, smaller α
  slice(1)

best_1se  <- tuning_results %>%
  filter(s == "lambda.1se") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  slice(1)

print(as.data.frame(best_min),  row.names = FALSE)
# > alpha alpha_idx          s   lambda lambda_idx nzero  dev_ratio      cvm     cvsd     cvlo     cvup
# >     0         1 lambda.min 1.957261        100     4 0.05352143 8310.048 58.19281 8251.855 8368.241
print(as.data.frame(best_1se), row.names = FALSE)
# > alpha alpha_idx          s   lambda lambda_idx nzero  dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.7         8 lambda.1se 7.601409         15     3 0.04314506 8395.434 94.91873 8300.515 8490.353

best_alpha_min     <- best_min$alpha
best_lambda_min    <- best_min$lambda
best_alpha_idx_min <- best_min$alpha_idx

best_alpha_1se     <- best_1se$alpha
best_lambda_1se    <- best_1se$lambda
best_alpha_idx_1se <- best_1se$alpha_idx

message(sprintf("Winner @ lambda.min : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_min, best_lambda_min, best_min$nzero, best_min$cvm, best_min$cvsd))
# > Winner @ lambda.min : alpha = 0.00 | lambda = 1.957261 | nzero = 4 | CVM = 8310.04809 (± 94.66112)
message(sprintf("Winner @ lambda.1se : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_1se, best_lambda_1se, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
# > Winner @ lambda.1se : alpha = 0.70 | lambda = 7.601409 | nzero = 3 | CVM = 8395.43397 (± 94.91873)

##### ---- Predict & evaluate on TRAIN / VALID / TEST for both winners ----
cv_best_min  <- cv_list[[best_alpha_idx_min]]
cv_best_1se  <- cv_list[[best_alpha_idx_1se]]

# Coefficients & importances
coef(cv_best_min,  s = best_lambda_min)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             s=1.957261
# > (Intercept) 521.236974
# > EXERPRAC     -1.855744
# > STUDYHMW      1.891380
# > WORKPAY      -6.361346
# > WORKHOME     -1.812967
coef(cv_best_1se,  s = best_lambda_1se)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >              s=4.584437
# > (Intercept) 514.6421790
# > EXERPRAC     -0.5139624
# > STUDYHMW      .        
# > WORKPAY      -5.0020993
# > WORKHOME     -0.3038639

caret::varImp(cv_best_min$glmnet.fit,  lambda = best_lambda_min)
# >           Overall
# > EXERPRAC 1.855744
# > STUDYHMW 1.891380
# > WORKPAY  6.361346
# > WORKHOME 1.812967
caret::varImp(cv_best_1se$glmnet.fit,  lambda = best_lambda_1se)
# >            Overall
# > EXERPRAC 0.5139624
# > STUDYHMW 0.0000000
# > WORKPAY  5.0020993
# > WORKHOME 0.3038639

# Predictions
pred_train_min <- as.numeric(predict(cv_best_min, newx = X_train, s = best_lambda_min))
pred_valid_min <- as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min))
pred_test_min  <- as.numeric(predict(cv_best_min,  newx = X_test,  s = best_lambda_min))

pred_train_1se <- as.numeric(predict(cv_best_1se, newx = X_train, s = best_lambda_1se))
pred_valid_1se <- as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se))
pred_test_1se  <- as.numeric(predict(cv_best_1se,  newx = X_test,  s = best_lambda_1se))

# Performance Metrics
metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Results tables
metric_results_min <- tibble::tibble(
  Rule    = "lambda.min",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"], metrics_valid_min["rmse"], metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],  metrics_valid_min["mae"],  metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"], metrics_valid_min["bias"], metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],   metrics_valid_min["r2"],   metrics_test_min["r2"])
)

metric_results_1se <- tibble::tibble(
  Rule    = "lambda.1se",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"], metrics_valid_1se["rmse"], metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],  metrics_valid_1se["mae"],  metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"], metrics_valid_1se["bias"], metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],   metrics_valid_1se["r2"],   metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.min   Training 91.06868 73.19982 -3.758605e-13 3.606879 0.05352143
# > lambda.min Validation 88.00754 70.62964 -3.015818e+00 2.744678 0.05265334
# > lambda.min       Test 91.32308 73.70615 -1.073580e+00 3.420891 0.06852296

print(as.data.frame(metric_results_1se),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.1se   Training 91.56652 73.71684 -2.914863e-13 3.681987 0.04314506
# > lambda.1se Validation 88.36183 70.85843 -2.394296e+00 2.945672 0.04501063
# > lambda.1se       Test 92.22802 74.48796 -6.115866e-01 3.611393 0.04997108

print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.min   Training 91.06868 73.19982 -3.758605e-13 3.606879 0.05352143
# > lambda.min Validation 88.00754 70.62964 -3.015818e+00 2.744678 0.05265334
# > lambda.min       Test 91.32308 73.70615 -1.073580e+00 3.420891 0.06852296
# > lambda.1se   Training 91.56652 73.71684 -2.914863e-13 3.681987 0.04314506
# > lambda.1se Validation 88.36183 70.85843 -2.394296e+00 2.945672 0.04501063
# > lambda.1se       Test 92.22802 74.48796 -6.115866e-01 3.611393 0.04997108

# Sanity check: wrapper vs underlying at the same λ
# all.equal(as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min)),
#           as.numeric(predict(cv_best_min$glmnet.fit, newx = X_valid, s = best_lambda_min)))
# # > [1] TRUE
# all.equal(as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se)),
#           as.numeric(predict(cv_best_1se$glmnet.fit, newx = X_valid, s = best_lambda_1se)))
# # > [1] TRUE



## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH (best_alpha_min, best_lambda_min) to all plausible values in mathematics.


### ---- 1.1) Use best min results from "Tuning: grouped = TRUE (Default), keep = TRUE" ----

#### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

tic("Fitting main glmnet models (fixed best_alpha_min, best_lambda_min)")
main_models <- lapply(pvmaths, function(pv) {
  
  # TRAIN (final weights)
  X_train <- as.matrix(train_data[, oos])
  y_train <- train_data[[pv]]
  w_train <- train_data[[final_wt]]
  
  # Fit glmnet at chosen alpha + lambda
  mod <- glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    weights = w_train,
    alpha = best_alpha_min,       # best_alpha_min = 0
    lambda = best_lambda_min,     # best_lambda_min = 1.957261
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda_min))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(oos, collapse = " + "))),
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha_min, best_lambda_min): 1.327 sec elapsed

# Quick look
main_models[[1]]$formula
main_models[[1]]$coefs
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  521.237130   -1.855877    1.891441   -6.361276   -1.812961 

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coefs
# >      (Intercept)  EXERPRAC STUDYHMW   WORKPAY  WORKHOME
# > [1,]    521.2371 -1.855877 1.891441 -6.361276 -1.812961
# > [2,]    521.9773 -2.121721 2.050260 -6.068429 -1.878735
# > [3,]    521.2003 -2.138503 2.154773 -6.173953 -1.939328
# > [4,]    523.9237 -2.309738 2.072642 -6.245360 -2.116344
# > [5,]    522.5072 -2.091797 1.999270 -6.263057 -2.007584
# > [6,]    521.1111 -2.276992 2.274406 -6.141983 -2.079890
# > [7,]    522.7089 -2.272537 1.896715 -6.196792 -1.795410
# > [8,]    520.4163 -2.130702 2.383581 -6.389677 -1.980256
# > [9,]    521.8108 -2.167699 2.000283 -6.427460 -1.824257
# > [10,]   521.2441 -2.206120 2.265452 -6.497906 -1.762293

main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  521.813680   -2.157169    2.098882   -6.276589   -1.919706 

# --- Weighted R² on TRAIN (point estimates per PV) ---
main_r2s_weighted <- sapply(1:M, function(i) {
  model  <- main_models[[i]]$mod
  X_train   <- as.matrix(train_data[, oos])
  y_train <- train_data[[pvmaths[i]]]
  w      <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.0560753

#### ---- Replicate models using BRR replicate weights ----

set.seed(123)

tic("Fitting replicate glmnet models (BRR weights)")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    X_train <- as.matrix(train_data[, oos])
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    mod <- glmnet(
      x = X_train,
      y = y_train,
      family = "gaussian",
      weights = w_train,
      alpha = best_alpha_min,
      lambda = best_lambda_min,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda_min))
    coefs  <- coefs_matrix[, 1]
    names(coefs) <- rownames(coefs_matrix)
    
    list(
      formula = as.formula(paste(pv, "~", paste(oos, collapse = " + "))),
      mod     = mod,
      coefs   = coefs
    )
  })
})
toc()
# > Fitting replicate glmnet models (BRR weights): 2.711 sec elapsed

# Example inspect
replicate_models[[1]][[1]]$formula
replicate_models[[1]][[1]]$coefs

# --- Replicate weighted R² on TRAIN (G x M) (optional diagnostic) ---
rep_r2_weighted <- matrix(NA_real_, nrow = G, ncol = M)
for (m in 1:M) {
  X_train   <- as.matrix(train_data[, oos])
  y_train <- train_data[[pvmaths[m]]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)  # 80 x 10

#### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

# Organize replicate coefficients: rep_coefs[[m]] is G x (p+1) matrix for PV m
rep_coefs <- lapply(replicate_models, function(m) {
  do.call(rbind, lapply(m, function(g) g$coefs))
})

# BRR sampling variance per PV then average across PVs
sampling_var_matrix_coef <- sapply(1:M, function(m) {
  sweep(rep_coefs[[m]], 2, main_coefs[m, ])^2 |>
    colSums() / (G * (1 - k)^2)
})
# Average sampling variance across PVs (Rubin)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef)

# Imputation variance across PVs
imputation_var_coef <- colSums(
  (main_coefs - matrix(main_coef, nrow = M, ncol = length(main_coef), byrow = TRUE))^2
) / (M - 1)

# Total variance & SE for coefficients
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
se_final_coef  <- sqrt(var_final_coef)

# Z-tests (large-sample normal) for coefficients
Estimate   <- main_coef
`Std. Error` <- se_final_coef
`z value`  <- Estimate / `Std. Error`
p_z        <- 2 * pnorm(-abs(`z value`))
`Pr(>|z|)` <- format.pval(p_z, digits = 3, eps = .Machine$double.eps)
z_Signif   <- symnum(p_z, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols = c("***", "**", "*", ".", ""))

# t-test
# library(survey)
# repw <- train_data[, rep_wts]          # matrix/data.frame of replicate weights
# des <- svrepdesign(
#   weights           = ~ W_FSTUWT,
#   repweights        = repw,
#   type              = "Fay",
#   rho               = 0.5,             # Fay’s factor k = 0.5 in PISA
#   combined.weights  = TRUE,
#   mse               = TRUE,
#   data              = train_data
# )
# dof_t <- degf(des)                             # design-based df
# dof_t
# # > [1] 79

dof_t <- G-1
`t value`  <- `z value`                         # same ratio; distribution differs
p_t        <- 2 * pt(-abs(`t value`), df = dof_t)
`Pr(>|t|)` <- format.pval(p_t, digits = 3, eps = .Machine$double.eps)
t_Signif   <- symnum(p_t, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols   = c("***","**","*",".",""))

##### ---- Final Outputs ----
# Coefficient table (glmnet): pooled estimates with BRR+Rubin SE
coef_table <- tibble::tibble(
  Term         = names(Estimate),
  Estimate     = as.numeric(Estimate),
  `Std. Error` = as.numeric(`Std. Error`),
  `z value`    = as.numeric(`z value`),
  `Pr(>|z|)`   = `Pr(>|z|)`,
  z_Signif     = as.character(z_Signif),
  `t value`    = as.numeric(`t value`),
  `Pr(>|t|)`   = `Pr(>|t|)`,
  t_Signif     = as.character(t_Signif),
)
print(as.data.frame(coef_table), row.names = FALSE)
# >        Term   Estimate Std. Error    z value Pr(>|z|) z_Signif   t value Pr(>|t|) t_Signif
# >  (Intercept) 521.813680  1.1069664 471.39071   <2e-16      *** 471.39071   <2e-16      ***
# >     EXERPRAC  -2.157169  0.1410339 -15.29538   <2e-16      *** -15.29538   <2e-16      ***
# >     STUDYHMW   2.098882  0.1827162  11.48712   <2e-16      ***  11.48712   <2e-16      ***
# >      WORKPAY  -6.276589  0.1532752 -40.94981   <2e-16      *** -40.94981   <2e-16      ***
# >     WORKHOME  -1.919706  0.1362882 -14.08563   <2e-16      *** -14.08563   <2e-16      ***

# (Optional) R-squared (TRAIN) SE via BRR + Rubin 
sampling_var_r2s_weighted <- sapply(1:M, function(m) {
  sum((rep_r2_weighted[, m] - main_r2s_weighted[m])^2) / (G * (1 - k)^2)
})
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
se_final_r2_weighted  <- sqrt(var_final_r2_weighted)
r2_weighted_table <- tibble::tibble(
  Metric       = "R-squared (Weighted, Train)",
  Estimate     = main_r2_weighted,
  `Std. Error` = se_final_r2_weighted
)
print(as.data.frame(r2_weighted_table), row.names = FALSE)
# >                      Metric   Estimate  Std. Error
# >  R-squared (Weighted, Train) 0.0560753 0.002219868

#### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- as.matrix(train_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
train_metrics_main

# Combine across plausible values
train_metric_main <- colMeans(train_metrics_main)

# --- Replicate predictions for training data ---
tic("Computing train_metrics_replicates (glmnet)")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- train_data[[pvmaths[m]]]
    w <- train_data[[rep_wts[g]]]
    X_train <- as.matrix(train_data[, oos])
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing train_metrics_replicates (glmnet): 1.522 sec elapsed
class(train_metrics_replicates[[1]]); dim(train_metrics_replicates[[1]])
# > [1] "matrix" "array" 
# > [1] 80  6

# BRR sampling variance on TRAIN (vectorized across metrics)
sampling_var_matrix_train <- sapply(1:M, function(m) {
  sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
    colSums() / (G * (1 - k)^2)
})
sampling_var_train <- rowMeans(sampling_var_matrix_train)

# Rubin imputation variance on TRAIN
imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)

# Total variance and SE
var_final_train <- sampling_var_train + (1 + 1/M) * imputation_var_train
se_final_train  <- sqrt(var_final_train)

# Confidence intervals
ci_lower_train  <- train_metric_main - z_crit * se_final_train
ci_upper_train  <- train_metric_main + z_crit * se_final_train
ci_length_train <- ci_upper_train - ci_lower_train

# Format results
train_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train,   scientific = FALSE),
  CI_lower       = format(ci_lower_train,   scientific = FALSE),
  CI_upper       = format(ci_upper_train,   scientific = FALSE),
  CI_length      = format(ci_length_train,  scientific = FALSE)
)
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                CI_lower                CI_upper            CI_length
# >       MSE 8282.8760413742620585253 95.5821430918978478530 8095.53848334898884787 8470.2135993995361786801 374.675116050547330815
# >      RMSE   91.0091172680310052101  0.5255868984525799981   89.97898587631783585   92.0392486597441745744   2.060262783426338729
# >       MAE   73.0722239979660344034  0.3194079489768992253   72.44619592159550336   73.6982520743365654425   1.252056152741062078
# >      Bias   -0.0000000000006862956  0.0000000000004722203   -0.00000000000161183    0.0000000000002392391   0.000000000001851069
# >     Bias%    3.6129475426387367420  0.0561348988819719980    3.50292516255427389    3.7229699227231995984   0.220044760168925713
# > R-squared    0.0560752990112748789  0.0022198675690237985    0.05172443852553976    0.0604261594970100027   0.008701720971470248

#### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- as.matrix(valid_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
valid_metric_main <- colMeans(valid_metrics_main)

# Replicate predictions on validation set
tic("Computing valid_metrics_replicates (glmnet)")
valid_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- valid_data[[pvmaths[m]]]
    w <- valid_data[[rep_wts[g]]]
    X_valid <- as.matrix(valid_data[, oos])
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 0.877 sec elapsed

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
valid_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(valid_metric_main, scientific = FALSE),
  Standard_error = format(se_final_valid,   scientific = FALSE),
  CI_lower       = format(ci_lower_valid,   scientific = FALSE),
  CI_upper       = format(ci_upper_valid,   scientific = FALSE),
  CI_length      = format(ci_length_valid,  scientific = FALSE)
)
print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower      CI_upper    CI_length
# >       MSE  7729.71175399  192.589247910 7352.24376428 8107.17974371 754.93597942
# >      RMSE    87.91343121    1.091276169   85.77456922   90.05229319   4.27772398
# >       MAE    70.55451643    0.930990435   68.72980871   72.37922415   3.64941545
# >      Bias    -2.05257945    1.326146859   -4.65177953    0.54662063   5.19840016
# >     Bias%     2.93823121    0.296416094    2.35726634    3.51919608   1.16192974
# > R-squared     0.05481652    0.006971983    0.04115168    0.06848136   0.02732967

#### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- as.matrix(test_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
test_metric_main <- colMeans(test_metrics_main)

# Replicate predictions on test set
tic("Computing test_metrics_replicates (glmnet)")
test_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- test_data[[pvmaths[m]]]
    w <- test_data[[rep_wts[g]]]
    X_test <- as.matrix(test_data[, oos])
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates (glmnet): 0.775 sec elapsed

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
test_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(test_metric_main, scientific = FALSE),
  Standard_error = format(se_final_test,   scientific = FALSE),
  CI_lower       = format(ci_lower_test,   scientific = FALSE),
  CI_upper       = format(ci_upper_test,   scientific = FALSE),
  CI_length      = format(ci_length_test,  scientific = FALSE)
)
print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower      CI_upper    CI_length
# >       MSE  8155.88290335  150.280725161 7861.3380945 8450.42771223 589.08961777
# >      RMSE    90.30691492    0.831252320   88.6776903   91.93613953   3.25844922
# >       MAE    73.07828110    0.810082479   71.4905486   74.66601359   3.17546497
# >      Bias    -1.36106948    0.839215505   -3.0059016    0.28376269   3.28966433
# >     Bias%     3.24759970    0.184109059    2.8867526    3.60844683   0.72169425
# > R-squared     0.06548008    0.004772479    0.0561262    0.07483397   0.01870777

#### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.

# Helper
evaluate_split <- function(split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           oos, pvmaths, best_lambda_min) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    X     <- as.matrix(split_data[, oos])
    y     <- split_data[[pvmaths[i]]]
    w     <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X, s = best_lambda_min))
    compute_metrics(y_true = y, y_pred = y_pred, w = w)
  }) |> t() |> as.data.frame()
  main_point <- colMeans(main_metrics_df)   # length 6: mse, rmse, mae, bias, bias_pct, r2
  
  # Replicate metrics across PVs
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      X     <- as.matrix(split_data[, oos])
      y     <- split_data[[pvmaths[m]]]
      w     <- split_data[[rep_wts[g]]]
      y_pred <- as.numeric(predict(model, newx = X, s = best_lambda_min))
      compute_metrics(y_true = y, y_pred = y_pred, w = w)
    }) |> t()
  })
  
  # BRR sampling variance, averaged across PVs
  sampling_var_matrix <- sapply(1:M, function(m) {
    sweep(replicate_metrics[[m]], 2, unlist(main_metrics_df[m, ]))^2 |>
      colSums() / (G * (1 - k)^2)
  })
  sampling_var <- rowMeans(sampling_var_matrix)
  
  # Rubin imputation variance across PVs
  imputation_var <- colSums((main_metrics_df - matrix(main_point, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
  
  # Total variance and CIs
  var_final <- sampling_var + (1 + 1/M) * imputation_var
  se_final  <- sqrt(var_final)
  ci_lower  <- main_point - z_crit * se_final
  ci_upper  <- main_point + z_crit * se_final
  ci_length <- ci_upper - ci_lower
  
  tibble::tibble(
    Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
    Point_estimate = format(main_point, scientific = FALSE),
    Standard_error = format(se_final,  scientific = FALSE),
    CI_lower       = format(ci_lower,  scientific = FALSE),
    CI_upper       = format(ci_upper,  scientific = FALSE),
    CI_length      = format(ci_length, scientific = FALSE)
  )
}

# Evaluate on each split
train_eval <- evaluate_split(train_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, oos, pvmaths, best_lambda_min)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, oos, pvmaths, best_lambda_min)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, oos, pvmaths, best_lambda_min)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

#### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                CI_lower                CI_upper            CI_length
# >       MSE 8282.8760413742620585253 95.5821430918978478530 8095.53848334898884787 8470.2135993995361786801 374.675116050547330815
# >      RMSE   91.0091172680310052101  0.5255868984525799981   89.97898587631783585   92.0392486597441745744   2.060262783426338729
# >       MAE   73.0722239979660344034  0.3194079489768992253   72.44619592159550336   73.6982520743365654425   1.252056152741062078
# >      Bias   -0.0000000000006862956  0.0000000000004722203   -0.00000000000161183    0.0000000000002392391   0.000000000001851069
# >     Bias%    3.6129475426387367420  0.0561348988819719980    3.50292516255427389    3.7229699227231995984   0.220044760168925713
# > R-squared    0.0560752990112748789  0.0022198675690237985    0.05172443852553976    0.0604261594970100027   0.008701720971470248

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower      CI_upper    CI_length
# >       MSE  7729.71175399  192.589247910 7352.24376428 8107.17974371 754.93597942
# >      RMSE    87.91343121    1.091276169   85.77456922   90.05229319   4.27772398
# >       MAE    70.55451643    0.930990435   68.72980871   72.37922415   3.64941545
# >      Bias    -2.05257945    1.326146859   -4.65177953    0.54662063   5.19840016
# >     Bias%     2.93823121    0.296416094    2.35726634    3.51919608   1.16192974
# > R-squared     0.05481652    0.006971983    0.04115168    0.06848136   0.02732967

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower      CI_upper    CI_length
# >       MSE  8155.88290335  150.280725161 7861.3380945 8450.42771223 589.08961777
# >      RMSE    90.30691492    0.831252320   88.6776903   91.93613953   3.25844922
# >       MAE    73.07828110    0.810082479   71.4905486   74.66601359   3.17546497
# >      Bias    -1.36106948    0.839215505   -3.0059016    0.28376269   3.28966433
# >     Bias%     3.24759970    0.184109059    2.8867526    3.60844683   0.72169425
# > R-squared     0.06548008    0.004772479    0.0561262    0.07483397   0.01870777

### ---- 1.2) Use best 1se results from "Tuning: grouped = TRUE (Default), keep = TRUE" ----

#### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

tic("Fitting main glmnet models (fixed best_alpha_1se, best_lambda_1se)")
main_models <- lapply(pvmaths, function(pv) {
  
  # TRAIN (final weights)
  X_train <- as.matrix(train_data[, oos])
  y_train <- train_data[[pv]]
  w_train <- train_data[[final_wt]]
  
  # Fit glmnet at chosen alpha + lambda
  mod <- glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    weights = w_train,
    alpha = best_alpha_1se,       # best_alpha_1se = 0
    lambda = best_lambda_1se,     # best_lambda_1se = 1.957261
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda_1se))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(oos, collapse = " + "))),
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha_1se, best_lambda_1se): 1.297 sec elapsed

# Quick look
main_models[[1]]$formula
main_models[[1]]$coefs
# >  (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME
# >  518.4279568  -0.8243642   0.0000000  -5.5092297  -0.6368568

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coefs
# >      (Intercept)  EXERPRAC STUDYHMW   WORKPAY  WORKHOME
# > [1,]    518.4280 -0.8243642 0.00000000 -5.509230 -0.6368568
# > [2,]    519.3445 -1.0838787 0.09993616 -5.213574 -0.6896642
# > [3,]    518.5649 -1.1003287 0.20521631 -5.319143 -0.7507444
# > [4,]    521.3034 -1.2732984 0.12337796 -5.390968 -0.9294886
# > [5,]    519.8803 -1.0524915 0.04781453 -5.409825 -0.8194109
# > [6,]    518.4802 -1.2407730 0.32757517 -5.286345 -0.8933976
# > [7,]    519.9288 -1.2456711 0.00000000 -5.341817 -0.6171155
# > [8,]    517.7751 -1.0925199 0.43756170 -5.537635 -0.7928605
# > [9,]    519.1812 -1.1293223 0.04821176 -5.576214 -0.6333122
# > [10,]   518.6038 -1.1690894 0.31796622 -5.648096 -0.5717779

main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# > 519.1490087  -1.1211737   0.1607660  -5.4232847  -0.7334629

# --- Weighted R² on TRAIN (point estimates per PV) ---
main_r2s_weighted <- sapply(1:M, function(i) {
  model  <- main_models[[i]]$mod
  X_train   <- as.matrix(train_data[, oos])
  y_train <- train_data[[pvmaths[i]]]
  w      <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.04920709

#### ---- Replicate models using BRR replicate weights ----

set.seed(123)

tic("Fitting replicate glmnet models (BRR weights)")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    X_train <- as.matrix(train_data[, oos])
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    mod <- glmnet(
      x = X_train,
      y = y_train,
      family = "gaussian",
      weights = w_train,
      alpha = best_alpha_1se,
      lambda = best_lambda_1se,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda_1se))
    coefs  <- coefs_matrix[, 1]
    names(coefs) <- rownames(coefs_matrix)
    
    list(
      formula = as.formula(paste(pv, "~", paste(oos, collapse = " + "))),
      mod     = mod,
      coefs   = coefs
    )
  })
})
toc()
# > Fitting replicate glmnet models (BRR weights): 1.828 sec elapsed

# Example inspect
replicate_models[[1]][[1]]$formula
replicate_models[[1]][[1]]$coefs

# --- Replicate weighted R² on TRAIN (G x M) (optional diagnostic) ---
rep_r2_weighted <- matrix(NA_real_, nrow = G, ncol = M)
for (m in 1:M) {
  X_train   <- as.matrix(train_data[, oos])
  y_train <- train_data[[pvmaths[m]]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)  # 80 x 10

#### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

# Organize replicate coefficients: rep_coefs[[m]] is G x (p+1) matrix for PV m
rep_coefs <- lapply(replicate_models, function(m) {
  do.call(rbind, lapply(m, function(g) g$coefs))
})

# BRR sampling variance per PV then average across PVs
sampling_var_matrix_coef <- sapply(1:M, function(m) {
  sweep(rep_coefs[[m]], 2, main_coefs[m, ])^2 |>
    colSums() / (G * (1 - k)^2)
})
# Average sampling variance across PVs (Rubin)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef)

# Imputation variance across PVs
imputation_var_coef <- colSums(
  (main_coefs - matrix(main_coef, nrow = M, ncol = length(main_coef), byrow = TRUE))^2
) / (M - 1)

# Total variance & SE for coefficients
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
se_final_coef  <- sqrt(var_final_coef)

# Z-tests (large-sample normal) for coefficients
Estimate   <- main_coef
`Std. Error` <- se_final_coef
`z value`  <- Estimate / `Std. Error`
p_z        <- 2 * pnorm(-abs(`z value`))
`Pr(>|z|)` <- format.pval(p_z, digits = 3, eps = .Machine$double.eps)
z_Signif   <- symnum(p_z, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols = c("***", "**", "*", ".", ""))

# t-test
# library(survey)
# repw <- train_data[, rep_wts]          # matrix/data.frame of replicate weights
# des <- svrepdesign(
#   weights           = ~ W_FSTUWT,
#   repweights        = repw,
#   type              = "Fay",
#   rho               = 0.5,             # Fay’s factor k = 0.5 in PISA
#   combined.weights  = TRUE,
#   mse               = TRUE,
#   data              = train_data
# )
# dof_t <- degf(des)                             # design-based df
# dof_t
# # > [1] 79

dof_t <- G-1
`t value`  <- `z value`                         # same ratio; distribution differs
p_t        <- 2 * pt(-abs(`t value`), df = dof_t)
`Pr(>|t|)` <- format.pval(p_t, digits = 3, eps = .Machine$double.eps)
t_Signif   <- symnum(p_t, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols   = c("***","**","*",".",""))

##### ---- Final Outputs ----
# Coefficient table (glmnet): pooled estimates with BRR+Rubin SE
coef_table <- tibble::tibble(
  Term         = names(Estimate),
  Estimate     = as.numeric(Estimate),
  `Std. Error` = as.numeric(`Std. Error`),
  `z value`    = as.numeric(`z value`),
  `Pr(>|z|)`   = `Pr(>|z|)`,
  z_Signif     = as.character(z_Signif),
  `t value`    = as.numeric(`t value`),
  `Pr(>|t|)`   = `Pr(>|t|)`,
  t_Signif     = as.character(t_Signif),
)
print(as.data.frame(coef_table), row.names = FALSE)
# >        Term   Estimate Std. Error    z value Pr(>|z|) z_Signif   t value Pr(>|t|) t_Signif
# > (Intercept) 519.1490087  1.1090088 468.119848  < 2e-16      *** 468.119848  < 2e-16      ***
# >    EXERPRAC  -1.1211737  0.1410055  -7.951276 1.85e-15      ***  -7.951276 1.08e-11      ***
# >    STUDYHMW   0.1607660  0.1683704   0.954835     0.34            0.954835    0.343         
# >     WORKPAY  -5.4232847  0.1555261 -34.870570  < 2e-16      *** -34.870570  < 2e-16      ***
# >    WORKHOME  -0.7334629  0.1353603  -5.418598 6.01e-08      ***  -5.418598 6.32e-07      ***

# (Optional) R-squared (TRAIN) SE via BRR + Rubin 
sampling_var_r2s_weighted <- sapply(1:M, function(m) {
  sum((rep_r2_weighted[, m] - main_r2s_weighted[m])^2) / (G * (1 - k)^2)
})
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
se_final_r2_weighted  <- sqrt(var_final_r2_weighted)
r2_weighted_table <- tibble::tibble(
  Metric       = "R-squared (Weighted, Train)",
  Estimate     = main_r2_weighted,
  `Std. Error` = se_final_r2_weighted
)
print(as.data.frame(r2_weighted_table), row.names = FALSE)
# >                      Metric    Estimate  Std. Error
# >  R-squared (Weighted, Train) 0.04920709 0.002168203

#### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- as.matrix(train_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
train_metrics_main

# Combine across plausible values
train_metric_main <- colMeans(train_metrics_main)

# --- Replicate predictions for training data ---
tic("Computing train_metrics_replicates (glmnet)")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- train_data[[pvmaths[m]]]
    w <- train_data[[rep_wts[g]]]
    X_train <- as.matrix(train_data[, oos])
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing train_metrics_replicates (glmnet): 1.522 sec elapsed
class(train_metrics_replicates[[1]]); dim(train_metrics_replicates[[1]])
# > [1] "matrix" "array" 
# > [1] 80  6

# BRR sampling variance on TRAIN (vectorized across metrics)
sampling_var_matrix_train <- sapply(1:M, function(m) {
  sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
    colSums() / (G * (1 - k)^2)
})
sampling_var_train <- rowMeans(sampling_var_matrix_train)

# Rubin imputation variance on TRAIN
imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)

# Total variance and SE
var_final_train <- sampling_var_train + (1 + 1/M) * imputation_var_train
se_final_train  <- sqrt(var_final_train)

# Confidence intervals
ci_lower_train  <- train_metric_main - z_crit * se_final_train
ci_upper_train  <- train_metric_main + z_crit * se_final_train
ci_length_train <- ci_upper_train - ci_lower_train

# Format results
train_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train,   scientific = FALSE),
  CI_lower       = format(ci_lower_train,   scientific = FALSE),
  CI_upper       = format(ci_upper_train,   scientific = FALSE),
  CI_length      = format(ci_length_train,  scientific = FALSE)
)
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                CI_lower                 CI_upper              CI_length
# >       MSE 8343.1401800273670232855 95.7500257416604370064 8155.473578054929930659 8530.8067819998050254071 375.333203944875094749
# >      RMSE   91.3396108738002112659  0.5245908845516273900   90.311431633461012325   92.3677901141394102069   2.056358480678397882
# >       MAE   73.3347353945783595464  0.3011926908127218194   72.744408568178712926   73.9250622209780061667   1.180653652799293241
# >      Bias   -0.0000000000006568124  0.0000000000004566251   -0.000000000001551781    0.0000000000002381563   0.000000000001789937
# >     Bias%    3.6718640372129165428  0.0566011334495797880    3.560927854167594830    3.7828002202582382552   0.221872366090643425
# > R-squared    0.0492070885198075819  0.0021682033196068428    0.044957488102217981    0.0534566889373971829   0.008499200835179202

#### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- as.matrix(valid_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
valid_metric_main <- colMeans(valid_metrics_main)

# Replicate predictions on validation set
tic("Computing valid_metrics_replicates (glmnet)")
valid_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- valid_data[[pvmaths[m]]]
    w <- valid_data[[rep_wts[g]]]
    X_valid <- as.matrix(valid_data[, oos])
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_1se))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 0.877 sec elapsed

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
valid_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(valid_metric_main, scientific = FALSE),
  Standard_error = format(se_final_valid,   scientific = FALSE),
  CI_lower       = format(ci_lower_valid,   scientific = FALSE),
  CI_upper       = format(ci_upper_valid,   scientific = FALSE),
  CI_length      = format(ci_length_valid,  scientific = FALSE)
)
print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower      CI_upper    CI_length
# >       MSE  7748.85818594  186.415963501 7383.48961134 8114.22676054 730.73714921
# >      RMSE    88.02262492    1.055311067   85.95425323   90.09099660   4.13674337
# >       MAE    70.61274434    0.862401579   68.92246831   72.30302037   3.38055207
# >      Bias    -1.58783723    1.325492576   -4.18575494    1.01008048   5.19583542
# >     Bias%     3.08653806    0.297153249    2.50412839    3.66894772   1.16481933
# > R-squared     0.05246618    0.005619243    0.04145266    0.06347969   0.02202703

#### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- as.matrix(test_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
test_metric_main <- colMeans(test_metrics_main)

# Replicate predictions on test set
tic("Computing test_metrics_replicates (glmnet)")
test_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- test_data[[pvmaths[m]]]
    w <- test_data[[rep_wts[g]]]
    X_test <- as.matrix(test_data[, oos])
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_1se))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates (glmnet): 0.78 sec elapsed

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
test_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(test_metric_main, scientific = FALSE),
  Standard_error = format(se_final_test,   scientific = FALSE),
  CI_lower       = format(ci_lower_test,   scientific = FALSE),
  CI_upper       = format(ci_upper_test,   scientific = FALSE),
  CI_length      = format(ci_length_test,  scientific = FALSE)
)
print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower      CI_upper    CI_length
# >       MSE  8255.50204089  153.140785216 7955.35161730 8555.6524645 600.30084718
# >      RMSE    90.85672543    0.841560414   89.20729733   92.5061535   3.29885621
# >       MAE    73.55809692    0.791025932   72.00771458   75.1084793   3.10076467
# >      Bias    -0.97260966    0.839285298   -2.61757861    0.6723593   3.28993791
# >     Bias%     3.39662894    0.184485314    3.03504437    3.7582135   0.72316914
# > R-squared     0.05407028    0.003905287    0.04641606    0.0617245   0.01530844

#### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.

# Helper
evaluate_split <- function(split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           oos, pvmaths, best_lambda_1se) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    X     <- as.matrix(split_data[, oos])
    y     <- split_data[[pvmaths[i]]]
    w     <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X, s = best_lambda_1se))
    compute_metrics(y_true = y, y_pred = y_pred, w = w)
  }) |> t() |> as.data.frame()
  main_point <- colMeans(main_metrics_df)   # length 6: mse, rmse, mae, bias, bias_pct, r2
  
  # Replicate metrics across PVs
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      X     <- as.matrix(split_data[, oos])
      y     <- split_data[[pvmaths[m]]]
      w     <- split_data[[rep_wts[g]]]
      y_pred <- as.numeric(predict(model, newx = X, s = best_lambda_1se))
      compute_metrics(y_true = y, y_pred = y_pred, w = w)
    }) |> t()
  })
  
  # BRR sampling variance, averaged across PVs
  sampling_var_matrix <- sapply(1:M, function(m) {
    sweep(replicate_metrics[[m]], 2, unlist(main_metrics_df[m, ]))^2 |>
      colSums() / (G * (1 - k)^2)
  })
  sampling_var <- rowMeans(sampling_var_matrix)
  
  # Rubin imputation variance across PVs
  imputation_var <- colSums((main_metrics_df - matrix(main_point, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
  
  # Total variance and CIs
  var_final <- sampling_var + (1 + 1/M) * imputation_var
  se_final  <- sqrt(var_final)
  ci_lower  <- main_point - z_crit * se_final
  ci_upper  <- main_point + z_crit * se_final
  ci_length <- ci_upper - ci_lower
  
  tibble::tibble(
    Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
    Point_estimate = format(main_point, scientific = FALSE),
    Standard_error = format(se_final,  scientific = FALSE),
    CI_lower       = format(ci_lower,  scientific = FALSE),
    CI_upper       = format(ci_upper,  scientific = FALSE),
    CI_length      = format(ci_length, scientific = FALSE)
  )
}

# Evaluate on each split
train_eval <- evaluate_split(train_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, oos, pvmaths, best_lambda_1se)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, oos, pvmaths, best_lambda_1se)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, oos, pvmaths, best_lambda_1se)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

#### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                CI_lower                 CI_upper              CI_length
# >       MSE 8343.1401800273670232855 95.7500257416604370064 8155.473578054929930659 8530.8067819998050254071 375.333203944875094749
# >      RMSE   91.3396108738002112659  0.5245908845516273900   90.311431633461012325   92.3677901141394102069   2.056358480678397882
# >       MAE   73.3347353945783595464  0.3011926908127218194   72.744408568178712926   73.9250622209780061667   1.180653652799293241
# >      Bias   -0.0000000000006568124  0.0000000000004566251   -0.000000000001551781    0.0000000000002381563   0.000000000001789937
# >     Bias%    3.6718640372129165428  0.0566011334495797880    3.560927854167594830    3.7828002202582382552   0.221872366090643425
# > R-squared    0.0492070885198075819  0.0021682033196068428    0.044957488102217981    0.0534566889373971829   0.008499200835179202


print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower      CI_upper    CI_length
# >       MSE  7748.85818594  186.415963501 7383.48961134 8114.22676054 730.73714921
# >      RMSE    88.02262492    1.055311067   85.95425323   90.09099660   4.13674337
# >       MAE    70.61274434    0.862401579   68.92246831   72.30302037   3.38055207
# >      Bias    -1.58783723    1.325492576   -4.18575494    1.01008048   5.19583542
# >     Bias%     3.08653806    0.297153249    2.50412839    3.66894772   1.16481933
# > R-squared     0.05246618    0.005619243    0.04145266    0.06347969   0.02202703

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower      CI_upper    CI_length
# >       MSE  8255.50204089  153.140785216 7955.35161730 8555.6524645 600.30084718
# >      RMSE    90.85672543    0.841560414   89.20729733   92.5061535   3.29885621
# >       MAE    73.55809692    0.791025932   72.00771458   75.1084793   3.10076467
# >      Bias    -0.97260966    0.839285298   -2.61757861    0.6723593   3.28993791
# >     Bias%     3.39662894    0.184485314    3.03504437    3.7582135   0.72316914
# > R-squared     0.05407028    0.003905287    0.04641606    0.0617245   0.01530844
