# ---- I. Predictive Modelling: Version 1.5 ----

# Fit with glmnet: alpha α (=1, default), lambda λ (grid), with manual K-folds + tracks out-of-fold (OOF) predictions/metrics.

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

## ---- Main model using final student weights (W_FSTUWT) ---- 

### ---- Fit Elastic Net for PV1MATH only: cross-validation with glmnet ----

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

# Outcome distribution by fold
print(do.call(rbind, tapply(y_train, foldid, function(v) c(mean = mean(v), sd = sd(v)))))
# >       mean       sd
# > 1 489.4925 92.81556
# > 2 492.0924 93.06908
# > 3 489.7786 90.84370
# > 4 492.2019 92.46838
# > 5 490.3393 92.04746

# --- Fit on full TRAIN to fix the lambda path (matches cv.glmnet alignment="lambda") ---
tic("glmnet fit (full TRAIN)")
fit <- glmnet(
  x = X_train, 
  y = y_train,
  family = "gaussian",
  weights = w_train,
  alpha = 1,
  standardize = TRUE,
  intercept   = TRUE
)
toc()
# > glmnet fit (full TRAIN): 0.005 sec elapsed

# --- Manual CV using cv_folds; collect OOF predictions and fold MSEs ---
oof_pred <- matrix(NA_real_, nrow = nrow(X_train), ncol = length(fit$lambda))
fold_mse <- matrix(NA_real_, nrow = length(cv_folds), ncol = length(fit$lambda))

tic("Manual K-fold CV (glmnet only, OOF tracking)")
for (k in seq_along(cv_folds)) {
  idx_valid <- cv_folds[[k]]
  idx_train  <- setdiff(seq_len(nrow(X_train)), idx_valid)
  
  fit_k <- glmnet(
    x = X_train[idx_train, , drop = FALSE],
    y = y_train[idx_train],
    family   = "gaussian",
    weights  = w_train[idx_train],
    alpha    = 1,                         # Default
    lambda  = fit$lambda,                 # align fold paths to λ’s
    standardize = TRUE,                   # Default
    intercept   = TRUE                    # Default
  )
  
  pred_k <- predict(fit_k, newx = X_train[idx_valid, , drop = FALSE], s = fit$lambda)
  oof_pred[idx_valid, ] <- pred_k
  
  wk <- w_train[idx_valid]; yk <- y_train[idx_valid]
  fold_mse[k, ] <- colSums(wk * (yk - pred_k)^2) / sum(wk)
}
toc()
# > Manual K-fold CV (glmnet only, OOF tracking): 0.044 sec elapsed

# --- CV error summaries (two ways) + λ selection ---

# One-liner normalized fold weights (weighted by within-fold weight totals)
w_fold_norm <- tapply(w_train, foldid, sum) / sum(w_train)

# "grouped=TRUE" (fold means with fold weights)
cvm_grouped <- as.numeric(t(fold_mse) %*% w_fold_norm)

# Weighted SE across folds 
cvsd_grouped <- sqrt(
  colSums(sweep(sweep(fold_mse, 2, cvm_grouped, "-")^2, 1, w_fold_norm, "*") ) / (length(cv_folds) - 1)
)

# "grouped=FALSE" (observation-level aggregation of OOF residuals)
cvm_ungrouped <- colSums(w_train * (y_train - oof_pred)^2) / sum(w_train)
cvsd_ungrouped <- sqrt(
  colSums( w_train * (sweep((y_train - oof_pred)^2, 2, cvm_ungrouped, "-")^2) ) /
    sum(w_train) / (nrow(X_train) - 1)
)

# Means agree for squared error; SE differs by construction
stopifnot(isTRUE(all.equal(cvm_grouped, cvm_ungrouped, tolerance = 1e-12)))

# λ.min (same for grouped/ungrouped here)
cvmin          <- min(cvm_grouped)
lambda_min     <- max(fit$lambda[cvm_grouped <= cvmin])
idx_min        <- match(lambda_min, fit$lambda)

# 1-SE rules (both ways)
lambda_1se_grouped   <- max(fit$lambda[cvm_grouped   <= (cvm_grouped[idx_min]   + cvsd_grouped[idx_min])])
lambda_1se_ungrouped <- max(fit$lambda[cvm_ungrouped <= (cvm_ungrouped[idx_min] + cvsd_ungrouped[idx_min])])

cat(sprintf("\nSelected lambdas →  lambda.min = %.6f,  lambda.1se(grouped) = %.6f,  lambda.1se(ungrouped) = %.6f\n\n",
            lambda_min, lambda_1se_grouped, lambda_1se_ungrouped))
# > Selected lambdas →  lambda.min = 0.067143,  lambda.1se(grouped) = 4.025131,  lambda.1se(ungrouped) = 5.839776

### ---- Predict and evaluate performance on training/validation/test datasets ----

# Predict with lambda.min
pred_train_min <- as.numeric(predict(fit, newx = X_train, s = lambda_min))
pred_valid_min <- as.numeric(predict(fit, newx = X_valid, s = lambda_min))
pred_test_min  <- as.numeric(predict(fit, newx = X_test,  s = lambda_min))

metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

metric_results_min <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"], metrics_valid_min["rmse"], metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],  metrics_valid_min["mae"],  metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"], metrics_valid_min["bias"], metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],   metrics_valid_min["r2"],   metrics_test_min["r2"])
)

# Predict with lambda.1se (grouped)
pred_train_1se_grouped <- as.numeric(predict(fit, newx = X_train, s = lambda_1se_grouped))
pred_valid_1se_grouped <- as.numeric(predict(fit, newx = X_valid, s = lambda_1se_grouped))
pred_test_1se_grouped  <- as.numeric(predict(fit, newx = X_test,  s = lambda_1se_grouped))

metrics_train_1se_grouped <- compute_metrics(y_train, pred_train_1se_grouped, w_train)
metrics_valid_1se_grouped <- compute_metrics(y_valid, pred_valid_1se_grouped, w_valid)
metrics_test_1se_grouped  <- compute_metrics(y_test,  pred_test_1se_grouped,  w_test)

metric_results_1se_grouped <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se_grouped["rmse"], metrics_valid_1se_grouped["rmse"], metrics_test_1se_grouped["rmse"]),
  MAE     = c(metrics_train_1se_grouped["mae"],  metrics_valid_1se_grouped["mae"],  metrics_test_1se_grouped["mae"]),
  Bias    = c(metrics_train_1se_grouped["bias"], metrics_valid_1se_grouped["bias"], metrics_test_1se_grouped["bias"]),
  `Bias%` = c(metrics_train_1se_grouped["bias_pct"], metrics_valid_1se_grouped["bias_pct"], metrics_test_1se_grouped["bias_pct"]),
  R2      = c(metrics_train_1se_grouped["r2"],   metrics_valid_1se_grouped["r2"],   metrics_test_1se_grouped["r2"])
)

# Predict with lambda.1se (ungrouped)
pred_train_1se_ungrouped <- as.numeric(predict(fit, newx = X_train, s = lambda_1se_ungrouped))
pred_valid_1se_ungrouped <- as.numeric(predict(fit, newx = X_valid, s = lambda_1se_ungrouped))
pred_test_1se_ungrouped  <- as.numeric(predict(fit, newx = X_test,  s = lambda_1se_ungrouped))

metrics_train_1se_ungrouped <- compute_metrics(y_train, pred_train_1se_ungrouped, w_train)
metrics_valid_1se_ungrouped <- compute_metrics(y_valid, pred_valid_1se_ungrouped, w_valid)
metrics_test_1se_ungrouped  <- compute_metrics(y_test,  pred_test_1se_ungrouped,  w_test)

metric_results_1se_ungrouped <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se_ungrouped["rmse"], metrics_valid_1se_ungrouped["rmse"], metrics_test_1se_ungrouped["rmse"]),
  MAE     = c(metrics_train_1se_ungrouped["mae"],  metrics_valid_1se_ungrouped["mae"],  metrics_test_1se_ungrouped["mae"]),
  Bias    = c(metrics_train_1se_ungrouped["bias"], metrics_valid_1se_ungrouped["bias"], metrics_test_1se_ungrouped["bias"]),
  `Bias%` = c(metrics_train_1se_ungrouped["bias_pct"], metrics_valid_1se_ungrouped["bias_pct"], metrics_test_1se_ungrouped["bias_pct"]),
  R2      = c(metrics_train_1se_ungrouped["r2"],   metrics_valid_1se_ungrouped["r2"],   metrics_test_1se_ungrouped["r2"])
)

print(as.data.frame(metric_results_min),           row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051
print(as.data.frame(metric_results_1se_grouped),   row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.41481 73.55493 -3.269085e-13 3.664657 0.04631327
# > Validation 88.23838 70.74959 -2.525736e+00 2.901086 0.04767726
# >       Test 92.03300 74.28495 -6.772109e-01 3.577457 0.05398446
print(as.data.frame(metric_results_1se_ungrouped), row.names = FALSE)
# >   Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.61485 73.76314 -2.953923e-13 3.685288 0.04213472
# > Validation 88.40407 70.89232 -2.356312e+00 2.956835 0.04409728
# >       Test 92.28553 74.54139 -5.943704e-01 3.618750 0.04878589



