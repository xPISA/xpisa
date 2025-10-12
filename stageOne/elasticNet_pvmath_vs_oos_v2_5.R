# ---- II. Predictive Modelling: Version 2.5 ----

# Tune hyperparameters (alpha α and lambda λ) using glmnet, with manual K-folds + tracks out-of-fold (OOF) predictions/metrics.

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

# --- Tuning ---
alpha_grid <- seq(0, 1, by = 0.1)   # 0=ridge, 1=lasso

# Storage
mod_list       <- vector("list", length(alpha_grid))  # glmnet path per alpha (on full TRAIN)
per_alpha_list <- vector("list", length(alpha_grid))  # tidy rows per alpha for rule-based selection

# Precompute normalized fold weights for grouped summaries
w_fold_norm <- tapply(w_train, foldid, sum) / sum(w_train)

tic("Grid over alpha (glmnet-only manual 5-fold CV with OOF tracking)")
for (i in seq_along(alpha_grid)) {
  alpha <- alpha_grid[i]
  message(sprintf("alpha = %.1f — fitting TRAIN path & doing manual CV (interpolated to master λ) ...", alpha))
  
  # ---- 1) Fit on full TRAIN to fix the reference λ grid (regularization path) ----
  fit <- glmnet(
    x = X_train, 
    y = y_train,
    family = "gaussian",
    weights = w_train,
    alpha = alpha,
    standardize = TRUE,
    intercept   = TRUE
  )
  mod_list[[i]] <- fit
  
  # ---- 2) Manual CV over the fixed reference λ grid; track OOF preds ----
  oof_pred <- matrix(NA_real_, nrow = n_cv, ncol = length(fit$lambda))       # fit$lambda: reference λ grid (from full‑train fit)
  fold_mse <- matrix(NA_real_, nrow = num_folds, ncol = length(fit$lambda))
  
  for (k in seq_along(cv_folds)) {
    idx_valid <- cv_folds[[k]]
    idx_train_k <- setdiff(seq_len(n_cv), idx_valid)
    
    # No 'lambda = fit$lambda' here; let the fold pick its path (cv.glmnet behavior)
    fit_k <- glmnet(
      x = X_train[idx_train_k, , drop = FALSE],
      y = y_train[idx_train_k],
      family   = "gaussian",
      weights  = w_train[idx_train_k],
      alpha    = alpha,
      standardize = TRUE,
      intercept   = TRUE
    )
    
    # Align fold predictions to the reference λ grid (like cv.glmnet alignment="lambda")
    pred_k <- predict(
      fit_k,
      newx = X_train[idx_valid, , drop = FALSE],
      s    = fit$lambda,
      exact = FALSE
    )
    oof_pred[idx_valid, ] <- pred_k
    
    wk <- w_train[idx_valid]; yk <- y_train[idx_valid]
    fold_mse[k, ] <- colSums(wk * (yk - pred_k)^2) / sum(wk)
  }
  
  # ---- 3) CV summaries: grouped (fold-weighted) & ungrouped (row-level) ----
  # grouped (matches cv.glmnet grouped=TRUE): fold-weighted mean MSE
  cvm_grouped <- as.numeric(t(fold_mse) %*% w_fold_norm)
  # weighted sample SD across folds (cv.glmnet-style scale)
  cvsd_grouped <- sqrt(
    colSums( sweep(sweep(fold_mse, 2, cvm_grouped, "-")^2, 1, w_fold_norm, "*") ) /
      (num_folds - 1)
  )
  
  # ungrouped (matches cv.glmnet grouped=FALSE): pooled OOF over rows
  res2_oof       <- (y_train - oof_pred)^2
  cvm_ungrouped  <- colSums(w_train * res2_oof) / sum(w_train)
  cvsd_ungrouped <- sqrt(
    colSums( w_train * (sweep(res2_oof, 2, cvm_ungrouped, "-")^2) ) / sum(w_train) / (n_cv - 1)
  )
  
  # Means agreement check
  stopifnot(isTRUE(all.equal(cvm_grouped, cvm_ungrouped)))
  
  # ---- 4) Rule-based λ selection at this α (tie-breaks toward larger λ) ----
  # λ.min
  cvmin        <- min(cvm_grouped, na.rm = TRUE)
  lambda_min   <- max(fit$lambda[cvm_grouped <= cvmin + 0])
  idx_min      <- match(lambda_min, fit$lambda)
  
  # 1‑SE rules
  lambda_1se_grouped   <- max(fit$lambda[cvm_grouped   <= (cvm_grouped[idx_min]   + cvsd_grouped[idx_min])])
  lambda_1se_ungrouped <- max(fit$lambda[cvm_ungrouped <= (cvm_ungrouped[idx_min] + cvsd_ungrouped[idx_min])])
  
  # Tidy rows for selection across α (use df as nzero)
  pick <- function(l) match(l, fit$lambda)
  per_alpha_list[[i]] <- tibble::tibble(
    alpha       = alpha,
    alpha_idx   = i,
    s           = c("lambda.min", "lambda.1se_grouped", "lambda.1se_ungrouped"),
    lambda      = c(lambda_min,   lambda_1se_grouped,   lambda_1se_ungrouped),
    lambda_idx  = c(pick(lambda_min), pick(lambda_1se_grouped), pick(lambda_1se_ungrouped)),
    nzero       = fit$df[c(pick(lambda_min), pick(lambda_1se_grouped), pick(lambda_1se_ungrouped))],
    dev_ratio   = as.numeric(fit$dev.ratio[c(pick(lambda_min), pick(lambda_1se_grouped), pick(lambda_1se_ungrouped))]),
    cvm_grouped = c(cvm_grouped[pick(lambda_min)],
                    cvm_grouped[pick(lambda_1se_grouped)],
                    cvm_grouped[pick(lambda_1se_ungrouped)]),
    cvsd_grouped = c(cvsd_grouped[pick(lambda_min)],
                     cvsd_grouped[pick(lambda_1se_grouped)],
                     cvsd_grouped[pick(lambda_1se_ungrouped)]),
    cvm_ungrouped = c(cvm_ungrouped[pick(lambda_min)],
                      cvm_ungrouped[pick(lambda_1se_grouped)],
                      cvm_ungrouped[pick(lambda_1se_ungrouped)]),
    cvsd_ungrouped = c(cvsd_ungrouped[pick(lambda_min)],
                       cvsd_ungrouped[pick(lambda_1se_grouped)],
                       cvsd_ungrouped[pick(lambda_1se_ungrouped)])
  )
}
toc()
# > Grid over alpha (glmnet-only manual 5-fold CV with OOF tracking): 1.855 sec elapsed

##### ---- Explore the tuning results ----
tuning_results <- dplyr::bind_rows(per_alpha_list)

# Peek at top candidates by grouped CVM regardless of rule
tuning_results %>%
  dplyr::arrange(cvm_grouped, nzero, dplyr::desc(lambda), alpha) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx                    s      lambda lambda_idx nzero  dev_ratio cvm_grouped cvsd_grouped cvm_ungrouped cvsd_ungrouped
# >   0.0         1           lambda.min  1.95726075        100     4 0.05352143    8310.048     58.19281      8310.048       94.66112
# >   0.1         2           lambda.min  0.73689611         61     4 0.05353111    8310.258     58.24930      8310.258       94.66966
# >   0.2         3           lambda.min  0.25395695         65     4 0.05353768    8310.297     58.28652      8310.297       94.67795
# >   0.3         4           lambda.min  0.18581163         64     4 0.05353779    8310.312     58.28966      8310.312       94.67837
# >   0.5         6           lambda.min  0.12235682         63     4 0.05353783    8310.317     58.29708      8310.317       94.67812
# >   0.4         5           lambda.min  0.15294602         63     4 0.05353765    8310.321     58.29005      8310.321       94.67826
# >   0.7         8           lambda.min  0.09591890         62     4 0.05353760    8310.335     58.29203      8310.335       94.67842
# >   0.8         9           lambda.min  0.08392904         62     4 0.05353767    8310.337     58.29291      8310.337       94.67859
# >   0.9        10           lambda.min  0.07460359         62     4 0.05353773    8310.339     58.29360      8310.339       94.67871
# >   0.6         7           lambda.min  0.11190538         62     4 0.05353751    8310.339     58.29257      8310.339       94.67836
# >   1.0        11           lambda.min  0.06714323         62     4 0.05353777    8310.341     58.29415      8310.341       94.67881
# >   0.8         9   lambda.1se_grouped  4.58443661         19     3 0.04684211    8362.782     58.05622      8362.782       94.82459
# >   0.2         3   lambda.1se_grouped 15.22431926         21     4 0.04702745    8363.283     59.17683      8363.283       94.81821
# >   0.7         8   lambda.1se_grouped  5.23935613         19     3 0.04675936    8363.425     58.07059      8363.425       94.82550
# >   0.6         7   lambda.1se_grouped  6.11258215         19     3 0.04664677    8364.305     58.09043      8364.305       94.82701
# >   0.3         4   lambda.1se_grouped 11.13911439         20     4 0.04676437    8364.329     58.65193      8364.329       94.82794
# >   0.0         1   lambda.1se_grouped 61.17840939         63     4 0.04648180    8364.969     59.27289      8364.969       94.74087
# >   0.1         2   lambda.1se_grouped 25.27899462         23     4 0.04679828    8365.288     59.82835      8365.288       94.80813
# >   0.5         6   lambda.1se_grouped  7.33509858         19     3 0.04648516    8365.576     58.11956      8365.576       94.82964
# >   0.4         5   lambda.1se_grouped  9.16887322         19     3 0.04623502    8367.558     58.16429      8367.558       94.83462
# >   1.0        11   lambda.1se_grouped  4.02513082         18     3 0.04631327    8368.111     57.47850      8368.111       94.83666
# >   0.9        10   lambda.1se_grouped  4.47236757         18     3 0.04625534    8368.550     57.49203      8368.550       94.83739
# >   0.7         8 lambda.1se_ungrouped  7.60140865         15     3 0.04314506    8395.434     56.64284      8395.434       94.91873
# >   0.2         3 lambda.1se_ungrouped 22.08788051         17     3 0.04284309    8396.383     57.76041      8396.383       94.94124
# >   0.0         1 lambda.1se_ungrouped 88.75939696         59     4 0.04270578    8396.604     59.76914      8396.604       94.92529
# >   0.3         4 lambda.1se_ungrouped 16.16094773         16     3 0.04287335    8396.763     57.22251      8396.763       94.93250
# >   0.6         7 lambda.1se_ungrouped  8.86831009         15     3 0.04295050    8396.964     56.67157      8396.964       94.92544
# >   0.5         6 lambda.1se_ungrouped 10.64197211         15     3 0.04267323    8399.161     56.71596      8399.161       94.93561
# >   0.1         2 lambda.1se_ungrouped 36.67549289         19     3 0.04203576    8402.123     58.79676      8402.123       94.99349
# >   0.4         5 lambda.1se_ungrouped 13.30246514         15     3 0.04224893    8402.547     56.78701      8402.547       94.95232
# >   1.0        11 lambda.1se_ungrouped  5.83977563         14     3 0.04213472    8402.898     56.97637      8402.898       94.94526
# >   0.9        10 lambda.1se_ungrouped  6.48863958         14     3 0.04203642    8403.758     56.95438      8403.758       94.94940
# >   0.8         9 lambda.1se_ungrouped  7.29971953         14     3 0.04191219    8404.844     56.92913      8404.844       94.95474

# --- Select winners across α for each rule ---
best_min <- tuning_results %>%
  dplyr::filter(s == "lambda.min") %>%
  dplyr::arrange(cvm_grouped, nzero, dplyr::desc(lambda), alpha) %>%
  dplyr::slice(1)

best_1se_grouped <- tuning_results %>%
  dplyr::filter(s == "lambda.1se_grouped") %>%
  dplyr::arrange(cvm_grouped, nzero, dplyr::desc(lambda), alpha) %>%
  dplyr::slice(1)

best_1se_ungrouped <- tuning_results %>%
  dplyr::filter(s == "lambda.1se_ungrouped") %>%
  dplyr::arrange(cvm_ungrouped, nzero, dplyr::desc(lambda), alpha) %>%
  dplyr::slice(1)

print(as.data.frame(best_min),  row.names = FALSE)
# > alpha alpha_idx          s   lambda lambda_idx nzero  dev_ratio cvm_grouped cvsd_grouped cvm_ungrouped cvsd_ungrouped
# >     0         1 lambda.min 1.957261        100     4 0.05352143    8310.048     58.19281      8310.048       94.66112
print(as.data.frame(best_1se_grouped), row.names = FALSE)
# > alpha alpha_idx                  s   lambda lambda_idx nzero  dev_ratio cvm_grouped cvsd_grouped cvm_ungrouped cvsd_ungrouped
# >   0.8         9 lambda.1se_grouped 4.584437         19     3 0.04684211    8362.782     58.05622      8362.782       94.82459
print(as.data.frame(best_1se_ungrouped), row.names = FALSE)
# > alpha alpha_idx                    s   lambda lambda_idx nzero  dev_ratio cvm_grouped cvsd_grouped cvm_ungrouped cvsd_ungrouped
# >   0.7         8 lambda.1se_ungrouped 7.601409         15     3 0.04314506    8395.434     56.64284      8395.434       94.91873

##### ---- Predict & evaluate on TRAIN / VALID / TEST for winners ----
best_alpha_idx_min           <- best_min$alpha_idx
best_lambda_min              <- best_min$lambda

best_alpha_idx_1se_grouped   <- best_1se_grouped$alpha_idx
best_lambda_1se_grouped      <- best_1se_grouped$lambda

best_alpha_idx_1se_ungrouped <- best_1se_ungrouped$alpha_idx
best_lambda_1se_ungrouped    <- best_1se_ungrouped$lambda

mod_best_min           <- mod_list[[best_alpha_idx_min]]
mod_best_1se_grouped   <- mod_list[[best_alpha_idx_1se_grouped]]
mod_best_1se_ungrouped <- mod_list[[best_alpha_idx_1se_ungrouped]]

# Coefficients & importances
coef(mod_best_min,           s = best_lambda_min)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             s=1.957261
# > (Intercept) 521.236974
# > EXERPRAC     -1.855744
# > STUDYHMW      1.891380
# > WORKPAY      -6.361346
# > WORKHOME     -1.812967
coef(mod_best_1se_grouped,   s = best_lambda_1se_grouped)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >              s=4.584437
# > (Intercept) 518.4288701
# > EXERPRAC     -0.8246734
# > STUDYHMW      .        
# > WORKPAY      -5.5091581
# > WORKHOME     -0.6367917
coef(mod_best_1se_ungrouped, s = best_lambda_1se_ungrouped)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >              s=7.601409
# > (Intercept) 514.6421790
# > EXERPRAC     -0.5139624
# > STUDYHMW      .        
# > WORKPAY      -5.0020993
# > WORKHOME     -0.3038639

caret::varImp(mod_best_min,           lambda = best_lambda_min)
# >           Overall
# > EXERPRAC 1.855744
# > STUDYHMW 1.891380
# > WORKPAY  6.361346
# > WORKHOME 1.812967
caret::varImp(mod_best_1se_grouped,   lambda = best_lambda_1se_grouped)
# >            Overall
# > EXERPRAC 0.8246734
# > STUDYHMW 0.0000000
# > WORKPAY  5.5091581
# > WORKHOME 0.6367917
caret::varImp(mod_best_1se_ungrouped, lambda = best_lambda_1se_ungrouped)
# >            Overall
# > EXERPRAC 0.5139624
# > STUDYHMW 0.0000000
# > WORKPAY  5.0020993
# > WORKHOME 0.3038639

# --- lambda.min ---
pred_train_min <- as.numeric(predict(mod_best_min, newx = X_train, s = best_lambda_min))
pred_valid_min <- as.numeric(predict(mod_best_min, newx = X_valid, s = best_lambda_min))
pred_test_min  <- as.numeric(predict(mod_best_min,  newx = X_test,  s = best_lambda_min))

metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

metric_results_min <- tibble::tibble(
  Rule    = "lambda.min",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"], metrics_valid_min["rmse"], metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],  metrics_valid_min["mae"],  metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"], metrics_valid_min["bias"], metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],   metrics_valid_min["r2"],   metrics_test_min["r2"])
)
print(as.data.frame(metric_results_min), row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.min   Training 91.06868 73.19982 -3.758605e-13 3.606879 0.05352143
# > lambda.min Validation 88.00754 70.62964 -3.015818e+00 2.744678 0.05265334
# > lambda.min       Test 91.32308 73.70615 -1.073580e+00 3.420891 0.06852296

# --- lambda.1se (grouped) ---
pred_train_1se_grouped <- as.numeric(predict(mod_best_1se_grouped, newx = X_train, s = best_lambda_1se_grouped))
pred_valid_1se_grouped <- as.numeric(predict(mod_best_1se_grouped, newx = X_valid, s = best_lambda_1se_grouped))
pred_test_1se_grouped  <- as.numeric(predict(mod_best_1se_grouped,  newx = X_test,  s = best_lambda_1se_grouped))

metrics_train_1se_grouped <- compute_metrics(y_train, pred_train_1se_grouped, w_train)
metrics_valid_1se_grouped <- compute_metrics(y_valid, pred_valid_1se_grouped, w_valid)
metrics_test_1se_grouped  <- compute_metrics(y_test,  pred_test_1se_grouped,  w_test)

metric_results_1se_grouped <- tibble::tibble(
  Rule    = "lambda.1se_grouped",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se_grouped["rmse"], metrics_valid_1se_grouped["rmse"], metrics_test_1se_grouped["rmse"]),
  MAE     = c(metrics_train_1se_grouped["mae"],  metrics_valid_1se_grouped["mae"],  metrics_test_1se_grouped["mae"]),
  Bias    = c(metrics_train_1se_grouped["bias"], metrics_valid_1se_grouped["bias"], metrics_test_1se_grouped["bias"]),
  `Bias%` = c(metrics_train_1se_grouped["bias_pct"], metrics_valid_1se_grouped["bias_pct"], metrics_test_1se_grouped["bias_pct"]),
  R2      = c(metrics_train_1se_grouped["r2"],   metrics_valid_1se_grouped["r2"],   metrics_test_1se_grouped["r2"])
)
print(as.data.frame(metric_results_1se_grouped), row.names = FALSE)
# >               Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.1se_grouped   Training 91.38946 73.52732 -3.623010e-13 3.661776 0.04684211
# > lambda.1se_grouped Validation 88.21869 70.73364 -2.553865e+00 2.892379 0.04810221
# > lambda.1se_grouped       Test 92.00023 74.25070 -6.904099e-01 3.571384 0.05465809

# --- lambda.1se (ungrouped) ---
pred_train_1se_ungrouped <- as.numeric(predict(mod_best_1se_ungrouped, newx = X_train, s = best_lambda_1se_ungrouped))
pred_valid_1se_ungrouped <- as.numeric(predict(mod_best_1se_ungrouped, newx = X_valid, s = best_lambda_1se_ungrouped))
pred_test_1se_ungrouped  <- as.numeric(predict(mod_best_1se_ungrouped,  newx = X_test,  s = best_lambda_1se_ungrouped))

metrics_train_1se_ungrouped <- compute_metrics(y_train, pred_train_1se_ungrouped, w_train)
metrics_valid_1se_ungrouped <- compute_metrics(y_valid, pred_valid_1se_ungrouped, w_valid)
metrics_test_1se_ungrouped  <- compute_metrics(y_test,  pred_test_1se_ungrouped,  w_test)

metric_results_1se_ungrouped <- tibble::tibble(
  Rule    = "lambda.1se_ungrouped",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se_ungrouped["rmse"], metrics_valid_1se_ungrouped["rmse"], metrics_test_1se_ungrouped["rmse"]),
  MAE     = c(metrics_train_1se_ungrouped["mae"],  metrics_valid_1se_ungrouped["mae"],  metrics_test_1se_ungrouped["mae"]),
  Bias    = c(metrics_train_1se_ungrouped["bias"], metrics_valid_1se_ungrouped["bias"], metrics_test_1se_ungrouped["bias"]),
  `Bias%` = c(metrics_train_1se_ungrouped["bias_pct"], metrics_valid_1se_ungrouped["bias_pct"], metrics_test_1se_ungrouped["bias_pct"]),
  R2      = c(metrics_train_1se_ungrouped["r2"],   metrics_valid_1se_ungrouped["r2"],   metrics_test_1se_ungrouped["r2"])
)
print(as.data.frame(metric_results_1se_ungrouped), row.names = FALSE)
# >                 Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.1se_ungrouped   Training 91.56652 73.71684 -2.914863e-13 3.681987 0.04314506
# > lambda.1se_ungrouped Validation 88.36183 70.85843 -2.394296e+00 2.945672 0.04501063
# > lambda.1se_ungrouped       Test 92.22802 74.48796 -6.115866e-01 3.611393 0.04997108

# ---- Combined view (same print style as earlier versions)
print(as.data.frame(dplyr::bind_rows(
  metric_results_min,
  metric_results_1se_grouped,
  metric_results_1se_ungrouped
)), row.names = FALSE)
# >                 Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# >           lambda.min   Training 91.06868 73.19982 -3.758605e-13 3.606879 0.05352143
# >           lambda.min Validation 88.00754 70.62964 -3.015818e+00 2.744678 0.05265334
# >           lambda.min       Test 91.32308 73.70615 -1.073580e+00 3.420891 0.06852296
# >   lambda.1se_grouped   Training 91.38946 73.52732 -3.623010e-13 3.661776 0.04684211
# >   lambda.1se_grouped Validation 88.21869 70.73364 -2.553865e+00 2.892379 0.04810221
# >   lambda.1se_grouped       Test 92.00023 74.25070 -6.904099e-01 3.571384 0.05465809
# > lambda.1se_ungrouped   Training 91.56652 73.71684 -2.914863e-13 3.681987 0.04314506
# > lambda.1se_ungrouped Validation 88.36183 70.85843 -2.394296e+00 2.945672 0.04501063
# > lambda.1se_ungrouped       Test 92.22802 74.48796 -6.115866e-01 3.611393 0.04997108
