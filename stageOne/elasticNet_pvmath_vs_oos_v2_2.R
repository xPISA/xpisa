# ---- II. Predictive Modelling: Version 2.2 ----

# Tune hyperparameters (both alpha α and lambda λ) using cv.glmnet

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

# Grid over elastic‑net mixing parameter
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso
# options: alpha_grid <- seq(0, 1, by = 0.05), alpha_grid <- sort(unique(c(seq(0, 1, by = 0.1), 0.001, 0.005, 0.01, 0.05))), etc. 

# Storage
cv_list        <- vector("list", length(alpha_grid))  # cv.glmnet fits per alpha
per_alpha_list <- vector("list", length(alpha_grid))  # two rows per alpha: lambda.min & lambda.1se

# Parallel backend (for cv.glmnet parallelization)
registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Grid over alpha (cv.glmnet, 5-fold)")
for (i in seq_along(alpha_grid)) {
  alpha <- alpha_grid[i]
  message(sprintf("Fitting cv.glmnet for alpha = %.1f (internal 5-fold CV on TRAIN)", alpha))
  
  # Fix fold randomness 
  set.seed(123)
  
  cvmod <- cv.glmnet(
    x = X_train,
    y = y_train,
    weights = w_train,
    type.measure = "mse", 
    nfolds = 5,               # <-
    parallel = TRUE,          # <- Try parallel = FALSE, display a bit differently ;)
    trace.it = 1,
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
# > Grid over alpha (cv.glmnet, 5-fold): 2.019 sec elapsed

#### ---- Explore tuning results ----
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by CV MSE (lower is better) irrespective of rule
tuning_results %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  #head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx          s       lambda lambda_idx nzero  dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.0         1 lambda.min   2.83965022         96     4 0.05350156 8310.109 135.0460 8175.063 8445.155
# >   0.1         2 lambda.min   0.73689611         61     4 0.05353111 8310.368 134.9168 8175.452 8445.285
# >   0.5         6 lambda.min   0.12235682         63     4 0.05353783 8310.423 134.8240 8175.599 8445.247
# >   0.2         3 lambda.min   0.30589205         63     4 0.05353662 8310.424 134.8526 8175.571 8445.277
# >   0.3         4 lambda.min   0.18581163         64     4 0.05353779 8310.443 134.8304 8175.613 8445.274
# >   0.4         5 lambda.min   0.15294602         63     4 0.05353765 8310.455 134.8334 8175.621 8445.288
# >   0.6         7 lambda.min   0.11190538         62     4 0.05353751 8310.465 134.8373 8175.627 8445.302
# >   0.7         8 lambda.min   0.09591890         62     4 0.05353760 8310.472 134.8343 8175.638 8445.306
# >   0.8         9 lambda.min   0.08392904         62     4 0.05353767 8310.475 134.8329 8175.642 8445.308
# >   0.9        10 lambda.min   0.07460359         62     4 0.05353773 8310.477 134.8318 8175.645 8445.309
# >   1.0        11 lambda.min   0.06714323         62     4 0.05353777 8310.478 134.8310 8175.647 8445.309
# >   0.4         5 lambda.1se  16.02286635         13     3 0.03896658 8432.910 141.7096 8291.201 8574.620
# >   0.9        10 lambda.1se   7.81558935         12     2 0.03860868 8433.463 140.9051 8292.558 8574.368
# >   0.8         9 lambda.1se   8.79253802         12     2 0.03843441 8435.039 140.9336 8294.105 8575.972
# >   0.0         1 lambda.1se 128.77468748         55     4 0.03795227 8436.766 140.1952 8296.571 8576.961
# >   0.7         8 lambda.1se  10.04861488         12     2 0.03820923 8437.012 140.9834 8296.028 8577.995
# >   0.2         3 lambda.1se  29.19887813         14     3 0.03833663 8437.436 141.9041 8295.532 8579.340
# >   0.6         7 lambda.1se  11.72338403         12     2 0.03790759 8439.616 141.0516 8298.565 8580.668
# >   0.3         4 lambda.1se  21.36382180         13     3 0.03808452 8440.250 141.9185 8298.332 8582.169
# >   0.1         2 lambda.1se  48.48284319         16     3 0.03772012 8441.275 141.7160 8299.559 8582.991
# >   0.5         6 lambda.1se  14.06806083         12     2 0.03748390 8443.240 141.1258 8302.114 8584.365
# >   1.0        11 lambda.1se   7.71983970         11     1 0.03691772 8443.798 141.0794 8302.719 8584.878

# Choose winners under each rule separately (clean and explicit)
best_min  <- tuning_results %>%
  filter(s == "lambda.min") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%   # tie-breaks: fewer nonzeros (nzero), larger λ, smaller α
  slice(1)

best_1se  <- tuning_results %>%
  filter(s == "lambda.1se") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  slice(1)

print(as.data.frame(best_min),  row.names = FALSE)
# > alpha alpha_idx          s  lambda lambda_idx nzero  dev_ratio      cvm    cvsd     cvlo     cvup
# >     0         1 lambda.min 2.83965         96     4 0.05350156 8310.109 135.046 8175.063 8445.155
print(as.data.frame(best_1se), row.names = FALSE)
# > alpha alpha_idx          s   lambda lambda_idx nzero  dev_ratio     cvm     cvsd     cvlo    cvup
# >   0.4         5 lambda.1se 16.02287         13     3 0.03896658 8432.91 141.7096 8291.201 8574.62

best_alpha_min     <- best_min$alpha
best_lambda_min    <- best_min$lambda
best_alpha_idx_min <- best_min$alpha_idx

best_alpha_1se     <- best_1se$alpha
best_lambda_1se    <- best_1se$lambda
best_alpha_idx_1se <- best_1se$alpha_idx

message(sprintf("Winner @ lambda.min : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_min, best_lambda_min, best_min$nzero, best_min$cvm, best_min$cvsd))
# > Winner @ lambda.min : alpha = 0.00 | lambda = 2.839650 | nzero = 4 | CVM = 8310.10875 (± 135.04602)
message(sprintf("Winner @ lambda.1se : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_1se, best_lambda_1se, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
# > Winner @ lambda.1se : alpha = 0.40 | lambda = 16.022866 | nzero = 3 | CVM = 8432.91010 (± 141.70959)

### ---- Predict & evaluate on TRAIN / VALID / TEST for both winners ----
cv_best_min  <- cv_list[[best_alpha_idx_min]]
cv_best_1se  <- cv_list[[best_alpha_idx_1se]]

# Coefficients & importances
coef(cv_best_min,  s = best_lambda_min)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >              s=2.83965
# > (Intercept) 521.167910
# > EXERPRAC     -1.846305
# > STUDYHMW      1.862984
# > WORKPAY      -6.306849
# > WORKHOME     -1.799307
coef(cv_best_1se,  s = best_lambda_1se)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >              s=16.02287
# > (Intercept) 511.9312740
# > EXERPRAC     -0.3413858
# > STUDYHMW      .        
# > WORKPAY      -4.3675187
# > WORKHOME     -0.1191435
caret::varImp(cv_best_min$glmnet.fit,  lambda = best_lambda_min)
# >           Overall
# > EXERPRAC 1.846305
# > STUDYHMW 1.862984
# > WORKPAY  6.306849
# > WORKHOME 1.799307
caret::varImp(cv_best_1se$glmnet.fit,  lambda = best_lambda_1se)
# >            Overall
# > EXERPRAC 0.3413858
# > STUDYHMW 0.0000000
# > WORKPAY  4.3675187
# > WORKHOME 0.1191435

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
# > lambda.min   Training 91.06964 73.20327 -2.842227e-13 3.608504 0.05350156
# > lambda.min Validation 88.00491 70.62650 -3.006045e+00 2.748281 0.05271003
# > lambda.min       Test 91.33127 73.71567 -1.065989e+00 3.424498 0.06835572

print(as.data.frame(metric_results_1se),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.1se   Training 91.76624 73.91490 -2.877281e-13 3.701157 0.03896658
# > lambda.1se Validation 88.53872 71.02572 -2.267634e+00 2.991144 0.04118313
# > lambda.1se       Test 92.47675 74.71938 -5.457834e-01 3.647633 0.04483972

print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.min   Training 91.06964 73.20327 -2.842227e-13 3.608504 0.05350156
# > lambda.min Validation 88.00491 70.62650 -3.006045e+00 2.748281 0.05271003
# > lambda.min       Test 91.33127 73.71567 -1.065989e+00 3.424498 0.06835572
# > lambda.1se   Training 91.76624 73.91490 -2.877281e-13 3.701157 0.03896658
# > lambda.1se Validation 88.53872 71.02572 -2.267634e+00 2.991144 0.04118313
# > lambda.1se       Test 92.47675 74.71938 -5.457834e-01 3.647633 0.04483972


# Sanity check: wrapper vs underlying at the same λ
# all.equal(as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min)),
#           as.numeric(predict(cv_best_min$glmnet.fit, newx = X_valid, s = best_lambda_min)))
# # > [1] TRUE
# all.equal(as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se)),
#           as.numeric(predict(cv_best_1se$glmnet.fit, newx = X_valid, s = best_lambda_1se)))
# # > [1] TRUE
