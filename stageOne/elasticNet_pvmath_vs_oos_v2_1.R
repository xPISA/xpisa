# ---- II. Predictive Modelling: Version 2.1 ----

# Tune hyperparameters (both alpha α and lambda λ) using glmnet only

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

### ---- Fit main model using final student weights (W_FSTUWT) on the training data ---- 

#### ---- Tuning for PV1MATH only: glmnet ----

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
mod_list        <- vector("list", length(alpha_grid))      # glmnet fits per alpha
per_alpha_list  <- vector("list", length(alpha_grid))      # per‑lambda metrics per alpha

tic("Grid over alpha (glmnet) -> select by Validation RMSE")
for (i in seq_along(alpha_grid)) {
  alpha   <- alpha_grid[i]
  message(sprintf("Fitting glmnet path for alpha = %.1f", alpha))
  
  # Fit the whole lambda path at this alpha on TRAIN
  mod <- glmnet(
    x = X_train, 
    y = y_train,
    family     = "gaussian",
    weights    = w_train,
    alpha      = alpha,
    standardize= TRUE,
    intercept  = TRUE
  )
  mod_list[[i]] <- mod
  
  # Predictions across the path
  pred_valid_matrix <- predict(mod, newx = X_valid, s = mod$lambda)
  pred_train_matrix <- predict(mod, newx = X_train, s = mod$lambda)
  
  # RMSE curves (weighted)
  rmse_valid_vec <- apply(pred_valid_matrix, 2, function(p) w_rmse(y_valid, p, w_valid))
  rmse_train_vec <- apply(pred_train_matrix, 2, function(p) w_rmse(y_train, p, w_train))
  
  # Collect tidy rows per alpha (one row per lambda on this path)
  per_alpha_list[[i]] <- tibble(
    alpha       = alpha,
    alpha_idx   = i,
    lambda      = as.numeric(mod$lambda),
    lambda_idx  = seq_along(mod$lambda),
    df          = mod$df,
    dev_ratio   = as.numeric(mod$dev.ratio),
    rmse_valid  = as.numeric(rmse_valid_vec),
    rmse_train  = as.numeric(rmse_train_vec)
  )
}
toc()
# > Grid over alpha (glmnet) -> select by Validation RMSE: 1.128 sec elapsed

##### ---- Explore tuning results ----

# Stack all alpha–lambda candidates
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by Validation RMSE
tuning_results %>%
  arrange(rmse_valid) %>%
  head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx    lambda lambda_idx df  dev_ratio rmse_valid rmse_train
# >   0.0         1  8.671881         84  4 0.05322086   87.99720   91.08314
# >   0.0         1  9.517379         83  4 0.05316185   87.99729   91.08598
# >   0.0         1  7.901495         85  4 0.05327103   87.99736   91.08073
# >   0.0         1 10.445312         82  4 0.05309257   87.99770   91.08931
# >   0.0         1  7.199548         86  4 0.05331358   87.99772   91.07868
# >   0.1         2  4.736822         41  4 0.05319925   87.99799   91.08418
# >   0.1         2  4.316016         42  4 0.05325474   87.99806   91.08151
# >   0.1         2  5.198656         40  4 0.05313328   87.99821   91.08736
# >   0.0         1  6.559960         87  4 0.05334963   87.99822   91.07695
# >.  0.1         2  3.932593         43  4 0.05330137   87.99835   91.07927

# Choose the best (Validation RMSE); tie‑break: fewer non‑zero coefs → larger lambda → smaller alpha
best_row <- tuning_results %>%
  arrange(rmse_valid, df, desc(lambda), alpha) %>%
  slice(1)
print(as.data.frame(best_row), row.names = FALSE)
# > alpha alpha_idx   lambda lambda_idx df  dev_ratio rmse_valid rmse_train
# >     0         1 8.671881         84  4 0.05322086    87.9972   91.08314

best_alpha      <- best_row$alpha
best_lambda     <- best_row$lambda
best_alpha_idx  <- best_row$alpha_idx
best_df         <- best_row$df
best_rmse_valid <- best_row$rmse_valid

message(sprintf(
  "Selected: alpha = %.2f | lambda = %.6f | df = %d | Valid RMSE = %.5f",
  best_alpha, best_lambda, best_df, best_rmse_valid
))
# > Selected: alpha = 0.00 | lambda = 8.671881 | df = 4 | Valid RMSE = 87.99720

#### ---- Predict and evaluate performance on training/validation/test datasets ----
best_mod <- mod_list[[best_alpha_idx]]

coef(best_mod, s = best_lambda)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             s=8.671881
# > (Intercept) 520.714141
# > EXERPRAC     -1.785846
# > STUDYHMW      1.691682
# > WORKPAY      -5.969659
# > WORKHOME     -1.715343
varImp(best_mod, lambda = best_lambda)
# >           Overall
# > EXERPRAC 1.785846
# > STUDYHMW 1.691682
# > WORKPAY  5.969659
# > WORKHOME 1.715343

pred_train_best <- as.numeric(predict(best_mod, newx = X_train, s = best_lambda))
pred_valid_best <- as.numeric(predict(best_mod, newx = X_valid, s = best_lambda))
pred_test_best  <- as.numeric(predict(best_mod,  newx = X_test,  s = best_lambda))

metrics_train_best <- compute_metrics(y_true = y_train, y_pred = pred_train_best, w = w_train)
metrics_valid_best <- compute_metrics(y_true = y_valid, y_pred = pred_valid_best, w = w_valid)
metrics_test_best  <- compute_metrics(y_true = y_test,  y_pred = pred_test_best,  w = w_test)

metric_results <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_best["rmse"], metrics_valid_best["rmse"], metrics_test_best["rmse"]),
  MAE     = c(metrics_train_best["mae"],  metrics_valid_best["mae"],  metrics_test_best["mae"]),
  Bias    = c(metrics_train_best["bias"], metrics_valid_best["bias"], metrics_test_best["bias"]),
  `Bias%` = c(metrics_train_best["bias_pct"], metrics_valid_best["bias_pct"], metrics_test_best["bias_pct"]),
  R2      = c(metrics_train_best["r2"],   metrics_valid_best["r2"],   metrics_test_best["r2"])
)

print(as.data.frame(metric_results), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.08314 73.23093 -2.840842e-13 3.618566 0.05322086
# > Validation 87.99720 70.61768 -2.945335e+00 2.770650 0.05287589
# >       Test 91.38917 73.77597 -1.019268e+00 3.446746 0.06717423

## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH (best_alpha, best_lambda) to all plausible values in mathematics.

### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

tic("Fitting main glmnet models (fixed best_alpha, best_lambda)")
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
    alpha = best_alpha,       # best_alpha = 0
    lambda = best_lambda,     # best_lambda = 8.671881
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(oos, collapse = " + "))),
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha, best_lambda): 1.634 sec elapsed

# Quick look
main_models[[1]]$formula
main_models[[1]]$coefs
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  520.714353   -1.786167    1.691868   -5.969469   -1.715343

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coefs
# >      (Intercept)  EXERPRAC STUDYHMW   WORKPAY  WORKHOME
# > [1,]    520.7144 -1.786167 1.691868 -5.969469 -1.715343
# > [2,]    521.4381 -2.027081 1.832240 -5.701027 -1.773726
# > [3,]    520.6720 -2.044449 1.930428 -5.805036 -1.830492
# > [4,]    523.3085 -2.206123 1.847078 -5.874577 -1.997340
# > [5,]    521.9354 -2.004485 1.784940 -5.886309 -1.895864
# > [6,]    520.5560 -2.171883 2.035311 -5.777912 -1.959403
# > [7,]    522.1227 -2.169219 1.689962 -5.822615 -1.701880
# > [8,]    519.9182 -2.036781 2.140576 -6.003972 -1.865690
# > [9,]    521.2555 -2.074773 1.788878 -6.038977 -1.728415
# > [10,]    520.7404 -2.105980 2.032483 -6.099684 -1.666669

main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  521.266125   -2.062694    1.877376   -5.897958   -1.813482 

# --- Weighted R² on TRAIN (point estimates per PV) ---
main_r2s_weighted <- sapply(1:M, function(i) {
  model  <- main_models[[i]]$mod
  X_train   <- as.matrix(train_data[, oos])
  y_train <- train_data[[pvmaths[i]]]
  w      <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.05576098

### ---- Replicate models using BRR replicate weights ----

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
      alpha = best_alpha,
      lambda = best_lambda,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda))
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
# > Fitting replicate glmnet models (BRR weights): 3.685 sec elapsed

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
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)  # 80 x 10

### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

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

#### ---- Final Outputs ----
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
# >        Term   Estimate Std. Error    z value Pr(>|z|) z_Signif    t value Pr(>|t|) t_Signif
# > (Intercept) 521.266125  2.6228868 198.737562  < 2e-16      *** 198.737562  < 2e-16      ***
# >    EXERPRAC  -2.062694  0.3068317  -6.722559 1.79e-11      ***  -6.722559 2.52e-09      ***
# >    STUDYHMW   1.877376  0.4292656   4.373461 1.22e-05      ***   4.373461 3.69e-05      ***
# >     WORKPAY  -5.897958  0.3995711 -14.760723  < 2e-16      *** -14.760723  < 2e-16      ***
# >    WORKHOME  -1.813482  0.3207772  -5.653402 1.57e-08      ***  -5.653402 2.41e-07      ***

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
# >                     Metric   Estimate  Std. Error
# > R-squared (Weighted, Train) 0.05576098 0.005902793

### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- as.matrix(train_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
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
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
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
# >       MSE 8285.6337290988103632117 181.878988812903514827 7929.157461480956044397 8642.109996716664682026 712.9525352357086376
# >      RMSE   91.0242680053883219671   0.999893732768025290   89.064512300795669830   92.984023709980974104   3.9195114091853043
# >       MAE   73.0917202838497246375   0.777864202114617487   71.567134462842091125   74.616306104857358150   3.0491716420152670
# >      Bias   -0.0000000000006719578   0.000000000002552648   -0.000000000005675056    0.000000000004331141   0.0000000000100062
# >     Bias%    3.6253280172608968179   0.093990872016213270    3.441109293233605371    3.809546741288188265   0.3684374480545829
# > R-squared    0.0557609830134192172   0.005902792607586635    0.044191722094340143    0.067330243932498285   0.0231385218381581

### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- as.matrix(valid_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda))
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
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 1.294 sec elapsed

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
# >       MSE  7727.53444151   343.10630014 7055.05845037 8400.01043265 1344.9519823
# >      RMSE    87.90110096     1.94860199   84.08191125   91.72029068    7.6383794
# >       MAE    70.54343322     1.73358048   67.14567792   73.94118851    6.7955106
# >      Bias    -1.97779386     3.04506753   -7.94601656    3.99042883   11.9364454
# >     Bias%     2.96574682     0.66561119    1.66117285    4.27032078    2.6091479
# > R-squared     0.05508253     0.01149623    0.03255033    0.07761473    0.0450644

### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- as.matrix(test_data[, oos])
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda))
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
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates (glmnet): 1.883 sec elapsed

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
# >    Metric Point_estimate Standard_error      CI_lower      CI_upper     CI_length
# >       MSE  8163.89679963   276.81392038 7621.35148527 8706.44211400 1085.09062872
# >      RMSE    90.35123327     1.53195666   87.34865340   93.35381314    6.00515974
# >       MAE    73.12828962     1.39704422   70.39013326   75.86644597    5.47631271
# >      Bias    -1.30298378     2.46921145   -6.14254929    3.53658173    9.67913102
# >     Bias%     3.27376059     0.53936010    2.21663422    4.33088696    2.11425274
# > R-squared     0.06456447     0.01093801    0.04312636    0.08600258    0.04287622

### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----
# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.
evaluate_split <- function(split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           oos, pvmaths, best_lambda) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    X     <- as.matrix(split_data[, oos])
    y     <- split_data[[pvmaths[i]]]
    w     <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X, s = best_lambda))
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
      y_pred <- as.numeric(predict(model, newx = X, s = best_lambda))
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
                             M, G, k, z_crit, oos, pvmaths, best_lambda)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, oos, pvmaths, best_lambda)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, oos, pvmaths, best_lambda)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                CI_lower                CI_upper            CI_length
# >       MSE 8285.6337290988103632117 181.878988812903514827 7929.157461480956044397 8642.109996716664682026 712.9525352357086376
# >      RMSE   91.0242680053883219671   0.999893732768025290   89.064512300795669830   92.984023709980974104   3.9195114091853043
# >       MAE   73.0917202838497246375   0.777864202114617487   71.567134462842091125   74.616306104857358150   3.0491716420152670
# >      Bias   -0.0000000000006719578   0.000000000002552648   -0.000000000005675056    0.000000000004331141   0.0000000000100062
# >     Bias%    3.6253280172608968179   0.093990872016213270    3.441109293233605371    3.809546741288188265   0.3684374480545829
# > R-squared    0.0557609830134192172   0.005902792607586635    0.044191722094340143    0.067330243932498285   0.0231385218381581
print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower      CI_upper    CI_length
# >       MSE  7727.53444151   343.10630014 7055.05845037 8400.01043265 1344.9519823
# >      RMSE    87.90110096     1.94860199   84.08191125   91.72029068    7.6383794
# >       MAE    70.54343322     1.73358048   67.14567792   73.94118851    6.7955106
# >      Bias    -1.97779386     3.04506753   -7.94601656    3.99042883   11.9364454
# >     Bias%     2.96574682     0.66561119    1.66117285    4.27032078    2.6091479
# > R-squared     0.05508253     0.01149623    0.03255033    0.07761473    0.0450644
print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower      CI_upper     CI_length
# >       MSE  8163.89679963   276.81392038 7621.35148527 8706.44211400 1085.09062872
# >      RMSE    90.35123327     1.53195666   87.34865340   93.35381314    6.00515974
# >       MAE    73.12828962     1.39704422   70.39013326   75.86644597    5.47631271
# >      Bias    -1.30298378     2.46921145   -6.14254929    3.53658173    9.67913102
# >     Bias%     3.27376059     0.53936010    2.21663422    4.33088696    2.11425274
# > R-squared     0.06456447     0.01093801    0.04312636    0.08600258    0.04287622
