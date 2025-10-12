# ---- I. Predictive Modelling: Version 1.5 ----

# xgb.train with manual K-folds (fixed hyperparameters); tracks out-of-fold (OOF) predictions/metrics.

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


## ---- Main model (final weight) using xgb.cv to choose best_iter per rule ----

### ---- Cross-validation (CV) for PV1MATH only: xgb.cv + Manual Folds + OOF ----

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
num_folds <- 5
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

# Fixed hyperparameters 
nrounds <- 100
params <- list(
  objective   = "reg:squarederror",
  max_depth   = 6,
  eta         = 0.3,
  eval_metric = "rmse",
  nthread     = max(1, parallel::detectCores() - 1)
)

# --- Manual CV via xgb.train (train one booster per fold to full nrounds) ---
dtrain_fold <- vector("list", num_folds)
dvalid_fold <- vector("list", num_folds)
mod_fold    <- vector("list", num_folds)
eval_fold   <- vector("list", num_folds)   # per-iter {train_rmse, valid_rmse}

set.seed(123)
for (k in seq_len(num_folds)) {
  valid_idx_k <- cv_folds[[k]]
  train_idx_k <- setdiff(seq_len(n_cv), valid_idx_k)
  
  dtrain_fold[[k]] <- xgb.DMatrix(as.matrix(X_train[train_idx_k, , drop = FALSE]),
                                  label = y_train[train_idx_k],
                                  weight = w_train[train_idx_k])
  dvalid_fold[[k]] <- xgb.DMatrix(as.matrix(X_train[valid_idx_k, , drop = FALSE]),
                                  label = y_train[valid_idx_k],
                                  weight = w_train[valid_idx_k])
  
  mod_fold[[k]] <- xgb.train(
    params  = params,
    data    = dtrain_fold[[k]],
    nrounds = nrounds,
    watchlist = list(train = dtrain_fold[[k]], valid = dvalid_fold[[k]]),
    verbose = 1,
    early_stopping_rounds=NULL 
  )
  
  eval_fold[[k]] <- as.data.frame(mod_fold[[k]]$evaluation_log)  # cols: iter, train_rmse, valid_rmse
  stopifnot(nrow(eval_fold[[k]]) == nrounds)
}

# Aggregate fold logs to mimic xgb.cv$evaluation_log (population SD like xgb.cv)
train_rmse_matrix <- do.call(cbind, lapply(eval_fold, function(df) df$train_rmse))
valid_rmse_matrix <- do.call(cbind, lapply(eval_fold, function(df) df$valid_rmse))

cv_eval_log <- tibble::tibble(
  iter             = seq_len(nrounds),
  train_rmse_mean  = rowMeans(train_rmse_matrix),
  train_rmse_std   = sqrt(pmax(0, rowMeans(train_rmse_matrix^2) - rowMeans(train_rmse_matrix)^2)),
  test_rmse_mean   = rowMeans(valid_rmse_matrix),
  test_rmse_std    = sqrt(pmax(0, rowMeans(valid_rmse_matrix^2) - rowMeans(valid_rmse_matrix)^2))
)

# CV-best by fold-mean validation RMSE
best_iter_by_cv <- which.min(cv_eval_log$test_rmse_mean)
best_rmse_by_cv <- cv_eval_log$test_rmse_mean[best_iter_by_cv]

# --- Reconstruct pooled OOF predictions per iteration ---
oof_pred_matrix <- matrix(NA_real_, nrow = n_cv, ncol = nrounds)
rmse_oof_by_iter <- numeric(nrounds)

for (iter in seq_len(nrounds)) {
  for (k in seq_len(num_folds)) {
    idx <- cv_folds[[k]]
    oof_pred_matrix[idx, iter] <- predict( 
      mod_fold[[k]],
      dvalid_fold[[k]],
      iterationrange = c(1, iter + 1)
    ) # Consider streaming instead of storing to save memory
  }
  rmse_oof_by_iter[iter] <- sqrt(sum(w_train * (y_train - oof_pred_matrix[, iter])^2) / sum(w_train))
}

best_iter_by_oof <- which.min(rmse_oof_by_iter)
best_rmse_by_oof <- rmse_oof_by_iter[best_iter_by_oof]
rmse_oof_at_final_iter <- rmse_oof_by_iter[nrounds]

# Create a cv_mod-like object for downstream compatibility
cv_mod <- list(
  evaluation_log = cv_eval_log,
  models = mod_fold,
  pred = oof_pred_matrix[, nrounds]   # OOF snapshot at final iteration
)

#### ---- Explore CV output ----
cv_mod$evaluation_log |> head(10) |> print(row.names = FALSE)

message(sprintf("CV-selected best_iter = %d | test_rmse_mean = %.5f (± %.5f)",
                best_iter_by_cv,
                best_rmse_by_cv,
                cv_mod$evaluation_log$test_rmse_std[best_iter_by_cv]))
# > CV-selected best_iter = 13 | test_rmse_mean = 90.60277 (± 1.48136)

message(sprintf("OOF-selected best_iter = %d | pooled OOF RMSE = %.5f",
                best_iter_by_oof,
                best_rmse_by_oof))
# > OOF-selected best_iter = 13 | pooled OOF RMSE = 90.63768

# Visualize CV-mean curves
cv_mod$evaluation_log %>%
  tidyr::pivot_longer(cols = c(train_rmse_mean, test_rmse_mean),
                      names_to = "Dataset", values_to = "RMSE") %>%
  ggplot2::ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_vline(xintercept = best_iter_by_cv, linetype = 2, alpha = 0.4) +
  ggplot2::annotate("text",
                    x = best_iter_by_cv,
                    y = max(cv_mod$evaluation_log$test_rmse_mean, na.rm = TRUE) * 1.02,
                    label = paste("Best iter =", best_iter_by_cv), size = 3, vjust = 0) +
  ggplot2::labs(title = "Manual CV (xgb.train): RMSE mean across folds",
                x = "Boosting Round", y = "RMSE", color = "Dataset") +
  ggplot2::theme_minimal()

# Visualize pooled weighted OOF curve
tibble::tibble(iter = seq_len(nrounds), OOF_RMSE = rmse_oof_by_iter) %>%
  ggplot2::ggplot(aes(x = iter, y = OOF_RMSE)) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_vline(xintercept = best_iter_by_oof, linetype = 2, alpha = 0.4) +
  ggplot2::annotate("text",
                    x = best_iter_by_oof,
                    y = max(rmse_oof_by_iter, na.rm = TRUE) * 1.02,
                    label = paste("Best iter =", best_iter_by_oof), size = 3, vjust = 0) +
  ggplot2::labs(title = "Manual CV (xgb.train): pooled weighted OOF RMSE",
                x = "Boosting Round", y = "OOF RMSE") +
  ggplot2::theme_minimal()


### ---- Refit on full TRAIN (for deployment) ----
set.seed(123)
main_model <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = nrounds,
  watchlist = list(train = dtrain, valid = dvalid),
  verbose = 1
)

# VALID-selected cutoff from refit log
best_iter <- which.min(main_model$evaluation_log$valid_rmse)
best_rmse <- min(main_model$evaluation_log$valid_rmse)

#### ---- Explore refit, importance, and quick tree ----
main_model  # str(main_model)
names(main_model$evaluation_log)
print(as.data.frame(main_model$evaluation_log)) 

xgb.importance(model = main_model)
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.3068270 0.2394453 0.2222222
# > 2: STUDYHMW 0.2471694 0.2611407 0.2607579
# > 3: EXERPRAC 0.2324982 0.2502567 0.2512845
# > 4: WORKHOME 0.2135054 0.2491572 0.2657354
xgb.importance(model = main_model, trees = 0:(best_iter - 1))
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.3613280 0.2308251 0.1976369
# > 2: STUDYHMW 0.2423058 0.2676605 0.2857143
# > 3: EXERPRAC 0.2114040 0.2719123 0.2341568
# > 4: WORKHOME 0.1849622 0.2296021 0.2824919

main_model$evaluation_log |>
  pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = best_iter, linetype = 2, alpha = 0.4) +
  annotate("text",
           x = best_iter,
           y = max(main_model$evaluation_log$valid_rmse) * 1.02,
           label = paste("Best iter =", best_iter), size = 3, vjust = 0
  ) +
  labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal()

xgb.plot.importance(importance_matrix = xgb.importance(model = main_model),
                    top_n = NULL,                        # Top n features (you only have 4 in oos)
                    measure = "Gain",                    # Can also use "Cover" or "Frequency"
                    rel_to_first = TRUE,
                    xlab = "Relative Importance")

xgb.plot.tree(model = main_model, trees = 0)                  # or trees = 1, 2, etc. 
xgb.plot.tree(model = main_model, trees = best_iter-1)  # trees with lowest VALID rmse


### ---- Predict and evaluate performance on training/validation/test datasets ----

# Predictions at CV-selected cutoff
pred_cv_train <- predict(main_model, dtrain, iterationrange = c(1, best_iter_by_cv + 1))
pred_cv_valid <- predict(main_model, dvalid, iterationrange = c(1, best_iter_by_cv + 1))
pred_cv_test  <- predict(main_model, dtest,  iterationrange = c(1, best_iter_by_cv + 1))

# Predictions at OOF-selected cutoff
pred_oof_train <- predict(main_model, dtrain, iterationrange = c(1, best_iter_by_oof + 1))
pred_oof_valid <- predict(main_model, dvalid, iterationrange = c(1, best_iter_by_oof + 1))
pred_oof_test  <- predict(main_model, dtest,  iterationrange = c(1, best_iter_by_oof + 1))

# Predictions at VALID-selected cutoff (from refit)
pred_train <- predict(main_model, dtrain, iterationrange = c(1, best_iter + 1))
pred_valid <- predict(main_model, dvalid, iterationrange = c(1, best_iter + 1))
pred_test  <- predict(main_model, dtest,  iterationrange = c(1, best_iter + 1))

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
  R2      = c(compute_metrics(y_train, pred_oof_train, w_train)["r2"],
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
              compute_metrics(y_valid, pred_valid, w_valid)["mae"],
              compute_metrics(y_test,  pred_test,  w_test )["mae"]),
  Bias    = c(compute_metrics(y_train, pred_train, w_train)["bias"],
              compute_metrics(y_valid, pred_valid, w_valid)["bias"],
              compute_metrics(y_test,  pred_test,  w_test )["bias"]),
  `Bias%` = c(compute_metrics(y_train, pred_train, w_train)["bias_pct"],
              compute_metrics(y_valid, pred_valid, w_valid)["bias_pct"],
              compute_metrics(y_test,  pred_test,  w_test )["bias_pct"]),
  R2      = c(compute_metrics(y_train, pred_train, w_train)["r2"],
              compute_metrics(y_valid, pred_valid, w_valid)["r2"],
              compute_metrics(y_test,  pred_test,  w_test )["r2"])
)

# Print combined metrics
print(as.data.frame(dplyr::bind_rows(metrics_cv, metrics_oof, metrics)), row.names = FALSE)
# >      Model    Dataset     RMSE      MAE      Bias    Bias%         R2
# >    CV-best   Training 85.14164 68.14055 -4.869022 2.232743 0.17271213
# >    CV-best Validation 86.89550 69.58232 -8.804809 1.370829 0.07644301
# >    CV-best       Test 90.05394 71.79991 -6.757713 2.059296 0.09423298
# >   OOF-best   Training 85.14164 68.14055 -4.869022 2.232743 0.17271213
# >   OOF-best Validation 86.89550 69.58232 -8.804809 1.370829 0.07644301
# >   OOF-best       Test 90.05394 71.79991 -6.757713 2.059296 0.09423298
# > VALID-best   Training 84.62298 67.69093 -2.386232 2.718319 0.18276069
# > VALID-best Validation 86.82276 69.36116 -6.337820 1.870945 0.07798852
# > VALID-best       Test 89.87706 71.56219 -4.195269 2.576558 0.09778761

# Summary messages
message(sprintf(
  "Summary | CV-best iter = %d | OOF-best iter = %d | VALID-best iter = %d",
  best_iter_by_cv, best_iter_by_oof, best_iter
))
# > Summary | CV-best iter = 13 | OOF-best iter = 13 | VALID-best iter = 15
message(sprintf(
  "OOF RMSE @ CV-best = %.5f | @ OOF-best = %.5f | @ VALID-best = %.5f",
  rmse_oof_by_iter[best_iter_by_cv], rmse_oof_by_iter[best_iter_by_oof], rmse_oof_by_iter[best_iter]
))
# > OOF RMSE @ CV-best = 90.63768 | @ OOF-best = 90.63768 | @ VALID-best = 90.73744

