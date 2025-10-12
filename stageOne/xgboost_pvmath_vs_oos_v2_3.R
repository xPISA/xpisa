# ---- II. Predictive Modelling: Version 2.3 ----

# xgb.cv with manual K-folds + hyperparameter tuning

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

### ---- Tune XGBoost model for PV1MATH only: xgb.cv + Manual Folds ----

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


# Hyperparameter grid (3 × 3 × 4 = 36) 
grid <- expand.grid(
  nrounds   = c(100, 200, 300),     
  max_depth = c(4, 6, 8),
  eta       = c(0.01, 0.05, 0.1, 0.3),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

# Containers
tuning_results <- tibble::tibble(
  grid_id = integer(),
  param_name = character(),
  nrounds = integer(),
  max_depth = integer(),
  eta = double(),
  best_iter_cv = integer(),          # argmin over test_rmse_mean
  train_rmse_mean = double(),        # at best_iter_cv
  train_rmse_std  = double(),        # at best_iter_cv
  test_rmse_mean  = double(),        # at best_iter_cv  (selector)
  test_rmse_std   = double()         # at best_iter_cv
)

cv_eval_list <- vector("list", nrow(grid))   # per-config evaluation logs
cv_mod_list  <- vector("list", nrow(grid))   # store cv objects (audit)

# Track overall best (by CV fold-mean on test)
best_cv_rmse   <- Inf
best_grid_id   <- NA_integer_
best_params    <- list()
best_iter_cv   <- NA_integer_
best_cv_eval   <- NULL
best_row_full  <- NULL

# Tuning loop: xgb.cv with manual folds
set.seed(123)
tictoc::tic("Tuning (xgb.cv, manual folds)")
for (i in seq_len(nrow(grid))) {
  row <- grid[i, ]
  message(sprintf("CV %02d/%02d | nrounds=%d, max_depth=%d, eta=%.3f",
                  i, nrow(grid), row$nrounds, row$max_depth, row$eta))
  
  params <- list(
    objective   = "reg:squarederror",
    max_depth   = row$max_depth,
    eta         = row$eta,
    eval_metric = "rmse",
    nthread     = max(1, parallel::detectCores() - 1)
    # seed = 123  
  )
  
  cv_mod <- xgb.cv(
    params  = params,
    data    = dtrain,
    nrounds = row$nrounds,
    nfold   = num_folds,         # ignored when `folds` provided; harmless
    prediction = FALSE,
    showsd  = TRUE,
    verbose = 1,
    stratified = FALSE,
    folds   = cv_folds,          # <-- manual, fixed folds
    early_stopping_rounds = NULL
  )
  
  # ---- Best iteration by fold-mean validation RMSE (no temp cv_eval var) ----
  best_iter_cv_i   <- which.min(cv_mod$evaluation_log$test_rmse_mean)
  best_test_mean_i <- cv_mod$evaluation_log$test_rmse_mean[best_iter_cv_i]
  best_test_std_i  <- cv_mod$evaluation_log$test_rmse_std[best_iter_cv_i]
  best_train_mean_i<- cv_mod$evaluation_log$train_rmse_mean[best_iter_cv_i]
  best_train_std_i <- cv_mod$evaluation_log$train_rmse_std[best_iter_cv_i]
  
  # Save per-config log and cv object
  cv_eval_list[[i]] <- dplyr::mutate(as.data.frame(cv_mod$evaluation_log), grid_id = i)
  cv_mod_list[[i]]  <- cv_mod
  
  # Append summary row
  tuning_results <- dplyr::add_row(
    tuning_results,
    grid_id = i,
    param_name = sprintf("nrounds=%d, max_depth=%d, eta=%.3f", row$nrounds, row$max_depth, row$eta),
    nrounds = row$nrounds,
    max_depth = row$max_depth,
    eta = row$eta,
    best_iter_cv = best_iter_cv_i,
    train_rmse_mean = best_train_mean_i,
    train_rmse_std  = best_train_std_i,
    test_rmse_mean  = best_test_mean_i,
    test_rmse_std   = best_test_std_i
  )
  
  # Update global best (strict '<' keeps first in ties — favors smaller nrounds by grid order)
  if (best_test_mean_i < best_cv_rmse) {
    best_cv_rmse <- best_test_mean_i
    best_grid_id <- i
    best_params  <- params
    best_iter_cv <- best_iter_cv_i
    best_cv_eval <- as.data.frame(cv_mod$evaluation_log)  # store for plotting
    best_row_full<- row
  }
}
tictoc::toc()
# > Tuning (xgb.cv, manual folds): 123.124 sec elapsed

#### ---- Explore tuning output ----
tuning_results %>% 
  arrange(test_rmse_mean) %>% 
  head(10) %>% 
  as.data.frame() %>% 
  print(row.names = FALSE)
# > grid_id                          param_name nrounds max_depth  eta best_iter_cv train_rmse_mean train_rmse_std test_rmse_mean test_rmse_std
# >      19 nrounds=100, max_depth=4, eta=0.100     100         4 0.10           50        87.27138      0.3012648       89.44827      1.217818
# >      20 nrounds=200, max_depth=4, eta=0.100     200         4 0.10           50        87.27138      0.3012648       89.44827      1.217818
# >      21 nrounds=300, max_depth=4, eta=0.100     300         4 0.10           50        87.27138      0.3012648       89.44827      1.217818
# >      11 nrounds=200, max_depth=4, eta=0.050     200         4 0.05          102        87.29514      0.2836398       89.45365      1.220706
# >      12 nrounds=300, max_depth=4, eta=0.050     300         4 0.05          102        87.29514      0.2836398       89.45365      1.220706
# >      10 nrounds=100, max_depth=4, eta=0.050     100         4 0.05           99        87.34661      0.2816726       89.45510      1.220395
# >      28 nrounds=100, max_depth=4, eta=0.300     100         4 0.30           15        87.37651      0.2707659       89.51472      1.288254
# >      29 nrounds=200, max_depth=4, eta=0.300     200         4 0.30           15        87.37651      0.2707659       89.51472      1.288254
# >      30 nrounds=300, max_depth=4, eta=0.300     300         4 0.30           15        87.37651      0.2707659       89.51472      1.288254
# >      22 nrounds=100, max_depth=6, eta=0.100     100         6 0.10           44        84.13944      0.3316562       90.30762      1.440544

best_params
best_row_full
# >    nrounds max_depth eta
# > 19     100         4 0.1

message(sprintf("Selected grid_id = %d | nrounds=%d, max_depth=%d, eta=%.3f | best_iter_cv = %d | cv_min = %.5f",
                best_grid_id, best_row_full$nrounds, best_row_full$max_depth, best_row_full$eta,
                best_iter_cv, best_cv_rmse))
# > Selected grid_id = 19 | nrounds=100, max_depth=4, eta=0.100 | best_iter_cv = 50 | cv_min = 89.44827

# Visualize CV curves (means across folds) for the selected config
best_cv_eval %>%
  tidyr::pivot_longer(cols = c(train_rmse_mean, test_rmse_mean),
                      names_to = "Dataset", values_to = "RMSE") %>%
  ggplot2::ggplot(ggplot2::aes(x = iter, y = RMSE, color = Dataset)) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_vline(xintercept = best_iter_cv, linetype = 2, alpha = 0.4) +
  ggplot2::annotate("text",
                    x = best_iter_cv,
                    y = max(best_cv_eval$test_rmse_mean, na.rm = TRUE) * 1.02,
                    label = paste("Best iter =", best_iter_cv), size = 3, vjust = 0) +
  ggplot2::labs(title = "xgb.cv RMSE over Boosting Rounds (means across folds)",
                x = "Boosting Round", y = "RMSE", color = "Dataset") +
  ggplot2::theme_minimal()


### ---- Refit final booster on full TRAIN at CV-selected iteration ----
set.seed(123)
main_model <- xgb.train(
  params  = best_params,
  data    = dtrain,
  nrounds = best_iter_cv,                          # train exactly to the CV-selected iteration
  watchlist = list(train = dtrain, valid = dvalid),
  verbose = 1,
  early_stopping_rounds = NULL
)

#### ---- Explore refit, importance, and quick tree ----
main_model
print(as.data.frame(main_model$evaluation_log), row.names = FALSE)
# > iter train_rmse valid_rmse
# >    1  461.11274  462.34838
# >    2  416.82702  418.17171
# >  ...
# >   49   87.61153   86.14725
# >   50   87.58393   86.12982

xgb.importance(model = main_model)  # <=> xgb.importance(model = main_model, trees = 0:(best_iter_cv - 1))
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
    title = "XGBoost RMSE over Boosting Rounds (refit at best_iter_cv)",
    x = "Boosting Round", y = "RMSE", color = "Dataset"
  ) +
  ggplot2::theme_minimal()

# Optional tree snapshots
xgb.plot.tree(model = main_model, trees = 0)
xgb.plot.tree(model = main_model, trees = max(0, best_iter_cv - 1))


### ---- Predict and evaluate performance on TRAIN/VALID/TEST ----
pred_train <- predict(main_model, dtrain)  # exactly best_iter_cv trees
pred_valid <- predict(main_model, dvalid)
pred_test  <- predict(main_model, dtest)

# Optional equivalence checks
stopifnot(all.equal(
  predict(main_model, dtrain, iterationrange = c(1, best_iter_cv + 1)), pred_train
))
stopifnot(all.equal(
  predict(main_model, dvalid, iterationrange = c(1, best_iter_cv + 1)), pred_valid
))
stopifnot(all.equal(
  predict(main_model, dtest,  iterationrange = c(1, best_iter_cv + 1)), pred_test
))

metrics_train <- compute_metrics(y_true = y_train, y_pred = pred_train, w = w_train)
metrics_valid <- compute_metrics(y_true = y_valid, y_pred = pred_valid, w = w_valid)
metrics_test  <- compute_metrics(y_true = y_test,  y_pred = pred_test,  w = w_test)

metric_results <- tibble::tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE     = c(metrics_train["rmse"],     metrics_valid["rmse"],     metrics_test["rmse"]),
  MAE      = c(metrics_train["mae"],      metrics_valid["mae"],      metrics_test["mae"]),
  Bias     = c(metrics_train["bias"],     metrics_valid["bias"],     metrics_test["bias"]),
  `Bias%`  = c(metrics_train["bias_pct"], metrics_valid["bias_pct"], metrics_test["bias_pct"]),
  R2       = c(metrics_train["r2"],       metrics_valid["r2"],       metrics_test["r2"])
)
print(as.data.frame(metric_results), row.names = FALSE)
# >    Dataset     RMSE      MAE      Bias    Bias%         R2
# >   Training 87.58393 70.23074 -2.587390 2.846265 0.12456984
# > Validation 86.12982 68.85730 -6.044312 1.960081 0.09264719
# >       Test 89.51025 71.48393 -4.269235 2.597415 0.10513692

# Summary message
message(sprintf("Selected config -> grid_id=%d | nrounds cap=%d | max_depth=%d | eta=%.3f | best_iter_cv=%d | cv_min=%.5f",
                best_grid_id, best_row_full$nrounds, best_row_full$max_depth, best_row_full$eta, best_iter_cv, best_cv_rmse))
# > Selected config -> grid_id=19 | nrounds cap=100 | max_depth=4 | eta=0.100 | best_iter_cv=50 | cv_min=89.44827
