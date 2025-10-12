# ---- II. Predictive Modelling: Version 2.2 ----

# xgb.cv (internal folds) with hyperparameter tuning

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

### ---- Tune XGBoost model for PV1MATH only: xgb.cv (internal folds) ----

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

# Hyperparameter grid (3 × 3 × 4 = 36 combinations)
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
  train_rmse_mean = double(),        # train_rmse_mean at best_iter_cv
  train_rmse_std  = double(),        # train_rmse_std  at best_iter_cv
  test_rmse_mean  = double(),        # test_rmse_mean  at best_iter_cv  (selector)
  test_rmse_std   = double()         # test_rmse_std   at best_iter_cv
)

cv_eval_list <- vector("list", nrow(grid))   # per-config evaluation logs
cv_mod_list  <- vector("list", nrow(grid))   

# Track overall best (by CV fold-mean on test)
best_cv_rmse   <- Inf
best_grid_id   <- NA_integer_
best_params    <- list()
best_iter_cv   <- NA_integer_
best_cv_eval   <- NULL
best_row_full  <- NULL

set.seed(123)
tictoc::tic("Tuning (xgb.cv, internal folds)")
for (i in seq_len(nrow(grid))) {
  row <- grid[i, ]
  
  set.seed(123)
  
  message(sprintf("CV %02d/%02d | nrounds=%d, max_depth=%d, eta=%.3f",
                  i, nrow(grid), row$nrounds, row$max_depth, row$eta))
  
  params <- list(
    objective   = "reg:squarederror",
    max_depth   = row$max_depth,
    eta         = row$eta,
    eval_metric = "rmse",
    nthread     = max(1, parallel::detectCores() - 1)
    #seed = 123  
  )
  
  cv_mod <- xgb.cv(
    params  = params,
    data    = dtrain,
    nrounds = row$nrounds,
    nfold   = 5,
    prediction = FALSE,
    showsd  = TRUE,
    verbose = TRUE,
    stratified = FALSE,
    early_stopping_rounds = NULL
  )
  
  # Best iteration by fold-mean validation RMSE
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
  
  # Update global best (strict '<' keeps first in case of ties — favors smaller nrounds by grid order)
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
# > Tuning (xgb.cv, internal folds): 128.102 sec elapsed

#### ---- Explore tuning output ----

tuning_results %>% 
  arrange(test_rmse_mean) %>% 
  head(10) %>% 
  as.data.frame() %>% 
  print(row.names = FALSE)
# > grid_id                          param_name nrounds max_depth  eta best_iter_cv train_rmse_mean train_rmse_std test_rmse_mean test_rmse_std
# >      11 nrounds=200, max_depth=4, eta=0.050     200         4 0.05          101        87.30551      0.2878251       89.42718      1.226467
# >      12 nrounds=300, max_depth=4, eta=0.050     300         4 0.05          101        87.30551      0.2878251       89.42718      1.226467
# >      10 nrounds=100, max_depth=4, eta=0.050     100         4 0.05           99        87.34344      0.2868435       89.42888      1.224456
# >      19 nrounds=100, max_depth=4, eta=0.100     100         4 0.10           53        87.18464      0.3030843       89.47921      1.266622
# >      20 nrounds=200, max_depth=4, eta=0.100     200         4 0.10           53        87.18464      0.3030843       89.47921      1.266622
# >      21 nrounds=300, max_depth=4, eta=0.100     300         4 0.10           53        87.18464      0.3030843       89.47921      1.266622
# >      28 nrounds=100, max_depth=4, eta=0.300     100         4 0.30           15        87.37652      0.2721233       89.52342      1.300937
# >      29 nrounds=200, max_depth=4, eta=0.300     200         4 0.30           15        87.37652      0.2721233       89.52342      1.300937
# >      30 nrounds=300, max_depth=4, eta=0.300     300         4 0.30           15        87.37652      0.2721233       89.52342      1.300937
# >      13 nrounds=100, max_depth=6, eta=0.050     100         6 0.05           95        83.94781      0.3383981       90.29572      1.463622

best_params
best_row_full
# > nrounds max_depth  eta
# >      11     200         4 0.05

message(sprintf("Selected grid_id = %d | nrounds=%d, max_depth=%d, eta=%.3f | best_iter_cv = %d | cv_min = %.5f",
                best_grid_id, best_row_full$nrounds, best_row_full$max_depth, best_row_full$eta,
                best_iter_cv, best_cv_rmse))
# > Selected grid_id = 11 | nrounds=200, max_depth=4, eta=0.050 | best_iter_cv = 101 | cv_min = 89.42718


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
# >    1  485.79125  486.96908
# >    2  462.34541  463.58341
# >  ...
# >   99   87.64847   86.09023
# >  100   87.63519   86.07881
# >  101   87.61843   86.06715

xgb.importance(model = main_model)  # <=> xgb.importance(model = main_model, trees = 0:(best_iter_cv - 1))
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.4402620 0.2402601 0.2099010
# > 2: STUDYHMW 0.2211476 0.2649644 0.2825083
# > 3: EXERPRAC 0.2146435 0.3030674 0.2825083
# > 4: WORKHOME 0.1239469 0.1917081 0.2250825


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
pred_train <- predict(main_model, dtrain)  # model has exactly best_iter_cv trees
pred_valid <- predict(main_model, dvalid)
pred_test  <- predict(main_model, dtest)

# Optional equivalence checks (iterationrange vs default)
stopifnot(all.equal(
  predict(main_model, dtrain, iterationrange = c(1, best_iter_cv + 1)),
  pred_train
))
stopifnot(all.equal(
  predict(main_model, dvalid, iterationrange = c(1, best_iter_cv + 1)),
  pred_valid
))
stopifnot(all.equal(
  predict(main_model, dtest,  iterationrange = c(1, best_iter_cv + 1)),
  pred_test
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
# >   Training 87.61843 70.27426 -2.823713 2.799273 0.12388001
# > Validation 86.06715 68.86083 -6.273885 1.912833 0.09396712
# >       Test 89.56018 71.53846 -4.603330 2.530886 0.10413815

# Summary message
message(sprintf("Selected config -> grid_id=%d | nrounds cap=%d | max_depth=%d | eta=%.3f | best_iter_cv=%d | cv_min=%.5f",
                best_grid_id, best_row_full$nrounds, best_row_full$max_depth, best_row_full$eta, best_iter_cv, best_cv_rmse))
# > Selected config -> grid_id=11 | nrounds cap=200 | max_depth=4 | eta=0.050 | best_iter_cv=101 | cv_min=89.42718
