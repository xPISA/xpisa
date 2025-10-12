# ---- II. Predictive Modelling: Version 2.1 ----

# xgb.train with hyperparameter tuning

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

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("tidyverse","xgboost","Matrix","haven","broom","tictoc","DiagrammeR"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         tidyverse             xgboost              Matrix               haven               broom              tictoc          DiagrammeR 
# > "tidyverse 2.0.0"  "xgboost 1.7.11.1"      "Matrix 1.7.3"       "haven 2.5.4"       "broom 1.0.8"      "tictoc 1.2.1" "DiagrammeR 1.0.11"

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

## ---- 2. PV1MATH only ----

# --- Remark ---
# 1) Repeat the same process for PV2MATH - PV10MATH.
# 2) Apply best results from PV1MATH to all plausible values in mathematics. 

### ---- Fit main model using final student weights (W_FSTUWT) on the training data ----

#### ---- Tune XGBoost model for PV1MATH only: xgb.train ----

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

# Define hyperparameter grid (3 × 3 × 4 = 36 combinations)
grid <- expand.grid(
  nrounds = c(100, 200, 300),       # Max number of boosting iterations.
  max_depth = c(4, 6, 8),           # Maximum depth of a tree. Default: 6.
  eta = c(0.01, 0.05, 0.1, 0.3),    # eta control the learning rate: 0 < eta < 1.
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

# Initialize results table to track tuning performance
tuning_results <- tibble(
  grid_id = integer(),
  param_name = character(),
  nrounds = integer(),
  max_depth = integer(),
  eta = double(),
  best_iter_in_grid = integer(),     # Best iteration (boosting round) with lowest valid RMSE
  train_rmse = double(),             # Train RMSE at best iteration
  valid_rmse = double()              # Validation RMSE at best iteration
)

# Store all trained models
model_list <- vector("list", nrow(grid))  

# Store evaluation logs for all runs (for visualization later)
eval_log_list <- list()

# Track best model and its metrics
best_rmse <- Inf        # best valid rmse
best_model <- NULL      # model with best valid rmse
best_params <- list()
best_eval_log <- NULL
best_iter <- NULL
best_grid_id <- NULL

set.seed(123)                                          
tic("Tuning (xgb.train)")
for (i in seq_len(nrow(grid))) {                           # nrow(grid) = 36
  row <- grid[i, ]
  
  # Print the current hyperparameter combination
  message(sprintf("Fitting model %d/%d: nrounds = %d, max_depth = %d, eta = %.3f", 
                  i, nrow(grid), row$nrounds, row$max_depth, row$eta))
  
  # Set model parameters
  params <- list(
    objective = "reg:squarederror",                         # Specify the learning task and the corresponding learning objective: reg:squarederror Regression with squared loss (Default)
    max_depth = row$max_depth,                          
    eta = row$eta,                                      
    eval_metric = "rmse",                                   # Default: metric will be assigned according to objective(rmse for regression) 
    nthread = max(1, parallel::detectCores() - 1)           # Manually specify number of thread
    #seed = 123                                             # Random number seed for reproducibility (e.g. tune subsample, colsample_bytree for regularization)
  )
  
  # Train model on current combination of hyperparameters
  mod <- xgb.train(                                         # xgb.train, ?xgb.train
    params = params,
    data = dtrain,
    nrounds = row$nrounds,
    watchlist = list(train = dtrain, valid = dvalid),       # Fit on TRAIN and evaluate on VALID
    verbose = 1,                                            # If 0, xgboost will stay silent. If 1 (default), it will print information about performance. If 2, some additional information will be printed out.
    early_stopping_rounds = NULL  
  )
  
  # Save current model
  model_list[[i]] <- mod  
  
  # Extract log of RMSE over boosting rounds
  eval_log_i <- mod$evaluation_log
  best_iter_i <- which.min(eval_log_i$valid_rmse)           # Best iteration (boosting round) with lowest VALID rmse
  best_train_rmse_i <- eval_log_i$train_rmse[best_iter_i]   # TRAIN rmse at best iteration (not the lowest) 
  best_valid_rmse_i <- eval_log_i$valid_rmse[best_iter_i]   # Lowest VALID rmse
  
  # # --- (Optional) Verify: logged RMSE is truly *weighted* (VALID & TRAIN) ---
  # # Predict at the selected iteration for this config
  # pred_valid_i <- predict(mod, dvalid, iterationrange = c(1, best_iter_i + 1))
  # pred_train_i <- predict(mod, dtrain, iterationrange = c(1, best_iter_i + 1))
  # 
  # # sanity: iterationrange vs ntreelimit equivalence
  # stopifnot(all.equal(
  #   predict(mod, dvalid, iterationrange = c(1, best_iter_i + 1)),
  #   predict(mod, dvalid, ntreelimit = best_iter_i)
  # ))
  # 
  # # Manual weighted RMSE (use your compute_metrics helper for consistency)
  # wrmse_valid_i <- compute_metrics(y_true = y_valid, y_pred = pred_valid_i, w = w_valid)["rmse"]
  # wrmse_train_i <- compute_metrics(y_true = y_train, y_pred = pred_train_i, w = w_train)["rmse"]
  # 
  # # Compare with xgboost's evaluation_log at best_iter_i
  # if (!isTRUE(all.equal(best_valid_rmse_i, wrmse_valid_i))) {
  #   warning(sprintf(
  #     "Weighted VALID RMSE mismatch (grid_id=%d, iter=%d): log=%.8f vs manual=%.8f",
  #     i, best_iter_i, best_valid_rmse_i, wrmse_valid_i
  #   ))
  # }
  # if (!isTRUE(all.equal(best_train_rmse_i, wrmse_train_i))) {
  #   warning(sprintf(
  #     "Weighted TRAIN RMSE mismatch (grid_id=%d, iter=%d): log=%.8f vs manual=%.8f",
  #     i, best_iter_i, best_train_rmse_i, wrmse_train_i
  #   ))
  # }
  # # --- end verification ---
  
  # Store log with grid ID for plotting later
  eval_log_list[[i]] <- eval_log_i |> mutate(grid_id = i)
  
  # Save current result summary to tuning_results
  tuning_results <- add_row(tuning_results,
                            grid_id = i,
                            param_name = paste0("nrounds=", row$nrounds, ", max_depth=", row$max_depth, ", eta=", row$eta),
                            nrounds = row$nrounds,
                            max_depth = row$max_depth,
                            eta = row$eta,
                            best_iter_in_grid = best_iter_i,
                            train_rmse = best_train_rmse_i,
                            valid_rmse = best_valid_rmse_i
  )
  
  # Update best model tracking if current RMSE is better
  # NOTE: strict '<' means exact ties keep the first encountered config.
  # With expand.grid ordering (nrounds slowest varying: 100 <- 200 <- 300),
  # ties favor smaller nrounds (simpler), which is desirable.
  if (best_valid_rmse_i < best_rmse) {
    best_rmse <- best_valid_rmse_i
    best_model <- mod
    best_params <- params
    best_eval_log <- eval_log_i
    best_iter <- best_iter_i
    best_grid_id <- i
  }
}
toc()
# > Tuning: 32.582 sec elapsed

##### ----  Explore tuning output ----
length(model_list)  #36
model_list[[1]]
print(as.data.frame(model_list[[1]]$evaluation_log))
length(eval_log_list)  #36
print(as.data.frame(eval_log_list[[1]]))

tuning_results
#tuning_results <- tuning_results %>% mutate(gap = valid_rmse - train_rmse)
print(head(as.data.frame(tuning_results %>% arrange(valid_rmse))), row.names = FALSE)   # Tie break: e.g. grid_id 11 vs 12, smaller full nrounds is chosen in tuning
# >  grid_id                         param_name nrounds max_depth  eta best_iter_in_grid train_rmse valid_rmse
# >       11 nrounds=200, max_depth=4, eta=0.05     200         4 0.05               180   86.88122   85.92491
# >       12 nrounds=300, max_depth=4, eta=0.05     300         4 0.05               180   86.88122   85.92491 
# >       19  nrounds=100, max_depth=4, eta=0.1     100         4 0.10                86   86.90028   85.97932
# >       20  nrounds=200, max_depth=4, eta=0.1     200         4 0.10                86   86.90028   85.97932
# >       21  nrounds=300, max_depth=4, eta=0.1     300         4 0.10                86   86.90028   85.97932
# >       10 nrounds=100, max_depth=4, eta=0.05     100         4 0.05               100   87.63519   86.07881
print(head(as.data.frame(tuning_results %>% arrange(valid_rmse, best_iter_in_grid, nrounds))), row.names = FALSE)  # Tie-break
print(as.data.frame(tuning_results %>% arrange(valid_rmse)), row.names = FALSE)  # Print all
#print(tuning_results %>% arrange(valid_rmse), n = Inf)

# --- Explore best_model ---
best_model                  # str(best_model)                 
best_model$evaluation_log   # <=> best_eval_log
best_rmse
# > [1] 85.92491
best_params
best_params$max_depth; best_params$eta
# > [1] 4
# > [1] 0.05
best_iter
# > [1] 180
best_grid_id
# > [1] 11
grid[best_grid_id, ]
# >    nrounds max_depth  eta
# > 11     200         4 0.05

print(as.data.frame(best_model$evaluation_log))  # grid_id=11 nrounds=200, max_depth=4, eta=0.05
best_model$evaluation_log |>                     # <=> model_list[[best_grid_id]]$evaluation_log |>
  pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = best_iter, linetype = 2, alpha = 0.4) +
  annotate("text", x = best_iter, y = max(best_model$evaluation_log$valid_rmse) + 5,
           label = paste("Best iter =", best_iter), size = 3, vjust = 0) + 
  labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal()                                # Visualizing the best path

print(as.data.frame(model_list[[12]]$evaluation_log))  # Comparing a tie configuration: grid_id=12 nrounds=300, max_depth=4, eta=0.05
model_list[[12]]$evaluation_log |>
  pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = which.min(model_list[[12]]$evaluation_log$valid_rmse), linetype = 2, alpha = 0.4) +
  annotate("text", x = which.min(model_list[[12]]$evaluation_log$valid_rmse), y = max(model_list[[12]]$evaluation_log$valid_rmse) + 5,
           label = paste("Best iter =", which.min(model_list[[12]]$evaluation_log$valid_rmse)), size = 3, vjust = 0) + 
  labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal()

xgb.importance(model = best_model)                            #?xgb.importance
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.4075640 0.1903748 0.1893333
# > 2: STUDYHMW 0.2310851 0.2576967 0.2863333
# > 3: EXERPRAC 0.2201593 0.3060447 0.2736667
# > 4: WORKHOME 0.1411917 0.2458839 0.2506667

xgb.importance(model = best_model, trees = 0:(best_iter - 1))  # 0-based indices
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.4113538 0.1979506 0.1903704
# > 2: STUDYHMW 0.2303721 0.2482283 0.2833333
# > 3: EXERPRAC 0.2191372 0.3101510 0.2755556
# > 4: WORKHOME 0.1391370 0.2436702 0.2507407

xgb.plot.importance(importance_matrix = xgb.importance(model = best_model, trees = 0:(best_iter - 1)),
                    top_n = NULL,                              # Top n features (you only have 4 in oos)
                    measure = "Gain",                          # Can also use "Cover" or "Frequency"
                    rel_to_first = TRUE,
                    xlab = "Relative Importance")

xgb.plot.tree(model = best_model, trees = 0)               # or trees = 1, 2, etc. 
xgb.plot.tree(model = best_model, trees = best_iter-1)     # trees = best_iter-1 as trees = 0 may not be representative


#### ---- Predict and evaluate performance on the training/validation/test datasets ----

# Predict using best_model at best_iter 
pred_train <- predict(best_model, dtrain, iterationrange = c(1, best_iter + 1))   # ?predict.xgb.Booster; Can try predinteraction = TRUE or predcontrib = TRUE (SHAP)
pred_valid <- predict(best_model, dvalid, iterationrange = c(1, best_iter + 1))   # iterationrange = c(1, best_iter + 1) <=> ntreelimit = best_iter (depreciated)  
pred_test  <- predict(best_model, dtest, iterationrange = c(1, best_iter + 1))

# Compute weighted metrics
metrics_train <- compute_metrics(y_true = y_train, y_pred = pred_train, w = w_train)
metrics_valid <- compute_metrics(y_true = y_valid, y_pred = pred_valid, w = w_valid)
metrics_test  <- compute_metrics(y_true = y_test,  y_pred = pred_test,  w = w_test)

# Combine and display results 
metric_results <- tibble::tibble(
  Dataset = c("Training", "Validation", "Test"),
  #MSE     = c(metrics_train["mse"],      metrics_valid["mse"],      metrics_test["mse"]),       # optional
  RMSE     = c(metrics_train["rmse"],     metrics_valid["rmse"],     metrics_test["rmse"]),
  MAE      = c(metrics_train["mae"],      metrics_valid["mae"],      metrics_test["mae"]),
  Bias     = c(metrics_train["bias"],     metrics_valid["bias"],     metrics_test["bias"]),
  `Bias%`  = c(metrics_train["bias_pct"], metrics_valid["bias_pct"], metrics_test["bias_pct"]),
  R2       = c(metrics_train["r2"],       metrics_valid["r2"],       metrics_test["r2"])
)

print(as.data.frame(metric_results), row.names = FALSE)
# >    Dataset     RMSE      MAE        Bias    Bias%         R2
# >   Training 86.88122 69.60903 -0.04917408 3.323704 0.13856116
# > Validation 85.92491 68.63517 -3.60089285 2.439683 0.09695929
# >       Test 89.53851 71.55340 -1.67299135 3.117512 0.10457167

##### ---- Explore SHAP ----

# Remark: SHAP is for interpretation; do not feed these into evaluation metrics.

####### ---- Per-feature SHAP ----
# Interpretation: For any row, prediction ≈ BIAS + sum(feature SHAPs).

# If plain predictions at the tuned round don’t already exist, compute them now:
if (!exists("pred_train")) {
  pred_train <- predict(best_model, dtrain, iterationrange = c(1, best_iter + 1))
  pred_valid <- predict(best_model, dvalid, iterationrange = c(1, best_iter + 1))
  pred_test  <- predict(best_model, dtest,  iterationrange = c(1, best_iter + 1))
}

# Helper: get SHAP (predcontrib) + sanity checks, aligned to best_iter
get_shap <- function(model, dmat, preds, features) {
  pred_shap <- predict(model, dmat, predcontrib = TRUE, iterationrange = c(1, best_iter + 1))  # n x (p+1); last col = "BIAS"
  # shape checks
  stopifnot(ncol(pred_shap) == length(features) + 1L)
  stopifnot(colnames(pred_shap)[ncol(pred_shap)] == "BIAS")
  # SHAP rows must sum to the prediction (squared-error objective ⇒ raw scores)
  stopifnot(isTRUE(all.equal(rowSums(pred_shap), preds, tolerance = 1e-6)))
  pred_shap
}

shap_train <- get_shap(best_model, dtrain, pred_train, oos)
shap_valid <- get_shap(best_model, dvalid, pred_valid, oos)
shap_test  <- get_shap(best_model, dtest,  pred_test,  oos)

# Helper: weighted global importance (mean |SHAP|), excluding BIAS
global_shap <- function(shap_mat, w, features) {
  tibble::tibble(
    feature = features,
    mean_abs_shap = colSums(abs(shap_mat[, features, drop = FALSE]) * w) / sum(w)
  ) |>
    dplyr::arrange(dplyr::desc(mean_abs_shap))
}

global_importance_train <- global_shap(shap_train, w_train, oos)
global_importance_valid <- global_shap(shap_valid, w_valid, oos)
global_importance_test  <- global_shap(shap_test,  w_test,  oos)

print(as.data.frame(global_importance_train), row.names=FALSE)
# >  feature mean_abs_shap
# >  WORKPAY     14.289040
# > EXERPRAC     10.418595
# > STUDYHMW      8.056279
# > WORKHOME      6.396722
print(as.data.frame(global_importance_valid), row.names=FALSE)
# >  feature mean_abs_shap
# >  WORKPAY     14.834166
# > EXERPRAC     10.588882
# > STUDYHMW      7.965808
# > WORKHOME      6.465970
print(as.data.frame(global_importance_test), row.names=FALSE)
# >  feature mean_abs_shap
# >  WORKPAY     14.686324
# > EXERPRAC     10.254455
# > STUDYHMW      8.152814
# > WORKHOME      6.312677

# Simple bar plots
plot_global <- function(tab, title) {
  ggplot2::ggplot(tab, ggplot2::aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::labs(x = "Feature", y = "Weighted mean |SHAP|", title = title) +
    ggplot2::theme_minimal()
}
plot_global(global_importance_train, "Global SHAP importance (training)")
plot_global(global_importance_valid, "Global SHAP importance (validation)")
plot_global(global_importance_test,  "Global SHAP importance (test)")



####### ---- SHAP Interactions ----

######## ---- SHAP Interactions (training) ----
# Assumes these already exist: best_model, dtrain, train_data, w_train, oos, pred_train, shap_train
# Note: the interaction tensor can be heavy when feature count grows.

# 1) Interaction tensor at tuned round (best_iter)
interaction_tensor_train <- predict(
  best_model, dtrain, predinteraction = TRUE, iterationrange = c(1, best_iter + 1)  # end is exclusive
)

# 2) Identity check on one row (features-only sum equals predcontrib SHAP)
example_row_id <- 1L
interaction_matrix_example <- interaction_tensor_train[example_row_id, oos, oos, drop = FALSE][1, , ]
shap_from_interactions_example <- rowSums(interaction_matrix_example)
stopifnot(isTRUE(all.equal(
  shap_from_interactions_example,
  shap_train[example_row_id, oos],
  tolerance = 1e-6
)))

# 3) Weighted mean |interaction| per pair over training rows
interaction_avg_matrix_full <- apply(
  interaction_tensor_train[, oos, oos, drop = FALSE],
  c(2, 3),
  function(x) stats::weighted.mean(abs(x), w_train, na.rm = TRUE)
)
interaction_avg_matrix_full <- (interaction_avg_matrix_full + t(interaction_avg_matrix_full)) / 2
diag(interaction_avg_matrix_full) <- NA   # drop main effects on the diagonal

# 4) Unique-pair table (upper triangle only)
interaction_avg_matrix_masked <- interaction_avg_matrix_full
interaction_avg_matrix_masked[lower.tri(interaction_avg_matrix_masked, diag = TRUE)] <- NA

interaction_table <- as.data.frame(as.table(interaction_avg_matrix_masked)) |>
  dplyr::mutate(
    Var1 = factor(Var1, levels = oos),
    Var2 = factor(Var2, levels = oos)
  ) |>
  dplyr::filter(!is.na(Freq)) |>
  dplyr::arrange(dplyr::desc(Freq)) |>
  dplyr::rename(feature_i = Var1, feature_j = Var2, mean_abs_interaction = Freq)

print(interaction_table)
# >   feature_i feature_j mean_abs_interaction
# > 1  EXERPRAC  WORKHOME             2.294732
# > 2  EXERPRAC   WORKPAY             1.942468
# > 3  EXERPRAC  STUDYHMW             1.793645
# > 4  STUDYHMW   WORKPAY             1.597582
# > 5  STUDYHMW  WORKHOME             1.455196
# > 6   WORKPAY  WORKHOME             1.211943

# 5) Share of interactions in overall explanation (heuristic)
main_abs_mean <- colSums(abs(shap_train[, oos, drop = FALSE]) * w_train, na.rm = TRUE) / sum(w_train, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (training): %.2f%%\n", 100 * interaction_share))
# > Interaction share (training): 20.82%

# 6) Heatmap
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = oos),
    feature_j = factor(feature_j, levels = oos)
  )

ggplot2::ggplot(interaction_heat_df, ggplot2::aes(feature_i, feature_j, fill = mean_abs_interaction)) +
  ggplot2::geom_tile(na.rm = TRUE) +
  ggplot2::scale_x_discrete(drop = FALSE) +
  ggplot2::scale_y_discrete(drop = FALSE) +
  ggplot2::coord_equal() +
  ggplot2::labs(x = "", y = "", fill = "Weighted mean |interaction|",
                title = "SHAP interaction heatmap (training)") +
  ggplot2::theme_minimal()

# 7) Per-feature interaction strength (use FULL matrix)
interaction_strength_table <- tibble::tibble(
  feature = oos,
  mean_abs_interaction_with_others = sapply(oos, function(fi) {
    partners <- setdiff(oos, fi)
    mean(interaction_avg_matrix_full[fi, partners], na.rm = TRUE)
  })
) |>
  dplyr::arrange(dplyr::desc(mean_abs_interaction_with_others))

print(as.data.frame(interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         2.010282
# > WORKHOME                         1.653958
# > STUDYHMW                         1.615474
# >  WORKPAY                         1.583998

ggplot2::ggplot(
  interaction_strength_table,
  ggplot2::aes(x = stats::reorder(feature, mean_abs_interaction_with_others),
               y = mean_abs_interaction_with_others)
) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::labs(x = "Feature", y = "Mean |interaction| with others",
                title = "Per-feature interaction strength (training)") +
  ggplot2::theme_minimal()

# 8) Inspect one pair in detail (dependence-style plot of the interaction term)
as_plain_num <- function(x) {
  x |> haven::zap_missing() |> haven::zap_labels() |> vctrs::vec_data()
}

interaction_pair <- c("EXERPRAC", "STUDYHMW")  # change as needed
interaction_pair_df <- tibble::tibble(
  x_i = as_plain_num(train_data[[interaction_pair[1]]]),
  x_j = as_plain_num(train_data[[interaction_pair[2]]]),
  shap_interaction_ij = interaction_tensor_train[, interaction_pair[1], interaction_pair[2]],
  w   = w_train
) |> tidyr::drop_na()

# Option A — points colored by continuous x_j; ONE weighted smooth line
ggplot2::ggplot(interaction_pair_df, ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j)) +
  ggplot2::geom_point(alpha = 0.35) +
  ggplot2::geom_smooth(
    mapping = ggplot2::aes(x = x_i, y = shap_interaction_ij, group = 1, weight = w),
    inherit.aes = FALSE, method = "loess", se = FALSE, color = "black"
  ) +
  ggplot2::labs(x = interaction_pair[1],
                y = paste("SHAP interaction:", paste(interaction_pair, collapse = " × ")),
                color = interaction_pair[2],
                title = "Interaction effect (training)") +
  ggplot2::theme_minimal()

# Option B — separate smooths by quantile bins
interaction_pair_df_bins <- interaction_pair_df |>
  dplyr::mutate(x_j_bin = factor(dplyr::ntile(x_j, 4), labels = paste0("Q", 1:4)))

ggplot2::ggplot(interaction_pair_df_bins,
                ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j_bin)) +
  ggplot2::geom_point(alpha = 0.25) +
  ggplot2::geom_smooth(ggplot2::aes(weight = w), method = "loess", se = FALSE) +
  ggplot2::labs(x = interaction_pair[1],
                y = paste("SHAP interaction:", paste(interaction_pair, collapse = " × ")),
                color = paste0(interaction_pair[2], " quantile"),
                title = "Interaction effect by quantile (training)") +
  ggplot2::theme_minimal()


######## ---- SHAP Interactions (validation) ----
# Assumes: best_model, dvalid, valid_data, w_valid, oos, pred_valid, shap_valid

# 1) Interaction tensor
interaction_tensor_valid <- predict(
  best_model, dvalid, predinteraction = TRUE, iterationrange = c(1, best_iter + 1)
)

# 2) Identity check
example_row_id <- 1L
interaction_matrix_example <- interaction_tensor_valid[example_row_id, oos, oos, drop = FALSE][1, , ]
shap_from_interactions_example <- rowSums(interaction_matrix_example)
stopifnot(isTRUE(all.equal(
  shap_from_interactions_example,
  shap_valid[example_row_id, oos],
  tolerance = 1e-6
)))

# 3) Weighted mean |interaction| per pair
interaction_avg_matrix_full <- apply(
  interaction_tensor_valid[, oos, oos, drop = FALSE],
  c(2, 3),
  function(x) stats::weighted.mean(abs(x), w_valid, na.rm = TRUE)
)
interaction_avg_matrix_full <- (interaction_avg_matrix_full + t(interaction_avg_matrix_full)) / 2
diag(interaction_avg_matrix_full) <- NA

# 4) Unique-pair table
interaction_avg_matrix_masked <- interaction_avg_matrix_full
interaction_avg_matrix_masked[lower.tri(interaction_avg_matrix_masked, diag = TRUE)] <- NA

interaction_table <- as.data.frame(as.table(interaction_avg_matrix_masked)) |>
  dplyr::mutate(
    Var1 = factor(Var1, levels = oos),
    Var2 = factor(Var2, levels = oos)
  ) |>
  dplyr::filter(!is.na(Freq)) |>
  dplyr::arrange(dplyr::desc(Freq)) |>
  dplyr::rename(feature_i = Var1, feature_j = Var2, mean_abs_interaction = Freq)

print(interaction_table)
# >   feature_i feature_j mean_abs_interaction
# > 1  EXERPRAC  WORKHOME             2.347426
# > 2  EXERPRAC   WORKPAY             2.047872
# > 3  EXERPRAC  STUDYHMW             1.770241
# > 4  STUDYHMW   WORKPAY             1.630730
# > 5  STUDYHMW  WORKHOME             1.465484
# > 6   WORKPAY  WORKHOME             1.272215

# 5) Share of interactions
main_abs_mean <- colSums(abs(shap_valid[, oos, drop = FALSE]) * w_valid, na.rm = TRUE) / sum(w_valid, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (validation): %.2f%%\n", 100 * interaction_share))
# > Interaction share (validation): 20.91%

# 6) Heatmap
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = oos),
    feature_j = factor(feature_j, levels = oos)
  )

ggplot2::ggplot(interaction_heat_df, ggplot2::aes(feature_i, feature_j, fill = mean_abs_interaction)) +
  ggplot2::geom_tile(na.rm = TRUE) +
  ggplot2::scale_x_discrete(drop = FALSE) +
  ggplot2::scale_y_discrete(drop = FALSE) +
  ggplot2::coord_equal() +
  ggplot2::labs(x = "", y = "", fill = "Weighted mean |interaction|",
                title = "SHAP interaction heatmap (validation)") +
  ggplot2::theme_minimal()

# 7) Per-feature interaction strength
interaction_strength_table <- tibble::tibble(
  feature = oos,
  mean_abs_interaction_with_others = sapply(oos, function(fi) {
    partners <- setdiff(oos, fi)
    mean(interaction_avg_matrix_full[fi, partners], na.rm = TRUE)
  })
) |>
  dplyr::arrange(dplyr::desc(mean_abs_interaction_with_others))

print(as.data.frame(interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         2.055180
# > WORKHOME                         1.695042
# >  WORKPAY                         1.650272
# > STUDYHMW                         1.622152

ggplot2::ggplot(
  interaction_strength_table,
  ggplot2::aes(x = stats::reorder(feature, mean_abs_interaction_with_others),
               y = mean_abs_interaction_with_others)
) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::labs(x = "Feature", y = "Mean |interaction| with others",
                title = "Per-feature interaction strength (validation)") +
  ggplot2::theme_minimal()


######## ---- SHAP Interactions (test) ----
# Assumes: best_model, dtest, test_data, w_test, oos, pred_test, shap_test

# 1) Interaction tensor
interaction_tensor_test <- predict(
  best_model, dtest, predinteraction = TRUE, iterationrange = c(1, best_iter + 1)
)

# 2) Identity check
example_row_id <- 1L
interaction_matrix_example <- interaction_tensor_test[example_row_id, oos, oos, drop = FALSE][1, , ]
shap_from_interactions_example <- rowSums(interaction_matrix_example)
stopifnot(isTRUE(all.equal(
  shap_from_interactions_example,
  shap_test[example_row_id, oos],
  tolerance = 1e-6
)))

# 3) Weighted mean |interaction| per pair
interaction_avg_matrix_full <- apply(
  interaction_tensor_test[, oos, oos, drop = FALSE],
  c(2, 3),
  function(x) stats::weighted.mean(abs(x), w_test, na.rm = TRUE)
)
interaction_avg_matrix_full <- (interaction_avg_matrix_full + t(interaction_avg_matrix_full)) / 2
diag(interaction_avg_matrix_full) <- NA

# 4) Unique-pair table
interaction_avg_matrix_masked <- interaction_avg_matrix_full
interaction_avg_matrix_masked[lower.tri(interaction_avg_matrix_masked, diag = TRUE)] <- NA

interaction_table <- as.data.frame(as.table(interaction_avg_matrix_masked)) |>
  dplyr::mutate(
    Var1 = factor(Var1, levels = oos),
    Var2 = factor(Var2, levels = oos)
  ) |>
  dplyr::filter(!is.na(Freq)) |>
  dplyr::arrange(dplyr::desc(Freq)) |>
  dplyr::rename(feature_i = Var1, feature_j = Var2, mean_abs_interaction = Freq)

print(interaction_table)
# >   feature_i feature_j mean_abs_interaction
# > 1  EXERPRAC  WORKHOME             2.309679
# > 2  EXERPRAC   WORKPAY             1.943255
# > 3  EXERPRAC  STUDYHMW             1.766477
# > 4  STUDYHMW   WORKPAY             1.611564
# > 5  STUDYHMW  WORKHOME             1.476861
# > 6   WORKPAY  WORKHOME             1.258267

# 5) Share of interactions
main_abs_mean <- colSums(abs(shap_test[, oos, drop = FALSE]) * w_test, na.rm = TRUE) / sum(w_test, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (test): %.2f%%\n", 100 * interaction_share))
# > Interaction share (test): 20.83%

# 6) Heatmap
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = oos),
    feature_j = factor(feature_j, levels = oos)
  )

ggplot2::ggplot(interaction_heat_df, ggplot2::aes(feature_i, feature_j, fill = mean_abs_interaction)) +
  ggplot2::geom_tile(na.rm = TRUE) +
  ggplot2::scale_x_discrete(drop = FALSE) +
  ggplot2::scale_y_discrete(drop = FALSE) +
  ggplot2::coord_equal() +
  ggplot2::labs(x = "", y = "", fill = "Weighted mean |interaction|",
                title = "SHAP interaction heatmap (test)") +
  ggplot2::theme_minimal()

# 7) Per-feature interaction strength
interaction_strength_table <- tibble::tibble(
  feature = oos,
  mean_abs_interaction_with_others = sapply(oos, function(fi) {
    partners <- setdiff(oos, fi)
    mean(interaction_avg_matrix_full[fi, partners], na.rm = TRUE)
  })
) |>
  dplyr::arrange(dplyr::desc(mean_abs_interaction_with_others))

print(as.data.frame(interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         2.006471
# > WORKHOME                         1.681603
# > STUDYHMW                         1.618300
# >  WORKPAY                         1.604362

ggplot2::ggplot(
  interaction_strength_table,
  ggplot2::aes(x = stats::reorder(feature, mean_abs_interaction_with_others),
               y = mean_abs_interaction_with_others)
) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::labs(x = "Feature", y = "Mean |interaction| with others",
                title = "Per-feature interaction strength (test)") +
  ggplot2::theme_minimal()

######## ---- ** SHAP Interactions (training / validation / test) ** ----

# Remark: Consolidated SHAP-interaction helpers for train/valid/test.

# --- Helpers ---

# Convert SPSS-labelled vectors to plain numeric
as_plain_num <- function(x) {
  x |> haven::zap_missing() |> haven::zap_labels() |> vctrs::vec_data()
}

# Core computation: interaction tensor, averages, tables, share, sanity checks
compute_shap_interactions <- function(model, dmat, data_frame, w_vec, features,
                                      split = "split", ntrees = best_iter) {
  stopifnot(all(features %in% colnames(data_frame)))
  
  # Predictions & SHAP at tuned round
  preds    <- predict(model, dmat, iterationrange = c(1, best_iter + 1))
  shap_mat <- predict(model, dmat, predcontrib = TRUE, iterationrange = c(1, best_iter + 1))  # n x (p+1), last col "BIAS"
  
  # SHAP sanity checks
  stopifnot(ncol(shap_mat) == length(features) + 1L)
  stopifnot(colnames(shap_mat)[ncol(shap_mat)] == "BIAS")
  stopifnot(isTRUE(all.equal(rowSums(shap_mat), preds, tolerance = 1e-6)))
  
  # Interactions at tuned round: n × (p+1) × (p+1)
  interaction_tensor <- predict(model, dmat, predinteraction = TRUE, iterationrange = c(1, best_iter + 1))
  
  # Identity (one row): sum of interactions across columns == per-feature SHAP
  example_row_id <- 1L
  interaction_matrix_example <- interaction_tensor[example_row_id, features, features, drop = FALSE][1, , ]
  shap_from_interactions_example <- rowSums(interaction_matrix_example)
  stopifnot(isTRUE(all.equal(shap_from_interactions_example,
                             shap_mat[example_row_id, features], tolerance = 1e-6)))
  
  # Weighted mean |interaction| per feature-pair across rows
  interaction_avg_matrix_full <- apply(
    interaction_tensor[, features, features, drop = FALSE],
    c(2, 3),
    function(x) stats::weighted.mean(abs(x), w_vec, na.rm = TRUE)
  )
  
  # Enforce symmetry; drop diagonal (main effects)
  interaction_avg_matrix_full <- (interaction_avg_matrix_full + t(interaction_avg_matrix_full)) / 2
  diag(interaction_avg_matrix_full) <- NA
  
  # Mask one triangle for a unique-pair table/heatmap
  interaction_avg_matrix_masked <- interaction_avg_matrix_full
  interaction_avg_matrix_masked[lower.tri(interaction_avg_matrix_masked, diag = TRUE)] <- NA
  
  # Sorted unique-pair table
  interaction_table <- as.data.frame(as.table(interaction_avg_matrix_masked)) |>
    dplyr::mutate(
      Var1 = factor(Var1, levels = features),
      Var2 = factor(Var2, levels = features)
    ) |>
    dplyr::filter(!is.na(Freq)) |>
    dplyr::arrange(dplyr::desc(Freq)) |>
    dplyr::rename(feature_i = Var1, feature_j = Var2, mean_abs_interaction = Freq)
  
  # Per-feature interaction strength (use FULL matrix)
  interaction_strength_table <- tibble::tibble(
    feature = features,
    mean_abs_interaction_with_others = sapply(features, function(fi) {
      partners <- setdiff(features, fi)
      mean(interaction_avg_matrix_full[fi, partners], na.rm = TRUE)
    })
  ) |>
    dplyr::arrange(dplyr::desc(mean_abs_interaction_with_others))
  
  # Heuristic share of interactions vs main effects
  main_abs_mean <- colSums(abs(shap_mat[, features, drop = FALSE]) * w_vec, na.rm = TRUE) / sum(w_vec, na.rm = TRUE)
  total_main  <- sum(main_abs_mean, na.rm = TRUE)
  total_pairs <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
  interaction_share <- total_pairs / (total_main + total_pairs)
  
  list(
    split = split,
    features = features,
    data = data_frame,
    weights = w_vec,
    interaction_tensor = interaction_tensor,
    interaction_avg_matrix_full = interaction_avg_matrix_full,
    interaction_avg_matrix_masked = interaction_avg_matrix_masked,
    interaction_table = interaction_table,
    interaction_strength_table = interaction_strength_table,
    interaction_share = interaction_share
  )
}

# Heatmap of pairwise mean |interaction|
plot_interaction_heatmap <- function(obj, title_prefix = "SHAP interaction heatmap") {
  heat_df <- tibble::as_tibble(obj$interaction_avg_matrix_masked, rownames = "feature_i") |>
    tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
    dplyr::mutate(
      feature_i = factor(feature_i, levels = obj$features),
      feature_j = factor(feature_j, levels = obj$features)
    )
  
  ggplot2::ggplot(heat_df, ggplot2::aes(feature_i, feature_j, fill = mean_abs_interaction)) +
    ggplot2::geom_tile(na.rm = TRUE) +
    ggplot2::scale_x_discrete(drop = FALSE) +
    ggplot2::scale_y_discrete(drop = FALSE) +
    ggplot2::coord_equal() +
    ggplot2::labs(x = "", y = "", fill = "Weighted mean |interaction|",
                  title = sprintf("%s (%s)", title_prefix, obj$split)) +
    ggplot2::theme_minimal()
}

# Bar chart: per-feature interaction strength
plot_interaction_strength <- function(obj, title_prefix = "Per-feature interaction strength") {
  tbl <- obj$interaction_strength_table
  print(as.data.frame(tbl), row.names = FALSE)
  
  ggplot2::ggplot(tbl,
                  ggplot2::aes(x = stats::reorder(feature, mean_abs_interaction_with_others),
                               y = mean_abs_interaction_with_others)) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::labs(x = "Feature", y = "Mean |interaction| with others",
                  title = sprintf("%s (%s)", title_prefix, obj$split)) +
    ggplot2::theme_minimal()
}

# Pairwise dependence-style plot of one interaction term
plot_interaction_pair <- function(obj, pair, quantiles = FALSE,
                                  title_prefix = "Interaction effect") {
  stopifnot(length(pair) == 2, all(pair %in% obj$features))
  
  df <- tibble::tibble(
    x_i = as_plain_num(obj$data[[pair[1]]]),
    x_j = as_plain_num(obj$data[[pair[2]]]),
    shap_interaction_ij = obj$interaction_tensor[, pair[1], pair[2]],
    w = obj$weights
  ) |> tidyr::drop_na()
  
  if (!quantiles) {
    ggplot2::ggplot(df, ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j)) +
      ggplot2::geom_point(alpha = 0.35) +
      ggplot2::geom_smooth(
        mapping = ggplot2::aes(x = x_i, y = shap_interaction_ij, group = 1, weight = w),
        inherit.aes = FALSE, method = "loess", se = FALSE, color = "black"
      ) +
      ggplot2::labs(x = pair[1],
                    y = paste("SHAP interaction:", paste(pair, collapse = " × ")),
                    color = pair[2],
                    title = sprintf("%s (%s)", title_prefix, obj$split)) +
      ggplot2::theme_minimal()
  } else {
    df_bins <- df |>
      dplyr::mutate(x_j_bin = factor(dplyr::ntile(x_j, 4), labels = paste0("Q", 1:4)))
    
    ggplot2::ggplot(df_bins,
                    ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j_bin)) +
      ggplot2::geom_point(alpha = 0.25) +
      ggplot2::geom_smooth(ggplot2::aes(weight = w), method = "loess", se = FALSE) +
      ggplot2::labs(x = pair[1],
                    y = paste("SHAP interaction:", paste(pair, collapse = " × ")),
                    color = paste0(pair[2], " quantile"),
                    title = sprintf("%s by quantile (%s)", title_prefix, obj$split)) +
      ggplot2::theme_minimal()
  }
}

# --- Run for all three splits (best_model @ best_iter) ---

shap_int_train <- compute_shap_interactions(
  model = best_model, dmat = dtrain, data_frame = train_data,
  w_vec = w_train, features = oos, split = "training"
)

shap_int_valid <- compute_shap_interactions(
  model = best_model, dmat = dvalid, data_frame = valid_data,
  w_vec = w_valid, features = oos, split = "validation"
)

shap_int_test <- compute_shap_interactions(
  model = best_model, dmat = dtest, data_frame = test_data,
  w_vec = w_test, features = oos, split = "test"
)

# --- Quick access & examples ---

# Tables / shares
shap_int_train$interaction_table |> as.data.frame() |> print(row.names = FALSE)
# > feature_i feature_j mean_abs_interaction
# >  EXERPRAC  WORKHOME             2.294732
# >  EXERPRAC   WORKPAY             1.942468
# >  EXERPRAC  STUDYHMW             1.793644
# >  STUDYHMW   WORKPAY             1.597582
# >  STUDYHMW  WORKHOME             1.455196
# >  WORKPAY  WORKHOME             1.211944
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_train$split, 100 * shap_int_train$interaction_share))
# > Interaction share (training): 20.82%
print(as.data.frame(shap_int_train$interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         2.010282
# > WORKHOME                         1.653957
# > STUDYHMW                         1.615474
# >  WORKPAY                         1.583998

shap_int_valid$interaction_table |> as.data.frame() |> print(row.names = FALSE)
# >  feature_i feature_j mean_abs_interaction
# >   EXERPRAC  WORKHOME             2.347427
# >   EXERPRAC   WORKPAY             2.047872
# >   EXERPRAC  STUDYHMW             1.770241
# >   STUDYHMW   WORKPAY             1.630730
# >   STUDYHMW  WORKHOME             1.465484
# >    WORKPAY  WORKHOME             1.272215
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_valid$split, 100 * shap_int_valid$interaction_share))
# > Interaction share (validation): 20.91%
print(as.data.frame(shap_int_valid$interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         2.055180
# > WORKHOME                         1.695042
# >  WORKPAY                         1.650272
# > STUDYHMW                         1.622152

shap_int_test$interaction_table  |> as.data.frame() |> print(row.names = FALSE)
# > feature_i feature_j mean_abs_interaction
# >  EXERPRAC  WORKHOME             2.309679
# >  EXERPRAC   WORKPAY             1.943255
# >  EXERPRAC  STUDYHMW             1.766477
# >  STUDYHMW   WORKPAY             1.611563
# >  STUDYHMW  WORKHOME             1.476861
# >   WORKPAY  WORKHOME             1.258267
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_test$split, 100 * shap_int_test$interaction_share))
# > Interaction share (test): 20.83%
print(as.data.frame(shap_int_test$interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         2.006470
# > WORKHOME                         1.681603
# > STUDYHMW                         1.618300
# >  WORKPAY                         1.604362

# Heatmaps
plot_interaction_heatmap(shap_int_train)
plot_interaction_heatmap(shap_int_valid)
plot_interaction_heatmap(shap_int_test)

# Per-feature interaction strength
plot_interaction_strength(shap_int_train)
# > feature mean_abs_interaction_with_others
# > EXERPRAC                         2.010282
# > WORKHOME                         1.653957
# > STUDYHMW                         1.615474
# >  WORKPAY                         1.583998
plot_interaction_strength(shap_int_valid)
# > feature mean_abs_interaction_with_others
# > EXERPRAC                         2.055180
# > WORKHOME                         1.695042
# >  WORKPAY                         1.650272
# > STUDYHMW                         1.622152
plot_interaction_strength(shap_int_test)
# > feature mean_abs_interaction_with_others
# > EXERPRAC                         2.006470
# > WORKHOME                         1.681603
# > STUDYHMW                         1.618300
# >  WORKPAY                         1.604362

# Pairwise dependence (continuous & quantile versions)
plot_interaction_pair(shap_int_valid, pair = c("EXERPRAC","STUDYHMW"))
plot_interaction_pair(shap_int_valid, pair = c("EXERPRAC","STUDYHMW"), quantiles = TRUE)





## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH to all plausible values in mathematics. 

### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

# Fit one XGBoost model per plausible value
tic("Fitting main models")
main_models <- lapply(pvmaths, function(pv) {
  
  X_train <- train_data[, oos]
  y_train <- train_data[[pv]]
  w_train <- train_data[[final_wt]]
  
  X_valid <- valid_data[, oos]
  y_valid <- valid_data[[pv]]
  weight_valid <- valid_data[[final_wt]]
  
  dtrain <- xgb.DMatrix(
    data = as.matrix(X_train),
    label = y_train,
    weight = w_train
  )
  
  dvalid <- xgb.DMatrix(
    data = as.matrix(X_valid),
    label = y_valid,
    weight = weight_valid
  )
  
  mod <- xgb.train(
    params = list(
      objective = "reg:squarederror",                  
      max_depth = best_params$max_depth,               # best_params$max_depth = 4
      eta = best_params$eta,                           # best_params$eta = 0.05
      eval_metric = "rmse",                            
      nthread = max(1, parallel::detectCores() - 1) 
    ),
    data = dtrain,
    nrounds = best_iter,                               # best_iter = 180
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
# > Fitting main models: 9.132 sec elapsed

main_models[[1]]             # Inspect first model: mod object, r2, and importance
main_models[[1]]$formula
main_models[[1]]$importance
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.4113538 0.1979506 0.1903704
# > 2: STUDYHMW 0.2303721 0.2482283 0.2833333
# > 3: EXERPRAC 0.2191372 0.3101510 0.2755556
# > 4: WORKHOME 0.1391370 0.2436702 0.2507407
main_models[[1]]$importance$Feature
main_models[[2]]$importance$Feature

main_models[[1]]$mod$evaluation_log
which.min(main_models[[1]]$mod$evaluation_log$eval_rmse)
# > [1] 180
min(main_models[[1]]$mod$evaluation_log$eval_rmse)
# > 85.92491
which.min(main_models[[2]]$mod$evaluation_log$eval_rmse)
# > [1] 138
min(main_models[[2]]$mod$evaluation_log$eval_rmse)
# > 86.20352
which.min(main_models[[3]]$mod$evaluation_log$eval_rmse)
# > [1] 125
min(main_models[[3]]$mod$evaluation_log$eval_rmse)
# > 88.12071
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
# >       EXERPRAC  STUDYHMW   WORKPAY  WORKHOME
# >  [1,] 0.2191372 0.2303721 0.4113538 0.1391370
# >  [2,] 0.2434393 0.2155113 0.4029170 0.1381324
# >  [3,] 0.2245437 0.2312644 0.3950371 0.1491548
# >  [4,] 0.2349889 0.2103203 0.4049976 0.1496932
# >  [5,] 0.2338689 0.2185989 0.4069152 0.1406170
# >  [6,] 0.2208963 0.2356716 0.3984187 0.1450134
# >  [7,] 0.2382402 0.2218124 0.4040186 0.1359288
# >  [8,] 0.2203249 0.2210595 0.4241437 0.1344719
# >  [9,] 0.2280041 0.2122333 0.4201257 0.1396369
# > [10,] 0.2221361 0.2278423 0.4201320 0.1298896

# Mean of gain importance across PVs
main_importance <- colMeans(main_importance_matrix)
main_importance
# >  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME 
# > 0.2285580 0.2224686 0.4088059 0.1401675

# Display ranked importance
tibble(
  Variable = names(main_importance),
  Importance = main_importance
) |> arrange(desc(Importance))
# > # A tibble: 4 × 2
# > Variable Importance
# > <chr>         <dbl>
# > 1 WORKPAY       0.409
# > 2 STUDYHMW      0.229
# > 3 EXERPRAC      0.222
# > 4 WORKHOME      0.140

# --- Estimates of Manual weighted R² (XGBoost models on training data) ---

main_r2s_weighted <- sapply(1:M, function(i) {
  pv <- pvmaths[i]
  model <- main_models[[i]]$mod
  
  # Extract weighted true values and predictions on training data
  y_train <- train_data[[pv]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, oos]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  
  y_pred <- predict(model, dtrain)  # prediction from fitted xgb model
  
  # Weighted mean and sums
  y_bar <- sum(w * y_train) / sum(w)
  sse <- sum(w * (y_train - y_pred)^2)
  sst <- sum(w * (y_train - y_bar)^2)
  
  # Weighted R²
  r2 <- 1 - sse / sst
  return(r2)
})

main_r2s_weighted
# > [1] 0.1385612 0.1389852 0.1404915 0.1459622 0.1401975 0.1419122 0.1413771 0.1395262 0.1389365 0.1453945

# Final Rubin's Step 2: mean of R² across plausible values
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.1411344

# --- Use helper function for all five metrics ---
main_metrics <- sapply(1:M, function(i) {
  pv <- pvmaths[i]
  model <- main_models[[i]]$mod
  
  y_train <- train_data[[pv]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, oos]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  y_pred <- predict(model, dtrain)
  
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
main_metrics
# >         mse     rmse      mae        bias bias_pct        r2
# > 1  7548.346 86.88122 69.60903 -0.04917408 3.323704 0.1385612
# > 2  7453.035 86.33096 69.15426 -0.04917094 3.268289 0.1389852
# > 3  7653.501 87.48429 69.97697 -0.04911159 3.401368 0.1404915
# > 4  7514.426 86.68579 69.34255 -0.04912727 3.304153 0.1459622
# > 5  7592.757 87.13643 69.70087 -0.04914092 3.339679 0.1401975
# > 6  7562.852 86.96466 69.50477 -0.04894043 3.361325 0.1419122
# > 7  7465.435 86.40275 69.21384 -0.04911576 3.309820 0.1413771
# > 8  7566.662 86.98656 69.68548 -0.04907257 3.346052 0.1395262
# > 9  7634.608 87.37625 69.85156 -0.04907771 3.389926 0.1389365
# > 10 7373.629 85.86984 69.02493 -0.04914780 3.239757 0.1453945

### ---- Replicate models using BRR replicate weights ----

set.seed(123)

tic("Fitting replicate models")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    X_train <- train_data[, oos]
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    X_valid <- valid_data[, oos]
    y_valid <- valid_data[[pv]]
    weight_valid <- valid_data[[w]]
    
    dtrain <- xgb.DMatrix(
      data = as.matrix(X_train),
      label = y_train,
      weight = w_train
    )
    
    dvalid <- xgb.DMatrix(
      data = as.matrix(X_valid),
      label = y_valid,
      weight = weight_valid
    )
    
    mod <- xgb.train(
      params = list(
        objective = "reg:squarederror",
        max_depth = best_params$max_depth,             # best_params$max_depth = 4
        eta = best_params$eta,                         # best_params$eta = 0.05
        eval_metric = "rmse",
        nthread = max(1, parallel::detectCores() - 1)
      ),
      data = dtrain,
      nrounds = best_iter,                             # best_iter = 180    
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
# > Fitting replicate models: 424.142 sec elapsed

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
# > 0.2101773 0.2396407 0.3918714 0.1583106

# --- Weighted R² across (M × G) ---
rep_r2_weighted  <- matrix(NA, nrow = G, ncol = M)

for (m in 1:M) {
  pv <- pvmaths[m]
  y_train <- train_data[[pv]]
  
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w <- train_data[[rep_wts[g]]]  # Replicate weight g
    
    X_train <- train_data[, oos]
    dtrain <- xgb.DMatrix(data = as.matrix(X_train))
    
    y_pred <- predict(model, dtrain)
    
    y_bar <- sum(w * y_train) / sum(w)
    sse <- sum(w * (y_train- y_pred)^2)
    sst <- sum(w * (y_train - y_bar)^2)
    
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
# > [1] 0.0003330413

imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
imputation_var_r2_weighted
# > [1] 6.896563e-06

var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
var_final_r2_weighted
# > [1] 0.0003406275

se_final_r2_weighted <- sqrt(var_final_r2_weighted)
se_final_r2_weighted
# > [1] 0.0184561

#### ---- Final Outputs ----

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
# > R-squared (Weighted) 0.1411344  0.0184561
print(as.data.frame(importance_table), row.names = FALSE)
# > Variable Importance Std. Error        CV
# >  WORKPAY  0.4088059 0.04178985 0.1022242
# > EXERPRAC  0.2285580 0.02745546 0.1201247
# > STUDYHMW  0.2224686 0.02596741 0.1167239
# > WORKHOME  0.1401675 0.03161805 0.2255733

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
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, oos]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  y_pred <- predict(model, dtrain)
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()

train_metrics_main
# >         mse     rmse      mae        bias bias_pct        r2
# > 1  7548.346 86.88122 69.60903 -0.04917408 3.323704 0.1385612
# > 2  7453.035 86.33096 69.15426 -0.04917094 3.268289 0.1389852
# > 3  7653.501 87.48429 69.97697 -0.04911159 3.401368 0.1404915
# > 4  7514.426 86.68579 69.34255 -0.04912727 3.304153 0.1459622
# > 5  7592.757 87.13643 69.70087 -0.04914092 3.339679 0.1401975
# > 6  7562.852 86.96466 69.50477 -0.04894043 3.361325 0.1419122
# > 7  7465.435 86.40275 69.21384 -0.04911576 3.309820 0.1413771
# > 8  7566.662 86.98656 69.68548 -0.04907257 3.346052 0.1395262
# > 9  7634.608 87.37625 69.85156 -0.04907771 3.389926 0.1389365
# > 10 7373.629 85.86984 69.02493 -0.04914780 3.239757 0.1453945
train_metrics_main$r2   # = main_r2s_weighted
# > [1] 0.1385612 0.1389852 0.1404915 0.1459622 0.1401975 0.1419122 0.1413771 0.1395262 0.1389365 0.1453945

train_metric_main <- colMeans(train_metrics_main)
train_metric_main 
# >           mse          rmse           mae          bias      bias_pct            r2 
# > 7536.52502073   86.81187336   69.50642743   -0.04910791    3.32840738    0.14113440  
train_metric_main["r2"]  # = main_r2_weighted
# >        r2 
# > 0.1411344 

# --- Replicate predictions for training data ---
tic("Computing train_metrics_replicates")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- train_data[[pvmaths[m]]]
    w <- train_data[[rep_wts[g]]]
    X_train <- train_data[, oos]
    dtrain <- xgb.DMatrix(data = as.matrix(X_train))
    y_pred <- predict(model, dtrain)
    compute_metrics(y_train, y_pred, w)
  }) |> t()
}) # a list of M=10 matrices, each of shape 80x5
toc()
# > Computing train_metrics_replicates: 14.348 sec elapsed
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
# > mse      3.922124e+04 3.684780e+04 3.552012e+04 2.833462e+04 4.065179e+04 4.104985e+04 3.316905e+04 4.105551e+04 4.172701e+04 3.425965e+04
# > rmse     1.311570e+00 1.247274e+00 1.171416e+00 9.500013e-01 1.351550e+00 1.369779e+00 1.121161e+00 1.371197e+00 1.379821e+00 1.172632e+00
# > mae      9.212247e-01 1.015724e+00 9.053512e-01 7.295544e-01 1.021169e+00 1.255223e+00 8.072398e-01 1.072509e+00 9.983063e-01 9.363370e-01
# > bias     4.115837e-08 4.375059e-08 3.763726e-08 3.242932e-08 3.849001e-08 6.474332e-08 3.343725e-08 3.318679e-08 2.948530e-08 3.492402e-08
# > bias_pct 5.951694e-03 5.590731e-03 7.332435e-03 4.594773e-03 6.314084e-03 7.186497e-03 7.900774e-03 6.748308e-03 7.666507e-03 5.341635e-03
# > r2       3.576020e-04 3.201196e-04 3.202319e-04 2.725866e-04 3.584683e-04 3.499137e-04 3.164074e-04 3.714499e-04 3.455682e-04 3.180655e-04

# <=> Equivalent codes
# sampling_var_matrix_train <- sapply(1:M, function(m) {
#   sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
#     colMeans() / (1 - k)^2
# })
# sampling_var_matrix_train

# Debugged and check consistency
sampling_var_matrix_train["r2", ]
# > [1] 0.0003576020 0.0003201196 0.0003202319 0.0002725866 0.0003584683 0.0003499137 0.0003164074 0.0003714499 0.0003455682 0.0003180655
sampling_var_r2s_weighted
# > [1] 0.0003576020 0.0003201196 0.0003202319 0.0002725866 0.0003584683 0.0003499137 0.0003164074 0.0003714499 0.0003455682 0.0003180655
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
sampling_var_r2_weighted
# > [1] 0.000333041

sampling_var_train <- rowMeans(sampling_var_matrix_train)
sampling_var_train                                           # sampling_var_train['r2'] = sampling_var_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 3.718366e+04 1.244640e+00 9.662638e-01 3.892422e-08 6.462744e-03 3.330413e-04 

imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
imputation_var_train                                              # imputation_var_train ['r2'] = imputation_var_r2_weighted 
# >         mse         rmse          mae         bias     bias_pct           r2 
# > 7.472735e+03 2.485166e-01 9.880575e-02 4.632440e-09 2.562065e-03 6.896563e-06

var_final_train <- sampling_var_train + (1 + 1/M) * imputation_var_train
var_final_train                                              # var_final_train ['r2'] = var_final_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 4.540367e+04 1.518008e+00 1.074950e+00 4.401991e-08 9.281015e-03 3.406275e-04 

se_final_train <- sqrt(var_final_train)
se_final_train                                         # se_final_train ['r2'] = se_final_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 2.130814e+02 1.232075e+00 1.036798e+00 2.098092e-04 9.633802e-02 1.845610e-02 

# Confidence intervals
ci_lower_train <- train_metric_main - z_crit * se_final_train
ci_upper_train <- train_metric_main + z_crit * se_final_train
ci_length_train <- ci_upper_train - ci_lower_train

train_eval <- tibble(
  Metric = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train, scientific = FALSE),
  ci_lower = format(ci_lower_train, scientific = FALSE),
  ci_upper = format(ci_upper_train, scientific = FALSE),
  ci_length = format(ci_length_train, scientific = FALSE)
)

print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      ci_lower      ci_upper     ci_length
# >       MSE  7536.52502073 213.0813755898 7118.89319880 7954.15684266 835.263643864
# >      RMSE    86.81187336   1.2320748370   84.39705105   89.22669566   4.829644614
# >       MAE    69.50642743   1.0367980240   67.47434065   71.53851422   4.064173572
# >      Bias    -0.04910791   0.0002098092   -0.04951913   -0.04869669   0.000822437
# >     Bias%     3.32840738   0.0963380246    3.13958832    3.51722644   0.377638117
# > R-squared     0.14113440   0.0184560974    0.10496111    0.17730769   0.072346572

### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- valid_data[, oos]
  dvalid <- xgb.DMatrix(data = as.matrix(X_valid))
  y_pred <- predict(model, dvalid)
  compute_metrics(y_train, y_pred, w)
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
    y_train <- valid_data[[pvmaths[m]]]
    w <- valid_data[[rep_wts[g]]]
    X_valid <- valid_data[, oos]
    dvalid <- xgb.DMatrix(data = as.matrix(X_valid))
    y_pred <- predict(model, dvalid)
    compute_metrics(y_train, y_pred, w)
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
  ci_length = format(ci_length_valid, scientific = FALSE)
)

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower     CI_upper     ci_length
# >       MSE   7389.7153679   345.70953636 6712.13712754 8067.2936083 1355.15648074
# >      RMSE     85.9582601     2.00291124   82.03262624   89.8838940    7.85126777
# >       MAE     68.7567142     1.80725589   65.21455771   72.2988706    7.08431290
# >      Bias     -2.3537830     2.92275080   -8.08226928    3.3747033   11.45697261
# >     Bias%      2.6985846     0.64323829    1.43786074    3.9593085    2.52144775
# > R-squared      0.0963971     0.01867842    0.05978808    0.1330061    0.07321805

### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- test_data[, oos]
  dtest <- xgb.DMatrix(data = as.matrix(X_test))
  y_pred <- predict(model, dtest)
  compute_metrics(y_train, y_pred, w)
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
    y_train <- test_data[[pvmaths[m]]]
    w <- test_data[[rep_wts[g]]]
    X_test <- test_data[, oos]
    dtest <- xgb.DMatrix(data = as.matrix(X_test))
    y_pred <- predict(model, dtest)
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates: 3.299 sec elapsed

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
# >    Metric Point_estimate Standard_error      CI_lower     CI_upper     CI_length
# >       MSE   7841.1992624   281.91671298 7288.65265831 8393.7458665 1105.09320817
# >      RMSE     88.5479117     1.58874669   85.43402538   91.6617980    6.22777259
# >       MAE     71.2435290     1.39884133   68.50185035   73.9852076    5.48335727
# >      Bias     -1.9014746     2.47150454   -6.74553447    2.9425853    9.68811977
# >     Bias%      2.9594148     0.53018719    1.92026696    3.9985626    2.07829560
# > R-squared      0.1015129     0.01907534    0.06412591    0.1388999    0.07477394

### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process to avoid redundancy.

# Evaluation function 
evaluate_split <- function(split_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, oos, pvmaths) {
  
  # Main plausible values loop
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    y_train <- split_data[[pvmaths[i]]]
    w <- split_data[[final_wt]]
    features <- split_data[, oos]
    dmat <- xgb.DMatrix(data = as.matrix(features))
    y_pred <- predict(model, dmat)
    compute_metrics(y_train, y_pred, w)
  }) |> t() |> as.data.frame()
  
  main_point <- colMeans(main_metrics_df)
  
  # Replicate loop
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      y_train <- split_data[[pvmaths[m]]]
      w <- split_data[[rep_wts[g]]]
      features <- split_data[, oos]
      dmat <- xgb.DMatrix(data = as.matrix(features))
      y_pred <- predict(model, dmat)
      compute_metrics(y_train, y_pred, w)
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
# >    Metric Point_estimate Standard_error      CI_lower      CI_upper     CI_length
# >       MSE  7536.52502073 213.0813755898 7118.89319880 7954.15684266 835.263643864
# >      RMSE    86.81187336   1.2320748370   84.39705105   89.22669566   4.829644614
# >       MAE    69.50642743   1.0367980240   67.47434065   71.53851422   4.064173572
# >      Bias    -0.04910791   0.0002098092   -0.04951913   -0.04869669   0.000822437
# >     Bias%     3.32840738   0.0963380246    3.13958832    3.51722644   0.377638117
# > R-squared     0.14113440   0.0184560974    0.10496111    0.17730769   0.072346572

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower     CI_upper     CI_length
# >       MSE   7389.7153679   345.70953636 6712.13712754 8067.2936083 1355.15648074
# >      RMSE     85.9582601     2.00291124   82.03262624   89.8838940    7.85126777
# >       MAE     68.7567142     1.80725589   65.21455771   72.2988706    7.08431290
# >      Bias     -2.3537830     2.92275080   -8.08226928    3.3747033   11.45697261
# >     Bias%      2.6985846     0.64323829    1.43786074    3.9593085    2.52144775
# > R-squared      0.0963971     0.01867842    0.05978808    0.1330061    0.07321805

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower     CI_upper     CI_length
# >       MSE   7841.1992624   281.91671298 7288.65265831 8393.7458665 1105.09320817
# >      RMSE     88.5479117     1.58874669   85.43402538   91.6617980    6.22777259
# >       MAE     71.2435290     1.39884133   68.50185035   73.9852076    5.48335727
# >      Bias     -1.9014746     2.47150454   -6.74553447    2.9425853    9.68811977
# >     Bias%      2.9594148     0.53018719    1.92026696    3.9985626    2.07829560
# > R-squared      0.1015129     0.01907534    0.06412591    0.1388999    0.07477394
