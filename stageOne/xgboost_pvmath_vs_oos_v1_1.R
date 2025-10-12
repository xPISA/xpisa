# ---- I. Predictive Modelling: Version 1.1 ----

# xgb.train with fixed hyperparameters

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

### ---- Fit XGBoost model for PV1MATH only: xgb.train ----

# Define target plausible value
pv1math <- pvmaths[1]

# Prepare training and validation data
X_train <- train_data[, oos]
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

X_valid <- valid_data[, oos]
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

# Create DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train, weight = w_train)  # ?xgb.DMatrix; Use nthread = max(1, parallel::detectCores() - 1) to speed up if needed
dvalid <- xgb.DMatrix(data = as.matrix(X_valid), label = y_valid, weight = w_valid)

# Train model on fixed hyperparameters (most of which are default values)
set.seed(123)
main_model <- xgb.train(
  params = list(
    objective = "reg:squarederror",                  # Default: metric will be assigned according to objective(rmse for regression) 
    max_depth = 6,                                   # Default: 6
    eta = 0.3,                                       # Default: 0.3
    eval_metric = "rmse",                            # Default: rmse for regression 
    nthread = max(1, parallel::detectCores() - 1)    # Manually specify number of thread
    #seed = 123                                      # Random number seed for reproducibility (e.g. tune subsample, colsample_bytree for regularization)
  ),
  data = dtrain,
  nrounds = 100,                                     # Max number of boosting iterations.
  watchlist = list(train = dtrain, valid = dvalid),  # Track the performance of each round's model
  verbose = 1,                                       # If 0, xgboost will stay silent. If 1 (default), it will print information about performance. If 2, some additional information will be printed out.
  early_stopping_rounds=NULL                         # Default: NULL
)

#### ---- Explore Fit ---- 
main_model  # str(main_model)
names(main_model$evaluation_log)
print(as.data.frame(main_model$evaluation_log)) 

best_iter <- which.min(main_model$evaluation_log$valid_rmse)  # 15
best_rmse <- min(main_model$evaluation_log$valid_rmse)        # 86.82276

main_model$evaluation_log |>
  pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = best_iter, linetype = 2, alpha = 0.4) +
  annotate("text", x = best_iter, y = max(main_model$evaluation_log$valid_rmse) + 5,
           label = paste("Best iter =", best_iter), size = 3, vjust = 0) + 
  labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal()

xgb.importance(model = main_model)
# > Feature      Gain     Cover Frequency
# > <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.3068270 0.2394453 0.2222222
# > 2: STUDYHMW 0.2471694 0.2611407 0.2607579
# > 3: EXERPRAC 0.2324982 0.2502567 0.2512845
# > 4: WORKHOME 0.2135054 0.2491572 0.2657354

xgb.plot.importance(importance_matrix = xgb.importance(model = main_model),
                    top_n = NULL,                        # Top n features (you only have 4 in oos)
                    measure = "Gain",                    # Can also use "Cover" or "Frequency"
                    rel_to_first = TRUE,
                    xlab = "Relative Importance")
xgb.plot.tree(model = main_model, trees = 0)            # or trees = 1, 2, etc. 
xgb.plot.tree(model = main_model, trees = best_iter-1)  # trees with lowest VALID rmse


### ---- Predict and evaluate performance on training/validation/test datasets ----

# Prepare test data 
X_test <- test_data[, oos]
y_test <- test_data[[pv1math]]
w_test <- test_data[[final_wt]]
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test, weight = w_test)

# Predict using refitted model
pred_train <- predict(main_model, dtrain, iterationrange = c(1, best_iter + 1))   # ?predict.xgb.Booster; Can try predinteraction = TRUE or predcontrib = TRUE (SHAP)
pred_valid <- predict(main_model, dvalid, iterationrange = c(1, best_iter + 1))
pred_test  <- predict(main_model, dtest, iterationrange = c(1, best_iter + 1))

# Sanity check
stopifnot(all.equal(predict(main_model, dtrain, iterationrange = c(1, best_iter + 1)), 
                    predict(main_model, dtrain, ntreelimit = best_iter)))
stopifnot(all.equal(predict(main_model, dvalid, iterationrange = c(1, best_iter + 1)), 
                    predict(main_model, dvalid, ntreelimit = best_iter)))
stopifnot(all.equal(predict(main_model, dtest, iterationrange = c(1, best_iter + 1)), 
                    predict(main_model, dtest, ntreelimit = best_iter)))

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
# >    Dataset     RMSE      MAE      Bias    Bias%         R2
# >   Training 84.62298 67.69093 -2.386232 2.718319 0.18276069
# > Validation 86.82276 69.36116 -6.337820 1.870945 0.07798852
# >       Test 89.87706 71.56219 -4.195269 2.576558 0.09778761


#### ---- Explore SHAP ----

# Remark: Do not feed these into evaluation metrics

##### ---- Per-feature SHAP ----

# Interpretation: For any row, prediction ≈ BIAS + sum(feature SHAPs). Units are the same as target (PISA points).

# Standard plain predictions (needed for SHAP sanity checks)
#pred_train <- predict(main_model, dtrain)
#pred_valid <- predict(main_model, dvalid)
#pred_test  <- predict(main_model, dtest)

# Helper: get SHAP(predcontrib) + sanity checks
get_shap <- function(model, dmat, preds, features) {
  pred_shap <- predict(model, dmat, predcontrib = TRUE)  # n x (p+1); last col = "BIAS"
  # shape checks
  stopifnot(ncol(pred_shap) == length(features) + 1L)
  stopifnot(colnames(pred_shap)[ncol(pred_shap)] == "BIAS")
  # SHAP rows must sum to the prediction (squared-error objective ⇒ raw scores)
  stopifnot(isTRUE(all.equal(rowSums(pred_shap), preds, tolerance = 1e-6)))
  pred_shap
}

shap_train <- get_shap(main_model, dtrain, pred_train, oos)
shap_valid <- get_shap(main_model, dvalid, pred_valid, oos)
shap_test  <- get_shap(main_model, dtest,  pred_test,  oos)

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
# >  WORKPAY     15.660982
# > EXERPRAC     12.737663
# > STUDYHMW     12.043624
# > WORKHOME      9.914024
print(as.data.frame(global_importance_valid), row.names=FALSE)
# >  feature mean_abs_shap
# >  WORKPAY     16.020378
# > EXERPRAC     12.829390
# > STUDYHMW     11.785662
# > WORKHOME      9.966754
print(as.data.frame(global_importance_test), row.names=FALSE)
# >  feature mean_abs_shap
# >  WORKPAY     16.252410
# > STUDYHMW     12.250275
# > EXERPRAC     12.242939
# > WORKHOME      9.782268

# Plots (reuse a small plotting helper)
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


##### ---- SHAP Interactions ----

###### ---- SHAP Interactions (training) ----

# Assumes these already exist: main_model, dtrain, train_data, w_train, oos, pred_train, shap_train
# Replace 'train' with valid/test analogs to explore other splits.
# Note: the interaction tensor is heavy when feature count grows. 

# 1) Interaction tensor: n × (p+1) × (p+1) (includes BIAS row/col)
interaction_tensor_train <- predict(main_model, dtrain, predinteraction = TRUE)

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
# enforce symmetry (defensive; should already be symmetric)
interaction_avg_matrix_full <- (interaction_avg_matrix_full + t(interaction_avg_matrix_full)) / 2
diag(interaction_avg_matrix_full) <- NA   # drop main effects on the diagonal

# 4) Unique-pair table (mask one triangle only for display)
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
# > 1  EXERPRAC  WORKHOME             5.379000
# > 2  STUDYHMW  WORKHOME             5.069080
# > 3  EXERPRAC  STUDYHMW             4.985623
# > 4  STUDYHMW   WORKPAY             4.027050
# > 5  EXERPRAC   WORKPAY             3.990908
# > 6   WORKPAY  WORKHOME             3.506468

# 5) Share of interactions in overall explanation (heuristic)
main_abs_mean <- colSums(abs(shap_train[, oos, drop = FALSE]) * w_train, na.rm = TRUE) / sum(w_train, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (training): %.2f%%\n", 100 * interaction_share))
# > Interaction share (training): 34.87%

# 6) Heatmap (keep all feature labels)
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = oos),
    feature_j = factor(feature_j, levels = oos)
  )

ggplot2::ggplot(interaction_heat_df,
                ggplot2::aes(feature_i, feature_j, fill = mean_abs_interaction)) +
  ggplot2::geom_tile(na.rm = TRUE) +
  ggplot2::scale_x_discrete(drop = FALSE) +
  ggplot2::scale_y_discrete(drop = FALSE) +
  ggplot2::coord_equal() +
  ggplot2::labs(x = "", y = "", fill = "Weighted mean |interaction|",
                title = "SHAP interaction heatmap (training)") +
  ggplot2::theme_minimal()

# 7) Per-feature interaction strength (use FULL matrix -> no NaN)
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
# > EXERPRAC                         4.785177
# > STUDYHMW                         4.693918
# > WORKHOME                         4.651516
# >  WORKPAY                         3.841475

ggplot2::ggplot(interaction_strength_table,
                ggplot2::aes(x = reorder(feature, mean_abs_interaction_with_others),
                             y = mean_abs_interaction_with_others)) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::labs(x = "Feature", y = "Mean |interaction| with others",
                title = "Per-feature interaction strength (training)") +
  ggplot2::theme_minimal()

# 8) Inspect one pair in detail (dependence-style plot of the interaction term)
#    Convert SPSS-labelled columns to plain numeric.
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
ggplot2::ggplot(interaction_pair_df,
                ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j)) +
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


###### ---- SHAP Interactions (validation) ----

# Assumes these already exist: main_model, dvalid, valid_data, w_valid, oos, pred_valid, shap_valid
# Replace 'valid' with train/test analogs to explore other splits.
# Note: the interaction tensor is heavy when feature count grows.

# 1) Interaction tensor
interaction_tensor_valid <- predict(main_model, dvalid, predinteraction = TRUE)

# 2) Identity check on one row
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
# > 1  EXERPRAC  WORKHOME             5.391957
# > 2  STUDYHMW  WORKHOME             5.093887
# > 3  EXERPRAC  STUDYHMW             4.982787
# > 4  EXERPRAC   WORKPAY             4.157292
# > 5  STUDYHMW   WORKPAY             3.905291
# > 6   WORKPAY  WORKHOME             3.550695

# 5) Share of interactions
main_abs_mean <- colSums(abs(shap_valid[, oos, drop = FALSE]) * w_valid, na.rm = TRUE) / sum(w_valid, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (validation): %.2f%%\n", 100 * interaction_share))
# > Interaction share (validation): 34.86%

# 6) Heatmap
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = oos),
    feature_j = factor(feature_j, levels = oos)
  )

ggplot2::ggplot(interaction_heat_df,
                ggplot2::aes(feature_i, feature_j, fill = mean_abs_interaction)) +
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
# > EXERPRAC                         4.844012
# > WORKHOME                         4.678846
# > STUDYHMW                         4.660655
# >  WORKPAY                         3.871092

ggplot2::ggplot(interaction_strength_table,
                ggplot2::aes(x = reorder(feature, mean_abs_interaction_with_others),
                             y = mean_abs_interaction_with_others)) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::labs(x = "Feature", y = "Mean |interaction| with others",
                title = "Per-feature interaction strength (validation)") +
  ggplot2::theme_minimal()

# 8) Dependence-style plot for one pair
as_plain_num <- function(x) {
  x |> haven::zap_missing() |> haven::zap_labels() |> vctrs::vec_data()
}

interaction_pair <- c("EXERPRAC", "STUDYHMW")  # change as needed
interaction_pair_df <- tibble::tibble(
  x_i = as_plain_num(valid_data[[interaction_pair[1]]]),
  x_j = as_plain_num(valid_data[[interaction_pair[2]]]),
  shap_interaction_ij = interaction_tensor_valid[, interaction_pair[1], interaction_pair[2]],
  w   = w_valid
) |> tidyr::drop_na()

ggplot2::ggplot(interaction_pair_df,
                ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j)) +
  ggplot2::geom_point(alpha = 0.35) +
  ggplot2::geom_smooth(
    mapping = ggplot2::aes(x = x_i, y = shap_interaction_ij, group = 1, weight = w),
    inherit.aes = FALSE, method = "loess", se = FALSE, color = "black"
  ) +
  ggplot2::labs(x = interaction_pair[1],
                y = paste("SHAP interaction:", paste(interaction_pair, collapse = " × ")),
                color = interaction_pair[2],
                title = "Interaction effect (validation)") +
  ggplot2::theme_minimal()

interaction_pair_df_bins <- interaction_pair_df |>
  dplyr::mutate(x_j_bin = factor(dplyr::ntile(x_j, 4), labels = paste0("Q", 1:4)))

ggplot2::ggplot(interaction_pair_df_bins,
                ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j_bin)) +
  ggplot2::geom_point(alpha = 0.25) +
  ggplot2::geom_smooth(ggplot2::aes(weight = w), method = "loess", se = FALSE) +
  ggplot2::labs(x = interaction_pair[1],
                y = paste("SHAP interaction:", paste(interaction_pair, collapse = " × ")),
                color = paste0(interaction_pair[2], " quantile"),
                title = "Interaction effect by quantile (validation)") +
  ggplot2::theme_minimal()


###### ---- SHAP Interactions (test) ----

# Assumes these already exist: main_model, dtest, test_data, w_test, oos, pred_test, shap_test
# Replace 'test' with train/valid analogs to explore other splits.
# Note: the interaction tensor is heavy when feature count grows.

# 1) Interaction tensor
interaction_tensor_test <- predict(main_model, dtest, predinteraction = TRUE)

# 2) Identity check on one row
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
# > 1  EXERPRAC  WORKHOME             5.468155
# > 2  STUDYHMW  WORKHOME             5.134742
# > 3  EXERPRAC  STUDYHMW             4.868743
# > 4  STUDYHMW   WORKPAY             3.886177
# > 5  EXERPRAC   WORKPAY             3.867489
# > 6   WORKPAY  WORKHOME             3.472474

# 5) Share of interactions
main_abs_mean <- colSums(abs(shap_test[, oos, drop = FALSE]) * w_test, na.rm = TRUE) / sum(w_test, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (test): %.2f%%\n", 100 * interaction_share))
# > Interaction share (test): 34.57%

# 6) Heatmap
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = oos),
    feature_j = factor(feature_j, levels = oos)
  )

ggplot2::ggplot(interaction_heat_df,
                ggplot2::aes(feature_i, feature_j, fill = mean_abs_interaction)) +
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
# > EXERPRAC                         4.734795
# > WORKHOME                         4.691790
# > STUDYHMW                         4.629887
# >  WORKPAY                         3.742046

ggplot2::ggplot(interaction_strength_table,
                ggplot2::aes(x = reorder(feature, mean_abs_interaction_with_others),
                             y = mean_abs_interaction_with_others)) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::labs(x = "Feature", y = "Mean |interaction| with others",
                title = "Per-feature interaction strength (test)") +
  ggplot2::theme_minimal()

# 8) Dependence-style plot for one pair
as_plain_num <- function(x) {
  x |> haven::zap_missing() |> haven::zap_labels() |> vctrs::vec_data()
}

interaction_pair <- c("EXERPRAC", "STUDYHMW")  # change as needed
interaction_pair_df <- tibble::tibble(
  x_i = as_plain_num(test_data[[interaction_pair[1]]]),
  x_j = as_plain_num(test_data[[interaction_pair[2]]]),
  shap_interaction_ij = interaction_tensor_test[, interaction_pair[1], interaction_pair[2]],
  w   = w_test
) |> tidyr::drop_na()

ggplot2::ggplot(interaction_pair_df,
                ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j)) +
  ggplot2::geom_point(alpha = 0.35) +
  ggplot2::geom_smooth(
    mapping = ggplot2::aes(x = x_i, y = shap_interaction_ij, group = 1, weight = w),
    inherit.aes = FALSE, method = "loess", se = FALSE, color = "black"
  ) +
  ggplot2::labs(x = interaction_pair[1],
                y = paste("SHAP interaction:", paste(interaction_pair, collapse = " × ")),
                color = interaction_pair[2],
                title = "Interaction effect (test)") +
  ggplot2::theme_minimal()

interaction_pair_df_bins <- interaction_pair_df |>
  dplyr::mutate(x_j_bin = factor(dplyr::ntile(x_j, 4), labels = paste0("Q", 1:4)))

ggplot2::ggplot(interaction_pair_df_bins,
                ggplot2::aes(x = x_i, y = shap_interaction_ij, color = x_j_bin)) +
  ggplot2::geom_point(alpha = 0.25) +
  ggplot2::geom_smooth(ggplot2::aes(weight = w), method = "loess", se = FALSE) +
  ggplot2::labs(x = interaction_pair[1],
                y = paste("SHAP interaction:", paste(interaction_pair, collapse = " × ")),
                color = paste0(interaction_pair[2], " quantile"),
                title = "Interaction effect by quantile (test)") +
  ggplot2::theme_minimal()


###### ---- ** SHAP Interactions (training/validation/test) ** ----

# Remark: Consolidated SHAP-interaction helpers for train/valid/test.

# --- Helpers ---

# Convert SPSS-labelled vectors to plain numeric
as_plain_num <- function(x) {
  x |> haven::zap_missing() |> haven::zap_labels() |> vctrs::vec_data()
}

# Core computation: interaction tensor, averages, tables, share, sanity checks
compute_shap_interactions <- function(model, dmat, data_frame, w_vec, features,
                                      preds = NULL, shap_mat = NULL, split = "split") {
  stopifnot(all(features %in% colnames(data_frame)))
  
  # Cache preds / SHAP if not provided
  if (is.null(preds))    preds    <- predict(model, dmat)
  if (is.null(shap_mat)) shap_mat <- predict(model, dmat, predcontrib = TRUE)
  
  # SHAP sanity checks
  stopifnot(ncol(shap_mat) == length(features) + 1L)
  stopifnot(colnames(shap_mat)[ncol(shap_mat)] == "BIAS")
  stopifnot(isTRUE(all.equal(rowSums(shap_mat), preds, tolerance = 1e-6)))
  
  # Interactions: n × (p+1) × (p+1)
  interaction_tensor <- predict(model, dmat, predinteraction = TRUE)
  
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
  
  # Enforce symmetry defensively (should already be symmetric)
  interaction_avg_matrix_full <- (interaction_avg_matrix_full +
                                    t(interaction_avg_matrix_full)) / 2
  
  # Drop diagonal (main effects) for pair-only summaries
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
  
  # Per-feature interaction strength (use FULL matrix; no NA from masking)
  interaction_strength_table <- tibble::tibble(
    feature = features,
    mean_abs_interaction_with_others = sapply(features, function(fi) {
      partners <- setdiff(features, fi)
      mean(interaction_avg_matrix_full[fi, partners], na.rm = TRUE)
    })
  ) |>
    dplyr::arrange(dplyr::desc(mean_abs_interaction_with_others))
  
  # Heuristic "share" of interactions vs main effects (for context)
  main_abs_mean <- colSums(abs(shap_mat[, features, drop = FALSE]) * w_vec, na.rm = TRUE) /
    sum(w_vec, na.rm = TRUE)
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
  
  ggplot2::ggplot(heat_df,
                  ggplot2::aes(feature_i, feature_j, fill = mean_abs_interaction)) +
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
                  ggplot2::aes(x = reorder(feature, mean_abs_interaction_with_others),
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
    # Continuous colour; single weighted smooth (avoid color-drop warning)
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
    # Separate smooths by quantile bins (no scales::cut_number; use dplyr::ntile)
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

# --- Run for all three splits ---

# Training
shap_int_train <- compute_shap_interactions(
  model    = main_model,
  dmat     = dtrain,
  data_frame = train_data,
  w_vec    = w_train,
  features = oos,
  preds    = pred_train,
  shap_mat = shap_train,
  split    = "training"
)

# Validation
shap_int_valid <- compute_shap_interactions(
  model    = main_model,
  dmat     = dvalid,
  data_frame = valid_data,
  w_vec    = w_valid,
  features = oos,
  preds    = pred_valid,
  shap_mat = shap_valid,
  split    = "validation"
)

# Test
shap_int_test <- compute_shap_interactions(
  model    = main_model,
  dmat     = dtest,
  data_frame = test_data,
  w_vec    = w_test,
  features = oos,
  preds    = pred_test,
  shap_mat = shap_test,
  split    = "test"
)


# --- Quick access & examples ---

# Tables / shares
shap_int_train$interaction_table %>% as.data.frame() %>% print(row.names = FALSE)
# >   feature_i feature_j mean_abs_interaction
# >    EXERPRAC  WORKHOME             5.379000
# >    STUDYHMW  WORKHOME             5.069080
# >    EXERPRAC  STUDYHMW             4.985623
# >    STUDYHMW   WORKPAY             4.027050
# >    EXERPRAC   WORKPAY             3.990908
# >     WORKPAY  WORKHOME             3.506468
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_train$split, 100 * shap_int_train$interaction_share))
# > Interaction share (training): 34.87%
print(as.data.frame(shap_int_train$interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         4.785177
# > STUDYHMW                         4.693918
# > WORKHOME                         4.651516
# >  WORKPAY                         3.841475

shap_int_valid$interaction_table
# >   feature_i feature_j mean_abs_interaction
# > 1  EXERPRAC  WORKHOME             5.391957
# > 2  STUDYHMW  WORKHOME             5.093887
# > 3  EXERPRAC  STUDYHMW             4.982787
# > 4  EXERPRAC   WORKPAY             4.157292
# > 5  STUDYHMW   WORKPAY             3.905291
# > 6   WORKPAY  WORKHOME             3.550695
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_valid$split, 100 * shap_int_valid$interaction_share))
# > Interaction share (training): 34.86%
print(as.data.frame(shap_int_valid$interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         4.844012
# > WORKHOME                         4.678846
# > STUDYHMW                         4.660655
# >  WORKPAY                         3.871092

shap_int_test$interaction_table
# >   feature_i feature_j mean_abs_interaction
# >  1  EXERPRAC  WORKHOME             5.468155
# >  2  STUDYHMW  WORKHOME             5.134742
# >  3  EXERPRAC  STUDYHMW             4.868743
# >  4  STUDYHMW   WORKPAY             3.886177
# >  5  EXERPRAC   WORKPAY             3.867489
# >  6   WORKPAY  WORKHOME             3.472474
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_test$split, 100 * shap_int_test$interaction_share))
# > Interaction share (training): 34.57%
print(as.data.frame(shap_int_test$interaction_strength_table), row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# >  EXERPRAC                         4.734795
# >  WORKHOME                         4.691790
# >  STUDYHMW                         4.629887
# >   WORKPAY                         3.742046

# Heatmaps
plot_interaction_heatmap(shap_int_train)
plot_interaction_heatmap(shap_int_valid)
plot_interaction_heatmap(shap_int_test)

# Per-feature interaction strength (table is printed inside the plot helper)
plot_interaction_strength(shap_int_train)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         4.785177
# > STUDYHMW                         4.693918
# > WORKHOME                         4.651516
# >  WORKPAY                         3.841475
plot_interaction_strength(shap_int_valid)
# >  feature mean_abs_interaction_with_others
# > EXERPRAC                         4.844012
# > WORKHOME                         4.678846
# > STUDYHMW                         4.660655
# >  WORKPAY                         3.871092
plot_interaction_strength(shap_int_test)
# >  feature mean_abs_interaction_with_others
# >  EXERPRAC                         4.734795
# >  WORKHOME                         4.691790
# >  STUDYHMW                         4.629887
# >   WORKPAY                         3.742046

# Pairwise dependence (continuous & quantile versions)
plot_interaction_pair(shap_int_valid, pair = c("EXERPRAC","STUDYHMW"))
plot_interaction_pair(shap_int_valid, pair = c("EXERPRAC","STUDYHMW"), quantiles = TRUE)

