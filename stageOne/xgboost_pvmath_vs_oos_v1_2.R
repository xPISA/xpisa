# ---- I. Predictive Modelling: Version 1.2 ----

# xgb.cv with fixed hyperparameters

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


## ---- Main model (final weight) using xgb.cv to choose best_iter ----

### ---- Cross-validation (CV) for PV1MATH only: xgb.cv ----

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

# Cross-validate with xgb.cv on fixed hyperparameters (most of which are default values)
set.seed(123)
cv_mod <- xgb.cv(
  params = list(
    objective = "reg:squarederror",
    max_depth = 6,                      # default
    eta = 0.3,                          # default
    eval_metric = "rmse",
    nthread = max(1, parallel::detectCores() - 1)
  ),
  data = dtrain,
  nrounds = 100,                        # same cap as using xgb.train
  nfold = 5,                            # K-fold CV 
  prediction = FALSE,                   # <- 
  showsd = TRUE,                        # report per-iter mean±sd across folds
  verbose = TRUE,
  stratified = FALSE,
  early_stopping_rounds = NULL          # manual selection below
)


#### ---- Explore CV output ----

# Inspect cv_mod
cv_mod
str(cv_mod)
names(cv_mod$evaluation_log)
print(head(as.data.frame(cv_mod$evaluation_log), 10))

# Select the iteration with minimal mean test RMSE across folds
best_iter <- which.min(cv_mod$evaluation_log$test_rmse_mean)
best_rmse <- cv_mod$evaluation_log$test_rmse_mean[best_iter]
message(sprintf("CV-selected best_iter = %d | test_rmse_mean = %.5f (± %.5f)",
                best_iter,
                best_rmse,
                cv_mod$evaluation_log$test_rmse_std[best_iter]))
# > CV-selected best_iter = 13 | test_rmse_mean = 90.62659 (± 1.43272)

# Visualize CV RMSE curves (means across folds)
cv_mod$evaluation_log %>%
  as.data.frame() %>%
  tidyr::pivot_longer(
    cols = c(train_rmse_mean, test_rmse_mean),
    names_to = "Dataset", values_to = "RMSE"
  ) %>%
  ggplot2::ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_vline(xintercept = best_iter, linetype = 2, alpha = 0.4) +
  ggplot2::annotate("text",
                    x = best_iter,
                    y = max(cv_mod$evaluation_log$test_rmse_mean, na.rm = TRUE) * 1.02,
                    label = paste("Best iter =", best_iter), size = 3, vjust = 0
  ) +
  ggplot2::labs(
    title = "xgb.cv RMSE over Boosting Rounds (means across folds)",
    x = "Boosting Round", y = "RMSE", color = "Dataset"
  ) +
  ggplot2::theme_minimal()

### ---- Refit final booster on full TRAIN using best_iter ----
set.seed(123)
main_model <- xgb.train(
  params = list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.3,
    eval_metric = "rmse",
    nthread = max(1, parallel::detectCores() - 1)
  ),
  data = dtrain,
  nrounds = best_iter,
  watchlist = list(train = dtrain, valid = dvalid),
  verbose = 1,
  early_stopping_rounds = NULL
)

#### ---- Explore refit, importance, and quick tree ----
main_model  # str(main_model)
print(as.data.frame(main_model$evaluation_log)) 

xgb.importance(model = main_model)
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  WORKPAY 0.3679126 0.2434563 0.1895911
# > 2: STUDYHMW 0.2382677 0.2620173 0.2800496
# > 3: EXERPRAC 0.2143397 0.2619964 0.2478315
# > 4: WORKHOME 0.1794800 0.2325300 0.2825279

main_model$evaluation_log |>
  pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
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

xgb.plot.tree(model = main_model, trees = 0)            # or trees = 1, 2, etc. 
xgb.plot.tree(model = main_model, trees = best_iter-1)  # trees with lowest VALID rmse

### ---- Predict and evaluate performance on training/validation/test datasets ----
pred_train <- predict(main_model, dtrain)
pred_valid <- predict(main_model, dvalid)
pred_test  <- predict(main_model, dtest)

# Optional equivalence checks
stopifnot(all.equal(
  predict(main_model, dtrain, iterationrange = c(1, best_iter + 1)),
  pred_train
))
stopifnot(all.equal(
  predict(main_model, dvalid, iterationrange = c(1, best_iter + 1)),
  pred_valid
))
stopifnot(all.equal(
  predict(main_model, dtest,  iterationrange = c(1, best_iter + 1)),
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
# >   Training 85.14164 68.14055 -4.869022 2.232743 0.17271213
# > Validation 86.89550 69.58232 -8.804809 1.370829 0.07644301
# >       Test 90.05394 71.79991 -6.757713 2.059296 0.09423298
