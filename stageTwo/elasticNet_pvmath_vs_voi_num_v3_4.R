# ---- III. Predictive Modelling: Version 3.4 ----
#
# Nested Cross-Validation (glmnet):
#   inner  - cv.glmnet with MANUAL K-folds (grouped = TRUE, keep = TRUE), select lambda.min AND lambda.1se across α-grid
#   outer  - glmnet exact refit on OUTER-TRAIN at the inner winner’s (alpha, lambda.*); evaluate on OUTER-HOLDOUT
#
# This script runs both rules side-by-side and reports per-fold and aggregated metrics
# for:   Rule ∈ { "lambda.min", "lambda.1se" }.
#
# Remark: glmnet cannot handle NAs?

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

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("tidyverse","glmnet","Matrix","haven","broom","tictoc","caret", "doParallel"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         tidyverse            glmnet            Matrix             haven             broom            tictoc             caret          doParallel  
# > "tidyverse 2.0.0"   "glmnet 4.1.10"    "Matrix 1.7.3"     "haven 2.5.4"     "broom 1.0.8"    "tictoc 1.2.1"     "caret 7.0.1" "doParallel 1.0.17"  

# Load data
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE)
dim(pisa_2022_canada_merged)   # 23073  1699

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
voi_num <- c(# --- Student Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### Subject-specific beliefs, attitudes, feelings and behaviours (Module 7)
  "MATHMOT",     # Relative motivation to do well in mathematics compared to other core subjects
  "MATHEASE",    # Perception of mathematics as easier than other core subjects
  "MATHPREF",    #  Preference of mathematics over other core subjects
  ### Out-of-school experiences (Module 10)
  "EXERPRAC",    # Exercising or practising a sport before or after school
  "STUDYHMW",    # Studying for school or homework before or after school
  "WORKPAY",     # Working for pay before or after school
  "WORKHOME",    # Working in household or taking care of family members
  ## Derived variables based on IRT scaling
  ### Economic, social and cultural status (Module 2)
  "HOMEPOS",     # Home possessions (Components of ESCS)
  "ICTRES",      # ICT resources (at home)
  ### Educational pathways and post-secondary aspirations (Module 3)
  "INFOSEEK",    # Information seeking regarding future career
  ### School culture and climate (Module 6)
  "BULLIED",     # Being bullied
  "FEELSAFE",    # Feeling safe
  "BELONG",      # Sense of belonging
  ### Subject-specific beliefs, attitudes, feelings, and behaviours (Module 7)
  "GROSAGR",     # Growth mindset
  "ANXMAT",      # Mathematics anxiety
  "MATHEFF",     # Mathematics self-efficacy: Formal and applied mathematics
  "MATHEF21",    # Mathematics self-efficacy: Mathematical reasoning and 21st century mathematics
  "MATHPERS",    # Proactive mathematics study behaviour
  "FAMCON",      # Subjective familiarity with mathematics concepts
  ### General social and emotional characteristics (Module 8)
  "ASSERAGR",    # Assertiveness
  "COOPAGR",     # Cooperation
  "CURIOAGR",    # Curiosity
  "EMOCOAGR",    # Emotional control
  "EMPATAGR",    # Empathy
  "PERSEVAGR",   # Perseverance
  "STRESAGR",    # Stress resistance
  ### Exposure to mathematics content (Module 15)
  "EXPOFA",      # Exposure to formal and applied mathematics tasks
  "EXPO21ST",    # Exposure to mathematical reasoning and 21st century mathematics tasks
  ### Mathematics teacher behaviour (Module 16)
  "COGACRCO",    # Cognitive activation in mathematics: Foster reasoning
  "COGACMCO",    # Cognitive activation in mathematics: Encourage mathematical thinking
  "DISCLIM",     # Disciplinary climate in mathematics
  ### Parental/guardian involvement and support (Module 19)
  "FAMSUP",      # Family support
  ### Creative thinking (Module 20)
  "CREATFAM",    # Creative peers and family environment 
  "CREATSCH",    # Creative school and class environment
  "CREATEFF",    # Creative thinking self-efficacy, 
  "CREATOP",     # Creativity and openness to intellect
  "IMAGINE",     # Imagination and adventurousness
  "OPENART",     # Openness to art and reflection
  "CREATAS",     # Participation in creative activities at school
  "CREATOOS",    # Participation in creative activities outside of school
  ### Global crises (Module 21)
  "FAMSUPSL",    # Family support for self-directed learning
  "FEELLAH",     # Feelings about learning at home 
  "PROBSELF",    # Problems with self-directed learning
  "SDLEFF",      # Self-directed learning self-efficacy
  "SCHSUST",     # School actions to sustain learning 
  "LEARRES",     # Types of learning resources used while school was closed
  ## Complex composite index
  "ESCS",        # Index of economic, social and cultural status (ESCS)
  # --- School Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### Out-of-school experiences (Module 10)
  "MACTIV",      # Mathematics-related extra-curricular activities at school (ordinal/numeric)
  ### Organisation of student learning at school (Module 14)
  "ABGMATH",     # Ability grouping for mathematics classes (ordinal/numeric)
  ### Teacher qualification, training, and professional development (Module 17)
  "MTTRAIN",     # Mathematics teacher training
  ### Creative thinking (Module 20)
  "CREENVSC",    # Creative school environment 
  "OPENCUL",     # Openness culture/climate 
  ### Global crises (Module 21)
  "DIGPREP")     # Preparedness for Digital Learning (WLE) (numeric)
stopifnot(!anyDuplicated(voi_num))  # Fail if any duplicates 
length(voi_num)
# > [1] 53

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)    # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                  # Final student weight

# Prepare modeling data
temp_data <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                  # IDs
         all_of(final_wt), all_of(rep_wts),   # Weights
         all_of(pvmaths), all_of(voi_num)) %>%    # PVs + predictors
  filter(if_all(all_of(voi_num), ~ !is.na(.)))    # Listwise deletion for predictors

dim(temp_data)  # 6944 x 146

# Check summaries
summary(temp_data[[final_wt]])
summary(temp_data[, pvmaths])
sapply(temp_data[, pvmaths], sd)
summary(temp_data[, voi_num])
sapply(temp_data[, voi_num], sd)

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

### ---- Random Train/Validation/Test (80/10/10) split ----
set.seed(123)                      # Ensure reproducibility
n <- nrow(temp_data)               # 6944
indices <- sample(n)               # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.80 * n)         # 5555
n_valid <- floor(0.10 * n)         # 694
n_test  <- n - n_train - n_valid   # 695
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
# >                  train    valid     test
# > n              5555.00   694.00   695.00
# > w_mean           16.02    16.06    17.22
# > w_sum         88995.65 11148.48 11970.28
# > w_min             1.05     1.05     1.05
# > w_max           132.19    88.54    76.99
# > w_sd             13.85    13.11    14.07
# > n_schools       670.00   394.00   403.00
# > pv1math_mean    520.97   520.70   529.25
# > pv1math_sd       90.32    83.95    94.10
# > exerprac_mean     4.19     3.77     4.26
# > exerprac_sd       3.26     3.03     3.27

## ---- Main model using final student weights (W_FSTUWT) ----

### ---- Nested Cross-Validation for PV1MATH (grouped=TRUE, keep=TRUE; lambda.min & lambda.1se) ----

# Target plausible value
pv1math <- pvmaths[1]   # "PV1MATH"

# TRAIN/VALID/TEST matrices (VALID/TEST kept but not used for nested-CV choices)
X_train <- as.matrix(train_data[, voi_num])
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

X_valid <- as.matrix(valid_data[, voi_num])
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

X_test  <- as.matrix(test_data[,  voi_num])
y_test  <- test_data[[pv1math]]
w_test  <- test_data[[final_wt]]

# --- Helpers: folds and diagnostics ---

# Build K manual folds (indices) with a fixed seed over N rows
make_folds <- function(n_cv, num_folds = 5L, seed = 123L) {
  set.seed(seed)
  cv_order <- sample.int(n_cv)
  bounds <- floor(seq(0, n_cv, length.out = num_folds + 1))
  stopifnot(all(diff(bounds) > 0))
  folds <- vector("list", num_folds)
  for (k in seq_len(num_folds)) folds[[k]] <- cv_order[(bounds[k] + 1):bounds[k + 1]]
  stopifnot(identical(sort(unlist(folds)), seq_len(n_cv)))
  folds
}

# Weight diagnostics per fold
print_weight_balance <- function(w, folds, title = "Weight balance") {
  s <- tibble::tibble(
    fold   = seq_along(folds),
    n      = vapply(folds, length, integer(1)),
    w_sum  = vapply(folds, function(idx) sum(w[idx]), numeric(1)),
    w_mean = vapply(folds, function(idx) mean(w[idx]), numeric(1)),
    w_med  = vapply(folds, function(idx) median(w[idx]), numeric(1)),
    w_effn = vapply(folds, function(idx) { wi <- w[idx]; (sum(wi)^2) / sum(wi^2) }, numeric(1))
  ) |>
    dplyr::mutate(w_share = w_sum / sum(w_sum))
  message(title)
  print(s, n = Inf, width = Inf)
  cat(sprintf("Weight share range: [%.3f, %.3f]\n", min(s$w_share), max(s$w_share)))
  cat(sprintf("Max/Min share ratio: %.3f\n", max(s$w_share) / min(s$w_share)))
  cat(sprintf("Coeff. of variation of shares: %.3f\n\n", stats::sd(s$w_share) / mean(s$w_share)))
}

# α-grid
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso

# --- Outer folds over TRAIN ---
num_folds_outer <- 5L
n_cv_outer <- nrow(X_train)
outer_folds <- make_folds(n_cv_outer, num_folds_outer, seed = 123L)

print_weight_balance(w_train, outer_folds, title = "Outer CV: weight balance over TRAIN")
# > Outer CV: weight balance over TRAIN
# > # A tibble: 5 × 7
# >    fold     n  w_sum w_mean w_med w_effn w_share
# >   <int> <int>  <dbl>  <dbl> <dbl>  <dbl>   <dbl>
# > 1     1  1111 17147.   15.4  9.12   630.   0.193
# > 2     2  1111 17872.   16.1 10.2    653.   0.201
# > 3     3  1111 18283.   16.5 10.2    621.   0.205
# > 4     4  1111 17925.   16.1  9.39   633.   0.201
# > 5     5  1111 17769.   16.0  9.57   647.   0.200
# > Weight share range: [0.193, 0.205]
# > Max/Min share ratio: 1.066
# > Coeff. of variation of shares: 0.023

# Storage (separate containers for each rule)
outer_oof_pred_min      <- rep(NA_real_, n_cv_outer)
outer_oof_pred_1se      <- rep(NA_real_, n_cv_outer)

outer_metrics_min       <- vector("list", num_folds_outer)
outer_metrics_1se       <- vector("list", num_folds_outer)

inner_winners_rows_min  <- vector("list", num_folds_outer)
inner_winners_rows_1se  <- vector("list", num_folds_outer)

# Parallel backend for cv.glmnet (inner loop)
doParallel::registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Nested Cross-Validation (glmnet; grouped=TRUE, keep=TRUE; lambda.min & lambda.1se)")
for (o in seq_len(num_folds_outer)) {
  message(sprintf("\n==== Outer fold %d/%d ====", o, num_folds_outer))
  outer_hold_idx  <- outer_folds[[o]]
  outer_train_idx <- setdiff(seq_len(n_cv_outer), outer_hold_idx)
  
  # OUTER-TRAIN / OUTER-HOLDOUT splits
  X_train_outer <- X_train[outer_train_idx, , drop = FALSE]
  y_train_outer <- y_train[outer_train_idx]
  w_train_outer <- w_train[outer_train_idx]
  
  X_hold_outer  <- X_train[outer_hold_idx, , drop = FALSE]
  y_hold_outer  <- y_train[outer_hold_idx]
  w_hold_outer  <- w_train[outer_hold_idx]
  
  # Inner folds (on OUTER-TRAIN)
  n_cv_inner      <- length(outer_train_idx)
  num_folds_inner <- 5L
  inner_folds     <- make_folds(n_cv_inner, num_folds_inner, seed = 123L)
  
  # foldid (1..K for each OUTER-TRAIN row)
  foldid_inner <- integer(n_cv_inner)
  for (k in seq_along(inner_folds)) foldid_inner[inner_folds[[k]]] <- k
  stopifnot(all(foldid_inner %in% seq_len(num_folds_inner)), length(foldid_inner) == n_cv_inner)
  
  print_weight_balance(w_train_outer, inner_folds,
                       title = sprintf("Inner CV (outer fold %d): weight balance over OUTER-TRAIN", o))
  
  # --- Inner tuner over α-grid using cv.glmnet (grouped=TRUE, keep=TRUE) ---
  per_alpha_list_min <- vector("list", length(alpha_grid))   # one row per α at lambda.min
  per_alpha_list_1se <- vector("list", length(alpha_grid))   # one row per α at lambda.1se
  
  tic(sprintf("Inner tuning (outer fold %d)", o))
  for (i in seq_along(alpha_grid)) {
    alpha <- alpha_grid[i]
    message(sprintf("  [Inner α %d/%d] alpha = %.1f  (manual 5-fold on OUTER-TRAIN)", i, length(alpha_grid), alpha))
    
    set.seed(123)
    cvmod <- cv.glmnet(
      x = X_train_outer,
      y = y_train_outer,
      weights = w_train_outer,
      type.measure = "mse",
      foldid = foldid_inner,      # manual inner folds
      grouped = TRUE,             # as requested
      keep = TRUE,                # keep OOF preds (cvmod$fit.preval)
      parallel = TRUE,
      trace.it = 0,
      alpha = alpha,
      family = "gaussian",
      standardize = TRUE,
      intercept = TRUE
    )
    
    # ---- lambda.min row (per α) ----
    idx_min     <- cvmod$index["min", 1]
    per_alpha_list_min[[i]] <- tibble::tibble(
      alpha       = alpha,
      alpha_idx   = i,
      s           = "lambda.min",
      lambda      = cvmod$lambda[idx_min],
      lambda_idx  = idx_min,
      nzero       = cvmod$nzero[idx_min],
      dev_ratio   = as.numeric(cvmod$glmnet.fit$dev.ratio[idx_min]),
      cvm         = as.numeric(cvmod$cvm[idx_min]),
      cvsd        = as.numeric(cvmod$cvsd[idx_min]),
      cvlo        = as.numeric(cvmod$cvlo[idx_min]),
      cvup        = as.numeric(cvmod$cvup[idx_min])
    )
    
    # ---- lambda.1se row (per α) ----
    idx_1se     <- cvmod$index["1se", 1]
    per_alpha_list_1se[[i]] <- tibble::tibble(
      alpha       = alpha,
      alpha_idx   = i,
      s           = "lambda.1se",
      lambda      = cvmod$lambda[idx_1se],
      lambda_idx  = idx_1se,
      nzero       = cvmod$nzero[idx_1se],
      dev_ratio   = as.numeric(cvmod$glmnet.fit$dev.ratio[idx_1se]),
      cvm         = as.numeric(cvmod$cvm[idx_1se]),
      cvsd        = as.numeric(cvmod$cvsd[idx_1se]),
      cvlo        = as.numeric(cvmod$cvlo[idx_1se]),
      cvup        = as.numeric(cvmod$cvup[idx_1se])
    )
  } # α loop
  toc()
  
  # Collate inner tuning results (one row per α per rule)
  tuning_results_min <- dplyr::bind_rows(per_alpha_list_min)
  tuning_results_1se <- dplyr::bind_rows(per_alpha_list_1se)
  
  # Winners across α (tie-breaks: fewer nzero → larger λ → smaller α)
  best_min <- tuning_results_min %>%
    dplyr::arrange(cvm, nzero, dplyr::desc(lambda), alpha) %>%
    dplyr::slice(1)
  
  best_1se <- tuning_results_1se %>%
    dplyr::arrange(cvm, nzero, dplyr::desc(lambda), alpha) %>%
    dplyr::slice(1)
  
  print(as.data.frame(best_min),  row.names = FALSE)
  print(as.data.frame(best_1se),  row.names = FALSE)
  message(sprintf("Inner winner @ lambda.min (outer %d): alpha=%.2f | lambda=%.6f | nzero=%d | CVM=%.5f (± %.5f)",
                  o, best_min$alpha, best_min$lambda, best_min$nzero, best_min$cvm, best_min$cvsd))
  message(sprintf("Inner winner @ lambda.1se (outer %d): alpha=%.2f | lambda=%.6f | nzero=%d | CVM=%.5f (± %.5f)",
                  o, best_1se$alpha, best_1se$lambda, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
  
  # Record winners (for audit)
  inner_winners_rows_min[[o]] <- tibble::tibble(
    outer_fold   = o,
    alpha        = best_min$alpha,
    alpha_idx    = best_min$alpha_idx,
    s            = best_min$s,
    lambda       = best_min$lambda,
    lambda_idx   = best_min$lambda_idx,
    nzero        = best_min$nzero,
    dev_ratio    = best_min$dev_ratio,
    cvm          = best_min$cvm,
    cvsd         = best_min$cvsd
  )
  inner_winners_rows_1se[[o]] <- tibble::tibble(
    outer_fold   = o,
    alpha        = best_1se$alpha,
    alpha_idx    = best_1se$alpha_idx,
    s            = best_1se$s,
    lambda       = best_1se$lambda,
    lambda_idx   = best_1se$lambda_idx,
    nzero        = best_1se$nzero,
    dev_ratio    = best_1se$dev_ratio,
    cvm          = best_1se$cvm,
    cvsd         = best_1se$cvsd
  )
  
  # --- Outer refit at the inner winners; evaluate on OUTER‑HOLDOUT ---
  
  # lambda.min
  fit_outer_min <- glmnet(
    x = X_train_outer, y = y_train_outer, weights = w_train_outer,
    family = "gaussian",
    alpha = best_min$alpha,
    lambda = best_min$lambda,
    standardize = TRUE, intercept = TRUE
  )
  pred_outer_hold_min <- as.numeric(predict(fit_outer_min, newx = X_hold_outer))
  fold_metrics_min <- compute_metrics(y_true = y_hold_outer, y_pred = pred_outer_hold_min, w = w_hold_outer)
  outer_metrics_min[[o]] <- fold_metrics_min
  outer_oof_pred_min[outer_hold_idx] <- pred_outer_hold_min
  message(sprintf("Outer %d @ lambda.min | RMSE=%.5f | MAE=%.5f | Bias=%.5f | Bias%%=%.3f | R2=%.5f",
                  o, fold_metrics_min["rmse"], fold_metrics_min["mae"], fold_metrics_min["bias"],
                  fold_metrics_min["bias_pct"], fold_metrics_min["r2"]))
  
  # lambda.1se
  fit_outer_1se <- glmnet(
    x = X_train_outer, y = y_train_outer, weights = w_train_outer,
    family = "gaussian",
    alpha = best_1se$alpha,
    lambda = best_1se$lambda,
    standardize = TRUE, intercept = TRUE
  )
  pred_outer_hold_1se <- as.numeric(predict(fit_outer_1se, newx = X_hold_outer))
  fold_metrics_1se <- compute_metrics(y_true = y_hold_outer, y_pred = pred_outer_hold_1se, w = w_hold_outer)
  outer_metrics_1se[[o]] <- fold_metrics_1se
  outer_oof_pred_1se[outer_hold_idx] <- pred_outer_hold_1se
  message(sprintf("Outer %d @ lambda.1se | RMSE=%.5f | MAE=%.5f | Bias=%.5f | Bias%%=%.3f | R2=%.5f",
                  o, fold_metrics_1se["rmse"], fold_metrics_1se["mae"], fold_metrics_1se["bias"],
                  fold_metrics_1se["bias_pct"], fold_metrics_1se["r2"]))
}
toc()
# > Nested Cross-Validation (glmnet; grouped=TRUE, keep=TRUE; lambda.min & lambda.1se): 6.085 sec elapsed

# Stop the parallel backend:
# doParallel::stopImplicitCluster()

#### ---- Aggregations over outer holds (both rules) ----

# Helpers to aggregate & print for a given rule
aggregate_rule <- function(outer_metrics_list, outer_oof_pred_vec, rule_label) {
  outer_metrics_matrix <- do.call(rbind, outer_metrics_list)
  metrics_outer_mean <- tibble::tibble(
    Rule = rule_label,
    Aggregation = "Simple-mean (outer folds)",
    RMSE = mean(outer_metrics_matrix[, "rmse"]),
    MAE  = mean(outer_metrics_matrix[, "mae"]),
    Bias = mean(outer_metrics_matrix[, "bias"]),
    `Bias%` = mean(outer_metrics_matrix[, "bias_pct"]),
    R2   = mean(outer_metrics_matrix[, "r2"])
  )
  stopifnot(all(!is.na(outer_oof_pred_vec)))
  pooled_vec <- compute_metrics(y_true = y_train, y_pred = outer_oof_pred_vec, w = w_train)
  metrics_outer_pooled <- tibble::tibble(
    Rule = rule_label,
    Aggregation = "Pooled-weighted OOF",
    RMSE = pooled_vec["rmse"],
    MAE  = pooled_vec["mae"],
    Bias = pooled_vec["bias"],
    `Bias%` = pooled_vec["bias_pct"],
    R2   = pooled_vec["r2"]
  )
  list(mean = metrics_outer_mean, pooled = metrics_outer_pooled,
       fold_matrix = outer_metrics_matrix)
}

agg_min  <- aggregate_rule(outer_metrics_min, outer_oof_pred_min, "lambda.min")
agg_1se  <- aggregate_rule(outer_metrics_1se, outer_oof_pred_1se, "lambda.1se")

# Report both rules side-by-side
message("\n==== Nested-CV aggregated performance (both rules) ====")
print(as.data.frame(dplyr::bind_rows(agg_min$mean, agg_min$pooled,
                                     agg_1se$mean, agg_1se$pooled)),
      row.names = FALSE)
# >       Rule               Aggregation     RMSE      MAE       Bias    Bias%        R2
# > lambda.min Simple-mean (outer folds) 69.08040 54.79631 0.08042310 1.982847 0.4127779
# > lambda.min       Pooled-weighted OOF 69.08524 54.78584 0.06727356 1.979963 0.4149784
# > lambda.1se Simple-mean (outer folds) 70.04903 55.62512 0.02189872 2.152405 0.3967233
# > lambda.1se       Pooled-weighted OOF 70.05290 55.61414 0.01171655 2.150153 0.3984752

# Inner winners tables (for audit)
inner_winners_table_min <- dplyr::bind_rows(inner_winners_rows_min)
inner_winners_table_1se <- dplyr::bind_rows(inner_winners_rows_1se)

message("\n==== Inner winners by outer fold — lambda.min ====")
print(as.data.frame(inner_winners_table_min), row.names = FALSE)
# > outer_fold alpha alpha_idx          s   lambda lambda_idx nzero dev_ratio      cvm      cvsd
# >          1   0.1         2 lambda.min 3.220739         55    50 0.4377548 4761.418 129.7359
# >          2   0.0         1 lambda.min 6.524812         97    53 0.4539486 4742.163 183.8103
# >          3   0.6         7 lambda.min 1.201905         46    39 0.4251878 4835.732 141.7866
# >          4   0.6         7 lambda.min 1.118646         47    42 0.4458259 4674.146 200.2680
# >          5   0.1         2 lambda.min 4.493036         51    45 0.4291806 4820.220 136.8975
message("\n==== Inner winners by outer fold — lambda.1se ====")
print(as.data.frame(inner_winners_table_1se), row.names = FALSE)
# > outer_fold alpha alpha_idx          s    lambda lambda_idx nzero  dev_ratio      cvm     cvsd
# >          1   0.2         3 lambda.1se 10.351574         35    28 0.4161577 4871.049 159.2154
# >          2   0.6         7 lambda.1se  4.599169         32    25 0.4239048 4907.910 212.6347
# >          3   0.3         4 lambda.1se  8.842124         32    25 0.3992829 4957.339 162.4014
# >          4   0.2         3 lambda.1se 13.547978         32    23 0.4142501 4846.426 200.3954
# >          5   0.1         2 lambda.1se 19.906941         35    32 0.4026266 4935.389 155.0611

# Per-fold outer holdout metrics tables
outer_folds_table_min <- tibble::tibble(
  outer_fold = seq_len(num_folds_outer),
  RMSE   = agg_min$fold_matrix[, "rmse"],
  MAE    = agg_min$fold_matrix[, "mae"],
  Bias   = agg_min$fold_matrix[, "bias"],
  `Bias%`= agg_min$fold_matrix[, "bias_pct"],
  R2     = agg_min$fold_matrix[, "r2"]
)
outer_folds_table_1se <- tibble::tibble(
  outer_fold = seq_len(num_folds_outer),
  RMSE   = agg_1se$fold_matrix[, "rmse"],
  MAE    = agg_1se$fold_matrix[, "mae"],
  Bias   = agg_1se$fold_matrix[, "bias"],
  `Bias%`= agg_1se$fold_matrix[, "bias_pct"],
  R2     = agg_1se$fold_matrix[, "r2"]
)

message("\n==== Per-fold outer holdout metrics — lambda.min ====")
print(as.data.frame(outer_folds_table_min), row.names = FALSE)
# > outer_fold     RMSE      MAE       Bias    Bias%         R2
# >          1 68.94541 55.18858  1.8718123 2.362098 0.4210022
# >          2 69.79294 55.87104  3.5045389 2.534447 0.3492882
# >          3 66.90360 53.00364 -0.2447814 1.871310 0.4606318
# >          4 70.86578 55.57958 -1.4587819 1.854344 0.3876495
# >          5 68.89428 54.33872 -3.2706725 1.292036 0.4453178
message("\n==== Per-fold outer holdout metrics — lambda.1se ====")
print(as.data.frame(outer_folds_table_1se), row.names = FALSE)
# > outer_fold     RMSE      MAE       Bias    Bias%         R2
# >          1 70.17619 55.98092  1.7202387 2.511365 0.4001456
# >          2 69.38975 56.00375  2.2601237 2.410364 0.3567847
# >          3 68.38187 53.70931  0.0486163 2.118996 0.4365333
# >          4 72.19869 56.83452 -0.5758465 2.238710 0.3643975
# >          5 70.09863 55.59710 -3.3436387 1.482592 0.4257554

# Consolidated: inner winners + outer holdout performance — both rules
message("\n==== Consolidated (winners + outer performance) — both rules ====")
consolidated_min <- inner_winners_table_min %>%
  dplyr::left_join(outer_folds_table_min, by = "outer_fold") %>%
  dplyr::mutate(Rule = "lambda.min")

consolidated_1se <- inner_winners_table_1se %>%
  dplyr::left_join(outer_folds_table_1se, by = "outer_fold") %>%
  dplyr::mutate(Rule = "lambda.1se")

consolidated_both <- dplyr::bind_rows(consolidated_min, consolidated_1se) %>%
  dplyr::relocate(Rule, outer_fold)

print(as.data.frame(consolidated_both), row.names = FALSE)
# >        Rule outer_fold alpha alpha_idx          s    lambda lambda_idx nzero dev_ratio      cvm      cvsd     RMSE      MAE      Bias     Bias%       R2
# >  lambda.min          1   0.1         2 lambda.min  3.220739         55    50 0.4377548 4761.418 129.7359 68.94541 55.18858  1.8718123 2.362098 0.4210022
# >  lambda.min          2   0.0         1 lambda.min  6.524812         97    53 0.4539486 4742.163 183.8103 69.79294 55.87104  3.5045389 2.534447 0.3492882
# >  lambda.min          3   0.6         7 lambda.min  1.201905         46    39 0.4251878 4835.732 141.7866 66.90360 53.00364 -0.2447814 1.871310 0.4606318
# >  lambda.min          4   0.6         7 lambda.min  1.118646         47    42 0.4458259 4674.146 200.2680 70.86578 55.57958 -1.4587819 1.854344 0.3876495
# >  lambda.min          5   0.1         2 lambda.min  4.493036         51    45 0.4291806 4820.220 136.8975 68.89428 54.33872 -3.2706725 1.292036 0.4453178
# >  lambda.1se          1   0.2         3 lambda.1se 10.351574         35    28 0.4161577 4871.049 159.2154 70.17619 55.98092  1.7202387 2.511365 0.4001456
# >  lambda.1se          2   0.6         7 lambda.1se  4.599169         32    25 0.4239048 4907.910 212.6347 69.38975 56.00375  2.2601237 2.410364 0.3567847
# >  lambda.1se          3   0.3         4 lambda.1se  8.842124         32    25 0.3992829 4957.339 162.4014 68.38187 53.70931  0.0486163 2.118996 0.4365333
# >  lambda.1se          4   0.2         3 lambda.1se 13.547978         32    23 0.4142501 4846.426 200.3954 72.19869 56.83452 -0.5758465 2.238710 0.3643975
# >  lambda.1se          5   0.1         2 lambda.1se 19.906941         35    32 0.4026266 4935.389 155.0611 70.09863 55.59710 -3.3436387 1.482592 0.4257554