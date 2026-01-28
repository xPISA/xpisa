# ---- III. Predictive Modelling: Version 3.4 ----
#
# Nested Cross-Validation: 
#  inner - xgb.cv with manual K-folds + OOF reconstruction (tuning hyperparameters)
#  outer - xgb.train (evaluating performance)

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

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("tidyverse","xgboost","Matrix","haven","broom","tictoc","DiagrammeR"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         tidyverse             xgboost              Matrix               haven               broom              tictoc          DiagrammeR 
# > "tidyverse 2.0.0"  "xgboost 1.7.11.1"      "Matrix 1.7.3"       "haven 2.5.4"       "broom 1.0.8"      "tictoc 1.2.1" "DiagrammeR 1.0.11"

# Load data
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE)
dim(pisa_2022_canada_merged)   # 23073  1699

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
stopifnot(!anyDuplicated(voi_num))                                      # Fail if any duplicates 
stopifnot(all(sapply(pisa_2022_canada_merged[, voi_num], is.numeric)))  # Ensure all predictors are numeric
length(voi_num)
# > [1] 53

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)   # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                 # Final student weight

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

### ---- Random Train/Validation/Test (80/10/10) split ----
set.seed(123)          # Ensure reproducibility
n <- nrow(temp_data)   # 20003
indices <- sample(n)   # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.80 * n)         # 5555
n_valid <- floor(0.10 * n)         # 694
n_test  <- n - n_train - n_valid   # 695

# Assign indices
train_idx <- indices[1:n_train]
valid_idx <- indices[(n_train + 1):(n_train + n_valid)]
test_idx  <- indices[(n_train + n_valid + 1):n]

# Subset the data
train_data <- temp_data[train_idx, ]
valid_data <- temp_data[valid_idx, ]
test_data  <- temp_data[test_idx, ]

## ---- Main model using final student weights (W_FSTUWT) ----
### ---- Nested Cross-Validation for PV1MATH only ----

# Define target plausible value
pv1math <- pvmaths[1]   # "PV1MATH"

# Prepare training/validation/test split columns
X_train <- train_data[, voi_num]
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

X_valid <- valid_data[, voi_num]
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

X_test  <- test_data[,  voi_num]
y_test  <- test_data[[pv1math]]
w_test  <- test_data[[final_wt]]

# Create DMatrix (note: dvalid/dtest not used in nested CV; kept for symmetry)
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train, weight = w_train)
dvalid <- xgb.DMatrix(data = as.matrix(X_valid), label = y_valid, weight = w_valid)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test,  weight = w_test)

# --- Helpers: folds and weights ---

# Build K manual folds (indices) with a fixed seed; over N rows
make_folds <- function(n_cv, num_folds = 5L, seed = 123L) {
  set.seed(seed)
  cv_order <- sample.int(n_cv)
  bounds <- floor(seq(0, n_cv, length.out = num_folds + 1))
  cv_folds <- vector("list", num_folds)
  for (k in seq_len(num_folds)) cv_folds[[k]] <- cv_order[(bounds[k] + 1):bounds[k + 1]]
  stopifnot(identical(sort(unlist(cv_folds)), seq_len(n_cv)))
  cv_folds
}

# Print weight diagnostics 
print_weight_balance <- function(w, folds, title = "Weight balance") {
  s <- tibble::tibble(
    fold   = seq_along(folds),
    n      = vapply(folds, length, integer(1)),
    w_sum  = vapply(folds, function(idx) sum(w[idx]), numeric(1)),
    w_mean = vapply(folds, function(idx) mean(w[idx]), numeric(1)),
    w_med  = vapply(folds, function(idx) median(w[idx]), numeric(1)),
    w_effn = vapply(folds, function(idx) {
      wi <- w[idx]; (sum(wi)^2) / sum(wi^2)
    }, numeric(1))
  ) |>
    dplyr::mutate(w_share = w_sum / sum(w_sum))
  message(title)
  print(s, n = Inf, width = Inf)
  cat(sprintf("Weight share range: [%.3f, %.3f]\n", min(s$w_share), max(s$w_share)))
  cat(sprintf("Max/Min share ratio: %.3f\n", max(s$w_share) / min(s$w_share)))
  cat(sprintf("Coeff. of variation of shares: %.3f\n\n", stats::sd(s$w_share) / mean(s$w_share)))
}

# Assert xgboost callback exists 
if (is.null(tryCatch(getFromNamespace("cb.cv.predict", "xgboost"), error = function(e) NULL))) {
  stop("xgboost build lacks cb.cv.predict(save_models=TRUE). Update/repair your xgboost installation.")
}

# --- Define the hyperparameter grid  ---
grid <- expand.grid(
  nrounds   = c(100, 200, 300),
  max_depth = c(4, 6, 8),
  eta       = c(0.01, 0.05, 0.10, 0.30),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)
stopifnot(is.data.frame(grid), nrow(grid) >= 1)

# --- Outer folds over TRAIN (X_train / y_train / w_train) ---
num_folds_outer <- 5L
n_cv_outer <- nrow(X_train)
outer_folds <- make_folds(n_cv_outer, num_folds_outer)

# Show outer-fold weight diagnostics
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

# Storage for outer predictions/metrics (@ OOF-best only)
outer_oof_pred      <- rep(NA_real_, n_cv_outer)
outer_metrics       <- vector("list", num_folds_outer)
inner_winners_rows  <- vector("list", num_folds_outer)

# --- Nested loop: For each outer fold, run inner tuner on OUTER-TRAIN ---
tictoc::tic("Nested Cross-Validation")
for (o in seq_len(num_folds_outer)) {
  message(sprintf("\n==== Outer fold %d/%d ====", o, num_folds_outer))
  outer_hold_idx  <- outer_folds[[o]]
  outer_train_idx <- setdiff(seq_len(n_cv_outer), outer_hold_idx)
  
  # OUTER-TRAIN / OUTER-HOLDOUT matrices
  dtrain_outer   <- xgb.DMatrix(
    data  = as.matrix(X_train[outer_train_idx, , drop = FALSE]),
    label = y_train[outer_train_idx],
    weight= w_train[outer_train_idx]
  )
  dholdout_outer <- xgb.DMatrix(
    data  = as.matrix(X_train[outer_hold_idx, , drop = FALSE]),
    label = y_train[outer_hold_idx],
    weight= w_train[outer_hold_idx]
  )
  
  # --- Inner folds (on OUTER-TRAIN) ---
  n_cv_inner <- length(outer_train_idx)
  num_folds_inner <- 5L
  inner_folds_local <- make_folds(n_cv_inner, num_folds_inner)
  
  # Map inner folds (local -> absolute indices in full TRAIN)
  inner_folds_abs <- lapply(inner_folds_local, function(idxs) outer_train_idx[idxs])
  stopifnot(all(vapply(inner_folds_abs, function(v) all(v %in% outer_train_idx), logical(1))))
  
  # Inner weight diagnostics 
  print_weight_balance(w_train[outer_train_idx], inner_folds_local,
                       title = sprintf("Inner CV (outer fold %d): weight balance over OUTER-TRAIN", o))
  
  # Held-out DMatrices per inner fold (reused for OOF predictions)
  dvalid_fold_inner <- lapply(seq_len(num_folds_inner), function(k) {
    idx_abs <- inner_folds_abs[[k]]
    xgb.DMatrix(
      as.matrix(X_train[idx_abs, , drop = FALSE]),
      label = y_train[idx_abs], weight = w_train[idx_abs]
    )
  })
  
  # --- Inner tuner (CV+OOF tracking) ---
  best_rmse_by_cv_inner  <- Inf
  best_rmse_by_oof_inner <- Inf
  best_cv_info  <- NULL
  best_oof_info <- NULL
  
  # Containers (per inner run, for inspection)
  tuning_results <- tibble::tibble(
    grid_id = integer(),
    param_name = character(),
    nrounds = integer(),
    max_depth = integer(),
    eta = double(),
    best_iter_by_cv = integer(),
    best_rmse_by_cv = double(),
    test_rmse_std_at_cvbest   = double(),
    train_rmse_mean_at_cvbest = double(),
    train_rmse_std_at_cvbest  = double(),
    best_iter_by_oof = integer(),
    best_rmse_by_oof = double(),
    rmse_oof_at_cvbest_iter   = double(),
    rmse_oof_at_final_iter    = double()
  )
  cv_eval_list   <- vector("list", nrow(grid))
  oof_curve_list <- vector("list", nrow(grid))
  
  tictoc::tic(sprintf("Inner tuning (outer fold %d)", o))
  for (i in seq_len(nrow(grid))) {
    row <- grid[i, ]
    message(sprintf("  [Inner %d/%d] nrounds=%d, max_depth=%d, eta=%.3f",
                    i, nrow(grid), row$nrounds, row$max_depth, row$eta))
    
    params <- list(
      objective   = "reg:squarederror",
      max_depth   = row$max_depth,
      eta         = row$eta,
      eval_metric = "rmse",
      nthread     = max(1, parallel::detectCores() - 1)
    )
    
    # 1) xgb.cv on OUTER-TRAIN with manual inner folds
    cv_mod <- xgb.cv(
      params  = params,
      data    = dtrain_outer,
      nrounds = row$nrounds,
      folds   = inner_folds_local,    # local (relative) folds for dtrain_outer
      showsd  = TRUE,
      verbose = FALSE,
      early_stopping_rounds = NULL,
      prediction = TRUE,
      stratified = FALSE,
      callbacks = list(getFromNamespace("cb.cv.predict", "xgboost")(save_models = TRUE))
    )
    
    # CV rule (fold-mean)
    best_iter_by_cv_i <- which.min(cv_mod$evaluation_log$test_rmse_mean)
    best_rmse_by_cv_i <- cv_mod$evaluation_log$test_rmse_mean[best_iter_by_cv_i]
    
    # Sanity: ensure fold models exist
    stopifnot(!is.null(cv_mod$models), length(cv_mod$models) == num_folds_inner)
    
    # 2) Manual OOF reconstruction across inner folds, pooled weighted
    oof_pred_matrix_i  <- matrix(NA_real_, nrow = n_cv_inner, ncol = row$nrounds)
    rmse_oof_by_iter_i <- numeric(row$nrounds)
    for (iter in seq_len(row$nrounds)) {
      for (k in seq_len(num_folds_inner)) {
        idx_abs <- inner_folds_abs[[k]]
        idx_loc <- match(idx_abs, outer_train_idx)  # absolute -> local positions (1..n_cv_inner)
        stopifnot(!any(is.na(idx_loc)))
        oof_pred_matrix_i[idx_loc, iter] <- predict(
          cv_mod$models[[k]],
          dvalid_fold_inner[[k]],
          iterationrange = c(1, iter + 1)
        )
      }
      rmse_oof_by_iter_i[iter] <- sqrt(
        sum(w_train[outer_train_idx] * (y_train[outer_train_idx] - oof_pred_matrix_i[, iter])^2) /
          sum(w_train[outer_train_idx])
      )
    }
    best_iter_by_oof_i        <- which.min(rmse_oof_by_iter_i)
    best_rmse_by_oof_i        <- rmse_oof_by_iter_i[best_iter_by_oof_i]
    rmse_oof_at_cvbest_iter_i <- rmse_oof_by_iter_i[best_iter_by_cv_i]
    rmse_oof_at_final_iter_i  <- sqrt(
      sum(w_train[outer_train_idx] * (y_train[outer_train_idx] - cv_mod$pred)^2) /
        sum(w_train[outer_train_idx])
    )
    # At last iter, reconstructed OOF equals cv_mod$pred
    stopifnot(isTRUE(all.equal(oof_pred_matrix_i[, row$nrounds], cv_mod$pred)))
    
    # Save logs for this config
    cv_eval_list[[i]]   <- dplyr::mutate(as.data.frame(cv_mod$evaluation_log), grid_id = i)
    oof_curve_list[[i]] <- rmse_oof_by_iter_i
    tuning_results <- dplyr::add_row(
      tuning_results,
      grid_id = i,
      param_name = sprintf("nrounds=%d, max_depth=%d, eta=%.3f", row$nrounds, row$max_depth, row$eta),
      nrounds = row$nrounds,
      max_depth = row$max_depth,
      eta = row$eta,
      best_iter_by_cv = best_iter_by_cv_i,
      best_rmse_by_cv = best_rmse_by_cv_i,
      test_rmse_std_at_cvbest   = cv_mod$evaluation_log$test_rmse_std[best_iter_by_cv_i],
      train_rmse_mean_at_cvbest = cv_mod$evaluation_log$train_rmse_mean[best_iter_by_cv_i],
      train_rmse_std_at_cvbest  = cv_mod$evaluation_log$train_rmse_std[best_iter_by_cv_i],
      best_iter_by_oof = best_iter_by_oof_i,
      best_rmse_by_oof = best_rmse_by_oof_i,
      rmse_oof_at_cvbest_iter   = rmse_oof_at_cvbest_iter_i,
      rmse_oof_at_final_iter    = rmse_oof_at_final_iter_i
    )
    
    # Track winners (CV rule and OOF rule)
    if (best_rmse_by_cv_i < best_rmse_by_cv_inner) {
      best_rmse_by_cv_inner <- best_rmse_by_cv_i
      best_cv_info <- list(
        grid_id = i, params = params, row = row,
        best_iter_by_cv  = best_iter_by_cv_i,
        best_iter_by_oof = best_iter_by_oof_i,
        cv_eval  = cv_eval_list[[i]],
        oof_curve= oof_curve_list[[i]]
      )
    }
    if (best_rmse_by_oof_i < best_rmse_by_oof_inner) {
      best_rmse_by_oof_inner <- best_rmse_by_oof_i
      best_oof_info <- list(
        grid_id = i, params = params, row = row,
        best_iter_by_cv  = best_iter_by_cv_i,
        best_iter_by_oof = best_iter_by_oof_i,
        cv_eval  = cv_eval_list[[i]],
        oof_curve= oof_curve_list[[i]]
      )
    }
  } # grid loop
  tictoc::toc()
  
  # Inspect winners (optional summaries)
  message("Top configs by CV fold-mean (inner):")
  tuning_results %>%
    dplyr::arrange(best_rmse_by_cv) %>% head(5) %>% as.data.frame() %>% print(row.names = FALSE)
  
  message("Top configs by pooled OOF RMSE (inner):")
  tuning_results %>%
    dplyr::arrange(best_rmse_by_oof) %>% head(5) %>% as.data.frame() %>% print(row.names = FALSE)
  
  # Assert CV winner and OOF winner are the same config
  stopifnot(best_cv_info$grid_id == best_oof_info$grid_id)
  
  # Record the inner winners (one row per outer fold)
  inner_winners_rows[[o]] <- tibble::tibble(
    outer_fold       = o,
    grid_id          = best_oof_info$grid_id,
    n_cap            = best_oof_info$row$nrounds,
    max_depth        = best_oof_info$row$max_depth,
    eta              = best_oof_info$row$eta,
    best_iter_by_cv  = best_oof_info$best_iter_by_cv,
    best_rmse_by_cv  = min(best_oof_info$cv_eval$test_rmse_mean),
    best_iter_by_oof = best_oof_info$best_iter_by_oof,
    best_rmse_by_oof = min(best_oof_info$oof_curve)
  )
  
  message(sprintf("Inner winners agree (grid %d): n_cap=%d, max_depth=%d, eta=%.3f | CV-best iter=%d | OOF-best iter=%d",
                  best_oof_info$grid_id, best_oof_info$row$nrounds,
                  best_oof_info$row$max_depth, best_oof_info$row$eta,
                  best_oof_info$best_iter_by_cv, best_oof_info$best_iter_by_oof))
  
  # --- Outer refit: once to n_cap (no early stopping; no watchlist logging of holdout) ---
  main_model_outer <- xgb.train(
    params  = list(                                  # <=> params = best_oof_info$params
      objective = "reg:squarederror",
      max_depth = best_oof_info$params$max_depth,
      eta       = best_oof_info$params$eta,
      eval_metric = "rmse",
      nthread   = max(1, parallel::detectCores() - 1)
    ),
    data    = dtrain_outer,
    nrounds = best_oof_info$row$nrounds,             # n_cap
    verbose = 1,
    early_stopping_rounds = NULL
  )
  
  # Predict OUTER-HOLDOUT at inner OOF-best
  pred_outer_hold <- predict(
    main_model_outer, dholdout_outer,
    iterationrange = c(1, best_oof_info$best_iter_by_oof + 1)
  )
  
  # Metrics on OUTER-HOLDOUT @ OOF-best only
  fold_metrics <- compute_metrics(
    y_true = y_train[outer_hold_idx],
    y_pred = pred_outer_hold,
    w      = w_train[outer_hold_idx]
  )
  outer_metrics[[o]] <- fold_metrics
  
  # Save into global outer OOF vector
  outer_oof_pred[outer_hold_idx] <- pred_outer_hold
  
  message(sprintf("Outer fold %d metrics @ OOF-best | RMSE=%.5f | MAE=%.5f | Bias=%.5f | Bias%%=%.3f | R2=%.5f",
                  o, fold_metrics["rmse"], fold_metrics["mae"], fold_metrics["bias"],
                  fold_metrics["bias_pct"], fold_metrics["r2"]))
}
tictoc::toc()
# > Nested Cross-Validation: 2562.404 sec elapsed

#### ---- Aggregation over outer holds ----

# 1) Simple average of the 5 per-fold metrics
outer_metrics_matrix <- do.call(rbind, outer_metrics)
metrics_outer_mean <- tibble::tibble(
  Aggregation = "Simple-mean (outer folds)",
  RMSE   = mean(outer_metrics_matrix[, "rmse"]),
  MAE    = mean(outer_metrics_matrix[, "mae"]),
  Bias   = mean(outer_metrics_matrix[, "bias"]),
  `Bias%`= mean(outer_metrics_matrix[, "bias_pct"]),
  R2     = mean(outer_metrics_matrix[, "r2"])
)

# 2) Pooled weighted OOF across all outer holdouts
stopifnot(all(!is.na(outer_oof_pred)))
metrics_outer_pooled_vec <- compute_metrics(
  y_true = y_train,
  y_pred = outer_oof_pred,
  w      = w_train
)
metrics_outer_pooled <- tibble::tibble(
  Aggregation = "Pooled-weighted OOF",
  RMSE   = metrics_outer_pooled_vec["rmse"],
  MAE    = metrics_outer_pooled_vec["mae"],
  Bias   = metrics_outer_pooled_vec["bias"],
  `Bias%`= metrics_outer_pooled_vec["bias_pct"],
  R2     = metrics_outer_pooled_vec["r2"]
)

# Report both
message("\n==== Nested-CV aggregated performance (@ inner OOF-best only) ====")
print(as.data.frame(dplyr::bind_rows(metrics_outer_mean, metrics_outer_pooled)), row.names = FALSE)
# >               Aggregation     RMSE      MAE      Bias    Bias%        R2
# > Simple-mean (outer folds) 67.08319 52.96145 -1.079706 1.617121 0.4463448
# >       Pooled-weighted OOF 67.08886 52.94919 -1.088901 1.615858 0.4483011

# Inner winners table (one row per outer fold) 
inner_winners_table <- dplyr::bind_rows(inner_winners_rows)
message("\n==== Inner winners (by outer fold) ====")
print(as.data.frame(inner_winners_table), row.names = FALSE)
# > outer_fold grid_id n_cap max_depth  eta best_iter_by_cv best_rmse_by_cv best_iter_by_oof best_rmse_by_oof
# >          1      11   200         4 0.05             194        66.66252              194         66.71581
# >          2      12   300         4 0.05             234        66.62505              234         66.67123
# >          3      19   100         4 0.10              77        67.48883               77         67.48858
# >          4      12   300         4 0.05             265        66.55232              265         66.62907
# >          5      19   100         4 0.10              81        66.43799               81         66.47821

# Return per-fold table for audit
outer_folds_table <- tibble::tibble(
  outer_fold = seq_len(num_folds_outer),
  RMSE   = outer_metrics_matrix[, "rmse"],
  MAE    = outer_metrics_matrix[, "mae"],
  Bias   = outer_metrics_matrix[, "bias"],
  `Bias%`= outer_metrics_matrix[, "bias_pct"],
  R2     = outer_metrics_matrix[, "r2"]
)
print(as.data.frame(outer_folds_table), row.names = FALSE)
# > outer_fold     RMSE      MAE       Bias     Bias%        R2
# >          1 66.48365 53.55831 -0.6876014 1.6595417 0.4616113
# >          2 67.89107 53.91570  4.2791805 2.5901163 0.3842689
# >          3 65.60659 51.39518 -3.0757766 1.2600809 0.4813417
# >          4 68.27600 52.88080 -1.7112060 1.6016042 0.4315881
# >          5 67.15863 53.05724 -4.2031262 0.9742637 0.4729139

# Consolidated table: join inner winners with outer-fold performance
message("\n==== Consolidated table: inner winners + outer holdout performance ====")
inner_winners_table %>%
  dplyr::left_join(outer_folds_table, by = "outer_fold") %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > outer_fold grid_id n_cap max_depth  eta best_iter_by_cv best_rmse_by_cv best_iter_by_oof best_rmse_by_oof     RMSE      MAE       Bias     Bias%        R2
# >          1      11   200         4 0.05             194        66.66252              194         66.71581 66.48365 53.55831 -0.6876014 1.6595417 0.4616113
# >          2      12   300         4 0.05             234        66.62505              234         66.67123 67.89107 53.91570  4.2791805 2.5901163 0.3842689
# >          3      19   100         4 0.10              77        67.48883               77         67.48858 65.60659 51.39518 -3.0757766 1.2600809 0.4813417
# >          4      12   300         4 0.05             265        66.55232              265         66.62907 68.27600 52.88080 -1.7112060 1.6016042 0.4315881
# >          5      19   100         4 0.10              81        66.43799               81         66.47821 67.15863 53.05724 -4.2031262 0.9742637 0.4729139