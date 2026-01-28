# ---- II. Predictive Modelling: Version 2.4 ----

# xgb.cv with manual K-folds + OOF reconstruction (tuning hyperparameters)

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
  select(CNTSCHID, CNTSTUID,                      # IDs
         all_of(final_wt), all_of(rep_wts),       # Weights
         all_of(pvmaths), all_of(voi_num)) %>%    # PVs + predictors
  filter(if_all(all_of(voi_num), ~ !is.na(.)))    # Listwise deletion for predictors

# To compare performance without listwise deletion 
# temp_data <- pisa_2022_canada_merged %>%
#   select(CNTSCHID, CNTSTUID,                    # IDs
#          all_of(final_wt), all_of(rep_wts),     # Weights
#          all_of(pvmaths), all_of(voi_num))      # PVs + predictors

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

## ---- 2. PV1MATH only ----

# --- Remark ---
# 1) Repeat the same process for PV2MATH - PV10MATH.
# 2) Apply best results from PV1MATH to all plausible values in mathematics. 

### ---- Fit main model using final student weights (W_FSTUWT) on the training data----

#### ---- Tune XGBoost model for PV1MATH only: xgb.cv + Manual Folds + OOF----

# Define target plausible value
pv1math <- pvmaths[1]   # "PV1MATH"

# Prepare training/validation/test data
X_train <- train_data[, voi_num]
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

X_valid <- valid_data[, voi_num]
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

X_test  <- test_data[,  voi_num]
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
# > 1     1  1111 17147.   15.4  9.12   630.   0.193
# > 2     2  1111 17872.   16.1 10.2    653.   0.201
# > 3     3  1111 18283.   16.5 10.2    621.   0.205
# > 4     4  1111 17925.   16.1  9.39   633.   0.201
# > 5     5  1111 17769.   16.0  9.57   647.   0.200
# > Weight share range: [0.193, 0.205]
# > Max/Min share ratio: 1.066
# > Coeff. of variation of shares: 0.023

# Held-out DMatrices per fold (reused for OOF predictions)
dvalid_fold <- lapply(seq_len(num_folds), function(k) {
  idx <- cv_folds[[k]]
  xgb.DMatrix(as.matrix(X_train[idx, , drop = FALSE]),
              label = y_train[idx], weight = w_train[idx])
})

# Ensure callback exists; then call directly inside xgb.cv()
if (is.null(tryCatch(getFromNamespace("cb.cv.predict", "xgboost"), error = function(e) NULL))) {
  stop("xgboost build lacks cb.cv.predict(save_models=TRUE). Update/repair your xgboost installation.")
}

# Hyperparameter grid (3 × 3 × 4 = 36) 
grid <- expand.grid(
  nrounds   = c(100, 200, 300),
  max_depth = c(4, 6, 8),
  eta       = c(0.01, 0.05, 0.10, 0.30),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

# Containers across grid (include BOTH CV- and OOF-based summaries per config)
tuning_results <- tibble::tibble(
  grid_id = integer(),
  param_name = character(),
  nrounds = integer(),
  max_depth = integer(),
  eta = double(),
  # CV-based (fold-mean)
  best_iter_by_cv = integer(),
  best_rmse_by_cv = double(),
  test_rmse_std_at_cvbest   = double(),
  train_rmse_mean_at_cvbest = double(),
  train_rmse_std_at_cvbest  = double(),
  # OOF-based (pooled, weighted)
  best_iter_by_oof = integer(),
  best_rmse_by_oof = double(),
  rmse_oof_at_cvbest_iter   = double(),
  rmse_oof_at_final_iter    = double()
)

cv_eval_list   <- vector("list", nrow(grid))   # per-config evaluation_log (means & sds)
oof_curve_list <- vector("list", nrow(grid))   # per-config OOF RMSE curve (per iteration)

# Track BOTH winners across the grid (renamed per request; no "_overall")
best_rmse_by_cv  <- Inf
best_rmse_by_oof <- Inf
best_cv_info  <- NULL   # list(grid_id, params, row, best_iter_by_cv, best_iter_by_oof, cv_eval, oof_curve)
best_oof_info <- NULL

# Grid search 
set.seed(123)
tictoc::tic("Tuning (xgb.cv + Manual OOF)")
for (i in seq_len(nrow(grid))) {
  row <- grid[i, ]
  message(sprintf("Grid %d/%d: nrounds=%d, max_depth=%d, eta=%.3f",
                  i, nrow(grid), row$nrounds, row$max_depth, row$eta))
  
  params <- list(
    objective   = "reg:squarederror",
    max_depth   = row$max_depth,
    eta         = row$eta,
    eval_metric = "rmse",
    nthread     = max(1, parallel::detectCores() - 1)
  )
  
  # 1) Run xgb.cv to get CV curves (means ± sd) and OOF snapshot @ final iter + fold models
  cv_mod <- xgb.cv(
    params  = params,
    data    = dtrain,
    nrounds = row$nrounds,
    folds   = cv_folds,
    showsd  = TRUE,
    verbose = TRUE,   # requested logical TRUE
    early_stopping_rounds = NULL,
    prediction = TRUE,
    stratified = FALSE,
    callbacks = list(getFromNamespace("cb.cv.predict", "xgboost")(save_models = TRUE))
  )
  
  # CV-best iteration by validation RMSE (fold-mean) — avoid temp 'ev'
  best_iter_by_cv_i <- which.min(cv_mod$evaluation_log$test_rmse_mean)
  best_rmse_by_cv_i <- cv_mod$evaluation_log$test_rmse_mean[best_iter_by_cv_i]
  
  # 2) OOF curve reconstruction using the saved fold boosters 
  oof_pred_matrix_i <- matrix(NA_real_, nrow = n_cv, ncol = row$nrounds)
  rmse_oof_by_iter_i <- numeric(row$nrounds)
  for (iter in seq_len(row$nrounds)) {
    for (k in seq_len(num_folds)) {
      idx <- cv_folds[[k]]
      oof_pred_matrix_i[idx, iter] <- predict(
        cv_mod$models[[k]], dvalid_fold[[k]],
        iterationrange = c(1, iter + 1)
      )
    }
    rmse_oof_by_iter_i[iter] <- sqrt(
      sum(w_train * (y_train - oof_pred_matrix_i[, iter])^2) / sum(w_train)
    )
  }
  best_iter_by_oof_i <- which.min(rmse_oof_by_iter_i)
  best_rmse_by_oof_i <- rmse_oof_by_iter_i[best_iter_by_oof_i]
  rmse_oof_at_cvbest_iter_i <- rmse_oof_by_iter_i[best_iter_by_cv_i]
  rmse_oof_at_final_iter_i  <- sqrt(sum(w_train * (y_train - cv_mod$pred)^2) / sum(w_train))
  stopifnot(isTRUE(all.equal(oof_pred_matrix_i[, row$nrounds], cv_mod$pred)))
  
  # Save per-config artifacts
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
  
  # Track best-by-CV across grid
  if (best_rmse_by_cv_i < best_rmse_by_cv) {
    best_rmse_by_cv <- best_rmse_by_cv_i
    best_cv_info <- list(
      grid_id = i, params = params, row = row,
      best_iter_by_cv  = best_iter_by_cv_i,
      best_iter_by_oof = best_iter_by_oof_i,
      cv_eval  = cv_eval_list[[i]],
      oof_curve= oof_curve_list[[i]]
    )
  }
  # Track best-by-OOF across grid
  if (best_rmse_by_oof_i < best_rmse_by_oof) {
    best_rmse_by_oof <- best_rmse_by_oof_i
    best_oof_info <- list(
      grid_id = i, params = params, row = row,
      best_iter_by_cv  = best_iter_by_cv_i,
      best_iter_by_oof = best_iter_by_oof_i,
      cv_eval  = cv_eval_list[[i]],
      oof_curve= oof_curve_list[[i]]
    )
  }
}
tictoc::toc()
# > Tuning (xgb.cv + Manual OOF): 695.128 sec elapsed

##### ---- Explore tuning output (both rules) ----
message("Top configs by CV fold-mean (test_rmse_mean @ CV-best):")
tuning_results %>% dplyr::arrange(best_rmse_by_cv) %>% head(10) %>% as.data.frame() %>% print(row.names = FALSE)
# > grid_id                          param_name nrounds max_depth  eta best_iter_by_cv best_rmse_by_cv test_rmse_std_at_cvbest train_rmse_mean_at_cvbest train_rmse_std_at_cvbest best_iter_by_oof best_rmse_by_oof rmse_oof_at_cvbest_iter rmse_oof_at_final_iter
# >      11 nrounds=200, max_depth=4, eta=0.050     200         4 0.05             166        66.77491               1.3798175                  47.64497                0.1315420              166         66.77939                66.77939               66.80400
# >      12 nrounds=300, max_depth=4, eta=0.050     300         4 0.05             166        66.77491               1.3798175                  47.64497                0.1315420              166         66.77939                66.77939               66.98657
# >      14 nrounds=200, max_depth=6, eta=0.050     200         6 0.05             199        66.84798               1.2561452                  26.68397                0.4499043              199         66.85173                66.85173               66.85576
# >      15 nrounds=300, max_depth=6, eta=0.050     300         6 0.05             199        66.84798               1.2561452                  26.68397                0.4499043              199         66.85173                66.85173               66.95288
# >      20 nrounds=200, max_depth=4, eta=0.100     200         4 0.10             101        67.02349               0.9669641                  45.43514                0.3057141              101         67.02352                67.02352               67.15253
# >      21 nrounds=300, max_depth=4, eta=0.100     300         4 0.10             101        67.02349               0.9669641                  45.43514                0.3057141              101         67.02352                67.02352               67.30712
# >      19 nrounds=100, max_depth=4, eta=0.100     100         4 0.10              91        67.03722               0.8948536                  46.61440                0.2545616               91         67.03661                67.03661               67.05199
# >      22 nrounds=100, max_depth=6, eta=0.100     100         6 0.10              96        67.11708               0.8488830                  27.41266                0.7103732               96         67.12035                67.12035               67.16355
# >      23 nrounds=200, max_depth=6, eta=0.100     200         6 0.10              96        67.11708               0.8488830                  27.41266                0.7103732               96         67.12035                67.12035               67.42764
# >      24 nrounds=300, max_depth=6, eta=0.100     300         6 0.10              96        67.11708               0.8488830                  27.41266                0.7103732               96         67.12035                67.12035               67.75880

message("Top configs by pooled OOF RMSE (min across iterations):")
tuning_results %>% dplyr::arrange(best_rmse_by_oof) %>% head(10) %>% as.data.frame() %>% print(row.names = FALSE)
# > grid_id                          param_name nrounds max_depth  eta best_iter_by_cv best_rmse_by_cv test_rmse_std_at_cvbest train_rmse_mean_at_cvbest train_rmse_std_at_cvbest best_iter_by_oof best_rmse_by_oof rmse_oof_at_cvbest_iter rmse_oof_at_final_iter
# >      11 nrounds=200, max_depth=4, eta=0.050     200         4 0.05             166        66.77491               1.3798175                  47.64497                0.1315420              166         66.77939                66.77939               66.80400
# >      12 nrounds=300, max_depth=4, eta=0.050     300         4 0.05             166        66.77491               1.3798175                  47.64497                0.1315420              166         66.77939                66.77939               66.98657
# >      14 nrounds=200, max_depth=6, eta=0.050     200         6 0.05             199        66.84798               1.2561452                  26.68397                0.4499043              199         66.85173                66.85173               66.85576
# >      15 nrounds=300, max_depth=6, eta=0.050     300         6 0.05             199        66.84798               1.2561452                  26.68397                0.4499043              199         66.85173                66.85173               66.95288
# >      20 nrounds=200, max_depth=4, eta=0.100     200         4 0.10             101        67.02349               0.9669641                  45.43514                0.3057141              101         67.02352                67.02352               67.15253
# >      21 nrounds=300, max_depth=4, eta=0.100     300         4 0.10             101        67.02349               0.9669641                  45.43514                0.3057141              101         67.02352                67.02352               67.30712
# >      19 nrounds=100, max_depth=4, eta=0.100     100         4 0.10              91        67.03722               0.8948536                  46.61440                0.2545616               91         67.03661                67.03661               67.05199
# >      22 nrounds=100, max_depth=6, eta=0.100     100         6 0.10              96        67.11708               0.8488830                  27.41266                0.7103732               96         67.12035                67.12035               67.16355
# >      23 nrounds=200, max_depth=6, eta=0.100     200         6 0.10              96        67.11708               0.8488830                  27.41266                0.7103732               96         67.12035                67.12035               67.42764
# >      24 nrounds=300, max_depth=6, eta=0.100     300         6 0.10              96        67.11708               0.8488830                  27.41266                0.7103732               96         67.12035                67.12035               67.75880
stopifnot(!is.null(best_cv_info), !is.null(best_oof_info))

message(sprintf("CV winner -> grid_id=%d | cap=%d | max_depth=%d | eta=%.3f | CV-best iter=%d | OOF-best iter=%d | CV-min=%.5f | OOF-min=%.5f",
                best_cv_info$grid_id, best_cv_info$row$nrounds, best_cv_info$row$max_depth, best_cv_info$row$eta,
                best_cv_info$best_iter_by_cv, best_cv_info$best_iter_by_oof,
                best_rmse_by_cv, min(best_cv_info$oof_curve)))
# > CV winner -> grid_id=11 | cap=200 | max_depth=4 | eta=0.050 | CV-best iter=166 | OOF-best iter=166 | CV-min=66.77491 | OOF-min=66.77939

message(sprintf("OOF winner -> grid_id=%d | cap=%d | max_depth=%d | eta=%.3f | CV-best iter=%d | OOF-best iter=%d | OOF-min=%.5f | CV-min@that=%.5f",
                best_oof_info$grid_id, best_oof_info$row$nrounds, best_oof_info$row$max_depth, best_oof_info$row$eta,
                best_oof_info$best_iter_by_cv, best_oof_info$best_iter_by_oof,
                best_rmse_by_oof, min(best_oof_info$cv_eval$test_rmse_mean)))
# > OOF winner -> grid_id=11 | cap=200 | max_depth=4 | eta=0.050 | CV-best iter=166 | OOF-best iter=166 | OOF-min=66.77939 | CV-min@that=66.77491

#### ---- Refit with OOF winner config ----

# Sanity: confirm CV-winner and OOF-winner are the SAME config (as assumed here).
stopifnot(best_cv_info$grid_id == best_oof_info$grid_id)

# Use the OOF-winner config to refit 
set.seed(123)
main_model <- xgb.train(
  params = list(                                            # <=> params = best_oof_info$params
    objective = "reg:squarederror",                         # Specify the learning task and the corresponding learning objective: reg:squarederror Regression with squared loss (Default)
    max_depth = best_oof_info$params$max_depth,                          
    eta = best_oof_info$params$eta,                                      
    eval_metric = "rmse",                                   # Default: metric will be assigned according to objective(rmse for regression) 
    nthread = max(1, parallel::detectCores() - 1)           # Manually specify number of thread
    #seed = 123                                             # Random number seed for reproducibility (e.g. tune subsample, colsample_bytree for regularization)
  ),
  data    = dtrain,
  nrounds = best_oof_info$row$nrounds,
  watchlist = list(train = dtrain, valid = dvalid),
  verbose = 1,
  early_stopping_rounds = NULL
)

# VALID-selected cutoff from refit 
best_iter <- which.min(main_model$evaluation_log$valid_rmse)  # 174
best_rmse <- min(main_model$evaluation_log$valid_rmse)        # 65.70228

##### ---- Explore refit, importance, and quick tree ----
main_model
print(as.data.frame(main_model$evaluation_log), row.names = FALSE)
# > iter train_rmse valid_rmse
# >    1  502.33164  501.02708
# >    2  477.73534  476.40862
# >  ...
# >  199   47.85030   65.76805
# >  200   47.77914   65.78005

xgb.importance(model = main_model) %>% head()    
# >     Feature       Gain      Cover  Frequency
# >      <char>      <num>      <num>      <num>
# > 1:  MATHEFF 0.40490042 0.07662199 0.04419703
# > 2:     ESCS 0.05621067 0.04199433 0.04251012
# > 3: FAMSUPSL 0.04832341 0.04297436 0.04251012
# > 4:   ANXMAT 0.04136378 0.03491656 0.02766532
# > 5:   FAMCON 0.02743916 0.03774013 0.03103914
# > 6: MATHEF21 0.02421468 0.03557355 0.02361673
xgb.importance(model = main_model, trees = 0:(best_oof_info$best_iter_by_oof - 1)) %>% head() 
# >     Feature      Gain     Cover Frequency
# >      <char>     <num>     <num>     <num>
# > 1:  MATHEFF 0.41831806 0.09058278 0.04665314
# > 2:     ESCS 0.05705440 0.04766530 0.04421907
# > 3: FAMSUPSL 0.04921920 0.04812288 0.04665314
# > 4:   ANXMAT 0.04237571 0.03737265 0.03083164
# > 5:   FAMCON 0.02746064 0.04394888 0.03286004
# > 6: MATHEF21 0.02469909 0.03831558 0.02555781

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
  geom_vline(xintercept = best_iter, linetype = 2, alpha = 0.4) +
  annotate("text", x = best_iter, y = max(main_model$evaluation_log$valid_rmse) + 5,
           label = paste("Best iter =", best_oof_info$best_iter_by_oof), size = 3, vjust = 0) + 
  ggplot2::labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round", y = "RMSE", color = "Dataset"
  ) +
  ggplot2::theme_minimal()

# Tree snapshots
xgb.plot.tree(model = main_model, trees = 0)
xgb.plot.tree(model = main_model, trees = max(0, best_cv_info$best_iter_by_cv - 1))

#### ---- Predict and evaluate performance on TRAIN/VALID/TEST ----

# CV-selected cutoff
pred_cv_train <- predict(main_model, dtrain, iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))
pred_cv_valid <- predict(main_model, dvalid, iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))
pred_cv_test  <- predict(main_model, dtest,  iterationrange = c(1, best_cv_info$best_iter_by_cv + 1))

# OOF-selected cutoff
pred_oof_train <- predict(main_model, dtrain, iterationrange = c(1, best_oof_info$best_iter_by_oof+ 1))
pred_oof_valid <- predict(main_model, dvalid, iterationrange = c(1, best_oof_info$best_iter_by_oof+ 1))
pred_oof_test  <- predict(main_model, dtest,  iterationrange = c(1, best_oof_info$best_iter_by_oof+ 1))

# VALID-selected cutoff (from refit)
pred_train <- predict(main_model, dtrain, iterationrange = c(1, best_iter + 1))
pred_valid <- predict(main_model, dvalid, iterationrange = c(1, best_iter + 1))
pred_test  <- predict(main_model, dtest,  iterationrange = c(1, best_iter + 1))

# Consistency check: VALID RMSE equals watchlist at best_iter
stopifnot(isTRUE(all.equal(
  unname(compute_metrics(y_valid, pred_valid, w_valid)["rmse"]),
  main_model$evaluation_log$valid_rmse[best_iter]
)))

# Metrics 
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
  R2      = c(compute_metrics(y_train, pred_oof_train,  w_train)["r2"],
              compute_metrics(y_valid, pred_oof_valid,  w_valid)["r2"],
              compute_metrics(y_test,  pred_oof_test,   w_test )["r2"])
)

metrics <- tibble::tibble(  # VALID-best (v1.4 naming)
  Model   = "VALID-best",
  Dataset = c("Training","Validation","Test"),
  RMSE    = c(compute_metrics(y_train, pred_train, w_train)["rmse"],
              compute_metrics(y_valid, pred_valid, w_valid)["rmse"],
              compute_metrics(y_test,  pred_test,  w_test )["rmse"]),
  MAE     = c(compute_metrics(y_train, pred_train, w_train)["mae"],
              compute_metrics(y_valid, pred_valid,  w_valid)["mae"],
              compute_metrics(y_test,  pred_test,   w_test )["mae"]),
  Bias    = c(compute_metrics(y_train, pred_train, w_train)["bias"],
              compute_metrics(y_valid, pred_valid,  w_valid)["bias"],
              compute_metrics(y_test,  pred_test,   w_test )["bias"]),
  `Bias%` = c(compute_metrics(y_train, pred_train,  w_train)["bias_pct"],
              compute_metrics(y_valid, pred_valid,   w_valid)["bias_pct"],
              compute_metrics(y_test,  pred_test,    w_test )["bias_pct"]),
  R2      = c(compute_metrics(y_train, pred_train,  w_train)["r2"],
              compute_metrics(y_valid, pred_valid,   w_valid)["r2"],
              compute_metrics(y_test,  pred_test,    w_test )["r2"])
)

# Side-by-side comparison 
print(as.data.frame(dplyr::bind_rows(metrics_cv, metrics_oof, metrics)), row.names = FALSE)
# >      Model    Dataset     RMSE      MAE        Bias     Bias%        R2
# >    CV-best   Training 49.87196 39.21827 -0.10321867 1.2985313 0.6951303
# >    CV-best Validation 65.78340 51.80086 -1.31198326 1.2972147 0.3860387
# >    CV-best       Test 64.96092 50.72562 -6.03279893 0.4825073 0.5234076
# >   OOF-best   Training 49.87196 39.21827 -0.10321867 1.2985313 0.6951303
# >   OOF-best Validation 65.78340 51.80086 -1.31198326 1.2972147 0.3860387
# >   OOF-best       Test 64.96092 50.72562 -6.03279893 0.4825073 0.5234076
# > VALID-best   Training 49.32077 38.75174 -0.06843546 1.2828857 0.7018320
# > VALID-best Validation 65.70228 51.74808 -1.38334543 1.2745293 0.3875520
# > VALID-best       Test 64.82269 50.64938 -5.95540185 0.4893086 0.5254336

# Summary 
message(sprintf(
  "Summary | CV-best iter = %d | OOF-best iter = %d | VALID-best iter = %d | n_cap (refit) = %d",
  best_oof_info$best_iter_by_cv, best_oof_info$best_iter_by_oof, best_iter, best_oof_info$row$nrounds
))
# > Summary | CV-best iter = 166 | OOF-best iter = 166 | VALID-best iter = 174 | n_cap (refit) = 200
message(sprintf(
  "OOF RMSE @ CV-best = %.5f | @ OOF-best = %.5f | @ VALID-best = %.5f",
  best_oof_info$oof_curve[best_oof_info$best_iter_by_cv], best_oof_info$oof_curve[best_oof_info$best_iter_by_oof], best_oof_info$oof_curve[best_iter]
))
# > OOF RMSE @ CV-best = 66.77939 | @ OOF-best = 66.77939 | @ VALID-best = 66.79064

## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH to all plausible values in mathematics. 

### ---- Fit main models using final weight (W_FSTUWT) ----

set.seed(123)

# Fit one XGBoost model per plausible value
tic("Fitting main models")
main_models <- lapply(pvmaths, function(pv) {
  
  X_train <- train_data[, voi_num]
  y_train <- train_data[[pv]]
  w_train <- train_data[[final_wt]]
  
  X_valid <- valid_data[, voi_num]
  y_valid <- valid_data[[pv]]
  w_valid <- valid_data[[final_wt]]
  
  dtrain <- xgb.DMatrix(
    data = as.matrix(X_train),
    label = y_train,
    weight = w_train
  )
  
  dvalid <- xgb.DMatrix(
    data = as.matrix(X_valid),
    label = y_valid,
    weight = w_valid
  )
  
  mod <- xgb.train(
    params = list(
      objective = "reg:squarederror",                  
      max_depth = best_oof_info$params$max_depth,               # best_oof_info$params$max_depth = 4
      eta = best_oof_info$params$eta,                           # best_oof_info$params$eta = 0.05
      eval_metric = "rmse",                            
      nthread = max(1, parallel::detectCores() - 1) 
    ),
    data = dtrain,
    nrounds = best_oof_info$best_iter_by_oof,                   # best_oof_info$best_iter_by_oof = 166
    watchlist = list(train = dtrain, eval = dvalid),
    verbose = 1,                               
    early_stopping_rounds=NULL               
  )
  
  list(
    mod = mod,
    formula = as.formula(paste(pv, "~", paste(voi_num, collapse = " + "))),
    importance = xgb.importance(model = mod)
  )
})
toc()
# > Fitting main models: 28.861 sec elapsed

main_models[[1]]             # Inspect first model: mod object, r2, and importance
main_models[[1]]$formula
main_models[[1]]$importance %>% head() %>% print(row.names=FALSE)
# >  Feature       Gain      Cover  Frequency
# >   <char>      <num>      <num>      <num>
# >  MATHEFF 0.41831806 0.09058278 0.04665314
# >     ESCS 0.05705440 0.04766530 0.04421907
# > FAMSUPSL 0.04921920 0.04812288 0.04665314
# >   ANXMAT 0.04237571 0.03737265 0.03083164
# >   FAMCON 0.02746064 0.04394888 0.03286004
# > MATHEF21 0.02469909 0.03831558 0.02555781
main_models[[1]]$importance$Feature
main_models[[2]]$importance$Feature

main_models[[1]]$mod$evaluation_log
which.min(main_models[[1]]$mod$evaluation_log$eval_rmse)
# > [1] 166
min(main_models[[1]]$mod$evaluation_log$eval_rmse)
# > 65.78341
which.min(main_models[[2]]$mod$evaluation_log$eval_rmse)
# > [1] 163
min(main_models[[2]]$mod$evaluation_log$eval_rmse)
# > 68.07645
which.min(main_models[[3]]$mod$evaluation_log$eval_rmse)
# > [1] 166
min(main_models[[3]]$mod$evaluation_log$eval_rmse)
# > 65.51495
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
# > # A tibble: 10 × 3
# >  pv       best_nround best_eval_rmse
# >  <chr>          <int>          <dbl>
# >1 PV1MATH          166           65.8
# >2 PV2MATH          163           68.1
# >3 PV3MATH          166           65.5
# >4 PV4MATH          164           67.1
# >5 PV5MATH          161           66.4
# >6 PV6MATH          166           67.9
# >7 PV7MATH          152           69.1
# >8 PV8MATH          164           66.1
# >9 PV9MATH          140           67.8
# >10 PV10MATH        128           70.3

# # --- Variable Importance ---
# 
# # Gain-based Variable Importance Matrix (10 PVs × p predictors) 
# main_importance_matrix <- do.call(rbind, lapply(main_models, function(m) {
#   # Extract Gain column from xgb.importance()
#   main_importance <- m$importance
#   setNames(main_importance$Gain, main_importance$Feature)[voi_num]  # Ensure correct column order and all variables present
# }))
# dim(main_importance_matrix)  # 10x53
# main_importance_matrix[, 1: 6]
# # >            MATHMOT     MATHEASE     MATHPREF   EXERPRAC     STUDYHMW    WORKPAY
# # >  [1,] 7.658559e-05 3.600987e-07 1.995973e-04 0.01715437 0.0009945422 0.02185493
# # >  [2,] 1.449202e-05 8.629997e-06 4.276851e-04 0.02003683 0.0010911087 0.02425707
# # >  [3,] 9.830436e-05 6.948579e-05 1.863857e-04 0.01495500 0.0010861422 0.03204153
# # >  [4,] 4.657499e-04 9.386579e-06 1.042503e-04 0.02100573 0.0017915209 0.02984059
# # >  [5,]           NA 2.131156e-05 1.904485e-04 0.02002977 0.0011670372 0.02780895
# # >  [6,] 2.208581e-06 2.895649e-04 4.621869e-05 0.01706586 0.0004597595 0.02559769
# # >  [7,] 9.391622e-05 3.152710e-06 3.758198e-04 0.02073120 0.0023527137 0.02764404
# # >  [8,] 1.314410e-04 3.012300e-04 4.593325e-04 0.01829539 0.0008006057 0.03141363
# # >  [9,]           NA 3.498187e-04 7.794546e-06 0.02088210 0.0003854466 0.02491970
# # > [10,] 2.210727e-05 1.858645e-04 2.177621e-04 0.01923594 0.0006597048 0.03329804
# 
# # Mean of gain importance across PVs
# main_importance <- colMeans(main_importance_matrix)
# main_importance[1: 6]
# # > MATHMOT     MATHEASE     MATHPREF     EXERPRAC     STUDYHMW      WORKPAY 
# # >      NA 0.0001238805 0.0002215294 0.0189392173 0.0010788582 0.0278676176 
# # main_importance <- colMeans(main_importance_matrix, na.rm=TRUE)
# # main_importance[1: 6]
# # # >      MATHMOT     MATHEASE     MATHPREF     EXERPRAC     STUDYHMW      WORKPAY 
# # # > 0.0001131006 0.0001238805 0.0002215294 0.0189392173 0.0010788582 0.0278676176

# --- Variable Importance (main): M × p, fill NA with 0 ---
main_importance_matrix <- do.call(rbind, lapply(main_models, function(m) {
  gain_vec <- setNames(m$importance$Gain, m$importance$Feature)[voi_num]
  gain_vec[is.na(gain_vec)] <- 0
  gain_vec
}))
dimnames(main_importance_matrix) <- list(pvmaths, voi_num)

# quick checks
stopifnot(all(dim(main_importance_matrix) == c(M, length(voi_num))))
stopifnot(!anyNA(main_importance_matrix))
if (any(abs(rowSums(main_importance_matrix) - 1) > 1e-6)) warning("Row sums not ~1.")  # each PV's gains should sum ~ 1

# Mean of gain importance across PVs
main_importance <- colMeans(main_importance_matrix)
main_importance[1: 6]
# >      MATHMOT     MATHEASE     MATHPREF     EXERPRAC     STUDYHMW      WORKPAY 
# > 0.0000904805 0.0001238805 0.0002215294 0.0189392173 0.0010788582 0.0278676176 

# Display ranked importance
tibble(
  Variable = names(main_importance),
  Importance = main_importance
) |> 
  arrange(desc(Importance)) |> 
  head() 
# > # A tibble: 4 × 2
# > Variable Importance
# > <chr>         <dbl>
# > 1 MATHEFF      0.423 
# > 2 ESCS         0.0588
# > 3 FAMSUPSL     0.0504
# > 4 ANXMAT       0.0446
# > 5 FAMCON       0.0288
# > 6 WORKPAY      0.0279

# --- Estimates of Manual weighted R² (XGBoost models on training data) ---

main_r2s_weighted <- sapply(1:M, function(i) {
  pv <- pvmaths[i]
  model <- main_models[[i]]$mod
  
  # Extract weighted true values and predictions on training data
  y_true <- train_data[[pv]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, voi_num]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  
  y_pred <- predict(model, dtrain)  # prediction from fitted xgb model
  
  # Weighted mean and sums
  y_bar <- sum(w * y_true) / sum(w)
  sse <- sum(w * (y_true - y_pred)^2)
  sst <- sum(w * (y_true - y_bar)^2)
  
  # Weighted R²
  r2 <- 1 - sse / sst
  return(r2)
})

main_r2s_weighted
# > [1] 0.6951303 0.6961154 0.6962064 0.6960417 0.6948999 0.7033021 0.6965448 0.7011998 0.6993118 0.7038768

# Final Rubin's Step 2: mean of R² across plausible values
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.6982629

# --- Use helper function for all five metrics ---
main_metrics <- sapply(1:M, function(i) {
  pv <- pvmaths[i]
  model <- main_models[[i]]$mod
  
  y_true <- train_data[[pv]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, voi_num]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  y_pred <- predict(model, dtrain)
  
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
main_metrics
# >         mse     rmse      mae      bias bias_pct        r2
# > 1  2487.213 49.87196 39.21827 -0.1032187 1.298531 0.6951303
# > 2  2427.633 49.27102 38.83615 -0.1035517 1.263172 0.6961154
# > 3  2467.884 49.67780 38.94456 -0.1039515 1.297609 0.6962064
# > 4  2409.015 49.08172 38.60388 -0.1036017 1.242688 0.6960417
# > 5  2409.341 49.08504 38.48143 -0.1038262 1.236003 0.6948999
# > 6  2346.565 48.44136 38.09013 -0.1035184 1.231488 0.7033021
# > 7  2379.514 48.78026 38.44469 -0.1032316 1.242507 0.6965448
# > 8  2415.591 49.14866 38.32167 -0.1035625 1.271779 0.7011998
# > 9  2384.916 48.83561 38.26823 -0.1032805 1.242896 0.6993118
# > 10 2330.688 48.27720 38.07287 -0.1036052 1.197837 0.7038768

### ---- Replicate models using BRR replicate weights (XGBoost, fixed hyperparameters) ----

set.seed(123)

tic("Fitting replicate models")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    X_train <- train_data[, voi_num]
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    X_valid <- valid_data[, voi_num]
    y_valid <- valid_data[[pv]]
    w_valid <- valid_data[[w]]
    
    dtrain <- xgb.DMatrix(
      data = as.matrix(X_train),
      label = y_train,
      weight = w_train
    )
    
    dvalid <- xgb.DMatrix(
      data = as.matrix(X_valid),
      label = y_valid,
      weight = w_valid
    )
    
    mod <- xgb.train(
      params = list(
        objective = "reg:squarederror",
        max_depth = best_oof_info$params$max_depth,             # best_oof_info$params$max_depth = 4
        eta = best_oof_info$params$eta,                         # best_oof_info$params$eta = 0.05
        eval_metric = "rmse",
        nthread = max(1, parallel::detectCores() - 1)
      ),
      data = dtrain,
      nrounds = best_oof_info$best_iter_by_oof,                 # best_oof_info$best_iter_by_oof = 50   
      watchlist = list(train = dtrain, eval = dvalid),
      verbose = 1,
      early_stopping_rounds = NULL
    )
    
    list(
      mod = mod,
      formula = as.formula(paste(pv, "~", paste(voi_num, collapse = " + "))),
      importance = xgb.importance(model = mod)
    )
  })
})
toc()
# > Fitting replicate models: 2137.261 sec elapsed

replicate_models[[1]][[1]]$mod        
replicate_models[[1]][[1]]$formula    
replicate_models[[1]][[1]]$importance 

# # --- Gain-Based Variable Importance Matrix for Replicates: (M × G × p) ---
# rep_importance_array <- array(NA, dim = c(M, G, length(voi_num)),
#                               dimnames = list(NULL, NULL, voi_num))
# 
# for (m in 1:M) {
#   for (g in 1:G) {
#     imp <- replicate_models[[m]][[g]]$importance
#     rep_importance_array[m, g, ] <- setNames(imp$Gain, imp$Feature)[voi_num]
#   }
# }
# 
# # Check structure
# dim(rep_importance_array)  # 10 x 80 x 53
# rep_importance_array[1, 1, ]  # e.g., PV1, BRR replicate 1

# --- Gain-Based Variable Importance Matrix for Replicates: (M × G × p) ---
rep_importance_array <- array(NA, dim = c(M, G, length(voi_num)),
                              dimnames = list(NULL, NULL, voi_num))

for (m in 1:M) {
  for (g in 1:G) {
    imp <- replicate_models[[m]][[g]]$importance
    gain_vec <- setNames(imp$Gain, imp$Feature)[voi_num]
    gain_vec[is.na(gain_vec)] <- 0
    rep_importance_array[m, g, ] <- gain_vec
  }
}

# quick checks
stopifnot(all(dim(rep_importance_array) == c(M, G, length(voi_num))))
stopifnot(!anyNA(rep_importance_array))
# optional: each (PV, replicate) slice should sum ~ 1
if (any(abs(apply(rep_importance_array, c(1, 2), sum) - 1) > 1e-6)) {
  warning("Some (PV, replicate) gain rows do not sum to ~1.")
}

# Check structure
dim(rep_importance_array)  # 10 x 80 x 53
rep_importance_array[1, 1, ]  # e.g., PV1, BRR replicate 1

# --- Weighted R² across (M × G) ---
rep_r2_weighted  <- matrix(NA, nrow = G, ncol = M)

for (m in 1:M) {
  pv <- pvmaths[m]
  y_true <- train_data[[pv]]
  
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w <- train_data[[rep_wts[g]]]  # Replicate weight g
    
    X_train <- train_data[, voi_num]
    dtrain <- xgb.DMatrix(data = as.matrix(X_train))
    
    y_pred <- predict(model, dtrain)
    
    y_bar <- sum(w * y_true) / sum(w)
    sse <- sum(w * (y_true - y_pred)^2)
    sst <- sum(w * (y_true - y_bar)^2)
    
    rep_r2_weighted [g, m] <- 1 - sse / sst
  }
}

# Check output
dim(rep_r2_weighted )       # 80 x 10
rep_r2_weighted 

### ---- Rubin + BRR for Standard Errors (SEs) ----

# --- Rubin + BRR: Gain-Based Variable Importance ---
sampling_var_importance   <- setNames(numeric(length(voi_num)), voi_num)
imputation_var_importance <- setNames(numeric(length(voi_num)), voi_num)
var_final_importance      <- setNames(numeric(length(voi_num)), voi_num)
se_final_importance       <- setNames(numeric(length(voi_num)), voi_num)
cv_final_importance       <- setNames(numeric(length(voi_num)), voi_num)

for (var in voi_num) {
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
# > [1] 2.401175e-05

imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
imputation_var_r2_weighted
# > [1] 1.160932e-05

var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
var_final_r2_weighted
# > [1] 3.678201e-05

se_final_r2_weighted <- sqrt(var_final_r2_weighted)
se_final_r2_weighted
# > [1] 0.006064817

#### ---- Final Output Tables ----

# --- Variable Importance Table (mean ± SE) ---
importance_table <- tibble(
  Variable = voi_num,  # e.g., "EXERPRAC", ...
  Importance = main_importance[voi_num],
  `Std. Error` = se_final_importance[voi_num],
  `CV` = cv_final_importance[voi_num]
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
# >               Metric  Estimate  Std. Error
# > R-squared (Weighted) 0.6982629 0.006064817

as.data.frame(importance_table) %>% head() %>% print(row.names = FALSE)
# > Variable Importance  Std. Error         CV
# >  MATHEFF 0.42277849 0.010850673 0.02566515
# >     ESCS 0.05875985 0.005138198 0.08744403
# > FAMSUPSL 0.05039168 0.002756465 0.05470081
# >   ANXMAT 0.04459412 0.004852402 0.10881260
# >   FAMCON 0.02882975 0.004043838 0.14026615
# >  WORKPAY 0.02786762 0.004019501 0.14423556

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
  y_true <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, voi_num]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  y_pred <- predict(model, dtrain)
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()

train_metrics_main
# >         mse     rmse      mae       bias bias_pct        r2
# > 1  2487.213 49.87196 39.21827 -0.1032187 1.298531 0.6951303
# > 2  2427.633 49.27102 38.83615 -0.1035517 1.263172 0.6961154
# > 3  2467.884 49.67780 38.94456 -0.1039515 1.297609 0.6962064
# > 4  2409.015 49.08172 38.60388 -0.1036017 1.242688 0.6960417
# > 5  2409.341 49.08504 38.48143 -0.1038262 1.236003 0.6948999
# > 6  2346.565 48.44136 38.09013 -0.1035184 1.231488 0.7033021
# > 7  2379.514 48.78026 38.44469 -0.1032316 1.242507 0.6965448
# > 8  2415.591 49.14866 38.32167 -0.1035625 1.271779 0.7011998
# > 9  2384.916 48.83561 38.26823 -0.1032805 1.242896 0.6993118
# > 10 2330.688 48.27720 38.07287 -0.1036052 1.197837 0.7038768
train_metrics_main$r2   # = main_r2s_weighted
# > [1] 0.6951303 0.6961154 0.6962064 0.6960417 0.6948999 0.7033021 0.6965448 0.7011998 0.6993118 0.7038768

train_metric_main <- colMeans(train_metrics_main)
train_metric_main 
# >          mse         rmse          mae         bias     bias_pct           r2
# > 2405.8361044   49.0470633   38.5281874   -0.1035348    1.2524510    0.6982629
train_metric_main["r2"]  # = main_r2_weighted
# >        r2 
# > 0.6982629

# --- Replicate predictions for training data ---
tic("Computing train_metrics_replicates")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_true <- train_data[[pvmaths[m]]]
    w <- train_data[[rep_wts[g]]]
    X_train <- train_data[, voi_num]
    dtrain <- xgb.DMatrix(data = as.matrix(X_train))
    y_pred <- predict(model, dtrain)
    compute_metrics(y_true, y_pred, w)
  }) |> t()
}) # a list of M=10 matrices, each of shape 80x5
toc()
# > Computing train_metrics_replicates: 7.66 sec elapsed
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
# > mse      1.790886e+03 1.586521e+03 1.647497e+03 1.899662e+03 1.510202e+03 1.260408e+03 1.222245e+03 2.079576e+03 1.320521e+03 1.302919e+03
# > rmse     1.867152e-01 1.692697e-01 1.729416e-01 2.049192e-01 1.623038e-01 1.387350e-01 1.326188e-01 2.240452e-01 1.431418e-01 1.445707e-01
# > mae      1.801720e-01 1.802483e-01 1.504353e-01 1.869650e-01 1.594858e-01 1.558026e-01 1.339613e-01 1.749261e-01 1.448963e-01 1.354861e-01
# > bias     2.257217e-08 1.099087e-08 9.487222e-09 8.059113e-09 7.389409e-09 1.299955e-08 1.029794e-08 9.984477e-09 2.209977e-08 8.283134e-09
# > bias_pct 2.299795e-04 1.977599e-04 2.681534e-04 2.749457e-04 1.718410e-04 1.966619e-04 1.923026e-04 3.055297e-04 1.731331e-04 1.564933e-04
# > r2       2.631137e-05 2.487412e-05 2.412654e-05 3.003006e-05 2.333965e-05 1.939238e-05 1.943825e-05 3.147892e-05 2.037502e-05 2.075119e-05

# <=> Equivalent codes
# sampling_var_matrix_train <- sapply(1:M, function(m) {
#   sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
#     colMeans() / (1 - k)^2
# })
# sampling_var_matrix_train

# Debugged and check consistency
sampling_var_matrix_train["r2", ]
# > [1] 2.631137e-05 2.487412e-05 2.412654e-05 3.003006e-05 2.333965e-05 1.939238e-05 1.943825e-05 3.147892e-05 2.037502e-05 2.075119e-05
sampling_var_r2s_weighted
# > [1] 2.631137e-05 2.487412e-05 2.412654e-05 3.003006e-05 2.333965e-05 1.939238e-05 1.943825e-05 3.147892e-05 2.037502e-05 2.075119e-05
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
sampling_var_r2_weighted
# > [1] 2.401175e-05

sampling_var_train <- rowMeans(sampling_var_matrix_train)
sampling_var_train                                           # sampling_var_train['r2'] = sampling_var_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 1.562044e+03 1.679261e-01 1.602379e-01 1.221637e-08 2.166800e-04 2.401175e-05  

imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
imputation_var_train                                              # imputation_var_train ['r2'] = imputation_var_r2_weighted 
# >         mse         rmse          mae         bias     bias_pct           r2 
# > 2.372832e+03 2.463193e-01 1.409707e-01 5.843621e-08 9.588120e-04 1.160932e-05 

var_final_train <- sampling_var_train + (1 + 1/M) * imputation_var_train
var_final_train                                              # var_final_train ['r2'] = var_final_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 4.172159e+03 4.388774e-01 3.153057e-01 7.649620e-08 1.271373e-03 3.678201e-05 

se_final_train <- sqrt(var_final_train)
se_final_train                                         # se_final_train ['r2'] = se_final_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 6.459225e+01 6.624782e-01 5.615209e-01 2.765795e-04 3.565632e-02 6.064817e-03 

# Confidence intervals
ci_lower <- train_metric_main - z_crit * se_final_train
ci_upper <- train_metric_main + z_crit * se_final_train
ci_length <- ci_upper - ci_lower

train_eval <- tibble(
  Metric = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train, scientific = FALSE),
  CI_lower = format(ci_lower, scientific = FALSE),
  CI_upper = format(ci_upper, scientific = FALSE),
  CI_length = format(ci_length, scientific = FALSE)
)

print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   2405.8361044  64.5922481321 2279.2376244 2532.4345844 253.196960039
# >      RMSE     49.0470633   0.6624781986   47.7486299   50.3454967   2.596866820
# >       MAE     38.5281874   0.5615208619   37.4276267   39.6287480   2.201121332
# >      Bias     -0.1035348   0.0002765795   -0.1040769   -0.1029927   0.001084172
# >     Bias%      1.2524510   0.0356563213    1.1825659    1.3223362   0.139770211
# > R-squared      0.6982629   0.0060648170    0.6863761    0.7101497   0.023773646

### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_true <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- valid_data[, voi_num]
  dvalid <- xgb.DMatrix(data = as.matrix(X_valid))
  y_pred <- predict(model, dvalid)
  compute_metrics(y_true, y_pred, w)
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
    y_true <- valid_data[[pvmaths[m]]]
    w <- valid_data[[rep_wts[g]]]
    X_valid <- valid_data[, voi_num]
    dvalid <- xgb.DMatrix(data = as.matrix(X_valid))
    y_pred <- predict(model, dvalid)
    compute_metrics(y_true, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates: 1.573 sec elapsed

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
  CI_length = format(ci_length_valid, scientific = FALSE)
)

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper   CI_length
# >       MSE   4552.0111860   230.37038546 4100.4935274 5003.5288446 903.0353172
# >      RMSE     67.4518216     1.69730089   64.1251730   70.7784702   6.6532972
# >       MAE     53.6255437     1.48682431   50.7114216   56.5396658   5.8282442
# >      Bias     -0.5667035     1.54751833   -3.5997837    2.4663766   6.0661604
# >     Bias%      1.5941289     0.34469121    0.9185466    2.2697113   1.3511647
# > R-squared      0.3865938     0.01886389    0.3496213    0.4235664   0.0739451

### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_true <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- test_data[, voi_num]
  dtest <- xgb.DMatrix(data = as.matrix(X_test))
  y_pred <- predict(model, dtest)
  compute_metrics(y_true, y_pred, w)
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
    y_true <- test_data[[pvmaths[m]]]
    w <- test_data[[rep_wts[g]]]
    X_test <- test_data[, voi_num]
    dtest <- xgb.DMatrix(data = as.matrix(X_test))
    y_pred <- predict(model, dtest)
    compute_metrics(y_true, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates: 2.144 sec elapsed

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
# >     Metric  Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >        MSE    4264.7384866   253.85216714 3767.1973816 4762.2795916 995.08220998
# >       RMSE      65.2821528     1.94075384   61.4783452   69.0859604   7.60761525
# >        MAE      51.6731310     1.70089976   48.3394287   55.0068332   6.66740456
# >       Bias      -4.6941319     1.60982595   -7.8493327   -1.5389310   6.31040175
# >      Bias%       0.8021684     0.29130170    0.2312275    1.3731092   1.14188167
# >  R-squared       0.5142798     0.01592205    0.4830731    0.5454864   0.06241329

### ---- ** Predictive Performance on Training/Validation/Test Data (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process to avoid redundancy.

# Evaluation function 
evaluate_split <- function(split_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, voi_num, pvmaths) {
  
  # Main plausible values loop
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    y_true <- split_data[[pvmaths[i]]]
    w <- split_data[[final_wt]]
    features <- split_data[, voi_num]
    dmat <- xgb.DMatrix(data = as.matrix(features))
    y_pred <- predict(model, dmat)
    compute_metrics(y_true, y_pred, w)
  }) |> t() |> as.data.frame()
  
  main_point <- colMeans(main_metrics_df)
  
  # Replicate loop
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      y_true <- split_data[[pvmaths[m]]]
      w <- split_data[[rep_wts[g]]]
      features <- split_data[, voi_num]
      dmat <- xgb.DMatrix(data = as.matrix(features))
      y_pred <- predict(model, dmat)
      compute_metrics(y_true, y_pred, w)
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
train_eval <- evaluate_split(train_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, voi_num, pvmaths)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, voi_num, pvmaths)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, voi_num, pvmaths)

# Display
print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

### ---- Summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   2405.8361044  64.5922481321 2279.2376244 2532.4345844 253.196960039
# >      RMSE     49.0470633   0.6624781986   47.7486299   50.3454967   2.596866820
# >       MAE     38.5281874   0.5615208619   37.4276267   39.6287480   2.201121332
# >      Bias     -0.1035348   0.0002765795   -0.1040769   -0.1029927   0.001084172
# >     Bias%      1.2524510   0.0356563213    1.1825659    1.3223362   0.139770211
# > R-squared      0.6982629   0.0060648170    0.6863761    0.7101497   0.023773646

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper   CI_length
# >       MSE   4552.0111860   230.37038546 4100.4935274 5003.5288446 903.0353172
# >      RMSE     67.4518216     1.69730089   64.1251730   70.7784702   6.6532972
# >       MAE     53.6255437     1.48682431   50.7114216   56.5396658   5.8282442
# >      Bias     -0.5667035     1.54751833   -3.5997837    2.4663766   6.0661604
# >     Bias%      1.5941289     0.34469121    0.9185466    2.2697113   1.3511647
# > R-squared      0.3865938     0.01886389    0.3496213    0.4235664   0.0739451

print(as.data.frame(test_eval), row.names = FALSE)
# >     Metric  Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >        MSE    4264.7384866   253.85216714 3767.1973816 4762.2795916 995.08220998
# >       RMSE      65.2821528     1.94075384   61.4783452   69.0859604   7.60761525
# >        MAE      51.6731310     1.70089976   48.3394287   55.0068332   6.66740456
# >       Bias      -4.6941319     1.60982595   -7.8493327   -1.5389310   6.31040175
# >      Bias%       0.8021684     0.29130170    0.2312275    1.3731092   1.14188167
# >  R-squared       0.5142798     0.01592205    0.4830731    0.5454864   0.06241329
