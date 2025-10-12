# ---- II. Predictive Modelling: Version 2.4 ----

# Tune hyperparameters (alpha α and lambda λ) using cv.glmnet, with manual K-folds + tracks out-of-fold (OOF) predictions/metrics.

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
set.seed(123)          # Ensure reproducibility
n <- nrow(temp_data)   # 20003
indices <- sample(n)   # Randomly shuffle row indices

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

## ---- 2. PV1MATH only ----

# --- Remark ---
# 1) Repeat the same process for PV2MATH - PV10MATH.
# 2) Apply best results from PV1MATH to all plausible values in mathematics. 

## ---- Main model using final student weights (W_FSTUWT) ---- 

### ---- Tuning Elastic Net for PV1MATH only: cv.glmnet ----

# Target plausible value
pv1math <- pvmaths[1]

# Training data
X_train <- as.matrix(train_data[, voi_num])
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

# Validation data 
X_valid <- as.matrix(valid_data[, voi_num])
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

# Test data 
X_test <- as.matrix(test_data[, voi_num])
y_test <- test_data[[pv1math]]
w_test <- test_data[[final_wt]]

# --- Fixed Manual CV folds on TRAIN (reproducible) ---
set.seed(123)
num_folds <- 5L
n_cv      <- nrow(X_train)
stopifnot(n_cv >= num_folds)                                                             # basic guard
cv_order  <- sample.int(n_cv)                                                            # random permutation
bounds    <- floor(seq(0, n_cv, length.out = num_folds + 1))
stopifnot(all(diff(bounds) > 0))                                                         # no empty folds
cv_folds  <- vector("list", num_folds)
for (k in seq_len(num_folds)) cv_folds[[k]] <- cv_order[(bounds[k] + 1):bounds[k + 1]]
stopifnot(identical(sort(unlist(cv_folds)), seq_len(n_cv)))

# Convert cv_folds (list of indices) -> foldid (vector 1..K for each row)
foldid <- integer(n_cv)
for (k in seq_along(cv_folds)) foldid[cv_folds[[k]]] <- k
stopifnot(all(foldid %in% seq_len(num_folds)), length(foldid) == n_cv)

# Weight balance across folds
tapply(w_train, foldid, sum)
# >        1        2        3        4        5 
# > 17147.27 17871.52 18282.63 17925.20 17769.03 
table(foldid)
# > foldid
# >    1    2    3    4    5 
# > 1111 1111 1111 1111 1111 

# Further check CV-fold weight balance 
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

# Outcome distribution by fold
print(do.call(rbind, tapply(y_train, foldid, function(v) c(mean = mean(v), sd = sd(v)))))
# >       mean       sd
# > 1 507.7791 91.30825
# > 2 508.0027 88.45793
# > 3 509.6627 92.20887
# > 4 508.5530 88.68829
# > 5 510.3219 89.14069

#### ---- 1) Tuning: grouped = TRUE (Default), keep = TRUE ----

# Grid over elastic‑net mixing parameter
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso
# options: alpha_grid <- seq(0, 1, by = 0.05), alpha_grid <- sort(unique(c(seq(0, 1, by = 0.1), 0.001, 0.005, 0.01, 0.05))), etc. 

# Storage
cv_list        <- vector("list", length(alpha_grid))  # cv.glmnet fits per alpha
per_alpha_list <- vector("list", length(alpha_grid))  # two rows per alpha: lambda.min & lambda.1se

# Parallel backend (for cv.glmnet parallelization)
registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Grid over alpha (cv.glmnet, manual 5-fold)")
for (i in seq_along(alpha_grid)) {
  alpha <- alpha_grid[i]
  message(sprintf("Fitting cv.glmnet for alpha = %.1f (manual 5-fold CV on TRAIN)", alpha))
  
  # Fix fold randomness 
  set.seed(123)
  
  cvmod <- cv.glmnet(
    x = X_train,
    y = y_train,
    weights = w_train,
    type.measure = "mse", 
    foldid = foldid,           # <-
    grouped = TRUE,            # <-                              
    keep = TRUE,               # <-                               
    parallel = TRUE,          
    trace.it = 0,
    alpha = alpha,
    family = "gaussian",
    standardize = TRUE,
    intercept = TRUE
  )
  cv_list[[i]] <- cvmod
  
  # Indices
  idx_min <- cvmod$index["min", 1]
  idx_1se <- cvmod$index["1se", 1]
  
  # Lambdas and path metadata at those indices
  lambda_min <- cvmod$lambda[idx_min]      # == cvmod$lambda.min
  lambda_1se <- cvmod$lambda[idx_1se]      # == cvmod$lambda.1se
  
  nzero_min  <- cvmod$nzero[idx_min]
  nzero_1se  <- cvmod$nzero[idx_1se]
  
  dev_min    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_min])
  dev_1se    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_1se])
  
  # CV summaries (in-built)
  cvm_min    <- as.numeric(cvmod$cvm[idx_min])
  cvsd_min   <- as.numeric(cvmod$cvsd[idx_min])
  cvup_min   <- as.numeric(cvmod$cvup[idx_min])
  cvlo_min   <- as.numeric(cvmod$cvlo[idx_min])
  
  cvm_1se    <- as.numeric(cvmod$cvm[idx_1se])
  cvsd_1se   <- as.numeric(cvmod$cvsd[idx_1se])
  cvup_1se   <- as.numeric(cvmod$cvup[idx_1se])
  cvlo_1se   <- as.numeric(cvmod$cvlo[idx_1se])
  
  # Two rows per α: one for lambda.min, one for lambda.1se
  per_alpha_list[[i]] <- tibble::tibble(
    alpha       = alpha,
    alpha_idx   = i,
    s           = c("lambda.min", "lambda.1se"),
    lambda      = c(lambda_min,   lambda_1se),
    lambda_idx  = c(idx_min,      idx_1se),
    nzero       = c(nzero_min,    nzero_1se),
    dev_ratio   = c(dev_min,      dev_1se),
    cvm         = c(cvm_min,      cvm_1se),
    cvsd        = c(cvsd_min,     cvsd_1se),
    cvlo        = c(cvlo_min,     cvlo_1se),
    cvup        = c(cvup_min,     cvup_1se)
  )
}
toc()
# > Grid over alpha (cv.glmnet, manual 5-fold): 2.32 sec elapsed

##### ---- Explore the first tuning results ----
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by CV MSE (lower is better) irrespective of rule
tuning_results %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  #head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx          s     lambda lambda_idx nzero dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.6         7 lambda.min  1.1140300         47    42 0.4352170 4764.684 89.56346 4675.121 4854.248
# >   0.7         8 lambda.min  0.9548829         47    41 0.4352598 4764.699 89.75413 4674.945 4854.453
# >   0.8         9 lambda.min  0.8355225         47    41 0.4352926 4764.704 89.89214 4674.812 4854.597
# >   0.9        10 lambda.min  0.7426867         47    41 0.4353175 4764.717 89.99991 4674.717 4854.717
# >   0.5         6 lambda.min  1.3368360         47    42 0.4351530 4764.718 89.31297 4675.405 4854.031
# >   1.0        11 lambda.min  0.6684180         47    41 0.4353370 4764.722 90.08862 4674.633 4854.811
# >   0.4         5 lambda.min  1.6710450         47    42 0.4350516 4764.817 88.96074 4675.856 4853.777
# >   0.3         4 lambda.min  2.0301253         48    42 0.4353499 4765.005 89.12933 4675.876 4854.134
# >   0.2         3 lambda.min  2.7746619         49    42 0.4354522 4765.456 88.87110 4676.584 4854.327
# >   0.1         2 lambda.min  4.6071461         51    45 0.4353364 4767.671 87.95128 4679.720 4855.622
# >   0.0         1 lambda.min  5.8135590         98    53 0.4371044 4779.595 91.45740 4688.137 4871.052
# >   0.5         6 lambda.1se  4.0825041         35    29 0.4179845 4840.249 83.02700 4757.222 4923.276
# >   0.4         5 lambda.1se  5.1031301         35    29 0.4175541 4843.104 83.19232 4759.911 4926.296
# >   0.2         3 lambda.1se  9.2995642         36    34 0.4177705 4845.776 84.17664 4761.600 4929.953
# >   0.3         4 lambda.1se  6.8041735         35    30 0.4167324 4848.758 83.45499 4765.303 4932.213
# >   1.0        11 lambda.1se  2.2402716         34    27 0.4161316 4850.120 82.62264 4767.497 4932.742
# >   0.9        10 lambda.1se  2.4891907         34    27 0.4160519 4850.511 82.69733 4767.814 4933.209
# >   0.8         9 lambda.1se  2.8003395         34    27 0.4159696 4851.066 82.77614 4768.290 4933.842
# >   0.1         2 lambda.1se 15.4413231         38    36 0.4182343 4851.306 84.92019 4766.386 4936.226
# >   0.7         8 lambda.1se  3.2003880         34    27 0.4158439 4851.848 82.88303 4768.965 4934.731
# >   0.6         7 lambda.1se  3.7337860         34    27 0.4156547 4853.002 83.01470 4769.987 4936.016
# >   0.0         1 lambda.1se 34.0501362         79    53 0.4207341 4862.424 80.95175 4781.472 4943.376

# Choose winners under each rule separately 
best_min  <- tuning_results %>%
  filter(s == "lambda.min") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%   # tie-breaks: fewer nonzeros (nzero), larger λ, smaller α
  slice(1)

best_1se  <- tuning_results %>%
  filter(s == "lambda.1se") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  slice(1)

print(as.data.frame(best_min),  row.names = FALSE)
# > alpha alpha_idx          s  lambda lambda_idx nzero dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.6         7 lambda.min 1.11403         47    42  0.435217 4764.684 89.56346 4675.121 4854.248
print(as.data.frame(best_1se), row.names = FALSE)
# > alpha alpha_idx          s   lambda lambda_idx nzero dev_ratio      cvm   cvsd     cvlo     cvup
# >   0.5         6 lambda.1se 4.082504         35    29 0.4179845 4840.249 83.027 4757.222 4923.276

best_alpha_min     <- best_min$alpha
best_lambda_min    <- best_min$lambda
best_alpha_idx_min <- best_min$alpha_idx

best_alpha_1se     <- best_1se$alpha
best_lambda_1se    <- best_1se$lambda
best_alpha_idx_1se <- best_1se$alpha_idx

message(sprintf("Winner @ lambda.min : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_min, best_lambda_min, best_min$nzero, best_min$cvm, best_min$cvsd))
# > Winner @ lambda.min : alpha = 0.60 | lambda = 1.114030 | nzero = 42 | CVM = 4764.68406 (± 89.56346)
message(sprintf("Winner @ lambda.1se : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_1se, best_lambda_1se, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
# > Winner @ lambda.1se : alpha = 0.50 | lambda = 4.082504 | nzero = 29 | CVM = 4840.24927 (± 83.02700)

##### ---- Predict & evaluate on TRAIN / VALID / TEST for both winners ----
cv_best_min  <- cv_list[[best_alpha_idx_min]]
cv_best_min
cv_best_1se  <- cv_list[[best_alpha_idx_1se]]
cv_best_1se

# Coefficients & importances
coef(cv_best_min,  s = best_lambda_min) %>% head()
# > 6 x 1 sparse Matrix of class "dgCMatrix"
# >               s=1.11403
# > (Intercept) 517.5748683
# > MATHMOT       .        
# > MATHEASE      2.7200581
# > MATHPREF      2.2764623
# > EXERPRAC     -2.6604818
# > STUDYHMW     -0.1180583
coef(cv_best_1se,  s = best_lambda_1se) %>% head()
# > 6 x 1 sparse Matrix of class "dgCMatrix"
# >              s=4.082504
# > (Intercept) 520.1301314
# > MATHMOT       .        
# > MATHEASE      0.8163205
# > MATHPREF      0.7600454
# > EXERPRAC     -2.2009855
# > STUDYHMW      .        

varImp(cv_best_min$glmnet.fit,  lambda = best_lambda_min) %>% head()
# >            Overall
# > MATHMOT  0.0000000
# > MATHEASE 2.7200581
# > MATHPREF 2.2764623
# > EXERPRAC 2.6604818
# > STUDYHMW 0.1180583
# > WORKPAY  3.0259446
varImp(cv_best_1se$glmnet.fit,  lambda = best_lambda_1se) %>% head()
# >            Overall
# > MATHMOT  0.0000000
# > MATHEASE 0.8163205
# > MATHPREF 0.7600454
# > EXERPRAC 2.2009855
# > STUDYHMW 0.0000000
# > WORKPAY  2.8624211

# Predictions
pred_train_min <- as.numeric(predict(cv_best_min, newx = X_train, s = best_lambda_min))
pred_valid_min <- as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min))
pred_test_min  <- as.numeric(predict(cv_best_min,  newx = X_test,  s = best_lambda_min))

pred_train_1se <- as.numeric(predict(cv_best_1se, newx = X_train, s = best_lambda_1se))
pred_valid_1se <- as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se))
pred_test_1se  <- as.numeric(predict(cv_best_1se,  newx = X_test,  s = best_lambda_1se))

# Performance Metrics
metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Results tables
metric_results_min <- tibble::tibble(
  Rule    = "lambda.min",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"], metrics_valid_min["rmse"], metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],  metrics_valid_min["mae"],  metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"], metrics_valid_min["bias"], metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],   metrics_valid_min["r2"],   metrics_test_min["r2"])
)

metric_results_1se <- tibble::tibble(
  Rule    = "lambda.1se",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"], metrics_valid_1se["rmse"], metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],  metrics_valid_1se["mae"],  metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"], metrics_valid_1se["bias"], metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],   metrics_valid_1se["r2"],   metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias     Bias%        R2
# > lambda.min   Training 67.87973 53.84353 -4.881329e-13 1.9360120 0.4352170
# > lambda.min Validation 66.25781 51.30309  1.612303e+00 1.9219869 0.3771514
# > lambda.min       Test 67.74694 52.70898 -5.560840e+00 0.7296187 0.4816511

print(as.data.frame(metric_results_1se),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias     Bias%        R2
# > lambda.1se   Training 68.90752 54.72335 -4.334360e-13 2.0594705 0.4179845
# > lambda.1se Validation 67.10729 52.23215  1.212293e+00 1.9522026 0.3610782
# > lambda.1se       Test 68.21446 53.27576 -5.787738e+00 0.7928585 0.4744722
print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias     Bias%        R2
# > lambda.min   Training 67.87973 53.84353 -4.881329e-13 1.9360120 0.4352170
# > lambda.min Validation 66.25781 51.30309  1.612303e+00 1.9219869 0.3771514
# > lambda.min       Test 67.74694 52.70898 -5.560840e+00 0.7296187 0.4816511
# > lambda.1se   Training 68.90752 54.72335 -4.334360e-13 2.0594705 0.4179845
# > lambda.1se Validation 67.10729 52.23215  1.212293e+00 1.9522026 0.3610782
# > lambda.1se       Test 68.21446 53.27576 -5.787738e+00 0.7928585 0.4744722

# Sanity check: wrapper vs underlying at the same λ
# all.equal(as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min)),
#           as.numeric(predict(cv_best_min$glmnet.fit, newx = X_valid, s = best_lambda_min)))
# # > [1] TRUE
# all.equal(as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se)),
#           as.numeric(predict(cv_best_1se$glmnet.fit, newx = X_valid, s = best_lambda_1se)))
# # > [1] TRUE

#### ---- 2) Tuning: grouped = FALSE, keep = TRUE ----

# Grid over elastic‑net mixing parameter
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso
# options: alpha_grid <- seq(0, 1, by = 0.05), alpha_grid <- sort(unique(c(seq(0, 1, by = 0.1), 0.001, 0.005, 0.01, 0.05))), etc. 

# Storage
cv_list        <- vector("list", length(alpha_grid))  # cv.glmnet fits per alpha
per_alpha_list <- vector("list", length(alpha_grid))  # two rows per alpha: lambda.min & lambda.1se

# Parallel backend (for cv.glmnet parallelization)
registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Grid over alpha (cv.glmnet, manual 5-fold)")
for (i in seq_along(alpha_grid)) {
  alpha <- alpha_grid[i]
  message(sprintf("Fitting cv.glmnet for alpha = %.1f (manual 5-fold CV on TRAIN)", alpha))
  
  # Fix fold randomness 
  set.seed(123)
  
  cvmod <- cv.glmnet(
    x = X_train,
    y = y_train,
    weights = w_train,
    type.measure = "mse", 
    foldid = foldid,            # <-
    grouped = FALSE,            # <-                              
    keep = TRUE,                # <-                               
    parallel = TRUE,          
    trace.it = 0,
    alpha = alpha,
    family = "gaussian",
    standardize = TRUE,
    intercept = TRUE
  )
  cv_list[[i]] <- cvmod
  
  # Indices
  idx_min <- cvmod$index["min", 1]
  idx_1se <- cvmod$index["1se", 1]
  
  # Lambdas and path metadata at those indices
  lambda_min <- cvmod$lambda[idx_min]      # == cvmod$lambda.min
  lambda_1se <- cvmod$lambda[idx_1se]      # == cvmod$lambda.1se
  
  nzero_min  <- cvmod$nzero[idx_min]
  nzero_1se  <- cvmod$nzero[idx_1se]
  
  dev_min    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_min])
  dev_1se    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_1se])
  
  # CV summaries (in-built)
  cvm_min    <- as.numeric(cvmod$cvm[idx_min])
  cvsd_min   <- as.numeric(cvmod$cvsd[idx_min])
  cvup_min   <- as.numeric(cvmod$cvup[idx_min])
  cvlo_min   <- as.numeric(cvmod$cvlo[idx_min])
  
  cvm_1se    <- as.numeric(cvmod$cvm[idx_1se])
  cvsd_1se   <- as.numeric(cvmod$cvsd[idx_1se])
  cvup_1se   <- as.numeric(cvmod$cvup[idx_1se])
  cvlo_1se   <- as.numeric(cvmod$cvlo[idx_1se])
  
  # Two rows per α: one for lambda.min, one for lambda.1se
  per_alpha_list[[i]] <- tibble::tibble(
    alpha       = alpha,
    alpha_idx   = i,
    s           = c("lambda.min", "lambda.1se"),
    lambda      = c(lambda_min,   lambda_1se),
    lambda_idx  = c(idx_min,      idx_1se),
    nzero       = c(nzero_min,    nzero_1se),
    dev_ratio   = c(dev_min,      dev_1se),
    cvm         = c(cvm_min,      cvm_1se),
    cvsd        = c(cvsd_min,     cvsd_1se),
    cvlo        = c(cvlo_min,     cvlo_1se),
    cvup        = c(cvup_min,     cvup_1se)
  )
}
toc()
# > Grid over alpha (cv.glmnet, manual 5-fold): 2.07 sec elapsed

##### ---- Explore the second tuning results ----
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by CV MSE (lower is better) irrespective of rule
tuning_results %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  #head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx          s     lambda lambda_idx nzero dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.6         7 lambda.min  1.1140300         47    42 0.4352170 4764.684 93.16785 4671.516 4857.852
# >   0.7         8 lambda.min  0.9548829         47    41 0.4352598 4764.699 93.17640 4671.523 4857.876
# >   0.8         9 lambda.min  0.8355225         47    41 0.4352926 4764.704 93.18154 4671.523 4857.886
# >   0.9        10 lambda.min  0.7426867         47    41 0.4353175 4764.717 93.18573 4671.531 4857.903
# >   0.5         6 lambda.min  1.3368360         47    42 0.4351530 4764.718 93.15932 4671.559 4857.877
# >   1.0        11 lambda.min  0.6684180         47    41 0.4353370 4764.722 93.18927 4671.533 4857.911
# >   0.4         5 lambda.min  1.6710450         47    42 0.4350516 4764.817 93.14813 4671.669 4857.965
# >   0.3         4 lambda.min  2.0301253         48    42 0.4353499 4765.005 93.14825 4671.857 4858.153
# >   0.2         3 lambda.min  2.7746619         49    42 0.4354522 4765.456 93.14153 4672.314 4858.597
# >   0.1         2 lambda.min  4.6071461         51    45 0.4353364 4767.671 93.12395 4674.547 4860.795
# >   0.0         1 lambda.min  5.8135590         98    53 0.4371044 4779.595 93.41113 4686.183 4873.006
# >   0.4         5 lambda.1se  5.1031301         35    29 0.4175541 4843.104 94.42899 4748.675 4937.533
# >   0.2         3 lambda.1se  9.2995642         36    34 0.4177705 4845.776 94.36924 4751.407 4940.146
# >   0.3         4 lambda.1se  6.8041735         35    30 0.4167324 4848.758 94.48181 4754.276 4943.239
# >   1.0        11 lambda.1se  2.2402716         34    27 0.4161316 4850.120 94.65252 4755.467 4944.772
# >   0.9        10 lambda.1se  2.4891907         34    27 0.4160519 4850.511 94.65000 4755.861 4945.161
# >   0.8         9 lambda.1se  2.8003395         34    27 0.4159696 4851.066 94.64762 4756.418 4945.714
# >   0.1         2 lambda.1se 15.4413231         38    36 0.4182343 4851.306 94.33071 4756.976 4945.637
# >   0.7         8 lambda.1se  3.2003880         34    27 0.4158439 4851.848 94.64611 4757.202 4946.494
# >   0.6         7 lambda.1se  3.7337860         34    27 0.4156547 4853.002 94.64587 4758.356 4947.647
# >   0.5         6 lambda.1se  4.4805432         34    27 0.4153674 4854.889 94.65344 4760.236 4949.543
# >   0.0         1 lambda.1se 34.0501362         79    53 0.4207341 4862.424 94.20973 4768.214 4956.634

# Choose winners under each rule separately
best_min  <- tuning_results %>%
  filter(s == "lambda.min") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%   # tie-breaks: fewer nonzeros (nzero), larger λ, smaller α
  slice(1)

best_1se  <- tuning_results %>%
  filter(s == "lambda.1se") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  slice(1)

print(as.data.frame(best_min),  row.names = FALSE)
# > alpha alpha_idx          s  lambda lambda_idx nzero dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.6         7 lambda.min 1.11403         47    42  0.435217 4764.684 93.16785 4671.516 4857.852
print(as.data.frame(best_1se), row.names = FALSE)
# > alpha alpha_idx          s  lambda lambda_idx nzero dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.4         5 lambda.1se 5.10313         35    29 0.4175541 4843.104 94.42899 4748.675 4937.533

best_alpha_min     <- best_min$alpha
best_lambda_min    <- best_min$lambda
best_alpha_idx_min <- best_min$alpha_idx

best_alpha_1se     <- best_1se$alpha
best_lambda_1se    <- best_1se$lambda
best_alpha_idx_1se <- best_1se$alpha_idx

message(sprintf("Winner @ lambda.min : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_min, best_lambda_min, best_min$nzero, best_min$cvm, best_min$cvsd))
# > Winner @ lambda.min : alpha = 0.60 | lambda = 1.114030 | nzero = 42 | CVM = 4764.68406 (± 93.16785)
message(sprintf("Winner @ lambda.1se : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_1se, best_lambda_1se, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
# > Winner @ lambda.1se : alpha = 0.40 | lambda = 5.103130 | nzero = 29 | CVM = 4843.10362 (± 94.42899)

##### ---- Predict & evaluate on TRAIN / VALID / TEST for both winners ----
cv_best_min  <- cv_list[[best_alpha_idx_min]]
cv_best_1se  <- cv_list[[best_alpha_idx_1se]]

# Coefficients & importances
coef(cv_best_min,  s = best_lambda_min) %>% head()
# > 6 x 1 sparse Matrix of class "dgCMatrix"
# >               s=1.11403
# > (Intercept) 517.5748683
# > MATHMOT       .        
# > MATHEASE      2.7200581
# > MATHPREF      2.2764623
# > EXERPRAC     -2.6604818
# > STUDYHMW     -0.1180583
coef(cv_best_1se,  s = best_lambda_1se) %>% head()
# > 6 x 1 sparse Matrix of class "dgCMatrix"
# >               s=5.10313
# > (Intercept) 519.9731623
# > MATHMOT       .        
# > MATHEASE      0.9159452
# > MATHPREF      0.8455279
# > EXERPRAC     -2.1865588
# > STUDYHMW      .        

caret::varImp(cv_best_min$glmnet.fit,  lambda = best_lambda_min) %>% head()
# >            Overall
# > MATHMOT  0.0000000
# > MATHEASE 2.7200581
# > MATHPREF 2.2764623
# > EXERPRAC 2.6604818
# > STUDYHMW 0.1180583
# > WORKPAY  3.0259446
caret::varImp(cv_best_1se$glmnet.fit,  lambda = best_lambda_1se) %>% head()
# >            Overall
# > MATHMOT  0.0000000
# > MATHEASE 0.9159452
# > MATHPREF 0.8455279
# > EXERPRAC 2.1865588
# > STUDYHMW 0.0000000
# > WORKPAY  2.8536083

# Predictions
pred_train_min <- as.numeric(predict(cv_best_min, newx = X_train, s = best_lambda_min))
pred_valid_min <- as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min))
pred_test_min  <- as.numeric(predict(cv_best_min,  newx = X_test,  s = best_lambda_min))

pred_train_1se <- as.numeric(predict(cv_best_1se, newx = X_train, s = best_lambda_1se))
pred_valid_1se <- as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se))
pred_test_1se  <- as.numeric(predict(cv_best_1se,  newx = X_test,  s = best_lambda_1se))

# Performance Metrics
metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Results tables
metric_results_min <- tibble::tibble(
  Rule    = "lambda.min",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"], metrics_valid_min["rmse"], metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],  metrics_valid_min["mae"],  metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"], metrics_valid_min["bias"], metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],   metrics_valid_min["r2"],   metrics_test_min["r2"])
)

metric_results_1se <- tibble::tibble(
  Rule    = "lambda.1se",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"], metrics_valid_1se["rmse"], metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],  metrics_valid_1se["mae"],  metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"], metrics_valid_1se["bias"], metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],   metrics_valid_1se["r2"],   metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.min   Training 67.87973 53.84353 -4.881329e-13 1.9360120 0.4352170
# > lambda.min Validation 66.25781 51.30309  1.612303e+00 1.9219869 0.3771514
# > lambda.min       Test 67.74694 52.70898 -5.560840e+00 0.7296187 0.4816511

print(as.data.frame(metric_results_1se),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%        R2
# > lambda.1se   Training 68.93299 54.74834 -3.689229e-13 2.066835 0.4175541
# > lambda.1se Validation 67.08097 52.24331  1.207340e+00 1.957540 0.3615793
# > lambda.1se       Test 68.29907 53.33028 -5.788283e+00 0.801823 0.4731678

print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias    Bias%         R2
# > lambda.min   Training 67.87973 53.84353 -4.881329e-13 1.9360120 0.4352170
# > lambda.min Validation 66.25781 51.30309  1.612303e+00 1.9219869 0.3771514
# > lambda.min       Test 67.74694 52.70898 -5.560840e+00 0.7296187 0.4816511
# > lambda.1se   Training 68.93299 54.74834 -3.689229e-13 2.0668349 0.4175541
# > lambda.1se Validation 67.08097 52.24331  1.207340e+00 1.9575396 0.3615793
# > lambda.1se       Test 68.29907 53.33028 -5.788283e+00 0.8018230 0.4731678

# Sanity check: wrapper vs underlying at the same λ
# all.equal(as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min)),
#           as.numeric(predict(cv_best_min$glmnet.fit, newx = X_valid, s = best_lambda_min)))
# # > [1] TRUE
# all.equal(as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se)),
#           as.numeric(predict(cv_best_1se$glmnet.fit, newx = X_valid, s = best_lambda_1se)))
# # > [1] TRUE

## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH (best_alpha_min, best_lambda_min) to all plausible values in mathematics.


### ---- 1.1) Use best min results from "Tuning: grouped = TRUE (Default), keep = TRUE" ----

#### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

tic("Fitting main glmnet models (fixed best_alpha_min, best_lambda_min)")
main_models <- lapply(pvmaths, function(pv) {
  
  # TRAIN (final weights)
  X_train <- as.matrix(train_data[, voi_num])
  y_train <- train_data[[pv]]
  w_train <- train_data[[final_wt]]
  
  # Fit glmnet at chosen alpha + lambda
  mod <- glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    weights = w_train,
    alpha = best_alpha_min,       # best_alpha_min = 0
    lambda = best_lambda_min,     # best_lambda_min = 1.957261
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda_min))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(voi_num, collapse = " + "))),
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha_min, best_lambda_min): 1.523 sec elapsed

# Quick look
main_models[[1]]$formula
main_models[[1]]$coefs [1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 517.5736994   0.0000000   2.7212531   2.2755992  -2.6607364  -0.1172899 

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coefs[, 1:6]
# >       (Intercept)    MATHMOT MATHEASE MATHPREF  EXERPRAC   STUDYHMW
# >  [1,]    517.5737  0.0000000 2.721253 2.275599 -2.660736 -0.1172899
# >  [2,]    519.6306 -1.2223678 1.897873 3.250402 -2.880730 -0.4647726
# >  [3,]    516.0057 -1.3997406 4.924307 2.749535 -2.633424  0.0000000
# >  [4,]    517.3860  0.0000000 1.785653 5.400259 -3.025133 -0.2220097
# >  [5,]    523.9906 -0.9566022 3.886118 2.082004 -2.852850 -0.1985203
# >  [6,]    517.6682 -0.1876222 3.141074 1.795271 -2.769292  0.0000000
# >  [7,]    518.7157  0.0000000 1.200539 2.569180 -2.737537 -0.6378523
# >  [8,]    517.0580 -0.4304055 4.665853 5.882527 -2.863342  0.0000000
# >  [9,]    515.9801 -0.8551240 7.389052 0.000000 -2.904497  0.0000000
# > [10,]    513.0115  0.0000000 5.004562 1.605485 -2.840320  0.0000000

main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 517.7020148  -0.5051862   3.6616284   2.7610262  -2.8167861  -0.1640445 

# -- Visualize coefs ---
coef_df <- enframe(main_coef, name = "Term", value = "Estimate") %>%
  filter(Term != "(Intercept)") %>%                 # optional
  mutate(
    mag  = abs(Estimate),
    sign = if_else(Estimate >= 0, "Positive", "Negative")
  )
# symmetric x-axis (after coord_flip it's horizontal)
max_abs <- max(coef_df$mag, na.rm = TRUE)
ggplot(
  coef_df %>% mutate(Term = fct_reorder(Term, mag)), # order by |Estimate|
  aes(x = Term, y = Estimate, fill = sign)
) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.6, alpha = 0.6) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_y_continuous(limits = c(-max_abs, max_abs)) +  # put neg left, pos right
  scale_fill_manual(values = c("Negative" = "#b2cbea", "Positive" = "#2b6cb0")) +
  labs(title = "Pooled regression coefficients (ordered by |estimate|)",
       x = NULL, y = "Estimate", fill = NULL) +
  theme_minimal() +
  theme(legend.position = "top")

# --- Weighted R² on TRAIN (point estimates per PV) ---
main_r2s_weighted <- sapply(1:M, function(i) {
  model  <- main_models[[i]]$mod
  X_train   <- as.matrix(train_data[, voi_num])
  y_train <- train_data[[pvmaths[i]]]
  w      <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.4457822

#### ---- Replicate models using BRR replicate weights ----

set.seed(123)

tic("Fitting replicate glmnet models (BRR weights)")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    X_train <- as.matrix(train_data[, voi_num])
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    mod <- glmnet(
      x = X_train,
      y = y_train,
      family = "gaussian",
      weights = w_train,
      alpha = best_alpha_min,
      lambda = best_lambda_min,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda_min))
    coefs  <- coefs_matrix[, 1]
    names(coefs) <- rownames(coefs_matrix)
    
    list(
      formula = as.formula(paste(pv, "~", paste(voi_num, collapse = " + "))),
      mod     = mod,
      coefs   = coefs
    )
  })
})
toc()
# > Fitting replicate glmnet models (BRR weights): 5.166 sec elapsed

# Example inspect
replicate_models[[1]][[1]]$formula
replicate_models[[1]][[1]]$coefs

# --- Replicate weighted R² on TRAIN (G x M) (optional diagnostic) ---
rep_r2_weighted <- matrix(NA_real_, nrow = G, ncol = M)
for (m in 1:M) {
  X_train   <- as.matrix(train_data[, voi_num])
  y_train <- train_data[[pvmaths[m]]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)  # 80 x 10

#### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

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

##### ---- Final Outputs ----
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
as.data.frame(coef_table) %>% head() %>% print(row.names = FALSE)
# >        Term    Estimate Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# > (Intercept) 517.7020148  3.0507593 169.6961185  < 2e-16      *** 169.6961185  < 2e-16      ***
# >     MATHMOT  -0.5051862  0.7688460  -0.6570708 0.511135           -0.6570708 0.513045         
# >    MATHEASE   3.6616284  2.0425446   1.7926797 0.073024        .   1.7926797 0.076852        .
# >    MATHPREF   2.7610262  1.9041305   1.4500194 0.147053            1.4500194 0.151012         
# >    EXERPRAC  -2.8167861  0.1356894 -20.7590786  < 2e-16      *** -20.7590786  < 2e-16      ***
# >    STUDYHMW  -0.1640445  0.2396896  -0.6844037 0.493720           -0.6844037 0.495723         

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
# >                      Metric    Estimate  Std. Error
# >  R-squared (Weighted, Train)  0.4457822 0.006918145

#### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- as.matrix(train_data[, voi_num])
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
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
    X_train <- as.matrix(train_data[, voi_num])
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing train_metrics_replicates (glmnet): 2.171 sec elapsed
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
# >    Metric           Point_estimate         Standard_error                 CI_lower                 CI_upper               CI_length
# >       MSE 4418.7434614575095110922 84.4900544320431805545 4253.1459977188760603894 4584.3409251961429617950 331.1949274772669014055
# >      RMSE   66.4712447738467204772  0.6325565158291337475   65.2314567846354833591   67.7110327630579575953   2.4795759784224742361
# >       MAE   52.7779271817845057058  0.4807995233039162319   51.8355774323248041924   53.7202769312442072192   1.8846994989194030268
# >      Bias   -0.0000000000002541063  0.0000000000002432153   -0.0000000000007307996    0.0000000000002225869   0.0000000000009533865
# >     Bias%    1.8522215648750068873  0.0434866670249753826    1.7669892636983695056    1.9374538660516442690   0.1704646023532747634
# > R-squared    0.4457821862570495730  0.0069181446823458666    0.4322228718398143932    0.4593415006742847528   0.0271186288344703597

#### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- as.matrix(valid_data[, voi_num])
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_min))
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
    X_valid <- as.matrix(valid_data[, voi_num])
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 1.616 sec elapsed

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
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   4567.1579990   277.43050314 4023.4042046 5110.9117934 1087.50758873
# >      RMSE     67.5560613     2.04528974   63.5473671   71.5647555    8.01738846
# >       MAE     53.4054423     1.69387336   50.0855115   56.7253731    6.63986158
# >      Bias      1.6430899     1.40119182   -1.1031956    4.3893754    5.49257099
# >     Bias%      2.0533637     0.31816468    1.4297724    2.6769550    1.24718264
# > R-squared      0.3847536     0.02185354    0.3419215    0.4275858    0.08566431

#### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- as.matrix(test_data[, voi_num])
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_min))
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
    X_test <- as.matrix(test_data[, voi_num])
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates (glmnet): 0.965 sec elapsed

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
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4590.3829610   219.12411306 4160.9075913 5019.8583307 858.95073948
# >      RMSE     67.7373506     1.61612413   64.5698055   70.9048956   6.33509016
# >       MAE     53.1636001     1.22880994   50.7551768   55.5720233   4.81684643
# >      Bias     -4.5311294     1.48831579   -7.4481747   -1.6140840   5.83409071
# >     Bias%      0.9794910     0.26912466    0.4520163    1.5069656   1.05494929
# > R-squared      0.4769919     0.01421721    0.4491267    0.5048571   0.05573044

#### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.

# Helper
evaluate_split <- function(split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           voi_num, pvmaths, best_lambda_min) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    X     <- as.matrix(split_data[, voi_num])
    y     <- split_data[[pvmaths[i]]]
    w     <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X, s = best_lambda_min))
    compute_metrics(y_true = y, y_pred = y_pred, w = w)
  }) |> t() |> as.data.frame()
  main_point <- colMeans(main_metrics_df)   # length 6: mse, rmse, mae, bias, bias_pct, r2
  
  # Replicate metrics across PVs
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      X     <- as.matrix(split_data[, voi_num])
      y     <- split_data[[pvmaths[m]]]
      w     <- split_data[[rep_wts[g]]]
      y_pred <- as.numeric(predict(model, newx = X, s = best_lambda_min))
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
                             M, G, k, z_crit, voi_num, pvmaths, best_lambda_min)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, voi_num, pvmaths, best_lambda_min)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, voi_num, pvmaths, best_lambda_min)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

#### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                 CI_lower                 CI_upper               CI_length
# >       MSE 4418.7434614575095110922 84.4900544320431805545 4253.1459977188760603894 4584.3409251961429617950 331.1949274772669014055
# >      RMSE   66.4712447738467204772  0.6325565158291337475   65.2314567846354833591   67.7110327630579575953   2.4795759784224742361
# >       MAE   52.7779271817845057058  0.4807995233039162319   51.8355774323248041924   53.7202769312442072192   1.8846994989194030268
# >      Bias   -0.0000000000002541063  0.0000000000002432153   -0.0000000000007307996    0.0000000000002225869   0.0000000000009533865
# >     Bias%    1.8522215648750068873  0.0434866670249753826    1.7669892636983695056    1.9374538660516442690   0.1704646023532747634
# > R-squared    0.4457821862570495730  0.0069181446823458666    0.4322228718398143932    0.4593415006742847528   0.0271186288344703597

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   4567.1579990   277.43050314 4023.4042046 5110.9117934 1087.50758873
# >      RMSE     67.5560613     2.04528974   63.5473671   71.5647555    8.01738846
# >       MAE     53.4054423     1.69387336   50.0855115   56.7253731    6.63986158
# >      Bias      1.6430899     1.40119182   -1.1031956    4.3893754    5.49257099
# >     Bias%      2.0533637     0.31816468    1.4297724    2.6769550    1.24718264
# > R-squared      0.3847536     0.02185354    0.3419215    0.4275858    0.08566431

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4590.3829610   219.12411306 4160.9075913 5019.8583307 858.95073948
# >      RMSE     67.7373506     1.61612413   64.5698055   70.9048956   6.33509016
# >       MAE     53.1636001     1.22880994   50.7551768   55.5720233   4.81684643
# >      Bias     -4.5311294     1.48831579   -7.4481747   -1.6140840   5.83409071
# >     Bias%      0.9794910     0.26912466    0.4520163    1.5069656   1.05494929
# > R-squared      0.4769919     0.01421721    0.4491267    0.5048571   0.05573044

### ---- 1.2) Use best 1se results from "Tuning: grouped = TRUE (Default), keep = TRUE" ----

#### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

tic("Fitting main glmnet models (fixed best_alpha_1se, best_lambda_1se)")
main_models <- lapply(pvmaths, function(pv) {
  
  # TRAIN (final weights)
  X_train <- as.matrix(train_data[, voi_num])
  y_train <- train_data[[pv]]
  w_train <- train_data[[final_wt]]
  
  # Fit glmnet at chosen alpha + lambda
  mod <- glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    weights = w_train,
    alpha = best_alpha_1se,       # best_alpha_1se = 0
    lambda = best_lambda_1se,     # best_lambda_1se = 1.957261
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda_1se))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(voi_num, collapse = " + "))),
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha_1se, best_lambda_1se): 1.103 sec elapsed

# Quick look
main_models[[1]]$formula
main_models[[1]]$coefs[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 519.9712393   0.0000000   0.9223362   0.8446013  -2.1859906   0.0000000 

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coefs[,1:6]
# >      (Intercept) MATHMOT  MATHEASE   MATHPREF  EXERPRAC   STUDYHMW
# > [1,]    519.9712       0 0.9223362 0.84460125 -2.185991  0.0000000
# > [2,]    518.7611       0 0.3020417 1.65585466 -2.386759 -0.1598790
# > [3,]    517.5308       0 3.0207413 1.18320190 -2.119607  0.0000000
# > [4,]    519.5242       0 0.3384220 3.59474243 -2.501755  0.0000000
# > [5,]    523.2289       0 2.0051280 0.45867835 -2.393954  0.0000000
# > [6,]    519.3684       0 1.3115918 0.16588467 -2.246450  0.0000000
# > [7,]    520.5254       0 0.0000000 0.71789110 -2.312383 -0.1968145
# > [8,]    517.1422       0 2.8384049 4.06122450 -2.309177  0.0000000
# > [9,]    515.2380       0 4.7068924 0.00000000 -2.393160  0.0000000
# > [10,]   516.1473       0 3.1632302 0.02232452 -2.365913  0.0000000

main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef[1:6]
# >  (Intercept)      MATHMOT     MATHEASE     MATHPREF     EXERPRAC     STUDYHMW 
# > 518.74374385   0.00000000   1.86087885   1.27044034  -2.32151478  -0.03566935 

# -- Visualize coefs ---
coef_df <- enframe(main_coef, name = "Term", value = "Estimate") %>%
  filter(Term != "(Intercept)") %>%                 # optional
  mutate(
    mag  = abs(Estimate),
    sign = if_else(Estimate >= 0, "Positive", "Negative")
  )

# All coefficients sorted by |estimate| (descending) and display top 10
coef_df %>% 
  arrange(desc(mag), desc(Estimate)) %>%
  slice_head(n = 10) %>%
  transmute(Term, Estimate = round(Estimate, 3), `|Estimate|` = round(mag, 3))

# symmetric x-axis (after coord_flip it's horizontal)
max_abs <- max(coef_df$mag, na.rm = TRUE)
ggplot(
  coef_df %>% mutate(Term = fct_reorder(Term, mag)), # order by |Estimate|
  aes(x = Term, y = Estimate, fill = sign)
) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.6, alpha = 0.6) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_y_continuous(limits = c(-max_abs, max_abs)) +  # put neg left, pos right
  scale_fill_manual(values = c("Negative" = "#b2cbea", "Positive" = "#2b6cb0")) +
  labs(title = "Pooled regression coefficients (ordered by |estimate|)",
       x = NULL, y = "Estimate", fill = NULL) +
  theme_minimal() +
  theme(legend.position = "top")

# --- Weighted R² on TRAIN (point estimates per PV) ---
main_r2s_weighted <- sapply(1:M, function(i) {
  model  <- main_models[[i]]$mod
  X_train   <- as.matrix(train_data[, voi_num])
  y_train <- train_data[[pvmaths[i]]]
  w      <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.4277401

#### ---- Replicate models using BRR replicate weights ----

set.seed(123)

tic("Fitting replicate glmnet models (BRR weights)")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    X_train <- as.matrix(train_data[, voi_num])
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    mod <- glmnet(
      x = X_train,
      y = y_train,
      family = "gaussian",
      weights = w_train,
      alpha = best_alpha_1se,
      lambda = best_lambda_1se,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda_1se))
    coefs  <- coefs_matrix[, 1]
    names(coefs) <- rownames(coefs_matrix)
    
    list(
      formula = as.formula(paste(pv, "~", paste(voi_num, collapse = " + "))),
      mod     = mod,
      coefs   = coefs
    )
  })
})
toc()
# > Fitting replicate glmnet models (BRR weights): 4.733 sec elapsed

# Example inspect
replicate_models[[1]][[1]]$formula
replicate_models[[1]][[1]]$coefs

# --- Replicate weighted R² on TRAIN (G x M) (optional diagnostic) ---
rep_r2_weighted <- matrix(NA_real_, nrow = G, ncol = M)
for (m in 1:M) {
  X_train   <- as.matrix(train_data[, voi_num])
  y_train <- train_data[[pvmaths[m]]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)  # 80 x 10

#### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

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

##### ---- Final Outputs ----
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
as.data.frame(coef_table) %>% head() %>% print(row.names = FALSE)
# >        Term     Estimate  Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# > (Intercept) 518.74374385 2.496186144 207.8145274  < 2e-16      *** 207.8145274  < 2e-16      ***
# >     MATHMOT   0.00000000 0.002479268   0.0000000   1.0000            0.0000000 1.000000         
# >    MATHEASE   1.86087885 1.671859944   1.1130591   0.2657            1.1130591 0.269058         
# >    MATHPREF   1.27044034 1.570209084   0.8090899   0.4185            0.8090899 0.420894         
# >    EXERPRAC  -2.32151478 0.128980417 -17.9989709  < 2e-16      *** -17.9989709  < 2e-16      ***
# >    STUDYHMW  -0.03566935 0.082269610  -0.4335665   0.6646           -0.4335665 0.665785         

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
# >                      Metric    Estimate  Std. Error
# >  R-squared (Weighted, Train)  0.4277401 0.007222107

#### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- as.matrix(train_data[, voi_num])
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
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
    X_train <- as.matrix(train_data[, voi_num])
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
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
# >    Metric           Point_estimate         Standard_error                 CI_lower                 CI_upper               CI_length
# >       MSE 4562.5292456999768546666 83.5175483531163536099 4398.8378588507866879809 4726.2206325491670213523 327.3827736983803333715
# >      RMSE   67.5442797744234155743  0.6153146550932471204   66.3382852112809615619   68.7502743375658695868   2.4119891262849080249
# >       MAE   53.5757798924725676670  0.4943083290017316611   52.6069533703709950601   54.5446064145741402740   1.9376530442031452139
# >      Bias   -0.0000000000002682753  0.0000000000002403238   -0.0000000000007393012    0.0000000000002027506   0.0000000000009420518
# >     Bias%    1.9838909444758965339  0.0432103170882652196    1.8992002792223410257    2.0685816097294518201   0.1693813305071107944
# > R-squared    0.4277401310654539435  0.0072221071264166768    0.4135850612051871766    0.4418952009257207103   0.0283101397205335337

#### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- as.matrix(valid_data[, voi_num])
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_1se))
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
    X_valid <- as.matrix(valid_data[, voi_num])
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_1se))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 1.591 sec elapsed

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
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   4684.3379456   274.39014122 4146.5431511 5222.1327401 1075.58958902
# >      RMSE     68.4190426     1.99728672   64.5044326   72.3336527    7.82922008
# >       MAE     53.9677436     1.57630187   50.8782487   57.0572385    6.17898977
# >      Bias      1.3486409     1.45397149   -1.5010908    4.1983727    5.69946350
# >     Bias%      2.1126229     0.33124733    1.4633900    2.7618557    1.29846566
# > R-squared      0.3689365     0.02111864    0.3275447    0.4103283    0.08278355

#### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- as.matrix(test_data[, voi_num])
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_1se))
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
    X_test <- as.matrix(test_data[, voi_num])
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_1se))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates (glmnet): 1.211 sec elapsed

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
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4693.3672921   217.60472681 4266.8698646 5119.8647195 852.99485484
# >      RMSE     68.4938842     1.58388534   65.3895259   71.5982424   6.20871645
# >       MAE     54.0084733     1.26758458   51.5240532   56.4928935   4.96884027
# >      Bias     -4.8498475     1.46958364   -7.7301785   -1.9695165   5.76066203
# >     Bias%      1.0342493     0.26558152    0.5137191    1.5547795   1.04106043
# > R-squared      0.4652552     0.01312527    0.4395301    0.4909802   0.05145011

#### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.

# Helper
evaluate_split <- function(split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           voi_num, pvmaths, best_lambda_1se) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    X     <- as.matrix(split_data[, voi_num])
    y     <- split_data[[pvmaths[i]]]
    w     <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X, s = best_lambda_1se))
    compute_metrics(y_true = y, y_pred = y_pred, w = w)
  }) |> t() |> as.data.frame()
  main_point <- colMeans(main_metrics_df)   # length 6: mse, rmse, mae, bias, bias_pct, r2
  
  # Replicate metrics across PVs
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      X     <- as.matrix(split_data[, voi_num])
      y     <- split_data[[pvmaths[m]]]
      w     <- split_data[[rep_wts[g]]]
      y_pred <- as.numeric(predict(model, newx = X, s = best_lambda_1se))
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
                             M, G, k, z_crit, voi_num, pvmaths, best_lambda_1se)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, voi_num, pvmaths, best_lambda_1se)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, voi_num, pvmaths, best_lambda_1se)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

#### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                 CI_lower                 CI_upper               CI_length
# >       MSE 4562.5292456999768546666 83.5175483531163536099 4398.8378588507866879809 4726.2206325491670213523 327.3827736983803333715
# >      RMSE   67.5442797744234155743  0.6153146550932471204   66.3382852112809615619   68.7502743375658695868   2.4119891262849080249
# >       MAE   53.5757798924725676670  0.4943083290017316611   52.6069533703709950601   54.5446064145741402740   1.9376530442031452139
# >      Bias   -0.0000000000002682753  0.0000000000002403238   -0.0000000000007393012    0.0000000000002027506   0.0000000000009420518
# >     Bias%    1.9838909444758965339  0.0432103170882652196    1.8992002792223410257    2.0685816097294518201   0.1693813305071107944
# > R-squared    0.4277401310654539435  0.0072221071264166768    0.4135850612051871766    0.4418952009257207103   0.0283101397205335337

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   4684.3379456   274.39014122 4146.5431511 5222.1327401 1075.58958902
# >      RMSE     68.4190426     1.99728672   64.5044326   72.3336527    7.82922008
# >       MAE     53.9677436     1.57630187   50.8782487   57.0572385    6.17898977
# >      Bias      1.3486409     1.45397149   -1.5010908    4.1983727    5.69946350
# >     Bias%      2.1126229     0.33124733    1.4633900    2.7618557    1.29846566
# > R-squared      0.3689365     0.02111864    0.3275447    0.4103283    0.08278355

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4693.3672921   217.60472681 4266.8698646 5119.8647195 852.99485484
# >      RMSE     68.4938842     1.58388534   65.3895259   71.5982424   6.20871645
# >       MAE     54.0084733     1.26758458   51.5240532   56.4928935   4.96884027
# >      Bias     -4.8498475     1.46958364   -7.7301785   -1.9695165   5.76066203
# >     Bias%      1.0342493     0.26558152    0.5137191    1.5547795   1.04106043
# > R-squared      0.4652552     0.01312527    0.4395301    0.4909802   0.05145011
