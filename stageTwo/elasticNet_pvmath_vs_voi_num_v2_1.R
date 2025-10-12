# ---- II. Predictive Modelling: Version 2.1 ----

# Tune hyperparameters (alpha α and lambda λ) using glmnet only

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
# library(Matrix)     # (optional) if use sparse model matrices

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("tidyverse","glmnet","Matrix","haven","broom","tictoc","caret"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         tidyverse            glmnet            Matrix             haven             broom            tictoc            caret 
# > "tidyverse 2.0.0"   "glmnet 4.1.10"    "Matrix 1.7.3"     "haven 2.5.4"     "broom 1.0.8"    "tictoc 1.2.1"     "caret 7.0.1" 

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
         all_of(pvmaths), all_of(voi_num )) %>%    # PVs + predictors
  filter(if_all(all_of(voi_num ), ~ !is.na(.)))    # Listwise deletion for predictors

dim(temp_data)  # 6944 x 146

# Check summaries
summary(temp_data[[final_wt]])
summary(temp_data[, pvmaths])
sapply(temp_data[, pvmaths], sd)
summary(temp_data[, voi_num ])
sapply(temp_data[, voi_num ], sd)

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
# >                   train    valid     test
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

### ---- Fit main model using final student weights (W_FSTUWT) on the training data ---- 

#### ---- Tuning for PV1MATH only: glmnet ----

# Target plausible value
pv1math <- pvmaths[1]

# Training data
X_train <- as.matrix(train_data[, voi_num ])
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

# Validation data 
X_valid <- as.matrix(valid_data[, voi_num ])
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

# Test data 
X_test <- as.matrix(test_data[, voi_num ])
y_test <- test_data[[pv1math]]
w_test <- test_data[[final_wt]]

# Grid over elastic‑net mixing parameter
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso
# options: alpha_grid <- seq(0, 1, by = 0.05), alpha_grid <- sort(unique(c(seq(0, 1, by = 0.1), 0.001, 0.005, 0.01, 0.05))), etc. 

# Storage
mod_list        <- vector("list", length(alpha_grid))      # glmnet fits per alpha
per_alpha_list  <- vector("list", length(alpha_grid))      # per‑lambda metrics per alpha

tic("Grid over alpha (glmnet) -> select by Validation RMSE")
for (i in seq_along(alpha_grid)) {
  alpha   <- alpha_grid[i]
  message(sprintf("Fitting glmnet path for alpha = %.1f", alpha))
  
  # Fit the whole lambda path at this alpha on TRAIN
  mod <- glmnet(
    x = X_train, 
    y = y_train,
    family     = "gaussian",
    weights    = w_train,
    alpha      = alpha,
    standardize= TRUE,
    intercept  = TRUE
  )
  mod_list[[i]] <- mod
  
  # Predictions across the path
  pred_valid_matrix <- predict(mod, newx = X_valid, s = mod$lambda)
  pred_train_matrix <- predict(mod, newx = X_train, s = mod$lambda)
  
  # RMSE curves (weighted)
  rmse_valid_vec <- apply(pred_valid_matrix, 2, function(p) w_rmse(y_valid, p, w_valid))
  rmse_train_vec <- apply(pred_train_matrix, 2, function(p) w_rmse(y_train, p, w_train))
  
  # Collect tidy rows per alpha (one row per lambda on this path)
  per_alpha_list[[i]] <- tibble(
    alpha       = alpha,
    alpha_idx   = i,
    lambda      = as.numeric(mod$lambda),
    lambda_idx  = seq_along(mod$lambda),
    df          = mod$df,
    dev_ratio   = as.numeric(mod$dev.ratio),
    rmse_valid  = as.numeric(rmse_valid_vec),
    rmse_train  = as.numeric(rmse_train_vec)
  )
}
toc()
# > Grid over alpha (glmnet) -> select by Validation RMSE: 1.343 sec elapsed

##### ---- Explore tuning results ----

# Stack all alpha–lambda candidates
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by Validation RMSE
tuning_results %>%
  arrange(rmse_valid) %>%
  head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx   lambda lambda_idx df dev_ratio rmse_valid rmse_train
# >     0         1 17.75377         86 53 0.4312370   65.80472   68.11849
# >     0         1 16.17657         87 53 0.4321461   65.80623   68.06403
# >     0         1 19.48474         85 53 0.4302050   65.80915   68.18026
# >     0         1 14.73949         88 53 0.4329541   65.80993   68.01559
# >     0         1 13.43008         89 53 0.4336642   65.81688   67.97299
# >     0         1 21.38448         84 53 0.4290415   65.81874   68.24983
# >     0         1 12.23698         90 53 0.4342876   65.82648   67.93556
# >     0         1 23.46944         83 53 0.4277327   65.83429   68.32801
# >     0         1 11.14988         91 53 0.4348336   65.83824   67.90277
# >     0         1 10.15936         92 53 0.4353104   65.85170   67.87412

# Choose the best (Validation RMSE); tie‑break: fewer non‑zero coefs → larger lambda → smaller alpha
best_row <- tuning_results %>%
  arrange(rmse_valid, df, desc(lambda), alpha) %>%
  slice(1)
print(as.data.frame(best_row), row.names = FALSE)
# > alpha alpha_idx   lambda lambda_idx df dev_ratio rmse_valid rmse_train
# >     0         1 17.75377         86 53  0.431237   65.80472   68.11849

best_alpha      <- best_row$alpha
best_lambda     <- best_row$lambda
best_alpha_idx  <- best_row$alpha_idx
best_df         <- best_row$df
best_rmse_valid <- best_row$rmse_valid

message(sprintf(
  "Selected: alpha = %.2f | lambda = %.6f | df = %d | Valid RMSE = %.5f",
  best_alpha, best_lambda, best_df, best_rmse_valid
))
# > Selected: alpha = 0.00 | lambda = 17.753769 | df = 53 | Valid RMSE = 65.80472

#### ---- Predict and evaluate performance on training/validation/test datasets ----
best_mod <- mod_list[[best_alpha_idx]]

coef(best_mod, s = best_lambda) %>% head()
# > 6 x 1 sparse Matrix of class "dgCMatrix"
# >              s=17.75377
# > (Intercept) 517.8652021
# > MATHMOT      -2.5041223
# > MATHEASE      4.7751354
# > MATHPREF      4.3830439
# > EXERPRAC     -2.4278175
# > STUDYHMW     -0.4272713
varImp(best_mod, lambda = best_lambda) %>% head()
# >            Overall
# > MATHMOT  2.5041223
# > MATHEASE 4.7751354
# > MATHPREF 4.3830439
# > EXERPRAC 2.4278175
# > STUDYHMW 0.4272713
# > WORKPAY  2.9400311

pred_train_best <- as.numeric(predict(best_mod, newx = X_train, s = best_lambda))
pred_valid_best <- as.numeric(predict(best_mod, newx = X_valid, s = best_lambda))
pred_test_best  <- as.numeric(predict(best_mod,  newx = X_test,  s = best_lambda))

metrics_train_best <- compute_metrics(y_true = y_train, y_pred = pred_train_best, w = w_train)
metrics_valid_best <- compute_metrics(y_true = y_valid, y_pred = pred_valid_best, w = w_valid)
metrics_test_best  <- compute_metrics(y_true = y_test,  y_pred = pred_test_best,  w = w_test)

metric_results <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_best["rmse"], metrics_valid_best["rmse"], metrics_test_best["rmse"]),
  MAE     = c(metrics_train_best["mae"],  metrics_valid_best["mae"],  metrics_test_best["mae"]),
  Bias    = c(metrics_train_best["bias"], metrics_valid_best["bias"], metrics_test_best["bias"]),
  `Bias%` = c(metrics_train_best["bias_pct"], metrics_valid_best["bias_pct"], metrics_test_best["bias_pct"]),
  R2      = c(metrics_train_best["r2"],   metrics_valid_best["r2"],   metrics_test_best["r2"])
)

print(as.data.frame(metric_results), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 68.11849 54.04077 -4.052767e-13 2.0114125 0.4312370
# > Validation 65.80472 51.50513  1.292509e+00 1.9226634 0.3856408
# >       Test 68.73310 53.33290 -5.468503e+00 0.8495288 0.4664505

## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH (best_alpha, best_lambda) to all plausible values in mathematics.

### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

tic("Fitting main glmnet models (fixed best_alpha, best_lambda)")
main_models <- lapply(pvmaths, function(pv) {
  
  # TRAIN (final weights)
  X_train <- as.matrix(train_data[, voi_num ])
  y_train <- train_data[[pv]]
  w_train <- train_data[[final_wt]]
  
  # Fit glmnet at chosen alpha + lambda
  mod <- glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    weights = w_train,
    alpha = best_alpha,       # best_alpha = 0
    lambda = best_lambda,     # best_lambda = 8.671881
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(voi_num , collapse = " + "))),
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha, best_lambda): 6.606 sec elapsed

# Quick look
main_models[[1]]$formula
main_models[[1]]$coefs [1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW  
# > 517.8962495  -2.4652825   4.8007110   4.3747733  -2.4311872  -0.4317127

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coefs[, 1:6]
# >      (Intercept)     MATHMOT MATHEASE MATHPREF  EXERPRAC   STUDYHMW
# > [1,]    517.8962 -2.46528248 4.800711 4.374773 -2.431187 -0.4317127
# > [2,]    519.5984 -4.60230546 4.293952 5.036224 -2.581823 -0.6802777
# > [3,]    516.0707 -4.66647799 6.625039 4.889839 -2.377491 -0.3017808
# > [4,]    517.5889 -0.07792862 4.081685 6.392688 -2.709604 -0.5035949
# > [5,]    523.5515 -4.41323535 5.640916 4.188951 -2.556900 -0.4964423
# > [6,]    517.4072 -3.92970035 5.217411 4.166907 -2.482559 -0.2767835
# > [7,]    516.9973 -0.60122515 3.452506 4.016707 -2.491721 -0.8220998
# > [8,]    516.7387 -3.82438531 6.624460 7.408688 -2.600435 -0.1689491
# > [9,]    515.9362 -4.24642033 8.727501 2.306437 -2.620451 -0.1609251
# > [10,]    511.7495 -2.32310068 6.595578 3.680523 -2.568987 -0.1606222

main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW
# > 517.3534592  -3.1150062   5.6059759   4.6461737  -2.5421158  -0.4003188 

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
  X_train   <- as.matrix(train_data[, voi_num ])
  y_train <- train_data[[pvmaths[i]]]
  w      <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.4418933

### ---- Replicate models using BRR replicate weights ----

set.seed(123)

tic("Fitting replicate glmnet models (BRR weights)")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    X_train <- as.matrix(train_data[, voi_num ])
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    mod <- glmnet(
      x = X_train,
      y = y_train,
      family = "gaussian",
      weights = w_train,
      alpha = best_alpha,
      lambda = best_lambda,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda))
    coefs  <- coefs_matrix[, 1]
    names(coefs) <- rownames(coefs_matrix)
    
    list(
      formula = as.formula(paste(pv, "~", paste(voi_num , collapse = " + "))),
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
  X_train   <- as.matrix(train_data[, voi_num ])
  y_train <- train_data[[pvmaths[m]]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)  # 80 x 10

### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

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

#### ---- Final Outputs ----
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
# >        Term    Estimate Std. Error    z value Pr(>|z|) z_Signif    t value Pr(>|t|) t_Signif
# > (Intercept) 517.3534592  6.4388567 80.3486527  < 2e-16      *** 80.3486527  < 2e-16      ***
# >     MATHMOT  -3.1150062  5.5242767 -0.5638758 0.572839          -0.5638758 0.574436         
# >    MATHEASE   5.6059759  3.6819396  1.5225605 0.127869           1.5225605 0.131862         
# >    MATHPREF   4.6461737  3.8926552  1.1935744 0.232644           1.1935744 0.236216         
# >    EXERPRAC  -2.5421158  0.3541639 -7.1777957 7.08e-13      *** -7.1777957 3.41e-10      ***
# >    STUDYHMW  -0.4003188  0.4636670 -0.8633758 0.387931          -0.8633758 0.390545         

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
# >                      Metric  Estimate Std. Error
# > R-squared (Weighted, Train) 0.4418933 0.01451664

### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  X_train <- as.matrix(train_data[, voi_num ])
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
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
    X_train <- as.matrix(train_data[, voi_num ])
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
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
# >    Metric           Point_estimate         Standard_error                CI_lower               CI_upper              CI_length
# >       MSE 4449.7532250988106170553 144.899607251646216355 4165.755213511585679953 4733.75123668603555416 567.996023174449874205
# >      RMSE   66.7040754089274940952   1.086516516038095403   64.574542168884889293   68.83360864897009890   4.259066480085209605
# >       MAE   52.9847425984435886903   0.883861130308797982   51.252406615703478110   54.71707858118369927   3.464671965480221161
# >      Bias   -0.0000000000002364989   0.000000000001019646   -0.000000000002234967    0.00000000000176197   0.000000000003996937
# >     Bias%    1.9274083761824356564   0.070503626155948101    1.789223808137301308    2.06559294422757000   0.276369136090268697
# > R-squared    0.4418933297200217103   0.014516636838995492    0.413441244338943192    0.47034541510110023   0.056904170762157036

### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- as.matrix(valid_data[, voi_num ])
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda))
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
    X_valid <- as.matrix(valid_data[, voi_num ])
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 0.977 sec elapsed

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
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4527.5934815   437.57689464 3669.9585275 5385.2284355 1715.2699079
# >      RMSE     67.2626293     3.23835589   60.9155684   73.6096903   12.6941218
# >       MAE     53.3500174     2.64335840   48.1691302   58.5309047   10.3617745
# >      Bias      1.4138635     3.82052082   -6.0742197    8.9019467   14.9761664
# >     Bias%      2.0712899     0.79299419    0.5170498    3.6255299    3.1084801
# > R-squared      0.3900986     0.04296723    0.3058844    0.4743128    0.1684285

### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- as.matrix(test_data[, voi_num ])
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda))
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
    X_test <- as.matrix(test_data[, voi_num ])
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates (glmnet): 1.242 sec elapsed

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
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper   CI_length
# >       MSE   4717.8710136   461.37170743 3813.5990836 5622.1429437 1808.543860
# >      RMSE     68.6700473     3.34617773   62.1116595   75.2284351   13.116776
# >       MAE     53.7429015     2.57432227   48.6973225   58.7884804   10.091158
# >      Bias     -4.4941976     4.00479892  -12.3434592    3.3550641   15.698523
# >     Bias%      1.0888002     0.78502623   -0.4498229    2.6274234    3.077246
# > R-squared      0.4625157     0.03623052    0.3915052    0.5335262    0.142021

### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.

evaluate_split <- function(split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           voi_num , pvmaths, best_lambda) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    X     <- as.matrix(split_data[, voi_num ])
    y     <- split_data[[pvmaths[i]]]
    w     <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X, s = best_lambda))
    compute_metrics(y_true = y, y_pred = y_pred, w = w)
  }) |> t() |> as.data.frame()
  main_point <- colMeans(main_metrics_df)   # length 6: mse, rmse, mae, bias, bias_pct, r2
  
  # Replicate metrics across PVs
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      X     <- as.matrix(split_data[, voi_num ])
      y     <- split_data[[pvmaths[m]]]
      w     <- split_data[[rep_wts[g]]]
      y_pred <- as.numeric(predict(model, newx = X, s = best_lambda))
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
                             M, G, k, z_crit, voi_num , pvmaths, best_lambda)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, voi_num , pvmaths, best_lambda)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, voi_num , pvmaths, best_lambda)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                CI_lower               CI_upper              CI_length
# >       MSE 4449.7532250988106170553 144.899607251646216355 4165.755213511585679953 4733.75123668603555416 567.996023174449874205
# >      RMSE   66.7040754089274940952   1.086516516038095403   64.574542168884889293   68.83360864897009890   4.259066480085209605
# >       MAE   52.9847425984435886903   0.883861130308797982   51.252406615703478110   54.71707858118369927   3.464671965480221161
# >      Bias   -0.0000000000002364989   0.000000000001019646   -0.000000000002234967    0.00000000000176197   0.000000000003996937
# >     Bias%    1.9274083761824356564   0.070503626155948101    1.789223808137301308    2.06559294422757000   0.276369136090268697
# > R-squared    0.4418933297200217103   0.014516636838995492    0.413441244338943192    0.47034541510110023   0.056904170762157036

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4527.5934815   437.57689464 3669.9585275 5385.2284355 1715.2699079
# >      RMSE     67.2626293     3.23835589   60.9155684   73.6096903   12.6941218
# >       MAE     53.3500174     2.64335840   48.1691302   58.5309047   10.3617745
# >      Bias      1.4138635     3.82052082   -6.0742197    8.9019467   14.9761664
# >     Bias%      2.0712899     0.79299419    0.5170498    3.6255299    3.1084801
# > R-squared      0.3900986     0.04296723    0.3058844    0.4743128    0.1684285

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper   CI_length
# >       MSE   4717.8710136   461.37170743 3813.5990836 5622.1429437 1808.543860
# >      RMSE     68.6700473     3.34617773   62.1116595   75.2284351   13.116776
# >       MAE     53.7429015     2.57432227   48.6973225   58.7884804   10.091158
# >      Bias     -4.4941976     4.00479892  -12.3434592    3.3550641   15.698523
# >     Bias%      1.0888002     0.78502623   -0.4498229    2.6274234    3.077246
# > R-squared      0.4625157     0.03623052    0.3915052    0.5335262    0.142021

