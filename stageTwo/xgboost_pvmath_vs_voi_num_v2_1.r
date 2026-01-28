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
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE)
dim(pisa_2022_canada_merged)   # 23073 x 1278

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
n <- nrow(temp_data)   # 6944
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

### ---- Fit main model using final student weights (W_FSTUWT) on the training data ----

#### ---- Tune XGBoost model for PV1MATH only: xgb.train ----

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
# > Tuning (xgb.train): 185.46 sec elapsed

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
# >       11 nrounds=200, max_depth=4, eta=0.05     200         4 0.05               174   49.32077   65.70228
# >       12 nrounds=300, max_depth=4, eta=0.05     300         4 0.05               174   49.32077   65.70228
# >       14 nrounds=200, max_depth=6, eta=0.05     200         6 0.05               188   29.61840   65.80193
# >       20  nrounds=200, max_depth=4, eta=0.1     200         4 0.10               107   46.65733   65.86578
# >       21  nrounds=300, max_depth=4, eta=0.1     300         4 0.10               107   46.65733   65.86578
print(head(as.data.frame(tuning_results %>% arrange(valid_rmse, best_iter_in_grid, nrounds))), row.names = FALSE)  # Tie-break
print(as.data.frame(tuning_results %>% arrange(valid_rmse)), row.names = FALSE)  # Print all
#print(tuning_results %>% arrange(valid_rmse), n = Inf)

# --- Explore best_model ---
best_model                  # str(best_model)                 
best_model$evaluation_log   # <=> best_eval_log
best_rmse
# > [1] 65.70228
best_params
best_params$max_depth; best_params$eta
# > [1] 4
# > [1] 0.05
best_iter
# > [1] 174
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

xgb.importance(model = best_model) %>% head()                            #?xgb.importance
# >     Feature       Gain      Cover  Frequency
# >      <char>      <num>      <num>      <num>
# > 1:  MATHEFF 0.40490042 0.07662199 0.04419703
# > 2:     ESCS 0.05621067 0.04199433 0.04251012
# > 3: FAMSUPSL 0.04832341 0.04297436 0.04251012
# > 4:   ANXMAT 0.04136378 0.03491656 0.02766532
# > 5:   FAMCON 0.02743916 0.03774013 0.03103914
# > 6: MATHEF21 0.02421468 0.03557355 0.02361673

xgb.importance(model = best_model, trees = 0:(best_iter - 1)) %>% head()  # 0-based indices
# >     Feature       Gain      Cover  Frequency
# >      <char>      <num>      <num>      <num>
# > 1:  MATHEFF 0.41441651 0.08647937 0.04531371
# > 2:     ESCS 0.05677214 0.04661289 0.04337723
# > 3: FAMSUPSL 0.04910481 0.04670405 0.04608830
# > 4:   ANXMAT 0.04204422 0.03567853 0.02982184
# > 5:   FAMCON 0.02738327 0.04214522 0.03214562
# > 6: MATHEF21 0.02449374 0.03656461 0.02478699
sum(xgb.importance(model = best_model, trees = 0:(best_iter - 1))$Gain)
# > [1] 1

xgb.plot.importance(importance_matrix = xgb.importance(model = best_model, trees = 0:(best_iter - 1)),
                    top_n = NULL,                              # Top n features 
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
# >   Training 49.32077 38.75174 -0.06843546 1.2828857 0.7018320
# > Validation 65.70228 51.74808 -1.38334543 1.2745293 0.3875520
# >       Test 64.82269 50.64938 -5.95540185 0.4893086 0.5254336

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

shap_train <- get_shap(best_model, dtrain, pred_train, voi_num)
shap_valid <- get_shap(best_model, dvalid, pred_valid, voi_num)
shap_test  <- get_shap(best_model, dtest,  pred_test,  voi_num)

# Helper: weighted global importance (mean |SHAP|), excluding BIAS
global_shap <- function(shap_mat, w, features) {
  tibble::tibble(
    feature = features,
    mean_abs_shap = colSums(abs(shap_mat[, features, drop = FALSE]) * w) / sum(w)
  ) |>
    dplyr::arrange(dplyr::desc(mean_abs_shap))
}

global_importance_train <- global_shap(shap_train, w_train, voi_num)
global_importance_valid <- global_shap(shap_valid, w_valid, voi_num)
global_importance_test  <- global_shap(shap_test,  w_test,  voi_num)

as.data.frame(global_importance_train) %>% head() %>% print(row.names=FALSE) 
# >  feature mean_abs_shap
# >  MATHEFF     31.181601
# >     ESCS      9.553925
# > FAMSUPSL      9.110662
# >   ANXMAT      7.335649
# > MATHEF21      4.954477
# >   FAMCON      4.778166
as.data.frame(global_importance_valid) %>% head() %>% print(row.names=FALSE) 
# >  feature mean_abs_shap
# >  MATHEFF     31.434914
# >     ESCS     10.141874
# > FAMSUPSL      9.339531
# >   ANXMAT      7.507874
# > MATHEF21      4.853016
# >   FAMCON      4.845689
as.data.frame(global_importance_test) %>% head() %>% print(row.names=FALSE) 
# >  feature mean_abs_shap
# >  MATHEFF     32.830012
# >     ESCS      9.378453
# > FAMSUPSL      8.911687
# >   ANXMAT      7.737920
# > MATHEF21      5.298415
# >   FAMCON      4.898073

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
# Assumes these already exist: best_model, dtrain, train_data, w_train, voi_num, pred_train, shap_train
# Note: the interaction tensor can be heavy when feature count grows.

# 1) Interaction tensor at tuned round (best_iter)
interaction_tensor_train <- predict(
  best_model, dtrain, predinteraction = TRUE, iterationrange = c(1, best_iter + 1)  # end is exclusive
)

# 2) Identity check on one row (features-only sum equals predcontrib SHAP)
example_row_id <- 1L
interaction_matrix_example <- interaction_tensor_train[example_row_id, voi_num, voi_num, drop = FALSE][1, , ]
shap_from_interactions_example <- rowSums(interaction_matrix_example)
stopifnot(isTRUE(all.equal(
  shap_from_interactions_example,
  shap_train[example_row_id, voi_num],
  tolerance = 1e-6
)))

# 3) Weighted mean |interaction| per pair over training rows
interaction_avg_matrix_full <- apply(
  interaction_tensor_train[, voi_num, voi_num, drop = FALSE],
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
    Var1 = factor(Var1, levels = voi_num),
    Var2 = factor(Var2, levels = voi_num)
  ) |>
  dplyr::filter(!is.na(Freq)) |>
  dplyr::arrange(dplyr::desc(Freq)) |>
  dplyr::rename(feature_i = Var1, feature_j = Var2, mean_abs_interaction = Freq)

as.data.frame(interaction_table) %>% head() %>% print(row.names=FALSE)
# > feature_i feature_j mean_abs_interaction
# >    ANXMAT   MATHEFF            1.7066509
# >   MATHEFF      ESCS            1.4805777
# >   MATHEFF  FAMSUPSL            1.2011264
# >   GROSAGR   MATHEFF            1.0112280
# >  FAMSUPSL      ESCS            0.7822594
# >   MATHEFF  CREATSCH            0.6978199

# 5) Share of interactions in overall explanation (heuristic)
main_abs_mean <- colSums(abs(shap_train[, voi_num, drop = FALSE]) * w_train, na.rm = TRUE) / sum(w_train, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (training): %.2f%%\n", 100 * interaction_share))
# > Interaction share (training): 32.06%

# 6) Heatmap
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = voi_num),
    feature_j = factor(feature_j, levels = voi_num)
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
  feature = voi_num,
  mean_abs_interaction_with_others = sapply(voi_num, function(fi) {
    partners <- setdiff(voi_num, fi)
    mean(interaction_avg_matrix_full[fi, partners], na.rm = TRUE)
  })
) |>
  dplyr::arrange(dplyr::desc(mean_abs_interaction_with_others))

as.data.frame(interaction_strength_table) %>% head() %>% print(row.names=FALSE)
# >  feature mean_abs_interaction_with_others
# >  MATHEFF                        0.2574598
# >     ESCS                        0.1393350
# > FAMSUPSL                        0.1282611
# >   ANXMAT                        0.1222698
# > MATHEF21                        0.1049778
# >   FAMCON                        0.1045247

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
# Assumes: best_model, dvalid, valid_data, w_valid, voi_num, pred_valid, shap_valid

# 1) Interaction tensor
interaction_tensor_valid <- predict(
  best_model, dvalid, predinteraction = TRUE, iterationrange = c(1, best_iter + 1)
)

# 2) Identity check
example_row_id <- 1L
interaction_matrix_example <- interaction_tensor_valid[example_row_id, voi_num, voi_num, drop = FALSE][1, , ]
shap_from_interactions_example <- rowSums(interaction_matrix_example)
stopifnot(isTRUE(all.equal(
  shap_from_interactions_example,
  shap_valid[example_row_id, voi_num],
  tolerance = 1e-6
)))

# 3) Weighted mean |interaction| per pair
interaction_avg_matrix_full <- apply(
  interaction_tensor_valid[, voi_num, voi_num, drop = FALSE],
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
    Var1 = factor(Var1, levels = voi_num),
    Var2 = factor(Var2, levels = voi_num)
  ) |>
  dplyr::filter(!is.na(Freq)) |>
  dplyr::arrange(dplyr::desc(Freq)) |>
  dplyr::rename(feature_i = Var1, feature_j = Var2, mean_abs_interaction = Freq)

as.data.frame(interaction_table) %>% head() %>% print(row.names=FALSE)
# > feature_i feature_j mean_abs_interaction
# >    ANXMAT   MATHEFF            1.6454071
# >   MATHEFF      ESCS            1.5489270
# >   MATHEFF  FAMSUPSL            1.1551721
# >   GROSAGR   MATHEFF            1.0201575
# >  FAMSUPSL      ESCS            0.8390407
# >   MATHEFF  CREATSCH            0.7513035

# 5) Share of interactions
main_abs_mean <- colSums(abs(shap_valid[, voi_num, drop = FALSE]) * w_valid, na.rm = TRUE) / sum(w_valid, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (validation): %.2f%%\n", 100 * interaction_share))
# > Interaction share (validation): 32.02%

# 6) Heatmap
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = voi_num),
    feature_j = factor(feature_j, levels = voi_num)
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
  feature = voi_num,
  mean_abs_interaction_with_others = sapply(voi_num, function(fi) {
    partners <- setdiff(voi_num, fi)
    mean(interaction_avg_matrix_full[fi, partners], na.rm = TRUE)
  })
) |>
  dplyr::arrange(dplyr::desc(mean_abs_interaction_with_others))

as.data.frame(interaction_strength_table) %>% head() %>% print(row.names=FALSE)
# >  feature mean_abs_interaction_with_others
# >  MATHEFF                        0.2576357
# >     ESCS                        0.1451176
# > FAMSUPSL                        0.1285885
# >   ANXMAT                        0.1238359
# >   FAMCON                        0.1064312
# > MATHEF21                        0.1055714

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
# Assumes: best_model, dtest, test_data, w_test, voi_num, pred_test, shap_test

# 1) Interaction tensor
interaction_tensor_test <- predict(
  best_model, dtest, predinteraction = TRUE, iterationrange = c(1, best_iter + 1)
)

# 2) Identity check
example_row_id <- 1L
interaction_matrix_example <- interaction_tensor_test[example_row_id, voi_num, voi_num, drop = FALSE][1, , ]
shap_from_interactions_example <- rowSums(interaction_matrix_example)
stopifnot(isTRUE(all.equal(
  shap_from_interactions_example,
  shap_test[example_row_id, voi_num],
  tolerance = 1e-6
)))

# 3) Weighted mean |interaction| per pair
interaction_avg_matrix_full <- apply(
  interaction_tensor_test[, voi_num, voi_num, drop = FALSE],
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
    Var1 = factor(Var1, levels = voi_num),
    Var2 = factor(Var2, levels = voi_num)
  ) |>
  dplyr::filter(!is.na(Freq)) |>
  dplyr::arrange(dplyr::desc(Freq)) |>
  dplyr::rename(feature_i = Var1, feature_j = Var2, mean_abs_interaction = Freq)

as.data.frame(interaction_table) %>% head() %>% print(row.names=FALSE)
# > feature_i feature_j mean_abs_interaction
# >    ANXMAT   MATHEFF            1.7392153
# >   MATHEFF      ESCS            1.4692478
# >   MATHEFF  FAMSUPSL            1.1971104
# >   GROSAGR   MATHEFF            0.9878994
# >  FAMSUPSL      ESCS            0.7903515
# >   MATHEFF  CREATSCH            0.7229085

# 5) Share of interactions
main_abs_mean <- colSums(abs(shap_test[, voi_num, drop = FALSE]) * w_test, na.rm = TRUE) / sum(w_test, na.rm = TRUE)
total_main   <- sum(main_abs_mean, na.rm = TRUE)
total_pairs  <- sum(interaction_table$mean_abs_interaction, na.rm = TRUE)
interaction_share <- total_pairs / (total_main + total_pairs)
cat(sprintf("Interaction share (test): %.2f%%\n", 100 * interaction_share))
# > Interaction share (test): 31.84%

# 6) Heatmap
interaction_heat_df <- tibble::as_tibble(interaction_avg_matrix_masked, rownames = "feature_i") |>
  tidyr::pivot_longer(-feature_i, names_to = "feature_j", values_to = "mean_abs_interaction") |>
  dplyr::mutate(
    feature_i = factor(feature_i, levels = voi_num),
    feature_j = factor(feature_j, levels = voi_num)
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
  feature = voi_num,
  mean_abs_interaction_with_others = sapply(voi_num, function(fi) {
    partners <- setdiff(voi_num, fi)
    mean(interaction_avg_matrix_full[fi, partners], na.rm = TRUE)
  })
) |>
  dplyr::arrange(dplyr::desc(mean_abs_interaction_with_others))

as.data.frame(interaction_strength_table) %>% head() %>% print(row.names=FALSE)
# >  feature mean_abs_interaction_with_others
# >  MATHEFF                        0.2560069
# >     ESCS                        0.1377430
# > FAMSUPSL                        0.1278928
# >   ANXMAT                        0.1263353
# > MATHEF21                        0.1066722
# >   FAMCON                        0.1050001

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
  w_vec = w_train, features = voi_num, split = "training"
)

shap_int_valid <- compute_shap_interactions(
  model = best_model, dmat = dvalid, data_frame = valid_data,
  w_vec = w_valid, features = voi_num, split = "validation"
)

shap_int_test <- compute_shap_interactions(
  model = best_model, dmat = dtest, data_frame = test_data,
  w_vec = w_test, features = voi_num, split = "test"
)

# --- Quick access & examples ---

# Tables / shares
shap_int_train$interaction_table |> as.data.frame() |> head() |>print(row.names = FALSE)
# > feature_i feature_j mean_abs_interaction
# >    ANXMAT   MATHEFF            1.7066509
# >   MATHEFF      ESCS            1.4805777
# >   MATHEFF  FAMSUPSL            1.2011264
# >   GROSAGR   MATHEFF            1.0112280
# >  FAMSUPSL      ESCS            0.7822594
# >   MATHEFF  CREATSCH            0.6978199
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_train$split, 100 * shap_int_train$interaction_share))
# > Interaction share (training): 32.06%
as.data.frame(shap_int_train$interaction_strength_table)|> head() |>print(row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# >  MATHEFF                        0.2574598
# >     ESCS                        0.1393350
# > FAMSUPSL                        0.1282611
# >   ANXMAT                        0.1222698
# > MATHEF21                        0.1049778
# >   FAMCON                        0.1045247

shap_int_valid$interaction_table |> as.data.frame() |> head() |> print(row.names = FALSE)
# >  feature_i feature_j mean_abs_interaction
# >     ANXMAT   MATHEFF            1.6454071
# >    MATHEFF      ESCS            1.5489270
# >    MATHEFF  FAMSUPSL            1.1551721
# >    GROSAGR   MATHEFF            1.0201575
# >   FAMSUPSL      ESCS            0.8390407
# >    MATHEFF  CREATSCH            0.751303
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_valid$split, 100 * shap_int_valid$interaction_share))
# > Interaction share (validation): 32.02%
shap_int_valid$interaction_strength_table |> as.data.frame() |> head() |> print(row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# >  MATHEFF                        0.2576357
# >     ESCS                        0.1451176
# > FAMSUPSL                        0.1285885
# >   ANXMAT                        0.1238359
# >   FAMCON                        0.1064312
# > MATHEF21                        0.1055714

shap_int_test$interaction_table  |> as.data.frame() |> head() |> print(row.names = FALSE)
# > feature_i feature_j mean_abs_interaction
# >    ANXMAT   MATHEFF            1.7392153
# >   MATHEFF      ESCS            1.4692478
# >   MATHEFF  FAMSUPSL            1.1971104
# >   GROSAGR   MATHEFF            0.9878994
# >  FAMSUPSL      ESCS            0.7903515
# >   MATHEFF  CREATSCH            0.7229085
cat(sprintf("Interaction share (%s): %.2f%%\n",
            shap_int_test$split, 100 * shap_int_test$interaction_share))
# > Interaction share (test): 31.84%
shap_int_test$interaction_strength_table  |> as.data.frame() |> head() |> print(row.names = FALSE)
# >  feature mean_abs_interaction_with_others
# >  MATHEFF                        0.2560069
# >     ESCS                        0.1377430
# > FAMSUPSL                        0.1278928
# >   ANXMAT                        0.1263353
# > MATHEF21                        0.1066722
# >   FAMCON                        0.1050001

# Heatmaps
plot_interaction_heatmap(shap_int_train)
plot_interaction_heatmap(shap_int_valid)
plot_interaction_heatmap(shap_int_test)

# Per-feature interaction strength
plot_interaction_strength(shap_int_train)
plot_interaction_strength(shap_int_valid)
plot_interaction_strength(shap_int_test)

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
      max_depth = best_params$max_depth,               # best_params$max_depth = 4
      eta = best_params$eta,                           # best_params$eta = 0.05
      eval_metric = "rmse",                            
      nthread = max(1, parallel::detectCores() - 1) 
    ),
    data = dtrain,
    nrounds = best_iter,                               # best_iter = 174
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
# > Fitting main models: 29.9 sec elapsed

main_models[[1]]             # Inspect first model: mod object, r2, and importance
main_models[[1]]$formula
main_models[[1]]$importance %>% head() %>% print(row.names=FALSE)
# >     Feature      Gain      Cover  Frequency
# >      <char>     <num>      <num>      <num>
# >    MATHEFF 0.41441651 0.08647937 0.04531371
# >       ESCS 0.05677214 0.04661289 0.04337723
# >   FAMSUPSL 0.04910481 0.04670405 0.04608830
# >     ANXMAT 0.04204422 0.03567853 0.02982184
# >     FAMCON 0.02738327 0.04214522 0.03214562
# >   MATHEF21 0.02449374 0.03656461 0.02478699
main_models[[1]]$importance$Feature
main_models[[2]]$importance$Feature

main_models[[1]]$mod$evaluation_log
which.min(main_models[[1]]$mod$evaluation_log$eval_rmse)
# > [1] 174
min(main_models[[1]]$mod$evaluation_log$eval_rmse)
# > 65.70228
which.min(main_models[[2]]$mod$evaluation_log$eval_rmse)
# > [1] 170
min(main_models[[2]]$mod$evaluation_log$eval_rmse)
# > 68.04456
which.min(main_models[[3]]$mod$evaluation_log$eval_rmse)
# > [1] 173
min(main_models[[3]]$mod$evaluation_log$eval_rmse)
# > 65.49614
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
# >    pv       best_nround best_eval_rmse
# >    <chr>          <int>          <dbl>
# >  1 PV1MATH          174           65.7
# >  2 PV2MATH          170           68.0
# >  3 PV3MATH          173           65.5
# >  4 PV4MATH          169           67.1
# >  5 PV5MATH          161           66.4
# >  6 PV6MATH          167           67.8
# >  7 PV7MATH          152           69.1
# >  8 PV8MATH          172           65.9
# >  9 PV9MATH          140           67.8
# > 10 PV10MATH         128           70.3

# # --- Variable Importance (Use blocks below instead with NAs removed) ---
# 
# # Gain-based Variable Importance Matrix (10 PVs × p predictors) 
# main_importance_matrix <- do.call(rbind, lapply(main_models, function(m) {
#   # Extract Gain column from xgb.importance()
#   main_importance <- m$importance
#   setNames(main_importance$Gain, main_importance$Feature)[voi_num]  # Ensure correct column order and all variables present
# }))
# dim(main_importance_matrix)  # 10x53
# main_importance_matrix[, 1: 6]
# # # >            MATHMOT     MATHEASE     MATHPREF   EXERPRAC     STUDYHMW    WORKPAY
# # # >   [1,] 7.584888e-05 3.566348e-07 1.976773e-04 0.01717184 0.0009849754 0.02165380
# # # >   [2,] 1.560673e-05 8.553920e-06 5.623301e-04 0.01998682 0.0010814902 0.02405621
# # # >   [3,] 9.743285e-05 7.098876e-05 1.847467e-04 0.01482241 0.0011579780 0.03190271
# # # >   [4,] 4.617287e-04 9.305535e-06 1.033502e-04 0.02099655 0.0018839175 0.02958295
# # # >   [5,]           NA 2.973043e-05 1.884637e-04 0.01999291 0.0012142028 0.02752677
# # # >   [6,] 2.189407e-06 2.870510e-04 1.045956e-04 0.01695075 0.0005205074 0.02543347
# # # >   [7,] 9.313741e-05 3.126565e-06 3.781261e-04 0.02061056 0.0023916772 0.02758754
# # # >   [8,] 1.302493e-04 2.984991e-04 4.593591e-04 0.01836087 0.0008947926 0.03115778
# # # >   [9,]           NA 3.529295e-04 7.721674e-06 0.02069801 0.0004999667 0.02468672
# # # >  [10,] 2.584999e-05 1.842690e-04 2.158928e-04 0.01909483 0.0007276578 0.03302935
# 
# # Mean of gain importance across PVs
# main_importance <- colMeans(main_importance_matrix)
# main_importance[1: 6]
# # >  MATHMOT     MATHEASE     MATHPREF     EXERPRAC     STUDYHMW      WORKPAY 
# # >       NA 0.0001244810 0.0002402263 0.0188685563 0.0011357166 0.0276617273 
# # colMeans(main_importance_matrix, na.rm=TRUE)[1: 6]
# # # >      MATHMOT     MATHEASE     MATHPREF     EXERPRAC     STUDYHMW      WORKPAY 
# # # > 0.0001127554 0.0001244810 0.0002402263 0.0188685563 0.0011357166 0.0276617273 

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
# >       MATHMOT     MATHEASE     MATHPREF     EXERPRAC     STUDYHMW      WORKPAY 
# >  9.020433e-05 1.244810e-04 2.402263e-04 1.886856e-02 1.135717e-03 2.766173e-02 

# Display ranked importance
tibble(
  Variable = names(main_importance),
  Importance = main_importance
) |> 
  arrange(desc(Importance)) |> 
  head() 
# > # A tibble: 6 × 2
# > Variable Importance
# > <chr>         <dbl>
# > 1 MATHEFF      0.419 
# > 2 ESCS         0.0584
# > 3 FAMSUPSL     0.0502
# > 4 ANXMAT       0.0443
# > 5 FAMCON       0.0288
# > 6 WORKPAY      0.0277

# --- Estimates of Manual weighted R² (XGBoost models on training data) ---

main_r2s_weighted <- sapply(1:M, function(i) {
  pv <- pvmaths[i]
  model <- main_models[[i]]$mod
  
  # Extract weighted true values and predictions on training data
  y_train <- train_data[[pv]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, voi_num]
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
# > [1] 0.7018320 0.7022603 0.7023879 0.7020573 0.7021623 0.7094153 0.7023259 0.7075678 0.7058621 0.7099261

# Final Rubin's Step 2: mean of R² across plausible values
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.7045797

# --- Use helper function for all five metrics ---
main_metrics <- sapply(1:M, function(i) {
  pv <- pvmaths[i]
  model <- main_models[[i]]$mod
  
  y_train <- train_data[[pv]]
  w <- train_data[[final_wt]]
  X_train <- train_data[, voi_num]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  y_pred <- predict(model, dtrain)
  
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
main_metrics
# >         mse     rmse      mae        bias bias_pct        r2
# > 1  2432.539 49.32077 38.75174 -0.06843546 1.282886 0.7018320
# > 2  2378.543 48.77031 38.41342 -0.06837244 1.250991 0.7022603
# > 3  2417.668 49.16979 38.51621 -0.06884021 1.284243 0.7023879
# > 4  2361.339 48.59361 38.17702 -0.06838279 1.229079 0.7020573
# > 5  2351.991 48.49732 37.95125 -0.06870954 1.221235 0.7021623
# > 6  2298.216 47.93971 37.65380 -0.06874054 1.219296 0.7094153
# > 7  2334.182 48.31337 38.04026 -0.06833554 1.230440 0.7023259
# > 8  2364.110 48.62211 37.88254 -0.06833177 1.255071 0.7075678
# > 9  2332.962 48.30075 37.81824 -0.06865880 1.229479 0.7058621
# > 10 2283.077 47.78155 37.67906 -0.06829058 1.187406 0.7099261

### ---- Replicate models using BRR replicate weights ----

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
        max_depth = best_params$max_depth,             # best_params$max_depth = 4
        eta = best_params$eta,                         # best_params$eta = 0.05
        eval_metric = "rmse",
        nthread = max(1, parallel::detectCores() - 1)
      ),
      data = dtrain,
      nrounds = best_iter,                             # best_iter = 174   
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
# > Fitting replicate models: 2312.963 sec elapsed

replicate_models[[1]][[1]]$mod        
replicate_models[[1]][[1]]$formula    
replicate_models[[1]][[1]]$importance 

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
  y_train <- train_data[[pv]]
  
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w <- train_data[[rep_wts[g]]]  # Replicate weight g
    
    X_train <- train_data[, voi_num]
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
# > [1] 0.001471351

imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
imputation_var_r2_weighted
# > [1] 1.083988e-05

var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
var_final_r2_weighted
# > [1] 0.001483275

se_final_r2_weighted <- sqrt(var_final_r2_weighted)
se_final_r2_weighted
# > [1] 0.03851331

#### ---- Final Outputs ----

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
# >               Metric  Estimate Std. Error
# > R-squared (Weighted) 0.7045797 0.03851331

as.data.frame(importance_table) %>% head() %>% print(row.names = FALSE)
# > Variable Importance  Std. Error         CV
# >  MATHEFF 0.41920640 0.027587986 0.06581003
# >     ESCS 0.05840687 0.010606521 0.18159715
# > FAMSUPSL 0.05019359 0.009280770 0.18489952
# >   ANXMAT 0.04430813 0.009196541 0.20755876
# >   FAMCON 0.02878037 0.008306762 0.28862596
# >  WORKPAY 0.02766173 0.007705758 0.27857109

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
  X_train <- train_data[, voi_num]
  dtrain <- xgb.DMatrix(data = as.matrix(X_train))
  y_pred <- predict(model, dtrain)
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()

train_metrics_main
# >         mse     rmse      mae        bias bias_pct        r2
# > 1  2432.539 49.32077 38.75174 -0.06843546 1.282886 0.7018320
# > 2  2378.543 48.77031 38.41342 -0.06837244 1.250991 0.7022603
# > 3  2417.668 49.16979 38.51621 -0.06884021 1.284243 0.7023879
# > 4  2361.339 48.59361 38.17702 -0.06838279 1.229079 0.7020573
# > 5  2351.991 48.49732 37.95125 -0.06870954 1.221235 0.7021623
# > 6  2298.216 47.93971 37.65380 -0.06874054 1.219296 0.7094153
# > 7  2334.182 48.31337 38.04026 -0.06833554 1.230440 0.7023259
# > 8  2364.110 48.62211 37.88254 -0.06833177 1.255071 0.7075678
# > 9  2332.962 48.30075 37.81824 -0.06865880 1.229479 0.7058621
# > 10 2283.077 47.78155 37.67906 -0.06829058 1.187406 0.7099261
train_metrics_main$r2   # = main_r2s_weighted
# > [1] 0.7018320 0.7022603 0.7023879 0.7020573 0.7021623 0.7094153 0.7023259 0.7075678 0.7058621 0.7099261

train_metric_main <- colMeans(train_metrics_main)
train_metric_main 
# >           mse          rmse           mae          bias      bias_pct            r2 
# > 2355.46264704   48.53093073   38.08835184   -0.06850977    1.23901255    0.70457970 
train_metric_main["r2"]  # = main_r2_weighted
# >        r2 
# > 0.7045797 

# --- Replicate predictions for training data ---
tic("Computing train_metrics_replicates")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- train_data[[pvmaths[m]]]
    w <- train_data[[rep_wts[g]]]
    X_train <- train_data[, voi_num]
    dtrain <- xgb.DMatrix(data = as.matrix(X_train))
    y_pred <- predict(model, dtrain)
    compute_metrics(y_train, y_pred, w)
  }) |> t()
}) # a list of M=10 matrices, each of shape 80x5
toc()
# > Computing train_metrics_replicates: 7.827 sec elapsed
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
# > mse      1.060275e+05 9.689248e+04 1.022044e+05 1.194806e+05 8.555716e+04 8.049543e+04 7.966986e+04 1.281105e+05 7.687450e+04 8.087372e+04
# > rmse     1.129714e+01 1.054968e+01 1.095374e+01 1.315461e+01 9.408213e+00 9.052355e+00 8.819419e+00 1.410445e+01 8.512249e+00 9.162851e+00
# > mae      1.103456e+01 1.134464e+01 9.583228e+00 1.186807e+01 9.055373e+00 1.001859e+01 8.864692e+00 1.109570e+01 8.819973e+00 8.961098e+00
# > bias     4.681968e-07 6.839055e-07 5.511313e-07 5.046722e-07 4.869864e-07 4.614754e-07 4.097870e-07 6.535720e-07 3.126112e-07 9.816318e-07
# > bias_pct 1.365737e-02 1.265916e-02 1.697513e-02 1.687751e-02 1.035210e-02 1.302056e-02 1.260556e-02 1.796277e-02 1.057038e-02 1.060970e-02
# > r2       1.560059e-03 1.518060e-03 1.498695e-03 1.890625e-03 1.320034e-03 1.238217e-03 1.268393e-03 1.942122e-03 1.188393e-03 1.288912e-03

# <=> Equivalent codes
# sampling_var_matrix_train <- sapply(1:M, function(m) {
#   sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
#     colMeans() / (1 - k)^2
# })
# sampling_var_matrix_train

# Debugged and check consistency
sampling_var_matrix_train["r2", ]
# > [1] 0.001560059 0.001518060 0.001498695 0.001890625 0.001320034 0.001238217 0.001268393 0.001942122 0.001188393 0.001288912
sampling_var_r2s_weighted
# > [1] 0.001560059 0.001518060 0.001498695 0.001890625 0.001320034 0.001238217 0.001268393 0.001942122 0.001188393 0.001288912
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
sampling_var_r2_weighted
# > [1] 0.001471351

sampling_var_train <- rowMeans(sampling_var_matrix_train)
sampling_var_train                                           # sampling_var_train['r2'] = sampling_var_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 9.561861e+04 1.050147e+01 1.006459e+01 5.513970e-07 1.352902e-02 1.471351e-03 

imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
imputation_var_train                                              # imputation_var_train ['r2'] = imputation_var_r2_weighted 
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 2.215597e+03 2.348991e-01 1.366074e-01 4.168326e-08 8.892283e-04 1.083988e-05

var_final_train <- sampling_var_train + (1 + 1/M) * imputation_var_train
var_final_train                                              # var_final_train ['r2'] = var_final_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 9.805577e+04 1.075986e+01 1.021486e+01 5.972485e-07 1.450718e-02 1.483275e-03 

se_final_train <- sqrt(var_final_train)
se_final_train                                         # se_final_train ['r2'] = se_final_r2_weighted
# >          mse         rmse          mae         bias     bias_pct           r2 
# > 3.131386e+02 3.280222e+00 3.196069e+00 7.728186e-04 1.204457e-01 3.851331e-02

# Confidence intervals
ci_lower_train <- train_metric_main - z_crit * se_final_train
ci_upper_train <- train_metric_main + z_crit * se_final_train
ci_length_train <- ci_upper_train - ci_lower_train

train_eval <- tibble(
  Metric = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train, scientific = FALSE),
  CI_lower = format(ci_lower_train, scientific = FALSE),
  CI_upper = format(ci_upper_train, scientific = FALSE),
  CI_length = format(ci_length_train, scientific = FALSE)
)

print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error       CI_lower       CI_upper       CI_length
# >       MSE  2355.46264704 313.1385795953  1741.72230887  2969.20298522  1227.480676354
# >      RMSE    48.53093073   3.2802224880    42.10181280    54.96004867    12.858235875
# >       MAE    38.08835184   3.1960694412    31.82417085    44.35253284    12.528361994
# >      Bias    -0.06850977   0.0007728186    -0.07002446    -0.06699507     0.003029393
# >     Bias%     1.23901255   0.1204457359     1.00294325     1.47508186     0.472138609
# > R-squared     0.70457970   0.0385133087     0.62909500     0.78006440     0.150969396

### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  X_valid <- valid_data[, voi_num]
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
    X_valid <- valid_data[, voi_num]
    dvalid <- xgb.DMatrix(data = as.matrix(X_valid))
    y_pred <- predict(model, dvalid)
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates: 1.879 sec elapsed

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
# >    Metric Point_estimate Standard_error      CI_lower     CI_upper       CI_length
# >       MSE   4548.6336829    427.4422979 3710.86217342 5386.4051923    1675.5430189
# >      RMSE     67.4268885      3.1645392   61.22450567   73.6292714      12.4047657
# >       MAE     53.6141858      2.7468293   48.23049925   58.9978723      10.7673731
# >      Bias     -0.5703835      3.8271589   -8.07147698    6.9307100      15.0021870
# >     Bias%      1.5872118      0.7865029    0.04569446    3.1287291       3.0830346
# > R-squared      0.3870472      0.0486093    0.29177478    0.4823197       0.1905449

### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  X_test <- test_data[, voi_num]
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
    X_test <- test_data[, voi_num]
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
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4253.1779655   449.56392552 3372.0488628 5134.3070683 1762.2582055
# >      RMSE     65.1932205     3.42770879   58.4750347   71.9114062   13.4363715
# >       MAE     51.6006199     2.94641602   45.8257506   57.3754892   11.5497386
# >      Bias     -4.6787194     3.86196706  -12.2480357    2.8905970   15.1386327
# >     Bias%      0.7944627     0.74135935   -0.6585750    2.2475003    2.9060753
# > R-squared      0.5155909     0.04213463    0.4330085    0.5981732    0.1651647

### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process to avoid redundancy.

# Evaluation function 
evaluate_split <- function(split_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, voi_num, pvmaths) {
  
  # Main plausible values loop
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    y_train <- split_data[[pvmaths[i]]]
    w <- split_data[[final_wt]]
    features <- split_data[, voi_num]
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
      features <- split_data[, voi_num]
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
train_eval <- evaluate_split(train_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, voi_num, pvmaths)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, voi_num, pvmaths)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, voi_num, pvmaths)

# Display
print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

### ---- Summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error       CI_lower       CI_upper       CI_length
# >       MSE  2355.46264704 313.1385795953  1741.72230887  2969.20298522  1227.480676354
# >      RMSE    48.53093073   3.2802224880    42.10181280    54.96004867    12.858235875
# >       MAE    38.08835184   3.1960694412    31.82417085    44.35253284    12.528361994
# >      Bias    -0.06850977   0.0007728186    -0.07002446    -0.06699507     0.003029393
# >     Bias%     1.23901255   0.1204457359     1.00294325     1.47508186     0.472138609
# > R-squared     0.70457970   0.0385133087     0.62909500     0.78006440     0.150969396

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error      CI_lower     CI_upper       CI_length
# >       MSE   4548.6336829    427.4422979 3710.86217342 5386.4051923    1675.5430189
# >      RMSE     67.4268885      3.1645392   61.22450567   73.6292714      12.4047657
# >       MAE     53.6141858      2.7468293   48.23049925   58.9978723      10.7673731
# >      Bias     -0.5703835      3.8271589   -8.07147698    6.9307100      15.0021870
# >     Bias%      1.5872118      0.7865029    0.04569446    3.1287291       3.0830346
# > R-squared      0.3870472      0.0486093    0.29177478    0.4823197       0.1905449

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4253.1779655   449.56392552 3372.0488628 5134.3070683 1762.2582055
# >      RMSE     65.1932205     3.42770879   58.4750347   71.9114062   13.4363715
# >       MAE     51.6006199     2.94641602   45.8257506   57.3754892   11.5497386
# >      Bias     -4.6787194     3.86196706  -12.2480357    2.8905970   15.1386327
# >     Bias%      0.7944627     0.74135935   -0.6585750    2.2475003    2.9060753
# > R-squared      0.5155909     0.04213463    0.4330085    0.5981732    0.1651647

