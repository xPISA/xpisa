# ---- I. Predictive Modelling: Version 1.4 ----

# Fit with cv.glmnet: alpha α (=1, default), lambda λ (grid), with manual K-folds + tracks out-of-fold (OOF) predictions/metrics.
#
# Source codes: cv.glmnet, getAnywhere("cv.glmnet.raw"), getAnywhere("cvstats"), getAnywhere("cvcompute"), getAnywhere(getOptcv.glmnet)

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

# Load data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE)
dim(pisa_2022_student_canada)   # 23073 x 1278

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
oos <- c("EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME")  

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)    # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                  # Final student weight

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

# Small helper: weighted standard deviation
wsd <- function(x, w) {
  mu <- sum(w * x) / sum(w)
  sqrt( sum(w * (x - mu)^2) / sum(w) )
}

# Small helper: calculate weighted RMSE
w_rmse <- function(y_true, y_pred, w) sqrt(sum(w * (y_pred - y_true)^2) / sum(w))

### ---- Random Train/Validation/Test (70/15/15) split ----
set.seed(123)          # Ensure reproducibility
n <- nrow(temp_data)   # 20003
indices <- sample(n)   # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.70 * n)         # 14002
n_valid <- floor(0.15 * n)         # 3000
n_test  <- n - n_train - n_valid   # 3001
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
# > n              14002.00  3000.00  3001.00
# > w_mean            15.83    15.62    15.89
# > w_sum         221628.31 46871.57 47689.14
# > w_min              1.05     1.05     1.05
# > w_max            833.47   833.47    88.54
# > w_sd              15.42    20.24    13.90
# > n_schools        831.00   761.00   764.00
# > pv1math_mean     502.36   504.08   502.59
# > pv1math_sd        93.61    90.42    94.62
# > exerprac_mean      4.34     4.52     4.43
# > exerprac_sd        3.36     3.35     3.39




## ---- Main model using final student weights (W_FSTUWT) ---- 


### ---- Fit Elastic Net for PV1MATH only: glmnet.cv ----

# Target plausible value
pv1math <- pvmaths[1]

# Training data
X_train <- as.matrix(train_data[, oos])
y_train <- train_data[[pv1math]]
w_train <- train_data[[final_wt]]

# Validation data 
X_valid <- as.matrix(valid_data[, oos])
y_valid <- valid_data[[pv1math]]
w_valid <- valid_data[[final_wt]]

# Test data 
X_test <- as.matrix(test_data[, oos])
y_test <- test_data[[pv1math]]
w_test <- test_data[[final_wt]]

# Parallel backend (for cv.glmnet parallelization)
registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

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
# > 44064.98 45527.15 43272.60 44609.90 44153.68 
table(foldid)
# > foldid
# >    1    2    3    4    5 
# > 2800 2800 2801 2800 2801 

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

# Outcome distribution by fold
print(do.call(rbind, tapply(y_train, foldid, function(v) c(mean = mean(v), sd = sd(v)))))
# >       mean       sd
# > 1 489.4925 92.81556
# > 2 492.0924 93.06908
# > 3 489.7786 90.84370
# > 4 492.2019 92.46838
# > 5 490.3393 92.04746

#### ---- 1) Fit: grouped = TRUE (Default), keep = TRUE ----
# Timing cvfit
tic("Fitting with cv.glmnet")
# Elastic net path with default α = 1 (lasso)
set.seed(123)
cvmod1 <- cv.glmnet(                                         # cv.glmnet, ?cvglmnet
  x = X_train,
  y = y_train,
  weights = w_train,
  type.measure = "mse",                                      # loss to use for cv
  foldid = foldid,                                           # 5-fold cv on TRAIN
  grouped = TRUE,         # <-                               # Default: TRUE
  keep = TRUE,            # <-                               # Return the out‑of‑fold (OOF) predictions (“prevalidated array”); Default; FALSE
  parallel = TRUE,                                           # Enable parallel computing to speed up the computation process
  family = "gaussian",                                       # For regression
  trace.it = 1
)
toc()
# > Fitting with cv.glmnet: 0.878 sec elapsed

# # --- (Optional) An equivalent version with more details---
# set.seed(123)
# cvfit <- cv.glmnet(                                          # cv.glmnet, ?cvglmnet
#   x = X_train, 
#   y = y_train, 
#   weights = w_train, 
#   lambda = NULL,                                             # Default
#   type.measure = "mse",                                      # The default is type.measure="deviance", which uses squared-error for gaussian models (a.k.a type.measure="mse" there); RMSE = sqrt(MSE); c("default", "mse", "deviance", "class", "auc", "mae", "C")
#   foldid = foldid,                                           # prebuilt 5-fold split on TRAIN
#   alignment = c("lambda", "fraction"),                       # Default: lambda - the lambda values from the master fit (on all the data) are used to line up the predictions from each of the folds.
#   grouped = TRUE,                                            # Default
#   keep = TRUE,                                               # Default: FALSE
#   parallel = TRUE,                                           # Default: FALSE; Enable parallel computing to speed up the computation process
#   trace.it = 1,                                              # Default: 0
#   # Arguments that can be passed to glmnet ...
#   alpha = 1,                                                 # Default
#   family = "gaussian",                                       # For regression
#   #nlambda = 100,
#   #lambda.min.ratio = ifelse(nobs < nvars, 0.01, 1e-04),     # nobs <- nrow(X_train); nvars <- ncol(X_train)
#   standardize = TRUE,                                        # Default
#   intercept   = TRUE                                         # Default
# )
# 
# # Sanity check
# all.equal(cvmod1$lambda,     cvfit$lambda)
# # > [1] TRUE
# all.equal(cvmod1$cvm,        cvfit$cvm)
# # > [1] TRUE
# all.equal(cvmod1$cvsd,       cvfit$cvsd)
# # > [1] TRUE
# all.equal(cvmod1$lambda.min, cvfit$lambda.min)
# # > [1] TRUE
# all.equal(cvmod1$lambda.1se, cvfit$lambda.1se)
# # > [1] TRUE

##### ---- Explore the first cvfit ----

# Check object class
class(cvmod1)  # cv.glmnet object

# Check str
str(cvmod1)

# Print summary 
cvmod1
print(cvmod1)

# lambda, lambda.min and lambda.1se
cvmod1$lambda          # the values of lambda used in the fits.
length(cvmod1$lambda)  # n_lambda
# > [1] 62
cvmod1$lambda.min      # value of lambda that gives minimum cvm.
# > [1] 0.06714323
cvmod1$lambda.1se      # largest value of lambda such that error is within 1 standard error of the minimum.
# > [1] 4.025131
cvmod1$index           # a one column matrix with the indices of lambda.min and lambda.1se in the sequence of coefficients, fits etc.
# >     Lambda
# > min     62
# > 1se     18

# Other values
cvmod1$cvm           # The mean cross-validated error - a vector of length length(lambda).
# >  [1] 8759.018 8699.356 8645.501 8600.807 8563.717 8532.938 8507.399 8486.208 8468.626 8453.915 8439.830 8425.991 8413.812
# > [14] 8402.898 8392.577 8383.090 8374.972 8368.111 8361.908 8356.196 8349.949 8343.847 8338.184 8333.415 8329.461 8326.180
# > [27] 8323.460 8321.204 8319.333 8317.783 8316.498 8315.433 8314.550 8313.819 8313.214 8312.713 8312.299 8311.956 8311.671
# > [40] 8311.436 8311.241 8311.081 8310.948 8310.838 8310.748 8310.673 8310.612 8310.561 8310.520 8310.487 8310.459 8310.435
# > [53] 8310.416 8310.401 8310.388 8310.378 8310.370 8310.363 8310.358 8310.353 8310.350 8310.341
cvmod1$cvsd          # estimate of standard error of cvm.
# >  [1] 59.09483 59.64193 59.20523 58.80475 58.44815 58.13876 57.87683 57.66046 57.48636 57.41726 57.30197 57.14358 57.09166
# > [14] 56.97637 56.70378 56.88515 57.19206 57.47850 58.03696 58.38285 58.98277 59.14969 58.95580 58.76086 58.60547 58.48261
# > [27] 58.38652 58.31236 58.25604 58.21420 58.18400 58.16314 58.14969 58.14208 58.13903 58.13943 58.14412 58.14925 58.15564
# > [40] 58.16302 58.17104 58.17944 58.18799 58.19653 58.20494 58.21312 58.22100 58.22855 58.23572 58.24271 58.24885 58.25480
# > [53] 58.26040 58.26562 58.27047 58.27498 58.27915 58.28301 58.28658 58.28987 58.29289 58.29415
cvmod1$cvup          # upper curve = cvm+cvsd.
cvmod1$cvlo          # lower curve = cvm-cvsd.
cvmod1$nzero         # number of non-zero coefficients at each lambda.
cvmod1$name          # a text string indicating type of measure (for plotting purposes).
cvmod1$glmnet.fit    # a fitted glmnet object for the full data.
cvmod1$fit.preval    # <- keep=TRUE: this is the array of prevalidated fits (OOF).
cvmod1$foldid        # <- keep=TRUE: the fold assignments used

# Pooled weighted OOF MSE vs cvm (Fold‑weighted average of fold MSEs)
length(cvmod1$fit.preval)  # n_train x n_lambda
# > [1] 868124
all.equal(as.numeric(colSums(w_train * (y_train - cvmod1$fit.preval)^2) / sum(w_train)), as.numeric(cvmod1$cvm))
# > TRUE

# Coefs at lambda.min and lambda.1se
coef(cvmod1, s = "lambda.min")
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             lambda.min
# > (Intercept) 521.341812
# > EXERPRAC     -1.857451
# > STUDYHMW      1.919735
# > WORKPAY      -6.469187
# > WORKHOME     -1.821766
coef(cvmod1, s = "lambda.1se")
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             lambda.1se
# > (Intercept) 517.6976544
# > EXERPRAC     -0.7531747
# > STUDYHMW      .        
# > WORKPAY      -5.4703682
# > WORKHOME     -0.5614247

# --- Inspect CV curve --- 
# Plot MSE vs - Log(λ)
plot(cvmod1)                                   
abline(v = -log(cvmod1$lambda.min), lty = 2, col = "blue")
abline(v = -log(cvmod1$lambda.1se), lty = 2, col = "red")
legend("topright", c("lambda.min","lambda.1se"), lty = 2, col = c("blue","red"), bty = "n")

plot(log(cvmod1$lambda), cvmod1$cvm , pch = 19, col = "red",
     xlab = "log(Lambda)", ylab = cvmod1$name)

# --- predict and glmnet::assess.glmnet	---
predict(object = cvmod1,                              # ?predict.cv.glmnet
        newx = X_valid,
        s = c("lambda.1se", "lambda.min"))            # Default is the value s="lambda.1se" stored on the CV object.
# <=> 
predict(object = cvmod1,        
        newx = X_valid,
        s = c("lambda.1se", "lambda.min"),            
        exact = FALSE,       
        type = "link")
all.equal(predict(object = cvmod1,                     
                  newx = X_valid,
                  s = c("lambda.1se", "lambda.min")),
          predict(object = cvmod1,        
                  newx = X_valid,
                  s = c("lambda.1se", "lambda.min"),            
                  exact = FALSE,       
                  type = "link"))
# > [1] TRUE

predict(object = cvmod1,
        newx = X_valid,
        s = c("lambda.1se", "lambda.min"),   # Default is the value s="lambda.1se" stored on the CV object.           
        exact = FALSE,         
        type = "coefficient")                # Computes the coefficients at the requested values for s
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >              lambda.1se
# > (Intercept) 517.6976544
# > EXERPRAC     -0.7531747
# > STUDYHMW      .        
# > WORKPAY      -5.4703682
# > WORKHOME     -0.5614247
predict(object = cvmod1,
        newx = X_valid,
        s = "lambda.min",                    # <-                     
        exact = FALSE,         
        type = "coefficient")  
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# > lambda.min
# > (Intercept) 521.341812
# > EXERPRAC     -1.857451
# > STUDYHMW      1.919735
# > WORKPAY      -6.469187
# > WORKHOME     -1.821766

assess.glmnet(object = predict(object = cvmod1,      
                               newx = X_valid,
                               s = c("lambda.1se", "lambda.min"),  # Default: lambda.1se            
                               exact = FALSE) , 
              newy = y_valid,
              weights = w_valid,
              family = "gaussian") 

assess.glmnet(object = predict(object = cvmod1,      
                               newx = X_valid,
                               s = "lambda.min",                   # @lambda.min  
                               exact = FALSE) , 
              newy = y_valid,
              weights = w_valid,
              family = "gaussian") 

assess.glmnet(object = cvmod1, 
              newx = X_valid,
              newy = y_valid,
              weights = w_valid,
              family = "gaussian")   # VALID MSE and MAE, at lambda.1se

assess.glmnet(object = cvmod1, 
              newx = X_valid,
              newy = y_valid,
              weights = w_valid,
              family = "gaussian",
              s = "lambda.min")      # VALID MSE and MAE, at lambda.min

all.equal(assess.glmnet(object = predict(object = cvmod1,      
                                         newx = X_valid,
                                         s = c("lambda.1se", "lambda.min"),              
                                         exact = FALSE) , 
                        newy = y_valid,
                        weights = w_valid,
                        family = "gaussian"), 
          assess.glmnet(object = cvmod1, 
                        newx = X_valid,
                        newy = y_valid,
                        weights = w_valid,
                        family = "gaussian"))
# > [1] TRUE

sqrt(assess.glmnet(object = cvmod1, 
                   newx = X_valid,
                   newy = y_valid,
                   weights = w_valid,
                   family = "gaussian",)$mse)  # RMSE, -> rmse_valid_vector

assess.glmnet(object = cvmod1, 
              newx = X_valid,
              newy = y_valid,
              weights = w_valid,
              family = "gaussian",
              s       = c(cvmod1$lambda.min, cvmod1$lambda.1se))  # VALID MSE and MAE

sqrt(assess.glmnet(object = cvmod1, 
                   newx = X_valid,
                   newy = y_valid,
                   weights = w_valid,
                   family = "gaussian",
                   s       = c(cvmod1$lambda.min, cvmod1$lambda.1se))$mse)  # RMSE, -> rmse_valid_vector

assess.glmnet(
  object  = cvmod1$glmnet.fit,   # <- not the cv wrapper
  newx    = X_valid,
  newy    = y_valid,
  weights = w_valid,
  family  = "gaussian"
)

assess.glmnet(
  object  = predict(cvmod1, newx = X_valid, s = cvmod1$lambda),  # n × nlambda matrix
  newy    = y_valid,
  weights = w_valid,
  family  = "gaussian"
)

# --- (Option 1) Visualize Training vs Validation MSE, RMSE, MAE across Lambda: cvmod1 ---
assess_train <- assess.glmnet(object = cvmod1, s = cvmod1$lambda,
                              newx = X_train, newy = y_train,
                              weights = w_train, family = "gaussian")
assess_valid <- assess.glmnet(object = cvmod1, s = cvmod1$lambda,
                              newx = X_valid, newy = y_valid,
                              weights = w_valid, family = "gaussian")

par(mfrow = c(1, 3), mar = c(5,4,3,1))

# MSE
plot(log(cvmod1$lambda), as.numeric(assess_train$mse),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MSE",
     main = "Training vs Validation MSE across Lambda",
     ylim = range(c(as.numeric(assess_train$mse), as.numeric(assess_valid$mse))))
lines(log(cvmod1$lambda), as.numeric(assess_valid$mse), lwd = 2, col = "red")
abline(v = log(cvmod1$lambda[ which.min(assess_valid$mse) ]), lty = 2)
points(log(cvmod1$lambda[ which.min(assess_valid$mse) ]), min(as.numeric(assess_valid$mse)),
       pch = 19, col = "red")
abline(v = log(cvmod1$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod1$lambda.1se), lty = 3, col = "gray40")
legend("topright", c("Training","Validation","Best λ (valid)"),
       col = c("blue","red","black"), lty = c(1,1,2), lwd = c(2,2,1),
       pch = c(NA,NA,19), bty = "n")

# RMSE  (argmin RMSE == argmin MSE)
plot(log(cvmod1$lambda), sqrt(as.numeric(assess_train$mse)),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "RMSE",
     main = "Training vs Validation RMSE across Lambda",
     ylim = range(c(sqrt(as.numeric(assess_train$mse)), sqrt(as.numeric(assess_valid$mse)))))
lines(log(cvmod1$lambda), sqrt(as.numeric(assess_valid$mse)), lwd = 2, col = "red")
abline(v = log(cvmod1$lambda[ which.min(assess_valid$mse) ]), lty = 2)
points(log(cvmod1$lambda[ which.min(assess_valid$mse) ]), sqrt(min(as.numeric(assess_valid$mse))),
       pch = 19, col = "red")
abline(v = log(cvmod1$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod1$lambda.1se), lty = 3, col = "gray40")

# MAE
plot(log(cvmod1$lambda), as.numeric(assess_train$mae),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MAE",
     main = "Training vs Validation MAE across Lambda",
     ylim = range(c(as.numeric(assess_train$mae), as.numeric(assess_valid$mae))))
lines(log(cvmod1$lambda), as.numeric(assess_valid$mae), lwd = 2, col = "red")
abline(v = log(cvmod1$lambda[ which.min(assess_valid$mae) ]), lty = 2)
points(log(cvmod1$lambda[ which.min(assess_valid$mae) ]), min(as.numeric(assess_valid$mae)),
       pch = 19, col = "red")
abline(v = log(cvmod1$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod1$lambda.1se), lty = 3, col = "gray40")

par(mfrow = c(1, 1))

# --- (Option 2) Visualize Training vs Validation MSE, RMSE, MAE across Lambda: cvmod1$glmnet.fit ---
assess_train <- assess.glmnet(
  object  = cvmod1$glmnet.fit,
  newx    = X_train,
  newy    = y_train,
  weights = w_train,
  family  = "gaussian"
)

assess_valid <- assess.glmnet(
  object  = cvmod1$glmnet.fit,
  newx    = X_valid,
  newy    = y_valid,
  weights = w_valid,
  family  = "gaussian"
)

par(mfrow = c(1, 3), mar = c(5,4,3,1))

# MSE
plot(log(cvmod1$glmnet.fit$lambda),
     as.numeric(assess_train$mse),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MSE",
     main = "Training vs Validation MSE",
     ylim = range(c(as.numeric(assess_train$mse),
                    as.numeric(assess_valid$mse))))
lines(log(cvmod1$glmnet.fit$lambda),
      as.numeric(assess_valid$mse),
      lwd = 2, col = "red")
# Best λ by validation MSE
abline(v = log(cvmod1$glmnet.fit$lambda[which.min(assess_valid$mse)]), lty = 2)
points(log(cvmod1$glmnet.fit$lambda[which.min(assess_valid$mse)]),
       as.numeric(min(assess_valid$mse)),
       pch = 19, col = "red")
# CV-picked lambdas
abline(v = log(cvmod1$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod1$lambda.1se), lty = 3, col = "gray40")
legend("topright", c("Training","Validation","Best λ (valid)"),
       col = c("blue","red","black"), lty = c(1,1,2), lwd = c(2,2,1),
       pch = c(NA,NA,19), bty = "n", cex = 0.9)

# RMSE  (argmin RMSE == argmin MSE)
plot(log(cvmod1$glmnet.fit$lambda),
     sqrt(as.numeric(assess_train$mse)),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "RMSE",
     main = "Training vs Validation RMSE",
     ylim = range(c(sqrt(as.numeric(assess_train$mse)),
                    sqrt(as.numeric(assess_valid$mse)))))
lines(log(cvmod1$glmnet.fit$lambda),
      sqrt(as.numeric(assess_valid$mse)),
      lwd = 2, col = "red")
abline(v = log(cvmod1$glmnet.fit$lambda[which.min(assess_valid$mse)]), lty = 2)
points(log(cvmod1$glmnet.fit$lambda[which.min(assess_valid$mse)]),
       sqrt(as.numeric(min(assess_valid$mse))),
       pch = 19, col = "red")
abline(v = log(cvmod1$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod1$lambda.1se), lty = 3, col = "gray40")

# MAE
plot(log(cvmod1$glmnet.fit$lambda),
     as.numeric(assess_train$mae),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MAE",
     main = "Training vs Validation MAE",
     ylim = range(c(as.numeric(assess_train$mae),
                    as.numeric(assess_valid$mae))))
lines(log(cvmod1$glmnet.fit$lambda),
      as.numeric(assess_valid$mae),
      lwd = 2, col = "red")
abline(v = log(cvmod1$glmnet.fit$lambda[which.min(assess_valid$mae)]), lty = 2)
points(log(cvmod1$glmnet.fit$lambda[which.min(assess_valid$mae)]),
       as.numeric(min(assess_valid$mae)),
       pch = 19, col = "red")
abline(v = log(cvmod1$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod1$lambda.1se), lty = 3, col = "gray40")

par(mfrow = c(1, 1))

# Sanity check
all.equal(as.numeric(assess.glmnet(object = cvmod1, s = cvmod1$lambda, newx = X_train, newy = y_train,
                                   weights = w_train, family = "gaussian")$mse), 
          as.numeric(assess.glmnet(object = cvmod1, s = cvmod1$lambda, newx = X_train, newy = y_train,
                                   weights = w_train, family = "gaussian")$mse))
# > [1] TRUE
all.equal(as.numeric(assess.glmnet(object = cvmod1, s = cvmod1$lambda, newx = X_train, newy = y_train,
                                   weights = w_train, family = "gaussian")$mae), 
          as.numeric(assess.glmnet(object = cvmod1, s = cvmod1$lambda, newx = X_train, newy = y_train,
                                   weights = w_train, family = "gaussian")$mae))
# > [1] TRUE

##### ---- Predict and evaluate performance on training/validation/test datasets ----
###### ---- cvmod1 ----

# Predict with lambda.min 
pred_train_min <- as.numeric(predict(cvmod1, newx = X_train, s = "lambda.min"))
pred_valid_min <- as.numeric(predict(cvmod1, newx = X_valid, s = "lambda.min"))
pred_test_min  <- as.numeric(predict(cvmod1, newx = X_test,  s = "lambda.min"))

metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

# Predict with lambda.1se 
pred_train_1se <- as.numeric(predict(cvmod1, newx = X_train, s = "lambda.1se"))
pred_valid_1se <- as.numeric(predict(cvmod1, newx = X_valid, s = "lambda.1se"))
pred_test_1se  <- as.numeric(predict(cvmod1, newx = X_test,  s = "lambda.1se"))

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Combine results in a table
metric_results_min <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"],     metrics_valid_min["rmse"],     metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],      metrics_valid_min["mae"],      metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"],     metrics_valid_min["bias"],     metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],       metrics_valid_min["r2"],       metrics_test_min["r2"])
)

metric_results_1se <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"],     metrics_valid_1se["rmse"],     metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],      metrics_valid_1se["mae"],      metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"],     metrics_valid_1se["bias"],     metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],       metrics_valid_1se["r2"],       metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051

print(as.data.frame(metric_results_1se), row.names = FALSE)            # <-
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.41481 73.55493 -3.269085e-13 3.664657 0.04631327
# > Validation 88.23838 70.74959 -2.525736e+00 2.901086 0.04767726
# >       Test 92.03300 74.28495 -6.772109e-01 3.577457 0.05398446

print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051
# >   Training 91.41481 73.55493 -3.269085e-13 3.664657 0.04631327
# > Validation 88.23838 70.74959 -2.525736e+00 2.901086 0.04767726
# >       Test 92.03300 74.28495 -6.772109e-01 3.577457 0.05398446

###### ---- cvmod1$glmnet.fit ----

# Predict with lambda.min 
pred_train_min <- as.numeric(predict(cvmod1$glmnet.fit, newx = X_train, s = cvmod1$lambda.min))
pred_valid_min <- as.numeric(predict(cvmod1$glmnet.fit, newx = X_valid, s = cvmod1$lambda.min))
pred_test_min  <- as.numeric(predict(cvmod1$glmnet.fit, newx = X_test, s = cvmod1$lambda.min))

metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

# Predict with lambda.1se 
pred_train_1se <- as.numeric(predict(cvmod1$glmnet.fit, newx = X_train, s = cvmod1$lambda.1se))
pred_valid_1se <- as.numeric(predict(cvmod1$glmnet.fit, newx = X_valid, s = cvmod1$lambda.1se))
pred_test_1se  <- as.numeric(predict(cvmod1$glmnet.fit, newx = X_test, s = cvmod1$lambda.1se))

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Combine results in a table
metric_results_min <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"],     metrics_valid_min["rmse"],     metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],      metrics_valid_min["mae"],      metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"],     metrics_valid_min["bias"],     metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],       metrics_valid_min["r2"],       metrics_test_min["r2"])
)

metric_results_1se <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"],     metrics_valid_1se["rmse"],     metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],      metrics_valid_1se["mae"],      metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"],     metrics_valid_1se["bias"],     metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],       metrics_valid_1se["r2"],       metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051

print(as.data.frame(metric_results_1se), row.names = FALSE)            # <-
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.41481 73.55493 -3.269085e-13 3.664657 0.04631327
# > Validation 88.23838 70.74959 -2.525736e+00 2.901086 0.04767726
# >       Test 92.03300 74.28495 -6.772109e-01 3.577457 0.05398446

print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051
# >   Training 91.41481 73.55493 -3.269085e-13 3.664657 0.04631327
# > Validation 88.23838 70.74959 -2.525736e+00 2.901086 0.04767726
# >       Test 92.03300 74.28495 -6.772109e-01 3.577457 0.05398446

###### ---- Equivalence check ----
all.equal(as.numeric(predict(cvmod1, newx = X_train, s = "lambda.min")), 
          as.numeric(predict(cvmod1$glmnet.fit, newx = X_train, s = cvmod1$lambda.min)))
# > [1] TRUE
all.equal(as.numeric(predict(cvmod1, newx = X_valid, s = "lambda.min")), 
          as.numeric(predict(cvmod1$glmnet.fit, newx = X_valid, s = cvmod1$lambda.min)))
# > [1] TRUE
all.equal(as.numeric(predict(cvmod1, newx = X_test, s = "lambda.min")), 
          as.numeric(predict(cvmod1$glmnet.fit, newx = X_test, s = cvmod1$lambda.min)))
# > [1] TRUE

#### ---- 2) Fit: grouped = FALSE, keep = TRUE ----
tic("Fitting with cv.glmnet")
set.seed(123)
cvmod2 <- cv.glmnet(                                         
  x = X_train,
  y = y_train,
  weights = w_train,
  type.measure = "mse",                                     
  foldid = foldid,                                           
  grouped = FALSE,         # <-                               
  keep = TRUE,             # <-                               
  parallel = TRUE,                                           
  family = "gaussian",                                      
  trace.it = 1
)
toc()
# > Fitting with cv.glmnet: 0.178 sec elapsed

##### ---- Explore the second cvfit ----

# Check object class
class(cvmod2)  # cv.glmnet object

# Check str
str(cvmod2)

# Print summary 
cvmod2
print(cvmod2)

# lambda, lambda.min and lambda.1se
cvmod2$lambda          # the values of lambda used in the fits.
length(cvmod2$lambda)  # n_lambda
# > [1] 62
cvmod2$lambda.min      # value of lambda that gives minimum cvm.
# > [1] 0.06714323
cvmod2$lambda.1se      # largest value of lambda such that error is within 1 standard error of the minimum.
# > [1] 5.839776
cvmod2$index           # a one column matrix with the indices of lambda.min and lambda.1se in the sequence of coefficients, fits etc.
# >     Lambda
# > min     62
# > 1se     14

# Other values
cvmod2$cvm           # The mean cross-validated error - a vector of length length(lambda).
as.numeric(cvmod2$cvm)
# >  [1] 8759.018 8699.356 8645.501 8600.807 8563.717 8532.938 8507.399 8486.208 8468.626 8453.915 8439.830 8425.991 8413.812
# > [14] 8402.898 8392.577 8383.090 8374.972 8368.111 8361.908 8356.196 8349.949 8343.847 8338.184 8333.415 8329.461 8326.180
# > [27] 8323.460 8321.204 8319.333 8317.783 8316.498 8315.433 8314.550 8313.819 8313.214 8312.713 8312.299 8311.956 8311.671
# > [40] 8311.436 8311.241 8311.081 8310.948 8310.838 8310.748 8310.673 8310.612 8310.561 8310.520 8310.487 8310.459 8310.435
# > [53] 8310.416 8310.401 8310.388 8310.378 8310.370 8310.363 8310.358 8310.353 8310.350 8310.341
cvmod2$cvsd          # estimate of standard error of cvm.
as.numeric(cvmod2$cvsd)
# >  [1] 97.72855 97.17800 96.69944 96.31638 96.01014 95.76576 95.57121 95.41685 95.29488 95.19634 95.10959 95.04156 94.98819
# > [14] 94.94526 94.90671 94.87427 94.85154 94.83666 94.82364 94.81420 94.78866 94.75405 94.71685 94.68706 94.66455 94.64783
# > [27] 94.63576 94.62739 94.62196 94.61883 94.61749 94.61751 94.61858 94.62041 94.62280 94.62560 94.62864 94.63180 94.63502
# > [40] 94.63823 94.64139 94.64446 94.64743 94.65026 94.65296 94.65551 94.65791 94.66016 94.66226 94.66439 94.66621 94.66789
# > [53] 94.66944 94.67087 94.67220 94.67342 94.67455 94.67558 94.67654 94.67741 94.67821 94.67881
cvmod2$cvup          # upper curve = cvm+cvsd.
cvmod2$cvlo          # lower curve = cvm-cvsd.
cvmod2$nzero         # number of non-zero coefficients at each lambda.
cvmod2$name          # a text string indicating type of measure (for plotting purposes).
cvmod2$glmnet.fit    # a fitted glmnet object for the full data.
cvmod2$fit.preval    # <- keep=TRUE: this is the array of prevalidated fits (OOF).
cvmod2$foldid        # <- keep=TRUE: the fold assignments used

# Pooled weighted OOF MSE vs cvm (Fold‑weighted average of fold MSEs)
length(cvmod2$fit.preval)  # n_train x n_lambda
# > [1] 868124
all.equal(as.numeric(colSums(w_train * (y_train - cvmod2$fit.preval)^2) / sum(w_train)), as.numeric(cvmod2$cvm))
# > TRUE

# Coefs at lambda.min and lambda.1se
coef(cvmod2, s = "lambda.min")
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             lambda.min
# > (Intercept) 521.341812
# > EXERPRAC     -1.857451
# > STUDYHMW      1.919735
# > WORKPAY      -6.469187
# > WORKHOME     -1.821766
coef(cvmod2, s = "lambda.1se")
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             lambda.1se
# > (Intercept) 513.5993506
# > EXERPRAC     -0.4033480
# > STUDYHMW      .        
# > WORKPAY      -4.9934235
# > WORKHOME     -0.1872444

# --- Inspect CV curve --- 
# Plot MSE vs - Log(λ)
plot(cvmod2)                                   
abline(v = -log(cvmod2$lambda.min), lty = 2, col = "blue")
abline(v = -log(cvmod2$lambda.1se), lty = 2, col = "red")
legend("topright", c("lambda.min","lambda.1se"), lty = 2, col = c("blue","red"), bty = "n")

plot(log(cvmod2$lambda), cvmod2$cvm , pch = 19, col = "red",
     xlab = "log(Lambda)", ylab = cvmod2$name)

# --- predict and glmnet::assess.glmnet	---
predict(object = cvmod2,                              # ?predict.cv.glmnet
        newx = X_valid,
        s = c("lambda.1se", "lambda.min"))            # Default is the value s="lambda.1se" stored on the CV object.
# <=> 
predict(object = cvmod2,        
        newx = X_valid,
        s = c("lambda.1se", "lambda.min"),            
        exact = FALSE,       
        type = "link")
all.equal(predict(object = cvmod2,                     
                  newx = X_valid,
                  s = c("lambda.1se", "lambda.min")),
          predict(object = cvmod2,        
                  newx = X_valid,
                  s = c("lambda.1se", "lambda.min"),            
                  exact = FALSE,       
                  type = "link"))
# > [1] TRUE

predict(object = cvmod2,
        newx = X_valid,
        s = c("lambda.1se", "lambda.min"),   # Default is the value s="lambda.1se" stored on the CV object.           
        exact = FALSE,         
        type = "coefficient")                # Computes the coefficients at the requested values for s
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >              lambda.1se
# > (Intercept) 513.5993506
# > EXERPRAC     -0.4033480
# > STUDYHMW      .        
# > WORKPAY      -4.9934235
# > WORKHOME     -0.1872444
predict(object = cvmod2,
        newx = X_valid,
        s = "lambda.min",                    # <-                     
        exact = FALSE,         
        type = "coefficient")  
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# > lambda.min
# > (Intercept) 521.341812
# > EXERPRAC     -1.857451
# > STUDYHMW      1.919735
# > WORKPAY      -6.469187
# > WORKHOME     -1.821766

assess.glmnet(object = predict(object = cvmod2,      
                               newx = X_valid,
                               s = c("lambda.1se", "lambda.min"),  # Default: lambda.1se            
                               exact = FALSE) , 
              newy = y_valid,
              weights = w_valid,
              family = "gaussian") 

assess.glmnet(object = predict(object = cvmod2,      
                               newx = X_valid,
                               s = "lambda.min",                   # @lambda.min  
                               exact = FALSE) , 
              newy = y_valid,
              weights = w_valid,
              family = "gaussian") 

assess.glmnet(object = cvmod2, 
              newx = X_valid,
              newy = y_valid,
              weights = w_valid,
              family = "gaussian")   # VALID MSE and MAE, at lambda.1se

assess.glmnet(object = cvmod2, 
              newx = X_valid,
              newy = y_valid,
              weights = w_valid,
              family = "gaussian",
              s = "lambda.min")      # VALID MSE and MAE, at lambda.min

all.equal(assess.glmnet(object = predict(object = cvmod2,      
                                         newx = X_valid,
                                         s = c("lambda.1se", "lambda.min"),              
                                         exact = FALSE) , 
                        newy = y_valid,
                        weights = w_valid,
                        family = "gaussian"), 
          assess.glmnet(object = cvmod2, 
                        newx = X_valid,
                        newy = y_valid,
                        weights = w_valid,
                        family = "gaussian"))
# > [1] TRUE

sqrt(assess.glmnet(object = cvmod2, 
                   newx = X_valid,
                   newy = y_valid,
                   weights = w_valid,
                   family = "gaussian",)$mse)  # RMSE, -> rmse_valid_vector

assess.glmnet(object = cvmod2, 
              newx = X_valid,
              newy = y_valid,
              weights = w_valid,
              family = "gaussian",
              s       = c(cvmod2$lambda.min, cvmod2$lambda.1se))  # VALID MSE and MAE

sqrt(assess.glmnet(object = cvmod2, 
                   newx = X_valid,
                   newy = y_valid,
                   weights = w_valid,
                   family = "gaussian",
                   s       = c(cvmod2$lambda.min, cvmod2$lambda.1se))$mse)  # RMSE, -> rmse_valid_vector

assess.glmnet(
  object  = cvmod2$glmnet.fit,   # <- not the cv wrapper
  newx    = X_valid,
  newy    = y_valid,
  weights = w_valid,
  family  = "gaussian"
)

assess.glmnet(
  object  = predict(cvmod2, newx = X_valid, s = cvmod2$lambda),  # n × nlambda matrix
  newy    = y_valid,
  weights = w_valid,
  family  = "gaussian"
)

# --- (Option 1) Visualize Training vs Validation MSE, RMSE, MAE across Lambda: cvmod2 ---
assess_train <- assess.glmnet(object = cvmod2, s = cvmod2$lambda,
                              newx = X_train, newy = y_train,
                              weights = w_train, family = "gaussian")
assess_valid <- assess.glmnet(object = cvmod2, s = cvmod2$lambda,
                              newx = X_valid, newy = y_valid,
                              weights = w_valid, family = "gaussian")

par(mfrow = c(1, 3), mar = c(5,4,3,1))

# MSE
plot(log(cvmod2$lambda), as.numeric(assess_train$mse),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MSE",
     main = "Training vs Validation MSE across Lambda",
     ylim = range(c(as.numeric(assess_train$mse), as.numeric(assess_valid$mse))))
lines(log(cvmod2$lambda), as.numeric(assess_valid$mse), lwd = 2, col = "red")
abline(v = log(cvmod2$lambda[ which.min(assess_valid$mse) ]), lty = 2)
points(log(cvmod2$lambda[ which.min(assess_valid$mse) ]), min(as.numeric(assess_valid$mse)),
       pch = 19, col = "red")
abline(v = log(cvmod2$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod2$lambda.1se), lty = 3, col = "gray40")
legend("topright", c("Training","Validation","Best λ (valid)"),
       col = c("blue","red","black"), lty = c(1,1,2), lwd = c(2,2,1),
       pch = c(NA,NA,19), bty = "n")

# RMSE  (argmin RMSE == argmin MSE)
plot(log(cvmod2$lambda), sqrt(as.numeric(assess_train$mse)),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "RMSE",
     main = "Training vs Validation RMSE across Lambda",
     ylim = range(c(sqrt(as.numeric(assess_train$mse)), sqrt(as.numeric(assess_valid$mse)))))
lines(log(cvmod2$lambda), sqrt(as.numeric(assess_valid$mse)), lwd = 2, col = "red")
abline(v = log(cvmod2$lambda[ which.min(assess_valid$mse) ]), lty = 2)
points(log(cvmod2$lambda[ which.min(assess_valid$mse) ]), sqrt(min(as.numeric(assess_valid$mse))),
       pch = 19, col = "red")
abline(v = log(cvmod2$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod2$lambda.1se), lty = 3, col = "gray40")

# MAE
plot(log(cvmod2$lambda), as.numeric(assess_train$mae),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MAE",
     main = "Training vs Validation MAE across Lambda",
     ylim = range(c(as.numeric(assess_train$mae), as.numeric(assess_valid$mae))))
lines(log(cvmod2$lambda), as.numeric(assess_valid$mae), lwd = 2, col = "red")
abline(v = log(cvmod2$lambda[ which.min(assess_valid$mae) ]), lty = 2)
points(log(cvmod2$lambda[ which.min(assess_valid$mae) ]), min(as.numeric(assess_valid$mae)),
       pch = 19, col = "red")
abline(v = log(cvmod2$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod2$lambda.1se), lty = 3, col = "gray40")

par(mfrow = c(1, 1))

# --- (Option 2) Visualize Training vs Validation MSE, RMSE, MAE across Lambda: cvmod2$glmnet.fit ---
assess_train <- assess.glmnet(
  object  = cvmod2$glmnet.fit,
  newx    = X_train,
  newy    = y_train,
  weights = w_train,
  family  = "gaussian"
)

assess_valid <- assess.glmnet(
  object  = cvmod2$glmnet.fit,
  newx    = X_valid,
  newy    = y_valid,
  weights = w_valid,
  family  = "gaussian"
)

par(mfrow = c(1, 3), mar = c(5,4,3,1))

# MSE
plot(log(cvmod2$glmnet.fit$lambda),
     as.numeric(assess_train$mse),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MSE",
     main = "Training vs Validation MSE",
     ylim = range(c(as.numeric(assess_train$mse),
                    as.numeric(assess_valid$mse))))
lines(log(cvmod2$glmnet.fit$lambda),
      as.numeric(assess_valid$mse),
      lwd = 2, col = "red")
# Best λ by validation MSE
abline(v = log(cvmod2$glmnet.fit$lambda[which.min(assess_valid$mse)]), lty = 2)
points(log(cvmod2$glmnet.fit$lambda[which.min(assess_valid$mse)]),
       as.numeric(min(assess_valid$mse)),
       pch = 19, col = "red")
# CV-picked lambdas
abline(v = log(cvmod2$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod2$lambda.1se), lty = 3, col = "gray40")
legend("topright", c("Training","Validation","Best λ (valid)"),
       col = c("blue","red","black"), lty = c(1,1,2), lwd = c(2,2,1),
       pch = c(NA,NA,19), bty = "n", cex = 0.9)

# RMSE  (argmin RMSE == argmin MSE)
plot(log(cvmod2$glmnet.fit$lambda),
     sqrt(as.numeric(assess_train$mse)),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "RMSE",
     main = "Training vs Validation RMSE",
     ylim = range(c(sqrt(as.numeric(assess_train$mse)),
                    sqrt(as.numeric(assess_valid$mse)))))
lines(log(cvmod2$glmnet.fit$lambda),
      sqrt(as.numeric(assess_valid$mse)),
      lwd = 2, col = "red")
abline(v = log(cvmod2$glmnet.fit$lambda[which.min(assess_valid$mse)]), lty = 2)
points(log(cvmod2$glmnet.fit$lambda[which.min(assess_valid$mse)]),
       sqrt(as.numeric(min(assess_valid$mse))),
       pch = 19, col = "red")
abline(v = log(cvmod2$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod2$lambda.1se), lty = 3, col = "gray40")

# MAE
plot(log(cvmod2$glmnet.fit$lambda),
     as.numeric(assess_train$mae),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MAE",
     main = "Training vs Validation MAE",
     ylim = range(c(as.numeric(assess_train$mae),
                    as.numeric(assess_valid$mae))))
lines(log(cvmod2$glmnet.fit$lambda),
      as.numeric(assess_valid$mae),
      lwd = 2, col = "red")
abline(v = log(cvmod2$glmnet.fit$lambda[which.min(assess_valid$mae)]), lty = 2)
points(log(cvmod2$glmnet.fit$lambda[which.min(assess_valid$mae)]),
       as.numeric(min(assess_valid$mae)),
       pch = 19, col = "red")
abline(v = log(cvmod2$lambda.min), lty = 3, col = "gray40")
abline(v = log(cvmod2$lambda.1se), lty = 3, col = "gray40")

par(mfrow = c(1, 1))

# Sanity check
all.equal(as.numeric(assess.glmnet(object = cvmod2, s = cvmod2$lambda, newx = X_train, newy = y_train,
                                   weights = w_train, family = "gaussian")$mse), 
          as.numeric(assess.glmnet(object = cvmod2, s = cvmod2$lambda, newx = X_train, newy = y_train,
                                   weights = w_train, family = "gaussian")$mse))
# > [1] TRUE
all.equal(as.numeric(assess.glmnet(object = cvmod2, s = cvmod2$lambda, newx = X_train, newy = y_train,
                                   weights = w_train, family = "gaussian")$mae), 
          as.numeric(assess.glmnet(object = cvmod2, s = cvmod2$lambda, newx = X_train, newy = y_train,
                                   weights = w_train, family = "gaussian")$mae))
# > [1] TRUE

##### ---- Predict and evaluate performance on training/validation/test datasets ----
###### ---- cvmod2 ----

# Predict with lambda.min 
pred_train_min <- as.numeric(predict(cvmod2, newx = X_train, s = "lambda.min"))
pred_valid_min <- as.numeric(predict(cvmod2, newx = X_valid, s = "lambda.min"))
pred_test_min  <- as.numeric(predict(cvmod2, newx = X_test,  s = "lambda.min"))

metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

# Predict with lambda.1se 
pred_train_1se <- as.numeric(predict(cvmod2, newx = X_train, s = "lambda.1se"))
pred_valid_1se <- as.numeric(predict(cvmod2, newx = X_valid, s = "lambda.1se"))
pred_test_1se  <- as.numeric(predict(cvmod2, newx = X_test,  s = "lambda.1se"))

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Combine results in a table
metric_results_min <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"],     metrics_valid_min["rmse"],     metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],      metrics_valid_min["mae"],      metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"],     metrics_valid_min["bias"],     metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],       metrics_valid_min["r2"],       metrics_test_min["r2"])
)

metric_results_1se <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"],     metrics_valid_1se["rmse"],     metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],      metrics_valid_1se["mae"],      metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"],     metrics_valid_1se["bias"],     metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],       metrics_valid_1se["r2"],       metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051

print(as.data.frame(metric_results_1se), row.names = FALSE)            # <-
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.61485 73.76314 -2.953923e-13 3.685288 0.04213472
# > Validation 88.40407 70.89232 -2.356312e+00 2.956835 0.04409728
# >       Test 92.28553 74.54139 -5.943704e-01 3.618750 0.04878589

print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051
# >   Training 91.61485 73.76314 -2.953923e-13 3.685288 0.04213472
# > Validation 88.40407 70.89232 -2.356312e+00 2.956835 0.04409728
# >       Test 92.28553 74.54139 -5.943704e-01 3.618750 0.04878589

###### ---- cvmod2$glmnet.fit ----

# Predict with lambda.min 
pred_train_min <- as.numeric(predict(cvmod2$glmnet.fit, newx = X_train, s = cvmod2$lambda.min))
pred_valid_min <- as.numeric(predict(cvmod2$glmnet.fit, newx = X_valid, s = cvmod2$lambda.min))
pred_test_min  <- as.numeric(predict(cvmod2$glmnet.fit, newx = X_test, s = cvmod2$lambda.min))

metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

# Predict with lambda.1se 
pred_train_1se <- as.numeric(predict(cvmod2$glmnet.fit, newx = X_train, s = cvmod2$lambda.1se))
pred_valid_1se <- as.numeric(predict(cvmod2$glmnet.fit, newx = X_valid, s = cvmod2$lambda.1se))
pred_test_1se  <- as.numeric(predict(cvmod2$glmnet.fit, newx = X_test, s = cvmod2$lambda.1se))

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Combine results in a table
metric_results_min <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"],     metrics_valid_min["rmse"],     metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],      metrics_valid_min["mae"],      metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"],     metrics_valid_min["bias"],     metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],       metrics_valid_min["r2"],       metrics_test_min["r2"])
)

metric_results_1se <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"],     metrics_valid_1se["rmse"],     metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],      metrics_valid_1se["mae"],      metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"],     metrics_valid_1se["bias"],     metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],       metrics_valid_1se["r2"],       metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051

print(as.data.frame(metric_results_1se), row.names = FALSE)            # <-
# >   Training 91.61485 73.76314 -2.953923e-13 3.685288 0.04213472
# > Validation 88.40407 70.89232 -2.356312e+00 2.956835 0.04409728
# >       Test 92.28553 74.54139 -5.943704e-01 3.618750 0.04878589

print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.06790 73.19519 -3.246545e-13 3.604227 0.05353777
# > Validation 88.01263 70.63461 -3.029256e+00 2.739292 0.05254384
# >       Test 91.31290 73.69190 -1.083535e+00 3.415555 0.06873051
# >   Training 91.61485 73.76314 -2.953923e-13 3.685288 0.04213472
# > Validation 88.40407 70.89232 -2.356312e+00 2.956835 0.04409728
# >       Test 92.28553 74.54139 -5.943704e-01 3.618750 0.04878589

###### ---- Equivalence check ----
all.equal(as.numeric(predict(cvmod2, newx = X_train, s = "lambda.min")), 
          as.numeric(predict(cvmod2$glmnet.fit, newx = X_train, s = cvmod2$lambda.min)))
# > [1] TRUE
all.equal(as.numeric(predict(cvmod2, newx = X_valid, s = "lambda.min")), 
          as.numeric(predict(cvmod2$glmnet.fit, newx = X_valid, s = cvmod2$lambda.min)))
# > [1] TRUE
all.equal(as.numeric(predict(cvmod2, newx = X_test, s = "lambda.min")), 
          as.numeric(predict(cvmod2$glmnet.fit, newx = X_test, s = cvmod2$lambda.min)))
# > [1] TRUE

#### ---- 3) Compare the two fits----
# grouped=TRUE (fold‑level aggregation): 
#   For each λ, glmnet computes a fold MSE then takes a weighted mean across folds with weights;
#   SE is computed across folds (K replicates) → larger SE.
# grouped=FALSE (observation‑level aggregation): 
#   For each λ, glmnet stacks all OOF residuals and computes the pooled weighted mean;
#   SE is computed across observations (n replicates) → smaller SE.
# The means of MSE (cvm, additive loss) are algebraically identical. 
# The difference is the standard error (cvsd), which affects lambda.1se but not lambda.min.
# OOF predictions (if set keep=TRUE) are unchanged by grouped; grouped only affects how those OOF errors are summarized.

all.equal(cvmod1$lambda,cvmod2$lambda)
# > [1] TRUE

all.equal(cvmod1$cvm, as.numeric(cvmod2$cvm))  
# > [1] TRUE

all.equal(cvmod1$cvsd, as.numeric(cvmod2$cvsd))   # <-  
# > [1] "Mean relative difference: 0.632288"

all.equal(cvmod1$nzero,cvmod2$nzero)
# > [1] TRUE

cvmod1$lambda.min; cvmod2$lambda.min                        
# > [1] 0.06714323
# > [1] 0.06714323

cvmod1$lambda.1se;  cvmod2$lambda.1se             # <-                                 
# > [1] 4.025131
# > [1] 5.839776 

length(cvmod1$fit.preval);length(cvmod2$fit.preval)
# > [1] 868124
# > [1] 868124

all.equal(cvmod1$foldid,cvmod2$foldid)
# > [1] TRUE





