# ---- Script Description ----

# Remark: Coefficients are not standardized - variable importance plot.

# ---- I. Predictive Modelling: Version 1.1 ----

# Fit with glmnet: alpha α (=1, default), lambda λ (grid)

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

### ---- Fit Elastic Net for PV1MATH only: glmnet ----

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

# Timing the fit
tic("Fitting with glmnet")
# Fit elastic net path with default α = 1
mod <- glmnet(                                               # ??glmnet, ?glmnet, glmnet::glmnet, str(glmnet)
  x = X_train, 
  y = y_train,
  family = "gaussian",                                       # For regression
  weights = w_train,
  alpha = 1,                                                 # Elasticnet mixing parameter with 0≤α≤1; Default: 1, lasso penalty
  #nlambda = 100,
  #lambda.min.ratio = ifelse(nobs < nvars, 0.01, 1e-04),     # nobs <- nrow(X_train); nvars <- ncol(X_train)
  standardize = TRUE,                                        # x standardization; Default: TRUE
  intercept = TRUE,                                          # Intercept be fitted; Default: TRUE
  trace.it = 1                                               # Display progress bar; Default: 0
)
toc()
# > Fitting with glmnet: 1.755 sec elapsed

#### ---- Explore fitted model----

# Check the object class
class(mod)  # an object of class glmnet

# Check str
str(mod)

# Visualize the coefficients
plot(mod)                                 # <=> plot(mod, xvar = "lambda")
# Each curve correspond to a variable
plot(mod, xvar = "lambda", label = TRUE)  # Plot fit against the log-lambda value and with each curve labeled
plot(mod, xvar = "dev", label = TRUE)     # vs %deviance

# Print the summary of the glmnet path at each step
mod
print(mod)

# Extract lambda
mod$lambda
length(mod$lambda) 
# > [1] 62

# Extract df
mod$df

# Extract the deviance and dev.ratio from a glmnet object
deviance(mod)   # raw deviances at each λ, ranking to "mse" for gaussian
mod$nulldev     # the deviance of the intercept-only model (no predictors) - the baseline for comparing improvement
mod$dev.ratio   # %Dev = mod$dev.ratio * 100
all.equal(mod$dev.ratio, 1 - deviance(mod)/mod$nulldev) 
# > [1] TRUE

# --- Obtain the model coefficients at one or more 𝜆’s within the range of the sequence ---
# λ = 0.1
any(mod$lambda == 0.1)            # 0.1 is not in original lambda sequence
# > [1] FALSE
coef.apprx <- coef(mod, s = 0.1)  # ?coef.glmnet, ?predict.glmnet
# s: lambda
# Default: exact = FALSE
coef.exact <- coef(mod, s = 0.1, exact = TRUE, x = X_train, y = y_train, weights = w_train)
cbind(coef.exact[which(coef.exact != 0)], 
      coef.apprx[which(coef.apprx != 0)])
# >            [,1]       [,2]
# > [1,] 521.317564 521.318067
# > [2,]  -1.847909  -1.847906
# > [3,]   1.901640   1.901515
# > [4,]  -6.460963  -6.461062
# > [5,]  -1.810798  -1.810750

# λ = mod$lambda[1]
any(mod$lambda == mod$lambda[1])  # mod$lambda[1] = 19.57260747... is in original lambda sequence
# > [1] TRUE
coef.apprx <- coef(mod, mod$lambda[1])  
coef.exact <- coef(mod, s = mod$lambda[1], exact = TRUE, x = X_train, y = y_train, weights = w_train)
cbind(coef.exact[which(coef.exact != 0)], 
      coef.apprx[which(coef.apprx != 0)])
# >          [,1]       [,2]
# > [1,] 502.3584 502.3584

# Two λs
coef(mod, s = c(0.1, mod$lambda[1]))
# >5 x 2 sparse Matrix of class "dgCMatrix"
# >             s= 0.10000 s=19.57261
# > (Intercept) 521.318067   502.3584
# > EXERPRAC     -1.847906     .     
# > STUDYHMW      1.901515     .     
# > WORKPAY      -6.461062     .     
# > WORKHOME     -1.810750     .      

# --- broom::tidy.glmnet and broom::glance.glmnet	---
tidy(mod)
glance(mod)

tidied <- tidy(mod) %>% filter(term != "(Intercept)")

ggplot(tidied, aes(step, estimate, group = term, color = term)) +
  geom_line()

ggplot(tidied, aes(lambda, estimate, group = term, color = term)) +
  geom_line() +
  scale_x_log10()

ggplot(tidied, aes(lambda, dev.ratio)) +
  geom_line()

# --- caret::varImp	---
varImp(mod, lambda = 0.1)             #?varImp; no sign, all positive, no intercept
# >           Overall
# > EXERPRAC 1.847906
# > STUDYHMW 1.901515
# > WORKPAY  6.461062
# > WORKHOME 1.810750
varImp(mod, lambda = mod$lambda[1])
# >          Overall
# > EXERPRAC       0
# > STUDYHMW       0
# > WORKPAY        0
# > WORKHOME       0
varImp(mod, lambda = mod$lambda[-1])
# >            Overall
# > EXERPRAC 0.0000000
# > STUDYHMW 0.0000000
# > WORKPAY  0.6545619
# > WORKHOME 0.0000000
varImp(mod, lambda = mod$lambda[floor(length(mod$lambda)/2)])
# >           Overall
# > EXERPRAC 1.528060
# > STUDYHMW 1.296497
# > WORKPAY  6.184560
# > WORKHOME 1.443783

# --- predict and glmnet::assess.glmnet	---
predict(object = mod,          #?predict.glmnet
        newx = X_valid,
        s = NULL,              # Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create the model.
        exact = FALSE)         # Relevant only when predictions are made at values of s (lambda) different from those used in the fitting of the original model.Default: FALSE. 
# <=> 
predict(object = mod,        
        newx = X_valid,
        s = NULL,            
        exact = FALSE,       
        type = "link")

all.equal(predict(object = mod,          
                  newx = X_valid,
                  s = NULL,              
                  exact = FALSE),
          predict(object = mod,        
                  newx = X_valid,
                  s = NULL,            
                  exact = FALSE,       
                  type = "link"))
# > [1] TRUE

predict(object = mod,
        newx = X_valid,
        s = NULL,              
        exact = FALSE,         
        type = "coefficient")  # Computes the coefficients at the requested values for s

assess.glmnet(object = predict(object = mod,      
                               newx = X_valid,
                               s = NULL,              
                               exact = FALSE) , 
              newy = y_valid,
              weights = w_valid,
              family = "gaussian")

assess.glmnet(object = mod, 
              newx = X_valid,
              newy = y_valid,
              weights = w_valid,
              family = "gaussian")  # VALID MSE and MAE

all.equal(assess.glmnet(object = predict(object = mod,      
                                         newx = X_valid,
                                         s = NULL,              
                                         exact = FALSE) , 
                        newy = y_valid,
                        weights = w_valid,
                        family = "gaussian"), 
          assess.glmnet(object = mod, 
                        newx = X_valid,
                        newy = y_valid,
                        weights = w_valid,
                        family = "gaussian"))
# > [1] TRUE

sqrt(assess.glmnet(object = mod, 
                   newx = X_valid,
                   newy = y_valid,
                   weights = w_valid,
                   family = "gaussian")$mse)  # RMSE, -> rmse_valid_vector


# --- Visualize Training vs Validation MSE, RMSE, MAE across Lambda ---
assess_train <- assess.glmnet(object = mod, newx = X_train, newy = y_train,
                              weights = w_train, family = "gaussian")
assess_valid <- assess.glmnet(object = mod, newx = X_valid, newy = y_valid,
                              weights = w_valid, family = "gaussian")

par(mfrow = c(1, 3), mar = c(5,4,3,1))

# MSE
plot(log(mod$lambda), as.numeric(assess_train$mse),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "MSE",
     main = "Training vs Validation MSE across Lambda",
     ylim = range(c(as.numeric(assess_train$mse), as.numeric(assess_valid$mse))))
lines(log(mod$lambda), as.numeric(assess_valid$mse), lwd = 2, col = "red")
abline(v = log(mod$lambda[which.min(assess_valid$mse)]), lty = 2)
points(log(mod$lambda[which.min(assess_valid$mse)]), as.numeric(min(assess_valid$mse)), pch = 19, col = "red")
legend("topright", legend = c("Training", "Validation", "Best λ (valid)"),
       col = c("blue","red","black"), lty = c(1,1,2), lwd = c(2,2,1), pch = c(NA,NA,19), bty = "n")

# RMSE
plot(log(mod$lambda), as.numeric(sqrt(assess_train$mse)),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "RMSE",
     main = "Training vs Validation RMSE across Lambda",
     ylim = range(c(as.numeric(sqrt(assess_train$mse)), as.numeric(sqrt(assess_valid$mse)))))
lines(log(mod$lambda), as.numeric(sqrt(assess_valid$mse)), lwd = 2, col = "red")
abline(v = log(mod$lambda[which.min(sqrt(assess_valid$mse))]), lty = 2)
points(log(mod$lambda[which.min(sqrt(assess_valid$mse))]), as.numeric(min(sqrt(assess_valid$mse))), pch = 19, col = "red")
legend("topright", legend = c("Training", "Validation", "Best λ (valid)"),
       col = c("blue","red","black"), lty = c(1,1,2), lwd = c(2,2,1), pch = c(NA,NA,19), bty = "n")

# MAE
plot(log(mod$lambda), as.numeric(assess_train$mae),
     type = "l", lwd = 2, col = "blue",
     xlab = "log(Lambda)", ylab = "mae",
     main = "Training vs Validation MAE across Lambda",
     ylim = range(c(as.numeric(assess_train$mae), as.numeric(assess_valid$mae))))
lines(log(mod$lambda), as.numeric(assess_valid$mae), lwd = 2, col = "red")
abline(v = log(mod$lambda[which.min(assess_valid$mae)]), lty = 2)
points(log(mod$lambda[which.min(assess_valid$mae)]), as.numeric(min(assess_valid$mae)), pch = 19, col = "red")
legend("topright", legend = c("Training", "Validation", "Best λ (valid)"),
       col = c("blue","red","black"), lty = c(1,1,2), lwd = c(2,2,1), pch = c(NA,NA,19), bty = "n")

par(mfrow = c(1, 1))

### ---- Predict and evaluate performance on training/validation/test datasets ----

#### ---- Select best_lambda ----

# Predict for all lambdas
pred_train_matrix <- predict(mod, newx = X_train, s = mod$lambda)  # <=> predict(mod, newx = X_train), Matrix: n_train x n_lambda
pred_valid_matrix <- predict(mod, newx = X_valid, s = mod$lambda)  # <=> predict(mod, newx = X_valid), Matrix: n_valid x n_lambda

# Calculate rmse
rmse_train_vector <- apply(pred_train_matrix, 2, function(p) w_rmse(y_train, p, w_train))
rmse_valid_vector <- apply(pred_valid_matrix, 2, function(p) w_rmse(y_valid, p, w_valid))

# Sanity check
rmse_valid_vector == sqrt(assess.glmnet(object = mod, 
                                        newx = X_valid,
                                        newy = y_valid,
                                        weights = w_valid,
                                        family = "gaussian")$mse)

# Find best_lambda
best_lambda <- mod$lambda[ which.min(rmse_valid_vector)]
best_lambda
# > [1] 0.754236

# --- Explore best_lambda ---
coef(mod, best_lambda)
# > 5 x 1 sparse Matrix of class "dgCMatrix"
# >             s=0.754236
# > (Intercept) 520.831125
# > EXERPRAC     -1.657839
# > STUDYHMW      1.542173
# > WORKPAY      -6.296609
# > WORKHOME     -1.592756

varImp(mod, best_lambda) # same value but no signs
# >           Overall
# > EXERPRAC 1.657839
# > STUDYHMW 1.542173
# > WORKPAY  6.296609
# > WORKHOME 1.592756

# Coefficient paths with best-λ marker 
plot(mod, xvar = "lambda", label = TRUE)
abline(v = log(best_lambda), lty = 2)

# Plot Model size (df = #non-zero coefs) vs log(λ)
plot(
  log(mod$lambda), mod$df,
  type = "l", lwd = 2,
  xlab = "log(Lambda)", ylab = "Non-zero coefficients (df)",
  main = "Sparsity across Lambda"
)
abline(v = log(best_lambda), lty = 2)

# Plot %Dev (pseudo-R²) vs log(λ)
plot(
  log(mod$lambda), mod$dev.ratio * 100,
  type = "l", lwd = 2,
  xlab = "log(Lambda)", ylab = "%Dev (100 × dev.ratio)",
  main = "%Dev explained across Lambda"
)
abline(v = log(best_lambda), lty = 2)

# Visualize RMSE vs. log(λ) 
# ggplot
ggplot() + 
  geom_line(aes(x = log(mod$lambda), y = rmse_train_vector, color = "Train")) +
  geom_line(aes(x = log(mod$lambda), y = rmse_valid_vector, color = "Validation")) +
  geom_vline(xintercept = log(best_lambda), linetype = "dashed", color = "black") +
  labs(
    x = "log(Lambda)",
    y = "Weighted RMSE",
    title = "Train vs Validation RMSE across Lambda"
  ) +
  scale_color_manual(values = c("Train" = "blue", "Validation" = "red")) +
  theme_minimal() 

# base R
plot(
  log(mod$lambda), rmse_train_vector,
  type = "l", col = "blue", lwd = 2,
  xlab = "log(Lambda)", ylab = "Weighted RMSE",
  main = "Train vs Validation RMSE across Lambda",
  ylim = range(c(rmse_train_vector, rmse_valid_vector), na.rm = TRUE)
)
lines(log(mod$lambda), rmse_valid_vector, col = "red", lwd = 2)
abline(v = log(best_lambda), lty = 2, col = "black")
legend("topright", legend = c("Train", "Validation"),
       col = c("blue", "red"), lwd = 2, bty = "n")

# Generalization gap (Validation − Train RMSE) vs log(λ) 
plot(
  log(mod$lambda), rmse_valid_vector - rmse_train_vector,
  type = "l", lwd = 2,
  xlab = "log(Lambda)", ylab = "Validation − Train RMSE",
  main = "Generalization gap across Lambda"
)
abline(v = log(best_lambda), lty = 2)

# Variable importance (signed coefficients, sorted by |β|)
coef(mod, s = best_lambda) %>% 
  as.matrix() %>%
  { tibble(term = rownames(.), coefficient = as.numeric(.[, 1])) } %>%
  filter(term != "(Intercept)") %>%
  mutate(
    abs_coef = abs(coefficient),
    sign = case_when(
      coefficient >  0 ~ "positive",
      coefficient <  0 ~ "negative",
      TRUE             ~ "zero"
    )
  ) %>%
  arrange(desc(abs_coef)) %>%
  { 
    ggplot(., aes(x = reorder(term, abs_coef), y = coefficient, fill = sign)) +
      geom_col(width = 0.75) +
      geom_hline(yintercept = 0, linewidth = 0.6, linetype = "dashed") +
      coord_flip() +
      geom_text(
        aes(label = sprintf("%.2f", coefficient),
            hjust = ifelse(coefficient >= 0, -0.15, 1.15)),
        size = 3
      ) +
      scale_fill_manual(
        values = c(positive = "#1b9e77", zero = "#bdbdbd", negative = "#d95f02"),
        name   = "Sign"
      ) +
      labs(
        title = "Elastic Net Variable Importance (signed β) at best λ",
        subtitle = sprintf(
          "λ = %.6f   |   %%Dev = %.2f",
          best_lambda,
          100 * mod$dev.ratio[which.min(abs(mod$lambda - best_lambda))]
        ),
        x = "Predictor",
        y = "Coefficient (β)"
      ) +
      # exact same expansion behavior as your version:
      expand_limits(y = c(min(.$coefficient, 0) * 1.15,
                          max(.$coefficient, 0) * 1.15)) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "top")
  }


#### ---- Use best_lambda to predict and evaluate ----

# Predict using best_lambda
pred_train <- as.numeric(predict(mod, newx = X_train, s = best_lambda)) 
pred_valid <- as.numeric(predict(mod, newx = X_valid, s = best_lambda))
pred_test  <- as.numeric(predict(mod, newx = X_test,  s = best_lambda))

# Compute weighted metrics (uses your existing helper)
metrics_train <- compute_metrics(y_true = y_train, y_pred = pred_train, w = w_train)
metrics_valid <- compute_metrics(y_true = y_valid, y_pred = pred_valid, w = w_valid)
metrics_test  <- compute_metrics(y_true = y_test,  y_pred = pred_test,  w = w_test)

# Combine results in a table
metric_results <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train["rmse"],     metrics_valid["rmse"],     metrics_test["rmse"]),
  MAE     = c(metrics_train["mae"],      metrics_valid["mae"],      metrics_test["mae"]),
  Bias    = c(metrics_train["bias"],     metrics_valid["bias"],     metrics_test["bias"]),
  `Bias%` = c(metrics_train["bias_pct"], metrics_valid["bias_pct"], metrics_test["bias_pct"]),
  R2      = c(metrics_train["r2"],       metrics_valid["r2"],       metrics_test["r2"])
)

print(as.data.frame(metric_results), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias    Bias%         R2
# >   Training 91.08162 73.22339 -3.145380e-13 3.615077 0.05325252
# > Validation 88.00163 70.61354 -2.938738e+00 2.768314 0.05278064
# >       Test 91.39958 73.77060 -1.007781e+00 3.445324 0.06696172

# --- Visualization ---

# Residual diagnostics (Validation) at best-λ
plot(pred_valid, y_valid - pred_valid,                          # residual = y_valid - pred_valid
     xlab = "Fitted (Validation)", ylab = "Residual",
     main = "Residuals vs Fitted (Validation) at best lambda")
abline(h = 0, lty = 2)

qqnorm(y_valid - pred_valid); qqline(y_valid - pred_valid)


