# ---- Script Description ⚠️ Version Beta: ContainErrors ----
# This script conducts an in-depth explanatory and predictive modeling analysis 
# of PISA 2022 Canada student-level data using official PISA methodology. 
#
# Methodology:
# - Final student weights and 80 BRR replicate weights with Fay’s adjustment factor (k = 0.5) are applied
# - Rubin's Rules are used for combining across 10 plausible values (PV1–PV10)
# - Balanced Repeated Replication (BRR) is used to compute sampling variance
#
# Objectives:
# 1. Estimate the correlation between plausible values in mathematics (PV1MATH–PV10MATH)
#    and plausible values in reading (PV1READ–PV10READ), using official formulas not directly supported 
#    by the `intsvy` package. Results are verified against the IEA IDB Analyzer (SPSS Statistics).
#
# 2. Fit explanatory linear regression models to predict math plausible values from selected predictors,
#    using both final and replicate weights, and compute pooled estimates with total standard errors.
#
# 3. Evaluate predictive performance (RMSE, MAE, Bias, Bias%, R²) across training, validation, 
#    and test sets using a random 70/15/15 split. Uncertainty is computed using Rubin’s Rules and BRR.
#
# Final outputs include point estimates, standard errors, and 95% confidence intervals 
# for all evaluation metrics in a tidy tabular format.



# This script conducts an in-depth analysis of PISA 2022 Canada student-level 
#  data, using plausible values in reading (PV1READ-PV10READ) as the X 
#  (explanatory/independent/feature variable) and plausible values in 
#  mathematics (PV1MATH–PV10MATH) as the Y (response/dependent/target variable).

# Part I: Explanatory Modeling
# - Developed linear regression models manually using final student weights and 80 BRR replicate weights.
# - Followed official PISA methodology by combining Balanced Repeated Replication (BRR) with Rubin’s Rules 
#   to compute standard errors for regression coefficients and model fit statistics.
# - Validated manual results against the `intsvy::pisa.reg.pv()` function and the IEA IDB Analyzer with SPSS Statistics.
#   Consistency in estimates and uncertainty measures was confirmed.
# - Explored source code and documentation for `summary.lm()`, `weighted.mean`, and `pisa.reg.pv()` to ensure 
#   alignment between manual calculations and R internals.

# Part II: Predictive Modeling
# - Extended the analysis to a predictive setting by splitting the data into training, validation, and test sets.
# - Evaluated model performance using five metrics: RMSE, MAE, Bias%, R², and Adjusted R².
# - In addition to point estimates, calculated standard errors (SEs) for each performance metric using BRR and Rubin’s Rules, 
#   to quantify uncertainty and construct confidence intervals.

# This end-to-end implementation deepens familiarity with PISA methodology and provides a validated and reproducible framework 
# for both explanatory and predictive modeling under complex survey design.


# ---- I. Correlation: PVmMATH ~ PVmREAD ----

## ---- Set-up ----
setwd("~/projects/pisa")
library(haven)   # For reading SPSS .sav files
library(dplyr)   # For data manipulation
library(tibble)  # For tidy table outputs
library(broom)   # For tidying model output
library(intsvy)  # For analyze PISA data
library(weights) # For weighted correlation

# Load dataset
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE)

# Explore missingness
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "MATH")]))
# > [1] 0
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "READ")]))
# > [1] 0

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Data preparation
M <- 10  # Number of plausible values (PV1MATH to PV10MATH)
G <- 80  # Number of BRR replicate weights (W_FSTURWT1 to W_FSTURWT80)
k <- 0.5 # Fay's adjustment factor used in PISA's BRR method

pvmaths  <- paste0("PV", 1:M, "MATH")    # Plausible Values (PVs) in mathematics
pvreads  <- paste0("PV", 1:M, "READ")    # Plausible Values (PVs) in reading
rep_wts  <- paste0("W_FSTURWT", 1:G)     # 80 replicate weights
final_wt <- "W_FSTUWT"                   # Final student weight

temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID, W_FSTUWT,        # Select ID, weights
         all_of(rep_wts), all_of(pvmaths), all_of(pvreads))    # Include replicate weights + PVs


## ---- Correlation: Weighted Correlation Estimation ----

# --- Define helper function for weighted correlation ---
get_wtd_corr <- function(x, y, w) {
  # Assume no missing values in x, y, w
  if (length(x) < 3) return(NA_real_)
  
  mx <- weighted.mean(x, w)
  my <- weighted.mean(y, w)
  
  cov_xy <- sum(w * (x - mx) * (y - my)) / sum(w)
  var_x  <- sum(w * (x - mx)^2) / sum(w)
  var_y  <- sum(w * (y - my)^2) / sum(w)
  
  if (var_x <= 0 || var_y <= 0) return(NA_real_)
  
  return(cov_xy / sqrt(var_x * var_y))
}

# --- Compute correlation point estimates for each PV ---
main_corrs <- sapply(1:M, function(m) {
  get_wtd_corr(temp_data[[pvmaths[m]]], temp_data[[pvreads[m]]], temp_data[[final_wt]])
})
main_corrs
# > [1] 0.7432861 0.7557394 0.7451909 0.7499096 0.7548876 0.7456749 0.7522881 0.7515863 0.7522282 0.7526489

# --- Compute mean correlation across PVs (Rubin's rule) ---
main_corr <- mean(main_corrs)  # ρ̂
main_corr
# [1] 0.750344

# --- Compute replicate-based correlation matrix (M × G) ---
rep_corr_matrix <- matrix(NA_real_, nrow = M, ncol = G)
for (m in 1:M) {
  for (g in 1:G) {
    rep_corr_matrix[m, g] <- get_wtd_corr(
      temp_data[[pvmaths[m]]],
      temp_data[[pvreads[m]]],
      temp_data[[rep_wts[g]]]
    )
  }
}
dim(rep_corr_matrix)
# > [1] 10 80

# --- Compute sampling variance per PV (Fay-adjusted BRR) ---
sampling_var_corr_per_pv <- sapply(1:M, function(m) {
  replicate_corrs <- rep_corr_matrix[m, ]  # ρ̂_{m,g}
  sampling_var_corr_m <- mean((replicate_corrs - main_corrs[m])^2) / (1 - k)^2  # σ²(ρ̂ₘ)
  return(sampling_var_corr_m)
})
sampling_var_corr_per_pv
# [1] 2.731634e-05 2.390242e-05 2.859267e-05 2.107498e-05 2.456642e-05 2.423750e-05 2.106785e-05 2.368990e-05 2.220605e-05 2.134763e-05

# --- Compute average sampling variance across PVs ---
sampling_var_corr <- mean(sampling_var_corr_per_pv)  # σ²(ρ̂)
sampling_var_corr
# > [1] 2.380018e-05

# --- Compute imputation variance using Rubin's rule ---
imputation_var_corr <- sum((main_corrs - main_corr)^2) / (M - 1)  # σ²(test) = (1/(M-1)) * Σ (ρ̂ₘ -ρ̂)²
imputation_var_corr
# > [1] 1.803301e-05

# --- Compute final total error variance and standard error ---
var_final_corr <- sampling_var_corr + (1 + 1/M) * imputation_var_corr  # σ²(error)
var_final_corr
# > [1] 4.363649e-05
se_final_corr <- sqrt(var_final_corr)                                  # sqrt(σ²(error))
se_final_corr
# > [1] 0.006605792

## ---- Output final correlation estimate and variances ----
corr_results <- tibble::tibble(
  Step = c("Mean correlation ρ̂",
           "Sampling variance σ²(ρ̂)",
           "Imputation variance σ²(test)",
           "Total error variance σ²(error)",
           "Final standard error √σ²(error)"),
  Value = formatC(c(main_corr, sampling_var_corr, imputation_var_corr, var_final_corr, se_final_corr), 
                  format = "f", digits = 6)
)

print(as.data.frame(corr_results), row.names = FALSE)
# >                            Step    Value
# >              Mean correlation ρ̂ 0.750344
# >         Sampling variance σ²(ρ̂) 0.000024
# >    Imputation variance σ²(test) 0.000018
# >  Total error variance σ²(error) 0.000044
# > Final standard error √σ²(error) 0.006606



# ---- II. Explanatory Modelling: PVmMATH ~ PVmREAD ----

## ---- Set-up ----
setwd("~/projects/pisa")
library(haven)   # For reading SPSS .sav files
library(dplyr)   # For data manipulation
library(tibble)  # For tidy table outputs
library(broom)   # For tidying model output
library(intsvy)  # For analyze PISA data

# Load dataset
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE)

# Explore missingness
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "MATH")]))
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "READ")]))

# Explore summary statistics
summary(pisa_2022_student_canada[, paste0("PV", 1:10, "MATH")])
summary(pisa_2022_student_canada[, paste0("PV", 1:10, "READ")])

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Data preparation
M <- 10  # Number of plausible values (PV1MATH to PV10MATH)
G <- 80  # Number of BRR replicate weights (W_FSTURWT1 to W_FSTURWT80)
k <- 0.5 # Fay's adjustment factor used in PISA's BRR method

pvmaths  <- paste0("PV", 1:M, "MATH")    # Plausible Values (PVs)in mathematics
pvreads  <- paste0("PV", 1:M, "READ")    # Plausible Values (PVs)in reading
rep_wts  <- paste0("W_FSTURWT", 1:G)     # 80 replicate weights
final_wt <- "W_FSTUWT"                   # Final student weight

temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID, W_FSTUWT,        # Select ID, weights
         all_of(rep_wts), all_of(pvmaths), all_of(pvreads))    # Include replicate weights + PVs

## ---- Initial Exploration----

# Fit weighted linear model: Math PV1 ~ Gender 
mod <- lm(PV1MATH ~ PV1READ, data = temp_data, weights = temp_data[[final_wt]]) 

# Display regression summary
summary(mod)

# --- Analyze variance ---
anova(mod)

# SSR
SSR <- sum(anova(mod)[1, 2])
SSR

# SSE
SSE <- anova(mod)[2, 2]
SSE

# SST
null <- lm(PV1MATH ~ 1, data = temp_data, weights = temp_data[[final_wt]]) 
anova(null)

SST <- anova(null)[, 2]
SST

# R^2
SSR/SST

# sigma-hat
np1 <- anova(mod)[2, 1]
sqrt(SSE/np1)


# Plot diagnostics (2x2 layout)
par(mfrow = c(2, 2))
plot(mod)  # # TODO: consider coloring points by gender
par(mfrow = c(1, 1))

# Retrieve the internal source code used by summary.lm
getS3method("summary", "lm")

# Version for automation
#lm(as.formula(paste("PV1MATH", "~ PV1READ")), data = temp_data, weights = temp_data[[final_wt]]) %>%
#  summary()

# TODO:
# Use gender-based coloring in diagnostic plots

## ---- Fit Main Models using Final Student Weight (W_FSTUWT) ----

# For each plausible value (PV), fit: PVmMATH ~ PVmREAD using W_FSTUWT
main_models <- lapply(1:M, function(m) {
  formula <- as.formula(paste0(pvmaths[m], " ~ ", pvreads[m]))
  mod <- lm(formula, data = temp_data, weights = temp_data[[final_wt]])
  summ <- summary(mod)
  list(
    formula        = formula,              # Regression formula used
    mod            = mod,                  # Fitted lm model object
    summ           = summ,                 # Model summary (diagnostics, F-stat, etc.)
    coefs          = coef(mod),            # θ̂ₘ: Estimated coefficients (intercept, slope)
    r2             = summ$r.squared,       # R²: Proportion of variance explained
    adj_r2         = summ$adj.r.squared,   # Adjusted R²
    sigma          = summ$sigma,           # σ̂: Residual standard error
    fstat_val      = summ$fstatistic[["value"]],  # F-statistic value
    fstat_numdf    = summ$fstatistic[["numdf"]],  # Numerator degrees of freedom
    fstat_dendf    = summ$fstatistic[["dendf"]],  # Denominator degrees of freedom
    dof_model      = summ$df[1],           # Model DF (non-aliased coefficients)
    dof_residual   = summ$df[2],           # Residual DF = n - p
    dof_total      = summ$df[3]            # Total DF (model + residual)
  )
})
# Check first fitted model structure
main_models[[1]]
main_models[[2]]

main_models[[1]]$coefs
main_models[[2]]$coefs

## ---- Extract Estimates from Main Models (Rubin’s Step 1 & 2) ----

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs
# >       (Intercept)   PV1READ
# > [1,]    168.5908 0.6482272
# > [2,]    188.5821 0.6103437
# > [3,]    187.8780 0.6106393
# > [4,]    184.2774 0.6172270
# > [5,]    187.5909 0.6106041
# > [6,]    187.7387 0.6092078
# > [7,]    190.7875 0.6052069
# > [8,]    186.7155 0.6123744
# > [9,]    184.8894 0.6152872
# > [10,]   188.7872 0.6083374

# Rename for clarity
colnames(main_coefs) <- c("Intercept", "PVmREAD")
rownames(main_coefs) <- pvmaths  # pvmaths  <- paste0("PV", 1:M, "MATH") 
main_coefs
# >          Intercept   PVmREAD
# > PV1MATH   168.5908 0.6482272
# > PV2MATH   170.6940 0.6441958
# > PV3MATH   169.8827 0.6440868
# > PV4MATH   166.9828 0.6501220
# > PV5MATH   162.8552 0.6592350
# > PV6MATH   166.8017 0.6503792
# > PV7MATH   171.6272 0.6423905
# > PV8MATH   171.6708 0.6419811
# > PV9MATH   163.1490 0.6571757
# > PV10MATH  166.8554 0.6503970

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef
# > Intercept    PVmREAD 
# > 167.910971   0.648819

# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.5524742 0.5711421 0.5553095 0.5623644 0.5698552 0.5560311 0.5659374 0.5648819 0.5658472 0.5664804
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.5524548 0.5711235 0.5552902 0.5623454 0.5698366 0.5560119 0.5659185 0.5648631 0.5658284 0.5664616
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 248.6635 241.6448 247.9240 245.7833 242.8337 246.6946 242.6558 244.6913 245.0744 241.5740
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 28481.34 30725.37 28810.02 29646.37 30564.43 28894.36 30080.31 29951.39 30069.28 30146.89

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.5630323
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.5630134
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 244.7539
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 29736.98

## ---- Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
replicate_models <- lapply(1:M, function(m) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste0(pvmaths[m], " ~ ", pvreads[m]))
    mod <- lm(formula, data = temp_data, weights = temp_data[[w]])
    summ <- summary(mod)
    list(
      formula       = formula,                          # Regression formula
      mod           = mod,                              # Fitted lm model
      summ          = summ,                             # Summary output
      coefs         = coef(mod),                        # θ̂ₘg: Coefficients (intercept, slope)
      r2            = summ$r.squared,                   # R²
      adj_r2        = summ$adj.r.squared,               # Adjusted R²
      sigma         = summ$sigma,                       # Residual standard error
      fstat_val     = summ$fstatistic[["value"]],       # F-statistic
      fstat_numdf   = summ$fstatistic[["numdf"]],       # Numerator df
      fstat_dendf   = summ$fstatistic[["dendf"]],       # Denominator df
      dof_model     = summ$df[1],                       # Degrees of freedom (model)
      dof_residual  = summ$df[2],                       # Degrees of freedom (residual)
      dof_total     = summ$df[3]                        # Total degrees of freedom
    )
  })
})
# Structure: replicate_models[[m]][[g]] — list of 10 PVs × 80 replicates
replicate_models[[1]][[1]]  # Inspect one model
replicate_models[[1]]$coefs
replicate_models[[1]][[1]]$coefs


# --- Extract replicate fit estimates (Rubin's step 1)---
# Coefficients: rep_coefs[[m]] = G × p matrix for plausible value m 
rep_coefs <- lapply(replicate_models, function(m) {
  do.call(rbind, lapply(m, function(g) coef(g$mod)))  # G × p matrix for each PV
})             # list of ten 80x2 matrix
rep_coefs[[1]] # # View 80 coefficients (with replicate weights) for PV1MATH ~ PV1READ: 80x2 matrix

# Fit statistics: each rep_stat[[stat]] is a G × M matrix (replicates × PVs)---
rep_r2     <- sapply(replicate_models, function(m) sapply(m, function(g) g$r2))             # 80x2 matrix
class(rep_r2);dim(rep_r2)
# > [1] "matrix" "array" 
# > [1] 80 10
rep_adj_r2 <- sapply(replicate_models, function(m) sapply(m, function(g) g$adj_r2))         # 80x2 matrix
rep_sigma  <- sapply(replicate_models, function(m) sapply(m, function(g) g$sigma))          # 80x2 matrix
rep_fstat_val  <- sapply(replicate_models, function(m) sapply(m, function(g) g$fstat_val))  # 80x2 matrix

## ---- Rubin + BRR for Standard Errors (SEs)----
# --- Rubin + BRR for Coefficients ---
sampling_var_matrix_coef <- sapply(1:M, function(m) {
  coefs_m <- rep_coefs[[m]]                   # G×p=80x2 matrix for PV m: each row is θ̂ₘ_g (replicate estimates for PV m)
  sweep(coefs_m, 2, main_coefs[m, ])^2 |>     # Compute  (θ̂ₘ_g - θ̂ₘ)^2 element-wise
    colSums() / (G * (1 - k)^2)               # Compute BRR sampling variance with Fay's adjustment for each coefficient
})
sampling_var_matrix_coef 
# >                     [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]        [,9]        [,10]
# > (Intercept) 1.109334e+01 1.031750e+01 8.029376e+00 8.143299e+00 9.774924e+00 9.3477392216 1.093149e+01 1.015398e+01 1.21801e+01 1.071623e+01
# > PV1READ     5.033952e-05 4.295971e-05 3.554705e-05 3.436617e-05 4.627422e-05 0.0000386327 4.231929e-05 4.072383e-05 4.68876e-05 4.502253e-05

# Rename for clarity
colnames(sampling_var_matrix_coef) <- pvmaths  # pvmaths  <- paste0("PV", 1:M, "MATH") 
rownames(sampling_var_matrix_coef) <- c("Intercept", "PVmREAD")
sampling_var_matrix_coef
# >                PV1MATH      PV2MATH      PV3MATH      PV4MATH      PV5MATH      PV6MATH      PV7MATH      PV8MATH     PV9MATH     PV10MATH
# > Intercept 1.109334e+01 1.031750e+01 8.029376e+00 8.143299e+00 9.774924e+00 9.3477392216 1.093149e+01 1.015398e+01 1.21801e+01 1.071623e+01
# > PVmREAD   5.033952e-05 4.295971e-05 3.554705e-05 3.436617e-05 4.627422e-05 0.0000386327 4.231929e-05 4.072383e-05 4.68876e-05 4.502253e-05

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef 
# > Intercept      PVmREAD 
# > 1.006880e+01 4.230726e-05

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef
# > Intercept      PVmREAD 
# > 1.016509e+01 3.523902e-05

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef
# > Intercept      PVmREAD 
# > 2.125039e+01 8.107019e-05

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef
# > Intercept     PVmREAD 
# > 4.609815050 0.009003898

# --- Rubin + BRR for R², Adjusted R², Sigma, F-statistic ---

# Sampling variance σ²(θ̂ₘ)
sampling_var_r2 <- mean(sapply(1:M, function(m) sum((rep_r2[, m] - main_r2s[m])^2) / (G * (1 - k)^2)))
sampling_var_adj_r2 <- mean(sapply(1:M, function(m) sum((rep_adj_r2[, m] - main_adj_r2s[m])^2) / (G * (1 - k)^2)))
sampling_var_sigma <- mean(sapply(1:M, function(m) sum((rep_sigma[, m] - main_sigmas[m])^2) / (G * (1 - k)^2)))
sampling_var_fstat_val <- mean(sapply(1:M, function(m) sum((rep_fstat_val[, m] - main_fstats_val[m])^2) / (G * (1 - k)^2)))

# Imputation variance σ²₍test₎
imputation_var_r2     <- sum((main_r2s - main_r2)    ^ 2) / (M - 1)
imputation_var_adj_r2 <- sum((main_adj_r2s - main_adj_r2)^ 2) / (M - 1)
imputation_var_sigma  <- sum((main_sigmas - main_sigma) ^ 2) / (M - 1)
imputation_var_fstat_val  <- sum((main_fstats_val - main_fstat_val) ^ 2) / (M - 1)

# Total error variance σ²₍error₎
var_final_r2     <- sampling_var_r2     + (1 + 1/M) * imputation_var_r2
var_final_adj_r2 <- sampling_var_adj_r2 + (1 + 1/M) * imputation_var_adj_r2
var_final_sigma  <- sampling_var_sigma  + (1 + 1/M) * imputation_var_sigma
var_final_fstat_val  <- sampling_var_fstat_val  + (1 + 1/M) * imputation_var_fstat_val

# Final standard error σ₍error₎ = sqrt(σ²₍error₎)
se_final_r2     <- sqrt(var_final_r2)
se_final_r2
# > [1] 0.009906041
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.00990647
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 3.567683
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 1193.757


## ---- Significance Tests ----

# Pooled coefficient estimates and standard errors from Rubin + BRR
Estimate <- main_coef                     # θ̂: mean coefficient across 10 plausible values (PVs)
`Std. Error` <- se_final_coef             # √(σ²₍error₎): Rubin + BRR total variance
`t value` <- Estimate / `Std. Error`      # Standard t-statistic: θ̂ / SE(θ̂)

# Degrees of freedom: residual df from one PV model (assumes same sample size across PVs)
dof_resid <- main_models[[1]]$dof_residual                          # df = n - p, where p includes intercept
all(sapply(main_models, function(m) m$dof_residual) == dof_resid)   # TRUE – ensure consistent df across PVs

# Two-sided p-values using t-distribution
p_t <- 2 * pt(-abs(`t value`), df = dof_resid)                         # Two-sided p-value: P(|T| ≥ |t_obs|) under H₀
`Pr(>|t|)` <- format.pval(p_t, digits = 3, eps = .Machine$double.eps)  # formatted for display
t_Signif <- symnum(p_t, corr = FALSE, na = FALSE,                      # significance stars (t-test)
                   cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                   symbols = c("***", "**", "*", ".", ""))

# Optional: z-test approximation (for large n)
z_val <- `t value`                                           # Z ≈ t for large samples n
p_z <- 2 * pnorm(-abs(z_val))                                # Z ∼ N(0, 1) under H₀ → p = 2·P(Z ≥ |z_obs|)
`Pr(>|z|)` <- format.pval(p_z, digits = 3, eps = .Machine$double.eps)
z_Signif <- symnum(p_z, corr = FALSE, na = FALSE,            # significance stars (z-test)
                   cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                   symbols = c("***", "**", "*", ".", ""))

# F-test for overall model significance
f_stat_val <- main_fstat_val                                  # Pooled F-statistic across PVs
dof1 <- length(main_coef) - 1                                 # Exclude intercept (p - 1 predictors)
dof2 <- dof_resid                                             # Residual df = n - p
f_p <- pf(f_stat_val, dof1, dof2, lower.tail = FALSE)         # One-sided p-value: P(F > F_obs), F ∼ F(dof1, dof2) under H₀
f_sig <- symnum(f_p, corr = FALSE, na = FALSE,                # Significance stars for F-test
                cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1), 
                symbols = c("***", "**", "*", ".", ""))


## ---- Final Output Table ----

# Coefficient and significance results (Rubin + BRR)
result_table <- tibble(
  Term = names(main_coef),                       # Coefficient names
  Estimate,                                      # Rubin’s Rule pooled estimates (θ̂)
  `Std. Error`,                                  # Total SE from Rubin + BRR
  `t value`,                                     # θ̂ / SE(θ̂)
  `Pr(>|t|)`,                                     # two-sided p-value under H₀: θ = 0, t(df)
  t_Signif = as.character(t_Signif),             # Significance stars (t-test)
  `z value` = z_val,                             # Normal approx of t for large samples
  `Pr(>|z|)` = `Pr(>|z|)`,                        # p-value under standard normal
  z_Signif = as.character(z_Signif)              # Significance stars (z-test)
)

# Append Model Fit Statistics 
append_rows <- tibble(
  Term = c("R-squared", "Adjusted R-squared", "Residual Std. Error", "F-statistic"),
  Estimate = round(c(main_r2, main_adj_r2, main_sigma, main_fstat_val), 2),                     # Final mean estimates/Point estimates
  `Std. Error` = round(c(se_final_r2, se_final_adj_r2, se_final_sigma, se_final_fstat_val), 2), # Rubin+BRR SEs
  `t value` = c(NA, NA, NA, NA),                                                                # NA: not meaningful for these metrics
  `Pr(>|t|)` = c(NA, NA, NA, NA),
  t_Signif = as.character(c(NA, NA, NA, NA)),
  `z value` = c(NA, NA, NA, NA),
  `Pr(>|z|)` = c(NA, NA, NA, NA),
  z_Signif = as.character(c(NA, NA, NA, NA)),
  `Pr(>F)` = c(NA, NA, NA, format.pval(f_p, digits = 3, eps = .Machine$double.eps)),             # Only F-stat has p-value
  F_Signif = as.character(c(NA, NA, NA, f_sig))                                                  # F-stat significance
)


# Combine all results 
result_table <- bind_rows(result_table, append_rows)

# Print the final results
print(result_table)
# A tibble: 10 × 11
# > Term                 Estimate `Std. Error` `t value` `Pr(>|t|)` t_Signif `z value` `Pr(>|z|)` z_Signif `Pr(>F)` F_Signif
# > <chr>                   <dbl>        <dbl>     <dbl> <chr>      <chr>        <dbl> <chr>      <chr>    <chr>    <chr>   
# > 1 Intercept             168.         4.61         36.4 <2e-16     ***           36.4 <2e-16     ***      NA       NA      
# > 2 PVmREAD                 0.649      0.00900      72.1 <2e-16     ***           72.1 <2e-16     ***      NA       NA      
# > 3 R-squared               0.56       0.01         NA   NA         NA            NA   NA         NA       NA       NA      
# > 4 Adjusted R-squared      0.56       0.01         NA   NA         NA            NA   NA         NA       NA       NA      
# > 5 Residual Std. Error   245.         3.57         NA   NA         NA            NA   NA         NA       NA       NA      
# > 6 F-statistic         29737.      1194.           NA   NA         NA            NA   NA         NA       <2e-16   ***     

print(as.data.frame(result_table), row.names = FALSE)
# >                Term     Estimate   Std. Error  t value Pr(>|t|) t_Signif  z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           Intercept   167.910971 4.609815e+00 36.42467   <2e-16      *** 36.42467   <2e-16      ***   <NA>     <NA>
# >             PVmREAD     0.648819 9.003898e-03 72.05979   <2e-16      *** 72.05979   <2e-16      ***   <NA>     <NA>
# >           R-squared     0.560000 1.000000e-02       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared     0.560000 1.000000e-02       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error   244.750000 3.570000e+00       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 29736.980000 1.193760e+03       NA     <NA>     <NA>       NA     <NA>     <NA> <2e-16      ***

options(scipen = 999) 
print(as.data.frame(result_table), row.names = FALSE)
# >               Term     Estimate     Std. Error  t value Pr(>|t|) t_Signif  z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           Intercept   167.910971    4.609815050 36.42467   <2e-16      *** 36.42467   <2e-16      ***   <NA>     <NA>
# >             PVmREAD     0.648819    0.009003898 72.05979   <2e-16      *** 72.05979   <2e-16      ***   <NA>     <NA>
# >           R-squared     0.560000    0.010000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared     0.560000    0.010000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error   244.750000    3.570000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 29736.980000 1193.760000000       NA     <NA>     <NA>       NA     <NA>     <NA> <2e-16      ***

## ---- Compare results with intsvy package ----
# ⚠️ Do not use pisa.reg.pv() to regress one set of PVs on another — this is not its intended use.
# Use IEA IDB Analyzer in SPSS Statistics for correct estimation in such cases.
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="PV1READ", data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   185.58       7.39   25.12
# > PV1READ         0.61       0.01   41.98
# > R-squared       0.50       0.02   23.94

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=paste0("PV",1:10,"READ"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   124.14       4.05   30.66
# > PV1READ         0.05       0.07    0.70
# > PV2READ         0.10       0.07    1.44
# > PV3READ         0.07       0.06    1.06
# > PV4READ         0.05       0.07    0.79
# > PV5READ         0.10       0.07    1.48
# > PV6READ         0.05       0.07    0.72
# > PV7READ         0.09       0.06    1.38
# > PV8READ         0.07       0.07    1.09
# > PV9READ         0.09       0.07    1.28
# > PV10READ        0.07       0.07    1.03
# > R-squared       0.62       0.01   86.19


## ---- ⚠️ TO REVISE: Limitations and Notes to Check (PISA Explanatory Modeling) ----

# 1. Z-tests preferred over t-tests: BRR with Fay’s method lacks well-defined df.
#    Use z-based p-values; keep t-values for reference only.

# 2. F-statistic p-value is exploratory: BRR + Fay does not justify using pf().
#    Report F-value descriptively; avoid formal F-test inference.

# 3. Residual df assumed constant: missing values filtered before modeling,
#    so residual degrees of freedom are consistent across PVs.

# 4. R², Adjusted R², and Sigma SEs valid via Rubin + BRR
#    (consistent with intsvy or IEA IDB Analyzer with SPSS),
#    but not for hypothesis testing—report descriptively.
#    Same applies when estimating RMSE or R² in predictive modeling.

# 5. When adding predictors: check missingness and weight coverage.
#    Ensure consistent samples across PVs and replicates for valid BRR.

# 6. Consider to explore missing data as one category and compare results.



# ---- III. Predictive Modelling: PVmMATH ~ PVmREAD ----
## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)  # For reading .sav SPSS files
library(dplyr)  # For data wrangling
library(tibble) # For tidy data representation
library(broom)  # For tidy model outputs
library(intsvy) # For analyzing PISA data

# Load dataset
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE)

# Explore missingness
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "MATH")]))
# > [1] 0
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "READ")]))
# > [1] 0

# Explore summary statistics
summary(pisa_2022_student_canada[, paste0("PV", 1:10, "MATH")])
summary(pisa_2022_student_canada[, paste0("PV", 1:10, "READ")])

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Data preparation
M <- 10  # Number of plausible values (PV1MATH to PV10MATH)
G <- 80  # Number of BRR replicate weights (W_FSTURWT1 to W_FSTURWT80)
k <- 0.5 # Fay's adjustment factor used in PISA's BRR method
z_crit <- qnorm(0.975) # 95% confidence z-score

pvmaths  <- paste0("PV", 1:M, "MATH")    # Plausible Values (PVs)in mathematics
pvreads  <- paste0("PV", 1:M, "READ")    # Plausible Values (PVs)in reading
rep_wts  <- paste0("W_FSTURWT", 1:G)     # 80 replicate weights
final_wt <- "W_FSTUWT"                   # Final student weight

temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID, W_FSTUWT,        # Select ID, weights
         all_of(rep_wts), all_of(pvmaths), all_of(pvreads))    # Include replicate weights + PVs


## ---- Random Train/Validation/Test (70/15/15) split ----
set.seed(123)  # Ensure reproducibility

n <- nrow(temp_data) # 23031
indices <- sample(n)  # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.70 * n)        # 16151
n_valid <- floor(0.15 * n)        # 3460
n_test  <- n - n_train - n_valid  # 3462

# Assign indices
train_idx <- indices[1:n_train]
valid_idx <- indices[(n_train + 1):(n_train + n_valid)]
test_idx  <- indices[(n_train + n_valid + 1):n]

# Subset the data
train_data <- temp_data[train_idx, ]
valid_data <- temp_data[valid_idx, ]
test_data  <- temp_data[test_idx, ]


# Check if any y_true = 0; add epsilon if needed to avoid division by zero
summary(temp_data[, paste0("PV", 1:10, "MATH")]) # min > 0 for all 

## ---- Fit on the training data----

### ---- Fit main models using final student weight (W_FSTUWT) ----
# For each plausible value (PV), fit: PVmMATH ~ PVmREAD using W_FSTUWT
main_models <- lapply(1:M, function(m) {
  formula <- as.formula(paste0(pvmaths[m], " ~ ", pvreads[m]))
  mod <- lm(formula, data = train_data, weights = train_data[[final_wt]])
  summ <- summary(mod)
  list(
    formula        = formula,              # Regression formula used
    mod            = mod,                  # Fitted lm model object
    summ           = summ,                 # Model summary (diagnostics, F-stat, etc.)
    coefs          = coef(mod),            # θ̂ₘ: Estimated coefficients (intercept, slope)
    r2             = summ$r.squared,       # R²: Proportion of variance explained
    adj_r2         = summ$adj.r.squared,   # Adjusted R²
    sigma          = summ$sigma,           # σ̂: Residual standard error
    fstat_val      = summ$fstatistic[["value"]],  # F-statistic value
    fstat_numdf    = summ$fstatistic[["numdf"]],  # Numerator degrees of freedom
    fstat_dendf    = summ$fstatistic[["dendf"]],  # Denominator degrees of freedom
    dof_model      = summ$df[1],           # Model DF (non-aliased coefficients)
    dof_residual   = summ$df[2],           # Residual DF = n - p
    dof_total      = summ$df[3]            # Total DF (model + residual)
  )
})
# Check first fitted model structure
main_models[[1]]
main_models[[2]]

main_models[[1]]$coefs
# > (Intercept)     PV1READ 
# > 168.4967811   0.6476235
main_models[[2]]$coefs
# > (Intercept)     PV2READ 
# > 167.6558437   0.6485666 

# --- Extract Estimates from Main Models (Rubin’s Step 1 & 2) ---

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs
# >      (Intercept)   PV1READ
# > [1,]    168.4968 0.6476235
# > [2,]    167.6558 0.6485666
# > [3,]    167.4904 0.6483407
# > [4,]    165.0476 0.6524144
# > [5,]    162.2177 0.6598362
# > [6,]    166.6128 0.6508750
# > [7,]    169.3038 0.6460187
# > [8,]    168.7008 0.6473543
# > [9,]    161.9530 0.6586683
# > [10,]   165.5841 0.6526231

# Rename for clarity
colnames(main_coefs) <- c("Intercept", "PVmREAD")
rownames(main_coefs) <- pvmaths  # pvmaths  <- paste0("PV", 1:M, "MATH") 
main_coefs
# >          Intercept   PVmREAD
# > PV1MATH   168.4968 0.6476235
# > PV2MATH   167.6558 0.6485666
# > PV3MATH   167.4904 0.6483407
# > PV4MATH   165.0476 0.6524144
# > PV5MATH   162.2177 0.6598362
# > PV6MATH   166.6128 0.6508750
# > PV7MATH   169.3038 0.6460187
# > PV8MATH   168.7008 0.6473543
# > PV9MATH   161.9530 0.6586683
# > PV10MATH  165.5841 0.6526231

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef
# > Intercept     PVmREAD 
# > 166.3062913   0.6512321

# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.5512440 0.5755210 0.5531083 0.5605723 0.5623448 0.5544815 0.5658856 0.5655798 0.5640306 0.5650070
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.5512163 0.5754947 0.5530806 0.5605451 0.5623177 0.5544539 0.5658587 0.5655529 0.5640036 0.5649801
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 249.2066 240.8420 249.8965 246.7194 245.1783 248.1717 243.2921 245.3726 246.4098 243.0467
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 19837.15 21895.29 19987.27 20601.07 20749.91 20098.65 21050.87 21024.68 20892.59 20975.74

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.5617775
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.5617504
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 245.8136
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 20711.32

### ---- Fit Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
replicate_models <- lapply(1:M, function(m) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste0(pvmaths[m], " ~ ", pvreads[m]))
    mod <- lm(formula, data = temp_data, weights = temp_data[[w]])
    summ <- summary(mod)
    list(
      formula       = formula,                          # Regression formula
      mod           = mod,                              # Fitted lm model
      summ          = summ,                             # Summary output
      coefs         = coef(mod),                        # θ̂ₘg: Coefficients (intercept, slope)
      r2            = summ$r.squared,                   # R²
      adj_r2        = summ$adj.r.squared,               # Adjusted R²
      sigma         = summ$sigma,                       # Residual standard error
      fstat_val     = summ$fstatistic[["value"]],       # F-statistic
      fstat_numdf   = summ$fstatistic[["numdf"]],       # Numerator df
      fstat_dendf   = summ$fstatistic[["dendf"]],       # Denominator df
      dof_model     = summ$df[1],                       # Degrees of freedom (model)
      dof_residual  = summ$df[2],                       # Degrees of freedom (residual)
      dof_total     = summ$df[3]                        # Total degrees of freedom
    )
  })
})
# Structure: replicate_models[[m]][[g]] — list of 10 PVs × 80 replicates
replicate_models[[1]][[1]]  # Inspect one model
replicate_models[[1]][[1]]$coefs
# > (Intercept)     PV1READ 
# >  167.601692    0.650925


# --- Extract replicate fit estimates (Rubin's step 1)---
# Coefficients: rep_coefs[[m]] = G × p matrix for plausible value m 
rep_coefs <- lapply(replicate_models, function(m) {
  do.call(rbind, lapply(m, function(g) coef(g$mod)))  # G × p matrix for each PV
})             # 80x2 matrix per PV
rep_coefs[[1]] # 80x2 matrix

# Fit statistics: each rep_stat[[stat]] is a G × M matrix (replicates × PVs)---
rep_r2     <- sapply(replicate_models, function(m) sapply(m, function(g) g$r2))             # 80x10 matrix
class(rep_r2);dim(rep_r2)
# > [1] "matrix" "array" 
# > [1] 80 10
rep_r2[[1]][[1]]  # Single r² value
rep_r2[1, ]       # Across PVs for replicate 1
rep_r2[, 1]       # All replicates for PV1
rep_adj_r2 <- sapply(replicate_models, function(m) sapply(m, function(g) g$adj_r2))         # 80x10 matrix
rep_sigma  <- sapply(replicate_models, function(m) sapply(m, function(g) g$sigma))          # 80x10 matrix
rep_fstat_val  <- sapply(replicate_models, function(m) sapply(m, function(g) g$fstat_val))  # 80x10 matrix

### ---- Rubin + BRR for Standard Errors (SEs) ----
# --- Rubin + BRR for Coefficients ---
sampling_var_matrix_coef <- sapply(1:M, function(m) {
  coefs_m <- rep_coefs[[m]]                   # G×p=80x2 matrix for PV m: each row is θ̂ₘ_g (replicate estimates for PV m)
  sweep(coefs_m, 2, main_coefs[m, ])^2 |>     # Compute  (θ̂ₘ_g - θ̂ₘ)^2 element-wise
    colSums() / (G * (1 - k)^2)               # Compute BRR sampling variance with Fay's adjustment for each coefficient
})
sampling_var_matrix_coef 
# >                     [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > (Intercept) 1.108475e+01 4.904826e+01 3.005867e+01 2.176285e+01 1.125089e+01 9.467223e+00 3.315353e+01 4.482626e+01 1.798093e+01 1.876894e+01
# > PV1READ     5.254853e-05 1.209492e-04 1.006244e-04 5.161310e-05 4.697731e-05 3.891577e-05 9.625737e-05 1.488613e-04 5.516170e-05 6.777545e-05

# Rename for clarity
colnames(sampling_var_matrix_coef) <- pvmaths  # pvmaths  <- paste0("PV", 1:M, "MATH") 
rownames(sampling_var_matrix_coef) <- c("Intercept", "PVmREAD")
sampling_var_matrix_coef
# >                PV1MATH      PV2MATH      PV3MATH      PV4MATH      PV5MATH      PV6MATH      PV7MATH      PV8MATH      PV9MATH     PV10MATH
# > Intercept 1.108475e+01 4.904826e+01 3.005867e+01 2.176285e+01 1.125089e+01 9.467223e+00 3.315353e+01 4.482626e+01 1.798093e+01 1.876894e+01
# > PVmREAD   5.254853e-05 1.209492e-04 1.006244e-04 5.161310e-05 4.697731e-05 3.891577e-05 9.625737e-05 1.488613e-04 5.516170e-05 6.777545e-0

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef 
# >    Intercept      PVmREAD 
# > 2.474023e+01 7.796842e-05 

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef
# >    Intercept      PVmREAD 
# > 6.734225e+00 2.261024e-05

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef
# >    Intercept      PVmREAD 
# > 3.214788e+01 1.028397e-04 

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef
# >  Intercept    PVmREAD 
# > 5.66990967 0.01014099

# --- Rubin + BRR for R², Adjusted R², Sigma, F-statistic ---

# Sampling variance σ²(θ̂ₘ)
sampling_var_r2 <- mean(sapply(1:M, function(m) sum((rep_r2[, m] - main_r2s[m])^2) / (G * (1 - k)^2)))
sampling_var_adj_r2 <- mean(sapply(1:M, function(m) sum((rep_adj_r2[, m] - main_adj_r2s[m])^2) / (G * (1 - k)^2)))
sampling_var_sigma <- mean(sapply(1:M, function(m) sum((rep_sigma[, m] - main_sigmas[m])^2) / (G * (1 - k)^2)))
sampling_var_fstat_val <- mean(sapply(1:M, function(m) sum((rep_fstat_val[, m] - main_fstats_val[m])^2) / (G * (1 - k)^2)))

# Imputation variance σ²₍test₎
imputation_var_r2     <- sum((main_r2s - main_r2)    ^ 2) / (M - 1)
imputation_var_adj_r2 <- sum((main_adj_r2s - main_adj_r2)^ 2) / (M - 1)
imputation_var_sigma  <- sum((main_sigmas - main_sigma) ^ 2) / (M - 1)
imputation_var_fstat_val  <- sum((main_fstats_val - main_fstat_val) ^ 2) / (M - 1)

# Total error variance σ²₍error₎
var_final_r2     <- sampling_var_r2     + (1 + 1/M) * imputation_var_r2
var_final_adj_r2 <- sampling_var_adj_r2 + (1 + 1/M) * imputation_var_adj_r2
var_final_sigma  <- sampling_var_sigma  + (1 + 1/M) * imputation_var_sigma
var_final_fstat_val  <- sampling_var_fstat_val  + (1 + 1/M) * imputation_var_fstat_val

# Final standard error σ₍error₎ = sqrt(σ²₍error₎)
se_final_r2     <- sqrt(var_final_r2)
se_final_r2
# > [1] 0.01230824
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.01231235
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 4.676993
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 18109.93


### ---- Significance Tests ----

# Pooled coefficient estimates and standard errors from Rubin + BRR
Estimate <- main_coef                     # θ̂: mean coefficient across 10 plausible values (PVs)
`Std. Error` <- se_final_coef             # √(σ²₍error₎): Rubin + BRR total variance
`t value` <- Estimate / `Std. Error`      # Standard t-statistic: θ̂ / SE(θ̂)

# Degrees of freedom: residual df from one PV model (assumes same sample size across PVs)
dof_resid <- main_models[[1]]$dof_residual                          # df = n - p, where p includes intercept
all(sapply(main_models, function(m) m$dof_residual) == dof_resid)   # TRUE – ensure consistent df across PVs

# Two-sided p-values using t-distribution
p_t <- 2 * pt(-abs(`t value`), df = dof_resid)                         # Two-sided p-value: P(|T| ≥ |t_obs|) under H₀
`Pr(>|t|)` <- format.pval(p_t, digits = 3, eps = .Machine$double.eps)  # formatted for display
t_Signif <- symnum(p_t, corr = FALSE, na = FALSE,                      # significance stars (t-test)
                   cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                   symbols = c("***", "**", "*", ".", ""))

# Optional: z-test approximation (for large n)
z_val <- `t value`                                           # Z ≈ t for large samples n
p_z <- 2 * pnorm(-abs(z_val))                                # Z ∼ N(0, 1) under H₀ → p = 2·P(Z ≥ |z_obs|)
`Pr(>|z|)` <- format.pval(p_z, digits = 3, eps = .Machine$double.eps)
z_Signif <- symnum(p_z, corr = FALSE, na = FALSE,            # significance stars (z-test)
                   cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                   symbols = c("***", "**", "*", ".", ""))

# F-test for overall model significance
f_stat_val <- main_fstat_val                                  # Pooled F-statistic across PVs
dof1 <- length(main_coef) - 1                                 # Exclude intercept (p - 1 predictors)
dof2 <- dof_resid                                             # Residual df = n - p
f_p <- pf(f_stat_val, dof1, dof2, lower.tail = FALSE)         # One-sided p-value: P(F > F_obs), F ∼ F(dof1, dof2) under H₀
f_sig <- symnum(f_p, corr = FALSE, na = FALSE,                # Significance stars for F-test
                cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1), 
                symbols = c("***", "**", "*", ".", ""))


### ---- Final Output Table ----

# Coefficient and significance results (Rubin + BRR)
result_table <- tibble(
  Term = names(main_coef),                       # Coefficient names
  Estimate,                                      # Rubin’s Rule pooled estimates (θ̂)
  `Std. Error`,                                  # Total SE from Rubin + BRR
  `t value`,                                     # θ̂ / SE(θ̂)
  `Pr(>|t|)`,                                     # two-sided p-value under H₀: θ = 0, t(df)
  t_Signif = as.character(t_Signif),             # Significance stars (t-test)
  `z value` = z_val,                             # Normal approx of t for large samples
  `Pr(>|z|)` = `Pr(>|z|)`,                        # p-value under standard normal
  z_Signif = as.character(z_Signif)              # Significance stars (z-test)
)

# Append Model Fit Statistics 
append_rows <- tibble(
  Term = c("R-squared", "Adjusted R-squared", "Residual Std. Error", "F-statistic"),
  Estimate = round(c(main_r2, main_adj_r2, main_sigma, main_fstat_val), 2),                     # Final mean estimates/Point estimates
  `Std. Error` = round(c(se_final_r2, se_final_adj_r2, se_final_sigma, se_final_fstat_val), 2), # Rubin+BRR SEs
  `t value` = c(NA, NA, NA, NA),                                                                # NA: not meaningful for these metrics
  `Pr(>|t|)` = c(NA, NA, NA, NA),
  t_Signif = as.character(c(NA, NA, NA, NA)),
  `z value` = c(NA, NA, NA, NA),
  `Pr(>|z|)` = c(NA, NA, NA, NA),
  z_Signif = as.character(c(NA, NA, NA, NA)),
  `Pr(>F)` = c(NA, NA, NA, format.pval(f_p, digits = 3, eps = .Machine$double.eps)),             # Only F-stat has p-value
  F_Signif = as.character(c(NA, NA, NA, f_sig))                                                  # F-stat significance
)

# Combine all results 
result_table <- bind_rows(result_table, append_rows)

# Print the final results
print(result_table)
# > # A tibble: 6 × 11
# > Term                 Estimate `Std. Error` `t value` `Pr(>|t|)` t_Signif `z value` `Pr(>|z|)` z_Signif `Pr(>F)` F_Signif
# > <chr>                   <dbl>        <dbl>     <dbl> <chr>      <chr>        <dbl> <chr>      <chr>    <chr>    <chr>   
# > 1 Intercept             166.          5.67        29.3 <2e-16     ***           29.3 <2e-16     ***      NA       NA      
# > 2 PVmREAD                 0.651       0.0101      64.2 <2e-16     ***           64.2 <2e-16     ***      NA       NA      
# > 3 R-squared               0.56        0.01        NA   NA         NA            NA   NA         NA       NA       NA      
# > 4 Adjusted R-squared      0.56        0.01        NA   NA         NA            NA   NA         NA       NA       NA      
# > 5 Residual Std. Error   246.          4.68        NA   NA         NA            NA   NA         NA       NA       NA      
# > 6 F-statistic         20711.      18110.          NA   NA         NA            NA   NA         NA       <2e-16   ***

print(as.data.frame(result_table), row.names = FALSE)
# >               Term     Estimate   Std. Error  t value Pr(>|t|) t_Signif  z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >          Intercept 1.663063e+02 5.669910e+00 29.33138   <2e-16      *** 29.33138   <2e-16      ***   <NA>     <NA>
# >            PVmREAD 6.512321e-01 1.014099e-02 64.21780   <2e-16      *** 64.21780   <2e-16      ***   <NA>     <NA>
# >          R-squared 5.600000e-01 1.000000e-02       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Adjusted R-squared 5.600000e-01 1.000000e-02       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error 2.458100e+02 4.680000e+00       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 2.071132e+04 1.810993e+04       NA     <NA>     <NA>       NA     <NA>     <NA> <2e-16      ***

options(scipen = 999) 
print(as.data.frame(result_table), row.names = FALSE)
# >                Term      Estimate     Std. Error  t value Pr(>|t|) t_Signif  z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           Intercept   166.3062913     5.66990967 29.33138   <2e-16      *** 29.33138   <2e-16      ***   <NA>     <NA>
# >             PVmREAD     0.6512321     0.01014099 64.21780   <2e-16      *** 64.21780   <2e-16      ***   <NA>     <NA>
# >           R-squared     0.5600000     0.01000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared     0.5600000     0.01000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error   245.8100000     4.68000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 20711.3200000 18109.93000000       NA     <NA>     <NA>       NA     <NA>     <NA> <2e-16      ***
options(scipen = 0)  # reset if needed


## ---- Predict and evaluate on the training data (Weighted, Rubin + BRR)----
### ---- Helper: Compute weighted predictive performance metrics ----
compute_metrics <- function(y_true, y_pred, w) {           # w: weight
  resid <- y_true - y_pred                                 # Residuals: resid = observed - predicted (=> predicted - observed = -resid);
  rmse <- sqrt(sum(w * (-resid)^2) / sum(w))               # Root Mean Squared Error: RMSE = sqrt[ sum(w * (predicted - observed)^2) / sum(w) ]
  mae  <- sum(w * abs(-resid)) / sum(w)                    # Mean Absolute Error: MAE = sum(w * |predicted - observed|) / sum(w)
  bias <- sum(w * (-resid)) / sum(w)                       # Bias = sum(w * (predicted - observed)) / sum(w) 
  bias_pct <- 100 * sum(w * (-resid / y_true)) / sum(w)    # Percentage bias: %Bias = 100 * sum(w * (predicted - observed) / observed) / sum(w)
  y_bar <- sum(w * y_true) / sum(w)                        # R-squared = 1 - SSE / SST
  sse <- sum(w * resid^2)
  sst <- sum(w * (y_true - y_bar)^2)
  r2 <- 1 - (sse / sst)
  return(c(rmse = rmse, mae = mae, bias = bias, bias_pct = bias_pct, r2 = r2))
}

### ---- Predict on training data (for each PV) using main_models ----
# Output: 10 × 5 data.frame with metrics for each plausible value
train_metrics_main <- sapply(main_models, function(model) {
  y_pred <- predict(model$mod, newdata = train_data)
  y_true <- train_data[[as.character(model$formula[[2]])]]
  w      <- train_data[[final_wt]]
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
class(train_metrics_main); dim(train_metrics_main)
# > [1] "data.frame"
# > [1] 10  5
train_metrics_main
# >        rmse      mae          bias bias_pct        r2
# > 1  63.10384 50.15158  4.497866e-12 1.743918 0.5512440
# > 2  60.98576 48.71205 -5.851528e-13 1.625593 0.5755210
# > 3  63.27853 50.44350  8.961208e-13 1.765015 0.5531083
# > 4  62.47404 49.83779  9.218179e-13 1.732070 0.5605723
# > 5  62.08379 49.40872  2.496717e-12 1.690511 0.5623448
# > 6  62.84178 49.84314 -1.624693e-12 1.742142 0.5544815
# > 7  61.60616 48.98690  1.615891e-12 1.686203 0.5658856
# > 8  62.13301 49.28540  3.224236e-13 1.694894 0.5655798
# > 9  62.39563 49.66973  1.064369e-12 1.716753 0.5640306
# > 10 61.54404 48.84057  2.046670e-12 1.654498 0.5650070

options(scipen = 999) 
train_metrics_main
# >        rmse      mae                   bias bias_pct        r2
# > 1  63.10384 50.15158  0.0000000000044978656 1.743918 0.5512440
# > 2  60.98576 48.71205 -0.0000000000005851528 1.625593 0.5755210
# > 3  63.27853 50.44350  0.0000000000008961208 1.765015 0.5531083
# > 4  62.47404 49.83779  0.0000000000009218179 1.732070 0.5605723
# > 5  62.08379 49.40872  0.0000000000024967174 1.690511 0.5623448
# > 6  62.84178 49.84314 -0.0000000000016246928 1.742142 0.5544815
# > 7  61.60616 48.98690  0.0000000000016158906 1.686203 0.5658856
# > 8  62.13301 49.28540  0.0000000000003224236 1.694894 0.5655798
# > 9  62.39563 49.66973  0.0000000000010643685 1.716753 0.5640306
# > 10 61.54404 48.84057  0.0000000000020466698 1.654498 0.5650070
options(scipen = 0) 

# Obtain final mean estimate 
main_metrics <- colMeans(train_metrics_main)
main_metrics
# >.       rmse          mae         bias     bias_pct           r2 
# > 6.224466e+01 4.951794e+01 1.165203e-12 1.705160e+00 5.617775e-01 

### ---- Replicate predictions for each plausible value and replicate weight using replicate_models ----
# Output: list of 10 matrices; each matrix is 80 × 5 (replicates × metrics)
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]
    y_pred <- predict(model$mod, newdata = train_data)
    y_true <- train_data[[as.character(model$formula[[2]])]]
    w      <- train_data[[rep_wts[g]]]
    compute_metrics(y_true, y_pred, w)
  }) |> t()
})
class(train_metrics_replicates);length(train_metrics_replicates); class(train_metrics_replicates[[1]]);dim(train_metrics_replicates[[1]])
# > [1] "list"
# > [1] 10
# > [1] "matrix" "array" 
# > [1] 80  5

### ---- Combine Rubin's Rules + BRR for SE uncertainty estimation  ----

# Sampling variance 
sampling_var_matrix <- sapply(1:M, function(m) {
  sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |> colMeans() / (G * (1 - k)^2)
}) 
dim(sampling_var_matrix)  # 5 metrics × 10 PVs
# > [1]  5 10
sampling_var_matrix
# >                  [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > rmse     3.688813e-03 3.058107e-03 5.442181e-03 3.401805e-03 3.780586e-03 3.393339e-03 3.277552e-03 4.869682e-03 4.223274e-03 3.452685e-03
# > mae      2.266818e-03 2.392451e-03 3.789768e-03 2.269130e-03 2.634054e-03 2.453042e-03 2.181378e-03 3.701251e-03 2.847056e-03 2.448152e-03
# > bias     1.040107e-02 3.676562e-02 4.776655e-03 3.283344e-02 6.466481e-03 1.697226e-03 1.458360e-02 5.404201e-03 1.207857e-02 3.260871e-03
# > bias_pct 4.282290e-04 1.871963e-03 3.225502e-04 1.607416e-03 3.011980e-04 8.946331e-05 8.048680e-04 3.575560e-04 5.942126e-04 1.856152e-04
# > r2       9.950482e-07 8.843083e-07 1.069893e-06 8.204290e-07 8.423250e-07 7.841801e-07 8.712392e-07 1.045844e-06 7.967555e-07 8.762600e-07

sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 3.858802e-03 2.698310e-03 1.282677e-02 6.563071e-04 8.986282e-0

# Imputation variance
imputation_var <- colSums((train_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 5.289718e-01 3.272829e-01 2.839892e-24 1.876438e-03 5.298212e-05

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 5.857278e-01 3.627095e-01 1.282677e-02 2.720389e-03 5.917896e-05

# Final standard error
se_final <- sqrt(var_final)
se_final
# >        rmse         mae        bias    bias_pct          r2 
# > 0.765328556 0.602253683 0.113255350 0.052157346 0.007692786

# Confidence interval
ci_lower <- main_metrics - z_crit * se_final
ci_upper <- main_metrics + z_crit * se_final
ci_length <- ci_upper - ci_lower

### ---- Final train output ----
train_eval <- tibble::tibble(
  Metric    = c("RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate  = format(main_metrics, scientific = FALSE),
  Standard_error  = format(se_final, scientific = FALSE),
  CI_lower  = format(ci_lower, scientific = FALSE),
  CI_upper  = format(ci_upper, scientific = FALSE),
  CI_length = format(ci_length, scientific = FALSE))

print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric        Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE 62.244657516468222980    0.765328556 60.7446411 63.7446739 3.00003281
# >       MAE 49.517937938763573413    0.602253683 48.3375424 50.6983335 2.36079106
# >      Bias  0.000000000001165203    0.113255350 -0.2219764  0.2219764 0.44395281
# >     Bias%  1.705159754784218107    0.052157346  1.6029332  1.8073863 0.20445304
# > R-squared  0.561777499361724830    0.007692786  0.5466999  0.5768551 0.03015517


## ---- Predict and evaluate on the validation data (Weighted, Rubin + BRR)----
### ---- Predict on validation data (for each PV) using main_models ----
# Output: 10 × 5 data.frame with metrics for each plausible value
valid_metrics_main <- sapply(main_models, function(model) {
  y_pred <- predict(model$mod, newdata = valid_data)
  y_true <- valid_data[[as.character(model$formula[[2]])]]
  w      <- valid_data[[final_wt]]
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
class(valid_metrics_main); dim(valid_metrics_main)
# > [1] "data.frame"
# > [1] 10  5
valid_metrics_main
# >        rmse      mae       bias  bias_pct        r2
# > 1  62.22909 49.24974  0.2099619 1.7759860 0.5574304
# > 2  62.57159 48.92075 -1.2777027 1.4113320 0.5476096
# > 3  61.62811 48.62093  0.4685773 1.6543450 0.5592154
# > 4  61.92906 48.69524 -3.1492033 0.9545467 0.5632489
# > 5  60.32558 47.95536  1.3087153 1.7806203 0.5733457
# > 6  63.07094 49.80819  1.5828613 2.0392927 0.5408599
# > 7  62.49159 49.60105 -0.2140550 1.5756427 0.5466911
# > 8  60.94731 48.78561 -0.4075335 1.4339139 0.5654491
# > 9  62.15018 49.20442  0.3304630 1.7054942 0.5595023
# > 10 61.40749 48.54761  0.2916660 1.6246277 0.5567662

# Obtain final mean estimate 
main_metrics <- colMeans(valid_metrics_main)
main_metrics
# >        rmse         mae        bias    bias_pct          r2 
# > 61.87509435 48.93889001 -0.08562497  1.59558012  0.55701187 

### ---- Replicate predictions for each plausible value and replicate weight using replicate_models ----
# Output: list of 10 matrices; each matrix is 80 × 5 (replicates × metrics)
valid_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]
    y_pred <- predict(model$mod, newdata = valid_data)
    y_true <- valid_data[[as.character(model$formula[[2]])]]
    w      <- valid_data[[rep_wts[g]]]
    compute_metrics(y_true, y_pred, w)
  }) |> t()
})
class(valid_metrics_replicates);length(valid_metrics_replicates); class(valid_metrics_replicates[[1]]);dim(valid_metrics_replicates[[1]])
# > [1] "list"
# > [1] 10
# > [1] "matrix" "array" 
# > [1] 80  5

### ---- Combine Rubin's Rules + BRR for SE uncertainty estimation  ----

# Sampling variance 
sampling_var_matrix <- sapply(1:M, function(m) {
  sweep(valid_metrics_replicates[[m]], 2, unlist(valid_metrics_main[m, ]))^2 |> colMeans() / (G * (1 - k)^2)
}) 
dim(sampling_var_matrix)  # 5 metrics × 10 PVs
# > [1]  5 10
sampling_var_matrix
# >                  [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > rmse     1.465377e-02 1.756825e-02 2.247274e-02 1.542002e-02 1.536346e-02 1.576427e-02 1.514891e-02 1.350638e-02 2.234746e-02 2.022194e-02
# > mae      1.032512e-02 9.754642e-03 1.256300e-02 9.897930e-03 8.206594e-03 1.169360e-02 1.080358e-02 9.531786e-03 1.074949e-02 1.125687e-02
# > bias     2.316926e-02 4.550073e-02 2.363730e-02 4.415787e-02 2.396086e-02 2.316239e-02 2.933110e-02 2.232289e-02 2.453385e-02 2.306627e-02
# > bias_pct 1.181943e-03 2.290753e-03 9.664905e-04 2.181155e-03 1.189709e-03 1.079460e-03 1.577817e-03 1.130321e-03 1.279484e-03 1.174791e-03
# > r2       2.757997e-06 2.526615e-06 3.216652e-06 2.454218e-06 2.597743e-06 3.149144e-06 2.687110e-06 1.883536e-06 2.769445e-06 3.244784e-06
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 1.724672e-02 1.047826e-02 2.828425e-02 1.405192e-03 2.728724e-06 

# Imputation variance
imputation_var <- colSums((valid_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >        rmse          mae         bias     bias_pct           r2 
# > 6.711714e-01 2.962213e-01 1.818300e+00 8.349443e-02 9.377545e-05

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 0.7555352408 0.3363217335 2.0284144838 0.0932490658 0.0001058817  

# Final standard error
se_final <- sqrt(var_final)
se_final
# >       rmse        mae       bias   bias_pct         r2 
# > 0.86921530 0.57993252 1.42422417 0.30536710 0.01028988

# Confidence interval
ci_lower <- main_metrics - z_crit * se_final
ci_upper <- main_metrics + z_crit * se_final
ci_length <- ci_upper - ci_lower

### ---- Final validation output ----
valid_eval <- tibble::tibble(
  Metric    = c("RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate  = format(main_metrics, scientific = FALSE),
  Standard_error  = format(se_final, scientific = FALSE),
  CI_lower  = format(ci_lower, scientific = FALSE),
  CI_upper  = format(ci_upper, scientific = FALSE),
  CI_length = format(ci_length, scientific = FALSE))

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error   CI_lower   CI_upper CI_length
# >      RMSE    61.87509435     0.86921530 60.1714637 63.5787250 3.4072614
# >       MAE    48.93889001     0.57993252 47.8022432 50.0755369 2.2732937
# >      Bias    -0.08562497     1.42422417 -2.8770530  2.7058031 5.5828562
# >     Bias%     1.59558012     0.30536710  0.9970716  2.1940886 1.1970170
# > R-squared     0.55701187     0.01028988  0.5368441  0.5771797 0.0403356

## ---- Predict and evaluate on the test data (Weighted, Rubin + BRR)----
### ---- Predict on test data (for each PV) using main_models ----
# Output: 10 × 5 data.frame with metrics for each plausible value
test_metrics_main <- sapply(main_models, function(model) {
  y_pred <- predict(model$mod, newdata = test_data)
  y_true <- test_data[[as.character(model$formula[[2]])]]
  w      <- test_data[[final_wt]]
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
class(test_metrics_main); dim(test_metrics_main)
# > [1] "data.frame"
# > [1] 10  5
test_metrics_main
# >        rmse      mae      bias  bias_pct        r2
# > 1  64.14636 50.51715 -2.869676 1.1841280 0.5530537
# > 2  61.90479 48.74765 -4.215572 0.7080923 0.5720376
# > 3  62.66340 49.75878 -1.995045 1.2078528 0.5615950
# > 4  62.55764 50.09092 -2.074329 1.2415148 0.5687363
# > 5  60.90761 48.86434 -3.494948 0.8862391 0.5998444
# > 6  61.20976 49.10308 -1.127939 1.3741271 0.5774812
# > 7  60.76353 48.42551 -3.016466 0.9201291 0.5839203
# > 8  63.23471 50.21673 -1.256139 1.3843580 0.5607835
# > 9  61.48333 48.87214 -3.246644 0.9021738 0.5797216
# > 10 60.26515 47.48469 -1.228274 1.3042461 0.5824585

# Obtain final mean estimate 
main_metrics <- colMeans(test_metrics_main)
main_metrics
# > rmse        mae       bias   bias_pct         r2 
# > 61.9136285 49.2080984 -2.4525032  1.1112861  0.5739632

### ---- Replicate predictions for each plausible value and replicate weight using replicate_models ----
# Output: list of 10 matrices; each matrix is 80 × 5 (replicates × metrics)
test_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]
    y_pred <- predict(model$mod, newdata = test_data)
    y_true <- test_data[[as.character(model$formula[[2]])]]
    w      <- test_data[[rep_wts[g]]]
    compute_metrics(y_true, y_pred, w)
  }) |> t()
})
class(test_metrics_replicates);length(test_metrics_replicates); class(test_metrics_replicates[[1]]);dim(test_metrics_replicates[[1]])
# > [1] "list"
# > [1] 10
# > [1] "matrix" "array" 
# > [1] 80  5

### ---- Combine Rubin's Rules + BRR for SE uncertainty estimation  ----

# Sampling variance 
sampling_var_matrix <- sapply(1:M, function(m) {
  sweep(test_metrics_replicates[[m]], 2, unlist(test_metrics_main[m, ]))^2 |> colMeans() / (G * (1 - k)^2)
}) 
dim(sampling_var_matrix)  # 5 metrics × 10 PVs
# > [1]  5 10
sampling_var_matrix
# > [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > rmse     1.450486e-02 1.582581e-02 1.434909e-02 1.403979e-02 1.348575e-02 1.675089e-02 1.374773e-02 1.812376e-02 1.465491e-02 1.620818e-02
# > mae      9.890177e-03 1.193565e-02 9.906466e-03 1.008935e-02 1.046289e-02 1.375237e-02 9.181900e-03 1.337274e-02 9.013683e-03 1.067409e-02
# > bias     3.617683e-02 5.822479e-02 2.843299e-02 6.384486e-02 3.048412e-02 2.411273e-02 4.464846e-02 3.359568e-02 3.877643e-02 2.744902e-02
# > bias_pct 1.531786e-03 2.810715e-03 1.280740e-03 3.014417e-03 1.238149e-03 1.076878e-03 2.114807e-03 1.654820e-03 1.686490e-03 1.191081e-03
# > r2       3.003974e-06 2.646852e-06 2.920357e-06 3.177907e-06 2.219372e-06 2.671256e-06 2.746930e-06 2.847618e-06 2.513377e-06 2.404245e-06
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 1.516908e-02 1.082793e-02 3.857459e-02 1.759988e-03 2.715189e-06

# Imputation variance
imputation_var <- colSums((test_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 1.4933574410 0.8722478045 1.1505142470 0.0562175706 0.0001868442

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 1.6578622621 0.9703005166 1.3041402621 0.0635993160 0.000208243

# Final standard error
se_final <- sqrt(var_final)
se_final
# >       rmse        mae       bias   bias_pct         r2 
# > 1.28758000 0.98503833 1.14198961 0.25218905 0.01443065 

# Confidence interval
ci_lower <- main_metrics - z_crit * se_final
ci_upper <- main_metrics + z_crit * se_final
ci_length <- ci_upper - ci_lower

### ---- Final test output ----
test_eval <- tibble::tibble(
  Metric    = c("RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate  = format(main_metrics, scientific = FALSE),
  Standard_error  = format(se_final, scientific = FALSE),
  CI_lower  = format(ci_lower, scientific = FALSE),
  CI_upper  = format(ci_upper, scientific = FALSE),
  CI_length = format(ci_length, scientific = FALSE))

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE     61.9136285     1.28758000 59.3900181 64.4372389 5.04722086
# >       MAE     49.2080984     0.98503833 47.2774588 51.1387381 3.86127931
# >      Bias     -2.4525032     1.14198961 -4.6907617 -0.2142447 4.47651700
# >     Bias%      1.1112861     0.25218905  0.6170047  1.6055676 0.98856290
# > R-squared      0.5739632     0.01443065  0.5456796  0.6022468 0.05656713


## ---- Summary ----

print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric        Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE 62.244657516468222980    0.765328556 60.7446411 63.7446739 3.00003281
# >       MAE 49.517937938763573413    0.602253683 48.3375424 50.6983335 2.36079106
# >      Bias  0.000000000001165203    0.113255350 -0.2219764  0.2219764 0.44395281
# >     Bias%  1.705159754784218107    0.052157346  1.6029332  1.8073863 0.20445304
# > R-squared  0.561777499361724830    0.007692786  0.5466999  0.5768551 0.03015517

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error   CI_lower   CI_upper CI_length
# >      RMSE    61.87509435     0.86921530 60.1714637 63.5787250 3.4072614
# >       MAE    48.93889001     0.57993252 47.8022432 50.0755369 2.2732937
# >      Bias    -0.08562497     1.42422417 -2.8770530  2.7058031 5.5828562
# >     Bias%     1.59558012     0.30536710  0.9970716  2.1940886 1.1970170
# > R-squared     0.55701187     0.01028988  0.5368441  0.5771797 0.0403356

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE     61.9136285     1.28758000 59.3900181 64.4372389 5.04722086
# >       MAE     49.2080984     0.98503833 47.2774588 51.1387381 3.86127931
# >      Bias     -2.4525032     1.14198961 -4.6907617 -0.2142447 4.47651700
# >     Bias%      1.1112861     0.25218905  0.6170047  1.6055676 0.98856290
# > R-squared      0.5739632     0.01443065  0.5456796  0.6022468 0.05656713
