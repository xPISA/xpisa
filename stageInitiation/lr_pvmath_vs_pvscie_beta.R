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
#    and plausible values in science (PV1SCIE–PV10SCIE), using official formulas not directly supported 
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
#  data, using plausible values in science (PV1READ-PV10READ) as the X 
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


# ---- I. Correlation: PVmMATH ~ PVmSCIE ----

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
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "SCIE")]))
# > [1] 0

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Data preparation
M <- 10  # Number of plausible values (PV1MATH to PV10MATH)
G <- 80  # Number of BRR replicate weights (W_FSTURWT1 to W_FSTURWT80)
k <- 0.5 # Fay's adjustment factor used in PISA's BRR method

pvmaths  <- paste0("PV", 1:M, "MATH")    # Plausible Values (PVs) in mathematics
pvscies  <- paste0("PV", 1:M, "SCIE")    # Plausible Values (PVs) in reading
rep_wts  <- paste0("W_FSTURWT", 1:G)     # 80 replicate weights
final_wt <- "W_FSTUWT"                   # Final student weight

temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID, W_FSTUWT,        # Select ID, weights
         all_of(rep_wts), all_of(pvmaths), all_of(pvscies))    # Include replicate weights + PVs


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
  get_wtd_corr(temp_data[[pvmaths[m]]], temp_data[[pvscies[m]]], temp_data[[final_wt]])
})
main_corrs
# > [1] 0.8122608 0.8032595 0.8032586 0.8011411 0.8017726 0.8057819 0.7983808 0.8065182 0.8036891 0.8021107

# --- Compute mean correlation across PVs (Rubin's rule) ---
main_corr <- mean(main_corrs)  # ρ̂
main_corr
# > [1] 0.8038173

# --- Compute replicate-based correlation matrix (M × G) ---
rep_corr_matrix <- matrix(NA_real_, nrow = M, ncol = G)
for (m in 1:M) {
  for (g in 1:G) {
    rep_corr_matrix[m, g] <- get_wtd_corr(
      temp_data[[pvmaths[m]]],
      temp_data[[pvscies[m]]],
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
# >  [1] 1.431707e-05 1.132194e-05 1.523736e-05 1.485048e-05 1.460584e-05 1.430324e-05 1.803374e-05 1.275028e-05 1.246834e-05 1.586267e-05

# --- Compute average sampling variance across PVs ---
sampling_var_corr <- mean(sampling_var_corr_per_pv)  # σ²(ρ̂)
sampling_var_corr
# > [1] 1.43751e-05

# --- Compute imputation variance using Rubin's rule ---
imputation_var_corr <- sum((main_corrs - main_corr)^2) / (M - 1)  # σ²(test) = (1/(M-1)) * Σ (ρ̂ₘ -ρ̂)²
imputation_var_corr
# > [1] 1.409981e-05

# --- Compute final total error variance and standard error ---
var_final_corr <- sampling_var_corr + (1 + 1/M) * imputation_var_corr  # σ²(error)
var_final_corr
# > [1] 2.988489e-05

se_final_corr <- sqrt(var_final_corr)                                  # sqrt(σ²(error))
se_final_corr
# > [1] 0.005466707

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
# >              Mean correlation ρ̂ 0.803817
# >         Sampling variance σ²(ρ̂) 0.000014
# >    Imputation variance σ²(test) 0.000014
# >  Total error variance σ²(error) 0.000030
# > Final standard error √σ²(error) 0.005467








# ---- II. Explanatory Modelling: PVmMATH ~ PVmSCIE ----

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
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "SCIE")]))

# Explore summary statistics
summary(pisa_2022_student_canada[, paste0("PV", 1:10, "MATH")])
summary(pisa_2022_student_canada[, paste0("PV", 1:10, "SCIE")])

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Data preparation
M <- 10  # Number of plausible values (PV1MATH to PV10MATH)
G <- 80  # Number of BRR replicate weights (W_FSTURWT1 to W_FSTURWT80)
k <- 0.5 # Fay's adjustment factor used in PISA's BRR method

pvmaths  <- paste0("PV", 1:M, "MATH")    # Plausible Values (PVs)in mathematics
pvscies  <- paste0("PV", 1:M, "SCIE")    # Plausible Values (PVs)in science
rep_wts  <- paste0("W_FSTURWT", 1:G)     # 80 replicate weights
final_wt <- "W_FSTUWT"                   # Final student weight

temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID, W_FSTUWT,        # Select ID, weights
         all_of(rep_wts), all_of(pvmaths), all_of(pvscies))    # Include replicate weights + PVs

## ---- Initial Exploration----

# Fit weighted linear model: Math PV1 ~ Gender 
mod <- lm(PV1MATH ~ PV1SCIE, data = temp_data, weights = temp_data[[final_wt]]) 

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
  formula <- as.formula(paste0(pvmaths[m], " ~ ", pvscies[m]))
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
# >      (Intercept)   PV1SCIE
# > [1,]    105.6125 0.7613151
# > [2,]    111.4232 0.7500320
# > [3,]    104.7815 0.7630126
# > [4,]    113.0297 0.7449543
# > [5,]    113.9861 0.7436173
# > [6,]    104.6950 0.7603241
# > [7,]    114.9445 0.7411275
# > [8,]    106.4634 0.7574539
# > [9,]    110.2725 0.7507891
# > [10,]   114.5470 0.7412248

# Rename for clarity
colnames(main_coefs) <- c("Intercept", "PVmSCIE")
rownames(main_coefs) <- pvmaths  # pvmaths  <- paste0("PV", 1:M, "MATH") 
main_coefs
# >          Intercept   PVmSCIE
# > PV1MATH   105.6125 0.7613151
# > PV2MATH   111.4232 0.7500320
# > PV3MATH   104.7815 0.7630126
# > PV4MATH   113.0297 0.7449543
# > PV5MATH   113.9861 0.7436173
# > PV6MATH   104.6950 0.7603241
# > PV7MATH   114.9445 0.7411275
# > PV8MATH   106.4634 0.7574539
# > PV9MATH   110.2725 0.7507891
# > PV10MATH  114.5470 0.7412248

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef
# >    Intercept     PVmSCIE 
# > 109.9755230   0.7513851 

# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.6597676 0.6452259 0.6452244 0.6418270 0.6428393 0.6492845 0.6374119 0.6504716 0.6459162 0.6433816
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.6597529 0.6452105 0.6452090 0.6418115 0.6428238 0.6492693 0.6373961 0.6504564 0.6459009 0.6433662
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 216.8158 219.7843 221.4453 222.3527 221.2755 219.2607 221.7794 219.3089 221.3246 219.1027
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 44738.54 41959.11 41958.84 41342.02 41524.57 42711.66 40557.67 42935.08 42085.90 41622.81

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.646135
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.6461197
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 220.245
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 42143.62

## ---- Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
replicate_models <- lapply(1:M, function(m) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste0(pvmaths[m], " ~ ", pvscies[m]))
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
# > (Intercept)     PV1SCIE 
# > 105.8191385   0.7603428
replicate_models[[2]][[1]]$coefs
# > (Intercept)     PV2SCIE 
# > 110.5747731   0.7522028 


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
# >                     [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > (Intercept) 9.473264e+00 6.617298e+00 1.393712e+01 1.135282e+01 1.033106e+01 9.384365e+00 9.673194e+00 6.878254e+00 1.159006e+01 8.458564e+00
# > PV1SCIE     3.976013e-05 2.711446e-05 5.469013e-05 4.466815e-05 4.128424e-05 3.584322e-05 3.863106e-05 2.960818e-05 4.402015e-05 3.760529e-0

# Rename for clarity
colnames(sampling_var_matrix_coef) <- pvmaths  # pvmaths  <- paste0("PV", 1:M, "MATH") 
rownames(sampling_var_matrix_coef) <- c("Intercept", "PVmSCIE")
sampling_var_matrix_coef
# >                PV1MATH      PV2MATH      PV3MATH      PV4MATH      PV5MATH      PV6MATH      PV7MATH      PV8MATH      PV9MATH     PV10MATH
# > Intercept 9.473264e+00 6.617298e+00 1.393712e+01 1.135282e+01 1.033106e+01 9.384365e+00 9.673194e+00 6.878254e+00 1.159006e+01 8.458564e+00
# > PVmSCIE   3.976013e-05 2.711446e-05 5.469013e-05 4.466815e-05 4.128424e-05 3.584322e-05 3.863106e-05 2.960818e-05 4.402015e-05 3.760529e-05

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef 
# >    Intercept      PVmSCIE 
# > 9.7695987627 0.0000393225

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef
# >    Intercept      PVmSCIE 
# > 1.771325e+01 7.365247e-05 

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef
# >    Intercept      PVmSCIE 
# > 2.925417e+01 1.203402e-04 

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef
# >  Intercept    PVmSCIE 
# > 5.40871243 0.01096997

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
# > [1] 0.008796741
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.008797123
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 2.81066
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 1634.763


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
# > # A tibble: 6 × 11
# > Term                 Estimate `Std. Error` `t value` `Pr(>|t|)` t_Signif `z value` `Pr(>|z|)` z_Signif `Pr(>F)` F_Signif
# > <chr>                   <dbl>        <dbl>     <dbl> <chr>      <chr>        <dbl> <chr>      <chr>    <chr>    <chr>   
# > 1 Intercept             110.          5.41        20.3 <2e-16     ***           20.3 <2e-16     ***      NA       NA      
# > 2 PVmSCIE                 0.751       0.0110      68.5 <2e-16     ***           68.5 <2e-16     ***      NA       NA      
# > 3 R-squared               0.65        0.01        NA   NA         NA            NA   NA         NA       NA       NA      
# > 4 Adjusted R-squared      0.65        0.01        NA   NA         NA            NA   NA         NA       NA       NA      
# > 5 Residual Std. Error   220.          2.81        NA   NA         NA            NA   NA         NA       NA       NA      
# > 6 F-statistic         42144.       1635.          NA   NA         NA            NA   NA         NA       <2e-16   ***   

print(as.data.frame(result_table), row.names = FALSE)
# >                Term     Estimate   Std. Error  t value Pr(>|t|) t_Signif  z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           Intercept 1.099755e+02 5.408712e+00 20.33303   <2e-16      *** 20.33303   <2e-16      ***   <NA>     <NA>
# >             PVmSCIE 7.513851e-01 1.096997e-02 68.49473   <2e-16      *** 68.49473   <2e-16      ***   <NA>     <NA>
# >           R-squared 6.500000e-01 1.000000e-02       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared 6.500000e-01 1.000000e-02       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error 2.202400e+02 2.810000e+00       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 4.214362e+04 1.634760e+03       NA     <NA>     <NA>       NA     <NA>     <NA> <2e-16      ***

options(scipen = 999) 
print(as.data.frame(result_table), row.names = FALSE)
# >                Term      Estimate    Std. Error  t value Pr(>|t|) t_Signif  z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           Intercept   109.9755230    5.40871243 20.33303   <2e-16      *** 20.33303   <2e-16      ***   <NA>     <NA>
# >             PVmSCIE     0.7513851    0.01096997 68.49473   <2e-16      *** 68.49473   <2e-16      ***   <NA>     <NA>
# >           R-squared     0.6500000    0.01000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared     0.6500000    0.01000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error   220.2400000    2.81000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 42143.6200000 1634.76000000       NA     <NA>     <NA>       NA     <NA>     <NA> <2e-16      ***

## ---- Compare results with intsvy package ----
# ⚠️ Do not use pisa.reg.pv() to regress one set of PVs on another — this is not its intended use.
# Use IEA IDB Analyzer in SPSS Statistics for correct estimation in such cases.
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="PV1SCIE", data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   129.25       9.37   13.80
# > PV1SCIE         0.72       0.02   39.25
# > R-squared       0.59       0.03   21.19

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=paste0("PV",1:10,"SCIE"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)    61.20       4.09   14.95
# > PV1SCIE         0.10       0.08    1.24
# > PV2SCIE         0.08       0.08    1.00
# > PV3SCIE         0.10       0.08    1.20
# > PV4SCIE         0.07       0.08    0.83
# > PV5SCIE         0.07       0.09    0.84
# > PV6SCIE         0.10       0.09    1.13
# > PV7SCIE         0.07       0.08    0.95
# > PV8SCIE         0.09       0.08    1.07
# > PV9SCIE         0.09       0.08    1.10
# > PV10SCIE        0.07       0.08    0.87
# > R-squared       0.70       0.01   93.84


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



# ---- III. Predictive Modelling: PVmMATH ~ PVmSCIE ----
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
sum(is.na(pisa_2022_student_canada[, paste0("PV", 1:10, "SCIE")]))
# > [1] 0

# Explore summary statistics
summary(pisa_2022_student_canada[, paste0("PV", 1:10, "MATH")])
summary(pisa_2022_student_canada[, paste0("PV", 1:10, "SCIE")])

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Data preparation
M <- 10  # Number of plausible values (PV1MATH to PV10MATH)
G <- 80  # Number of BRR replicate weights (W_FSTURWT1 to W_FSTURWT80)
k <- 0.5 # Fay's adjustment factor used in PISA's BRR method
z_crit <- qnorm(0.975) # 95% confidence z-score

pvmaths  <- paste0("PV", 1:M, "MATH")    # Plausible Values (PVs)in mathematics
pvscies  <- paste0("PV", 1:M, "SCIE")    # Plausible Values (PVs)in reading
rep_wts  <- paste0("W_FSTURWT", 1:G)     # 80 replicate weights
final_wt <- "W_FSTUWT"                   # Final student weight

temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID, W_FSTUWT,        # Select ID, weights
         all_of(rep_wts), all_of(pvmaths), all_of(pvscies))    # Include replicate weights + PVs


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
  formula <- as.formula(paste0(pvmaths[m], " ~ ", pvscies[m]))
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
# > (Intercept)     PV1SCIE 
# > 105.5345631   0.7605606
main_models[[2]]$coefs
# > (Intercept)     PV2SCIE 
# > 110.2483807   0.7513949 

# --- Extract Estimates from Main Models (Rubin’s Step 1 & 2) ---

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs
# >.     (Intercept)   PV1SCIE
# > [1,]    105.5346 0.7605606
# > [2,]    110.2484 0.7513949
# > [3,]    101.2788 0.7687996
# > [4,]    113.2730 0.7427197
# > [5,]    112.6814 0.7456617
# > [6,]    102.1462 0.7645854
# > [7,]    112.5610 0.7446504
# > [8,]    103.2371 0.7630446
# > [9,]    107.2003 0.7551732
# > [10,]   113.1437 0.7432894

# Rename for clarity
colnames(main_coefs) <- c("Intercept", "PVmSCIE")
rownames(main_coefs) <- pvmaths  # pvmaths  <- paste0("PV", 1:M, "MATH") 
main_coefs
# >          Intercept   PVmSCIE
# > PV1MATH   105.5346 0.7605606
# > PV2MATH   110.2484 0.7513949
# > PV3MATH   101.2788 0.7687996
# > PV4MATH   113.2730 0.7427197
# > PV5MATH   112.6814 0.7456617
# > PV6MATH   102.1462 0.7645854
# > PV7MATH   112.5610 0.7446504
# > PV8MATH   103.2371 0.7630446
# > PV9MATH   107.2003 0.7551732
# > PV10MATH  113.1437 0.7432894

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef
# >   Intercept     PVmSCIE 
# > 108.1304419   0.7539879 

# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.6585752 0.6442755 0.6459810 0.6405934 0.6386489 0.6480406 0.6389354 0.6497359 0.6448638 0.6415214
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.6585540 0.6442535 0.6459591 0.6405712 0.6386266 0.6480188 0.6389131 0.6497142 0.6448418 0.6414992
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 217.3712 220.4758 222.4192 223.1273 222.7823 220.5797 221.8801 220.3276 222.3962 220.6380
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 31149.84 29248.50 29467.20 28783.40 28541.61 29734.14 28577.07 29956.21 29323.70 28899.71

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.6451171
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.6450951
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 221.1997
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 29368.14

### ---- Fit Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
replicate_models <- lapply(1:M, function(m) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste0(pvmaths[m], " ~ ", pvscies[m]))
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
# > (Intercept)     PV1SCIE 
# > 105.8191385   0.7603428
replicate_models[[2]][[1]]$coefs
# > (Intercept)     PV2SCIE 
# > 110.5747731   0.7522028


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
# > (Intercept) 9.517867e+00 1.264035e+01 6.512096e+01 1.215761e+01 1.591515e+01 37.502661408 3.390462e+01 5.102979e+01 4.988273e+01 1.551808e+01
# > PV1SCIE     4.214133e-05 3.499235e-05 1.930028e-04 7.443983e-05 5.227955e-05  0.000114084 9.092621e-05 1.617542e-04 1.210956e-04 5.012077e-05

# Rename for clarity
colnames(sampling_var_matrix_coef) <- pvmaths  # pvmaths  <- paste0("PV", 1:M, "MATH") 
rownames(sampling_var_matrix_coef) <- c("Intercept", "PVmSCIE")
sampling_var_matrix_coef
# >                PV1MATH      PV2MATH      PV3MATH      PV4MATH      PV5MATH      PV6MATH      PV7MATH      PV8MATH      PV9MATH     PV10MATH
# > Intercept 9.517867e+00 1.264035e+01 6.512096e+01 1.215761e+01 1.591515e+01 37.502661408 3.390462e+01 5.102979e+01 4.988273e+01 1.551808e+01
# > PVmSCIE   4.214133e-05 3.499235e-05 1.930028e-04 7.443983e-05 5.227955e-05  0.000114084 9.092621e-05 1.617542e-04 1.210956e-04 5.012077e-05

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef 
# >    Intercept      PVmSCIE 
# > 3.031898e+01 9.348367e-05 

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef
# >    Intercept      PVmSCIE 
# > 2.341209e+01 9.588768e-05

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef
# >    Intercept      PVmSCIE 
# > 5.607228e+01 1.989601e-04

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef
# >  Intercept    PVmSCIE 
# > 7.48814266 0.01410532 

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
# > [1] 0.009551118
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.009554541
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 3.495556
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 25624.17


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
# A tibble: 6 × 11
# > Term                 Estimate `Std. Error` `t value` `Pr(>|t|)` t_Signif `z value` `Pr(>|z|)` z_Signif `Pr(>F)` F_Signif
# > <chr>                   <dbl>        <dbl>     <dbl> <chr>      <chr>        <dbl> <chr>      <chr>    <chr>    <chr>   
# > 1 Intercept             108.          7.49        14.4 <2e-16     ***           14.4 <2e-16     ***      NA       NA      
# > 2 PVmSCIE                 0.754       0.0141      53.5 <2e-16     ***           53.5 <2e-16     ***      NA       NA      
# > 3 R-squared               0.65        0.01        NA   NA         NA            NA   NA         NA       NA       NA      
# > 4 Adjusted R-squared      0.65        0.01        NA   NA         NA            NA   NA         NA       NA       NA      
# > 5 Residual Std. Error   221.          3.5         NA   NA         NA            NA   NA         NA       NA       NA      
# > 6 F-statistic         29368.      25624.          NA   NA         NA            NA   NA         NA       <2e-16   ***   

print(as.data.frame(result_table), row.names = FALSE)
# >                Term     Estimate   Std. Error  t value Pr(>|t|) t_Signif  z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           Intercept 1.081304e+02 7.488143e+00 14.44022   <2e-16      *** 14.44022   <2e-16      ***   <NA>     <NA>
# >             PVmSCIE 7.539879e-01 1.410532e-02 53.45415   <2e-16      *** 53.45415   <2e-16      ***   <NA>     <NA>
# >           R-squared 6.500000e-01 1.000000e-02       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared 6.500000e-01 1.000000e-02       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error 2.212000e+02 3.500000e+00       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 2.936814e+04 2.562417e+04       NA     <NA>     <NA>       NA     <NA>     <NA> <2e-16      ***

options(scipen = 999) 
print(as.data.frame(result_table), row.names = FALSE)
# >                Term      Estimate     Std. Error  t value Pr(>|t|) t_Signif  z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           Intercept   108.1304419     7.48814266 14.44022   <2e-16      *** 14.44022   <2e-16      ***   <NA>     <NA>
# >             PVmSCIE     0.7539879     0.01410532 53.45415   <2e-16      *** 53.45415   <2e-16      ***   <NA>     <NA>
# >           R-squared     0.6500000     0.01000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared     0.6500000     0.01000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error   221.2000000     3.50000000       NA     <NA>     <NA>       NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 29368.1400000 25624.17000000       NA     <NA>     <NA>       NA     <NA>     <NA> <2e-16      ***
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
# > 1  55.04251 43.60905  4.764873e-13 1.346121 0.6585752
# > 2  55.82866 44.17177 -1.272858e-12 1.382138 0.6442755
# > 3  56.32076 44.38433  2.770843e-12 1.413128 0.6459810
# > 4  56.50006 44.83254  4.307044e-12 1.426601 0.6405934
# > 5  56.41270 44.65912 -6.001662e-13 1.424082 0.6386489
# > 6  55.85497 44.32494  3.552897e-13 1.400207 0.6480406
# > 7  56.18424 44.40822  4.132062e-13 1.420065 0.6389354
# > 8  55.79113 44.04454  6.253107e-13 1.383644 0.6497359
# > 9  56.31494 44.57140  3.269793e-12 1.424922 0.6448638
# > 10 55.86973 44.22628 -6.891561e-13 1.382615 0.6415214

options(scipen = 999) 
train_metrics_main
# > rmse      mae                   bias bias_pct        r2
# > 1  55.04251 43.60905  0.0000000000004764873 1.346121 0.6585752
# > 2  55.82866 44.17177 -0.0000000000012728581 1.382138 0.6442755
# > 3  56.32076 44.38433  0.0000000000027708433 1.413128 0.6459810
# > 4  56.50006 44.83254  0.0000000000043070436 1.426601 0.6405934
# > 5  56.41270 44.65912 -0.0000000000006001662 1.424082 0.6386489
# > 6  55.85497 44.32494  0.0000000000003552897 1.400207 0.6480406
# > 7  56.18424 44.40822  0.0000000000004132062 1.420065 0.6389354
# > 8  55.79113 44.04454  0.0000000000006253107 1.383644 0.6497359
# > 9  56.31494 44.57140  0.0000000000032697927 1.424922 0.6448638
# > 10 55.86973 44.22628 -0.0000000000006891561 1.382615 0.6415214
options(scipen = 0) 

# Obtain final mean estimate 
main_metrics <- colMeans(train_metrics_main)
main_metrics
# >         rmse          mae         bias     bias_pct           r2 
# > 5.601197e+01 4.432322e+01 9.655793e-13 1.400352e+00 6.451171e-01

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
# > rmse     3.518602e-03 3.078421e-03 3.134437e-03 3.047263e-03 3.282489e-03 3.230006e-03 3.455146e-03 3.452862e-03 3.143221e-03 3.333668e-03
# > mae      2.251404e-03 2.138492e-03 2.354421e-03 2.343369e-03 2.010295e-03 2.104720e-03 2.242081e-03 2.074867e-03 2.024347e-03 2.342458e-03
# > bias     1.152985e-02 1.346131e-02 1.519630e-02 4.182148e-02 3.815847e-03 7.849532e-03 1.779560e-02 7.460896e-03 3.287544e-02 7.004880e-03
# > bias_pct 4.933515e-04 6.432976e-04 9.344077e-04 1.693661e-03 2.143903e-04 4.805017e-04 9.605016e-04 5.031676e-04 1.718557e-03 3.904096e-04
# > r2       5.997901e-07 4.691260e-07 5.625009e-07 5.405349e-07 5.452180e-07 5.370303e-07 6.613906e-07 5.333143e-07 4.900035e-07 6.125025e-07

sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 3.267612e-03 2.188645e-03 1.588111e-02 8.032245e-04 5.551411e-07

# Imputation variance
imputation_var <- colSums((train_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >.        rmse          mae         bias     bias_pct           r2 
# > 1.869704e-01 1.183141e-01 3.440737e-24 6.971307e-04 3.621793e-05

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 2.089350e-01 1.323341e-01 1.588111e-02 1.570068e-03 4.039487e-05

# Final standard error
se_final <- sqrt(var_final)
se_final
# >        rmse         mae        bias    bias_pct          r2 
# > 0.457094115 0.363777555 0.126020291 0.039624088 0.006355695 

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
# >    Metric         Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE 56.0119689125987747502    0.457094115 55.1160809 56.9078569 1.79177600
# >       MAE 44.3232189473057047735    0.363777555 43.6102280 45.0362099 1.42598181
# >      Bias  0.0000000000009655793    0.126020291 -0.2469952  0.2469952 0.49399046
# >     Bias%  1.4003522482887047484    0.039624088  1.3226905  1.4780140 0.15532357
# > R-squared  0.6451171246275746451    0.006355695  0.6326602  0.6575741 0.02491387


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
# > 1  53.94042 42.68726 -0.5111205 1.2077063 0.6674757
# > 2  55.53951 44.10888 -1.4067248 1.0738673 0.6435793
# > 3  55.82117 43.76862 -1.8363250 0.8640130 0.6383683
# > 4  56.07365 44.21901 -3.7684040 0.5987363 0.6419344
# > 5  54.71224 43.50263  0.7669597 1.4268023 0.6490525
# > 6  55.50680 43.74919 -0.4162676 1.1668026 0.6443860
# > 7  55.63780 43.38533 -2.0659614 0.8829783 0.6406720
# > 8  54.13288 42.58249 -1.0399250 0.9622494 0.6571897
# > 9  55.89653 44.08441 -1.9622629 0.8969442 0.6436895
# > 10 54.71267 43.20019 -0.8597135 1.1235259 0.6481431

# Obtain final mean estimate 
main_metrics <- colMeans(valid_metrics_main)
main_metrics
# >      rmse       mae      bias  bias_pct        r2 
# > 55.197367 43.528799 -1.309975  1.020363  0.647449 

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
# > rmse     9.412927e-03 7.888179e-03 1.159175e-02 1.153227e-02 1.138208e-02 1.220279e-02 1.525496e-02 1.050595e-02 1.479307e-02 1.263682e-02
# > mae      6.626352e-03 6.670741e-03 6.552588e-03 8.137313e-03 6.742080e-03 8.153495e-03 9.474230e-03 7.026361e-03 8.237772e-03 6.823081e-03
# > bias     2.507782e-02 3.511407e-02 3.279952e-02 5.643913e-02 2.547474e-02 2.426637e-02 3.623671e-02 2.765523e-02 5.842575e-02 2.489004e-02
# > bias_pct 1.094229e-03 1.772254e-03 1.722685e-03 2.367384e-03 1.267843e-03 1.243488e-03 1.795296e-03 1.362480e-03 2.847510e-03 1.260493e-03
# > r2       1.927314e-06 1.825297e-06 2.925459e-06 1.943143e-06 2.575162e-06 2.660841e-06 3.422771e-06 1.649585e-06 2.797448e-06 2.527511e-06
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 1.172008e-02 7.444401e-03 3.463794e-02 1.673366e-03 2.425453e-06

# Imputation variance
imputation_var <- colSums((valid_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >        rmse          mae         bias     bias_pct           r2 
# > 5.817707e-01 3.289582e-01 1.483719e+00 5.277095e-02 7.736549e-05 

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 6.516679e-01 3.692985e-01 1.666729e+00 5.972142e-02 8.752749e-05  

# Final standard error
se_final <- sqrt(var_final)
se_final
# >        rmse         mae        bias    bias_pct          r2 
# > 0.807259478 0.607699315 1.291018711 0.244379655 0.009355613

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
# >    Metric Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE      55.197367    0.807259478 53.6151674 56.7795664 3.16439901
# >       MAE      43.528799    0.607699315 42.3377304 44.7198679 2.38213754
# >      Bias      -1.309975    1.291018711 -3.8403247  1.2203757 5.06070036
# >     Bias%       1.020363    0.244379655  0.5413873  1.4993379 0.95795065
# > R-squared       0.647449    0.009355613  0.6291124  0.6657857 0.03667333

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
# > 1  56.14099 44.31904 -2.605620 0.8315969 0.6576492
# > 2  55.94007 44.47611 -1.780098 0.9455726 0.6505354
# > 3  56.18877 44.53190 -1.719982 0.9535835 0.6475102
# > 4  56.65298 44.33944 -2.378864 0.9335586 0.6463059
# > 5  56.50162 44.55590 -2.427961 0.9037627 0.6556441
# > 6  54.96097 43.61331 -1.953449 0.8779461 0.6593461
# > 7  57.54720 45.07208 -1.757449 1.0476907 0.6268023
# > 8  56.66501 44.40151 -1.283550 1.0662096 0.6473065
# > 9  55.95335 44.05042 -3.530225 0.5566649 0.6519237
# > 10 55.38556 43.60683 -1.416364 0.9981829 0.6473367

# Obtain final mean estimate 
main_metrics <- colMeans(test_metrics_main)
main_metrics
# >       rmse        mae       bias   bias_pct         r2 
# > 56.1936512 44.2966540 -2.0853563  0.9114769  0.6490360

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
# >                  [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > rmse     1.352878e-02 1.297961e-02 2.291226e-02 1.478447e-02 1.382094e-02 1.560057e-02 1.816105e-02 1.880991e-02 1.096231e-02 1.585707e-02
# > mae      1.002577e-02 8.934601e-03 1.384055e-02 9.963531e-03 1.040658e-02 1.057350e-02 1.095060e-02 1.235681e-02 7.442766e-03 1.093441e-02
# > bias     2.769749e-02 2.135009e-02 3.516409e-02 6.493427e-02 1.836057e-02 1.929223e-02 3.415693e-02 2.554721e-02 5.467072e-02 2.072830e-02
# > bias_pct 1.168163e-03 9.194527e-04 1.830830e-03 2.753687e-03 8.167708e-04 9.773799e-04 1.623686e-03 1.247776e-03 2.610678e-03 9.211410e-04
# > r2       2.414832e-06 2.092956e-06 3.040190e-06 2.588112e-06 2.439378e-06 2.675551e-06 3.571269e-06 3.110566e-06 2.085464e-06 2.042840e-06
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 1.574170e-02 1.054291e-02 3.219019e-02 1.486956e-03 2.606116e-06 

# Imputation variance
imputation_var <- colSums((test_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 5.174949e-01 1.969787e-01 4.448770e-01 2.076596e-02 8.274839e-05 

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 5.849861e-01 2.272195e-01 5.215549e-01 2.432951e-02 9.362934e-05 

# Final standard error
se_final <- sqrt(var_final)
se_final
# >        rmse         mae        bias    bias_pct          r2 
# > 0.764843844 0.476675476 0.722187557 0.155979208 0.009676225 

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
# >      RMSE     56.1936512    0.764843844 54.6945848 57.6927176 2.99813277
# >       MAE     44.2966540    0.476675476 43.3623872 45.2309208 1.86853353
# >      Bias     -2.0853563    0.722187557 -3.5008179 -0.6698947 2.83092320
# >     Bias%      0.9114769    0.155979208  0.6057632  1.2171905 0.61142726
# > R-squared      0.6490360    0.009676225  0.6300710  0.6680011 0.03793011


## ---- Summary ----

print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric         Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE 56.0119689125987747502    0.457094115 55.1160809 56.9078569 1.79177600
# >       MAE 44.3232189473057047735    0.363777555 43.6102280 45.0362099 1.42598181
# >      Bias  0.0000000000009655793    0.126020291 -0.2469952  0.2469952 0.49399046
# >     Bias%  1.4003522482887047484    0.039624088  1.3226905  1.4780140 0.15532357
# > R-squared  0.6451171246275746451    0.006355695  0.6326602  0.6575741 0.02491387

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE      55.197367    0.807259478 53.6151674 56.7795664 3.16439901
# >       MAE      43.528799    0.607699315 42.3377304 44.7198679 2.38213754
# >      Bias      -1.309975    1.291018711 -3.8403247  1.2203757 5.06070036
# >     Bias%       1.020363    0.244379655  0.5413873  1.4993379 0.95795065
# > R-squared       0.647449    0.009355613  0.6291124  0.6657857 0.03667333

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >      RMSE     56.1936512    0.764843844 54.6945848 57.6927176 2.99813277
# >       MAE     44.2966540    0.476675476 43.3623872 45.2309208 1.86853353
# >      Bias     -2.0853563    0.722187557 -3.5008179 -0.6698947 2.83092320
# >     Bias%      0.9114769    0.155979208  0.6057632  1.2171905 0.61142726
# > R-squared      0.6490360    0.009676225  0.6300710  0.6680011 0.03793011



