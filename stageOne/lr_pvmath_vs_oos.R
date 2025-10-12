# ---- Script Description ----
#
# Purpose:
#   Performs three main tasks in analyze PISA 2022 Canada (student-level) data including: 
#     I.   Exploratory Data Analysis (EDA), including weighted correlation analysis.
#     II.  Explanatory Modelling using weighted linear regression with plausible values (PVs),
#          Balanced Repeated Replication (BRR), and Rubin’s Rules.
#     III. Predictive Modelling with multiple algorithms, evaluated using PVs, BRR, and Rubin’s Rules.
#
# Data
#   Source: CY08MSP_STU_QQQ_CAN.SAV (student-level, Canada). Metadata: metadata_student.csv.
#   y: PV1MATH–PV10MATH (plausible values in mathematics).
#   X: EXERPRAC, STUDYHMW, WORKPAY, WORKHOME
#     (See OECD, 2024, p. 397: Student questionnaire derived variables > Simple questionnaire indices > Out-of-school experiences, Module 10)
# 
# Sampling Weights
#   Final student weight W_FSTUWT is used in all data analysis.
#
# Data Split
#   Random student-level split: 70% TRAIN / 15% VALID / 15% TEST.
#   Optionally 80/10/10 if adding predictors and listwise deletion reduces sample size.
#
# Methodology:
#   - Follows official PISA procedures for PV handling, final student weights (W_FSTUWT),
#     and BRR replicate weights (W_FSTURWT1–W_FSTURWT80) with Fay’s k = 0.5.
#   - Combines BRR (sampling variance) and Rubin’s Rules (imputation variance) to estimate
#     total standard errors for parameters and performance metrics.
#   - Manual implementation verified with:
#       * `intsvy` R package outputs
#       * IEA IDB Analyzer (SPSS Statistics)
#     ensuring consistency across platforms.
#   - Investigated SAS macros, IEA IDB Analyzer (SPSS Statistics) code, 
#     and relevant R source code (e.g., stats::summary.lm, getAnywhere(intsvy.reg.pv), getAnywhere(pisa.reg.pv)) 
#     to verify methodological alignment and implementation details.
#
# References:
#   1. OECD (2024). *PISA 2022 Technical Report*. OECD Publishing. https://doi.org/10.1787/01820d6d-en
#   2. OECD (2009). *PISA Data Analysis Manual: SAS, Second Edition*. OECD Publishing. https://doi.org/10.1787/9789264056251-en
#   3. Zhang, Y., & Cutumisu, M. (2024). Predicting the Mathematics Literacy of Resilient Students from 
#      High‐performing Economies: A Machine Learning Approach. *Studies in Educational Evaluation, 83*, 101412.
# 
# TODO: MI degrees of freedom vs lm residual df.


# ---- I. Exploratory Data Analysis (EDA)----
## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)      # For reading SPSS .sav files
library(tidyverse)  # Includes dplyr, tidyr, purrr, ggplot2, tibble, etc.
library(tibble)     # For tidy table outputs
library(broom)      # For tidying model output
library(intsvy)     # For analyze PISA data

# Load data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE) # Preserve SPSS's user-defined missing values
dim(pisa_2022_student_canada)  # 23073 x 1278

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Constants
M <- 10                         # Number of plausible values
G <- 80                         # Number of BRR replicate weights
k <- 0.5                        # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)          # 95% z-critical value for confidence interval (CI)

# Variable names
pvmaths  <- paste0("PV", 1:M, "MATH")     # PV1MATH to PV10MATH
final_wt <- "W_FSTUWT"                    # Final student weight
rep_wts  <- paste0("W_FSTURWT", 1:G)      # BRR replicate weights: W_FSTURWT1 to W_FSTURWT80

# Define predictor variables of interest (see Zhang & Cutumisu, 2024)
oos <- c("EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME") # I have

# Subset data, checking missingness, and filter data
temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                    # Student and school IDs for sample size checks (OECD, 2009)
         all_of(final_wt), all_of(rep_wts),     # Final student weight + 80 BRR replicate weights
         all_of(pvmaths), all_of(oos))          # Target and predictor variables

sapply(temp_data[, oos], function(x) sum(is.na(x)))
# > EXERPRAC STUDYHMW  WORKPAY WORKHOME 
# >     2912     2887     2998     2944 
sum(is.na(temp_data[, pvmaths]))
# > [1] 0

temp_data <- temp_data %>% filter(if_all(all_of(oos), ~ !is.na(.))) # Listwise deletion: retain only complete cases on predictors
dim(temp_data)  # 20003 x 97


## ---- Variable Class ----

# For plausible values in mathematics
sapply(temp_data[pvmaths], class)   # All numeric

# For final student weight 
sapply(temp_data[final_wt], class)  # All numeric

# For BRR replicate weights 
sapply(temp_data[rep_wts], class)   # All numeric

# For predictors 
sapply(temp_data[oos], class)
# >       EXERPRAC              STUDYHMW              WORKPAY               WORKHOME             
# > [1,] "haven_labelled_spss" "haven_labelled_spss" "haven_labelled_spss" "haven_labelled_spss"
# > [2,] "haven_labelled"      "haven_labelled"      "haven_labelled"      "haven_labelled"     
# > [3,] "vctrs_vctr"          "vctrs_vctr"          "vctrs_vctr"          "vctrs_vctr"         
# > [4,] "double"              "double"              "double"              "double"  


## ---- Summary Statistics ----
# For plausible values in mathematics
# sapply(temp_data[pvmaths], summary)
# sapply(temp_data[pvmaths], sd)
rbind(sapply(temp_data[pvmaths], summary), `Std. Dev.` = sapply(temp_data[pvmaths], sd))
# >             PV1MATH   PV2MATH   PV3MATH   PV4MATH  PV5MATH   PV6MATH   PV7MATH  PV8MATH   PV9MATH  PV10MATH
# > Min.      186.44300 125.74700 135.37200 138.06600 171.5050 173.31600  99.96400 125.1450 171.46600 120.62600
# > 1st Qu.   436.65350 437.57375 436.74600 437.53825 436.5165 435.51575 436.88500 436.3510 435.96800 438.50925
# > Median    500.30200 500.42050 499.86250 500.32700 498.7140 498.61250 499.42600 499.1095 498.72500 500.27350
# > Mean      500.80329 501.03647 500.78064 500.97015 499.8443 499.69101 500.51639 500.4045 500.06540 501.26914
# > 3rd Qu.   563.08525 563.90950 563.30800 563.22075 561.7007 562.03275 562.34900 562.6555 562.42475 562.80150
# > Max.      851.46700 875.27500 835.73000 895.75300 860.1550 885.77800 821.98400 827.7000 869.94000 903.00900
# > Std. Dev.  90.81739  90.90119  90.96026  90.37942  90.9393  90.81622  90.11218  90.7654  90.83869  89.70215

# For final student weight
rbind(sapply(temp_data[final_wt], summary), `Std. Dev.` = sapply(temp_data[final_wt], sd))
# >            W_FSTUWT
# > Min.        1.04731
# > 1st Qu.     4.25281
# > Median      9.89608
# > Mean       16.14737
# > 3rd Qu.    26.29699
# > Max.      833.47350
# > Std. Dev.  16.53914

# For BRR replicate weights
rbind(sapply(temp_data[rep_wts], summary), `Std. Dev.` = sapply(temp_data[rep_wts], sd))

# For predictors 
rbind(sapply(temp_data[oos], summary), `Std. Dev.` = sapply(temp_data[oos], sd))
# >            EXERPRAC  STUDYHMW   WORKPAY  WORKHOME
# > Min.       0.000000  0.000000  0.000000  0.000000
# > 1st Qu.    2.000000  2.000000  0.000000  2.000000
# > Median     4.000000  4.000000  0.000000  4.000000
# > Mean       4.433635  4.380093  1.945008  4.619057
# > 3rd Qu.    7.000000  6.000000  3.000000  8.000000
# > Max.      10.000000 10.000000 10.000000 10.000000
# > Std. Dev.  3.387253  3.030304  2.833138  3.427242

## ---- Data Visualization ----

# For plausible values in mathematics
temp_data %>%
  select(all_of(pvmaths)) %>%
  pivot_longer(everything(), names_to = "PlausibleValue", values_to = "Score") %>%
  ggplot(aes(x = Score)) +
  geom_density(fill = "skyblue", color=NA, alpha = 0.6) +
  facet_wrap(~ PlausibleValue, scales = "free", ncol = 5) +
  theme_minimal() +
  labs(title = "Distribution of PV1MATH to PV10MATH", x = "Score", y = "Density")

# For final student weight
ggplot(temp_data, aes(x = .data[[final_wt]])) +
  geom_density(fill = "lightgreen", color=NA, alpha = 0.6) +
  theme_minimal() +
  labs(title = "Distribution of Final Student Weight", x = final_wt, y = "Density")

# For BRR replicate weights
temp_data %>%
  select(all_of(rep_wts)) %>%
  pivot_longer(everything(), names_to = "RepWeight", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_density(fill = "lightgreen", color=NA, alpha = 0.6) +
  facet_wrap(~ RepWeight, scales = "free", ncol = 8) +
  theme_minimal() +
  labs(title = "Distribution of 80 BRR Replicate Weights", x = "Weight", y = "Density")

# For predictors 
temp_data %>%
  select(all_of(oos)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = factor(Value))) +
  geom_bar(fill = "steelblue") +
  facet_wrap(~ Variable, scales = "free_y") +
  scale_x_discrete(drop = FALSE) +
  labs(x = "Value", y = "Count") +
  theme_minimal()


## ---- Individual Variable Exploration -----

# Remark: Student Questionnaire derived variables

### ---- Simple questionnaire indices: Out-of-school experiences (Module 10)----
#### ---- EXERPRAC: Exercise or practice a sport before or after school ----

class(pisa_2022_student_canada$EXERPRAC)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$EXERPRAC)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 

table(pisa_2022_student_canada$EXERPRAC)
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912
#prop.table(table(pisa_2022_student_canada$EXERPRAC)) 
round(prop.table(table(pisa_2022_student_canada$EXERPRAC)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 0.16 0.04 0.09 0.08 0.09 0.10 0.07 0.03 0.05 0.02 0.13 0.1

table(temp_data$EXERPRAC)
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 3781  979 2088 1815 2111 2199 1666  698 1252  391 3023 
#prop.table(table(temp_data$EXERPRAC)) 
round(prop.table(table(temp_data$EXERPRAC)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 0.19 0.05 0.11 0.09 0.11 0.11 0.08 0.03 0.06 0.02 0.14 

#c(summary(pisa_2022_student_canada$EXERPRAC), SD = sd(pisa_2022_student_canada$EXERPRAC))
c(summary(temp_data$EXERPRAC), SD = sd(temp_data$EXERPRAC))
# >     Min.   1st Qu.    Median      Mean   3rd Qu.      Max.        SD 
# > 0.000000  2.000000  4.000000  4.433635  7.000000 10.000000  3.387253

#attr(temp_data$EXERPRAC, "labels")
print(data.frame(
  Value = unname(attr(temp_data$EXERPRAC, "labels")),
  Label = names(attr(temp_data$EXERPRAC, "labels"))
)[order(attr(temp_data$EXERPRAC, "labels")), ], row.names = FALSE)
# > Value                                             Label
# >     0                             No exercise or sports
# >     1           1 time of exercising or sports per week
# >     2          2 times of exercising or sports per week
# >     3          3 times of exercising or sports per week
# >     4          4 times of exercising or sports per week
# >     5          5 times of exercising or sports per week
# >     6          6 times of exercising or sports per week
# >     7          7 times of exercising or sports per week
# >     8          8 times of exercising or sports per week
# >     9          9 times of exercising or sports per week
# >    10 10 or more times of exercising or sports per week
# >    95                                        Valid Skip
# >    97                                    Not Applicable
# >    98                                           Invalid
# >    99                                       No Response

hist(temp_data$EXERPRAC, breaks=11)
barplot(table(temp_data$EXERPRAC))
boxplot(temp_data$EXERPRAC)


#### ---- STUDYHMW: Studying for school or homework before or after school ----

class(pisa_2022_student_canada$STUDYHMW)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$STUDYHMW)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 

table(pisa_2022_student_canada$STUDYHMW)
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 2535 1295 2426 2224 2553 2511 2028  953 1125  411 2125 2887 
#prop.table(table(pisa_2022_student_canada$STUDYHMW)) 
round(prop.table(table(pisa_2022_student_canada$STUDYHMW)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 0.11 0.06 0.11 0.10 0.11 0.11 0.09 0.04 0.05 0.02 0.09 0.13

table(temp_data$STUDYHMW)
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 2510 1282 2406 2199 2535 2480 2016  942 1114  408 2111 
#prop.table(table(temp_data$STUDYHMW)) 
round(prop.table(table(temp_data$STUDYHMW)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 0.13 0.06 0.12 0.11 0.13 0.12 0.10 0.05 0.06 0.02 0.11

#c(summary(pisa_2022_student_canada$STUDYHMW), SD = sd(pisa_2022_student_canada$STUDYHMW))
c(summary(temp_data$STUDYHMW), SD = sd(temp_data$STUDYHMW))
# >     Min.   1st Qu.    Median      Mean   3rd Qu.      Max.        SD 
# > 0.000000  2.000000  4.000000  4.380093  6.000000 10.000000  3.030304 

#attr(temp_data$STUDYHMW, "labels")
print(data.frame(
  Value = unname(attr(temp_data$STUDYHMW, "labels")),
  Label = names(attr(temp_data$STUDYHMW, "labels"))
)[order(attr(temp_data$STUDYHMW, "labels")), ], row.names = FALSE)
# > Value                              Label
# >     0                        No studying
# >     1        1 time of studying per week
# >     2       2 times of studying per week
# >     3       3 times of studying per week
# >     4       4 times of studying per week
# >     5       5 times of studying per week
# >     6       6 times of studying per week
# >     7       7 times of studying per week
# >     8       8 times of studying per week
# >     9       9 times of studying per week
# >    10 10 or more times of study per week
# >    95                         Valid Skip
# >    97                     Not Applicable
# >    98                            Invalid
# >    99                        No Response

hist(temp_data$STUDYHMW, breaks=11)
barplot(table(temp_data$STUDYHMW))
boxplot(temp_data$STUDYHMW)


#### ---- WORKPAY: Working for pay before or after school ----

class(pisa_2022_student_canada$WORKPAY)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$WORKPAY)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 

table(pisa_2022_student_canada$WORKPAY)
# >     0     1     2     3     4     5     6     7     8     9    10    99 
# > 11157  1021  1777  1319  1287   814   915   252   542   140   851  2998 
#prop.table(table(pisa_2022_student_canada$WORKPAY)) 
round(prop.table(table(pisa_2022_student_canada$WORKPAY)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 0.48 0.04 0.08 0.06 0.06 0.04 0.04 0.01 0.02 0.01 0.04 0.13 

table(temp_data$WORKPAY)
# >     0     1     2     3     4     5     6     7     8     9    10 
# > 11133  1017  1771  1307  1276   810   913   250   539   138   849
#prop.table(table(temp_data$WORKPAY)) 
round(prop.table(table(temp_data$WORKPAY)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 0.19 0.05 0.11 0.09 0.11 0.11 0.08 0.03 0.06 0.02 0.14 

#c(summary(pisa_2022_student_canada$WORKPAY), SD = sd(pisa_2022_student_canada$WORKPAY))
c(summary(temp_data$WORKPAY), SD = sd(temp_data$WORKPAY))
# >      Min.   1st Qu.    Median      Mean   3rd Qu.      Max.        SD 
# >  0.000000  0.000000  0.000000  1.945008  3.000000 10.000000  2.833138  

#attr(temp_data$WORKPAY, "labels")
print(data.frame(
  Value = unname(attr(temp_data$WORKPAY, "labels")),
  Label = names(attr(temp_data$WORKPAY, "labels"))
)[order(attr(temp_data$WORKPAY, "labels")), ], row.names = FALSE)
# > Value                                             Label
# >     0                             No exercise or sports
# >     1           1 time of exercising or sports per week
# >     2          2 times of exercising or sports per week
# >     3          3 times of exercising or sports per week
# >     4          4 times of exercising or sports per week
# >     5          5 times of exercising or sports per week
# >     6          6 times of exercising or sports per week
# >     7          7 times of exercising or sports per week
# >     8          8 times of exercising or sports per week
# >     9          9 times of exercising or sports per week
# >    10 10 or more times of exercising or sports per week
# >    95                                        Valid Skip
# >    97                                    Not Applicable
# >    98                                           Invalid
# >    99                                       No Response

hist(temp_data$WORKPAY, breaks=11)
barplot(table(temp_data$WORKPAY))
boxplot(temp_data$WORKPAY)


#### ---- WORKHOME: Working in household/take care of family members before or after school ----

class(pisa_2022_student_canada$WORKHOME)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$WORKHOME)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 

table(pisa_2022_student_canada$WORKHOME)
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 3297 1298 2119 1748 1880 2256 1589  885 1165  556 3336 2944
#prop.table(table(pisa_2022_student_canada$WORKHOME)) 
round(prop.table(table(pisa_2022_student_canada$WORKHOME)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 0.14 0.06 0.09 0.08 0.08 0.10 0.07 0.04 0.05 0.02 0.14 0.13 

table(temp_data$WORKHOME)
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 3287 1290 2103 1733 1864 2238 1579  879 1160  553 3317
#prop.table(table(temp_data$WORKHOME)) 
round(prop.table(table(temp_data$WORKHOME)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 0.16 0.06 0.11 0.09 0.09 0.11 0.08 0.04 0.06 0.03 0.17 

#c(summary(pisa_2022_student_canada$WORKHOME), SD = sd(pisa_2022_student_canada$WORKHOME))
c(summary(temp_data$WORKHOME), SD = sd(temp_data$WORKHOME))
# >     Min.   1st Qu.    Median      Mean   3rd Qu.      Max.        SD 
# > 0.000000  2.000000  4.000000  4.619057  8.000000 10.000000  3.427242

#attr(temp_data$WORKHOME, "labels")
print(data.frame(
  Value = unname(attr(temp_data$WORKHOME, "labels")),
  Label = names(attr(temp_data$WORKHOME, "labels"))
)[order(attr(temp_data$WORKHOME, "labels")), ], row.names = FALSE)
# > Value                                                                          Label
# >     0                                 No work in household or care of family members
# >     1           1 time of working in household or caring for family members per week
# >     2          2 times of working in household or caring for family members per week
# >     3          3 times of working in household or caring for family members per week
# >     4          4 times of working in household or caring for family members per week
# >     5          5 times of working in household or caring for family members per week
# >     6          6 times of working in household or caring for family members per week
# >     7          7 times of working in household or caring for family members per week
# >     8          8 times of working in household or caring for family members per week
# >     9          9 times of working in household or caring for family members per week
# >    10 10 or more times of working in household or caring for family members per week
# >    95                                                                     Valid Skip
# >    97                                                                 Not Applicable
# >    98                                                                        Invalid
# >    99                                                                    No Response

hist(temp_data$WORKHOME, breaks=11)
barplot(table(temp_data$WORKHOME))
boxplot(temp_data$WORKHOME)



## ---- I.1 Correlation ----
### ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)      # To read .sav files
library(dplyr)      # For data manipulation
library(tibble)     # For tidy outputs

# Load data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE) # Preserve SPSS's user-defined missing values
dim(pisa_2022_student_canada)  # 23073 x 1278

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Constants
M <- 10                         # Number of plausible values
G <- 80                         # Number of BRR replicate weights
k <- 0.5                        # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)          # 95% z-critical value for confidence interval (CI)

# Variable names
pvmaths  <- paste0("PV", 1:M, "MATH")                       # PV1MATH to PV10MATH
final_wt <- "W_FSTUWT"                                      # Final student weight
rep_wts  <- paste0("W_FSTURWT", 1:G)                        # BRR replicate weights: W_FSTURWT1 to W_FSTURWT80
oos <- c( "EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME")    # Student Questionnaire derived variables > Simple questionnaire indices > Out-of-school experiences (Module 10)(OECD, 2009)

# Subset data, checking missingness, and filter data
temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                    # Student and school IDs for sample size checks (OECD, 2009)
         all_of(final_wt), all_of(rep_wts),     # Final student weight + 80 BRR replicate weights
         all_of(pvmaths), all_of(oos))          # Target and predictor variables

sapply(temp_data[, oos], function(x) sum(is.na(x)))
# > EXERPRAC STUDYHMW  WORKPAY WORKHOME 
# >     2912     2887     2998     2944 
sum(is.na(temp_data[, pvmaths]))
# > [1] 0

temp_data <- temp_data %>% filter(if_all(all_of(oos), ~ !is.na(.))) # Listwise deletion: retain only complete cases on predictors
dim(temp_data)  # 20003 x 97


### ---- Helper: Weighted correlation ----
get_wtd_corr <- function(x, y, w) {
  mx <- weighted.mean(x, w)                                 # Weighted mean of x
  my <- weighted.mean(y, w)                                 # Weighted mean of y
  cov_xy <- sum(w * (x - mx) * (y - my)) / sum(w)           # Weighted covariance
  var_x  <- sum(w * (x - mx)^2) / sum(w)                    # Weighted variance x
  var_y  <- sum(w * (y - my)^2) / sum(w)                    # Weighted variance y
  if (var_x <= 0 || var_y <= 0) return(NA_real_)            # Prevent invalid denominator
  return(cov_xy / sqrt(var_x * var_y))                      # Weighted correlation
}

### ---- Main function: compute mean correlation + SE using Rubin + BRR ----
compute_rubin_brr_corr <- function(var1, var2) {
  # Point estimates (ρ̂ₘ)
  main_corrs <- sapply(1:M, function(m) {
    get_wtd_corr(temp_data[[var1[m]]], temp_data[[var2[m]]], temp_data[[final_wt]])
  })
  main_corr <- mean(main_corrs) # Final mean estimate ρ̂
  
  # Replicate estimates (ρ̂ₘg)
  rep_corr_matrix <- matrix(NA_real_, M, G)
  for (m in 1:M) {
    for (g in 1:G) {
      rep_corr_matrix[m, g] <- get_wtd_corr(
        temp_data[[var1[m]]],
        temp_data[[var2[m]]],
        temp_data[[rep_wts[g]]]
      )
    }
  }
  
  # Sampling variance per PV: σ²(ρ̂ₘ)
  sampling_var_per_pv <- sapply(1:M, function(m) {
    sum((rep_corr_matrix[m, ] - main_corrs[m])^2) / (G * (1 - k)^2)
  })
  # <=> Equivalent codes
  # sampling_var_per_pv <- sapply(1:M, function(m) {
  #   mean((rep_corr_matrix[m, ] - main_corrs[m])^2) / (1 - k)^2
  # })
  sampling_var <- mean(sampling_var_per_pv)
  
  # Imputation variance
  imputation_var <- sum((main_corrs - main_corr)^2) / (M - 1)
  
  # Final error variance and standard error
  var_total <- sampling_var + (1 + 1/M) * imputation_var
  se_total <- sqrt(var_total)
  
  return(c(main_corr = main_corr, se = se_total))
}

### ---- Final output: Correlation Matrix ----
variables <- c(
  list(PVmMATH = pvmaths),                                       
  setNames(lapply(oos, function(v) rep(v, M)), oos)  
)
variable_names <- names(variables)

# Create n_vars x n_vars = 20 x 20 matrices for point estimates and standard errors
n_vars <- length(variable_names)
n_vars
# > [1] 5

point_estimate_matrix <- matrix(NA_real_, n_vars, n_vars, dimnames = list(variable_names, variable_names))
se_matrix             <- matrix(NA_real_, n_vars, n_vars, dimnames = list(variable_names, variable_names))

# Loop over variable pairs (only n_varsxn_vars)
for (i in 1:n_vars) {
  for (j in i:n_vars) {
    result <- compute_rubin_brr_corr(variables[[i]], variables[[j]])
    point_estimate_matrix[i, j] <- result["main_corr"]
    point_estimate_matrix[j, i] <- result["main_corr"]
    se_matrix[i, j] <- result["se"]
    se_matrix[j, i] <- result["se"]
  }
}

# Display results
cat("Point Estimates (ρ̂):\n")
print(round(point_estimate_matrix, 6))
# >            PVmMATH  EXERPRAC STUDYHMW   WORKPAY  WORKHOME
# > PVmMATH   1.000000 -0.124582 0.013298 -0.212767 -0.102605
# > EXERPRAC -0.124582  1.000000 0.261322  0.252191  0.245291
# > STUDYHMW  0.013298  0.261322 1.000000  0.092494  0.308906
# > WORKPAY  -0.212767  0.252191 0.092494  1.000000  0.201339
# > WORKHOME -0.102605  0.245291 0.308906  0.201339  1.000000

cat("\nStandard Errors (SE):\n")
print(round(se_matrix, 6))
# >           PVmMATH EXERPRAC STUDYHMW  WORKPAY WORKHOME
# > PVmMATH  0.000000 0.010228 0.012927 0.010133 0.010542
# > EXERPRAC 0.010228 0.000000 0.011003 0.010138 0.009641
# > STUDYHMW 0.012927 0.011003 0.000000 0.010823 0.010038
# > WORKPAY  0.010133 0.010138 0.010823 0.000000 0.010053
# > WORKHOME 0.010542 0.009641 0.010038 0.010053 0.000000


# ---- II. Explanatory Modelling----
## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)   # For reading SPSS .sav files
library(dplyr)   # For data manipulation
library(tibble)  # For tidy table outputs
library(broom)   # For tidying model output
library(intsvy)  # For analyze PISA data

# Load data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE) # Preserve SPSS's user-defined missing values
dim(pisa_2022_student_canada)  # 23073 x 1278

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Constants
M <- 10                         # Number of plausible values
G <- 80                         # Number of BRR replicate weights
k <- 0.5                        # Fay's adjustment factor (used in BRR)

# Variable names
pvmaths  <- paste0("PV", 1:M, "MATH")     # PV1MATH to PV10MATH
final_wt <- "W_FSTUWT"                    # Final student weight
rep_wts  <- paste0("W_FSTURWT", 1:G)      # BRR replicate weights: W_FSTURWT1 to W_FSTURWT80

# Define predictor variables of interest (see Zhang & Cutumisu, 2024)
oos <- c( "EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME") # I have

# Subset data, checking missingness, and filter data
temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                    # Student and school IDs for sample size checks (OECD, 2009)
         all_of(final_wt), all_of(rep_wts),     # Final student weight + 80 BRR replicate weights
         all_of(pvmaths), all_of(oos))          # Target and predictor variables

sapply(temp_data[, oos], function(x) sum(is.na(x)))
# > EXERPRAC STUDYHMW  WORKPAY WORKHOME 
# >     2912     2887     2998     2944 
sum(is.na(temp_data[, pvmaths]))
# > [1] 0

temp_data <- temp_data %>% filter(if_all(all_of(oos), ~ !is.na(.))) # Listwise deletion: retain only complete cases on predictors
dim(temp_data)  # 20003 x 97


## ---- Exploration with intsvy ----
library(intsvy)
# Null model
pisa.reg.pv(pvlabel=pvmaths, x="EXERPRAC", data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   517.52       2.09  247.76
# > EXERPRAC       -3.45       0.29  -12.06
# > R-squared       0.02       0.00    6.06
pisa.reg.pv(pvlabel=pvmaths, x="STUDYHMW", data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   500.52       2.24  223.73
# > STUDYHMW        0.41       0.40    1.03
# > R-squared       0.00       0.00    0.49
pisa.reg.pv(pvlabel=pvmaths, x="WORKPAY", data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   515.26       1.86  277.75
# > WORKPAY        -7.35       0.36  -20.15
# > R-squared       0.05       0.00   10.43
pisa.reg.pv(pvlabel=pvmaths, x="WORKHOME", data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   515.63       2.07  248.77
# > WORKHOME       -2.80       0.29   -9.56
# > R-squared       0.01       0.00    4.89
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "STUDYHMW"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   512.13       2.35  217.84
# > EXERPRAC       -3.81       0.32  -12.07
# > STUDYHMW        1.52       0.43    3.56
# > R-squared       0.02       0.00    5.96
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "WORKPAY"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   523.30       2.17  240.93
# > EXERPRAC       -2.10       0.30   -7.08
# > WORKPAY        -6.69       0.38  -17.84
# > R-squared       0.05       0.00   11.08
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "WORKHOME"), data=temp_data)
# > Estimate Std. Error t value
# > (Intercept)   525.13       2.31  227.25
# > EXERPRAC       -2.93       0.30   -9.81
# > WORKHOME       -2.09       0.30   -6.90
# > R-squared       0.02       0.00    7.20
pisa.reg.pv(pvlabel=pvmaths, x=c("STUDYHMW", "WORKPAY"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   510.58       2.39  213.71
# > STUDYHMW        1.54       0.43    3.62
# > WORKHOME       -3.22       0.31  -10.23
# > R-squared       0.01       0.00    5.22
pisa.reg.pv(pvlabel=pvmaths, x=c("STUDYHMW", "WORKHOME"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   510.58       2.39  213.71
# > STUDYHMW        1.54       0.43    3.62
# > WORKHOME       -3.22       0.31  -10.23
# > R-squared       0.01       0.00    5.22
pisa.reg.pv(pvlabel=pvmaths, x=c("WORKPAY", "WORKHOME"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   522.54       2.17  241.01
# > WORKPAY        -6.92       0.37  -18.51
# > WORKHOME       -1.70       0.30   -5.61
# > R-squared       0.05       0.00   11.08
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "STUDYHMW", "WORKPAY"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   517.35       2.42  213.60
# > EXERPRAC       -2.49       0.32   -7.86
# > STUDYHMW        1.70       0.43    3.97
# > WORKPAY        -6.75       0.38  -17.99
# > R-squared       0.05       0.01   10.60
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "STUDYHMW", "WORKHOME"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   518.89       2.47  210.45
# > EXERPRAC       -3.34       0.32  -10.43
# > STUDYHMW        2.30       0.44    5.26
# > WORKHOME       -2.62       0.31   -8.35
# > R-squared       0.03       0.00    7.42
pisa.reg.pv(pvlabel=pvmaths, x=c("STUDYHMW", "WORKPAY", "WORKHOME"), data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   516.87       2.48  208.38
# > STUDYHMW        1.75       0.42    4.18
# > WORKPAY        -6.98       0.38  -18.61
# > WORKHOME       -2.17       0.31   -7.04
# > R-squared       0.05       0.00   10.75
pisa.reg.pv(pvlabel=pvmaths, x=oos, data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   521.88       2.54  205.56
# > EXERPRAC       -2.22       0.32   -6.93
# > STUDYHMW        2.24       0.43    5.22
# > WORKPAY        -6.41       0.38  -17.10
# > WORKHOME       -1.85       0.31   -6.01
# > R-squared       0.06       0.01   11.14





## ---- Initial Exploration ----

# Fit weighted linear model: Math PV1 ~ Gender 
mod <- lm(as.formula(paste("PV1MATH ~", paste(oos, collapse = " + "))), data = temp_data, weights = temp_data[[final_wt]]) 

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
#lm(as.formula(paste("PV1MATH", "~ ESCS")), data = temp_data, weights = temp_data[[final_wt]]) %>%
#  summary()

# TODO:
# Use gender-based coloring in diagnostic plots







## ---- Fit main models using final student weight (W_FSTUWT) ----
main_models <- lapply(pvmaths, function(pv) {
  formula <- as.formula(paste(pv, "~", paste(oos, collapse = " + ")))
  mod <- lm(formula, data = temp_data, weights = temp_data[[final_wt]])
  summ <- summary(mod)                         # View source codes: stats::summary.lm
  list(
    formula = formula,                         # Regression formula used
    mod = mod,                                 # Fitted lm model object
    summ = summ,                               # Summary of lm (contains all diagnostics)
    coefs = coef(mod),                         # Named vector of regression coefficients
    r2 = summ$r.squared,                       # R-squared: proportion of variance explained
    adj_r2 = summ$adj.r.squared,               # Adjusted R-squared: accounts for model size
    sigma = summ$sigma,                        # Residual std. error 
    fstat_val   = summ$fstatistic[["value"]],  # F-statistic value for overall model
    fstat_numdf = summ$fstatistic[["numdf"]],  # Numerator df = # predictors
    fstat_dendf = summ$fstatistic[["dendf"]],  # Denominator df = residual df
    dof_model     = summ$df[1],                # Degrees of freedom for model (non-aliased)
    dof_residual = summ$df[2],                 # Residual degrees of freedom = n - p, where p is the number of estimated regression coefficients (including intercept, excluding aliased)
    dof_total     = summ$df[3]                 # Total number of coefficients (including aliased)
  )
}) # Option: split into main_models and main_results if needed
main_models[[1]] # View structure of the first fitted model
main_models[[2]] # View structure of the second fitted model

# --- Extract and average main fit estimates across plausible values (PVs) ---

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs
# >      (Intercept)  EXERPRAC STUDYHMW   WORKPAY  WORKHOME
# > [1,]    521.4261 -1.999764 2.182276 -6.483795 -1.826272
# > [2,]    522.0011 -2.172228 2.172222 -6.317461 -1.785746
# > [3,]    521.5095 -2.137629 2.200857 -6.421693 -1.798650
# > [4,]    524.0137 -2.504327 2.196171 -6.362508 -1.963067
# > [5,]    522.3519 -2.157700 2.146833 -6.442102 -1.943208
# > [6,]    521.1095 -2.286632 2.388111 -6.257987 -1.971225
# > [7,]    523.5319 -2.294608 1.974159 -6.452831 -1.806158
# > [8,]    520.6385 -2.204305 2.593554 -6.587783 -1.926711
# > [9,]    521.3785 -2.114590 2.107425 -6.339751 -1.805860
# > [10,]    520.8018 -2.304178 2.438010 -6.483202 -1.714962

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  521.876247   -2.217596    2.239962   -6.414911   -1.854186 


# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.05591431 0.05559200 0.05545877 0.06017327 0.05769659 0.05702688 0.05858203 0.06017129 0.05465749 0.05934516
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.05572548 0.05540310 0.05526984 0.05998529 0.05750811 0.05683826 0.05839372 0.05998330 0.05446840 0.05915701
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 360.4642 358.8323 362.7985 359.3594 359.9233 360.0556 357.7890 359.7721 361.6185 355.8345
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 296.0998 294.2925 293.5458 320.0976 306.1160 302.3478 311.1061 320.0864 289.0594 315.4144

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.05746178
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.05727325
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 359.6447
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 304.8166


## ---- Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste(pv, "~", paste(oos, collapse = " + ")))
    mod <- lm(formula, data = temp_data, weights = temp_data[[w]])
    summ <- summary(mod)
    list(
      formula = formula,                         # Regression formula
      mod = mod,                                 # Fitted lm model
      summ = summ,                               # Summary output
      coefs = coef(mod),                         # Model coefficients
      r2 = summ$r.squared,                       # R-squared
      adj_r2 = summ$adj.r.squared,               # Adjusted R-squared: math vs codes
      sigma = summ$sigma,                        # Residual std. error
      fstat_val   = summ$fstatistic[["value"]],  # F-statistic value
      fstat_numdf = summ$fstatistic[["numdf"]],  # Numerator df
      fstat_dendf = summ$fstatistic[["dendf"]],  # Denominator df
      dof_model    = summ$df[1],                 # Model degrees of freedom (non-aliased)
      dof_residual = summ$df[2],                 # Residual degrees of freedom
      dof_total    = summ$df[3]                  # Total coefficients (including aliased)
    )
  })
}) # Structure: replicate_models[[pv]][[replicate]] — list of 10 PVs × 80 replicates
replicate_models[[1]][[1]]


# --- Extract replicate fit estimates (Rubin's step 1)---
# Coefficients: rep_coefs[[m]] = G × p matrix for plausible value m 
rep_coefs <- lapply(replicate_models, function(m) {
  do.call(rbind, lapply(m, function(g) coef(g$mod)))  # G × p matrix for each PV
})             # list of ten 80x2 matrix
rep_coefs[[1]] # 80x2 matrix

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
# >                   [,1]       [,2]       [,3]       [,4]       [,5]       [,6]       [,7]       [,8]       [,9]      [,10]
# > (Intercept) 5.51907014 4.83295221 5.15119110 4.28403003 5.61326191 4.88063307 5.15583677 4.81065195 5.53594197 4.72467825
# > EXERPRAC    0.08239955 0.08552394 0.09149935 0.08479836 0.07686677 0.08194335 0.08642476 0.07637077 0.06197829 0.08621557
# > STUDYHMW    0.15182140 0.16112344 0.14760772 0.13893819 0.13521250 0.16489558 0.14738266 0.12460938 0.15817721 0.15233179
# > WORKPAY     0.12333806 0.11721001 0.14146572 0.12685483 0.12959622 0.13577660 0.12162420 0.12905063 0.15448509 0.12531835
# > WORKHOME    0.07093998 0.10275429 0.11476068 0.07894917 0.08115566 0.08796294 0.09638292 0.05900062 0.08190519 0.09111916

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef 
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  5.05082474  0.08140207  0.14820999  0.13047197  0.08649306 

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# > 1.267830256 0.018965139 0.032716823 0.009337024 0.007923409 

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  6.44543802  0.10226372  0.18419849  0.14074270  0.09520881 

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >   2.5387867   0.3197870   0.4291835   0.3751569   0.3085592  


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
# > [1] 0.00515688
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.005157911
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 4.119886
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 29.07708


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
print(as.data.frame(result_table), row.names = FALSE)
# >               Term   Estimate Std. Error    t value Pr(>|t|) t_Signif    z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >         (Intercept) 521.876247  2.5387867 205.561280  < 2e-16      *** 205.561280  < 2e-16      ***   <NA>     <NA>
# >            EXERPRAC  -2.217596  0.3197870  -6.934604 4.20e-12      ***  -6.934604 4.07e-12      ***   <NA>     <NA>
# >            STUDYHMW   2.239962  0.4291835   5.219124 1.82e-07      ***   5.219124 1.80e-07      ***   <NA>     <NA>
# >             WORKPAY  -6.414911  0.3751569 -17.099275  < 2e-16      *** -17.099275  < 2e-16      ***   <NA>     <NA>
# >            WORKHOME  -1.854186  0.3085592  -6.009173 1.90e-09      ***  -6.009173 1.86e-09      ***   <NA>     <NA>
# >           R-squared   0.060000  0.0100000         NA     <NA>     <NA>         NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared   0.060000  0.0100000         NA     <NA>     <NA>         NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error 359.640000  4.1200000         NA     <NA>     <NA>         NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 304.820000 29.0800000         NA     <NA>     <NA>         NA     <NA>     <NA> <2e-16      ***

## ---- Compare results with intsvy package  ----
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=oos, data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   521.88       2.54  205.56
# > EXERPRAC       -2.22       0.32   -6.93
# > STUDYHMW        2.24       0.43    5.22
# > WORKPAY        -6.41       0.38  -17.10
# > WORKHOME       -1.85       0.31   -6.01
# > R-squared       0.06       0.01   11.14

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=oos, data=temp_data, std=TRUE)
# >             Estimate Std. Error t value
# > (Intercept)     0.11       0.02    7.32
# > EXERPRAC       -0.08       0.01   -6.88
# > STUDYHMW        0.07       0.01    5.22
# > WORKPAY        -0.20       0.01  -16.99
# > WORKHOME       -0.07       0.01   -6.03
# > R-squared       0.06       0.01   11.14


# Remark: results are validated also using IEA IDB Analyzer, with consistency found. 



# ---- III. Predictive Modelling----
## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)   # For reading SPSS .sav files
library(tidyverse)  # or library(tidyr) + library(dplyr) + library(ggplot2)
# library(dplyr)   # For data manipulation
# library(tidyr)
# library(ggplot2) # For visualization
library(tibble)  # For tidy table outputs
library(broom)   # For tidying model output
library(intsvy)  # For analyze PISA data

# Load data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE) # Preserve SPSS's user-defined missing values
dim(pisa_2022_student_canada)  # 23073 x 1278

# Load metadata
metadata_student <- read.csv("data/pisa2022/metadata_student.csv") |> tibble::as_tibble()

# Constants
M <- 10                         # Number of plausible values
G <- 80                         # Number of BRR replicate weights
k <- 0.5                        # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)          # 95% z-critical value for confidence interval (CI)

# Variable names
pvmaths  <- paste0("PV", 1:M, "MATH")     # PV1MATH to PV10MATH
final_wt <- "W_FSTUWT"                    # Final student weight
rep_wts  <- paste0("W_FSTURWT", 1:G)      # BRR replicate weights: W_FSTURWT1 to W_FSTURWT80

# Define predictor variables of interest (see Zhang & Cutumisu, 2024)
oos <- c( "EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME") # I have

# Subset data, checking missingness, and filter data
temp_data <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                    # Student and school IDs for sample size checks (OECD, 2009)
         all_of(final_wt), all_of(rep_wts),     # Final student weight + 80 BRR replicate weights
         all_of(pvmaths), all_of(oos))          # Target and predictor variables

# The following codes give the same results
# temp_data <- pisa_2022_student_canada %>%
#   select(CNTSCHID, CNTSTUID,                    # Student and school IDs for sample size checks (OECD, 2009)
#          all_of(final_wt), all_of(rep_wts),     # Final student weight + 80 BRR replicate weights
#          all_of(pvmaths), all_of(oos)) %>%      # Target and predictor variables
#   filter(if_all(all_of(oos), ~ !is.na(.)))

sapply(temp_data[, oos], function(x) sum(is.na(x)))
# > EXERPRAC STUDYHMW  WORKPAY WORKHOME 
# >     2912     2887     2998     2944 
sum(is.na(temp_data[, pvmaths]))
# > [1] 0

temp_data <- temp_data %>% filter(if_all(all_of(oos), ~ !is.na(.))) # Listwise deletion: retain only complete cases on predictors
dim(temp_data)  # 20003 x 97


## ---- Random Train/Validation/Test (70/15/15) split ----
set.seed(123)  # Ensure reproducibility

n <- nrow(temp_data) # 16114
indices <- sample(n)  # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.70 * n)        # 11279
n_valid <- floor(0.15 * n)        # 2417
n_test  <- n - n_train - n_valid  # 2418

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
main_models <- lapply(pvmaths, function(pv) {
  formula <- as.formula(paste(pv, "~", paste(oos, collapse = " + ")))
  mod <- lm(formula, data = train_data, weights = train_data[[final_wt]])
  summ <- summary(mod)                         # View source codes: stats::summary.lm
  list(
    formula = formula,                         # Regression formula used
    mod = mod,                                 # Fitted lm model object
    summ = summ,                               # Summary of lm (contains all diagnostics)
    coefs = coef(mod),                         # Named vector of regression coefficients
    r2 = summ$r.squared,                       # R-squared: proportion of variance explained
    adj_r2 = summ$adj.r.squared,               # Adjusted R-squared: accounts for model size
    sigma = summ$sigma,                        # Residual std. error 
    fstat_val   = summ$fstatistic[["value"]],  # F-statistic value for overall model
    fstat_numdf = summ$fstatistic[["numdf"]],  # Numerator df = # predictors
    fstat_dendf = summ$fstatistic[["dendf"]],  # Denominator df = residual df
    dof_model     = summ$df[1],                # Degrees of freedom for model (non-aliased)
    dof_residual = summ$df[2],                 # Residual degrees of freedom = n - p, where p is the number of estimated regression coefficients (including intercept, excluding aliased)
    dof_total     = summ$df[3]                 # Total number of coefficients (including aliased)
  )
}) # Option: split into main_models and main_results if needed
main_models[[1]] # View structure of the first fitted model
main_models[[2]] # View structure of the second fitted model

# --- Extract and average main fit estimates across plausible values (PVs) ---

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs
# > (Intercept)  EXERPRAC STUDYHMW   WORKPAY  WORKHOME
# > [1,]    521.3903 -1.876956 1.956970 -6.485790 -1.844278
# > [2,]    522.1356 -2.151055 2.122041 -6.184907 -1.912558
# > [3,]    521.3551 -2.167629 2.228508 -6.290765 -1.974383
# > [4,]    524.1052 -2.341908 2.146976 -6.362691 -2.154661
# > [5,]    522.6755 -2.118644 2.069785 -6.382466 -2.043519
# > [6,]    521.2741 -2.309747 2.353125 -6.257122 -2.118796
# > [7,]    522.8818 -2.304628 1.964793 -6.315424 -1.825370
# > [8,]    520.5614 -2.159785 2.463427 -6.511891 -2.017261
# > [9,]    521.9739 -2.196356 2.069727 -6.550658 -1.854971
# > [10,]    521.3909 -2.237202 2.342007 -6.624295 -1.793063

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  521.974387   -2.186391    2.171736   -6.396601   -1.953886 


# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.05354006 0.05361088 0.05401633 0.05845384 0.05552031 0.05668234 0.05554228 0.05784387 0.05654770 0.05919030
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.05326958 0.05334043 0.05374599 0.05818477 0.05525040 0.05641277 0.05527238 0.05757463 0.05627808 0.05892144
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 362.3764 360.1565 365.2094 362.1807 363.4058 362.8282 360.5892 362.1929 363.9409 358.5121
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 197.9482 198.2249 199.8096 217.2433 205.7000 210.2639 205.7862 214.8372 209.7345 220.1525

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.05609479
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.05582505
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 362.1392
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 207.97


### ---- Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste(pv, "~", paste(oos, collapse = " + ")))
    mod <- lm(formula, data = train_data, weights = train_data[[w]])
    summ <- summary(mod)
    list(
      formula = formula,                         # Regression formula
      mod = mod,                                 # Fitted lm model
      summ = summ,                               # Summary output
      coefs = coef(mod),                         # Model coefficients
      r2 = summ$r.squared,                       # R-squared
      adj_r2 = summ$adj.r.squared,               # Adjusted R-squared: math vs codes
      sigma = summ$sigma,                        # Residual std. error
      fstat_val   = summ$fstatistic[["value"]],  # F-statistic value
      fstat_numdf = summ$fstatistic[["numdf"]],  # Numerator df
      fstat_dendf = summ$fstatistic[["dendf"]],  # Denominator df
      dof_model    = summ$df[1],                 # Model degrees of freedom (non-aliased)
      dof_residual = summ$df[2],                 # Residual degrees of freedom
      dof_total    = summ$df[3]                  # Total coefficients (including aliased)
    )
  })
}) # Structure: replicate_models[[pv]][[replicate]] — list of 10 PVs × 80 replicates
replicate_models[[1]][[1]]


# --- Extract replicate fit estimates (Rubin's step 1)---
# Coefficients: rep_coefs[[m]] = G × p matrix for plausible value m 
rep_coefs <- lapply(replicate_models, function(m) {
  do.call(rbind, lapply(m, function(g) coef(g$mod)))  # G × p matrix for each PV
})             # list of ten 80x2 matrix
rep_coefs[[1]] # 80x2 matrix

# Fit statistics: each rep_stat[[stat]] is a G × M matrix (replicates × PVs)---
rep_r2     <- sapply(replicate_models, function(m) sapply(m, function(g) g$r2))             # 80x2 matrix
class(rep_r2);dim(rep_r2)
# > [1] "matrix" "array" 
# > [1] 80 10
rep_adj_r2 <- sapply(replicate_models, function(m) sapply(m, function(g) g$adj_r2))         # 80x2 matrix
rep_sigma  <- sapply(replicate_models, function(m) sapply(m, function(g) g$sigma))          # 80x2 matrix
rep_fstat_val  <- sapply(replicate_models, function(m) sapply(m, function(g) g$fstat_val))  # 80x2 matrix


### ---- Rubin + BRR for Standard Errors (SEs)----
# --- Rubin + BRR for Coefficients ---
sampling_var_matrix_coef <- sapply(1:M, function(m) {
  coefs_m <- rep_coefs[[m]]                   # G×p=80x2 matrix for PV m: each row is θ̂ₘ_g (replicate estimates for PV m)
  sweep(coefs_m, 2, main_coefs[m, ])^2 |>     # Compute  (θ̂ₘ_g - θ̂ₘ)^2 element-wise
    colSums() / (G * (1 - k)^2)               # Compute BRR sampling variance with Fay's adjustment for each coefficient
})
sampling_var_matrix_coef 
# >                   [,1]       [,2]      [,3]      [,4]       [,5]       [,6]      [,7]       [,8]       [,9]     [,10]
# > (Intercept) 6.63333137 6.10258682 6.6702286 5.4058319 7.01192011 5.89540079 6.4049706 5.72768122 7.00525211 5.5663005
# > EXERPRAC    0.10325002 0.09887434 0.1152338 0.1014276 0.09076033 0.09580262 0.1076571 0.09399740 0.07439934 0.1038875
# > STUDYHMW    0.20875408 0.19658126 0.1920092 0.1990038 0.18255474 0.21294371 0.2026079 0.16653699 0.20098388 0.1966792
# > WORKPAY     0.16179674 0.15203532 0.2076979 0.1590343 0.16627466 0.18336475 0.1743670 0.16373214 0.19243334 0.1583323
# > WORKHOME    0.09603659 0.11637063 0.1214988 0.1120377 0.11276790 0.11376540 0.1210173 0.09035189 0.11081559 0.1167901

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef 
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >  6.24235040  0.09852901  0.19586547  0.17190685  0.11114518

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# > 1.04814471  0.01763460  0.02922593  0.02002599  0.01620703 

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >   7.3953096   0.1179271   0.2280140   0.1939354   0.1289729 

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef
# > (Intercept)    EXERPRAC    STUDYHMW     WORKPAY    WORKHOME 
# >   2.7194318   0.3434051   0.4775081   0.4403810   0.3591280 


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
# > [1] 0.005940901
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.005942599
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 4.970976
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 23.38713


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
print(as.data.frame(result_table), row.names = FALSE)
# >                Term   Estimate Std. Error    t value Pr(>|t|) t_Signif    z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >         (Intercept) 521.974387  2.7194318 191.942441  < 2e-16      *** 191.942441  < 2e-16      ***   <NA>     <NA>
# >            EXERPRAC  -2.186391  0.3434051  -6.366798 1.99e-10      ***  -6.366798 1.93e-10      ***   <NA>     <NA>
# >            STUDYHMW   2.171736  0.4775081   4.548061 5.46e-06      ***   4.548061 5.41e-06      ***   <NA>     <NA>
# >             WORKPAY  -6.396601  0.4403810 -14.525151  < 2e-16      *** -14.525151  < 2e-16      ***   <NA>     <NA>
# >            WORKHOME  -1.953886  0.3591280  -5.440640 5.40e-08      ***  -5.440640 5.31e-08      ***   <NA>     <NA>
# >           R-squared   0.060000  0.0100000         NA     <NA>     <NA>         NA     <NA>     <NA>   <NA>     <NA>
# >  Adjusted R-squared   0.060000  0.0100000         NA     <NA>     <NA>         NA     <NA>     <NA>   <NA>     <NA>
# > Residual Std. Error 362.140000  4.9700000         NA     <NA>     <NA>         NA     <NA>     <NA>   <NA>     <NA>
# >         F-statistic 207.970000 23.3900000         NA     <NA>     <NA>         NA     <NA>     <NA> <2e-16      ***

### ---- Compare results with intsvy package  ----
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=oos, data=train_data)
# >             Estimate Std. Error t value
# > (Intercept)   521.97       2.72  191.94
# > EXERPRAC       -2.19       0.34   -6.37
# > STUDYHMW        2.17       0.48    4.55
# > WORKPAY        -6.40       0.44  -14.53
# > WORKHOME       -1.95       0.36   -5.44
# > R-squared       0.06       0.01    9.44

# Remark: results are validated also using IEA IDB Analyzer, with consistency found. 


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
  y_pred <- predict(model$mod, train_data)
  y_true <- train_data[[as.character(model$formula[[2]])]]
  w      <- train_data[[final_wt]]
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
class(train_metrics_main); dim(train_metrics_main)
# > [1] "data.frame"
# > [1] 10  5
train_metrics_main  # train_metrics_main[,"r2"]=main_r2s
# >        rmse      mae         bias bias_pct         r2
# > 1  91.06779 73.19362 2.612822e-12 3.603169 0.05354006
# > 2  90.50991 72.77655 4.280074e-12 3.545121 0.05361088
# > 3  91.77974 73.58140 5.040959e-12 3.691862 0.05401633
# > 4  91.01860 72.84432 3.236104e-12 3.588507 0.05845384
# > 5  91.32648 73.10176 4.935674e-12 3.622317 0.05552031
# > 6  91.18133 73.14754 1.433361e-12 3.652116 0.05668234
# > 7  90.61865 72.88807 4.041727e-12 3.586280 0.05554228
# > 8  91.02167 73.07028 4.911728e-12 3.612763 0.05784387
# > 9  91.46096 73.44447 3.060399e-12 3.666965 0.05654770
# > 10 90.09666 72.64228 3.378924e-12 3.521097 0.05919030

# Obtain final mean estimate 
main_metrics <- colMeans(train_metrics_main)
main_metrics
# >         rmse          mae         bias     bias_pct           r2 
# > 9.100818e+01 7.306903e+01 3.693177e-12 3.609020e+00 5.609479e-02 

### ---- Replicate predictions for each plausible value and replicate weight using replicate_models ----
# Output: list of 10 matrices; each matrix is 80 × 5 (replicates × metrics)
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]
    y_pred <- predict(model$mod, train_data)
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
  sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |> colSums() / (G * (1 - k)^2)
}) 
dim(sampling_var_matrix)  # 5 metrics × 10 PVs
# > [1]  5 10
sampling_var_matrix
# >                  [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > rmse     6.641022e-01 6.688521e-01 8.337075e-01 6.489477e-01 8.056774e-01 6.236817e-01 8.107879e-01 8.102558e-01 7.699332e-01 7.172317e-01
# > mae      4.364312e-01 4.975593e-01 5.923611e-01 4.662594e-01 5.252866e-01 4.684748e-01 5.453171e-01 5.440861e-01 5.301746e-01 5.371407e-01
# > bias     5.736875e-23 1.081107e-22 1.212516e-22 7.257853e-23 1.335726e-22 3.239891e-23 9.025637e-23 1.243636e-22 6.357701e-23 6.795463e-23
# > bias_pct 4.400634e-03 3.834419e-03 7.584650e-03 4.520748e-03 5.275857e-03 5.543469e-03 8.649232e-03 5.797727e-03 6.558971e-03 5.187474e-03
# > r2       3.052955e-05 2.608080e-05 3.501491e-05 2.994011e-05 2.989844e-05 3.282912e-05 3.159553e-05 3.188115e-05 2.905075e-05 3.162321e-05

# <=> Equivalent codes
# sampling_var_matrix <- sapply(1:M, function(m) {
#   sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |> colMeans() / (1 - k)^2
# }) 
# sampling_var_matrix

sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 7.353177e-01 5.143091e-01 8.714327e-23 5.735318e-03 3.084436e-05

# Imputation variance
imputation_var <- colSums((train_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 2.407018e-01 8.634883e-02 1.365904e-24 2.783797e-03 4.045405e-06 

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 1.000090e+00 6.092928e-01 8.864576e-23 8.797494e-03 3.529430e-05 

# Final standard error
se_final <- sqrt(var_final)
se_final
# >         rmse          mae         bias     bias_pct           r2 
# > 1.000045e+00 7.805721e-01 9.415188e-12 9.379496e-02 5.940901e-03

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
# >    Metric        Point_estimate       Standard_error             CI_lower             CI_upper           CI_length
# >      RMSE 91.008177596236890849 1.000044871776648359 89.04812566463068890 92.96822952784309280 3.92010386321240389
# >       MAE 73.069030477250166200 0.780572104113458454 71.53913726585113864 74.59892368864919376 3.05978642279805513
# >      Bias  0.000000000003693177 0.000000000009415188 -0.00000000001476025  0.00000000002214661 0.00000000003690686
# >     Bias%  3.609019818879561914 0.093794959253481830  3.42518507681133588  3.79285456094778795 0.36766948413645206
# > R-squared  0.056094791680599332 0.005940900762988929  0.04445084014941451  0.06773874321178416 0.02328790306236966

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=oos, data=train_data)
# >             Estimate Std. Error t value
# > (Intercept)   521.97       2.72  191.94
# > EXERPRAC       -2.19       0.34   -6.37
# > STUDYHMW        2.17       0.48    4.55
# > WORKPAY        -6.40       0.44  -14.53
# > WORKHOME       -1.95       0.36   -5.44
# > R-squared       0.06       0.01    9.44


## ---- Predict and evaluate on the validation data (Weighted, Rubin + BRR)----
### ---- Predict on validation data (for each PV) using main_models ----
# Output: 10 × 5 data.frame with metrics for each plausible value
valid_metrics_main <- sapply(main_models, function(model) {
  y_pred <- predict(model$mod, valid_data)
  y_true <- valid_data[[as.character(model$formula[[2]])]]
  w      <- valid_data[[final_wt]]
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
class(valid_metrics_main); dim(valid_metrics_main)
# > [1] "data.frame"
# > [1] 10  5
valid_metrics_main
# >        rmse      mae       bias bias_pct         r2
# > 1  88.01501 70.63746 -3.0380957 2.736462 0.05249251
# > 2  88.20282 70.06570 -2.1046157 3.044303 0.05606270
# > 3  90.04831 72.44121 -2.5544316 3.002963 0.05255233
# > 4  88.31888 70.92134 -0.4628296 3.365787 0.05674432
# > 5  87.53046 70.18554 -0.5991454 3.209198 0.05832862
# > 6  87.47979 70.80282 -3.8401234 2.521645 0.05217894
# > 7  87.34035 70.11434 -3.2158802 2.638454 0.05881918
# > 8  86.90373 69.62067 -2.7543817 2.674542 0.06133422
# > 9  88.87943 71.30336 -1.6794237 3.016056 0.03808785
# > 10 86.49896 69.51619 -0.5135922 3.085834 0.05976739

# Obtain final mean estimate 
main_metrics <- colMeans(valid_metrics_main)
main_metrics
# >        rmse         mae        bias    bias_pct          r2 
# > 87.92177325 70.56086290 -2.07625193  2.92952438  0.05463681 

### ---- Replicate predictions for each plausible value and replicate weight using replicate_models ----
# Output: list of 10 matrices; each matrix is 80 × 5 (replicates × metrics)
valid_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]
    y_pred <- predict(model$mod, valid_data)
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
  sweep(valid_metrics_replicates[[m]], 2, unlist(valid_metrics_main[m, ]))^2 |> colSums() / (G * (1 - k)^2)
}) 
dim(sampling_var_matrix)  # 5 metrics × 10 PVs
# > [1]  5 10
sampling_var_matrix
# >                  [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > rmse     2.841997e+00 2.752126e+00 2.569142e+00 2.603124e+00 2.3266626410 2.0694817585 3.342068e+00 2.4591122772 2.5630168418 3.0656761797
# > mae      2.200722e+00 2.048617e+00 1.839495e+00 2.291722e+00 1.7934187869 1.6535149481 2.674819e+00 2.7048913154 2.0382340363 2.5892097065
# > bias     7.525027e+00 7.641858e+00 1.052940e+01 6.640374e+00 7.4500505296 7.6555726032 7.999331e+00 6.5547090373 7.4786979308 6.9869178468
# > bias_pct 3.624616e-01 3.830982e-01 4.879007e-01 3.276583e-01 0.3750302299 0.3528910225 3.537392e-01 0.3087253142 0.3416143152 0.3223539295
# > r2       9.052192e-05 9.388041e-05 1.003159e-04 9.654833e-05 0.0001075708 0.0001405706 8.048759e-05 0.0001004143 0.0000989376 0.0001198049
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 2.6592405891 2.1834643910 7.6461940533 0.3615472902 0.0001029052  

# Imputation variance
imputation_var <- colSums((valid_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >        rmse          mae         bias     bias_pct           r2 
# > 1.047993e+00 7.602330e-01 1.489868e+00 7.471192e-02 4.418074e-05  

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 3.8120328118 3.0197207021 9.2850489004 0.4437304054 0.0001515041 

# Final standard error
se_final <- sqrt(var_final)
se_final
# >      rmse       mae      bias  bias_pct        r2
# > 1.9524428 1.7377344 3.0471378 0.6661309 0.0123087

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
# >    Metric Point_estimate Standard_error   CI_lower    CI_upper   CI_length
# >      RMSE    87.92177325      1.9524428 84.0950557 91.74849078  7.65343506
# >       MAE    70.56086290      1.7377344 67.1549661 73.96675966  6.81179352
# >      Bias    -2.07625193      3.0471378 -8.0485323  3.89602846 11.94456077
# >     Bias%     2.92952438      0.6661309  1.6239318  4.23511700  2.61118523
# > R-squared     0.05463681      0.0123087  0.0305122  0.07876141  0.04824921    

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=oos, data=valid_data)
# >             Estimate Std. Error t value
# > (Intercept)   521.50       5.54   94.22
# > EXERPRAC       -2.85       0.71   -4.00
# > STUDYHMW        1.58       0.71    2.24
# > WORKPAY        -6.35       0.89   -7.12
# > WORKHOME       -0.27       0.79   -0.34
# > R-squared       0.06       0.01    5.12  # Estimate of R-squared differs since it fits and predict on valid_data, not fitting on the train_data and predict on valid_data


## ---- Predict and evaluate on the test data (Weighted, Rubin + BRR)----
### ---- Predict on test data (for each PV) using main_models ----
# Output: 10 × 5 data.frame with metrics for each plausible value
test_metrics_main <- sapply(main_models, function(model) {
  y_pred <- predict(model$mod, test_data)
  y_true <- test_data[[as.character(model$formula[[2]])]]
  w      <- test_data[[final_wt]]
  compute_metrics(y_true, y_pred, w)
}) |> t() |> as.data.frame()
class(test_metrics_main); dim(test_metrics_main)
# > [1] "data.frame"
# > [1] 10  5
test_metrics_main
# >        rmse      mae       bias bias_pct         r2
# > 1  91.30557 73.68468 -1.0909602 3.412643 0.06888007
# > 2  90.99335 74.09800 -0.3802106 3.519893 0.06375697
# > 3  89.91383 73.00931 -2.5362390 3.005934 0.06413399
# > 4  89.39225 72.57910 -1.7168142 3.123766 0.07088068
# > 5  89.62984 71.97611 -0.9149117 3.307864 0.06722458
# > 6  90.60855 73.12369 -1.4804492 3.171664 0.06179628
# > 7  89.57547 73.13125 -0.7320933 3.382357 0.07188760
# > 8  91.42165 73.94718 -2.4069274 3.098416 0.06878443
# > 9  90.56822 73.24347 -1.9036112 3.107544 0.06027604
# > 10 89.55679 71.85415 -0.6346463 3.262421 0.05931871

# Obtain final mean estimate 
main_metrics <- colMeans(test_metrics_main)
main_metrics
# >        rmse         mae        bias    bias_pct          r2 
# > 90.29655194 73.06469276 -1.37968629  3.23925027  0.06569394

### ---- Replicate predictions for each plausible value and replicate weight using replicate_models ----
# Output: list of 10 matrices; each matrix is 80 × 5 (replicates × metrics)
test_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]
    y_pred <- predict(model$mod, test_data)
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
  sweep(test_metrics_replicates[[m]], 2, unlist(test_metrics_main[m, ]))^2 |> colSums() / (G * (1 - k)^2)
}) 
dim(sampling_var_matrix)  # 5 metrics × 10 PVs
# > [1]  5 10
sampling_var_matrix
# >                  [,1]        [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > rmse     2.2124270905 1.048949051 1.6881209285 1.2239270429 1.6527752290 1.5741648268 1.6478158194 1.809791e+00 1.9675197527 1.8434974058
# > mae      1.5974062463 0.863109168 1.4730990645 1.0465405573 1.3633427688 1.3073180387 1.2117369118 1.430783e+00 1.3117279060 1.3916844780
# > bias     6.5925773936 6.168278618 5.0918606849 5.6881104458 5.4243363269 5.1765180275 5.2037208764 5.488911e+00 4.7406430116 5.3360530618
# > bias_pct 0.3155891625 0.288905878 0.2561594705 0.2695485938 0.2633465121 0.2331375034 0.2428769108 2.757592e-01 0.2249346681 0.2476863037
# > r2       0.0001102387 0.000118645 0.0001054372 0.0001327083 0.0001153139 0.0001102618 0.0001244351 9.697672e-05 0.0001138889 0.0001279991
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 1.6668988079 1.2996748249 5.4911009591 0.2617944240 0.0001155905 

# Imputation variance
imputation_var <- colSums((test_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 6.018426e-01 5.733401e-01 5.626595e-01 2.711906e-02 1.985371e-05  

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse         mae        bias    bias_pct          r2 
# > 2.3289256806 1.9303489779 6.1100263682 0.2916253949 0.0001374295  

# Final standard error
se_final <- sqrt(var_final)
se_final
# >       rmse        mae       bias   bias_pct         r2 
# > 1.52608181 1.38936999 2.47184675 0.54002351 0.01172303

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
# >    Metric Point_estimate Standard_error    CI_lower    CI_upper  CI_length
# >      RMSE    90.29655194     1.52608181 87.30548657 93.28761732 5.98213076
# >       MAE    73.06469276     1.38936999 70.34157761 75.78780791 5.44623030
# >      Bias    -1.37968629     2.47184675 -6.22441690  3.46504432 9.68946122
# >     Bias%     3.23925027     0.54002351  2.18082364  4.29767691 2.11685327
# > R-squared     0.06569394     0.01172303  0.04271721  0.08867066 0.04595345   

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=oos, data=test_data)
# >             Estimate Std. Error t value
# > (Intercept)   522.16       5.30   98.54
# > EXERPRAC       -1.77       0.73   -2.44
# > STUDYHMW        3.30       0.88    3.77
# > WORKPAY        -6.55       0.89   -7.36
# > WORKHOME       -3.09       0.67   -4.65
# > R-squared       0.07       0.01    5.62       # Estimate of R-squared differs since it fits and predict on test_data, not fitting on the train_data and predict on test_data


## ---- **Predict and evaluate on the training/validation/test data (Weighted, Rubin + BRR)** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process to avoid redundancy.

# Helper: Compute weighted predictive performance metrics
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

# Unified function to evaluate performance for any dataset split
evaluate_split <- function(split_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit) {
  # Main model predictions
  main_metrics_df <- sapply(main_models, function(model) {
    y_pred <- predict(model$mod, newdata = split_data)
    y_true <- split_data[[as.character(model$formula[[2]])]]
    w      <- split_data[[final_wt]]
    compute_metrics(y_true, y_pred, w)
  }) |> t() |> as.data.frame()
  
  # Final mean estimates
  main_point <- colMeans(main_metrics_df)
  
  # Replicate predictions
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]
      y_pred <- predict(model$mod, newdata = split_data)
      y_true <- split_data[[as.character(model$formula[[2]])]]
      w      <- split_data[[rep_wts[g]]]
      compute_metrics(y_true, y_pred, w)
    }) |> t()
  })
  
  # Sampling variance
  sampling_var_matrix <- sapply(1:M, function(m) {
    sweep(replicate_metrics[[m]], 2, unlist(main_metrics_df[m, ]))^2 |> colSums() / (G * (1 - k)^2)
  })
  sampling_var <- rowMeans(sampling_var_matrix)
  
  # Imputation variance
  imputation_var <- colSums((main_metrics_df - matrix(main_point, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
  
  # Final combined variance
  var_final <- sampling_var + (1 + 1/M) * imputation_var
  se_final <- sqrt(var_final)
  ci_lower <- main_point - z_crit * se_final
  ci_upper <- main_point + z_crit * se_final
  ci_length <- ci_upper - ci_lower
  
  # Format output
  tibble::tibble(
    Metric         = c("RMSE", "MAE", "Bias", "Bias%", "R-squared"),
    Point_estimate = format(main_point, scientific = FALSE),
    Standard_error = format(se_final, scientific = FALSE),
    CI_lower       = format(ci_lower, scientific = FALSE),
    CI_upper       = format(ci_upper, scientific = FALSE),
    CI_length      = format(ci_length, scientific = FALSE)
  )
}

train_eval <- evaluate_split(train_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval), row.names = FALSE)

## ---- Summary ----

print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric        Point_estimate       Standard_error             CI_lower             CI_upper           CI_length
# >      RMSE 91.008177596236890849 1.000044871776648359 89.04812566463068890 92.96822952784309280 3.92010386321240389
# >       MAE 73.069030477250166200 0.780572104113458454 71.53913726585113864 74.59892368864919376 3.05978642279805513
# >      Bias  0.000000000003693177 0.000000000009415188 -0.00000000001476025  0.00000000002214661 0.00000000003690686
# >     Bias%  3.609019818879561914 0.093794959253481830  3.42518507681133588  3.79285456094778795 0.36766948413645206
# > R-squared  0.056094791680599332 0.005940900762988929  0.04445084014941451  0.06773874321178416 0.02328790306236966

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error   CI_lower    CI_upper   CI_length
# >      RMSE    87.92177325      1.9524428 84.0950557 91.74849078  7.65343506
# >       MAE    70.56086290      1.7377344 67.1549661 73.96675966  6.81179352
# >      Bias    -2.07625193      3.0471378 -8.0485323  3.89602846 11.94456077
# >     Bias%     2.92952438      0.6661309  1.6239318  4.23511700  2.61118523
# > R-squared     0.05463681      0.0123087  0.0305122  0.07876141  0.04824921    

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error    CI_lower    CI_upper  CI_length
# >      RMSE    90.29655194     1.52608181 87.30548657 93.28761732 5.98213076
# >       MAE    73.06469276     1.38936999 70.34157761 75.78780791 5.44623030
# >      Bias    -1.37968629     2.47184675 -6.22441690  3.46504432 9.68946122
# >     Bias%     3.23925027     0.54002351  2.18082364  4.29767691 2.11685327
# > R-squared     0.06569394     0.01172303  0.04271721  0.08867066 0.04595345  



