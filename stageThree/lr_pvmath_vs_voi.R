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
#   X: voi_num (53) + voi_cat (5)
#     
# Sampling Weights
#   Final student weight W_FSTUWT is used in all data analysis.
#
# Data Split
#   Random student-level split: 80% TRAIN / 10% VALID / 10% TEST.
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
library(broom)      # For tidying model output
library(intsvy)     # For analyze PISA data
library(tictoc)     # For timing code execution

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("haven", "tidyverse", "broom","intsvy", "tictoc"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         haven         tidyverse             broom            intsvy            tictoc 
# > "haven 2.5.4" "tidyverse 2.0.0"     "broom 1.0.8"      "intsvy 2.9"    "tictoc 1.2.1"

# Load data
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE) # Preserve SPSS's user-defined missing values
dim(pisa_2022_canada_merged)   
# > [1] 23073  1699
stopifnot(                                                                    
  sum(is.na(pisa_2022_canada_merged[, paste0("PV", 1:10, "MATH")])) == 0,     # or: all(colSums(is.na(pisa_2022_canada_merged[, pvmaths, drop = FALSE])) == 0)
  sum(is.na(pisa_2022_canada_merged$W_FSTUWT)) == 0,                          # or: all(!is.na(pisa_2022_canada_merged[[final_wt]]))
  colSums(is.na(pisa_2022_canada_merged[paste0("W_FSTURWT", 1:80)])) == 0
)

# Load metadata + missing summary: full variables
metadata_missing_student <- read_csv("data/pisa2022/metadata_missing_student.csv", show_col_types = FALSE)
metadata_missing_school <- read_csv("data/pisa2022/metadata_missing_school.csv", show_col_types = FALSE)

# Load metadata + missing summary: student/school questionnaire derived variables
stuq_dvs_metadata_missing_student <- readr::read_csv("data/pisa2022/stuq_dvs_metadata_missing_student.csv", show_col_types = FALSE)
schq_dvs_metadata_missing_school <- read_csv("data/pisa2022/schq_dvs_metadata_missing_school.csv",  show_col_types = FALSE)

# Constants
M <- 10                         # Number of plausible values
G <- 80                         # Number of BRR replicate weights
k <- 0.5                        # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)          # 95% z-critical value for confidence interval (CI)

# y: Response/Target variables
pvmaths  <- paste0("PV", 1:M, "MATH")   # PV1MATH to PV10MATH

# X: Explanatory/Predictor variables 
# VOI = variables of interest: numeric (continuous + ordinal) + categorical (nominal)
voi_num <- c(
  # --- Student Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### Subject-specific beliefs, attitudes, feelings and behaviours (Module 7)
  "MATHMOT",     # Relative motivation to do well in mathematics compared to other core subjects
  "MATHEASE",    # Perception of mathematics as easier than other core subjects
  "MATHPREF",    # Preference of mathematics over other core subjects
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

voi_cat <- c(
  # --- Student Questionnaire Variables ---
  "REGION",      # REGION
  # --- Student Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### Basic demographics (Module 1)
  "ST004D01T",   # Student (Standardized) Gender
  ### Migration and language exposure (Module 4)
  "IMMIG",       # Index on immigrant background (OECD definition
  "LANGN",       # Language spoken at home
  # --- School Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### School type and infrastructure (Module 11)
  "SCHLTYPE")     # School type
stopifnot(!anyDuplicated(voi_cat))
stopifnot(all(sapply(pisa_2022_canada_merged[, voi_cat], is.numeric)))
length(voi_cat)

voi_all <- c(voi_num, voi_cat)
stopifnot(!anyDuplicated(voi_all))
voi_all

length(voi_num);length(voi_cat);length(voi_all)
# > [1] 53
# > [1] 5
# > [1] 58

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)   # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                 # Final student weight

length(rep_wts); length(final_wt)
# > [1] 80
# > [1] 1

### ---- Prepare data ----
temp_data <- pisa_2022_canada_merged %>%
  select(                      
    CNTSCHID, CNTSTUID,                                       # IDs
    all_of(final_wt), all_of(rep_wts),                        # Weights
    all_of(pvmaths), all_of(voi_all)                          # PVs + predictors
  ) %>%
  mutate(
    LANGN = na_if(LANGN, 999),                                # Missing 999 -> NA
    across(                                                   # label → factor for categorical VOIs
      all_of(voi_cat),
      ~ if (inherits(.x, "haven_labelled"))
        haven::as_factor(.x, levels = "labels") else as.factor(.x)
    )
  ) %>%                                                       # -> can compare performance without listwise deletion
  filter(IMMIG != "No Response") %>%                          # Treat "No Response" in `IMMIG` as missing values and drop
  filter(if_all(all_of(voi_all), ~ !is.na(.))) %>%            # Drop missing values; ST004D01T: User-defined NA `Not Applicable` is kept as a category
  droplevels()                                                # Drop levels not present   
dim(temp_data)
# > [1] 6755  151

# Quick invariants for weights & PVs after filtering
stopifnot(
  sum(is.na(temp_data[, pvmaths])) == 0,                        # or: all(colSums(is.na(temp_data[, pvmaths, drop = FALSE])) == 0)
  sum(is.na(temp_data$W_FSTUWT)) == 0,                          # or: all(!is.na(temp_data[[final_wt]]))
  colSums(is.na(temp_data[paste0("W_FSTURWT", 1:80)])) == 0
)

#### ---- Quick explore with intsvy ----

# Calculates mean achievement score and its standard error (weighed)
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), data = pisa_2022_canada_merged)
# >    Freq   Mean s.e.    SD  s.e
# > 1 23073 496.95 1.56 94.01 0.85
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), data = temp_data)
# >   Freq   Mean s.e.    SD  s.e
# > 1 6755 521.93 2.48 89.27 1.15

# Calculates percentiles for plausible values (weighed)
pisa.per.pv(pvlabel=paste0("PV",1:10,"MATH"), per=c(0, 25, 50, 75, 100), data=pisa_2022_canada_merged)  # all ten PVs in math
# >   Percentiles  Score Std. err.
# > 1           0 128.46     31.74  # <- min
# > 2          25 430.49      1.72  # <- 1st Qu.
# > 3          50 496.32      1.80  # <- mean
# > 4          75 561.63      2.16  # <- 3rd Qu.
# > 5         100 863.19     29.87  # <- max
pisa.per.pv(pvlabel=pvmaths[1], per=c(0, 25, 50, 75, 100), data=pisa_2022_canada_merged)                # PV1MATH
# >   Percentiles  Score Std. err.
# > 1           0 129.01       NaN
# > 2          25 430.23       NaN
# > 3          50 496.74       NaN
# > 4          75 561.90       NaN
# > 5         100 851.47       NaN

pisa.per.pv(pvlabel=paste0("PV",1:10,"MATH"), per=c(0, 25, 50, 75, 100), data=temp_data)
# >   Percentiles  Score Std. err.
# > 1           0 211.38     16.26
# > 2          25 460.01      2.87
# > 3          50 520.87      3.24
# > 4          75 583.09      3.35
# > 5         100 848.88     34.76
pisa.per.pv(pvlabel=pvmaths[1], per=c(0, 25, 50, 75, 100), data=temp_data) 
# >   Percentiles  Score Std. err.
# > 1           0 205.34       NaN
# > 2          25 459.50       NaN
# > 3          50 520.75       NaN
# > 4          75 585.05       NaN
# > 5         100 842.36       NaN

# --- By ST004D01T ---
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "ST004D01T", data = pisa_2022_canada_merged)
# >   ST004D01T  Freq   Mean  s.e.    SD   s.e
# > 1         1 11377 490.71  1.68 88.08  0.97
# > 2         2 11654 503.04  1.89 99.03  1.22
# > 3         7    42 454.07 20.95 67.85 11.85
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "ST004D01T", data = temp_data)
# >        ST004D01T Freq   Mean  s.e.    SD   s.e
# > 1         Female 3565 508.30  2.88 84.85  1.55
# > 2           Male 3173 536.64  2.88 91.53  1.53
# > 3 Not Applicable   17 462.03 23.41 68.87 13.55

df <- pisa_2022_canada_merged %>%
  select(                      
    CNTSCHID, CNTSTUID,                                       # IDs
    all_of(final_wt), all_of(rep_wts),                        # Weights
    all_of(pvmaths), all_of(voi_cat)                          # PVs + predictors
  ) %>%
  mutate(
    LANGN = na_if(LANGN, 999),                                # Missing 999 -> NA
    across(                                                   # label → factor for categorical VOIs
      all_of(voi_cat),
      ~ if (inherits(.x, "haven_labelled"))
        haven::as_factor(.x, levels = "labels") else as.factor(.x)
    )
  ) %>%                                                       # -> can compare performance without listwise deletion
  filter(IMMIG != "No Response") %>%                          # Treat "No Response" in `IMMIG` as missing values and drop
  filter(if_all(all_of(voi_cat), ~ !is.na(.))) %>%            # Drop missing values; ST004D01T: User-defined NA `Not Applicable` is kept as a category
  droplevels() 

pisa.reg.pv(pvlabel=pvmaths, x="ST004D01T", data=df)
# >                        Estimate Std. Error t value
# > (Intercept)               493.44       1.78  277.22
# > ST004D01TMale              15.10       1.83    8.23
# > ST004D01TNot Applicable   -35.06      21.56   -1.63
# > R-squared                   0.01       0.00    4.11
pisa.reg.pv(pvlabel=pvmaths, x="ST004D01T", data=temp_data)
# >                         Estimate Std. Error t value
# > (Intercept)               508.30       2.88  176.57  # <- reference level: Female
# > ST004D01TMale              28.35       2.97    9.53
# > ST004D01TNot Applicable   -46.26      23.29   -1.99
# > R-squared                   0.03       0.01    4.80

# --- By REGION ---
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "REGION", data = pisa_2022_canada_merged)
# >     REGION Freq   Mean s.e.    SD  s.e
# >  1   12401 1053 458.54 5.54 86.19 2.44
# >  2   12402  357 477.70 6.64 88.47 3.93
# >  3   12403 1590 470.32 3.60 91.17 2.43
# >  4   12404 1653 467.67 3.09 90.02 2.17
# >  5   12405 4137 513.62 3.89 93.56 1.90
# >  6   12406 5918 495.22 2.97 92.56 1.62
# >  7   12407 2629 470.47 2.68 85.59 1.72
# >  8   12408 2276 467.65 2.63 86.40 1.94
# >  9   12409 1330 503.54 5.70 98.29 2.62
# > 10  12410 2130 496.30 4.42 93.04 1.91
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "REGION", data = temp_data)
# >                                    Estimate Std. Error t value
# > (Intercept)                          484.10       6.08   79.63  # <- reference level: Newfoundland and Labrador
# > REGIONCanada: Prince Edward Island    31.19      13.53    2.30
# > REGIONCanada: Nova Scotia              9.67       9.47    1.02
# > REGIONCanada: New Brunswick           11.26       7.76    1.45
# > REGIONCanada: Quebec                  55.93       7.99    7.00
# > REGIONCanada: Ontario                 39.05       7.33    5.33
# > REGIONCanada: Manitoba                 7.82       6.82    1.15
# > REGIONCanada: Saskatchewan             8.25       7.62    1.08
# > REGIONCanada: Alberta                 39.01       9.72    4.01
# > REGIONCanada: British Columbia        36.74       8.81    4.17
# > R-squared                              0.02       0.00    4.93

pisa.reg.pv(pvlabel=pvmaths, x="REGION", data=df)
# >                                    Estimate Std. Error t value
# > (Intercept)                          460.87       5.63   81.83
# > REGIONCanada: Prince Edward Island    24.73       9.52    2.60
# > REGIONCanada: Nova Scotia             13.07       6.80    1.92
# > REGIONCanada: New Brunswick           10.35       6.27    1.65
# > REGIONCanada: Quebec                  56.54       6.82    8.30
# > REGIONCanada: Ontario                 39.31       6.50    6.04
# > REGIONCanada: Manitoba                12.70       5.46    2.33
# > REGIONCanada: Saskatchewan             9.94       6.48    1.53
# > REGIONCanada: Alberta                 44.55       7.75    5.75
# > REGIONCanada: British Columbia        40.22       7.34    5.48
# > R-squared                              0.02       0.00    5.91
pisa.reg.pv(pvlabel=pvmaths, x="REGION", data=temp_data)
# >                                    Estimate Std. Error t value
# > (Intercept)                          484.10       6.08   79.63
# > REGIONCanada: Prince Edward Island    31.19      13.53    2.30
# > REGIONCanada: Nova Scotia              9.67       9.47    1.02
# > REGIONCanada: New Brunswick           11.26       7.76    1.45
# > REGIONCanada: Quebec                  55.93       7.99    7.00
# > REGIONCanada: Ontario                 39.05       7.33    5.33
# > REGIONCanada: Manitoba                 7.82       6.82    1.15
# > REGIONCanada: Saskatchewan             8.25       7.62    1.08
# > REGIONCanada: Alberta                 39.01       9.72    4.01
# > REGIONCanada: British Columbia        36.74       8.81    4.17
# > R-squared                              0.02       0.00    4.93

# --- By IMMIG ---
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "IMMIG", data = pisa_2022_canada_merged)
# >   IMMIG  Freq   Mean s.e.     SD  s.e
# > 1     1 15053 496.83 1.79  91.25 0.91
# > 2     2  2565 517.04 3.40  95.57 2.44
# > 3     3  2845 498.73 3.67 100.10 2.04
# > 4     9  2610 464.63 4.85  88.97 2.79
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "IMMIG", data = temp_data)
# >                       IMMIG Freq   Mean s.e.    SD  s.e
# > 1            Native student 5013 518.16 2.59 87.01 1.31
# > 2 Second-Generation student  866 537.73 4.85 91.68 3.24
# > 3  First-Generation student  876 519.17 5.56 94.05 3.09

pisa.reg.pv(pvlabel=pvmaths, x="IMMIG", data=df)
# >                                Estimate Std. Error t value
# > (Intercept)                      496.93       1.79  277.01
# > IMMIGSecond-Generation student    20.17       3.60    5.61
# > IMMIGFirst-Generation student      2.24       4.01    0.56
# > R-squared                          0.01       0.00    2.84
pisa.reg.pv(pvlabel=pvmaths, x="IMMIG", data=temp_data)
# >                                Estimate Std. Error t value
# > (Intercept)                      518.16       2.59  200.37
# > IMMIGSecond-Generation student    19.57       4.91    3.98
# > IMMIGFirst-Generation student      1.01       5.85    0.17
# > R-squared                          0.01       0.00    2.06

# --- By LANGN ---
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "LANGN", data = pisa_2022_canada_merged)
# >   LANGN  Freq   Mean s.e.     SD  s.e
# > 1   313 14818 491.56 1.85  91.94 1.04
# > 2   493  3553 515.63 3.67  93.88 1.81
# > 3   807  3094 506.61 3.42 100.70 2.18
# > 4   999  1608 474.15 6.76  83.65 3.70
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "LANGN", data = temp_data)
# >                    LANGN Freq   Mean s.e.    SD  s.e
# > 1                English 4755 515.43 2.80 87.67 1.44
# > 2                 French 1089 540.77 4.34 85.00 2.39
# > 3 Another language (CAN)  911 528.38 5.46 96.04 3.36

pisa.reg.pv(pvlabel=pvmaths, x="LANGN", data=df)
# >                             Estimate Std. Error t value
# > (Intercept)                   493.78       1.90  259.67
# > LANGNFrench                    23.94       4.28    5.59
# > LANGNAnother language (CAN)    16.13       3.47    4.65
# > R-squared                       0.01       0.00    3.25
pisa.reg.pv(pvlabel=pvmaths, x="LANGN", data=temp_data)
# >                             Estimate Std. Error t value
# > (Intercept)                   515.43       2.80  184.17
# > LANGNFrench                    25.34       4.95    5.12
# > LANGNAnother language (CAN)    12.95       5.45    2.38
# > R-squared                       0.01       0.00    2.60

# --- By SCHLTYPE ---
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "SCHLTYPE", data = pisa_2022_canada_merged)
# >   SCHLTYPE  Freq   Mean  s.e.    SD  s.e
# > 1        1   980 543.95 12.25 87.33 3.99
# > 2        2   566 548.84  9.06 82.99 3.77
# > 3        3 21527 493.06  1.50 93.54 0.88
pisa.mean.pv(pvlabel = paste0("PV",1:10,"MATH"), by = "SCHLTYPE", data = temp_data)
# >                       SCHLTYPE Freq   Mean  s.e.    SD  s.e
# > 1          Private independent  373 566.60 13.47 81.74 4.59
# > 2 Private Government-dependent  232 562.88  9.13 82.79 5.35
# > 3                       Public 6150 517.74  2.23 88.81 1.16

pisa.reg.pv(pvlabel=pvmaths, x="SCHLTYPE", data=df) # df is pisa_2022_canada_merged with categorical variables converted
# >                                      Estimate Std. Error t value
# > (Intercept)                            546.08      12.60   43.34
# > SCHLTYPEPrivate Government-dependent    11.48      14.69    0.78
# > SCHLTYPEPublic                         -49.15      12.64   -3.89
# > R-squared                                0.02       0.01    3.48
pisa.reg.pv(pvlabel=pvmaths, x="SCHLTYPE", data=temp_data)
# >                                      Estimate Std. Error t value
# > (Intercept)                            566.60      13.47   42.05
# > SCHLTYPEPrivate Government-dependent    -3.72      14.21   -0.26
# > SCHLTYPEPublic                         -48.85      13.40   -3.65
# > R-squared                                0.02       0.01    2.20

# --- ESCS ---
pisa.mean(variable="ESCS", data=pisa_2022_canada_merged)
# >    Freq     Mean      s.e.       SD      s.e
# > 1 21396 7.191271 0.6460369 25.01528 1.091783
pisa.mean(variable="ESCS", data=temp_data)
# >   Freq      Mean       s.e.        SD         s.e
# > 1 6755 0.4635956 0.01931614 0.7254373 0.008650945
pisa.reg.pv(pvlabel=pvmaths, x="ESCS", data=pisa_2022_canada_merged)
# >             Estimate Std. Error t value
# > (Intercept)   483.67       1.32  366.93
# > ESCS           39.90       1.60   24.98
# > R-squared       0.10       0.01   13.02
pisa.reg.pv(pvlabel=pvmaths, x="ESCS", data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   504.38       2.11  238.90
# > ESCS           37.85       2.25   16.82
# > R-squared       0.09       0.01    9.26

pisa.mean(variable="ESCS", by="REGION", data=pisa_2022_canada_merged)
# >    REGION Freq      Mean      s.e.       SD       s.e
# >  1   12401 1009  4.195890 0.7064078 19.38122 1.6891502
# >  2   12402  316 13.535846 1.7709741 33.60023 1.9103175
# >  3   12403 1425  9.849041 0.7017382 29.23425 0.9469588
# >  4   12404 1567  4.755206 0.5558620 20.59325 1.2169475
# >  5   12405 3809  8.477281 1.6776169 27.10841 2.5494093
# >  6   12406 5512  7.624446 1.2741733 25.66875 2.0869961
# >  7   12407 2440  9.021617 1.5696638 28.22215 2.2662689
# >  8   12408 2181  4.036400 0.4310709 19.06878 1.0382449
# >  9   12409 1273  4.601054 1.2957563 19.91666 2.8717227
# > 10  12410 1864  6.797501 1.7331926 24.23652 3.0355779

df <- pisa_2022_canada_merged %>%
  dplyr::select(
    CNTSCHID, CNTSTUID,
    dplyr::all_of(c(final_wt, rep_wts, "ESCS", "REGION"))
  ) %>%
  dplyr::mutate(
    REGION = if (inherits(REGION, "haven_labelled"))
      haven::as_factor(REGION, levels = "labels")
    else as.factor(REGION)
  ) %>%
  dplyr::filter(!is.na(ESCS)) %>%
  droplevels() 

pisa.reg(y="ESCS", x="REGION", data=df)
# >                                    Estimate Std. Error t value
# > (Intercept)                            0.24       0.04    6.68
# > REGIONCanada: Prince Edward Island     0.09       0.07    1.41
# > REGIONCanada: Nova Scotia              0.03       0.05    0.63
# > REGIONCanada: New Brunswick            0.02       0.04    0.50
# > REGIONCanada: Quebec                   0.12       0.04    3.00
# > REGIONCanada: Ontario                  0.18       0.05    3.97
# > REGIONCanada: Manitoba                -0.06       0.04   -1.48
# > REGIONCanada: Saskatchewan            -0.03       0.04   -0.64
# > REGIONCanada: Alberta                  0.16       0.06    2.92
# > REGIONCanada: British Columbia         0.19       0.05    3.72
# > R-squared                              0.01       0.00    3.56
pisa.reg(y="ESCS", x="REGION", data=temp_data)
# >                                    Estimate Std. Error t value
# > (Intercept)                            0.36       0.04    8.53
# > REGIONCanada: Prince Edward Island     0.13       0.12    1.08
# > REGIONCanada: Nova Scotia             -0.01       0.07   -0.13
# > REGIONCanada: New Brunswick            0.03       0.06    0.60
# > REGIONCanada: Quebec                   0.10       0.05    1.79
# > REGIONCanada: Ontario                  0.14       0.06    2.36
# > REGIONCanada: Manitoba                -0.14       0.05   -2.56
# > REGIONCanada: Saskatchewan             0.01       0.05    0.23
# > REGIONCanada: Alberta                  0.07       0.08    0.95
# > REGIONCanada: British Columbia         0.16       0.07    2.37
# > R-squared                              0.01       0.00    2.24

# Try other combo: BULLY by SCHLTYPE, etc...

### ---- Sanity check (post-prep) ----

#### ---- Dimension ----
dim(pisa_2022_canada_merged); dim(temp_data)
# > [1] 23073  1699
# > [1] 6755  151

#### ---- Missingness ----
all(
  colSums(is.na(pisa_2022_canada_merged[pvmaths])) == 0,
  sum(is.na(pisa_2022_canada_merged$W_FSTUWT)) == 0,
  colSums(is.na(pisa_2022_canada_merged[paste0("W_FSTURWT", 1:80)])) == 0
)
# > [1] TRUE
all(
  colSums(is.na(temp_data[pvmaths])) == 0,
  sum(is.na(temp_data$W_FSTUWT)) == 0,
  colSums(is.na(temp_data[paste0("W_FSTURWT", 1:80)])) == 0
)
# > [1] TRUE

sapply(pisa_2022_canada_merged[, voi_num, drop=FALSE], \(x) sum(is.na(x)))  
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >    4150      4168      4123      2912      2887      2998      2944      1402      1424      5720      2973      2865      2942      4067      4239      4660      5014      4891 
# > FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP  CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >    4856      4300      3923      3997      4236      4055      3809      4294      4803      5072      4654      4878      3534      5434      5195      5197      4949      5108 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >    5329      5231      5735      6673      8525     12471      8478      8684      7666      8460      1677      2910      2922      2575      3036      3096      3008 
sapply(pisa_2022_canada_merged[, voi_cat, drop=FALSE], \(x) sum(is.na(x)))  
# > REGION ST004D01T     IMMIG     LANGN  SCHLTYPE 
# >      0        42      2610         0         0 
#                                     ⚠️

stopifnot(all(colSums(is.na(temp_data[, voi_all])) == 0)) 
sapply(temp_data[, voi_num, drop=FALSE], \(x) sum(is.na(x)))  # expect 0 if we intended to drop missing
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 
# > FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP   CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 
sapply(temp_data[, voi_cat, drop=FALSE], \(x) sum(is.na(x)))  
# > REGION ST004D01T     IMMIG     LANGN  SCHLTYPE 
# >      0         0         0         0         0 


#### ---- Data type/variable class  ----

# For plausible values in mathematics
sapply(pisa_2022_canada_merged[pvmaths], class)       # All numeric
sapply(temp_data[pvmaths], class)                     # All numeric

# For final student weight 
sapply(pisa_2022_canada_merged[final_wt], class)      # All numeric
sapply(temp_data[final_wt], class)                    # All numeric

# For BRR replicate weights 
sapply(pisa_2022_canada_merged[rep_wts], class)       # All numeric
sapply(temp_data[rep_wts], class)                     # All numeric

# For predictors 
all(sapply(pisa_2022_canada_merged[, voi_num, drop=FALSE], is.numeric))
# > [1] TRUE
all(sapply(pisa_2022_canada_merged[, voi_cat, drop=FALSE], is.factor))
# > [1] FALSE
sapply(temp_data[, voi_num, drop=FALSE], is.numeric)  # numeric predictors are truly numeric in temp_data
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 
# >  FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP  CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 
sapply(temp_data[, voi_cat, drop=FALSE], is.factor)   # categorical predictors are truly factors in temp_data

# Levels of voi_cat
lapply(temp_data[, voi_cat, drop = FALSE], levels)
# > $REGION
# > [1] "Canada: Newfoundland and Labrador" "Canada: Prince Edward Island"      "Canada: Nova Scotia"               "Canada: New Brunswick"             "Canada: Quebec"                   
# > [6] "Canada: Ontario"                   "Canada: Manitoba"                  "Canada: Saskatchewan"              "Canada: Alberta"                   "Canada: British Columbia"         

# > $ST004D01T
# > [1] "Female"         "Male"           "Not Applicable"

# > $IMMIG
# > [1] "Native student"            "Second-Generation student" "First-Generation student" 

# > $LANGN
# > [1] "English"                "French"                 "Another language (CAN)"

# > $SCHLTYPE
# > [1] "Private independent"          "Private Government-dependent" "Public"  

##### ---- Auto-classify VOIs by type ----
## Rules:
##  - categorical: is.factor(x)
##  - binary/numeric: numeric & exactly two distinct non-missing values
##  - ordinal/numeric: numeric, integer-valued (within tol), #levels in [min_ordinal_levels, max_ordinal_levels]
##  - continuous/numeric: numeric but not binary/ordinal
## Tunables:
##  - integer_tol: tolerance to decide "integer-like"
##  - min_ordinal_levels / max_ordinal_levels: bounds for treating integer-valued vars as ordinal

classify_voi_types <- function(df,
                               vars,
                               integer_tol = 1e-8,
                               min_ordinal_levels = 3L,
                               max_ordinal_levels = 20L) {
  stopifnot(all(vars %in% names(df)))
  
  purrr::map_dfr(vars, function(v) {
    x <- df[[v]]
    n_total <- length(x)
    n_miss  <- sum(is.na(x))
    x_nm    <- x[!is.na(x)]
    
    storage <- paste(class(x), collapse = "/")
    n_unique <- dplyr::n_distinct(x_nm)
    
    if (is.factor(x)) {
      tibble::tibble(
        variable   = v,
        storage    = storage,
        n_total    = n_total,
        n_missing  = n_miss,
        n_unique   = nlevels(x),
        min        = NA_real_,
        max        = NA_real_,
        integerish = NA,
        binary     = (nlevels(x) == 2),
        ordinal    = NA,
        type       = "categorical"
      )
    } else if (is.numeric(x)) {
      # integer-like check on unique values for stability
      uniq <- if (length(x_nm)) sort(unique(x_nm)) else numeric()
      integerish <- length(uniq) == 0 || all(abs(uniq - round(uniq)) < integer_tol)
      
      is_binary  <- (n_unique == 2L)
      is_ordinal <- (!is_binary) && integerish &&
        (n_unique >= min_ordinal_levels) &&
        (n_unique <= max_ordinal_levels)
      
      type <- dplyr::case_when(
        is_binary  ~ "binary/numeric",
        is_ordinal ~ "ordinal/numeric",
        TRUE       ~ "continuous/numeric"
      )
      
      tibble::tibble(
        variable   = v,
        storage    = storage,
        n_total    = n_total,
        n_missing  = n_miss,
        n_unique   = n_unique,
        min        = if (length(x_nm)) min(x_nm) else NA_real_,
        max        = if (length(x_nm)) max(x_nm) else NA_real_,
        integerish = isTRUE(integerish),
        binary     = is_binary,
        ordinal    = is_ordinal,
        type       = type
      )
    } else {
      # Rare: character etc. Treat as categorical fallback.
      tibble::tibble(
        variable   = v,
        storage    = storage,
        n_total    = n_total,
        n_missing  = n_miss,
        n_unique   = dplyr::n_distinct(x_nm),
        min        = NA_real_,
        max        = NA_real_,
        integerish = NA,
        binary     = NA,
        ordinal    = NA,
        type       = "categorical"
      )
    }
  }) %>%
    dplyr::mutate(
      type = factor(type,
                    levels = c("categorical", "binary/numeric",
                               "ordinal/numeric", "continuous/numeric"))
    ) %>%
    dplyr::arrange(type, variable)
}

###### ---- Run classification on data ----------------------------------
types_tbl <- classify_voi_types(
  df  = temp_data,
  vars = voi_all,
  integer_tol = 1e-8,
  min_ordinal_levels = 3L,
  max_ordinal_levels = 20L
)

## Quick overview
types_tbl %>%
  dplyr::count(type, name = "n_vars")

## Inspect details (first few)
types_tbl %>%
  dplyr::select(variable, type, storage, n_unique, min, max, integerish) %>%
  print(n = Inf)

###### ---- Extract ready-to-use sets ----------------------------------------------
voi_sets <- list(
  categorical        = types_tbl %>% dplyr::filter(type == "categorical")        %>% dplyr::pull(variable),
  binary_numeric     = types_tbl %>% dplyr::filter(type == "binary/numeric")     %>% dplyr::pull(variable),
  ordinal_numeric    = types_tbl %>% dplyr::filter(type == "ordinal/numeric")    %>% dplyr::pull(variable),
  continuous_numeric = types_tbl %>% dplyr::filter(type == "continuous/numeric") %>% dplyr::pull(variable)
)
length(voi_sets); length(voi_sets$categorical); length(voi_sets$binary_numeric); length(voi_sets$ordinal_numeric); length(voi_sets$continuous_numeric)
# > [1] 4
# > [1] 5
# > [1] 3
# > [1] 6
# > [1] 44
voi_sets
# > $categorical
# > [1] "IMMIG"     "LANGN"     "REGION"    "SCHLTYPE"  "ST004D01T"

# > $binary_numeric
# > [1] "MATHEASE" "MATHMOT"  "MATHPREF"

# > $ordinal_numeric
# > [1] "ABGMATH"  "EXERPRAC" "MACTIV"   "STUDYHMW" "WORKHOME" "WORKPAY" 

# > $continuous_numeric
# > [1] "ANXMAT"    "ASSERAGR"  "BELONG"    "BULLIED"   "COGACMCO"  "COGACRCO"  "COOPAGR"   "CREATAS"   "CREATEFF"  "CREATFAM"  "CREATOOS"  "CREATOP"   "CREATSCH"  "CREENVSC"  "CURIOAGR" 
# > [16] "DIGPREP"   "DISCLIM"   "EMOCOAGR"  "EMPATAGR"  "ESCS"      "EXPO21ST"  "EXPOFA"    "FAMCON"    "FAMSUP"    "FAMSUPSL"  "FEELLAH"   "FEELSAFE"  "GROSAGR"   "HOMEPOS"   "ICTRES"   
# > [31] "IMAGINE"   "INFOSEEK"  "LEARRES"   "MATHEF21"  "MATHEFF"   "MATHPERS"  "MTTRAIN"   "OPENART"   "OPENCUL"   "PERSEVAGR" "PROBSELF"  "SCHSUST"   "SDLEFF"    "STRESAGR" 

## Sanity check: union of all buckets equals voi_all
stopifnot(setequal(unlist(voi_sets, use.names = FALSE), voi_all))

#### ---- Summary Statistics ----

## -- pisa_2022_canada_merged --

## For output plausible values in mathematics
# Summary statistics: unweighted vs weighted (point estimates)
rbind(sapply(pisa_2022_canada_merged[pvmaths], summary), `Std. Dev.` = sapply(pisa_2022_canada_merged[pvmaths], sd)) # unweighted
# >             PV1MATH   PV2MATH   PV3MATH   PV4MATH   PV5MATH   PV6MATH   PV7MATH   PV8MATH   PV9MATH  PV10MATH
# > Min.      129.01000 123.22600 135.37200 136.05700  70.46700 173.31600  99.96400 125.14500 171.46600 120.62600
# > 1st Qu.   417.85200 419.40400 418.54300 418.81200 417.96800 417.59800 419.30400 418.32100 417.41300 419.20100
# > Median    482.94200 483.72300 483.65000 483.47300 483.05200 482.38000 483.67700 483.41400 482.86700 483.41200
# > Mean      484.47017 485.27534 484.95735 484.52161 484.03173 483.86503 484.82191 484.46046 483.95478 484.86189    # <=> sapply(pvmaths, \(v) mean(pisa_2022_canada_merged[[v]]))
# > 3rd Qu.   548.38800 549.39700 548.97600 548.57900 547.78800 547.80100 548.51400 548.48600 548.10500 548.44500
# > Max.      851.46700 875.27500 835.73000 895.75300 860.15500 885.77800 821.98400 827.70000 875.04500 903.00900
# > Std. Dev.  93.09758  92.86929  92.89535  93.02362  92.91727  92.68206  92.10989  92.96545  93.15956  91.92924

pisa.per.pv(pvlabel=pvmaths, per=c(0, 25, 50, 75, 100), data=pisa_2022_canada_merged)     # weighted: all ten PVs in math
# >   Percentiles  Score Std. err.
# > 1           0 128.46     31.74
# > 2          25 430.49      1.72
# > 3          50 496.32      1.80
# > 4          75 561.63      2.16
# > 5         100 863.19     29.87
pisa.per.pv(pvlabel=pvmaths[1], per=c(0, 25, 50, 75, 100), data=pisa_2022_canada_merged)  # weighted: only PV1MATH; repeat for others
# >   Percentiles  Score Std. err.
# > 1           0 129.01       NaN
# > 2          25 430.23       NaN
# > 3          50 496.74       NaN
# > 4          75 561.90       NaN
# > 5         100 851.47       NaN
sapply(pvmaths, \(pvmath) {
  out <- intsvy::pisa.per.pv(pvlabel = pvmath, per = c(0, 25, 50, 75, 100), data = pisa_2022_canada_merged) # Percentiles (0, 25, 50, 75, 100) for each PV, one PV at a time
  out$Score                                                                                                 # grab just the point estimates; SEs are NaN for single-PV calls
})
# >      PV1MATH PV2MATH PV3MATH PV4MATH PV5MATH PV6MATH PV7MATH PV8MATH PV9MATH PV10MATH
# > [1,]  129.01  123.23  135.37  136.06   70.47  173.32   99.96  125.14  171.47   120.63
# > [2,]  430.23  431.34  430.48  430.76  430.46  429.34  430.69  430.26  430.38   430.97
# > [3,]  496.74  496.97  496.11  496.53  496.29  496.08  496.63  496.26  495.89   495.66
# > [4,]  561.90  563.11  561.46  562.00  560.76  560.66  561.98  562.08  560.94   561.38
# > [5,]  851.47  875.28  835.73  895.75  860.16  885.78  821.98  827.70  875.04   903.01
#
# `rownames<-`(
#   sapply(pvmaths, \(pvmath)
#          intsvy::pisa.per.pv(pvlabel = pvmath,
#                              per = c(0, 25, 50, 75, 100),
#                              data = pisa_2022_canada_merged)$Score),
#   c("0", "25", "50", "75", "100")
# )
#
# structure(
#   sapply(pvmaths, \(pvmath)
#          intsvy::pisa.per.pv(pvlabel = pvmath,
#                              per = c(0, 25, 50, 75, 100),
#                              data = pisa_2022_canada_merged)$Score),
#   dimnames = list(c("Min.", "1st Qu.", "Median", "3rd Qu.", "Max."), pvmaths)
# )
(\(m) cbind(Percentiles = rownames(m), m))(
  structure(
    sapply(pvmaths, \(pv)
           intsvy::pisa.per.pv(pvlabel = pv, per = c(0,25,50,75,100),
                               data = pisa_2022_canada_merged)$Score),
    dimnames = list(c("Min.", "1st Qu.", "Median", "3rd Qu.", "Max."), pvmaths)
  )
)

(\(m) cbind(Percentiles = rownames(m), m))(
  `rownames<-`(
    sapply(pvmaths, \(pv)
           intsvy::pisa.per.pv(pvlabel = pv,
                               per = c(0,25,50,75,100),
                               data = pisa_2022_canada_merged)$Score),
    c("Min.", "1st Qu.", "Median", "3rd Qu.", "Max.")
  )
)


# Mean: unweighted vs weighted (point estimates)
sapply(pvmaths, \(v) mean(pisa_2022_canada_merged[[v]]))
# > PV1MATH  PV2MATH  PV3MATH  PV4MATH  PV5MATH  PV6MATH  PV7MATH  PV8MATH  PV9MATH PV10MATH 
# > 484.4702 485.2753 484.9574 484.5216 484.0317 483.8650 484.8219 484.4605 483.9548 484.8619 
sapply(pvmaths, \(v) weighted.mean(pisa_2022_canada_merged[[v]],
                                   pisa_2022_canada_merged$W_FSTUWT, na.rm=TRUE))
# > PV1MATH  PV2MATH  PV3MATH  PV4MATH  PV5MATH  PV6MATH  PV7MATH  PV8MATH  PV9MATH PV10MATH 
# > 496.9132 497.7168 497.1623 496.8984 496.8574 496.2980 497.3204 496.8787 496.5279 496.9057 

## For final student weight
rbind(sapply(pisa_2022_canada_merged[final_wt], summary), `Std. Dev.` = sapply(pisa_2022_canada_merged[final_wt], sd))
# >            W_FSTUWT
# > Min.        1.01499
# > 1st Qu.     3.74247
# > Median      8.85904
# > Mean       15.51213
# > 3rd Qu.    25.97300
# > Max.      833.47350
# > Std. Dev.  15.65097

## For BRR replicate weights
rbind(sapply(pisa_2022_canada_merged[rep_wts], summary), `Std. Dev.` = sapply(pisa_2022_canada_merged[rep_wts], sd))

## For inputs X
rbind(sapply(pisa_2022_canada_merged[voi_num], summary), `Std. Dev.` = sapply(pisa_2022_canada_merged[voi_num], sd))
summary(pisa_2022_canada_merged[, voi_cat])

## -- temp_data --

# For plausible values in mathematics
# sapply(temp_data[pvmaths], summary)
# sapply(temp_data[pvmaths], sd)
rbind(sapply(temp_data[pvmaths], summary), `Std. Dev.` = sapply(temp_data[pvmaths], sd))
# >             PV1MATH   PV2MATH   PV3MATH   PV4MATH  PV5MATH   PV6MATH   PV7MATH  PV8MATH   PV9MATH  PV10MATH
# > Min.      205.34000 214.89200 208.1770 220.47600 195.81300 244.63900 211.45100 221.4090 202.35200 189.22600
# > 1st Qu.   447.06400 450.31800 447.6065 449.68350 449.29300 446.00250 449.21550 446.4630 447.87050 448.79300
# > Median    509.03600 509.75200 508.4520 509.13500 509.24900 507.68300 508.92100 508.3250 508.56000 509.59000
# > Mean      510.53804 511.37593 510.3917 510.35401 509.90147 508.61530 509.98114 509.6892 509.86894 510.40403
# > 3rd Qu.   572.00100 573.09150 570.5080 570.35900 568.63600 569.87350 569.58250 571.0085 570.85500 570.31000
# > Max.      842.35800 875.27500 830.5470 836.22900 860.15500 885.77800 803.36800 802.4270 849.63700 903.00900
# > Std. Dev.  89.34665  89.18722  89.0885  88.46049  88.93493  88.91212  88.11456  89.2745  89.47526  88.02003

# For final student weight
rbind(sapply(temp_data[final_wt], summary), `Std. Dev.` = sapply(temp_data[final_wt], sd))
# >            W_FSTUWT
# > Min.        1.04731
# > 1st Qu.     4.29236
# > Median     10.15096
# > Mean       16.16633
# > 3rd Qu.    26.30068
# > Max.      132.18970
# > Std. Dev.  13.82717

# For BRR replicate weights
rbind(sapply(temp_data[rep_wts], summary), `Std. Dev.` = sapply(temp_data[rep_wts], sd))

# For predictors 
round(rbind(sapply(temp_data[voi_num], summary), `Std. Dev.` = sapply(temp_data[voi_num], sd)),4)
summary(temp_data[, voi_cat])  # by default it only shows the top 6 levels and collapses the rest into (Other)
# >                        REGION              ST004D01T                          IMMIG                         LANGN                              SCHLTYPE   
# > Canada: Ontario         :1596   Female        :3565   Native student           :5013   English               :4755   Private independent         : 373  
# > Canada: Quebec          :1179   Male          :3173   Second-Generation student: 866   French                :1089   Private Government-dependent: 232  
# > Canada: Manitoba        : 782   Not Applicable:  17   First-Generation student : 876   Another language (CAN): 911   Public                      :6150  
# > Canada: Saskatchewan    : 687                                                                                                                           
# > Canada: British Columbia: 681                                                                                                                           
# > Canada: New Brunswick   : 572                                                                                                                           
# > (Other)                 :1258  
table(temp_data$REGION, useNA = "ifany")
# > Canada: Newfoundland and Labrador      Canada: Prince Edward Island               Canada: Nova Scotia             Canada: New Brunswick                    Canada: Quebec 
# >                               357                                91                               380                               572                              1179 
# > Canada: Ontario                  Canada: Manitoba              Canada: Saskatchewan                   Canada: Alberta          Canada: British Columbia 
# >            1596                               782                               687                               430                               681

# --- *Weighted Counts ---


#### ---- Data Visualization ----

## -- pisa_2022_canada_merged --

# For plausible values in mathematics (unweighted)
pisa_2022_canada_merged %>%
  select(all_of(pvmaths)) %>%
  pivot_longer(everything(), names_to = "PlausibleValue", values_to = "Score") %>%
  ggplot(aes(x = Score)) +
  geom_density(fill = "skyblue", color=NA, alpha = 0.6) +
  facet_wrap(~ PlausibleValue, scales = "free", ncol = 5) +
  theme_minimal() +
  labs(title = "Distribution of PV1MATH to PV10MATH (unweighted, pisa_2022_canada_merged)", 
       x = "Score", y = "Density"
  ) + 
  theme(plot.title = element_text(hjust = 0.5),
        strip.text = element_text(hjust = 0.5)) 

# For plausible values in mathematics (weighted with final student weight)
pisa_2022_canada_merged |>
  select(W_FSTUWT, all_of(pvmaths)) |>
  pivot_longer(all_of(pvmaths), names_to = "PlausibleValue", values_to = "Score") |>
  ggplot(aes(x = Score, weight = W_FSTUWT)) +     # <- apply design weight
  geom_density(fill = "skyblue", color = NA, alpha = 0.6, na.rm = TRUE) +
  facet_wrap(~ PlausibleValue, scales = "free", ncol = 5) +
  theme_minimal() +
  labs(
    title = "Distributions of PV1MATH–PV10MATH (weighted, pisa_2022_canada_merged)",
    x = "Score", y = "Weighted density"
  ) +
  theme(plot.title = element_text(hjust = 0.5),
        strip.text  = element_text(hjust = 0.5))


# For final student weight
ggplot(pisa_2022_canada_merged, aes(x = .data[[final_wt]])) +
  geom_density(fill = "skyblue", color=NA, alpha = 0.6) +
  theme_minimal() +
  labs(title = "Distribution of Final Student Weight (pisa_2022_canada_merged)", x = final_wt, y = "Density") +
  theme(plot.title = element_text(hjust = 0.5))

# For BRR replicate weights
pisa_2022_canada_merged %>%
  select(all_of(rep_wts)) %>%
  pivot_longer(everything(), names_to = "RepWeight", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_density(fill = "skyblue", color=NA, alpha = 0.6) +
  facet_wrap(~ RepWeight, scales = "free", ncol = 8) +
  theme_minimal() +
  labs(title = "Distribution of 80 BRR Replicate Weights (pisa_2022_canada_merged)", x = "Weight", y = "Density") +
  theme(plot.title = element_text(hjust = 0.5),
        strip.text = element_text(hjust = 0.5))

# ## For predictors: numeric
# pisa_2022_canada_merged %>%
#   select(all_of(voi_num)) %>%
#   pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
#   ggplot(aes(x = factor(Value))) +
#   geom_bar(fill = "steelblue") +
#   facet_wrap(~ Variable, scales = "free_y") +
#   scale_x_discrete(drop = FALSE) +
#   labs(x = "Value", y = "Count") +
#   theme_minimal()
# # Box plot (order by median)
# pisa_2022_canada_merged %>%
#   select(all_of(voi_num)) %>%
#   pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
#   group_by(variable) %>% mutate(med = median(value, na.rm = TRUE)) %>% ungroup() %>%
#   ggplot(aes(x = reorder(variable, med), y = value)) +
#   geom_boxplot(outlier.alpha = 0.4) +
#   coord_flip() + labs(x = "", y = "Value") + theme_minimal()
# # Standardize per variable (z-scores) just for visulization
# pisa_2022_canada_merged %>%
#   select(all_of(voi_num)) %>%
#   mutate(across(everything(), ~ as.numeric(scale(.)))) %>%
#   pivot_longer(everything(), names_to = "variable", values_to = "z") %>%
#   ggplot(aes(variable, z)) +
#   geom_boxplot(outlier.alpha = 0.4) +
#   coord_flip() + labs(y = "Standardized value (z)") + theme_minimal()

## For predictors: voi_cat
# Simple distributions (one panel per categorical variable)
pisa_2022_canada_merged %>%
  mutate(across(all_of(voi_cat), ~ if (inherits(.x,"haven_labelled")) as_factor(.x,"labels") else as.factor(.x))) %>%
  select(all_of(voi_cat)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "level") %>%
  ggplot(aes(level)) +
  geom_bar() +
  facet_wrap(~ variable, scales = "free_y") +
  coord_flip() +
  labs(x = NULL, y = "Count") +
  theme_minimal()

# Weighted shares per variable: requires final weight W_FSTUWT
pisa_2022_canada_merged %>%
  mutate(across(all_of(voi_cat), ~ if (inherits(.x,"haven_labelled")) as_factor(.x,"labels") else as.factor(.x))) %>%
  select(all_of(voi_cat), W_FSTUWT) %>%
  pivot_longer(all_of(voi_cat), names_to = "variable", values_to = "level") %>%
  group_by(variable, level) %>%
  summarise(w_n = sum(W_FSTUWT, na.rm = TRUE), .groups = "drop_last") %>%
  mutate(w_pct = w_n / sum(w_n)) %>%
  ungroup() %>%
  ggplot(aes(x = reorder(level, w_pct), y = w_pct)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  facet_wrap(~ variable, scales = "free_y") +
  labs(x = NULL, y = "Weighted share (W_FSTUWT)") +
  theme_minimal()

# Dodged bars for a categorical vs categorical
# Example: LANGN by REGION (weighted)
pisa_2022_canada_merged %>%
  mutate(
    REGION = if (inherits(REGION,"haven_labelled")) as_factor(REGION,"labels") else as.factor(REGION),
    LANGN  = if (inherits(LANGN,"haven_labelled"))  as_factor(LANGN,"labels")  else as.factor(LANGN)
  ) %>%
  filter(!is.na(REGION), !is.na(LANGN)) %>%
  count(REGION, LANGN, wt = W_FSTUWT, name = "w_n") %>%
  group_by(REGION) %>% mutate(w_pct = w_n / sum(w_n)) %>% ungroup() %>%
  ggplot(aes(x = REGION, y = w_pct, fill = LANGN)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(x = NULL, y = "Weighted share", fill = "Test language") +
  theme_minimal()
# Example: SCHLTYPE by REGION (weighted)
pisa_2022_canada_merged %>%
  mutate(
    REGION = if (inherits(REGION,"haven_labelled")) as_factor(REGION,"labels") else as.factor(REGION),
    SCHLTYPE  = if (inherits(SCHLTYPE,"haven_labelled"))  as_factor(SCHLTYPE,"labels")  else as.factor(SCHLTYPE)
  ) %>%
  filter(!is.na(REGION), !is.na(SCHLTYPE)) %>%
  count(REGION, SCHLTYPE, wt = W_FSTUWT, name = "w_n") %>%
  group_by(REGION) %>% mutate(w_pct = w_n / sum(w_n)) %>% ungroup() %>%
  ggplot(aes(x = REGION, y = w_pct, fill = SCHLTYPE)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(x = NULL, y = "Weighted share", fill = "Test language") +
  theme_minimal()
# Example: IMMIG by REGION (weighted)
pisa_2022_canada_merged %>%
  mutate(
    REGION = if (inherits(REGION,"haven_labelled")) as_factor(REGION,"labels") else as.factor(REGION),
    IMMIG  = if (inherits(IMMIG,"haven_labelled"))  as_factor(IMMIG,"labels")  else as.factor(IMMIG)
  ) %>%
  filter(!is.na(REGION), !is.na(IMMIG)) %>%
  count(REGION, IMMIG, wt = W_FSTUWT, name = "w_n") %>%
  group_by(REGION) %>% mutate(w_pct = w_n / sum(w_n)) %>% ungroup() %>%
  ggplot(aes(x = REGION, y = w_pct, fill = IMMIG)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(x = NULL, y = "Weighted share", fill = "Test language") +
  theme_minimal()
# Example: ST004D01T by REGION (weighted)
pisa_2022_canada_merged %>%
  mutate(
    REGION = if (inherits(REGION,"haven_labelled")) as_factor(REGION,"labels") else as.factor(REGION),
    ST004D01T  = if (inherits(ST004D01T,"haven_labelled"))  as_factor(ST004D01T,"labels")  else as.factor(ST004D01T)
  ) %>%
  filter(!is.na(REGION), !is.na(ST004D01T)) %>%
  count(REGION, ST004D01T, wt = W_FSTUWT, name = "w_n") %>%
  group_by(REGION) %>% mutate(w_pct = w_n / sum(w_n)) %>% ungroup() %>%
  ggplot(aes(x = REGION, y = w_pct, fill = ST004D01T)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(x = NULL, y = "Weighted share", fill = "Test language") +
  theme_minimal()


## -- temp_data --
# For plausible values in mathematics (unweighted, temp_data)
temp_data %>%
  select(all_of(pvmaths)) %>%
  pivot_longer(everything(), names_to = "PlausibleValue", values_to = "Score") %>%
  ggplot(aes(x = Score)) +
  geom_density(fill = "lightgreen", color = NA, alpha = 0.6, na.rm = TRUE) +
  facet_wrap(~ PlausibleValue, scales = "free", ncol = 5) +
  theme_minimal() +
  labs(
    title = "Distribution of PV1MATH to PV10MATH (unweighted, temp_data)",
    x = "Score", y = "Density"
  ) +
  theme(plot.title = element_text(hjust = 0.5),
        strip.text  = element_text(hjust = 0.5))

# For plausible values in mathematics (weighted with final student weight, temp_data)
temp_data %>%
  select(W_FSTUWT, all_of(pvmaths)) %>%
  pivot_longer(all_of(pvmaths), names_to = "PlausibleValue", values_to = "Score") %>%
  ggplot(aes(x = Score, weight = W_FSTUWT)) +    # <- apply design weight
  geom_density(fill = "lightgreen", color = NA, alpha = 0.6, na.rm = TRUE) +
  facet_wrap(~ PlausibleValue, scales = "free", ncol = 5) +
  theme_minimal() +
  labs(
    title = "Distributions of PV1MATH–PV10MATH (weighted, temp_data)",
    x = "Score", y = "Weighted density"
  ) +
  theme(plot.title = element_text(hjust = 0.5),
        strip.text  = element_text(hjust = 0.5))


# For final student weight
ggplot(temp_data, aes(x = .data[[final_wt]])) +
  geom_density(fill = "lightgreen", color=NA, alpha = 0.6) +
  theme_minimal() +
  labs(title = "Distribution of Final Student Weight (temp_data)", x = final_wt, y = "Density") +
  theme(plot.title = element_text(hjust = 0.5))

# For BRR replicate weights
temp_data %>%
  select(all_of(rep_wts)) %>%
  pivot_longer(everything(), names_to = "RepWeight", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_density(fill = "lightgreen", color=NA, alpha = 0.6) +
  facet_wrap(~ RepWeight, scales = "free", ncol = 8) +
  theme_minimal() +
  labs(title = "Distribution of 80 BRR Replicate Weights (temp_data)", x = "Weight", y = "Density") +
  theme(plot.title = element_text(hjust = 0.5),
        strip.text = element_text(hjust = 0.5)) 

## For predictors: voi_num
temp_data %>%
  select(all_of(voi_num)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = factor(Value))) +
  geom_bar(fill = "steelblue") +
  facet_wrap(~ Variable, scales = "free_y") +
  scale_x_discrete(drop = FALSE) +
  labs(x = "Value", y = "Count") +
  theme_minimal()
# Box plot (order by median)
temp_data %>%
  select(all_of(voi_num)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  group_by(variable) %>% mutate(med = median(value, na.rm = TRUE)) %>% ungroup() %>%
  ggplot(aes(x = reorder(variable, med), y = value)) +
  geom_boxplot(outlier.alpha = 0.4) +
  coord_flip() + labs(x = "", y = "Value") + theme_minimal()
# Standardize per variable (z-scores) just for visulization
temp_data %>%
  select(all_of(voi_num)) %>%
  mutate(across(everything(), ~ as.numeric(scale(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "z") %>%
  ggplot(aes(variable, z)) +
  geom_boxplot(outlier.alpha = 0.4) +
  coord_flip() + labs(y = "Standardized value (z)") + theme_minimal()

## For predictors: voi_cat
# Simple distributions (one panel per categorical variable)
temp_data %>%
  mutate(across(all_of(voi_cat), ~ if (inherits(.x,"haven_labelled")) as_factor(.x,"labels") else as.factor(.x))) %>%
  select(all_of(voi_cat)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "level") %>%
  ggplot(aes(level)) +
  geom_bar() +
  facet_wrap(~ variable, scales = "free_y") +
  coord_flip() +
  labs(x = NULL, y = "Count") +
  theme_minimal()

# Weighted shares per variable: requires final weight W_FSTUWT
temp_data %>%
  mutate(across(all_of(voi_cat), ~ if (inherits(.x,"haven_labelled")) as_factor(.x,"labels") else as.factor(.x))) %>%
  select(all_of(voi_cat), W_FSTUWT) %>%
  pivot_longer(all_of(voi_cat), names_to = "variable", values_to = "level") %>%
  group_by(variable, level) %>%
  summarise(w_n = sum(W_FSTUWT, na.rm = TRUE), .groups = "drop_last") %>%
  mutate(w_pct = w_n / sum(w_n)) %>%
  ungroup() %>%
  ggplot(aes(x = reorder(level, w_pct), y = w_pct)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  facet_wrap(~ variable, scales = "free_y") +
  labs(x = NULL, y = "Weighted share (W_FSTUWT)") +
  theme_minimal()

# Dodged bars for a categorical vs categorical
# Example: LANGN by REGION (weighted)
temp_data %>%
  mutate(
    REGION = if (inherits(REGION,"haven_labelled")) as_factor(REGION,"labels") else as.factor(REGION),
    LANGN  = if (inherits(LANGN,"haven_labelled"))  as_factor(LANGN,"labels")  else as.factor(LANGN)
  ) %>%
  filter(!is.na(REGION), !is.na(LANGN)) %>%
  count(REGION, LANGN, wt = W_FSTUWT, name = "w_n") %>%
  group_by(REGION) %>% mutate(w_pct = w_n / sum(w_n)) %>% ungroup() %>%
  ggplot(aes(x = REGION, y = w_pct, fill = LANGN)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(x = NULL, y = "Weighted share", fill = "Test language") +
  theme_minimal()
# Example: SCHLTYPE by REGION (weighted)
temp_data %>%
  mutate(
    REGION = if (inherits(REGION,"haven_labelled")) as_factor(REGION,"labels") else as.factor(REGION),
    SCHLTYPE  = if (inherits(SCHLTYPE,"haven_labelled"))  as_factor(SCHLTYPE,"labels")  else as.factor(SCHLTYPE)
  ) %>%
  filter(!is.na(REGION), !is.na(SCHLTYPE)) %>%
  count(REGION, SCHLTYPE, wt = W_FSTUWT, name = "w_n") %>%
  group_by(REGION) %>% mutate(w_pct = w_n / sum(w_n)) %>% ungroup() %>%
  ggplot(aes(x = REGION, y = w_pct, fill = SCHLTYPE)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(x = NULL, y = "Weighted share", fill = "Test language") +
  theme_minimal()
# Example: IMMIG by REGION (weighted)
temp_data %>%
  mutate(
    REGION = if (inherits(REGION,"haven_labelled")) as_factor(REGION,"labels") else as.factor(REGION),
    IMMIG  = if (inherits(IMMIG,"haven_labelled"))  as_factor(IMMIG,"labels")  else as.factor(IMMIG)
  ) %>%
  filter(!is.na(REGION), !is.na(IMMIG)) %>%
  count(REGION, IMMIG, wt = W_FSTUWT, name = "w_n") %>%
  group_by(REGION) %>% mutate(w_pct = w_n / sum(w_n)) %>% ungroup() %>%
  ggplot(aes(x = REGION, y = w_pct, fill = IMMIG)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(x = NULL, y = "Weighted share", fill = "Test language") +
  theme_minimal()
# Example: ST004D01T by REGION (weighted)
temp_data %>%
  mutate(
    REGION = if (inherits(REGION,"haven_labelled")) as_factor(REGION,"labels") else as.factor(REGION),
    ST004D01T  = if (inherits(ST004D01T,"haven_labelled"))  as_factor(ST004D01T,"labels")  else as.factor(ST004D01T)
  ) %>%
  filter(!is.na(REGION), !is.na(ST004D01T)) %>%
  count(REGION, ST004D01T, wt = W_FSTUWT, name = "w_n") %>%
  group_by(REGION) %>% mutate(w_pct = w_n / sum(w_n)) %>% ungroup() %>%
  ggplot(aes(x = REGION, y = w_pct, fill = ST004D01T)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(x = NULL, y = "Weighted share", fill = "Test language") +
  theme_minimal()

### ---- Individual Variable Exploration -----

#### ---- 1) Numeric Variables ----

##### ---- Student Questionaire Derived Variables ----

###### ---- Simple questionnaire indices ----

####### ---- Subject-specific beliefs, attitudes, feelings and behaviours (Module 7) ----

######## ---- MATHMOT: Relative motivation to do well in mathematics compared to other core subjects ----
class(pisa_2022_canada_merged$MATHMOT)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$MATHMOT)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$MATHMOT))
# > [1] 4150
sum(is.na(temp_data$MATHMOT))
# > [1] 0
table(pisa_2022_canada_merged$MATHMOT, useNA="always")
# >     0     1     9  <NA> 
# > 18146   777  4150     0 
table(temp_data$MATHMOT, useNA = "always") 
# >    0    1 <NA> 
# > 6680  264    0 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$MATHMOT, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$MATHMOT,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$MATHMOT, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$MATHMOT,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$MATHMOT, useNA="always")),
           row.names=NULL)
# >   code                                                            label     n
# > 1    0 Not more motivated to do well in mathematics than other subjects 18146
# > 2    1     More motivated to do well in mathematics than other subjects   777
# > 3    9                                                      No Response  4150
# > 4   NA                                                             <NA>     0
data.frame(code=as.numeric(names(table(temp_data$MATHMOT, useNA="always"))),
           label=names(attr(temp_data$MATHMOT,"labels"))[
             match(as.numeric(names(table(temp_data$MATHMOT, useNA="always"))),
                   unname(attr(temp_data$MATHMOT,"labels")))],
           n=as.integer(table(temp_data$MATHMOT, useNA="always")),
           row.names=NULL) 
# >   code                                                            label    n
# > 1    0 Not more motivated to do well in mathematics than other subjects 6499
# > 2    1     More motivated to do well in mathematics than other subjects  256
# > 3   NA                                                             <NA>    0

######## ---- MATHEASE: Perception of mathematics as easier than other core subjects ----
class(pisa_2022_canada_merged$MATHEASE)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$MATHEASE)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$MATHEASE))
# > [1] 4168
sum(is.na(temp_data$MATHEASE))
# > [1] 0
table(pisa_2022_canada_merged$MATHEASE, useNA="always")
# >     0     1     9  <NA> 
# > 16310  2595  4168     0 
table(temp_data$MATHEASE, useNA = "always") 
# >    0    1 <NA> 
# > 5994  950    0 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$MATHEASE, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$MATHEASE,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$MATHEASE, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$MATHEASE,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$MATHEASE, useNA="always")),
           row.names=NULL)
# >   code                                                      label     n
# > 1    0 No perception of mathematics as easier than other subjects 16310
# > 2    1    Perception of mathematics as easier than other subjects  2595
# > 3    9                                                No Response  4168
# > 4   NA                                                       <NA>     0
data.frame(code=as.numeric(names(table(temp_data$MATHEASE, useNA="always"))),
           label=names(attr(temp_data$MATHEASE,"labels"))[
             match(as.numeric(names(table(temp_data$MATHEASE, useNA="always"))),
                   unname(attr(temp_data$MATHEASE,"labels")))],
           n=as.integer(table(temp_data$MATHEASE, useNA="always")),
           row.names=NULL) 
# >   code                                                      label    n
# > 1    0 No perception of mathematics as easier than other subjects 5994
# > 2    1    Perception of mathematics as easier than other subjects  950
# > 3   NA                                                       <NA>    0

######## ---- MATHPREF: Preference of mathematics over other core subjects ----
class(pisa_2022_canada_merged$MATHPREF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$MATHPREF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$MATHPREF))
# > [1] 4123
sum(is.na(temp_data$MATHPREF))
# > [1] 0
table(pisa_2022_canada_merged$MATHPREF, useNA="always")
# >     0     1     9  <NA> 
# > 16181  2769  4123     0 
table(temp_data$MATHPREF, useNA = "always") 
# >    0    1 <NA> 
# > 5915 1029    0  
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$MATHPREF, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$MATHPREF,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$MATHPREF, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$MATHPREF,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$MATHPREF, useNA="always")),
           row.names=NULL)
# >   code                                             label     n
# > 1    0 No preference for mathematics over other subjects 16181
# > 2    1    Preference for mathematics over other subjects  2769
# > 3    9                                       No Response  4123
# > 4   NA                                              <NA>     0
data.frame(code=as.numeric(names(table(temp_data$MATHPREF, useNA="always"))),
           label=names(attr(temp_data$MATHPREF,"labels"))[
             match(as.numeric(names(table(temp_data$MATHPREF, useNA="always"))),
                   unname(attr(temp_data$MATHPREF,"labels")))],
           n=as.integer(table(temp_data$MATHPREF, useNA="always")),
           row.names=NULL) 
# >   code                                             label    n
# > 1    0 No preference for mathematics over other subjects 5915
# > 2    1    Preference for mathematics over other subjects 1029
# > 3   NA                                              <NA>    0

####### ---- Out-of-school experiences (Module 10) ----

######## ---- EXERPRAC: Exercise or practice a sport before or after school ----
class(pisa_2022_canada_merged$EXERPRAC)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$EXERPRAC)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$EXERPRAC))
# > [1] 2912
sum(is.na(temp_data$EXERPRAC))
# > [1] 0
table(pisa_2022_canada_merged$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912    0 
#prop.table(table(pisa_2022_canada_merged$EXERPRAC)) 
round(prop.table(table(pisa_2022_canada_merged$EXERPRAC)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 0.16 0.04 0.09 0.08 0.09 0.10 0.07 0.03 0.05 0.02 0.13 0.13 
table(temp_data$EXERPRAC, useNA = "always") 
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 1324  337  793  696  764  789  593  258  408  122  860    0 
#prop.table(table(temp_data$EXERPRAC)) 
round(prop.table(table(temp_data$EXERPRAC)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 0.19 0.05 0.11 0.10 0.11 0.11 0.09 0.04 0.06 0.02 0.12
#c(summary(pisa_2022_canada_merged$EXERPRAC), SD = sd(pisa_2022_canada_merged$EXERPRAC))
c(summary(temp_data$EXERPRAC), SD = sd(temp_data$EXERPRAC))
# >     Min.   1st Qu.    Median      Mean   3rd Qu.     M ax.        SD 
# > 0.000000  2.000000  4.000000  4.224942  6.000000 10.000000  3.259276 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$EXERPRAC, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$EXERPRAC,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$EXERPRAC, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$EXERPRAC,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$EXERPRAC, useNA="always")),
           row.names=NULL)
# >    code                                             label    n
# > 1     0                             No exercise or sports 3807
# > 2     1           1 time of exercising or sports per week  984
# > 3     2          2 times of exercising or sports per week 2103
# > 4     3          3 times of exercising or sports per week 1833
# > 5     4          4 times of exercising or sports per week 2129
# > 6     5          5 times of exercising or sports per week 2222
# > 7     6          6 times of exercising or sports per week 1673
# > 8     7          7 times of exercising or sports per week  705
# > 9     8          8 times of exercising or sports per week 1264
# > 10    9          9 times of exercising or sports per week  391
# > 11   10 10 or more times of exercising or sports per week 3050
# > 12   99                                       No Response 2912
# > 13   NA                                              <NA>    0
data.frame(code=as.numeric(names(table(temp_data$EXERPRAC, useNA="always"))),
           label=names(attr(temp_data$EXERPRAC,"labels"))[
             match(as.numeric(names(table(temp_data$EXERPRAC, useNA="always"))),
                   unname(attr(temp_data$EXERPRAC,"labels")))],
           n=as.integer(table(temp_data$EXERPRAC, useNA="always")),
           row.names=NULL) 
# >    code                                             label    n
# > 1     0                             No exercise or sports 1324
# > 2     1           1 time of exercising or sports per week  337
# > 3     2          2 times of exercising or sports per week  793
# > 4     3          3 times of exercising or sports per week  696
# > 5     4          4 times of exercising or sports per week  764
# > 6     5          5 times of exercising or sports per week  789
# > 7     6          6 times of exercising or sports per week  593
# > 8     7          7 times of exercising or sports per week  258
# > 9     8          8 times of exercising or sports per week  408
# > 10    9          9 times of exercising or sports per week  122
# > 11   10 10 or more times of exercising or sports per week  860
# > 12   NA                                              <NA>    0

hist(pisa_2022_canada_merged$EXERPRAC, breaks=11)
barplot(table(pisa_2022_canada_merged$EXERPRAC))
boxplot(pisa_2022_canada_merged$EXERPRAC)

hist(temp_data$EXERPRAC, breaks=11)
barplot(table(temp_data$EXERPRAC))
boxplot(temp_data$EXERPRAC)

######## ---- STUDYHMW: Studying for school or homework before or after school ---- 
class(pisa_2022_canada_merged$STUDYHMW)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$STUDYHMW)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$STUDYHMW))
# > [1] 2887
sum(is.na(temp_data$STUDYHMW))
# > [1] 0
table(pisa_2022_canada_merged$STUDYHMW, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 2535 1295 2426 2224 2553 2511 2028  953 1125  411 2125 2887    0 
#prop.table(table(pisa_2022_canada_merged$STUDYHMW)) 
round(prop.table(table(pisa_2022_canada_merged$STUDYHMW)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 0.11 0.06 0.11 0.10 0.11 0.11 0.09 0.04 0.05 0.02 0.09 0.13 
table(temp_data$STUDYHMW, useNA = "always") 
# >   0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 729  425  854  809  940  905  740  349  403  138  652    0 
#prop.table(table(temp_data$STUDYHMW)) 
round(prop.table(table(temp_data$STUDYHMW)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 0.10 0.06 0.12 0.12 0.14 0.13 0.11 0.05 0.06 0.02 0.09 
#c(summary(pisa_2022_canada_merged$STUDYHMW), SD = sd(pisa_2022_canada_merged$STUDYHMW))
c(summary(temp_data$STUDYHMW), SD = sd(temp_data$STUDYHMW))
# >     Min.   1st Qu.    Median      Mean   3rd Qu.      Max.        SD 
# > 0.000000  2.000000  4.000000  4.423099  6.000000 10.000000  2.911309 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$STUDYHMW, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$STUDYHMW,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$STUDYHMW, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$STUDYHMW,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$STUDYHMW, useNA="always")),
           row.names=NULL)
# >    code                              label    n
# > 1     0                        No studying 2535
# > 2     1        1 time of studying per week 1295
# > 3     2       2 times of studying per week 2426
# > 4     3       3 times of studying per week 2224
# > 5     4       4 times of studying per week 2553
# > 6     5       5 times of studying per week 2511
# > 7     6       6 times of studying per week 2028
# > 8     7       7 times of studying per week  953
# > 9     8       8 times of studying per week 1125
# > 10    9       9 times of studying per week  411
# > 11   10 10 or more times of study per week 2125
# > 12   99                        No Response 2887
# > 13   NA                               <NA>    0
data.frame(code=as.numeric(names(table(temp_data$STUDYHMW, useNA="always"))),
           label=names(attr(temp_data$STUDYHMW,"labels"))[
             match(as.numeric(names(table(temp_data$STUDYHMW, useNA="always"))),
                   unname(attr(temp_data$STUDYHMW,"labels")))],
           n=as.integer(table(temp_data$STUDYHMW, useNA="always")),
           row.names=NULL) 
# >    code                              label   n
# > 1     0                        No studying 729
# > 2     1        1 time of studying per week 425
# > 3     2       2 times of studying per week 854
# > 4     3       3 times of studying per week 809
# > 5     4       4 times of studying per week 940
# > 6     5       5 times of studying per week 905
# > 7     6       6 times of studying per week 740
# > 8     7       7 times of studying per week 349
# > 9     8       8 times of studying per week 403
# > 10    9       9 times of studying per week 138
# > 11   10 10 or more times of study per week 652
# > 12   NA                               <NA>   0

hist(pisa_2022_canada_merged$STUDYHMW, breaks=11)
barplot(table(pisa_2022_canada_merged$STUDYHMW))
boxplot(pisa_2022_canada_merged$STUDYHMW)

hist(temp_data$STUDYHMW, breaks=11)
barplot(table(temp_data$STUDYHMW))
boxplot(temp_data$STUDYHMW)

######## ---- WORKPAY: Working for pay before or after school ----
class(pisa_2022_canada_merged$WORKPAY)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$WORKPAY)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$WORKPAY))
# > [1] 2998
sum(is.na(temp_data$WORKPAY))
# > [1] 0
table(pisa_2022_canada_merged$WORKPAY, useNA="always")
# >     0     1     2     3     4     5     6     7     8     9    10    99  <NA> 
# > 11157  1021  1777  1319  1287   814   915   252   542   140   851  2998     0 
#prop.table(table(pisa_2022_canada_merged$WORKPAY)) 
round(prop.table(table(pisa_2022_canada_merged$WORKPAY)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 0.48 0.04 0.08 0.06 0.06 0.04 0.04 0.01 0.02 0.01 0.04 0.13 
table(temp_data$WORKPAY, useNA = "always") 
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 4024  394  664  484  443  231  256   68  164   35  181    0 
#prop.table(table(temp_data$WORKPAY)) 
round(prop.table(table(temp_data$WORKPAY)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 0.58 0.06 0.10 0.07 0.06 0.03 0.04 0.01 0.02 0.01 0.03 
#c(summary(pisa_2022_student_canada$WORKPAY), SD = sd(pisa_2022_student_canada$WORKPAY))
c(summary(temp_data$WORKPAY), SD = sd(temp_data$WORKPAY))
# >     Min.   1st Qu.    Median      Mean   3rd Qu.      Max.        SD 
# > 0.000000  0.000000  0.000000  1.663306  3.000000 10.000000  2.546890 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$WORKPAY, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$WORKPAY,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$WORKPAY, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$WORKPAY,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$WORKPAY, useNA="always")),
           row.names=NULL)
# >    code                                        label     n
# > 1     0                              No work for pay 11157
# > 2     1           1 time of working for pay per week  1021
# > 3     2          2 times of working for pay per week  1777
# > 4     3          3 times of working for pay per week  1319
# > 5     4          4 times of working for pay per week  1287
# > 6     5          5 times of working for pay per week   814
# > 7     6          6 times of working for pay per week   915
# > 8     7          7 times of working for pay per week   252
# > 9     8          8 times of working for pay per week   542
# > 10    9          9 times of working for pay per week   140
# > 11   10 10 or more times of working for pay per week   851
# > 12   99                                  No Response  2998
# > 13   NA                                         <NA>     0
data.frame(code=as.numeric(names(table(temp_data$WORKPAY, useNA="always"))),
           label=names(attr(temp_data$WORKPAY,"labels"))[
             match(as.numeric(names(table(temp_data$WORKPAY, useNA="always"))),
                   unname(attr(temp_data$WORKPAY,"labels")))],
           n=as.integer(table(temp_data$WORKPAY, useNA="always")),
           row.names=NULL) 
# >    code                                        label    n
# > 1     0                              No work for pay 4024
# > 2     1           1 time of working for pay per week  394
# > 3     2          2 times of working for pay per week  664
# > 4     3          3 times of working for pay per week  484
# > 5     4          4 times of working for pay per week  443
# > 6     5          5 times of working for pay per week  231
# > 7     6          6 times of working for pay per week  256
# > 8     7          7 times of working for pay per week   68
# > 9     8          8 times of working for pay per week  164
# > 10    9          9 times of working for pay per week   35
# > 11   10 10 or more times of working for pay per week  181
# > 12   NA                                         <NA>    0

hist(pisa_2022_canada_merged$WORKPAY, breaks=11)
barplot(table(pisa_2022_canada_merged$WORKPAY))
boxplot(pisa_2022_canada_merged$WORKPAY)

hist(temp_data$WORKPAY, breaks=11)
barplot(table(temp_data$WORKPAY))
boxplot(temp_data$WORKPAY)

######## ---- WORKHOME: Working in household/take care of family members before or after school ----
class(pisa_2022_canada_merged$WORKHOME)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$WORKHOME)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$WORKHOME))
# > [1] 2944
sum(is.na(temp_data$WORKHOME))
# > [1] 0
table(pisa_2022_canada_merged$WORKHOME, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA>  
# > 3297 1298 2119 1748 1880 2256 1589  885 1165  556 3336 2944    0 
#prop.table(table(pisa_2022_canada_merged$WORKHOME)) 
round(prop.table(table(pisa_2022_canada_merged$WORKHOME)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10   99 
# > 0.14 0.06 0.09 0.08 0.08 0.10 0.07 0.04 0.05 0.02 0.14 0.13 
table(temp_data$WORKHOME, useNA = "always") 
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 1095  503  771  659  677  792  517  302  386  198 1044    0 
#prop.table(table(temp_data$WORKHOME)) 
round(prop.table(table(temp_data$WORKHOME)), 2) 
# >    0    1    2    3    4    5    6    7    8    9   10 
# > 0.16 0.07 0.11 0.09 0.10 0.11 0.07 0.04 0.06 0.03 0.15 
#c(summary(pisa_2022_student_canada$WORKHOME), SD = sd(pisa_2022_student_canada$WORKHOME))
c(summary(temp_data$WORKHOME), SD = sd(temp_data$WORKHOME))
# >     Min.   1st Qu.    Median      Mean   3rd Qu.      Max.        SD 
# > 0.000000  2.000000  4.000000  4.495392  7.000000 10.000000  3.359789 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$WORKHOME, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$WORKHOME,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$WORKHOME, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$WORKHOME,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$WORKHOME, useNA="always")),
           row.names=NULL)
# >    code                                                                          label    n
# > 1     0                                 No work in household or care of family members 3297
# > 2     1           1 time of working in household or caring for family members per week 1298
# > 3     2          2 times of working in household or caring for family members per week 2119
# > 4     3          3 times of working in household or caring for family members per week 1748
# > 5     4          4 times of working in household or caring for family members per week 1880
# > 6     5          5 times of working in household or caring for family members per week 2256
# > 7     6          6 times of working in household or caring for family members per week 1589
# > 8     7          7 times of working in household or caring for family members per week  885
# > 9     8          8 times of working in household or caring for family members per week 1165
# > 10    9          9 times of working in household or caring for family members per week  556
# > 11   10 10 or more times of working in household or caring for family members per week 3336
# > 12   99                                                                    No Response 2944
# > 13   NA                                                                           <NA>    0
data.frame(code=as.numeric(names(table(temp_data$WORKHOME, useNA="always"))),
           label=names(attr(temp_data$WORKHOME,"labels"))[
             match(as.numeric(names(table(temp_data$WORKHOME, useNA="always"))),
                   unname(attr(temp_data$WORKHOME,"labels")))],
           n=as.integer(table(temp_data$WORKHOME, useNA="always")),
           row.names=NULL) 
# >    code                                                                          label    n
# > 1     0                                 No work in household or care of family members 1095
# > 2     1           1 time of working in household or caring for family members per week  503
# > 3     2          2 times of working in household or caring for family members per week  771
# > 4     3          3 times of working in household or caring for family members per week  659
# > 5     4          4 times of working in household or caring for family members per week  677
# > 6     5          5 times of working in household or caring for family members per week  792
# > 7     6          6 times of working in household or caring for family members per week  517
# > 8     7          7 times of working in household or caring for family members per week  302
# > 9     8          8 times of working in household or caring for family members per week  386
# > 10    9          9 times of working in household or caring for family members per week  198
# > 11   10 10 or more times of working in household or caring for family members per week 1044
# > 12   NA                                                                           <NA>    0

hist(pisa_2022_canada_merged$WORKHOME, breaks=11)
barplot(table(pisa_2022_canada_merged$WORKHOME))
boxplot(pisa_2022_canada_merged$WORKHOME)

hist(temp_data$WORKHOME, breaks=11)
barplot(table(temp_data$WORKHOME))
boxplot(temp_data$WORKHOME)

###### ---- Derived variables based on IRT scaling ----

####### ---- Economic, social and cultural status (Module 2) ----

######## ---- HOMEPOS: Home possessions (Components of ESCS) ----
class(pisa_2022_canada_merged$HOMEPOS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$HOMEPOS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$HOMEPOS))
# > [1] 1402
sum(is.na(temp_data$HOMEPOS))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$HOMEPOS), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -6.8558 -0.2046  0.3659  0.3483  0.8954  8.8191    1402 
#summary(pisa_2022_canada_merged$HOMEPOS[!is.na(pisa_2022_canada_merged$HOMEPOS)])
summary(temp_data$HOMEPOS) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -6.58560 -0.09835  0.44125  0.43199  0.94222  4.34680 

######## ---- ICTRES: ICT resources (at home) ----
class(pisa_2022_canada_merged$ICTRES)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$ICTRES)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$ICTRES))
# > [1] 1424
sum(is.na(temp_data$ICTRES))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$ICTRES), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -5.6656 -0.2543  0.3628  0.3589  0.9494  5.2480    1424 
#summary(pisa_2022_canada_merged$ICTRES[!is.na(pisa_2022_canada_merged$ICTRES)])
summary(temp_data$ICTRES) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -5.0283 -0.1970  0.4001  0.4157  0.9490  5.2480 

####### ---- Educational pathways and post-secondary aspirations (Module 3) ----

######## ---- INFOSEEK: Information seeking regarding future career ----
class(pisa_2022_canada_merged$INFOSEEK)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$INFOSEEK)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$INFOSEEK))
# > [1] 5720
sum(is.na(temp_data$INFOSEEK))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$INFOSEEK), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.4211 -0.5771  0.0339 -0.0741  0.4978  2.6141    5720
#summary(pisa_2022_canada_merged$INFOSEEK[!is.na(pisa_2022_canada_merged$INFOSEEK)])
summary(temp_data$INFOSEEK) 
# >      Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -2.42110 -0.50310  0.04405 -0.06100  0.46105  2.61410 

####### ---- School culture and climate (Module 6) ----

######## ---- BULLIED: Being bullied ----
class(pisa_2022_canada_merged$BULLIED)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$BULLIED)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$BULLIED))
# > [1] 2973
sum(is.na(temp_data$BULLIED))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$BULLIED), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -1.2280 -1.2280 -0.2670 -0.2077  0.6132  4.6939    2973
#summary(pisa_2022_canada_merged$BULLIED[!is.na(pisa_2022_canada_merged$BULLIED)])
summary(temp_data$BULLIED) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -1.2280 -1.2280 -0.3264 -0.2669  0.5504  4.6939

######## ---- FEELSAFE: Feeling safe ----
class(pisa_2022_canada_merged$FEELSAFE)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$FEELSAFE)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$FEELSAFE))
# > [1] 2865
sum(is.na(temp_data$FEELSAFE))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$FEELSAFE), na.rm = TRUE)
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.     NA's 
# > -2.78860 -0.75600  0.28520  0.07853  1.12460  1.12460     2865 
#summary(pisa_2022_canada_merged$FEELSAFE[!is.na(pisa_2022_canada_merged$FEELSAFE)])
summary(temp_data$FEELSAFE) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# > -2.7886 -0.7560  0.3308  0.1037  1.1246  1.1246 

######## ---- BELONG: Sense of belonging ----
class(pisa_2022_canada_merged$BELONG)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$BELONG)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$BELONG))
# > [1] 2942
sum(is.na(temp_data$BELONG))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$BELONG), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.2877 -0.7630 -0.3261 -0.1658  0.2315  2.7849    2942
#summary(pisa_2022_canada_merged$BELONG[!is.na(pisa_2022_canada_merged$BELONG)])
summary(temp_data$BELONG) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -3.2583 -0.7334 -0.3261 -0.1610  0.2003  2.7562

####### ---- Subject-specific beliefs, attitudes, feelings, and behaviours (Module 7) ----

######## ---- GROSAGR: Growth mindset ----
class(pisa_2022_canada_merged$GROSAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$GROSAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$GROSAGR))
# > [1] 4067
sum(is.na(temp_data$GROSAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$GROSAGR), na.rm = TRUE)
# >     Min. 1st Qu. Median    Mean 3rd Qu.    Max.    NA's 
# > -3.6086 -0.3061 -0.0144  0.0214  0.6926  3.3724    4067 
#summary(pisa_2022_canada_merged$GROSAGR[!is.na(pisa_2022_canada_merged$GROSAGR)])
summary(temp_data$GROSAGR) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -3.59940 -0.30610 -0.01440  0.07811  0.77980  2.02940

######## ---- ANXMAT: Mathematics anxiety ----
class(pisa_2022_canada_merged$ANXMAT)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$ANXMAT)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$ANXMAT))
# > [1] 4239
sum(is.na(temp_data$ANXMAT))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$ANXMAT), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.3945 -0.5006  0.1748  0.1346  0.7889  2.6350    4239 
#summary(pisa_2022_canada_merged$ANXMAT[!is.na(pisa_2022_canada_merged$ANXMAT)])
summary(temp_data$ANXMAT) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -2.39450 -0.50060  0.10310  0.08833  0.77740  2.63500 

######## ---- MATHEFF: Mathematics self-efficacy: Formal and applied mathematics ----
class(pisa_2022_canada_merged$MATHEFF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$MATHEFF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$MATHEFF))
# > [1] 4660
sum(is.na(temp_data$MATHEFF))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$MATHEFF), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.5080 -0.7670 -0.2055 -0.1300  0.4622  2.3556    4660  
#summary(pisa_2022_canada_merged$MATHEFF[!is.na(pisa_2022_canada_merged$MATHEFF)])
summary(temp_data$MATHEFF) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -3.49350 -0.69210 -0.17850 -0.06096  0.52770  2.35560 

######## ---- MATHEF21: Mathematics self-efficacy: Mathematical reasoning and 21st century mathematics ----
class(pisa_2022_canada_merged$MATHEF21)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$MATHEF21)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$MATHEF21))
# > [1] 5014
sum(is.na(temp_data$MATHEF21))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$MATHEF21), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.4882 -0.3174  0.3229  0.2936  0.7528  2.9137    5014 
#summary(pisa_2022_canada_merged$MATHEF21[!is.na(pisa_2022_canada_merged$MATHEF21)])
summary(temp_data$MATHEF21) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -2.4607 -0.2636  0.3240  0.3051  0.7508  2.7911 

######## ---- MATHPERS: Proactive mathematics study behaviour ----
class(pisa_2022_canada_merged$MATHPERS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$MATHPERS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$MATHPERS))
# > [1] 4891
sum(is.na(temp_data$MATHPERS))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$MATHPERS), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.0955 -0.3964  0.1543  0.2386  0.7320  2.8491    4891 
#summary(pisa_2022_canada_merged$MATHPERS[!is.na(pisa_2022_canada_merged$MATHPERS)])
summary(temp_data$MATHPERS) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -3.0955 -0.3028  0.1842  0.2788  0.7250  2.8390 

######## ---- FAMCON: Subjective familiarity with mathematics concepts ----
class(pisa_2022_canada_merged$FAMCON)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$FAMCON)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$FAMCON))
# > [1] 4856
sum(is.na(temp_data$FAMCON))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$FAMCON), na.rm = TRUE)
# >     Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.9827 -0.0144  0.6709  0.8891  1.5232  4.8382    4856
#summary(pisa_2022_canada_merged$FAMCON[!is.na(pisa_2022_canada_merged$FAMCON)])
summary(temp_data$FAMCON) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -3.98270  0.04718  0.73545  0.95945  1.57258  4.83820

####### ---- General social and emotional characteristics (Module 8) ----

######## ---- ASSERAGR: Assertiveness ----
class(pisa_2022_canada_merged$ASSERAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$ASSERAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$ASSERAGR))
# > [1] 4300
sum(is.na(temp_data$ASSERAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$ASSERAGR), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -8.2332 -0.3593 -0.0014  0.0726  0.4792  7.2577    4300 
#summary(pisa_2022_canada_merged$ASSERAGR[!is.na(pisa_2022_canada_merged$ASSERAGR)])
summary(temp_data$ASSERAGR) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -8.22560 -0.36200  0.03750  0.09214  0.49875  7.25770 

######## ---- COOPAGR: Cooperation ----
class(pisa_2022_canada_merged$COOPAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$COOPAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$COOPAGR))
# > [1] 3923
sum(is.na(temp_data$COOPAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$COOPAGR), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -6.6331 -0.7191 -0.2193 -0.0811  0.3299  6.1327    3923
#summary(pisa_2022_canada_merged$COOPAGR[!is.na(pisa_2022_canada_merged$COOPAGR)])
summary(temp_data$COOPAGR) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -6.45170 -0.66765 -0.16695 -0.05574  0.35250  6.12650 

######## ---- CURIOAGR: Curiosity ----
class(pisa_2022_canada_merged$CURIOAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double"   
class(temp_data$CURIOAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double"   
sum(is.na(pisa_2022_canada_merged$CURIOAGR))
# > [1] 3997
sum(is.na(temp_data$CURIOAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$CURIOAGR), na.rm = TRUE)
# >     Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -4.9554 -0.6795 -0.1533  0.0090  0.4356  4.2195    3997 
#summary(pisa_2022_canada_merged$CURIOAGR[!is.na(pisa_2022_canada_merged$CURIOAGR)])
summary(temp_data$CURIOAGR) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -4.89130 -0.61365 -0.07505  0.07741  0.49205  4.09130 

######## ---- EMOCOAGR: Emotional control ----
class(pisa_2022_canada_merged$EMOCOAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$EMOCOAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$EMOCOAGR))
# > [1] 4236
sum(is.na(temp_data$EMOCOAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$EMOCOAGR), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -5.1892 -0.4265  0.0363  0.0537  0.5521  5.5872    4236
#summary(pisa_2022_canada_merged$EMOCOAGR[!is.na(pisa_2022_canada_merged$EMOCOAGR)])
summary(temp_data$EMOCOAGR) 
# >     Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# > -5.1892 -0.4223  0.0489  0.0734  0.5902  5.5872

######## ---- EMPATAGR: Empathy ----
class(pisa_2022_canada_merged$EMPATAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$EMPATAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$EMPATAGR))
# > [1] 4055
sum(is.na(temp_data$EMPATAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$EMPATAGR), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -6.4177 -0.6781 -0.1503 -0.0050  0.4162  4.6897    4055 
#summary(pisa_2022_canada_merged$EMPATAGR[!is.na(pisa_2022_canada_merged$EMPATAGR)])
summary(temp_data$EMPATAGR) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -6.07510 -0.60865 -0.10140  0.03607  0.45170  4.68970

######## ---- PERSEVAGR: Perseverance ----
class(pisa_2022_canada_merged$PERSEVAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$PERSEVAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$PERSEVAGR))
# > [1] 3809
sum(is.na(temp_data$PERSEVAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$PERSEVAGR), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -5.9854 -0.6467 -0.1816 -0.0179  0.3920  4.8909    3809
#summary(pisa_2022_canada_merged$PERSEVAGR[!is.na(pisa_2022_canada_merged$PERSEVAGR)])
summary(temp_data$PERSEVAGR) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -5.90880 -0.61745 -0.13920  0.02923  0.42433  4.77720

######## ---- STRESAGR: Stress resistance ----
class(pisa_2022_canada_merged$STRESAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$STRESAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$STRESAGR))
# > [1] 4294
sum(is.na(temp_data$STRESAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$STRESAGR), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -5.2897 -0.5654 -0.0046 -0.0969  0.4175  5.6241    4294 
#summary(pisa_2022_canada_merged$STRESAGR[!is.na(pisa_2022_canada_merged$STRESAGR)])
summary(temp_data$STRESAGR) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -5.2609 -0.5937 -0.0205 -0.1172  0.4282  5.6189 

####### ---- Exposure to mathematics content (Module 15) ----

######## ---- EXPOFA: Exposure to formal and applied mathematics tasks ----
class(pisa_2022_canada_merged$EXPOFA)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$EXPOFA)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$EXPOFA))
# > [1] 4803
sum(is.na(temp_data$EXPOFA))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$EXPOFA), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.0867 -0.3949  0.0772  0.0953  0.4850  2.6448    4803  
#summary(pisa_2022_canada_merged$EXPOFA[!is.na(pisa_2022_canada_merged$EXPOFA)])
summary(temp_data$EXPOFA) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -3.08670 -0.34802  0.08965  0.10857  0.47100  2.64480 

######## ---- EXPO21ST: Exposure to mathematical reasoning and 21st century mathematics tasks ----
class(pisa_2022_canada_merged$EXPO21ST)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$EXPO21ST)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$EXPO21ST))
# > [1] 5072
sum(is.na(temp_data$EXPO21ST))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$EXPO21ST), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.7453 -0.2905  0.2849  0.2587  0.7018  3.2705    5072  
#summary(pisa_2022_canada_merged$EXPO21ST[!is.na(pisa_2022_canada_merged$EXPO21ST)])
summary(temp_data$EXPO21ST) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -2.6364 -0.2465  0.2847  0.2751  0.7228  3.2705 

####### ---- Mathematics teacher behaviour (Module 16) ----

######## ---- COGACRCO: Cognitive activation in mathematics: Foster reasoning ----
class(pisa_2022_canada_merged$COGACRCO)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$COGACRCO)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$COGACRCO))
# > [1] 4654
sum(is.na(temp_data$COGACRCO))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$COGACRCO), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.9785 -0.3524  0.0441  0.1311  0.5504  3.7198    4654  
#summary(pisa_2022_canada_merged$COGACRCO[!is.na(pisa_2022_canada_merged$COGACRCO)])
summary(temp_data$COGACRCO) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -2.97850 -0.31898  0.09885  0.16397  0.56580  3.71980 

######## ---- COGACMCO: Cognitive activation in mathematics: Encourage mathematical thinking ----
class(pisa_2022_canada_merged$COGACMCO)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$COGACMCO)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$COGACMCO))
# > [1] 4878
sum(is.na(temp_data$COGACMCO))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$COGACMCO), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.1598 -0.3416  0.2170  0.2332  0.8118  2.6158    4878 
#summary(pisa_2022_canada_merged$COGACMCO[!is.na(pisa_2022_canada_merged$COGACMCO)])
summary(temp_data$COGACMCO) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -2.1598 -0.3378  0.2238  0.2404  0.7998  2.6158

######## ---- DISCLIM: Disciplinary climate in mathematics ----
class(pisa_2022_canada_merged$DISCLIM)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$DISCLIM)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$DISCLIM))
# > [1] 3534
sum(is.na(temp_data$DISCLIM))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$DISCLIM), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.4931 -0.6793 -0.1114 -0.0877  0.4571  1.8514    3534 
#summary(pisa_2022_canada_merged$DISCLIM[!is.na(pisa_2022_canada_merged$DISCLIM)])
summary(temp_data$DISCLIM) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -2.49310 -0.55320 -0.09030 -0.03604  0.47680  1.85140 

####### ---- Parental/guardian involvement and support (Module 19) ----

######## ---- FAMSUP: Family support ----
class(pisa_2022_canada_merged$FAMSUP)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$FAMSUP)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$FAMSUP))
# > [1] 5434
sum(is.na(temp_data$FAMSUP))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$FAMSUP), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.0631 -0.6511 -0.1242 -0.0067  0.5325  1.9583    5434 
#summary(pisa_2022_canada_merged$FAMSUP[!is.na(pisa_2022_canada_merged$FAMSUP)])
summary(temp_data$FAMSUP) 
# >      Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
# > -2.975100 -0.618650 -0.123350  0.006313  0.491150  1.958300

####### ---- Creative thinking (Module 20) ----

######## ---- CREATFAM: Creative peers and family environment  ----
class(pisa_2022_canada_merged$CREATFAM)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$CREATFAM)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$CREATFAM))
# > [1] 5195
sum(is.na(temp_data$CREATFAM))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$CREATFAM), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.7890 -0.5232 -0.0374  0.1395  0.7537  2.2394    5195  
#summary(pisa_2022_canada_merged$CREATFAM[!is.na(pisa_2022_canada_merged$CREATFAM)])
summary(temp_data$CREATFAM) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -2.7890 -0.4956 -0.0374  0.1507  0.6906  2.2394 

######## ---- CREATSCH: Creative school and class environment ----
class(pisa_2022_canada_merged$CREATSCH)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$CREATSCH)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$CREATSCH))
# > [1] 5197
sum(is.na(temp_data$CREATSCH))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$CREATSCH), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.6234 -0.4443  0.2737  0.2205  0.5352  2.8139    5197 
#summary(pisa_2022_canada_merged$CREATSCH[!is.na(pisa_2022_canada_merged$CREATSCH)])
summary(temp_data$CREATSCH) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -2.6234 -0.3960  0.2197  0.2264  0.5352  2.8139 

######## ---- CREATEFF: Creative thinking self-efficacy ----
class(pisa_2022_canada_merged$CREATEFF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$CREATEFF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$CREATEFF))
# > [1] 4949
sum(is.na(temp_data$CREATEFF))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$CREATEFF), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.9783 -0.4910  0.1309  0.1291  0.6159  2.5719    4949 
#summary(pisa_2022_canada_merged$CREATEFF[!is.na(pisa_2022_canada_merged$CREATEFF)])
summary(temp_data$CREATEFF) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -2.9210 -0.4737  0.0967  0.1206  0.5740  2.5641

######## ---- CREATOP: Creativity and openness to intellect ----
class(pisa_2022_canada_merged$CREATOP)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$CREATOP)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$CREATOP))
# > [1] 5108
sum(is.na(temp_data$CREATOP))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$CREATOP), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.9414 -0.5731  0.0287  0.0912  0.5122  2.9377    5108 
#summary(pisa_2022_canada_merged$CREATOP[!is.na(pisa_2022_canada_merged$CREATOP)])
summary(temp_data$CREATOP) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -2.9230 -0.5427  0.0246  0.1005  0.5268  2.9234 

######## ---- IMAGINE: Imagination and adventurousness ----
class(pisa_2022_canada_merged$IMAGINE)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$IMAGINE)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$IMAGINE))
# > [1] 5329
sum(is.na(temp_data$IMAGINE))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$IMAGINE), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.3982 -0.5572 -0.1308  0.0452  0.5749  2.5110    5329 
#summary(pisa_2022_canada_merged$IMAGINE[!is.na(pisa_2022_canada_merged$IMAGINE)])
summary(temp_data$IMAGINE) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -3.39820 -0.51070 -0.10750  0.08424  0.57930  2.51100

######## ---- OPENART: Openness to art and reflection ----
class(pisa_2022_canada_merged$OPENART)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$OPENART)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$OPENART))
# > [1] 5231
sum(is.na(temp_data$OPENART))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$OPENART), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.8504 -0.3483  0.1775  0.1662  0.6572  1.8963    5231 
#summary(pisa_2022_canada_merged$OPENART[!is.na(pisa_2022_canada_merged$OPENART)])
summary(temp_data$OPENART) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# > -2.8185 -0.3483  0.1775  0.1678  0.6258  1.8261 

######## ---- CREATAS: Participation in creative activities at school ----
class(pisa_2022_canada_merged$CREATAS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$CREATAS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$CREATAS))
# > [1] 5735
sum(is.na(temp_data$CREATAS))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$CREATAS), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -1.1205 -0.8544  0.0399  0.0515  0.4723  4.3998    5735
#summary(pisa_2022_canada_merged$CREATAS[!is.na(pisa_2022_canada_merged$CREATAS)])
summary(temp_data$CREATAS) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -1.12050 -0.69910  0.01570 -0.01847  0.39622  4.33120 

######## ---- CREATOOS: Participation in creative activities outside of school ----
class(pisa_2022_canada_merged$CREATOOS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$CREATOOS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$CREATOOS))
# > [1] 6673
sum(is.na(temp_data$CREATOOS))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$CREATOOS), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -0.8205 -0.8105 -0.3947 -0.0341  0.4752  4.6466    6673 
#summary(pisa_2022_canada_merged$CREATOOS[!is.na(pisa_2022_canada_merged$CREATOOS)])
summary(temp_data$CREATOOS) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# > -0.8205 -0.8105 -0.3947 -0.1122  0.4004  4.6466 

####### ---- Global crises (Module 21) ----

######## ---- FAMSUPSL: Family support for self-directed learning ----
class(pisa_2022_canada_merged$FAMSUPSL)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$FAMSUPSL)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$FAMSUPSL))
# > [1] 8525
sum(is.na(temp_data$FAMSUPSL))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$FAMSUPSL), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.3941 -0.7069 -0.0027 -0.0593  0.6049  2.4376    8525
#summary(pisa_2022_canada_merged$FAMSUPSL[!is.na(pisa_2022_canada_merged$FAMSUPSL)])
summary(temp_data$FAMSUPSL) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -2.38010 -0.71420 -0.01690 -0.08854  0.53707  2.43760

######## ---- FEELLAH: Feelings about learning at home  ----
class(pisa_2022_canada_merged$FEELLAH)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$FEELLAH)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$FEELLAH))
# > [1] 12471
sum(is.na(temp_data$FEELLAH))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$FEELLAH), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.6455 -0.6024 -0.0664 -0.0055  0.6049  3.1849   12471 
#summary(pisa_2022_canada_merged$FEELLAH[!is.na(pisa_2022_canada_merged$FEELLAH)])
summary(temp_data$FEELLAH) 
# >       Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# > -2.645500 -0.589900 -0.053700 -0.002558  0.597000  3.151700

######## ---- PROBSELF: Problems with self-directed learning ----
class(pisa_2022_canada_merged$PROBSELF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$PROBSELF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$PROBSELF))
# > [1] 8478
sum(is.na(temp_data$PROBSELF))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$PROBSELF), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.2306 -0.4959  0.2284  0.0820  0.7254  3.0976    8478 
#summary(pisa_2022_canada_merged$PROBSELF[!is.na(pisa_2022_canada_merged$PROBSELF)])
summary(temp_data$PROBSELF) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -2.23060 -0.51790  0.17840  0.03952  0.65145  3.09760

######## ---- SDLEFF: Self-directed learning self-efficacy ----
class(pisa_2022_canada_merged$SDLEFF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$SDLEFF)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$SDLEFF))
# > [1] 8684
sum(is.na(temp_data$SDLEFF))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$SDLEFF), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.5852 -0.6403  0.0506 -0.0159  0.3990  2.0817    8684 
#summary(pisa_2022_canada_merged$SDLEFF[!is.na(pisa_2022_canada_merged$SDLEFF)])
summary(temp_data$SDLEFF) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -2.58520 -0.59785  0.11170  0.01353  0.40800  2.08170 

######## ---- SCHSUST: School actions to sustain learning  ----
class(pisa_2022_canada_merged$SCHSUST)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$SCHSUST)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$SCHSUST))
# > [1] 7666
sum(is.na(temp_data$SCHSUST))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$SCHSUST), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.7592 -0.5174 -0.0080  0.0709  0.5636  2.7623    7666 
#summary(pisa_2022_canada_merged$SCHSUST[!is.na(pisa_2022_canada_merged$SCHSUST)])
summary(temp_data$SCHSUST) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -2.75920 -0.45010  0.03985  0.10958  0.58188  2.76230

####### ---- LEARRES: Types of learning resources used while school was closed ----
class(pisa_2022_canada_merged$LEARRES)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$LEARRES)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$LEARRES))
# > [1] 8460
sum(is.na(temp_data$LEARRES))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$LEARRES), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.3133 -0.4712 -0.0435 -0.0764  0.4162  3.0465    8460 
#summary(pisa_2022_canada_merged$LEARRES[!is.na(pisa_2022_canada_merged$LEARRES)])
summary(temp_data$LEARRES) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -3.31330 -0.45265 -0.04230 -0.08054  0.37728  3.04650 

###### ---- Complex composite index ----

######## ---- ESCS: Index of economic, social and cultural status ----
class(pisa_2022_canada_merged$ESCS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$ESCS)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$ESCS))
# > [1] 1677
sum(is.na(temp_data$ESCS))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$ESCS), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -5.7547 -0.2126  0.4427  0.3339  0.9317  3.9770    1677
#summary(pisa_2022_canada_merged$ESCS[!is.na(pisa_2022_canada_merged$ESCS)])
summary(temp_data$ESCS) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -5.5010 -0.1026  0.5313  0.4185  0.9886  2.7458

##### ---- School Questionnaire Derived Variables ----

###### ---- Simple questionnaire indices ---- 

####### ---- Out-of-school experiences (Module 10) ----

######## ---- MACTIV: Mathematics-related extra-curricular activities at school (ordinal/numeric) ----
class(pisa_2022_canada_merged$MACTIV)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$MACTIV)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
sum(is.na(pisa_2022_canada_merged$MACTIV))
# > [1] 2910
sum(is.na(temp_data$MACTIV))
# > [1] 0
table(pisa_2022_canada_merged$MACTIV, useNA="always")
# >    0    1    2    3    4    5 <NA> 
# > 2562 4033 4895 3857 2875 1941 2910
#table(temp_data$MACTIV, useNA="always")
table(temp_data$MACTIV, useNA = "always")    
# >   0    1    2    3    4    5 <NA> 
# > 801 1364 1722 1410  999  648    0 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$MACTIV, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$MACTIV,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$MACTIV, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$MACTIV,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$MACTIV, useNA="always")),
           row.names=NULL)
# >   code                                                      label    n
# > 1    0 No mathematics-related extra-curricular activities offered 2562
# > 2    1    1 mathematics-related extra-curricular activity offered 4033
# > 3    2  2 mathematics-related extra-curricular activities offered 4895
# > 4    3  3 mathematics-related extra-curricular activities offered 3857
# > 5    4  4 mathematics-related extra-curricular activities offered 2875
# > 6    5  5 mathematics-related extra-curricular activities offered 1941
# > 7   NA                                                       <NA> 2910
data.frame(code=as.numeric(names(table(temp_data$MACTIV, useNA="always"))),
           label=names(attr(temp_data$MACTIV,"labels"))[
             match(as.numeric(names(table(temp_data$MACTIV, useNA="always"))),
                   unname(attr(temp_data$MACTIV,"labels")))],
           n=as.integer(table(temp_data$MACTIV, useNA="always")),
           row.names=NULL)
# >   code                                                      label    n
# > 1    0 No mathematics-related extra-curricular activities offered  801
# > 2    1    1 mathematics-related extra-curricular activity offered 1364
# > 3    2  2 mathematics-related extra-curricular activities offered 1722
# > 4    3  3 mathematics-related extra-curricular activities offered 1410
# > 5    4  4 mathematics-related extra-curricular activities offered  999
# > 6    5  5 mathematics-related extra-curricular activities offered  648
# > 7   NA                                                       <NA>    0

####### ---- Organisation of student learning at school (Module 14) ----

######## ---- ABGMATH: Ability grouping for mathematics classes (ordinal/numeric) ----
class(pisa_2022_canada_merged$ABGMATH)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$ABGMATH)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
sum(is.na(pisa_2022_canada_merged$ABGMATH))
# > [1] 2922
sum(is.na(temp_data$ABGMATH))
# > [1] 0
table(pisa_2022_canada_merged$ABGMATH, useNA="always")
# >    1     2     3  <NA>
# > 2175 10946  7030  2922
#table(temp_data$ABGMATH, useNA="always")
table(temp_data$ABGMATH, useNA = "always")    
# >   1    2    3 <NA>
# > 769 3517 2469    0  
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$ABGMATH, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$ABGMATH,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$ABGMATH, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$ABGMATH,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$ABGMATH, useNA="always")),
           row.names=NULL)
# >   code                                                 label     n
# > 1    1                   No ability grouping for any classes  2175
# > 2    2 At least one form of ability grouping in some classes 10946
# > 3    3  At least one form of ability grouping in all classes  7030
# > 4   NA                                                  <NA>  2922
data.frame(code=as.numeric(names(table(temp_data$ABGMATH, useNA="always"))),
           label=names(attr(temp_data$ABGMATH,"labels"))[
             match(as.numeric(names(table(temp_data$ABGMATH, useNA="always"))),
                   unname(attr(temp_data$ABGMATH,"labels")))],
           n=as.integer(table(temp_data$ABGMATH, useNA="always")),
           row.names=NULL)
# >   code                                                 label    n
# > 1    1                   No ability grouping for any classes  769
# > 2    2 At least one form of ability grouping in some classes 3517
# > 3    3  At least one form of ability grouping in all classes 2469
# > 4   NA                                                  <NA>    0

##### ---- Teacher qualification, training, and professional development (Module 17) ---

######## ---- MTTRAIN: Mathematics teacher training ----
class(pisa_2022_canada_merged$MTTRAIN)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$MTTRAIN)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
sum(is.na(pisa_2022_canada_merged$MTTRAIN))
# > [1] 2575
sum(is.na(temp_data$MTTRAIN))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$MTTRAIN), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -1.6751 -0.0366  0.6955  0.3846  1.0982  1.0982    2575 
#summary(pisa_2022_canada_merged$FAMSUPSL[!is.na(pisa_2022_canada_merged$MTTRAIN)])
summary(temp_data$MTTRAIN) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# > -1.6751 -0.0622  0.6955  0.3537  1.0982  1.0982

####### ---- Creative thinking (Module 20) ----

######## ---- CREENVSC: Creative school environment ----
class(pisa_2022_canada_merged$CREENVSC)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$CREENVSC)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
sum(is.na(pisa_2022_canada_merged$CREENVSC))
# > [1] 3036
sum(is.na(temp_data$CREENVSC))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$CREENVSC), na.rm = TRUE)
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.     NA's 
# > -3.22370 -0.86710 -0.26860 -0.09676  0.43980  2.16310     3036
#summary(pisa_2022_canada_merged$FAMSUPSL[!is.na(pisa_2022_canada_merged$CREENVSC)])
summary(temp_data$CREENVSC) 
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -3.2237 -0.7641 -0.2686 -0.0927  0.4118  2.1631

######## ---- OPENCUL: Openness culture/climate  ----
class(pisa_2022_canada_merged$OPENCUL)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$OPENCUL)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
sum(is.na(pisa_2022_canada_merged$OPENCUL))
# > [1] 3096
sum(is.na(temp_data$OPENCUL))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$OPENCUL), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -2.0055 -0.3286  0.4217  0.2691  0.4769  3.3059    3096
#summary(pisa_2022_canada_merged$FAMSUPSL[!is.na(pisa_2022_canada_merged$OPENCUL)])
summary(temp_data$OPENCUL) 
# >     Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# > -2.0055 -0.3269  0.4217  0.2911  0.5408  3.3059

####### ---- Global crises (Module 21) ----

######## ---- DIGPREP: Preparedness for Digital Learning (WLE) (numeric)----

class(pisa_2022_canada_merged$DIGPREP)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$DIGPREP)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
sum(is.na(pisa_2022_canada_merged$DIGPREP))
# > [1] 3008
sum(is.na(temp_data$DIGPREP))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$DIGPREP), na.rm = TRUE)
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.1194 -0.6898 -0.1611 -0.1010  0.2816  2.9610    3008 
#summary(pisa_2022_canada_merged$FAMSUPSL[!is.na(pisa_2022_canada_merged$DIGPREP)])
summary(temp_data$DIGPREP) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -3.11940 -0.67490 -0.16110 -0.08791  0.31050  2.96100 

#### ---- 2) Categorical Variables ----
##### ---- Student Questionnaire Variables ----

######## ---- REGION: REGION ----
class(pisa_2022_canada_merged$REGION)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$REGION)
# > [1] "factor"
sum(is.na(pisa_2022_canada_merged$REGION))
# > [1] 0
sum(is.na(temp_data$REGION))
# > [1] 0
table(pisa_2022_canada_merged$REGION, useNA="always")
# > 12401 12402 12403 12404 12405 12406 12407 12408 12409 12410  <NA> 
# >  1053   357  1590  1653  4137  5918  2629  2276  1330  2130     0 
#table(temp_data$REGION, useNA="always")
table(droplevels(temp_data$REGION), useNA = "always")    # drop unused factor levels (cleaner tables)
# > Canada: Newfoundland and Labrador      Canada: Prince Edward Island               Canada: Nova Scotia             Canada: New Brunswick                    Canada: Quebec 
# >                               357                                91                               380                               572                              1179 
# >                   Canada: Ontario                  Canada: Manitoba              Canada: Saskatchewan                   Canada: Alberta          Canada: British Columbia 
# >                              1596                               782                               687                               430                               681 
# >                              <NA> 
# >                                 0 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$REGION, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$REGION,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$REGION, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$REGION,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$REGION, useNA="always")),
           row.names=NULL)
# >     code                             label    n
# > 1  12401 Canada: Newfoundland and Labrador 1053
# > 2  12402      Canada: Prince Edward Island  357
# > 3  12403               Canada: Nova Scotia 1590
# > 4  12404             Canada: New Brunswick 1653
# > 5  12405                    Canada: Quebec 4137
# > 6  12406                   Canada: Ontario 5918
# > 7  12407                  Canada: Manitoba 2629
# > 8  12408              Canada: Saskatchewan 2276
# > 9  12409                   Canada: Alberta 1330
# > 10 12410          Canada: British Columbia 2130
# > 11    NA                              <NA>    0
data.frame(
  code  = unname(attr(pisa_2022_canada_merged$REGION, "labels"))[
    match(
      names(table(droplevels(temp_data$REGION), useNA = "no")),
      names(attr(pisa_2022_canada_merged$REGION, "labels"))
    )
  ],
  label = names(table(droplevels(temp_data$REGION), useNA = "no")),
  n     = as.integer(table(droplevels(temp_data$REGION), useNA = "no")),
  row.names = NULL
)
# >     code                             label    n
# > 1  12401 Canada: Newfoundland and Labrador  357
# > 2  12402      Canada: Prince Edward Island   91
# > 3  12403               Canada: Nova Scotia  380
# > 4  12404             Canada: New Brunswick  572
# > 5  12405                    Canada: Quebec 1179
# > 6  12406                   Canada: Ontario 1596
# > 7  12407                  Canada: Manitoba  782
# > 8  12408              Canada: Saskatchewan  687
# > 9  12409                   Canada: Alberta  430
# > 10 12410          Canada: British Columbia  681

##### ---- Student Questionnaire Variables: Categorical ----

###### ---- Simple questionnaire indices ----

####### ---- Migration and language exposure (Module 4) ----

######## ---- ST004D01T: Student (Standardized) Gender ----
class(pisa_2022_canada_merged$ST004D01T)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$ST004D01T)
# > [1] "factor"
sum(is.na(pisa_2022_canada_merged$ST004D01T))
# > [1] 42
sum(is.na(temp_data$ST004D01T))
# > [1] 0
table(pisa_2022_canada_merged$ST004D01T, useNA="always")
# >     1     2     7  <NA> 
# > 11377 11654    42     0 
#table(temp_data$ST004D01T, useNA="always")
table(droplevels(temp_data$ST004D01T), useNA = "always") 
# > Female           Male Not Applicable           <NA> 
# >   3565           3173             17              0 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$ST004D01T, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$ST004D01T,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$ST004D01T, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$ST004D01T,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$ST004D01T, useNA="always")),
           row.names=NULL)
# >   code          label     n
# > 1    1         Female 11377
# > 2    2           Male 11654
# > 3    7 Not Applicable    42
# > 4   NA           <NA>     0
data.frame(
  code  = unname(attr(pisa_2022_canada_merged$ST004D01T, "labels"))[
    match(
      names(table(droplevels(temp_data$ST004D01T), useNA = "no")),
      names(attr(pisa_2022_canada_merged$ST004D01T, "labels"))
    )
  ],
  label = names(table(droplevels(temp_data$ST004D01T), useNA = "no")),
  n     = as.integer(table(droplevels(temp_data$ST004D01T), useNA = "no")),
  row.names = NULL
)
# >   code          label    n
# > 1    1         Female 3565
# > 2    2           Male 3173
# > 3    7 Not Applicable   17

####### ---- Migration and language exposure (Module 4) ----

######## ---- IMMIG: Index on immigrant background (OECD definition ----
class(pisa_2022_canada_merged$IMMIG)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$IMMIG)
# > [1] "factor"
sum(is.na(pisa_2022_canada_merged$IMMIG))
# > [1] 2610
sum(is.na(temp_data$IMMIG))
# > [1] 0
table(pisa_2022_canada_merged$IMMIG, useNA="always")
# >     1     2     3     9  <NA> 
# > 15053  2565  2845  2610     0 
table(droplevels(temp_data$IMMIG), useNA = "always") 
# > Native student Second-Generation student  First-Generation student                      <NA> 
# >           5013                       866                       876                         0 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$IMMIG, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$IMMIG,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$IMMIG, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$IMMIG,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$IMMIG, useNA="always")),
           row.names=NULL)
# >  code                     label     n
# >1    1            Native student 15053
# >2    2 Second-Generation student  2565
# >3    3  First-Generation student  2845
# >4    9               No Response  2610
# >5   NA                      <NA>     0
data.frame(
  code  = unname(attr(pisa_2022_canada_merged$IMMIG, "labels"))[
    match(
      names(table(droplevels(temp_data$IMMIG), useNA = "no")),
      names(attr(pisa_2022_canada_merged$IMMIG, "labels"))
    )
  ],
  label = names(table(droplevels(temp_data$IMMIG), useNA = "no")),
  n     = as.integer(table(droplevels(temp_data$IMMIG), useNA = "no")),
  row.names = NULL
)
# >  code                     label    n
# >1    1            Native student 5013
# >2    2 Second-Generation student  866
# >3    3  First-Generation student  876

######## ---- LANGN: Language spoken at home ----
class(pisa_2022_canada_merged$LANGN)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$LANGN)
# > [1] "factor"
sum(is.na(pisa_2022_canada_merged$LANGN))
# > [1] 0
sum(is.na(temp_data$LANGN))
# > [1] 0
table(pisa_2022_canada_merged$LANGN, useNA="always")
# >   313   493   807   999  <NA> 
# > 14818  3553  3094  1608     0 
#table(temp_data$LANGN, useNA="always")
table(droplevels(temp_data$LANGN), useNA = "always")    # drop unused factor levels (cleaner tables)
# > English                 French Another language (CAN)                   <NA> 
# >   4755                   1089                    911                      0 
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$LANGN, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$LANGN,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$LANGN, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$LANGN,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$LANGN, useNA="always")),
           row.names=NULL)
# >   code                  label     n
# > 1  313                English 14818
# > 2  493                 French  3553
# > 3  807 Another language (CAN)  3094
# > 4  999                Missing  1608
# > 5   NA                   <NA>     0
data.frame(
  code  = unname(attr(pisa_2022_canada_merged$LANGN, "labels"))[
    match(
      names(table(droplevels(temp_data$LANGN), useNA = "no")),
      names(attr(pisa_2022_canada_merged$LANGN, "labels"))
    )
  ],
  label = names(table(droplevels(temp_data$LANGN), useNA = "no")),
  n     = as.integer(table(droplevels(temp_data$LANGN), useNA = "no")),
  row.names = NULL
)
# >   code                  label    n
# > 1  313                English 4755
# > 2  493                 French 1089
# > 3  807 Another language (CAN)  911

##### ---- School Questionnaire Variables: Categorical ----

###### ---- Simple questionnaire indices ----

####### ---- School type and infrastructure (Module 11) ----

######## ---- SCHLTYPE: School type ----
class(pisa_2022_canada_merged$SCHLTYPE)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 
class(temp_data$SCHLTYPE)
# > [1] "factor"
sum(is.na(pisa_2022_canada_merged$SCHLTYPE))
# > [1] 0
sum(is.na(temp_data$SCHLTYPE))
# > [1] 0
table(pisa_2022_canada_merged$SCHLTYPE, useNA="always")
# >   1     2     3  <NA> 
# > 980   566 21527     0
#table(temp_data$SCHLTYPE, useNA="always")
table(droplevels(temp_data$SCHLTYPE), useNA = "always")    # drop unused factor levels (cleaner tables)
# > Private independent Private Government-dependent                       Public                         <NA> 
# >                 373                          232                         6150                            0
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$SCHLTYPE, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$SCHLTYPE,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$SCHLTYPE, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$SCHLTYPE,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$SCHLTYPE, useNA="always")),
           row.names=NULL)
# >   code                        label     n
# > 1    1          Private independent   980
# > 2    2 Private Government-dependent   566
# > 3    3                       Public 21527
# > 4   NA                         <NA>     0
data.frame(
  code  = unname(attr(pisa_2022_canada_merged$SCHLTYPE, "labels"))[
    match(
      names(table(droplevels(temp_data$SCHLTYPE), useNA = "no")),
      names(attr(pisa_2022_canada_merged$SCHLTYPE, "labels"))
    )
  ],
  label = names(table(droplevels(temp_data$SCHLTYPE), useNA = "no")),
  n     = as.integer(table(droplevels(temp_data$SCHLTYPE), useNA = "no")),
  row.names = NULL
)
# >   code                        label    n
# > 1    1          Private independent  373
# > 2    2 Private Government-dependent  232
# > 3    3                       Public 6150
## ---- I.i Correlation ----

# For numeric variables only.
# Setup is different from other parts in this R script. 

## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)      # For reading SPSS .sav files
library(tidyverse)  # Includes dplyr, tidyr, purrr, ggplot2, tibble, etc.
library(broom)      # For tidying model output
library(intsvy)     # For analyze PISA data
library(tictoc)       # For timing code execution

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("haven", "tidyverse", "broom","intsvy", "tictoc"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         haven         tidyverse             broom            intsvy            tictoc 
# > "haven 2.5.4" "tidyverse 2.0.0"     "broom 1.0.8"      "intsvy 2.9"    "tictoc 1.2.1"

# Load data
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE) # Preserve SPSS's user-defined missing values
dim(pisa_2022_canada_merged)   
# > [1] 23073  1699

# Load metadata + missing summary: full variables
metadata_missing_student <- read_csv("data/pisa2022/metadata_missing_student.csv", show_col_types = FALSE)
metadata_missing_school <- read_csv("data/pisa2022/metadata_missing_school.csv", show_col_types = FALSE)

# Load metadata + missing summary: student/school questionnaire derived variables
stuq_dvs_metadata_missing_student <- readr::read_csv("data/pisa2022/stuq_dvs_metadata_missing_student.csv", show_col_types = FALSE)
schq_dvs_metadata_missing_school <- read_csv("data/pisa2022/schq_dvs_metadata_missing_school.csv",  show_col_types = FALSE)

# Constants
M <- 10                         # Number of plausible values
G <- 80                         # Number of BRR replicate weights
k <- 0.5                        # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)          # 95% z-critical value for confidence interval (CI)

# Target varaible
pvmaths  <- paste0("PV", 1:M, "MATH")   # PV1MATH to PV10MATH

# Predictors (VOI = variables of interest): numeric (continuous + ordinal) + categorical (nominal)
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

length(rep_wts); length(final_wt)
# > [1] 80
# > [1] 1

#### ---- Prepare data ----

## Subset 
temp_data <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                    # Student and school IDs for sample size checks (OECD, 2009)
         all_of(final_wt), all_of(rep_wts),     # Final student weight + 80 BRR replicate weights
         all_of(pvmaths), all_of(voi_num))      # Target and predictor variables

dim(temp_data)
# > [1] 23073   146

# Quick check of missingness
sapply(temp_data[, voi_num], function(x) sum(is.na(x)))
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >    4150      4168      4123      2912      2887      2998      2944      1402      1424      5720      2973      2865      2942      4067      4239      4660      5014      4891 
# >  FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP  CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >    4856      4300      3923      3997      4236      4055      3809      4294      4803      5072      4654      4878      3534      5434      5195      5197      4949      5108 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >    5329      5231      5735      6673      8525     12471      8478      8684      7666      8460      1677      2910      2922      2575      3036      3096      3008 
stopifnot(
  sum(is.na(temp_data[, pvmaths])) == 0,
  sum(is.na(temp_data$W_FSTUWT)) == 0,
  colSums(is.na(temp_data[paste0("W_FSTURWT", 1:80)])) == 0
)

## Filter 
temp_data <- temp_data %>% filter(if_all(all_of(voi_num), ~ !is.na(.)))  # Listwise deletion: retain only complete cases on predictors
dim(temp_data) 
# > [1] 6944  146

# Quick check of missingness
sapply(temp_data[, voi_num], function(x) sum(is.na(x)))
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 
# >  FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP  CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 
stopifnot(
  sum(is.na(temp_data[, pvmaths])) == 0,
  sum(is.na(temp_data$W_FSTUWT)) == 0,
  colSums(is.na(temp_data[paste0("W_FSTURWT", 1:80)])) == 0
)

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
  setNames(lapply(voi_num, function(v) rep(v, M)), voi_num)  
)
variable_names <- names(variables)

# Create n_vars x n_vars = 20 x 20 matrices for point estimates and standard errors
n_vars <- length(variable_names)
n_vars
# > [1] 54

point_estimate_matrix <- matrix(NA_real_, n_vars, n_vars, dimnames = list(variable_names, variable_names))
se_matrix             <- matrix(NA_real_, n_vars, n_vars, dimnames = list(variable_names, variable_names))

# Loop over variable pairs (only n_vars x n_vars)
tic("Loop over variable pairs")
for (i in 1:n_vars) {
  for (j in i:n_vars) {
    result <- compute_rubin_brr_corr(variables[[i]], variables[[j]])
    point_estimate_matrix[i, j] <- result["main_corr"]
    point_estimate_matrix[j, i] <- result["main_corr"]
    se_matrix[i, j] <- result["se"]
    se_matrix[j, i] <- result["se"]
  }
}
toc()
# > Loop over variable pairs: 538.58 sec elapsed

# Display results
cat("Point Estimates (ρ̂):\n")
print(round(point_estimate_matrix[1:6, 1:6], 4))
# >          PVmMATH MATHMOT MATHEASE MATHPREF EXERPRAC STUDYHMW
# > PVmMATH   1.0000  0.0382   0.1323   0.1154  -0.0998  -0.0174
# > MATHMOT   0.0382  1.0000   0.2608   0.2843  -0.0118  -0.0479
# > MATHEASE  0.1323  0.2608   1.0000   0.4646   0.0000  -0.0540
# > MATHPREF  0.1154  0.2843   0.4646   1.0000  -0.0028  -0.0340
# > EXERPRAC -0.0998 -0.0118   0.0000  -0.0028   1.0000   0.2521
# > STUDYHMW -0.0174 -0.0479  -0.0540  -0.0340   0.2521   1.0000
View(round(point_estimate_matrix, 2))

cat("\nStandard Errors (SE):\n")
print(round(se_matrix[1:6, 1:6], 4))
# >          PVmMATH MATHMOT MATHEASE MATHPREF EXERPRAC STUDYHMW
# > PVmMATH   0.0000  0.0149   0.0174   0.0176   0.0155   0.0184
# > MATHMOT   0.0149  0.0000   0.0230   0.0233   0.0155   0.0129
# > MATHEASE  0.0174  0.0230   0.0000   0.0173   0.0141   0.0162
# > MATHPREF  0.0176  0.0233   0.0173   0.0000   0.0145   0.0150 
# > EXERPRAC  0.0155  0.0155   0.0141   0.0145   0.0000   0.0177
# > STUDYHMW  0.0184  0.0129   0.0162   0.0150   0.0177   0.0000
View(round(se_matrix, 2))

# Heatmap
point_estimate_matrix %>%
  as.data.frame() %>%
  rownames_to_column("Var1") %>%
  pivot_longer(-Var1, names_to = "Var2", values_to = "rho") %>%
  ggplot(aes(x = Var2, y = Var1, fill = rho)) +
  geom_tile() +
  scale_fill_gradient2(limits = c(-1, 1), oob = scales::squish) +
  coord_fixed() +
  labs(title = "Weighted Correlation Heatmap (ρ̂)",
       x = NULL, y = NULL, fill = "ρ̂") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        panel.grid = element_blank())

# --- Top correlations with PVmMATH (signed, sorted) ---
tibble(
  Variable = setdiff(colnames(point_estimate_matrix), "PVmMATH"),
  rho      = point_estimate_matrix["PVmMATH", setdiff(colnames(point_estimate_matrix), "PVmMATH")]
) %>%
  arrange(desc(abs(rho))) %>%
  mutate(abs_rho = abs(rho)) %>%
  #slice_head(n = 20) %>%  # <- change n to get more/less than top 20 by |rho|
  as.data.frame() %>%
  transform(rho = round(rho, 2), abs_rho = round(abs_rho, 2)) %>%
  print(row.names = FALSE)
# >  Variable   rho abs_rho
# >   MATHEFF  0.53    0.53
# >  MATHEF21  0.40    0.40
# >    FAMCON  0.33    0.33
# >    ANXMAT -0.33    0.33
# >      ESCS  0.31    0.31
# >   HOMEPOS  0.24    0.24
# >  CURIOAGR  0.22    0.22
# >   WORKPAY -0.19    0.19
# >  MATHPERS  0.18    0.18
# >  EMOCOAGR  0.18    0.18
# >  PROBSELF -0.17    0.17
# >    SDLEFF  0.17    0.17
# > PERSEVAGR  0.16    0.16
# >  STRESAGR  0.16    0.16
# >  FEELSAFE  0.15    0.15
# >   CREATOP  0.15    0.15
# >   GROSAGR  0.14    0.14
# >  MATHEASE  0.13    0.13
# >   BULLIED -0.13    0.13
# >  WORKHOME -0.12    0.12
# >  ASSERAGR  0.12    0.12
# >  FAMSUPSL -0.12    0.12
# >  MATHPREF  0.12    0.12
# >    ICTRES  0.11    0.11
# >  EXERPRAC -0.10    0.10
# >   OPENCUL  0.09    0.09
# >    BELONG  0.09    0.09
# >  CREATOOS -0.09    0.09
# >   DISCLIM  0.09    0.09
# >    MACTIV  0.08    0.08
# >   SCHSUST  0.08    0.08
# >  EXPO21ST  0.08    0.08
# >   OPENART -0.06    0.06
# >    EXPOFA  0.06    0.06
# >  CREATFAM  0.06    0.06
# >   FEELLAH  0.05    0.05
# >   CREATAS -0.05    0.05
# >   DIGPREP  0.04    0.04
# >  COGACRCO  0.04    0.04
# >  CREATSCH  0.04    0.04
# >   COOPAGR  0.04    0.04
# >   MATHMOT  0.04    0.04
# >   IMAGINE  0.03    0.03
# >   ABGMATH -0.03    0.03
# >   LEARRES  0.03    0.03
# >  CREENVSC  0.02    0.02
# >  EMPATAGR  0.02    0.02
# >  STUDYHMW -0.02    0.02
# >  COGACMCO  0.02    0.02
# >    FAMSUP -0.01    0.01
# >  CREATEFF  0.01    0.01
# >   MTTRAIN  0.00    0.00
# >  INFOSEEK  0.00    0.00

tibble(
  Variable = setdiff(colnames(point_estimate_matrix), "PVmMATH"),
  rho      = point_estimate_matrix["PVmMATH", setdiff(colnames(point_estimate_matrix), "PVmMATH")],
  se       = se_matrix["PVmMATH", setdiff(colnames(point_estimate_matrix), "PVmMATH")]
) %>%
  arrange(desc(abs(rho))) %>%
  mutate(abs_rho = abs(rho),
         lo_ci = rho - z_crit * se,
         up_ci = rho + z_crit * se) %>%
  #slice_head(n = 20) %>%  # <- change n to get more/less than top 20 by |rho|
  as.data.frame() %>%
  transform(rho = round(rho, 4), se = round(se, 4), abs_rho = round(abs_rho, 4), lo_ci = round(lo_ci, 4), up_ci = round(up_ci, 4)) %>%
  #write_csv("correlations_with_PVmMATH.csv") 
  print(row.names = FALSE)
# >  Variable     rho     se abs_rho   lo_ci   up_ci
# >   MATHEFF  0.5348 0.0114  0.5348  0.5125  0.5571
# >  MATHEF21  0.3952 0.0136  0.3952  0.3686  0.4218
# >    FAMCON  0.3349 0.0155  0.3349  0.3046  0.3652
# >.   ANXMAT -0.3313 0.0150  0.3313 -0.3607 -0.3019
# >      ESCS  0.3069 0.0169  0.3069  0.2738  0.3401
# >   HOMEPOS  0.2410 0.0153  0.2410  0.2109  0.2710
# >  CURIOAGR  0.2176 0.0153  0.2176  0.1877  0.2475
# >   WORKPAY -0.1893 0.0172  0.1893 -0.2231 -0.1555
# >  MATHPERS  0.1780 0.0167  0.1780  0.1452  0.2108
# >  EMOCOAGR  0.1772 0.0151  0.1772  0.1477  0.2068
# >  PROBSELF -0.1702 0.0170  0.1702 -0.2035 -0.1370
# >    SDLEFF  0.1668 0.0162  0.1668  0.1352  0.1985
# > PERSEVAGR  0.1650 0.0171  0.1650  0.1315  0.1985
# >  STRESAGR  0.1592 0.0145  0.1592  0.1309  0.1875
# >  FEELSAFE  0.1542 0.0169  0.1542  0.1210  0.1873
# >   CREATOP  0.1454 0.0145  0.1454  0.1171  0.1738
# >   GROSAGR  0.1449 0.0181  0.1449  0.1094  0.1803
# >  MATHEASE  0.1323 0.0174  0.1323  0.0981  0.1665
# >   BULLIED -0.1266 0.0186  0.1266 -0.1631 -0.0900
# >  WORKHOME -0.1221 0.0161  0.1221 -0.1537 -0.0904
# >  ASSERAGR  0.1212 0.0142  0.1212  0.0934  0.1490
# >  FAMSUPSL -0.1157 0.0156  0.1157 -0.1463 -0.0851
# >  MATHPREF  0.1154 0.0176  0.1154  0.0809  0.1499
# >    ICTRES  0.1132 0.0171  0.1132  0.0798  0.1467
# >  EXERPRAC -0.0998 0.0155  0.0998 -0.1301 -0.0695
# >   OPENCUL  0.0948 0.0333  0.0948  0.0296  0.1600
# >    BELONG  0.0931 0.0171  0.0931  0.0595  0.1266
# >  CREATOOS -0.0918 0.0156  0.0918 -0.1223 -0.0613
# >   DISCLIM  0.0907 0.0191  0.0907  0.0532  0.1282
# >    MACTIV  0.0845 0.0252  0.0845  0.0352  0.1338
# >   SCHSUST  0.0845 0.0168  0.0845  0.0516  0.1174
# >  EXPO21ST  0.0755 0.0195  0.0755  0.0372  0.1137
# >   OPENART -0.0598 0.0171  0.0598 -0.0933 -0.0262
# >    EXPOFA  0.0587 0.0176  0.0587  0.0241  0.0932
# >  CREATFAM  0.0582 0.0170  0.0582  0.0248  0.0916
# >   FEELLAH  0.0529 0.0170  0.0529  0.0196  0.0862
# >   CREATAS -0.0499 0.0183  0.0499 -0.0859 -0.0140
# >   DIGPREP  0.0436 0.0351  0.0436 -0.0252  0.1125
# >  COGACRCO  0.0421 0.0155  0.0421  0.0117  0.0726
# >  CREATSCH  0.0420 0.0180  0.0420  0.0068  0.0772
# >   COOPAGR  0.0390 0.0184  0.0390  0.0029  0.0751
# >   MATHMOT  0.0382 0.0149  0.0382  0.0089  0.0674
# >   IMAGINE  0.0306 0.0156  0.0306  0.0000  0.0612
# >   ABGMATH -0.0294 0.0295  0.0294 -0.0871  0.0284
# >   LEARRES  0.0291 0.0182  0.0291 -0.0067  0.0648
# >  CREENVSC  0.0235 0.0365  0.0235 -0.0481  0.0950
# >  EMPATAGR  0.0220 0.0178  0.0220 -0.0129  0.0569
# >  STUDYHMW -0.0174 0.0184  0.0174 -0.0536  0.0187
# >  COGACMCO  0.0172 0.0172  0.0172 -0.0166  0.0509
# >    FAMSUP -0.0095 0.0175  0.0095 -0.0437  0.0247
# >  CREATEFF  0.0066 0.0162  0.0066 -0.0251  0.0383
# >   MTTRAIN -0.0020 0.0238  0.0020 -0.0488  0.0447
# >  INFOSEEK  0.0015 0.0156  0.0015 -0.0290  0.0321

ggplot(
  tibble::tibble(
    Variable = setdiff(colnames(point_estimate_matrix), "PVmMATH"),
    rho = point_estimate_matrix["PVmMATH", setdiff(colnames(point_estimate_matrix), "PVmMATH")],
    se  = se_matrix["PVmMATH", setdiff(colnames(point_estimate_matrix), "PVmMATH")]
  ) %>%
    arrange(desc(abs(rho))) %>%
    mutate(
      lo_ci = rho - z_crit * se,
      up_ci = rho + z_crit * se,
      Variable = factor(Variable, levels = rev(Variable))  # largest |rho| at top
    )
  # %>% slice_head(n = 20)  # <- uncomment to limit to top N by |rho|
  ,
  aes(x = rho, y = Variable, fill = rho > 0)
) +
  geom_vline(xintercept = 0, color = "grey60") +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_errorbarh(aes(xmin = lo_ci, xmax = up_ci), height = 0.25, color = "grey25") +
  labs(
    title = "Correlations with PVmMATH (ordered by |ρ̂|)",
    x = expression(rho ~ "(weighted Pearson)"),
    y = NULL
  ) +
  scale_fill_manual(values = c("TRUE" = "#1b9e77", "FALSE" = "#d95f02")) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank())

# --- Top correlations among Xs (signed, sorted) ---    
as.data.frame(as.table(point_estimate_matrix[voi_num, voi_num, drop = FALSE])) %>%
  as_tibble() %>%
  rename(x = Var1, y = Var2, rho = Freq) %>%
  filter(x != y) %>%
  mutate(x = as.character(x), y = as.character(y)) %>%
  filter(x < y) %>%  # keep each pair once (upper triangle)
  left_join(
    as.data.frame(as.table(se_matrix[voi_num, voi_num, drop = FALSE])) %>%
      as_tibble() %>%
      rename(x = Var1, y = Var2, se = Freq) %>%
      mutate(x = as.character(x), y = as.character(y)),
    by = c("x","y")
  ) %>%
  mutate(
    abs_rho = abs(rho),
    lo_ci = rho - z_crit * se,
    up_ci = rho + z_crit * se
  ) %>%
  arrange(desc(abs_rho)) %>%
  # slice_head(n = 50) %>%   # <- uncomment to show top N pairs
  mutate(across(c(rho, se, abs_rho, lo_ci, up_ci), ~ round(.x, 4))) %>%
  #write_csv("pairwise_correlations_amongX.csv") %>%
  print(n = 50)

as.data.frame(as.table(point_estimate_matrix[voi_num, voi_num, drop = FALSE])) %>%
  as_tibble() %>%
  rename(x1 = Var1, x2 = Var2, rho = Freq) %>%
  mutate(x1 = as.character(x1), x2 = as.character(x2)) %>%
  filter(x1 != x2, x1 < x2) %>%                           # keep one direction
  left_join(
    as.data.frame(as.table(se_matrix[voi_num, voi_num, drop = FALSE])) %>%
      as_tibble() %>%
      rename(x1 = Var1, x2 = Var2, se = Freq) %>%
      mutate(x1 = as.character(x1), x2 = as.character(x2)),
    by = c("x1","x2")
  ) %>%
  filter(abs(rho) >= 0.50) %>%
  arrange(desc(abs(rho))) %>%
  mutate(lo_ci = rho - z_crit * se, up_ci = rho + z_crit * se) %>%
  mutate(across(c(rho, se, lo_ci, up_ci), ~ round(.x, 3))) %>%
  #write_csv("pairwise_correlations_amongXpartial.csv") %>%  # ->filter(abs(rho) >= 0.50) %>%was used
  print(n = Inf)
# > # A tibble: 8 × 6
# >  x1       x2         rho    se lo_ci up_ci
# >  <chr>    <chr>    <dbl> <dbl> <dbl> <dbl>
# >1 HOMEPOS  ICTRES   0.742 0.007 0.729 0.756
# >2 ESCS     HOMEPOS  0.711 0.009 0.694 0.727
# >3 MATHEF21 MATHEFF  0.702 0.009 0.684 0.72 
# >4 CREATAS  CREATOOS 0.604 0.022 0.56  0.648
# >5 CREATEFF CREATOP  0.569 0.012 0.545 0.594
# >6 EMOCOAGR STRESAGR 0.512 0.013 0.487 0.537
# >7 COGACMCO COGACRCO 0.511 0.012 0.488 0.534
# >8 EXPO21ST EXPOFA   0.505 0.016 0.474 0.537


# write.csv(round(point_estimate_matrix, 4), "point_estimate_correlation_matrix.csv")
# write.csv(round(se_matrix, 4), "se_correlation_matrix.csv")
# write.table(cbind(Row = rownames(round(point_estimate_matrix, 4)),
#                   as.data.frame(round(point_estimate_matrix, 4))),
#             "correlation_matrices_stacked.csv", sep = ",",
#             row.names = FALSE, col.names = TRUE)
# 
# cat("\n", file = "correlation_matrices_stacked.csv", append = TRUE)
# 
# write.table(cbind(Row = rownames(round(se_matrix, 4)),
#                   as.data.frame(round(se_matrix, 4))),
#             "correlation_matrices_stacked.csv", sep = ",",
#             row.names = FALSE, col.names = TRUE, append = TRUE)

# ---- II. Explanatory Modelling----
## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)      # For reading SPSS .sav files
library(tidyverse)  # Includes dplyr, tidyr, purrr, ggplot2, tibble, etc.
library(broom)      # For tidying model output
library(intsvy)     # For analyze PISA data
library(tictoc)     # For timing code execution

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("haven", "tidyverse", "broom","intsvy", "tictoc"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         haven         tidyverse             broom            intsvy            tictoc 
# > "haven 2.5.4" "tidyverse 2.0.0"     "broom 1.0.8"      "intsvy 2.9"    "tictoc 1.2.1"

# Load data
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE) # Preserve SPSS's user-defined missing values
dim(pisa_2022_canada_merged)   
# > [1] 23073  1699
stopifnot(                                                                    
  sum(is.na(pisa_2022_canada_merged[, paste0("PV", 1:10, "MATH")])) == 0,                        # or: all(colSums(is.na(pisa_2022_canada_merged[, pvmaths, drop = FALSE])) == 0)
  sum(is.na(pisa_2022_canada_merged$W_FSTUWT)) == 0,                          # or: all(!is.na(pisa_2022_canada_merged[[final_wt]]))
  colSums(is.na(pisa_2022_canada_merged[paste0("W_FSTURWT", 1:80)])) == 0
)

# Load metadata + missing summary: full variables
metadata_missing_student <- read_csv("data/pisa2022/metadata_missing_student.csv", show_col_types = FALSE)
metadata_missing_school <- read_csv("data/pisa2022/metadata_missing_school.csv", show_col_types = FALSE)

# Load metadata + missing summary: student/school questionnaire derived variables
stuq_dvs_metadata_missing_student <- readr::read_csv("data/pisa2022/stuq_dvs_metadata_missing_student.csv", show_col_types = FALSE)
schq_dvs_metadata_missing_school <- read_csv("data/pisa2022/schq_dvs_metadata_missing_school.csv",  show_col_types = FALSE)

# Constants
M <- 10                         # Number of plausible values
G <- 80                         # Number of BRR replicate weights
k <- 0.5                        # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)          # 95% z-critical value for confidence interval (CI)

# y: responce/dependent variables
pvmaths  <- paste0("PV", 1:M, "MATH")   # PV1MATH to PV10MATH

# X: covariates/explanatory variables (VOI = variables of interest): numeric (continuous + ordinal) + categorical (nominal)
voi_num <- c(
  # --- Student Questionnaire Derived Variables ---
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

voi_cat <- c(
  # --- Student Questionnaire Variables ---
  "REGION",      # REGION
  # --- Student Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### Basic demographics (Module 1)
  "ST004D01T",   # Student (Standardized) Gender
  ### Migration and language exposure (Module 4)
  "IMMIG",       # Index on immigrant background (OECD definition
  "LANGN",       # Language spoken at home
  # --- School Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### School type and infrastructure (Module 11)
  "SCHLTYPE")     # School type
stopifnot(!anyDuplicated(voi_cat))
stopifnot(all(sapply(pisa_2022_canada_merged[, voi_cat], is.numeric)))
length(voi_cat)

voi_all <- c(voi_num, voi_cat)
stopifnot(!anyDuplicated(voi_all))
voi_all

length(voi_num);length(voi_cat);length(voi_all)
# > [1] 53
# > [1] 5
# > [1] 58

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)   # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                 # Final student weight

length(rep_wts); length(final_wt)
# > [1] 80
# > [1] 1

### ---- Prepare data ----
temp_data <- pisa_2022_canada_merged %>%
  select(                      
    CNTSCHID, CNTSTUID,                                       # IDs
    all_of(final_wt), all_of(rep_wts),                        # Weights
    all_of(pvmaths), all_of(voi_all)                          # PVs + predictors
  ) %>%
  mutate(
    LANGN = na_if(LANGN, 999),                                # Missing 999 -> NA
    across(                                                   # label → factor for categorical VOIs
      all_of(voi_cat),
      ~ if (inherits(.x, "haven_labelled"))
        haven::as_factor(.x, levels = "labels") else as.factor(.x)
    )
  ) %>%                                                       # -> can compare performance without listwise deletion
  filter(IMMIG != "No Response") %>%                          # Treat "No Response" in `IMMIG` as missing values and drop
  filter(if_all(all_of(voi_all), ~ !is.na(.))) %>%            # Drop missing values; ST004D01T: User-defined NA `Not Applicable` is kept as a category
  droplevels()                                                # Drop levels not present   
dim(temp_data)
# > [1] 6755  151

# Quick invariants for weights & PVs after filtering
stopifnot(
  sum(is.na(temp_data[, pvmaths])) == 0,                        # or: all(colSums(is.na(temp_data[, pvmaths, drop = FALSE])) == 0)
  sum(is.na(temp_data$W_FSTUWT)) == 0,                          # or: all(!is.na(temp_data[[final_wt]]))
  colSums(is.na(temp_data[paste0("W_FSTURWT", 1:80)])) == 0
)

## ---- Initial Exploration ----

# Fit weighted linear model: Math PV1 ~ Gender 
mod <- lm(as.formula(paste("PV1MATH ~", paste(voi_all, collapse = " + "))), data = temp_data, weights = temp_data[[final_wt]]) 

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

### ---- Explore with intsvy ----
#library(intsvy)

pisa.mean(variable="BULLIED", by="REGION", data=pisa_2022_canada_merged)
pisa.mean(variable="BULLIED", by="ST004D01T", data=pisa_2022_canada_merged)
pisa.mean(variable="BULLIED", by="IMMIG", data=pisa_2022_canada_merged)
pisa.mean(variable="BULLIED", by="LANGN", data=pisa_2022_canada_merged)
pisa.mean(variable="BULLIED", by="SCHLTYPE", data=pisa_2022_canada_merged)

# --- Out-of-School Experience (oos) ---
# Null model
pisa.reg.pv(pvlabel=pvmaths, x="EXERPRAC", data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x="STUDYHMW", data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x="WORKPAY", data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x="WORKHOME", data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "STUDYHMW"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "WORKPAY"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "WORKHOME"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("STUDYHMW", "WORKPAY"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("STUDYHMW", "WORKHOME"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("WORKPAY", "WORKHOME"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "STUDYHMW", "WORKPAY"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "STUDYHMW", "WORKHOME"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("STUDYHMW", "WORKPAY", "WORKHOME"), data=temp_data)
pisa.reg.pv(pvlabel=pvmaths, x=c("EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME"), data=temp_data)

## ---- Fit main models using final student weight (W_FSTUWT) ----
tic("Fitting main models")
main_models <- lapply(pvmaths, function(pv) {
  formula <- as.formula(paste(pv, "~", paste(voi_all, collapse = " + ")))
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
toc()
# > Fitting main models: 2.684 sec elapsed
main_models[[1]] # View structure of the first fitted model
main_models[[2]] # View structure of the second fitted model

# --- Extract and average main fit estimates across plausible values (PVs) ---

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs[, 1:6]
# >      (Intercept)     MATHMOT MATHEASE   MATHPREF  EXERPRAC   STUDYHMW
# > [1,]    494.8378  2.877223589 5.997186 -0.4210800 -2.835007 -1.0622968
# > [2,]    503.6690  1.167463372 4.393129  2.7283678 -2.988386 -1.2531694
# > [3,]    502.5249  0.376134167 6.627748  1.5216877 -2.780771 -1.1958908
# > [4,]    498.4865  4.455321796 4.760063  2.9410972 -3.337528 -1.0031583
# > [5,]    501.8722  2.102795924 5.821944  0.4521781 -3.078614 -1.2449127
# > [6,]    497.6119  1.277123471 5.021934  1.2797424 -3.035915 -1.0206512
# > [7,]    501.9429  5.448352300 3.457612  1.2393519 -2.956716 -1.3511980
# > [8,]    501.2598 -0.836456804 6.637128  5.0012159 -2.981532 -0.9545580
# > [9,]    492.4047 -0.001058674 8.709587 -2.3328020 -3.024421 -0.9303493
# > [10,]   500.4354  2.819282022 6.973485  0.3447682 -3.037162 -0.7923084

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  499.504514    1.968618    5.839982    1.275453   -3.005605   -1.080849 

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
# > # A tibble: 10 × 3
# > Term                               Estimate `|Estimate|`
# > <chr>                                 <dbl>        <dbl>
# > 1 MATHEFF                                24.9         24.9
# > 2 REGIONCanada: Ontario                  22.8         22.8
# > 3 REGIONCanada: Alberta                  21.0         21.0
# > 4 REGIONCanada: British Columbia         19.3         19.3
# > 5 REGIONCanada: Prince Edward Island     18.7         18.7
# > 6 REGIONCanada: Quebec                   18.5         18.5
# > 7 ESCS                                   17.7         17.7
# > 8 LANGNFrench                            17.1         17.1
# > 9 HOMEPOS                                16.8         16.8
# > 10 ST004D01TMale                         13.2         13.2

# Display non-zero coefs
coef_df %>%
  filter(Term != "(Intercept)", abs(Estimate) > 0) %>%
  arrange(desc(mag), desc(Estimate)) %>%
  transmute(Term, Estimate = round(Estimate, 3), `|Estimate|` = round(mag, 3)) %>%
  as.data.frame() %>%
  #head() %>%
  print(row.names=TRUE)

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

# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.4535374 0.4619118 0.4681609 0.4655361 0.4617986 0.4569466 0.4530554 0.4756708 0.4665320 0.4656987
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.4478144 0.4562765 0.4625911 0.4599388 0.4561621 0.4512593 0.4473274 0.4701796 0.4609451 0.4601031
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 268.8156 263.9725 265.2834 262.7127 263.6366 264.5952 265.7876 262.0160 265.0216 262.6824
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 79.24849 81.96792 84.05302 83.17128 81.93060 80.34544 79.09453 86.62452 83.50479 83.22567

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.4628848
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.4572597
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 264.4524
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 82.31663

## ---- Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
tic("Fitting replicate models")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste(pv, "~", paste(voi_all, collapse = " + ")))
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
toc()
# > Fitting replicate models: 35.634 sec elapsed
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
sampling_var_matrix_coef[1:6, ]
# >                    [,1]        [,2]        [,3]        [,4]        [,5]        [,6]        [,7]        [,8]        [,9]       [,10]
# > (Intercept) 185.6326911 133.5582091 170.5640086 129.3259787 154.6583380 113.9336076 164.8218895 144.5189114 168.6316447 139.1882047
# > MATHMOT      34.1106891  32.3510524  29.3038561  32.1539647  29.4257363  23.7232353  31.8718359  27.7498528  27.8723568  29.8786318
# > MATHEASE     14.0223410  14.9565164  11.7372069  11.7119601  14.2701940  13.2785317  16.2680924  15.6143412  12.5375453  16.8374110
# > MATHPREF     15.8865193  16.3138789  15.9029296  16.1434663  16.4222215  14.0614665  19.1464587  17.5310290  13.2781291  19.6365946
# > EXERPRAC      0.1449533   0.1394045   0.1369791   0.1479410   0.1354914   0.1395043   0.1380321   0.1181272   0.1342848   0.1375226
# > STUDYHMW      0.1723942   0.1653131   0.1973243   0.1863082   0.1721654   0.1271864   0.1596557   0.1928032   0.1356106   0.1536653

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 150.4833483  29.8441211  14.1234140  16.4322693   0.1372240   0.1662426 

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 13.19662592  3.91817030  2.26722380  4.03003094  0.02229909  0.03053261 

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 164.9996369  34.1541085  16.6173602  20.8653034   0.1617530   0.1998285 

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  12.8452184   5.8441516   4.0764396   4.5678554   0.4021853   0.4470218   

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
# > [1] 0.01413551
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.01428355
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 5.688552
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 4.706435

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
# >                                 Term      Estimate Std. Error      t value Pr(>|t|) t_Signif      z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >                          (Intercept) 499.504514496 12.8452184 38.886416505  < 2e-16      *** 38.886416505  < 2e-16      ***   <NA>     <NA>
# >                              MATHMOT   1.968618116  5.8441516  0.336852675 0.736239           0.336852675 0.736228            <NA>     <NA>
# >                             MATHEASE   5.839981556  4.0764396  1.432618182 0.152014           1.432618182 0.151967            <NA>     <NA>
# >                             MATHPREF   1.275452733  4.5678554  0.279223532 0.780082           0.279223532 0.780073            <NA>     <NA>
# >                             EXERPRAC  -3.005605185  0.4021853 -7.473185007 8.83e-14      *** -7.473185007 7.83e-14      ***   <NA>     <NA>
# >                             STUDYHMW  -1.080849290  0.4470218 -2.417889415 0.015637        * -2.417889415 0.015611        *   <NA>     <NA>
# >                              WORKPAY  -3.778939216  0.4976670 -7.593309448 3.54e-14      *** -7.593309448 3.12e-14      ***   <NA>     <NA>
# >                             WORKHOME  -0.699248033  0.3826489 -1.827388150 0.067686        . -1.827388150 0.067641        .   <NA>     <NA>
# >                              HOMEPOS  16.814105724  2.9206181  5.757036671 8.94e-09      ***  5.757036671 8.56e-09      ***   <NA>     <NA>
# >                               ICTRES -10.586745365  2.1276748 -4.975734643 6.66e-07      *** -4.975734643 6.50e-07      ***   <NA>     <NA>
# >                             INFOSEEK  -0.819608844  1.5976204 -0.513018510 0.607955          -0.513018510 0.607938            <NA>     <NA>
# >                              BULLIED  -5.089331605  1.5003274 -3.392147383 0.000697      *** -3.392147383 0.000693      ***   <NA>     <NA>
# >                             FEELSAFE   2.094715532  1.3343392  1.569852378 0.116497           1.569852378 0.116449            <NA>     <NA>
# >                               BELONG  -4.442784093  1.7846391 -2.489457953 0.012818        * -2.489457953 0.012794        *   <NA>     <NA>
# >                              GROSAGR   4.634880122  1.4493467  3.197909937 0.001391       **  3.197909937 0.001384       **   <NA>     <NA>
# >                               ANXMAT  -7.006311910  1.3047689 -5.369772143 8.15e-08      *** -5.369772143 7.88e-08      ***   <NA>     <NA>
# >                              MATHEFF  24.888878052  1.6631680 14.964740396  < 2e-16      *** 14.964740396  < 2e-16      ***   <NA>     <NA>
# >                             MATHEF21   2.826031712  2.3235038  1.216280208 0.223921           1.216280208 0.223878            <NA>     <NA>
# >                             MATHPERS   3.986155859  1.4886920  2.677622892 0.007433       **  2.677622892 0.007415       **   <NA>     <NA>
# >                               FAMCON   6.472166165  1.0326471  6.267548740 3.90e-10      ***  6.267548740 3.67e-10      ***   <NA>     <NA>
# >                             ASSERAGR   3.161125232  1.3237016  2.388094971 0.016964        *  2.388094971 0.016936        *   <NA>     <NA>
# >                              COOPAGR  -2.256542784  1.5854859 -1.423249946 0.154710          -1.423249946 0.154664            <NA>     <NA>
# >                             CURIOAGR   6.617477099  1.3415356  4.932763118 8.30e-07      ***  4.932763118 8.11e-07      ***   <NA>     <NA>
# >                             EMOCOAGR   3.141844046  1.3877792  2.263936504 0.023610        *  2.263936504 0.023578        *   <NA>     <NA>
# >                             EMPATAGR  -0.221952429  1.5478442 -0.143394553 0.885983          -0.143394553 0.885979            <NA>     <NA>
# >                            PERSEVAGR   0.226019768  1.4791902  0.152799661 0.878561           0.152799661 0.878556            <NA>     <NA>
# >                             STRESAGR  -1.985203912  1.5495698 -1.281132324 0.200192          -1.281132324 0.200147            <NA>     <NA>
# >                               EXPOFA   0.103991965  1.7875070  0.058177096 0.953609           0.058177096 0.953608            <NA>     <NA>
# >                             EXPO21ST  -0.919877422  1.8684351 -0.492325069 0.622506          -0.492325069 0.622490            <NA>     <NA>
# >                             COGACRCO   1.200358581  1.5223028  0.788515012 0.430423           0.788515012 0.430396            <NA>     <NA>
# >                             COGACMCO  -4.759693830  1.6347968 -2.911489563 0.003609       ** -2.911489563 0.003597       **   <NA>     <NA>
# >                              DISCLIM  -0.271355093  1.4673052 -0.184934325 0.853286          -0.184934325 0.853281            <NA>     <NA>
# >                               FAMSUP  -2.325302364  1.5228209 -1.526970356 0.126816          -1.526970356 0.126768            <NA>     <NA>
# >                             CREATFAM  -1.919875311  1.6737403 -1.147056876 0.251399          -1.147056876 0.251358            <NA>     <NA>
# >                             CREATSCH  -4.427631153  1.5982774 -2.770251972 0.005617       ** -2.770251972 0.005601       **   <NA>     <NA>
# >                             CREATEFF  -7.390133556  1.6959104 -4.357620356 1.33e-05      *** -4.357620356 1.31e-05      ***   <NA>     <NA>
# >                              CREATOP   5.284388222  1.8616455  2.838557669 0.004545       **  2.838557669 0.004532       **   <NA>     <NA>
# >                              IMAGINE  -0.004743614  1.4358692 -0.003303653 0.997364          -0.003303653 0.997364            <NA>     <NA>
# >                              OPENART  -1.991196585  1.6201949 -1.228985849 0.219120          -1.228985849 0.219077            <NA>     <NA>
# >                              CREATAS   0.874332739  1.7422113  0.501852281 0.615788           0.501852281 0.615771            <NA>     <NA>
# >                             CREATOOS  -4.320939673  1.5984640 -2.703182426 0.006885       ** -2.703182426 0.006868       **   <NA>     <NA>
# >                             FAMSUPSL -10.283526009  1.4688396 -7.001122749 2.79e-12      *** -7.001122749 2.54e-12      ***   <NA>     <NA>
# >                              FEELLAH  -2.931796826  1.6089514 -1.822178606 0.068473        . -1.822178606 0.068428        .   <NA>     <NA>
# >                             PROBSELF  -0.151475065  1.4383501 -0.105311678 0.916132          -0.105311678 0.916128            <NA>     <NA>
# >                               SDLEFF   1.270093302  1.5746207  0.806602703 0.419924           0.806602703 0.419895            <NA>     <NA>
# >                              SCHSUST   4.509677041  1.5458092  2.917356748 0.003542       **  2.917356748 0.003530       **   <NA>     <NA>
# >                              LEARRES   0.239155652  1.7143945  0.139498607 0.889060           0.139498607 0.889056            <NA>     <NA>
# >                                 ESCS  17.716443202  2.4272107  7.299095582 3.23e-13      ***  7.299095582 2.90e-13      ***   <NA>     <NA>
# >                               MACTIV   1.655452801  1.1127041  1.487774502 0.136858           1.487774502 0.136810            <NA>     <NA>
# >                              ABGMATH  -1.788625190  2.5688156 -0.696284005 0.486275          -0.696284005 0.486251            <NA>     <NA>
# >                              MTTRAIN  -0.048722348  1.6367716 -0.029767348 0.976253          -0.029767348 0.976253            <NA>     <NA>
# >                             CREENVSC  -1.903100761  1.7723613 -1.073765714 0.282966          -1.073765714 0.282928            <NA>     <NA>
# >                              OPENCUL   4.267552139  1.7886030  2.385969517 0.017062        *  2.385969517 0.017034        *   <NA>     <NA>
# >                              DIGPREP   0.555044338  1.5873686  0.349663167 0.726603           0.349663167 0.726591            <NA>     <NA>
# >   REGIONCanada: Prince Edward Island  18.684814865 11.8734981  1.573657117 0.115614           1.573657117 0.115567            <NA>     <NA>
# >            REGIONCanada: Nova Scotia   5.862415958  6.7072719  0.874038811 0.382129           0.874038811 0.382097            <NA>     <NA>
# >          REGIONCanada: New Brunswick   1.673789341  6.5241649  0.256552272 0.797532           0.256552272 0.797524            <NA>     <NA>
# >                 REGIONCanada: Quebec  18.508453317  6.8291923  2.710196551 0.006741       **  2.710196551 0.006724       **   <NA>     <NA>
# >                REGIONCanada: Ontario  22.814332086  5.8163332  3.922459598 8.85e-05      ***  3.922459598 8.76e-05      ***   <NA>     <NA>
# >               REGIONCanada: Manitoba   3.847775333  6.0351318  0.637562769 0.523780           0.637562769 0.523758            <NA>     <NA>
# >           REGIONCanada: Saskatchewan   8.020007347  6.7377023  1.190317854 0.233964           1.190317854 0.233921            <NA>     <NA>
# >                REGIONCanada: Alberta  20.986535281  6.5906717  3.184278679 0.001458       **  3.184278679 0.001451       **   <NA>     <NA>
# >       REGIONCanada: British Columbia  19.345814538  7.0989321  2.725172506 0.006444       **  2.725172506 0.006427       **   <NA>     <NA>
# >                        ST004D01TMale  13.155653220  2.8590076  4.601475423 4.27e-06      ***  4.601475423 4.20e-06      ***   <NA>     <NA>
# >              ST004D01TNot Applicable  -5.992278717 13.2460907 -0.452380921 0.651009          -0.452380921 0.650995            <NA>     <NA>
# >       IMMIGSecond-Generation student   4.042559538  3.9707827  1.018076240 0.308679           1.018076240 0.308642            <NA>     <NA>
# >        IMMIGFirst-Generation student  -6.837717696  4.7600116 -1.436491814 0.150909          -1.436491814 0.150862            <NA>     <NA>
# >                          LANGNFrench  17.129640854  5.3056436  3.228569823 0.001250       **  3.228569823 0.001244       **   <NA>     <NA>
# >          LANGNAnother language (CAN)  11.246840578  5.2742192  2.132418126 0.033009        *  2.132418126 0.032972        *   <NA>     <NA>
# > SCHLTYPEPrivate Government-dependent   4.887152832  8.6076464  0.567768773 0.570211           0.567768773 0.570192            <NA>     <NA>
# >                       SCHLTYPEPublic  -7.895959706  9.1788631 -0.860232869 0.389692          -0.860232869 0.389661            <NA>     <NA>
# >                            R-squared   0.460000000  0.0100000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                   Adjusted R-squared   0.460000000  0.0100000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                  Residual Std. Error 264.450000000  5.6900000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                          F-statistic  82.320000000  4.7100000           NA     <NA>     <NA>           NA     <NA>     <NA> <2e-16      ***
as.data.frame(result_table) %>% head(10) %>% print(row.names = FALSE)

# -- Visualize coefs: ADD bars + 95% CI from Rubin+BRR SEs ---
ggplot(
  coef_df %>%
    left_join(enframe(se_final_coef, name = "Term", value = "SE"), by = "Term") %>%
    mutate(
      lo95 = Estimate - z_crit * SE,
      hi95 = Estimate + z_crit * SE,
      Term = fct_reorder(Term, mag)  # order by |Estimate|
    ),
  aes(x = Term, y = Estimate, fill = sign)
) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.6, alpha = 0.6) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_errorbar(aes(ymin = lo95, ymax = hi95), width = 0.25, linewidth = 0.4) +
  coord_flip() +
  scale_y_continuous(limits = c(-max_abs, max_abs)) +  # negative left, positive right
  scale_fill_manual(values = c("Negative" = "#b2cbea", "Positive" = "#2b6cb0")) +
  labs(title = "Pooled regression coefficients with 95% CIs (ordered by |estimate|)",
       x = NULL, y = "Estimate", fill = NULL) +
  theme_minimal() +
  theme(legend.position = "top")

### ---- Compare results with intsvy ----
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=voi_all, data=temp_data)
# >                                      Estimate Std. Error t value
# > (Intercept)                            499.50      12.85   38.89
# > MATHMOT                                  1.97       5.84    0.34
# > MATHEASE                                 5.84       4.08    1.43
# > MATHPREF                                 1.28       4.57    0.28
# > EXERPRAC                                -3.01       0.40   -7.47
# > STUDYHMW                                -1.08       0.45   -2.42
# > WORKPAY                                 -3.78       0.50   -7.59
# > WORKHOME                                -0.70       0.38   -1.83
# > HOMEPOS                                 16.81       2.92    5.76
# > ICTRES                                 -10.59       2.13   -4.98
# > INFOSEEK                                -0.82       1.60   -0.51
# > BULLIED                                 -5.09       1.50   -3.39
# > FEELSAFE                                 2.09       1.33    1.57
# > BELONG                                  -4.44       1.78   -2.49
# > GROSAGR                                  4.63       1.45    3.20
# > ANXMAT                                  -7.01       1.30   -5.37
# > MATHEFF                                 24.89       1.66   14.96
# > MATHEF21                                 2.83       2.32    1.22
# > MATHPERS                                 3.99       1.49    2.68
# > FAMCON                                   6.47       1.03    6.27
# > ASSERAGR                                 3.16       1.32    2.39
# > COOPAGR                                 -2.26       1.59   -1.42
# > CURIOAGR                                 6.62       1.34    4.93
# > EMOCOAGR                                 3.14       1.39    2.26
# > EMPATAGR                                -0.22       1.55   -0.14
# > PERSEVAGR                                0.23       1.48    0.15
# > STRESAGR                                -1.99       1.55   -1.28
# > EXPOFA                                   0.10       1.79    0.06
# > EXPO21ST                                -0.92       1.87   -0.49
# > COGACRCO                                 1.20       1.52    0.79
# > COGACMCO                                -4.76       1.63   -2.91
# > DISCLIM                                 -0.27       1.47   -0.18
# > FAMSUP                                  -2.33       1.52   -1.53
# > CREATFAM                                -1.92       1.67   -1.15
# > CREATSCH                                -4.43       1.60   -2.77
# > CREATEFF                                -7.39       1.70   -4.36
# > CREATOP                                  5.28       1.86    2.84
# > IMAGINE                                  0.00       1.44    0.00
# > OPENART                                 -1.99       1.62   -1.23
# > CREATAS                                  0.87       1.74    0.50
# > CREATOOS                                -4.32       1.60   -2.70
# > FAMSUPSL                               -10.28       1.47   -7.00
# > FEELLAH                                 -2.93       1.61   -1.82
# > PROBSELF                                -0.15       1.44   -0.11
# > SDLEFF                                   1.27       1.57    0.81
# > SCHSUST                                  4.51       1.55    2.92
# > LEARRES                                  0.24       1.71    0.14
# > ESCS                                    17.72       2.43    7.30
# > MACTIV                                   1.66       1.11    1.49
# > ABGMATH                                 -1.79       2.57   -0.70
# > MTTRAIN                                 -0.05       1.64   -0.03
# > CREENVSC                                -1.90       1.77   -1.07
# > OPENCUL                                  4.27       1.79    2.39
# > DIGPREP                                  0.56       1.59    0.35
# > REGIONCanada: Prince Edward Island      18.68      11.87    1.57
# > REGIONCanada: Nova Scotia                5.86       6.71    0.87
# > REGIONCanada: New Brunswick              1.67       6.52    0.26
# > REGIONCanada: Quebec                    18.51       6.83    2.71
# > REGIONCanada: Ontario                   22.81       5.82    3.92
# > REGIONCanada: Manitoba                   3.85       6.04    0.64
# > REGIONCanada: Saskatchewan               8.02       6.74    1.19
# > REGIONCanada: Alberta                   20.99       6.59    3.18
# > REGIONCanada: British Columbia          19.35       7.10    2.73
# > ST004D01TMale                           13.16       2.86    4.60
# > ST004D01TNot Applicable                 -5.99      13.25   -0.45
# > IMMIGSecond-Generation student           4.04       3.97    1.02
# > IMMIGFirst-Generation student           -6.84       4.76   -1.44
# > LANGNFrench                             17.13       5.31    3.23
# > LANGNAnother language (CAN)             11.25       5.27    2.13
# > SCHLTYPEPrivate Government-dependent     4.89       8.61    0.57
# > SCHLTYPEPublic                          -7.90       9.18   -0.86
# > R-squared                                0.46       0.01   32.75

# Remark: results are validated also using IEA IDB Analyzer, with consistency found. 

# ---- III. Predictive Modelling ----
## ---- Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)      # For reading SPSS .sav files
library(tidyverse)  # Includes dplyr, tidyr, purrr, ggplot2, tibble, etc.
library(broom)      # For tidying model output
library(intsvy)     # For analyze PISA data
library(tictoc)     # For timing code execution

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("haven", "tidyverse", "broom","intsvy", "tictoc"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         haven         tidyverse             broom            intsvy            tictoc 
# > "haven 2.5.4" "tidyverse 2.0.0"     "broom 1.0.8"      "intsvy 2.9"    "tictoc 1.2.1"

# Load data
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE) # Preserve SPSS's user-defined missing values
dim(pisa_2022_canada_merged)   
# > [1] 23073  1699
stopifnot(                                                                    
  sum(is.na(pisa_2022_canada_merged[, paste0("PV", 1:10, "MATH")])) == 0,                        # or: all(colSums(is.na(pisa_2022_canada_merged[, pvmaths, drop = FALSE])) == 0)
  sum(is.na(pisa_2022_canada_merged$W_FSTUWT)) == 0,                          # or: all(!is.na(pisa_2022_canada_merged[[final_wt]]))
  colSums(is.na(pisa_2022_canada_merged[paste0("W_FSTURWT", 1:80)])) == 0
)

# Load metadata + missing summary: full variables
metadata_missing_student <- read_csv("data/pisa2022/metadata_missing_student.csv", show_col_types = FALSE)
metadata_missing_school <- read_csv("data/pisa2022/metadata_missing_school.csv", show_col_types = FALSE)

# Load metadata + missing summary: student/school questionnaire derived variables
stuq_dvs_metadata_missing_student <- readr::read_csv("data/pisa2022/stuq_dvs_metadata_missing_student.csv", show_col_types = FALSE)
schq_dvs_metadata_missing_school <- read_csv("data/pisa2022/schq_dvs_metadata_missing_school.csv",  show_col_types = FALSE)

# Constants
M <- 10                         # Number of plausible values
G <- 80                         # Number of BRR replicate weights
k <- 0.5                        # Fay's adjustment factor (used in BRR)
z_crit <- qnorm(0.975)          # 95% z-critical value for confidence interval (CI)

# y: target variables
pvmaths  <- paste0("PV", 1:M, "MATH")   # PV1MATH to PV10MATH

# X: features/predictors (VOI = variables of interest): numeric (continuous + ordinal) + categorical (nominal)
voi_num <- c(
  # --- Student Questionnaire Derived Variables ---
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

voi_cat <- c(
  # --- Student Questionnaire Variables ---
  "REGION",      # REGION
  # --- Student Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### Basic demographics (Module 1)
  "ST004D01T",   # Student (Standardized) Gender
  ### Migration and language exposure (Module 4)
  "IMMIG",       # Index on immigrant background (OECD definition
  "LANGN",       # Language spoken at home
  # --- School Questionnaire Derived Variables ---
  ## Simple questionnaire indices
  ### School type and infrastructure (Module 11)
  "SCHLTYPE")     # School type
stopifnot(!anyDuplicated(voi_cat))
stopifnot(all(sapply(pisa_2022_canada_merged[, voi_cat], is.numeric)))
length(voi_cat)

voi_all <- c(voi_num, voi_cat)
stopifnot(!anyDuplicated(voi_all))
voi_all

length(voi_num);length(voi_cat);length(voi_all)
# > [1] 53
# > [1] 5
# > [1] 58

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)   # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                 # Final student weight

length(rep_wts); length(final_wt)
# > [1] 80
# > [1] 1

### ---- Prepare data ----
temp_data <- pisa_2022_canada_merged %>%
  select(                      
    CNTSCHID, CNTSTUID,                                       # IDs
    all_of(final_wt), all_of(rep_wts),                        # Weights
    all_of(pvmaths), all_of(voi_all)                          # PVs + predictors
  ) %>%
  mutate(
    LANGN = na_if(LANGN, 999),                                # Missing 999 -> NA
    across(                                                   # label → factor for categorical VOIs
      all_of(voi_cat),
      ~ if (inherits(.x, "haven_labelled"))
        haven::as_factor(.x, levels = "labels") else as.factor(.x)
    )
  ) %>%                                                       # -> can compare performance without listwise deletion
  filter(IMMIG != "No Response") %>%                          # Treat "No Response" in `IMMIG` as missing values and drop
  filter(if_all(all_of(voi_all), ~ !is.na(.))) %>%            # Drop missing values; ST004D01T: User-defined NA `Not Applicable` is kept as a category
  droplevels()                                                # Drop levels not present   
dim(temp_data)
# > [1] 6755  151

# Quick invariants for weights & PVs after filtering
stopifnot(
  sum(is.na(temp_data[, pvmaths])) == 0,                        # or: all(colSums(is.na(temp_data[, pvmaths, drop = FALSE])) == 0)
  sum(is.na(temp_data$W_FSTUWT)) == 0,                          # or: all(!is.na(temp_data[[final_wt]]))
  colSums(is.na(temp_data[paste0("W_FSTURWT", 1:80)])) == 0
)

### ---- Random Train/Validation/Test (80/10/10) split ----
set.seed(123)  # Ensure reproducibility

n <- nrow(temp_data)  # 6755
indices <- sample(n)  # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.80 * n)        # 5404
n_valid <- floor(0.10 * n)        # 675
n_test  <- n - n_train - n_valid  # 676

# Assign indices
train_idx <- indices[1:n_train]
valid_idx <- indices[(n_train + 1):(n_train + n_valid)]
test_idx  <- indices[(n_train + n_valid + 1):n]

# Subset the data
train_data <- temp_data[train_idx, ]
valid_data <- temp_data[valid_idx, ]
test_data  <- temp_data[test_idx, ]

# Check if any y_true = 0; add epsilon if needed to avoid division by zero
summary(temp_data[, paste0("PV", 1:10, "MATH")]) # min > 0 for all (unweighted)
# >    PV1MATH         PV2MATH         PV3MATH         PV4MATH         PV5MATH         PV6MATH         PV7MATH         PV8MATH         PV9MATH         PV10MATH    
# > Min.   :205.3   Min.   :214.9   Min.   :208.2   Min.   :220.5   Min.   :195.8   Min.   :244.6   Min.   :211.5   Min.   :221.4   Min.   :202.4   Min.   :189.2  
# > 1st Qu.:447.1   1st Qu.:450.3   1st Qu.:447.6   1st Qu.:449.7   1st Qu.:449.3   1st Qu.:446.0   1st Qu.:449.2   1st Qu.:446.5   1st Qu.:447.9   1st Qu.:448.8  
# > Median :509.0   Median :509.8   Median :508.5   Median :509.1   Median :509.2   Median :507.7   Median :508.9   Median :508.3   Median :508.6   Median :509.6  
# > Mean   :510.5   Mean   :511.4   Mean   :510.4   Mean   :510.4   Mean   :509.9   Mean   :508.6   Mean   :510.0   Mean   :509.7   Mean   :509.9   Mean   :510.4  
# > 3rd Qu.:572.0   3rd Qu.:573.1   3rd Qu.:570.5   3rd Qu.:570.4   3rd Qu.:568.6   3rd Qu.:569.9   3rd Qu.:569.6   3rd Qu.:571.0   3rd Qu.:570.9   3rd Qu.:570.3  
# > Max.   :842.4   Max.   :875.3   Max.   :830.5   Max.   :836.2   Max.   :860.2   Max.   :885.8   Max.   :803.4   Max.   :802.4   Max.   :849.6   Max.   :903.0 
pisa.per.pv(pvlabel=paste0("PV",1:10,"MATH"), per=c(0, 25, 50, 75, 100), data=temp_data) # (weighted)
# >   Percentiles  Score Std. err.
# > 1           0 211.38     16.26
# > 2          25 460.01      2.87
# > 3          50 520.87      3.24
# > 4          75 583.09      3.35
# > 5         100 848.88     34.76

## ---- Fit on the training data----

# Remark: Fit on the training split; estimates usable for inference as in II (Explanatory Modelling).

### ---- Fit main models using final student weight (W_FSTUWT) ----
tic("Fitting main models")
main_models <- lapply(pvmaths, function(pv) {
  formula <- as.formula(paste(pv, "~", paste(voi_all, collapse = " + ")))
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
toc()
# > Fitting main models: 3.914 sec elapsed
main_models[[1]] # View structure of the first fitted model
main_models[[2]] # View structure of the second fitted model

# --- Extract and average main fit estimates across plausible values (PVs) ---

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs[, 1:6]
# >      (Intercept)    MATHMOT MATHEASE   MATHPREF  EXERPRAC   STUDYHMW
# > [1,]    491.5342  3.7750430 4.577347 -2.1201953 -2.859705 -0.7732774
# > [2,]    504.5275  1.4407941 3.279135  2.2042310 -2.941407 -1.1882206
# > [3,]    499.2086  1.4928107 5.438109 -0.1984654 -2.999454 -0.9120926
# > [4,]    495.1004  3.7838689 3.905065  1.6382725 -3.429878 -0.9418432
# > [5,]    501.2553  1.0692034 4.299356  1.0617284 -2.961355 -1.1191807
# > [6,]    493.6705  0.9487838 5.261726 -0.3652362 -2.990540 -0.7959367
# > [7,]    496.8278  5.4505637 1.865814  1.3113335 -2.959375 -1.2144663
# > [8,]    500.8264 -1.2911628 5.245712  5.2848762 -3.044626 -0.7631681
# > [9,]    490.0677  0.4408446 8.195320 -3.9535600 -3.289262 -0.5864951
# > [10,]   496.9523  4.3006969 4.705933 -0.6335653 -3.198626 -0.4249481

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 496.9970736   2.1411446   4.6773518   0.4229419  -3.0674227  -0.8719629 

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
# > # A tibble: 10 × 3
# >  Term                               Estimate `|Estimate|`
# >  <chr>                                 <dbl>        <dbl>
# >  1 MATHEFF                                24.9         24.9
# >  2 REGIONCanada: Ontario                  21.6         21.6
# >  3 LANGNFrench                            18.1         18.1
# >  4 HOMEPOS                                17.5         17.5
# >  5 ESCS                                   16.7         16.7
# >  6 REGIONCanada: British Columbia         16.3         16.3
# >  7 REGIONCanada: Prince Edward Island     15.5         15.5
# >  8 REGIONCanada: Quebec                   15.0         15.0
# >  9 REGIONCanada: Alberta                  14.5         14.5
# > 10 ST004D01TMale                          13.3         13.3

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

# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.4566465 0.4685328 0.4765871 0.4724161 0.4651433 0.4639110 0.4603711 0.4797458 0.4707146 0.4683980
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.4495146 0.4615568 0.4697169 0.4654912 0.4581228 0.4568744 0.4532880 0.4729171 0.4637673 0.4614202
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 268.0709 261.5257 261.1371 259.4670 261.4494 261.4348 261.8163 259.6394 263.1795 261.7631
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 64.02820 67.16407 69.36996 68.21922 66.25563 65.92821 64.99596 70.25370 67.75498 67.12772

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.4682466
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.4612669
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 261.9483
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 67.10977

### ---- Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
tic("Fitting replicate models")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste(pv, "~", paste(voi_all, collapse = " + ")))
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
toc()
# > Fitting replicate models: 28.466 sec elapsed
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
sampling_var_matrix_coef[1:6, ]
# >                    [,1]        [,2]        [,3]        [,4]        [,5]        [,6]        [,7]        [,8]        [,9]       [,10]
# > (Intercept) 228.5882529 178.1646691 205.9836421 152.3526160 190.8561818 144.5114283 198.1206597 194.8629467 202.4531805 177.7693227
# > MATHMOT      49.2173957  40.3713322  37.1424679  46.3079869  37.7201420  34.8565550  40.5225070  37.9224807  38.1805511  37.0748360
# > MATHEASE     21.8947006  21.1168684  17.3991114  17.6693145  19.4808899  20.8575766  21.7202316  22.8073735  18.7268962  22.2469683
# > MATHPREF     19.1644421  18.0453031  16.5441703  17.8035910  17.5703576  16.0377090  18.6636546  19.4354441  14.7237285  21.7338571
# > EXERPRAC      0.1642185   0.1603352   0.1684912   0.1606892   0.1597729   0.1573549   0.1382110   0.1531436   0.1638166   0.1568822
# > STUDYHMW      0.2168256   0.2140478   0.2503199   0.2588889   0.2467693   0.1937997   0.2012945   0.2427429   0.2402904   0.1979687

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 187.3662900  39.9316255  20.3919931  17.9722257   0.1582915   0.2262948 

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 20.77181162  4.34860578  2.69213973  6.35895633  0.03231136  0.06595634  

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 210.2152828  44.7150918  23.3533468  24.9670777   0.1938340   0.2988467

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  14.4988028   6.6869344   4.8325301   4.9967067   0.4402659   0.5466688 

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
# > [1] 0.01502838
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.01522564
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 6.063331
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 4.091744

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
# >                                 Term     Estimate Std. Error      t value Pr(>|t|) t_Signif      z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >                          (Intercept) 496.99707364 14.4988028 34.278490462  < 2e-16      *** 34.278490462  < 2e-16      ***   <NA>     <NA>
# >                              MATHMOT   2.14114463  6.6869344  0.320198240  0.74883           0.320198240 0.748818            <NA>     <NA>
# >                             MATHEASE   4.67735178  4.8325301  0.967888812  0.33314           0.967888812 0.333100            <NA>     <NA>
# >                             MATHPREF   0.42294195  4.9967067  0.084644141  0.93255           0.084644141 0.932544            <NA>     <NA>
# >                             EXERPRAC  -3.06742267  0.4402659 -6.967205501 3.62e-12      *** -6.967205501 3.23e-12      ***   <NA>     <NA>
# >                             STUDYHMW  -0.87196287  0.5466688 -1.595047925  0.11076          -1.595047925 0.110702            <NA>     <NA>
# >                              WORKPAY  -3.59173952  0.5800623 -6.191989390 6.39e-10      *** -6.191989390 5.94e-10      ***   <NA>     <NA>
# >                             WORKHOME  -0.61764640  0.4267779 -1.447231454  0.14789          -1.447231454 0.147832            <NA>     <NA>
# >                              HOMEPOS  17.47078435  3.2446673  5.384461006 7.58e-08      ***  5.384461006 7.27e-08      ***   <NA>     <NA>
# >                               ICTRES -11.30274080  2.3451247 -4.819675893 1.48e-06      *** -4.819675893 1.44e-06      ***   <NA>     <NA>
# >                             INFOSEEK   0.23971572  1.7893023  0.133971615  0.89343           0.133971615 0.893425            <NA>     <NA>
# >                              BULLIED  -4.85947285  1.5340590 -3.167722182  0.00155       ** -3.167722182 0.001536       **   <NA>     <NA>
# >                             FEELSAFE   2.12095539  1.6654618  1.273493889  0.20290           1.273493889 0.202843            <NA>     <NA>
# >                               BELONG  -4.62126766  1.9968753 -2.314249452  0.02069        * -2.314249452 0.020654        *   <NA>     <NA>
# >                              GROSAGR   4.48355830  1.4226383  3.151579882  0.00163       **  3.151579882 0.001624       **   <NA>     <NA>
# >                               ANXMAT  -6.98273322  1.2755864 -5.474136023 4.60e-08      *** -5.474136023 4.40e-08      ***   <NA>     <NA>
# >                              MATHEFF  24.88022661  1.8677759 13.320776973  < 2e-16      *** 13.320776973  < 2e-16      ***   <NA>     <NA>
# >                             MATHEF21   2.69065520  2.4016734  1.120325203  0.26263           1.120325203 0.262575            <NA>     <NA>
# >                             MATHPERS   3.37355158  1.6155943  2.088118072  0.03683        *  2.088118072 0.036787        *   <NA>     <NA>
# >                               FAMCON   6.80899560  1.1537873  5.901430339 3.83e-09      ***  5.901430339 3.60e-09      ***   <NA>     <NA>
# >                             ASSERAGR   3.48249538  1.4325771  2.430930582  0.01509        *  2.430930582 0.015060        *   <NA>     <NA>
# >                              COOPAGR  -1.00940608  1.5277762 -0.660702823  0.50883          -0.660702823 0.508803            <NA>     <NA>
# >                             CURIOAGR   6.29202919  1.5952822  3.944147983 8.11e-05      ***  3.944147983 8.01e-05      ***   <NA>     <NA>
# >                             EMOCOAGR   3.65064428  1.5992847  2.282673145  0.02249        *  2.282673145 0.022450        *   <NA>     <NA>
# >                             EMPATAGR  -0.70265034  1.6526588 -0.425163592  0.67073          -0.425163592 0.670717            <NA>     <NA>
# >                            PERSEVAGR   0.05114772  1.5863382  0.032242636  0.97428           0.032242636 0.974279            <NA>     <NA>
# >                             STRESAGR  -2.34353063  1.6641007 -1.408286556  0.15910          -1.408286556 0.159046            <NA>     <NA>
# >                               EXPOFA  -0.28430590  1.9087196 -0.148951110  0.88160          -0.148951110 0.881592            <NA>     <NA>
# >                             EXPO21ST  -0.07655839  2.0641321 -0.037089869  0.97041          -0.037089869 0.970413            <NA>     <NA>
# >                             COGACRCO   0.67628321  1.6966186  0.398606514  0.69020           0.398606514 0.690183            <NA>     <NA>
# >                             COGACMCO  -4.50097762  1.7210287 -2.615283242  0.00894       ** -2.615283242 0.008915       **   <NA>     <NA>
# >                              DISCLIM   0.23647781  1.6371292  0.144446644  0.88515           0.144446644 0.885148            <NA>     <NA>
# >                               FAMSUP  -3.45903909  1.7719340 -1.952126378  0.05098        . -1.952126378 0.050923        .   <NA>     <NA>
# >                             CREATFAM  -1.09305247  1.8739461 -0.583289187  0.55972          -0.583289187 0.559699            <NA>     <NA>
# >                             CREATSCH  -3.72680489  1.7962621 -2.074755647  0.03806        * -2.074755647 0.038009        *   <NA>     <NA>
# >                             CREATEFF  -7.13070719  1.7895966 -3.984533164 6.85e-05      *** -3.984533164 6.76e-05      ***   <NA>     <NA>
# >                              CREATOP   5.13156026  2.0281841  2.530125517  0.01143        *  2.530125517 0.011402        *   <NA>     <NA>
# >                              IMAGINE  -0.39745334  1.6096222 -0.246923371  0.80498          -0.246923371 0.804968            <NA>     <NA>
# >                              OPENART  -1.53892928  1.6195240 -0.950235576  0.34204          -0.950235576 0.341993            <NA>     <NA>
# >                              CREATAS   1.03602697  1.9394900  0.534174956  0.59324           0.534174956 0.593220            <NA>     <NA>
# >                             CREATOOS  -5.06005619  1.8955815 -2.669395174  0.00762       ** -2.669395174 0.007599       **   <NA>     <NA>
# >                             FAMSUPSL -11.50944667  1.5421220 -7.463382613 9.79e-14      *** -7.463382613 8.43e-14      ***   <NA>     <NA>
# >                              FEELLAH  -3.90572386  1.6337344 -2.390672499  0.01685        * -2.390672499 0.016818        *   <NA>     <NA>
# >                             PROBSELF  -0.65126086  1.5596045 -0.417580771  0.67627          -0.417580771 0.676254            <NA>     <NA>
# >                               SDLEFF   1.27299617  1.8158032  0.701065053  0.48329           0.701065053 0.483262            <NA>     <NA>
# >                              SCHSUST   4.62459060  1.5363411  3.010132796  0.00262       **  3.010132796 0.002611       **   <NA>     <NA>
# >                              LEARRES   0.18797477  1.7333943  0.108443169  0.91365           0.108443169 0.913644            <NA>     <NA>
# >                                 ESCS  16.72374813  2.6350557  6.346639401 2.38e-10      ***  6.346639401 2.20e-10      ***   <NA>     <NA>
# >                               MACTIV   2.33752280  1.1947742  1.956455710  0.05046        .  1.956455710 0.050411        .   <NA>     <NA>
# >                              ABGMATH  -1.23532482  2.7565708 -0.448138255  0.65407          -0.448138255 0.654053            <NA>     <NA>
# >                              MTTRAIN   0.26056483  1.7765981  0.146665034  0.88340           0.146665034 0.883396            <NA>     <NA>
# >                             CREENVSC  -1.49636189  1.9312829 -0.774802034  0.43849          -0.774802034 0.438457            <NA>     <NA>
# >                              OPENCUL   4.71033360  1.8690433  2.520184324  0.01176        *  2.520184324 0.011729        *   <NA>     <NA>
# >                              DIGPREP   0.86488487  1.6499152  0.524199587  0.60016           0.524199587 0.600140            <NA>     <NA>
# >   REGIONCanada: Prince Edward Island  15.52346367 13.1645666  1.179185318  0.23838           1.179185318 0.238324            <NA>     <NA>
# >            REGIONCanada: Nova Scotia   6.07732128  7.6899542  0.790293558  0.42939           0.790293558 0.429356            <NA>     <NA>
# >          REGIONCanada: New Brunswick  -0.29449801  6.3256098 -0.046556462  0.96287          -0.046556462 0.962867            <NA>     <NA>
# >                 REGIONCanada: Quebec  15.03648867  7.0458703  2.134085361  0.03288        *  2.134085361 0.032836        *   <NA>     <NA>
# >                REGIONCanada: Ontario  21.55809902  5.7526263  3.747522939  0.00018      ***  3.747522939 0.000179      ***   <NA>     <NA>
# >               REGIONCanada: Manitoba   0.04091311  6.4556048  0.006337611  0.99494           0.006337611 0.994943            <NA>     <NA>
# >           REGIONCanada: Saskatchewan   5.39943133  6.9644979  0.775279341  0.43821           0.775279341 0.438175            <NA>     <NA>
# >                REGIONCanada: Alberta  14.54105818  6.4545696  2.252831570  0.02431        *  2.252831570 0.024270        *   <NA>     <NA>
# >       REGIONCanada: British Columbia  16.29175492  7.0030646  2.326375088  0.02004        *  2.326375088 0.019999        *   <NA>     <NA>
# >                        ST004D01TMale  13.26424419  3.1344612  4.231746196 2.36e-05      ***  4.231746196 2.32e-05      ***   <NA>     <NA>
# >              ST004D01TNot Applicable   3.15587813 12.8643986  0.245318745  0.80622           0.245318745 0.806210            <NA>     <NA>
# >       IMMIGSecond-Generation student   1.89041228  4.5798909  0.412763610  0.67980           0.412763610 0.679780            <NA>     <NA>
# >        IMMIGFirst-Generation student  -6.89366540  5.0805886 -1.356863536  0.17488          -1.356863536 0.174825            <NA>     <NA>
# >                          LANGNFrench  18.07069808  5.7804440  3.126178188  0.00178       **  3.126178188 0.001771       **   <NA>     <NA>
# >          LANGNAnother language (CAN)  12.58015102  5.7815413  2.175916479  0.02961        *  2.175916479 0.029562        *   <NA>     <NA>
# > SCHLTYPEPrivate Government-dependent   4.55642219 10.0989740  0.451176744  0.65188           0.451176744 0.651862            <NA>     <NA>
# >                       SCHLTYPEPublic  -7.19987675 10.5190684 -0.684459541  0.49371          -0.684459541 0.493685            <NA>     <NA>
# >                            R-squared   0.47000000  0.0200000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                   Adjusted R-squared   0.46000000  0.0200000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                  Residual Std. Error 261.95000000  6.0600000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                          F-statistic  67.11000000  4.0900000           NA     <NA>     <NA>           NA     <NA>     <NA> <2e-16      ***

# --- Sorted table by |estimate|: Report for Inference---
result_table %>%
  mutate(abs_est = abs(Estimate)) %>%
  arrange(desc(abs_est)) %>%
  select(-abs_est) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# > Term     Estimate Std. Error      t value Pr(>|t|) t_Signif      z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >                            (Intercept) 496.99707364 14.4988028 34.278490462  < 2e-16      *** 34.278490462  < 2e-16      ***   <NA>     <NA>
# >                    Residual Std. Error 261.95000000  6.0600000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                            F-statistic  67.11000000  4.0900000           NA     <NA>     <NA>           NA     <NA>     <NA> <2e-16      ***
# >                                MATHEFF  24.88022661  1.8677759 13.320776973  < 2e-16      *** 13.320776973  < 2e-16      ***   <NA>     <NA>
# >                  REGIONCanada: Ontario  21.55809902  5.7526263  3.747522939  0.00018      ***  3.747522939 0.000179      ***   <NA>     <NA>
# >                            LANGNFrench  18.07069808  5.7804440  3.126178188  0.00178       **  3.126178188 0.001771       **   <NA>     <NA>
# >                                HOMEPOS  17.47078435  3.2446673  5.384461006 7.58e-08      ***  5.384461006 7.27e-08      ***   <NA>     <NA>
# >                                   ESCS  16.72374813  2.6350557  6.346639401 2.38e-10      ***  6.346639401 2.20e-10      ***   <NA>     <NA>
# >         REGIONCanada: British Columbia  16.29175492  7.0030646  2.326375088  0.02004        *  2.326375088 0.019999        *   <NA>     <NA>
# >     REGIONCanada: Prince Edward Island  15.52346367 13.1645666  1.179185318  0.23838           1.179185318 0.238324            <NA>     <NA>
# >                   REGIONCanada: Quebec  15.03648867  7.0458703  2.134085361  0.03288        *  2.134085361 0.032836        *   <NA>     <NA>
# >                  REGIONCanada: Alberta  14.54105818  6.4545696  2.252831570  0.02431        *  2.252831570 0.024270        *   <NA>     <NA>
# >                          ST004D01TMale  13.26424419  3.1344612  4.231746196 2.36e-05      ***  4.231746196 2.32e-05      ***   <NA>     <NA>
# >            LANGNAnother language (CAN)  12.58015102  5.7815413  2.175916479  0.02961        *  2.175916479 0.029562        *   <NA>     <NA>
# >                               FAMSUPSL -11.50944667  1.5421220 -7.463382613 9.79e-14      *** -7.463382613 8.43e-14      ***   <NA>     <NA>
# >                                 ICTRES -11.30274080  2.3451247 -4.819675893 1.48e-06      *** -4.819675893 1.44e-06      ***   <NA>     <NA>
# >                         SCHLTYPEPublic  -7.19987675 10.5190684 -0.684459541  0.49371          -0.684459541 0.493685            <NA>     <NA>
# >                               CREATEFF  -7.13070719  1.7895966 -3.984533164 6.85e-05      *** -3.984533164 6.76e-05      ***   <NA>     <NA>
# >                                 ANXMAT  -6.98273322  1.2755864 -5.474136023 4.60e-08      *** -5.474136023 4.40e-08      ***   <NA>     <NA>
# >          IMMIGFirst-Generation student  -6.89366540  5.0805886 -1.356863536  0.17488          -1.356863536 0.174825            <NA>     <NA>
# >                                 FAMCON   6.80899560  1.1537873  5.901430339 3.83e-09      ***  5.901430339 3.60e-09      ***   <NA>     <NA>
# >                               CURIOAGR   6.29202919  1.5952822  3.944147983 8.11e-05      ***  3.944147983 8.01e-05      ***   <NA>     <NA>
# >              REGIONCanada: Nova Scotia   6.07732128  7.6899542  0.790293558  0.42939           0.790293558 0.429356            <NA>     <NA>
# >             REGIONCanada: Saskatchewan   5.39943133  6.9644979  0.775279341  0.43821           0.775279341 0.438175            <NA>     <NA>
# >                                CREATOP   5.13156026  2.0281841  2.530125517  0.01143        *  2.530125517 0.011402        *   <NA>     <NA>
# >                               CREATOOS  -5.06005619  1.8955815 -2.669395174  0.00762       ** -2.669395174 0.007599       **   <NA>     <NA>
# >                                BULLIED  -4.85947285  1.5340590 -3.167722182  0.00155       ** -3.167722182 0.001536       **   <NA>     <NA>
# >                                OPENCUL   4.71033360  1.8690433  2.520184324  0.01176        *  2.520184324 0.011729        *   <NA>     <NA>
# >                               MATHEASE   4.67735178  4.8325301  0.967888812  0.33314           0.967888812 0.333100            <NA>     <NA>
# >                                SCHSUST   4.62459060  1.5363411  3.010132796  0.00262       **  3.010132796 0.002611       **   <NA>     <NA>
# >                                 BELONG  -4.62126766  1.9968753 -2.314249452  0.02069        * -2.314249452 0.020654        *   <NA>     <NA>
# >   SCHLTYPEPrivate Government-dependent   4.55642219 10.0989740  0.451176744  0.65188           0.451176744 0.651862            <NA>     <NA>
# >                               COGACMCO  -4.50097762  1.7210287 -2.615283242  0.00894       ** -2.615283242 0.008915       **   <NA>     <NA>
# >                                GROSAGR   4.48355830  1.4226383  3.151579882  0.00163       **  3.151579882 0.001624       **   <NA>     <NA>
# >                                FEELLAH  -3.90572386  1.6337344 -2.390672499  0.01685        * -2.390672499 0.016818        *   <NA>     <NA>
# >                               CREATSCH  -3.72680489  1.7962621 -2.074755647  0.03806        * -2.074755647 0.038009        *   <NA>     <NA>
# >                               EMOCOAGR   3.65064428  1.5992847  2.282673145  0.02249        *  2.282673145 0.022450        *   <NA>     <NA>
# >                                WORKPAY  -3.59173952  0.5800623 -6.191989390 6.39e-10      *** -6.191989390 5.94e-10      ***   <NA>     <NA>
# >                               ASSERAGR   3.48249538  1.4325771  2.430930582  0.01509        *  2.430930582 0.015060        *   <NA>     <NA>
# >                                 FAMSUP  -3.45903909  1.7719340 -1.952126378  0.05098        . -1.952126378 0.050923        .   <NA>     <NA>
# >                               MATHPERS   3.37355158  1.6155943  2.088118072  0.03683        *  2.088118072 0.036787        *   <NA>     <NA>
# >                ST004D01TNot Applicable   3.15587813 12.8643986  0.245318745  0.80622           0.245318745 0.806210            <NA>     <NA>
# >                               EXERPRAC  -3.06742267  0.4402659 -6.967205501 3.62e-12      *** -6.967205501 3.23e-12      ***   <NA>     <NA>
# >                               MATHEF21   2.69065520  2.4016734  1.120325203  0.26263           1.120325203 0.262575            <NA>     <NA>
# >                               STRESAGR  -2.34353063  1.6641007 -1.408286556  0.15910          -1.408286556 0.159046            <NA>     <NA>
# >                                 MACTIV   2.33752280  1.1947742  1.956455710  0.05046        .  1.956455710 0.050411        .   <NA>     <NA>
# >                                MATHMOT   2.14114463  6.6869344  0.320198240  0.74883           0.320198240 0.748818            <NA>     <NA>
# >                               FEELSAFE   2.12095539  1.6654618  1.273493889  0.20290           1.273493889 0.202843            <NA>     <NA>
# >         IMMIGSecond-Generation student   1.89041228  4.5798909  0.412763610  0.67980           0.412763610 0.679780            <NA>     <NA>
# >                                OPENART  -1.53892928  1.6195240 -0.950235576  0.34204          -0.950235576 0.341993            <NA>     <NA>
# >                               CREENVSC  -1.49636189  1.9312829 -0.774802034  0.43849          -0.774802034 0.438457            <NA>     <NA>
# >                                 SDLEFF   1.27299617  1.8158032  0.701065053  0.48329           0.701065053 0.483262            <NA>     <NA>
# >                                ABGMATH  -1.23532482  2.7565708 -0.448138255  0.65407          -0.448138255 0.654053            <NA>     <NA>
# >                               CREATFAM  -1.09305247  1.8739461 -0.583289187  0.55972          -0.583289187 0.559699            <NA>     <NA>
# >                                CREATAS   1.03602697  1.9394900  0.534174956  0.59324           0.534174956 0.593220            <NA>     <NA>
# >                                COOPAGR  -1.00940608  1.5277762 -0.660702823  0.50883          -0.660702823 0.508803            <NA>     <NA>
# >                               STUDYHMW  -0.87196287  0.5466688 -1.595047925  0.11076          -1.595047925 0.110702            <NA>     <NA>
# >                                DIGPREP   0.86488487  1.6499152  0.524199587  0.60016           0.524199587 0.600140            <NA>     <NA>
# >                               EMPATAGR  -0.70265034  1.6526588 -0.425163592  0.67073          -0.425163592 0.670717            <NA>     <NA>
# >                               COGACRCO   0.67628321  1.6966186  0.398606514  0.69020           0.398606514 0.690183            <NA>     <NA>
# >                               PROBSELF  -0.65126086  1.5596045 -0.417580771  0.67627          -0.417580771 0.676254            <NA>     <NA>
# >                               WORKHOME  -0.61764640  0.4267779 -1.447231454  0.14789          -1.447231454 0.147832            <NA>     <NA>
# >                              R-squared   0.47000000  0.0200000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                     Adjusted R-squared   0.46000000  0.0200000           NA     <NA>     <NA>           NA     <NA>     <NA>   <NA>     <NA>
# >                               MATHPREF   0.42294195  4.9967067  0.084644141  0.93255           0.084644141 0.932544            <NA>     <NA>
# >                                IMAGINE  -0.39745334  1.6096222 -0.246923371  0.80498          -0.246923371 0.804968            <NA>     <NA>
# >            REGIONCanada: New Brunswick  -0.29449801  6.3256098 -0.046556462  0.96287          -0.046556462 0.962867            <NA>     <NA>
# >                                 EXPOFA  -0.28430590  1.9087196 -0.148951110  0.88160          -0.148951110 0.881592            <NA>     <NA>
# >                                MTTRAIN   0.26056483  1.7765981  0.146665034  0.88340           0.146665034 0.883396            <NA>     <NA>
# >                               INFOSEEK   0.23971572  1.7893023  0.133971615  0.89343           0.133971615 0.893425            <NA>     <NA>
# >                                DISCLIM   0.23647781  1.6371292  0.144446644  0.88515           0.144446644 0.885148            <NA>     <NA>
# >                                LEARRES   0.18797477  1.7333943  0.108443169  0.91365           0.108443169 0.913644            <NA>     <NA>
# >                               EXPO21ST  -0.07655839  2.0641321 -0.037089869  0.97041          -0.037089869 0.970413            <NA>     <NA>
# >                              PERSEVAGR   0.05114772  1.5863382  0.032242636  0.97428           0.032242636 0.974279            <NA>     <NA>
# >                 REGIONCanada: Manitoba   0.04091311  6.4556048  0.006337611  0.99494           0.006337611 0.994943            <NA>     <NA>

# -- Visualize coefs: ADD bars + 95% CI from Rubin+BRR SEs ---
ggplot(
  coef_df %>%
    left_join(enframe(se_final_coef, name = "Term", value = "SE"), by = "Term") %>%
    mutate(
      lo95 = Estimate - z_crit * SE,
      hi95 = Estimate + z_crit * SE,
      Term = fct_reorder(Term, mag)  # order by |Estimate|
    ),
  aes(x = Term, y = Estimate, fill = sign)
) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.6, alpha = 0.6) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_errorbar(aes(ymin = lo95, ymax = hi95), width = 0.25, linewidth = 0.4) +
  coord_flip() +
  scale_y_continuous(limits = c(-max_abs, max_abs)) +  # negative left, positive right
  scale_fill_manual(values = c("Negative" = "#b2cbea", "Positive" = "#2b6cb0")) +
  labs(title = "Pooled regression coefficients with 95% CIs (ordered by |estimate|)",
       x = NULL, y = "Estimate", fill = NULL) +
  theme_minimal() +
  theme(legend.position = "top")


### ---- Compare results with intsvy package  ----
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=voi_all, data=train_data)
# >                                      Estimate Std. Error t value
# > (Intercept)                            497.00      14.50   34.28
# > MATHMOT                                  2.14       6.69    0.32
# > MATHEASE                                 4.68       4.83    0.97
# > MATHPREF                                 0.42       5.00    0.08
# > EXERPRAC                                -3.07       0.44   -6.97
# > STUDYHMW                                -0.87       0.55   -1.60
# > WORKPAY                                 -3.59       0.58   -6.19
# > WORKHOME                                -0.62       0.43   -1.45
# > HOMEPOS                                 17.47       3.24    5.38
# > ICTRES                                 -11.30       2.35   -4.82
# > INFOSEEK                                 0.24       1.79    0.13
# > BULLIED                                 -4.86       1.53   -3.17
# > FEELSAFE                                 2.12       1.67    1.27
# > BELONG                                  -4.62       2.00   -2.31
# > GROSAGR                                  4.48       1.42    3.15
# > ANXMAT                                  -6.98       1.28   -5.47
# > MATHEFF                                 24.88       1.87   13.32
# > MATHEF21                                 2.69       2.40    1.12
# > MATHPERS                                 3.37       1.62    2.09
# > FAMCON                                   6.81       1.15    5.90
# > ASSERAGR                                 3.48       1.43    2.43
# > COOPAGR                                 -1.01       1.53   -0.66
# > CURIOAGR                                 6.29       1.60    3.94
# > EMOCOAGR                                 3.65       1.60    2.28
# > EMPATAGR                                -0.70       1.65   -0.43
# > PERSEVAGR                                0.05       1.59    0.03
# > STRESAGR                                -2.34       1.66   -1.41
# > EXPOFA                                  -0.28       1.91   -0.15
# > EXPO21ST                                -0.08       2.06   -0.04
# > COGACRCO                                 0.68       1.70    0.40
# > COGACMCO                                -4.50       1.72   -2.62
# > DISCLIM                                  0.24       1.64    0.14
# > FAMSUP                                  -3.46       1.77   -1.95
# > CREATFAM                                -1.09       1.87   -0.58
# > CREATSCH                                -3.73       1.80   -2.07
# > CREATEFF                                -7.13       1.79   -3.98
# > CREATOP                                  5.13       2.03    2.53
# > IMAGINE                                 -0.40       1.61   -0.25
# > OPENART                                 -1.54       1.62   -0.95
# > CREATAS                                  1.04       1.94    0.53
# > CREATOOS                                -5.06       1.90   -2.67
# > FAMSUPSL                               -11.51       1.54   -7.46
# > FEELLAH                                 -3.91       1.63   -2.39
# > PROBSELF                                -0.65       1.56   -0.42
# > SDLEFF                                   1.27       1.82    0.70
# > SCHSUST                                  4.62       1.54    3.01
# > LEARRES                                  0.19       1.73    0.11
# > ESCS                                    16.72       2.64    6.35
# > MACTIV                                   2.34       1.19    1.96
# > ABGMATH                                 -1.24       2.76   -0.45
# > MTTRAIN                                  0.26       1.78    0.15
# > CREENVSC                                -1.50       1.93   -0.77
# > OPENCUL                                  4.71       1.87    2.52
# > DIGPREP                                  0.86       1.65    0.52
# > REGIONCanada: Prince Edward Island      15.52      13.16    1.18
# > REGIONCanada: Nova Scotia                6.08       7.69    0.79
# > REGIONCanada: New Brunswick             -0.29       6.33   -0.05
# > REGIONCanada: Quebec                    15.04       7.05    2.13
# > REGIONCanada: Ontario                   21.56       5.75    3.75
# > REGIONCanada: Manitoba                   0.04       6.46    0.01
# > REGIONCanada: Saskatchewan               5.40       6.96    0.78
# > REGIONCanada: Alberta                   14.54       6.45    2.25
# > REGIONCanada: British Columbia          16.29       7.00    2.33
# > ST004D01TMale                           13.26       3.13    4.23
# > ST004D01TNot Applicable                  3.16      12.86    0.25
# > IMMIGSecond-Generation student           1.89       4.58    0.41
# > IMMIGFirst-Generation student           -6.89       5.08   -1.36
# > LANGNFrench                             18.07       5.78    3.13
# > LANGNAnother language (CAN)             12.58       5.78    2.18
# > SCHLTYPEPrivate Government-dependent     4.56      10.10    0.45
# > SCHLTYPEPublic                          -7.20      10.52   -0.68
# > R-squared                                0.47       0.02   31.16

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
# >        rmse      mae          bias bias_pct        r2
# > 1  66.19353 52.28321 -1.708357e-12 1.756205 0.4566465
# > 2  64.57735 51.45576 -8.126883e-13 1.679805 0.4685328
# > 3  64.48139 51.26937  6.792947e-13 1.680185 0.4765871
# > 4  64.06900 51.06501 -7.684951e-13 1.641460 0.4724161
# > 5  64.55852 51.28514 -4.036661e-12 1.646603 0.4651433
# > 6  64.55492 50.68159 -4.195477e-13 1.664192 0.4639110
# > 7  64.64910 51.30963 -1.662940e-12 1.683454 0.4603711
# > 8  64.11158 50.74900 -9.980214e-13 1.668747 0.4797458
# > 9  64.98571 51.64116 -7.661090e-13 1.698846 0.4707146
# > 10 64.63598 51.42302 -8.423487e-13 1.678582 0.4683980

# Obtain final mean estimate 
main_metrics <- colMeans(train_metrics_main)
main_metrics
# >         rmse           mae          bias      bias_pct            r2
# > 6.468171e+01  5.131629e+01 -1.133587e-12  1.679808e+00  4.682466e-01  

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
# > rmse     9.718301e-01 8.616860e-01 7.938971e-01 7.835652e-01 9.518941e-01 8.057561e-01 8.398649e-01 7.967019e-01 8.298037e-01 8.872743e-01
# > mae      6.070917e-01 6.030670e-01 5.717431e-01 5.865017e-01 6.788021e-01 5.245598e-01 6.318187e-01 5.282576e-01 5.949367e-01 6.891471e-01
# > bias     3.342841e-23 1.826414e-23 1.144530e-23 1.821856e-23 9.324914e-23 1.641356e-23 2.710752e-23 2.261254e-23 1.871944e-23 1.844538e-23
# > bias_pct 3.117593e-03 2.520460e-03 2.866103e-03 2.415472e-03 2.737783e-03 2.436065e-03 2.735630e-03 3.056803e-03 2.837068e-03 3.173597e-03
# > r2       1.726556e-04 1.454222e-04 1.685408e-04 1.814620e-04 1.329098e-04 1.685961e-04 1.725877e-04 2.009477e-04 1.896781e-04 1.752453e-04

# <=> Equivalent codes
# sampling_var_matrix <- sapply(1:M, function(m) {
#   sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |> colMeans() / (1 - k)^2
# }) 
# sampling_var_matrix

sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 8.522273e-01 6.015925e-01 2.779040e-23 2.789657e-03 1.708045e-04  

# Imputation variance
imputation_var <- colSums((train_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 3.515437e-01 2.069276e-01 1.478812e-24 1.017034e-03 5.004345e-05 

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 1.238925e+00 8.292129e-01 2.941709e-23 3.908394e-03 2.258523e-04

# Final standard error
se_final <- sqrt(var_final)
se_final
# >         rmse          mae         bias     bias_pct           r2 
# > 1.113070e+00 9.106113e-01 5.423753e-12 6.251715e-02 1.502838e-02 

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
# >    Metric        Point_estimate       Standard_error             CI_lower              CI_upper           CI_length
# >      RMSE 64.681708312814194528 1.113070246303455413 62.50013071779629570 66.863285907832093358 4.36315519003579766
# >       MAE 51.316290087352832927 0.910611273747204852 49.53152478689217020 53.101055387813495656 3.56953060092132546
# >      Bias -0.000000000001133587 0.000000000005423753 -0.00000000001176395  0.000000000009496772 0.00000000002126072
# >     Bias%  1.679807819337236552 0.062517150600298804  1.55727645574458440  1.802339182929888706 0.24506272718530431
# > R-squared  0.468246624621800778 0.015028384702499646  0.43879153185908881  0.497701717384512743 0.05891018552542393

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=voi_all, data=train_data)

## ---- Predict and evaluate on the validation data (Weighted, Rubin + BRR) ----

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
# >        rmse      mae       bias bias_pct        r2
# > 1  68.73864 53.62992  2.590497 2.492438 0.4475643
# > 2  68.97261 54.58889 -0.331490 1.896150 0.4564004
# > 3  69.18272 54.14733  3.654444 2.725319 0.4566719
# > 4  66.60331 52.94375 -1.357211 1.499773 0.4676010
# > 5  67.67914 52.95962  4.113547 2.689925 0.4593932
# > 6  68.27290 54.93651  3.624939 2.572634 0.4433006
# > 7  70.19869 56.28153  2.136108 2.448391 0.4354094
# > 8  69.35396 53.86871  1.903173 2.422942 0.4653665
# > 9  67.10573 52.34728  3.579128 2.480566 0.4522891
# > 10 67.13796 53.77034  1.330229 2.156756 0.4772342

# Obtain final mean estimate 
main_metrics <- colMeans(valid_metrics_main)
main_metrics
# >       rmse        mae       bias   bias_pct         r2 
# > 68.3245659 53.9473870  2.1243364  2.3384895  0.4561231 

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
# > rmse      6.601347937  6.071147797  8.450117413 5.482866e+00  7.03810941  7.377872281  6.823573364  8.45123006  7.01437386  7.311277138
# > mae       4.473366175  4.797855472  5.623933218 4.111850e+00  6.49063586  5.462090009  4.308796376  5.29022876  5.38162841  4.535935782
# > bias     16.710779285 14.091923584 15.455229291 1.496775e+01 18.75158355 14.733793307 16.711875697 14.26072519 14.92302132 15.395508051
# > bias_pct  0.785597706  0.561829229  0.732749188 5.420120e-01  0.80069273  0.584473566  0.639799257  0.66674610  0.60734337  0.638043244
# > r2        0.001264835  0.001066204  0.001367532 9.230537e-04  0.00146583  0.001369598  0.001355366  0.00132136  0.00128888  0.001326077
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >        rmse          mae         bias     bias_pct          r2 
# > 7.062191478  5.047631968 15.600219122  0.655928639  0.001274874 

# Imputation variance
imputation_var <- colSums((valid_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 1.3528208218 1.2844172601 3.3017366857 0.1470223028 0.0001505955

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >        rmse          mae         bias     bias_pct           r2 
# > 8.550294382  6.460490954 19.232129476  0.817653172  0.001440529 

# Final standard error
se_final <- sqrt(var_final)
se_final
# >      rmse       mae      bias  bias_pct        r2 
# > 2.9240886 2.5417496 4.3854452 0.9042418 0.0379543 

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
# >     Metric Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >       RMSE     68.3245659      2.9240886 62.5934575 74.0556743 11.4622168
# >        MAE     53.9473870      2.5417496 48.9656494 58.9291246  9.9634753
# >       Bias      2.1243364      4.3854452 -6.4709783 10.7196510 17.1906292
# >      Bias%      2.3384895      0.9042418  0.5662082  4.1107708  3.5445626
# >  R-squared      0.4561231      0.0379543  0.3817340  0.5305121  0.1487781

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
# >        rmse      mae      bias  bias_pct        r2
# > 1  68.40399 54.87567 -4.6986994 0.8017878 0.4030221
# > 2  68.76593 54.93946 -1.8891204 1.4043130 0.3880518
# > 3  72.29676 57.68133 -2.7106206 1.5186148 0.3936901
# > 4  71.90423 55.90967 -2.1116624 1.5737882 0.3865613
# > 5  69.45961 54.42856 -1.6494107 1.5512543 0.4135110
# > 6  70.95728 57.20560 -0.1652165 1.9589470 0.3968159
# > 7  71.52639 55.14504 -1.2741834 1.7214479 0.3883490
# > 8  66.96281 53.53000 -1.8571139 1.3264199 0.4356243
# > 9  69.78436 55.90323 -3.0120634 1.3237838 0.4273607
# > 10 67.08284 53.84864 -4.7986007 0.6628395 0.4015306

# Obtain final mean estimate 
main_metrics <- colMeans(test_metrics_main)
main_metrics
# >       rmse        mae       bias   bias_pct         r2 
# > 69.7144190 55.3467190 -2.4166691  1.3843196  0.4034517 

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
# >                  [,1]         [,2]        [,3]         [,4]         [,5]       [,6]        [,7]         [,8]        [,9]        [,10]
# > rmse      6.743908573  6.985247095  8.24829026 11.982814116  8.688838996  7.5444390  9.76627749  7.333703659  8.72336622  5.707335200
# > mae       4.801262011  4.948125288  6.04049361  7.542195778  6.655968726  5.2462415  6.27880613  5.558047727  6.91603347  4.532954665
# > bias     11.832936557 12.421905619 14.61332304 14.942843003 12.316220463 12.9863662 13.99886954 10.539711247 12.29508346 11.113733232
# > bias_pct  0.488097120  0.550389053  0.69259195  0.634393376  0.564900957  0.5730151  0.56616356  0.455202951  0.55529462  0.431333219
# > r2        0.001526611  0.001389794  0.00104191  0.001625828  0.001558948  0.0015433  0.00160525  0.001536262  0.00165679  0.001408386
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >       rmse         mae        bias    bias_pct          r2 
# > 8.172422059  5.852012887 12.706099238  0.551138195  0.001489308 

# Imputation variance
imputation_var <- colSums((test_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 3.7194387443 1.8165131296 2.1077332363 0.1547718249 0.0002890707 

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2
# > 12.263804678  7.850177330 15.024605798  0.721387202  0.001807286 

# Final standard error
se_final <- sqrt(var_final)
se_final
# >       rmse        mae       bias   bias_pct         r2 
# > 3.50197154 2.80181679 3.87615864 0.84934516 0.04251218  

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
# >    Metric Point_estimate Standard_error    CI_lower   CI_upper  CI_length
# >      RMSE     69.7144190     3.50197154  62.8506809 76.578157 13.7274762
# >       MAE     55.3467190     2.80181679  49.8552590 60.838179 10.9829200
# >      Bias     -2.4166691     3.87615864 -10.0138005  5.180462 15.1942627
# >     Bias%      1.3843196     0.84934516  -0.2803663  3.049006  3.3293719
# > R-squared      0.4034517     0.04251218   0.3201293  0.486774  0.1666447

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
# >    Metric        Point_estimate       Standard_error             CI_lower              CI_upper           CI_length
# >      RMSE 64.681708312814194528 1.113070246303455413 62.50013071779629570 66.863285907832093358 4.36315519003579766
# >       MAE 51.316290087352832927 0.910611273747204852 49.53152478689217020 53.101055387813495656 3.56953060092132546
# >      Bias -0.000000000001133587 0.000000000005423753 -0.00000000001176395  0.000000000009496772 0.00000000002126072
# >     Bias%  1.679807819337236552 0.062517150600298804  1.55727645574458440  1.802339182929888706 0.24506272718530431
# > R-squared  0.468246624621800778 0.015028384702499646  0.43879153185908881  0.497701717384512743 0.05891018552542393

print(as.data.frame(valid_eval), row.names = FALSE)
# >     Metric Point_estimate Standard_error   CI_lower   CI_upper  CI_length
# >       RMSE     68.3245659      2.9240886 62.5934575 74.0556743 11.4622168
# >        MAE     53.9473870      2.5417496 48.9656494 58.9291246  9.9634753
# >       Bias      2.1243364      4.3854452 -6.4709783 10.7196510 17.1906292
# >      Bias%      2.3384895      0.9042418  0.5662082  4.1107708  3.5445626
# >  R-squared      0.4561231      0.0379543  0.3817340  0.5305121  0.1487781

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error    CI_lower   CI_upper  CI_length
# >      RMSE     69.7144190     3.50197154  62.8506809 76.578157 13.7274762
# >       MAE     55.3467190     2.80181679  49.8552590 60.838179 10.9829200
# >      Bias     -2.4166691     3.87615864 -10.0138005  5.180462 15.1942627
# >     Bias%      1.3843196     0.84934516  -0.2803663  3.049006  3.3293719
# > R-squared      0.4034517     0.04251218   0.3201293  0.486774  0.1666447

# --- Keep only four decimals ---
# library(dplyr)
# library(readr)   # for parse_number

train_eval %>%
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        64.6817         1.1131  62.5001  66.8633    4.3632
# >       MAE        51.3163         0.9106  49.5315  53.1011    3.5695
# >      Bias        -0.0000         0.0000  -0.0000   0.0000    0.0000
# >     Bias%         1.6798         0.0625   1.5573   1.8023    0.2451
# > R-squared         0.4682         0.0150   0.4388   0.4977    0.0589

valid_eval %>%
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        68.3246         2.9241  62.5935  74.0557   11.4622
# >       MAE        53.9474         2.5417  48.9656  58.9291    9.9635
# >      Bias         2.1243         4.3854  -6.4710  10.7197   17.1906
# >     Bias%         2.3385         0.9042   0.5662   4.1108    3.5446
# > R-squared         0.4561         0.0380   0.3817   0.5305    0.1488

test_eval %>%
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        69.7144         3.5020  62.8507  76.5782   13.7275
# >       MAE        55.3467         2.8018  49.8553  60.8382   10.9829
# >      Bias        -2.4167         3.8762 -10.0138   5.1805   15.1943
# >     Bias%         1.3843         0.8493  -0.2804   3.0490    3.3294
# > R-squared         0.4035         0.0425   0.3201   0.4868    0.1666

# --- Keep only two decimals ---
train_eval %>%
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          64.68           1.11    62.50    66.86      4.36
# >       MAE          51.32           0.91    49.53    53.10      3.57
# >      Bias          -0.00           0.00    -0.00     0.00      0.00
# >     Bias%           1.68           0.06     1.56     1.80      0.25
# > R-squared           0.47           0.02     0.44     0.50      0.06

valid_eval %>%
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# > Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          68.32           2.92    62.59    74.06     11.46
# >       MAE          53.95           2.54    48.97    58.93      9.96
# >      Bias           2.12           4.39    -6.47    10.72     17.19
# >     Bias%           2.34           0.90     0.57     4.11      3.54
# > R-squared           0.46           0.04     0.38     0.53      0.15

test_eval %>%
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          69.71           3.50    62.85    76.58     13.73
# >       MAE          55.35           2.80    49.86    60.84     10.98
# >      Bias          -2.42           3.88   -10.01     5.18     15.19
# >     Bias%           1.38           0.85    -0.28     3.05      3.33
# > R-squared           0.40           0.04     0.32     0.49      0.17

# --- Export to word---
library(officer); library(flextable)

fmt2 <- \(df) df %>%
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  )

performance_tables_lm_voi_all <- read_docx() %>%
  body_add_par("Performance - Linear Regression (lm, voi_all)", style = "Normal") %>%  # style = c('Normal', 'heading 1', 'heading 2', 'heading 3', 'centered', 'Image Caption', 'Table Caption', 'toc 1', 'toc 2', 'Balloon Text', 'graphic title', 'table title')
  body_add_par("Training performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(train_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Validation performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(valid_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Test performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(test_eval))  %>% align(j = 1:6, align = "right", part = "all") %>% autofit())

print(performance_tables_lm_voi_all, target = "performance_tables_lm_voi_all.docx")



