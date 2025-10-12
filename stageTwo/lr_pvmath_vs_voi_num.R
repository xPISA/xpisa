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
#   X: voi_num
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

### ---- Sanity check (post-prep) ----

#### ---- Dimension ----
dim(pisa_2022_canada_merged); dim(temp_data)
# > [1] 23073  1699
# > [1] 6944  146

#### ---- Missingness ----
all(
  colSums(is.na(pisa_2022_canada_merged[pvmaths])) == 0,
  sum(is.na(pisa_2022_canada_merged$W_FSTUWT)) == 0,
  colSums(is.na(pisa_2022_canada_merged[paste0("W_FSTURWT", 1:80)])) == 0
)
# > TRUE
sapply(pisa_2022_canada_merged[, voi_num, drop=FALSE], \(x) sum(is.na(x)))  
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >    4150      4168      4123      2912      2887      2998      2944      1402      1424      5720      2973      2865      2942      4067      4239      4660      5014      4891 
# > FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP  CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >    4856      4300      3923      3997      4236      4055      3809      4294      4803      5072      4654      4878      3534      5434      5195      5197      4949      5108 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >    5329      5231      5735      6673      8525     12471      8478      8684      7666      8460      1677      2910      2922      2575      3036      3096      3008 

sapply(temp_data[, voi_num, drop=FALSE], \(x) sum(is.na(x)))  # expect 0 if we intended to drop missing
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 
# > FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP   CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >       0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0 

#### ---- Data type/variable class  ----

# For plausible values in mathematics
sapply(pisa_2022_canada_merged[pvmaths], class)           # All numeric
sapply(temp_data[pvmaths], class)           # All numeric

# For final student weight 
sapply(pisa_2022_canada_merged[final_wt], class)          # All numeric
sapply(temp_data[final_wt], class)          # All numeric

# For BRR replicate weights 
sapply(pisa_2022_canada_merged[rep_wts], class)           # All numeric
sapply(temp_data[rep_wts], class)           # All numeric

# For predictors 
all(sapply(pisa_2022_canada_merged[, voi_num, drop=FALSE], is.numeric))
# > [1] TRUE
sapply(temp_data[, voi_num, drop=FALSE], is.numeric)  # numeric predictors are truly numeric in temp_data
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 
# >  FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP  CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 

#### ---- Summary Statistics ----

## -- pisa_2022_canada_merged --
# For plausible values in mathematics
rbind(sapply(temp_data[pvmaths], summary), `Std. Dev.` = sapply(temp_data[pvmaths], sd))
# >             PV1MATH   PV2MATH  PV3MATH   PV4MATH   PV5MATH   PV6MATH   PV7MATH   PV8MATH  PV9MATH  PV10MATH
# > Min.      195.11800 184.89700 177.8010 220.47600 195.81300 226.94800 211.45100 206.45700 202.3520 189.22600
# > 1st Qu.   445.98550 448.79950 446.4312 448.13225 447.52675 444.76525 447.60200 445.75375 446.6712 447.73875
# > Median    508.26500 508.86850 507.3615 508.27900 507.92250 506.53700 507.96050 507.60400 507.5785 508.87400
# > Mean      509.47861 510.15031 509.3026 509.38227 508.69354 507.54532 508.92247 508.72593 508.7861 509.34000
# > 3rd Qu.   570.77850 572.06750 569.6672 569.43025 567.75325 569.03925 568.68300 570.12475 570.1745 569.40300
# > Max.      842.35800 875.27500 830.5470 836.22900 860.15500 885.77800 803.36800 802.42700 849.6370 903.00900
# > Std. Dev.  89.71491  89.74231  89.5981  88.72675  89.34575  89.30593  88.37921  89.63438  89.9106  88.37838

# For final student weight
rbind(sapply(temp_data[final_wt], summary), `Std. Dev.` = sapply(temp_data[final_wt], sd))
# >            W_FSTUWT
# > Min.        1.04731
# > 1st Qu.     4.28074
# > Median     10.00889
# > Mean       16.14551
# > 3rd Qu.    26.29699
# > Max.      132.18970
# > Std. Dev.  13.79908

# For BRR replicate weights
rbind(sapply(temp_data[rep_wts], summary), `Std. Dev.` = sapply(temp_data[rep_wts], sd))

# For predictors 
rbind(sapply(temp_data[voi_num], summary), `Std. Dev.` = sapply(temp_data[voi_num], sd))

## -- temp_data --
# For plausible values in mathematics
# sapply(temp_data[pvmaths], summary)
# sapply(temp_data[pvmaths], sd)
rbind(sapply(temp_data[pvmaths], summary), `Std. Dev.` = sapply(temp_data[pvmaths], sd))
# >             PV1MATH   PV2MATH   PV3MATH   PV4MATH  PV5MATH   PV6MATH   PV7MATH  PV8MATH   PV9MATH  PV10MATH
# > Min.      195.11800 184.89700 177.8010 220.47600 195.81300 226.94800 211.45100 206.45700 202.3520 189.22600
# > 1st Qu.   445.98550 448.79950 446.4312 448.13225 447.52675 444.76525 447.60200 445.75375 446.6712 447.73875
# > Median    508.26500 508.86850 507.3615 508.27900 507.92250 506.53700 507.96050 507.60400 507.5785 508.87400
# > Mean      509.47861 510.15031 509.3026 509.38227 508.69354 507.54532 508.92247 508.72593 508.7861 509.34000
# > 3rd Qu.   570.77850 572.06750 569.6672 569.43025 567.75325 569.03925 568.68300 570.12475 570.1745 569.40300
# > Max.      842.35800 875.27500 830.5470 836.22900 860.15500 885.77800 803.36800 802.42700 849.6370 903.00900
# > Std. Dev.  89.71491  89.74231  89.5981  88.72675  89.34575  89.30593  88.37921  89.63438  89.9106  88.37838

# For final student weight
rbind(sapply(temp_data[final_wt], summary), `Std. Dev.` = sapply(temp_data[final_wt], sd))
# >            W_FSTUWT
# > Min.        1.04731
# > 1st Qu.     4.28074
# > Median     10.00889
# > Mean       16.14551
# > 3rd Qu.    26.29699
# > Max.      132.18970
# > Std. Dev.  13.79908

# For BRR replicate weights
rbind(sapply(temp_data[rep_wts], summary), `Std. Dev.` = sapply(temp_data[rep_wts], sd))

# For predictors 
rbind(sapply(temp_data[voi_num], summary), `Std. Dev.` = sapply(temp_data[voi_num], sd))

#### ---- Data Visualization ----

## -- pisa_2022_canada_merged --
# For plausible values in mathematics
pisa_2022_canada_merged %>%
  select(all_of(pvmaths)) %>%
  pivot_longer(everything(), names_to = "PlausibleValue", values_to = "Score") %>%
  ggplot(aes(x = Score)) +
  geom_density(fill = "skyblue", color=NA, alpha = 0.6) +
  facet_wrap(~ PlausibleValue, scales = "free", ncol = 5) +
  theme_minimal() +
  labs(title = "Distribution of PV1MATH to PV10MATH", x = "Score", y = "Density")

# For final student weight
ggplot(pisa_2022_canada_merged, aes(x = .data[[final_wt]])) +
  geom_density(fill = "lightgreen", color=NA, alpha = 0.6) +
  theme_minimal() +
  labs(title = "Distribution of Final Student Weight", x = final_wt, y = "Density")

# For BRR replicate weights
pisa_2022_canada_merged %>%
  select(all_of(rep_wts)) %>%
  pivot_longer(everything(), names_to = "RepWeight", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_density(fill = "lightgreen", color=NA, alpha = 0.6) +
  facet_wrap(~ RepWeight, scales = "free", ncol = 8) +
  theme_minimal() +
  labs(title = "Distribution of 80 BRR Replicate Weights", x = "Weight", y = "Density")

# # For predictors 
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


## -- temp_data --
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

### ---- Individual Variable Exploration -----

#### ---- Student Questionaire Derived Variables ----

##### ---- Simple questionnaire indices ----

###### ---- Subject-specific beliefs, attitudes, feelings and behaviours (Module 7) ----

####### ---- MATHMOT: Relative motivation to do well in mathematics compared to other core subjects ----
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

####### ---- MATHEASE: Perception of mathematics as easier than other core subjects ----
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

####### ---- MATHPREF: Preference of mathematics over other core subjects ----
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

###### ---- Out-of-school experiences (Module 10) ----

####### ---- EXERPRAC: Exercise or practice a sport before or after school ----
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

####### ---- STUDYHMW: Studying for school or homework before or after school ---- 
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

####### ---- WORKPAY: Working for pay before or after school ----
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

####### ---- WORKHOME: Working in household/take care of family members before or after school ----
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

##### ---- Derived variables based on IRT scaling ----

###### ---- Economic, social and cultural status (Module 2) ----

####### ---- HOMEPOS: Home possessions (Components of ESCS) ----
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

####### ---- ICTRES: ICT resources (at home) ----
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

###### ---- Educational pathways and post-secondary aspirations (Module 3) ----

####### ---- INFOSEEK: Information seeking regarding future career ----
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

###### ---- School culture and climate (Module 6) ----

####### ---- BULLIED: Being bullied ----
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

####### ---- FEELSAFE: Feeling safe ----
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

####### ---- BELONG: Sense of belonging ----
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

###### ---- Subject-specific beliefs, attitudes, feelings, and behaviours (Module 7) ----

####### ---- GROSAGR: Growth mindset ----
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

####### ---- ANXMAT: Mathematics anxiety ----
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

####### ---- MATHEFF: Mathematics self-efficacy: Formal and applied mathematics ----
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

####### ---- MATHEF21: Mathematics self-efficacy: Mathematical reasoning and 21st century mathematics ----
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

####### ---- MATHPERS: Proactive mathematics study behaviour ----
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

####### ---- FAMCON: Subjective familiarity with mathematics concepts ----
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

###### ---- General social and emotional characteristics (Module 8) ----

####### ---- ASSERAGR: Assertiveness ----
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

####### ---- COOPAGR: Cooperation ----
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

####### ---- CURIOAGR: Curiosity ----
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

####### ---- EMOCOAGR: Emotional control ----
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

####### ---- EMPATAGR: Empathy ----
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

####### ---- PERSEVAGR: Perseverance ----
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

####### ---- STRESAGR: Stress resistance ----
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

###### ---- Exposure to mathematics content (Module 15) ----

####### ---- EXPOFA: Exposure to formal and applied mathematics tasks ----
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

####### ---- EXPO21ST: Exposure to mathematical reasoning and 21st century mathematics tasks ----
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

###### ---- Mathematics teacher behaviour (Module 16) ----

####### ---- COGACRCO: Cognitive activation in mathematics: Foster reasoning ----
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

####### ---- COGACMCO: Cognitive activation in mathematics: Encourage mathematical thinking ----
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

####### ---- DISCLIM: Disciplinary climate in mathematics ----
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

###### ---- Parental/guardian involvement and support (Module 19) ----

####### ---- FAMSUP: Family support ----
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

###### ---- Creative thinking (Module 20) ----

####### ---- CREATFAM: Creative peers and family environment  ----
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

####### ---- CREATSCH: Creative school and class environment ----
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

####### ---- CREATEFF: Creative thinking self-efficacy ----
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

####### ---- CREATOP: Creativity and openness to intellect ----
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

####### ---- IMAGINE: Imagination and adventurousness ----
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

####### ---- OPENART: Openness to art and reflection ----
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

####### ---- CREATAS: Participation in creative activities at school ----
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

####### ---- CREATOOS: Participation in creative activities outside of school ----
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

###### ---- Global crises (Module 21) ----

####### ---- FAMSUPSL: Family support for self-directed learning ----
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

####### ---- FEELLAH: Feelings about learning at home  ----
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

####### ---- PROBSELF: Problems with self-directed learning ----
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

####### ---- SDLEFF: Self-directed learning self-efficacy ----
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

####### ---- SCHSUST: School actions to sustain learning  ----
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

##### ---- Complex composite index ----

####### ---- ESCS: Index of economic, social and cultural status ----
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

#### ---- School Questionnaire Derived Variables ----

##### ---- Simple questionnaire indices ---- 

###### ---- Out-of-school experiences (Module 10) ----

####### ---- MACTIV: Mathematics-related extra-curricular activities at school (ordinal/numeric) ----
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

###### ---- Organisation of student learning at school (Module 14) ----

###### ---- ABGMATH: Ability grouping for mathematics classes (ordinal/numeric) ----
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

####### ---- MTTRAIN: Mathematics teacher training ----
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

###### ---- Creative thinking (Module 20) ----

####### ---- CREENVSC: Creative school environment ----
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

####### ---- OPENCUL: Openness culture/climate  ----
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

###### ---- Global crises (Module 21) ----

####### ---- DIGPREP: Preparedness for Digital Learning (WLE) (numeric)----

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

## ---- I.i Correlation ----
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
print(round(point_estimate_matrix[1:6, 1:6], 6))
# >            PVmMATH   MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW
# > PVmMATH   1.000000  0.038183  0.132289  0.115414 -0.099777 -0.017415
# > MATHMOT   0.038183  1.000000  0.260818  0.284277 -0.011759 -0.047905
# > MATHEASE  0.132289  0.260818  1.000000  0.464649 -0.000044 -0.053973
# > MATHPREF  0.115414  0.284277  0.464649  1.000000 -0.002798 -0.033990
# > EXERPRAC -0.099777 -0.011759 -0.000044 -0.002798  1.000000  0.252129
# > STUDYHMW -0.017415 -0.047905 -0.053973 -0.033990  0.252129  1.000000

cat("\nStandard Errors (SE):\n")
print(round(se_matrix[1:6, 1:6], 6))
# >           PVmMATH  MATHMOT MATHEASE MATHPREF EXERPRAC STUDYHMW
# > PVmMATH  0.000000 0.014923 0.017448 0.017615 0.015460 0.018441
# > MATHMOT  0.014923 0.000000 0.022998 0.023273 0.015457 0.012930
# > MATHEASE 0.017448 0.022998 0.000000 0.017337 0.014083 0.016228
# > MATHPREF 0.017615 0.023273 0.017337 0.000000 0.014480 0.014962
# > EXERPRAC 0.015460 0.015457 0.014083 0.014480 0.000000 0.017737
# > STUDYHMW 0.018441 0.012930 0.016228 0.014962 0.017737 0.000000

# ---- II. Explanatory Modelling----
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
  formula <- as.formula(paste(pv, "~", paste(voi_num, collapse = " + ")))
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
# > Fitting main models: 1.766 sec elapsed
main_models[[1]] # View structure of the first fitted model
main_models[[2]] # View structure of the second fitted model

# --- Extract and average main fit estimates across plausible values (PVs) ---

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs[, 1:6]
# >      (Intercept)     MATHMOT MATHEASE   MATHPREF  EXERPRAC   STUDYHMW
# > [1,]    517.7045  2.62626424 5.812645  0.5545492 -2.871975 -0.7649009
# > [2,]    517.9243 -0.07987079 4.948286  3.6552795 -2.953582 -1.0328930
# > [3,]    515.2070 -0.69635683 6.750537  2.5641898 -2.830880 -0.8083437
# > [4,]    515.8971  2.32465656 5.156836  3.6192899 -3.305571 -0.7392423
# > [5,]    523.7478  0.80272280 6.182313  1.3811565 -3.082071 -1.0006322
# > [6,]    516.4392 -0.68589604 5.553969  2.4298706 -3.056447 -0.7352105
# > [7,]    516.0702  4.64857372 3.597671  2.1598339 -2.918427 -1.1324624
# > [8,]    514.7706 -1.86135762 6.643433  6.0159402 -2.958139 -0.5470533
# > [9,]    515.1987 -0.91688488 9.090008 -1.3361636 -3.065091 -0.5835628
# > [10,]   511.6412  1.48640482 7.385548  1.3197137 -3.054349 -0.4080198

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 516.4600477   0.7648256   6.1121246   2.2363660  -3.0096534  -0.7752321

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
# > [1] 0.4406115 0.4470595 0.4531012 0.4501005 0.4478831 0.4427759 0.4421043 0.4627855 0.4516192 0.4524189
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.4363085 0.4428061 0.4488943 0.4458705 0.4436360 0.4384896 0.4378128 0.4586530 0.4474008 0.4482068
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 272.0143 267.8767 269.0719 266.3372 267.1811 268.3630 268.4888 265.5499 268.8982 265.8422
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 102.3966 105.1067 107.7039 106.4068 105.4574 103.2993 103.0185 111.9890 107.0615 107.4078

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.449046
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.4448078
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 267.9623
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 105.9848

## ---- Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
tic("Fitting replicate models")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste(pv, "~", paste(voi_num, collapse = " + ")))
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
# > Fitting replicate models: 24.106 sec elapsed
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
# >                   [,1]       [,2]       [,3]       [,4]       [,5]       [,6]       [,7]       [,8]       [,9]      [,10]
# > (Intercept) 40.4455405 33.3109195 44.5719063 33.8155898 33.4819857 32.7994929 37.7962180 30.2045562 41.1455924 30.6374514
# > MATHMOT     32.5148920 30.1965227 30.5199736 32.5291500 28.4029531 23.5028225 30.0348952 28.0921797 27.3754495 28.5509674
# > MATHEASE    13.9770716 14.2726684 10.8427379 11.0325313 13.9691673 12.0893350 15.4334187 14.8373812 11.5715649 16.0476684
# > MATHPREF    15.4526756 16.0746834 14.7673224 15.6644304 16.5278601 13.4685583 18.8049597 16.7850278 12.9925582 18.5922604
# > EXERPRAC     0.1447829  0.1401971  0.1389922  0.1552187  0.1416329  0.1306707  0.1309604  0.1320388  0.1281163  0.1403693
# > STUDYHMW     0.1874655  0.1870218  0.2069933  0.1991546  0.1854170  0.1549344  0.1944705  0.2177548  0.1632732  0.1767991

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  35.8209253  29.1719806  13.4073545  15.9130336   0.1382979   0.1873284 

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  9.61222145  4.02019849  2.24166437  3.94720612  0.01834526  0.05250617 

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  46.3943689  33.5941989  15.8731853  20.2549604   0.1584777   0.2450852 

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >   6.8113412   5.7960503   3.9841166   4.5005511   0.3980926   0.4950608  

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
# > [1] 0.01315298
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.01325415
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 5.509568
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 5.667114

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
# >                  Term     Estimate Std. Error     t value Pr(>|t|) t_Signif     z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           (Intercept) 516.46004766  6.8113412 75.82354677  < 2e-16      *** 75.82354677  < 2e-16      ***   <NA>     <NA>
# >               MATHMOT   0.76482560  5.7960503  0.13195634  0.89502           0.13195634  0.89502            <NA>     <NA>
# >              MATHEASE   6.11212463  3.9841166  1.53412292  0.12505           1.53412292  0.12500            <NA>     <NA>
# >              MATHPREF   2.23636597  4.5005511  0.49690936  0.61927           0.49690936  0.61925            <NA>     <NA>
# >              EXERPRAC  -3.00965336  0.3980926 -7.56018408 4.55e-14      *** -7.56018408 4.02e-14      ***   <NA>     <NA>
# >              STUDYHMW  -0.77523209  0.4950608 -1.56593301  0.11741          -1.56593301  0.11736            <NA>     <NA>
# >               WORKPAY  -3.84123868  0.5042224 -7.61814344 2.92e-14      *** -7.61814344 2.57e-14      ***   <NA>     <NA>
# >              WORKHOME  -0.47705849  0.3752055 -1.27145913  0.20361          -1.27145913  0.20357            <NA>     <NA>
# >               HOMEPOS  15.29561280  2.7691319  5.52361288 3.44e-08      ***  5.52361288 3.32e-08      ***   <NA>     <NA>
# >                ICTRES -10.39734285  2.1002491 -4.95052849 7.58e-07      *** -4.95052849 7.40e-07      ***   <NA>     <NA>
# >              INFOSEEK  -1.09117360  1.5128621 -0.72126444  0.47077          -0.72126444  0.47075            <NA>     <NA>
# >               BULLIED  -4.52345579  1.4629579 -3.09199323  0.00200       ** -3.09199323  0.00199       **   <NA>     <NA>
# >              FEELSAFE   3.49657772  1.3309161  2.62719613  0.00863       **  2.62719613  0.00861       **   <NA>     <NA>
# >                BELONG  -3.06873543  1.7815121 -1.72254537  0.08502        . -1.72254537  0.08497        .   <NA>     <NA>
# >               GROSAGR   4.30499613  1.3417396  3.20851830  0.00134       **  3.20851830  0.00133       **   <NA>     <NA>
# >                ANXMAT  -7.08739081  1.2874043 -5.50517896 3.82e-08      *** -5.50517896 3.69e-08      ***   <NA>     <NA>
# >               MATHEFF  27.05889290  1.6339762 16.56015093  < 2e-16      *** 16.56015093  < 2e-16      ***   <NA>     <NA>
# >              MATHEF21   3.75585288  2.2889255  1.64088038  0.10087           1.64088038  0.10082            <NA>     <NA>
# >              MATHPERS   2.35998215  1.4597903  1.61665832  0.10600           1.61665832  0.10595            <NA>     <NA>
# >                FAMCON   6.30738631  1.0043654  6.27997198 3.59e-10      ***  6.27997198 3.39e-10      ***   <NA>     <NA>
# >              ASSERAGR   2.34979775  1.2270089  1.91506171  0.05553        .  1.91506171  0.05548        .   <NA>     <NA>
# >               COOPAGR  -2.09207041  1.6255508 -1.28699169  0.19814          -1.28699169  0.19810            <NA>     <NA>
# >              CURIOAGR   6.41146677  1.3148663  4.87613604 1.11e-06      ***  4.87613604 1.08e-06      ***   <NA>     <NA>
# >              EMOCOAGR   3.83712660  1.3168453  2.91387813  0.00358       **  2.91387813  0.00357       **   <NA>     <NA>
# >              EMPATAGR  -0.75485231  1.5085648 -0.50037777  0.61683          -0.50037777  0.61681            <NA>     <NA>
# >             PERSEVAGR   0.42909209  1.4652836  0.29283893  0.76965           0.29283893  0.76965            <NA>     <NA>
# >              STRESAGR  -0.94381789  1.4679332 -0.64295698  0.52027          -0.64295698  0.52025            <NA>     <NA>
# >                EXPOFA   0.31495929  1.7379437  0.18122526  0.85620           0.18122526  0.85619            <NA>     <NA>
# >              EXPO21ST  -1.12650162  1.7881577 -0.62997891  0.52873          -0.62997891  0.52871            <NA>     <NA>
# >              COGACRCO   1.32970769  1.4885905  0.89326629  0.37175           0.89326629  0.37171            <NA>     <NA>
# >              COGACMCO  -4.81408869  1.5925929 -3.02279922  0.00251       ** -3.02279922  0.00250       **   <NA>     <NA>
# >               DISCLIM  -0.52800058  1.3977151 -0.37775979  0.70562          -0.37775979  0.70561            <NA>     <NA>
# >                FAMSUP  -2.96067401  1.5367680 -1.92655886  0.05408        . -1.92655886  0.05403        .   <NA>     <NA>
# >              CREATFAM  -2.50958361  1.6584073 -1.51324923  0.13026          -1.51324923  0.13022            <NA>     <NA>
# >              CREATSCH  -5.22803036  1.6285411 -3.21025379  0.00133       ** -3.21025379  0.00133       **   <NA>     <NA>
# >              CREATEFF  -6.92723859  1.7010650 -4.07229501 4.71e-05      *** -4.07229501 4.66e-05      ***   <NA>     <NA>
# >               CREATOP   5.91482923  1.7984827  3.28878856  0.00101       **  3.28878856  0.00101       **   <NA>     <NA>
# >               IMAGINE   0.07983732  1.4673647  0.05440864  0.95661           0.05440864  0.95661            <NA>     <NA>
# >               OPENART  -4.25285996  1.5750075 -2.70021573  0.00695       ** -2.70021573  0.00693       **   <NA>     <NA>
# >               CREATAS   0.97019525  1.7017907  0.57010255  0.56863           0.57010255  0.56861            <NA>     <NA>
# >              CREATOOS  -4.74713624  1.6192522 -2.93168435  0.00338       ** -2.93168435  0.00337       **   <NA>     <NA>
# >              FAMSUPSL -10.31896656  1.4553114 -7.09055575 1.47e-12      *** -7.09055575 1.34e-12      ***   <NA>     <NA>
# >               FEELLAH  -3.15136997  1.5422922 -2.04330284  0.04106        * -2.04330284  0.04102        *   <NA>     <NA>
# >              PROBSELF  -0.06343109  1.4011443 -0.04527092  0.96389          -0.04527092  0.96389            <NA>     <NA>
# >                SDLEFF   1.62719002  1.5463875  1.05225241  0.29272           1.05225241  0.29268            <NA>     <NA>
# >               SCHSUST   3.95368330  1.6044840  2.46414635  0.01376        *  2.46414635  0.01373        *   <NA>     <NA>
# >               LEARRES   1.07012526  1.6370303  0.65369913  0.51333           0.65369913  0.51331            <NA>     <NA>
# >                  ESCS  18.44997799  2.4953164  7.39384324 1.60e-13      ***  7.39384324 1.43e-13      ***   <NA>     <NA>
# >                MACTIV   2.08931519  1.0076989  2.07335265  0.03818        *  2.07335265  0.03814        *   <NA>     <NA>
# >               ABGMATH  -0.71680957  2.5326022 -0.28303283  0.77716          -0.28303283  0.77715            <NA>     <NA>
# >               MTTRAIN   1.08386888  1.5499992  0.69927062  0.48441           0.69927062  0.48438            <NA>     <NA>
# >              CREENVSC  -3.75293088  1.7484839 -2.14639149  0.03188        * -2.14639149  0.03184        *   <NA>     <NA>
# >               OPENCUL   4.80631454  1.8516142  2.59574303  0.00946       **  2.59574303  0.00944       **   <NA>     <NA>
# >               DIGPREP   2.00789446  1.5737026  1.27590462  0.20203           1.27590462  0.20199            <NA>     <NA>
# >             R-squared   0.45000000  0.0100000          NA     <NA>     <NA>          NA     <NA>     <NA>   <NA>     <NA>
# >    Adjusted R-squared   0.44000000  0.0100000          NA     <NA>     <NA>          NA     <NA>     <NA>   <NA>     <NA>
# >   Residual Std. Error 267.96000000  5.5100000          NA     <NA>     <NA>          NA     <NA>     <NA>   <NA>     <NA>
# >           F-statistic 105.98000000  5.6700000          NA     <NA>     <NA>          NA     <NA>     <NA> <2e-16      ***
as.data.frame(result_table) %>% head(10) %>% print(row.names = FALSE)
# >        Term    Estimate Std. Error    t value Pr(>|t|) t_Signif    z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# > (Intercept) 516.4600477  6.8113412 75.8235468  < 2e-16      *** 75.8235468  < 2e-16      ***   <NA>     <NA>
# >     MATHMOT   0.7648256  5.7960503  0.1319563  0.89502           0.1319563  0.89502            <NA>     <NA>
# >    MATHEASE   6.1121246  3.9841166  1.5341229  0.12505           1.5341229  0.12500            <NA>     <NA>
# >    MATHPREF   2.2363660  4.5005511  0.4969094  0.61927           0.4969094  0.61925            <NA>     <NA>
# >    EXERPRAC  -3.0096534  0.3980926 -7.5601841 4.55e-14      *** -7.5601841 4.02e-14      ***   <NA>     <NA>
# >    STUDYHMW  -0.7752321  0.4950608 -1.5659330  0.11741          -1.5659330  0.11736            <NA>     <NA>
# >     WORKPAY  -3.8412387  0.5042224 -7.6181434 2.92e-14      *** -7.6181434 2.57e-14      ***   <NA>     <NA>
# >    WORKHOME  -0.4770585  0.3752055 -1.2714591  0.20361          -1.2714591  0.20357            <NA>     <NA>
# >     HOMEPOS  15.2956128  2.7691319  5.5236129 3.44e-08      ***  5.5236129 3.32e-08      ***   <NA>     <NA>
# >      ICTRES -10.3973428  2.1002491 -4.9505285 7.58e-07      *** -4.9505285 7.40e-07      ***   <NA>     <NA>

### ---- Compare results with intsvy ----
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=voi_num, data=temp_data)
# >             Estimate Std. Error t value
# > (Intercept)   516.46       6.81   75.82
# > MATHMOT         0.76       5.80    0.13
# > MATHEASE        6.11       3.98    1.53
# > MATHPREF        2.24       4.50    0.50
# > EXERPRAC       -3.01       0.40   -7.56
# > STUDYHMW       -0.78       0.50   -1.57
# > WORKPAY        -3.84       0.50   -7.62
# > WORKHOME       -0.48       0.38   -1.27
# > HOMEPOS        15.30       2.77    5.52
# > ICTRES        -10.40       2.10   -4.95
# > INFOSEEK       -1.09       1.51   -0.72
# > BULLIED        -4.52       1.46   -3.09
# > FEELSAFE        3.50       1.33    2.63
# > BELONG         -3.07       1.78   -1.72
# > GROSAGR         4.30       1.34    3.21
# > ANXMAT         -7.09       1.29   -5.51
# > MATHEFF        27.06       1.63   16.56
# > MATHEF21        3.76       2.29    1.64
# > MATHPERS        2.36       1.46    1.62
# > FAMCON          6.31       1.00    6.28
# > ASSERAGR        2.35       1.23    1.92
# > COOPAGR        -2.09       1.63   -1.29
# > CURIOAGR        6.41       1.31    4.88
# > EMOCOAGR        3.84       1.32    2.91
# > EMPATAGR       -0.75       1.51   -0.50
# > PERSEVAGR       0.43       1.47    0.29
# > STRESAGR       -0.94       1.47   -0.64
# > EXPOFA          0.31       1.74    0.18
# > EXPO21ST       -1.13       1.79   -0.63
# > COGACRCO        1.33       1.49    0.89
# > COGACMCO       -4.81       1.59   -3.02
# > DISCLIM        -0.53       1.40   -0.38
# > FAMSUP         -2.96       1.54   -1.93
# > CREATFAM       -2.51       1.66   -1.51
# > CREATSCH       -5.23       1.63   -3.21
# > CREATEFF       -6.93       1.70   -4.07
# > CREATOP         5.91       1.80    3.29
# > IMAGINE         0.08       1.47    0.05
# > OPENART        -4.25       1.58   -2.70
# > CREATAS         0.97       1.70    0.57
# > CREATOOS       -4.75       1.62   -2.93
# > FAMSUPSL      -10.32       1.46   -7.09
# > FEELLAH        -3.15       1.54   -2.04
# > PROBSELF       -0.06       1.40   -0.05
# > SDLEFF          1.63       1.55    1.05
# > SCHSUST         3.95       1.60    2.46
# > LEARRES         1.07       1.64    0.65
# > ESCS           18.45       2.50    7.39
# > MACTIV          2.09       1.01    2.07
# > ABGMATH        -0.72       2.53   -0.28
# > MTTRAIN         1.08       1.55    0.70
# > CREENVSC       -3.75       1.75   -2.15
# > OPENCUL         4.81       1.85    2.60
# > DIGPREP         2.01       1.57    1.28
# > R-squared       0.45       0.01   34.14

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=voi_num, data=temp_data, std=TRUE)
# >             Estimate Std. Error t value
# > (Intercept)     0.07       0.02    4.56
# > MATHMOT         0.00       0.01    0.13
# > MATHEASE        0.02       0.02    1.54
# > MATHPREF        0.01       0.02    0.50
# > EXERPRAC       -0.11       0.01   -7.50
# > STUDYHMW       -0.03       0.02   -1.56
# > WORKPAY        -0.11       0.01   -7.56
# > WORKHOME       -0.02       0.01   -1.27
# > HOMEPOS         0.14       0.02    5.49
# > ICTRES         -0.10       0.02   -4.91
# > INFOSEEK       -0.01       0.02   -0.72
# > BULLIED        -0.05       0.02   -3.09
# > FEELSAFE        0.04       0.01    2.62
# > BELONG         -0.03       0.02   -1.72
# > GROSAGR         0.05       0.01    3.20
# > ANXMAT         -0.10       0.02   -5.49
# > MATHEFF         0.34       0.02   17.11
# > MATHEF21        0.04       0.02    1.64
# > MATHPERS        0.02       0.02    1.62
# > FAMCON          0.10       0.02    6.29
# > ASSERAGR        0.03       0.01    1.92
# > COOPAGR        -0.02       0.02   -1.29
# > CURIOAGR        0.07       0.01    4.89
# > EMOCOAGR        0.04       0.02    2.91
# > EMPATAGR       -0.01       0.02   -0.50
# > PERSEVAGR       0.00       0.02    0.29
# > STRESAGR       -0.01       0.02   -0.64
# > EXPOFA          0.00       0.02    0.18
# > EXPO21ST       -0.01       0.02   -0.63
# > COGACRCO        0.01       0.02    0.89
# > COGACMCO       -0.05       0.02   -3.01
# > DISCLIM        -0.01       0.01   -0.38
# > FAMSUP         -0.03       0.02   -1.93
# > CREATFAM       -0.03       0.02   -1.52
# > CREATSCH       -0.06       0.02   -3.22
# > CREATEFF       -0.07       0.02   -4.08
# > CREATOP         0.06       0.02    3.28
# > IMAGINE         0.00       0.02    0.05
# > OPENART        -0.05       0.02   -2.71
# > CREATAS         0.01       0.02    0.57
# > CREATOOS       -0.05       0.02   -2.94
# > FAMSUPSL       -0.12       0.02   -7.06
# > FEELLAH        -0.03       0.02   -2.05
# > PROBSELF        0.00       0.02   -0.04
# > SDLEFF          0.02       0.02    1.05
# > SCHSUST         0.04       0.02    2.46
# > LEARRES         0.01       0.02    0.65
# > ESCS            0.15       0.02    7.45
# > MACTIV          0.03       0.02    2.07
# > ABGMATH        -0.01       0.02   -0.28
# > MTTRAIN         0.01       0.02    0.70
# > CREENVSC       -0.04       0.02   -2.15
# > OPENCUL         0.05       0.02    2.60
# > DIGPREP         0.02       0.02    1.28
# > R-squared       0.45       0.01   34.14

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

## ---- Random Train/Validation/Test (80/10/10) split ----
set.seed(123)  # Ensure reproducibility

n <- nrow(temp_data) # 16114
indices <- sample(n)  # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.80 * n)        # 5555
n_valid <- floor(0.10 * n)        # 694
n_test  <- n - n_train - n_valid  # 695

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
tic("Fitting main models")
main_models <- lapply(pvmaths, function(pv) {
  formula <- as.formula(paste(pv, "~", paste(voi_num, collapse = " + ")))
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
# > Fitting main models: 1.091 sec elapsed
main_models[[1]] # View structure of the first fitted model
main_models[[2]] # View structure of the second fitted model

# --- Extract and average main fit estimates across plausible values (PVs) ---

# Coefficient matrix: M (10) × p (2) for PV-specific estimates θ̂ₘ (Rubin's Step 1)
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))
main_coefs[, 1:6]
# >      (Intercept)   MATHMOT MATHEASE  MATHPREF  EXERPRAC    STUDYHMW
# > [1,]    517.7296 -3.982680 3.752141 3.6929353 -2.829478 -0.38453655
# > [2,]    519.9460 -6.782201 2.990970 4.7458955 -3.054977 -0.73937978
# > [3,]    516.1596 -6.798383 6.025690 4.1673247 -2.787930 -0.26911794
# > [4,]    517.4321 -1.270876 2.473903 6.4550548 -3.218052 -0.48712641
# > [5,]    524.3755 -6.461862 4.973932 3.5751405 -3.006894 -0.49500262
# > [6,]    517.7475 -5.988161 4.258326 3.3898115 -2.925868 -0.24353415
# > [7,]    516.9792 -1.462859 2.048557 3.4470676 -2.916638 -0.92930725
# > [8,]    516.8223 -5.916986 5.840040 7.5251570 -3.068600 -0.08046007
# > [9,]    515.8476 -6.123641 8.981218 0.8112273 -3.114676 -0.06490011
# > [10,]   510.6869 -3.917151 6.076279 2.6603158 -3.039227 -0.06043878

# Final mean estimate - regression coef: Averages across PVs (Rubin's step 2: θ̂)
main_coef  <- colMeans(main_coefs)
main_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 517.3726527  -4.8704800   4.7421055   4.0469930  -2.9962340  -0.3753804 

# Model fit statistics for each PV-specific model (Rubin's Step 1: θ̂ₘ)
main_r2s         <- sapply(main_models, function(m) m$r2)
main_r2s
# > [1] 0.4382615 0.4502053 0.4526300 0.4497259 0.4441551 0.4465673 0.4430294 0.4593376 0.4502474 0.4570792
main_adj_r2s     <- sapply(main_models, function(m) m$adj_r2)
main_adj_r2s
# > [1] 0.4328493 0.4449082 0.4473563 0.4444242 0.4387998 0.4412352 0.4376632 0.4541285 0.4449507 0.4518484
main_sigmas      <- sapply(main_models, function(m) m$sigma)
main_sigmas
# > [1] 272.2889 266.5637 268.2113 265.6231 266.4823 266.1062 265.8132 265.9181 265.5985 262.9285
main_fstats_val  <- sapply(main_models, function(m) m$fstat_val)
main_fstats_val
# > [1] 80.97759 84.99155 85.82783 84.82709 82.93672 83.75058 82.55932 88.18030 85.00602 87.38175

# Final mean estimate - fit statistics:averages across PVs (Rubin's step 2: θ̂)
main_r2         <- mean(main_r2s)
main_r2
# > [1] 0.4491239
main_adj_r2     <- mean(main_adj_r2s)
main_adj_r2 
# > [1] 0.4438164
main_sigma      <- mean(main_sigmas)
main_sigma
# > [1] 266.5534
main_fstat_val  <- mean(main_fstats_val)
main_fstat_val 
# > [1] 84.64387

### ---- Replicate models using BRR replicate weights (W_FSTURWT1–W_FSTURWT80) ----
tic("Fitting replicate models")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    formula <- as.formula(paste(pv, "~", paste(voi_num, collapse = " + ")))
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
# > Fitting replicate models: 22.359 sec elapsed
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
# >                   [,1]       [,2]       [,3]       [,4]       [,5]       [,6]       [,7]       [,8]       [,9]      [,10]
# > (Intercept) 45.0307252 39.0265356 53.4413603 43.4060315 41.3200943 41.6908352 43.3524570 38.9034062 46.0844840 37.7667016
# > MATHMOT     48.8306505 48.6484060 42.4222580 48.4726530 36.5029823 34.2051397 48.8768387 37.6183404 40.6616363 39.9108124
# > MATHEASE    19.7349281 18.0191486 14.9764260 15.6649833 19.0560700 17.8596455 21.8011557 23.3058539 17.0051576 18.4548479
# > MATHPREF    20.4927610 23.6912366 21.0559669 24.4641338 24.1837340 19.1363949 24.9831730 23.8950384 19.1279136 23.7236171
# > EXERPRAC     0.2036546  0.1808555  0.1861279  0.2124202  0.1866795  0.1802971  0.1889375  0.1820541  0.2030708  0.2041436
# > STUDYHMW     0.2784311  0.2157197  0.2976281  0.2624279  0.2675064  0.2484901  0.2399203  0.2877113  0.2479804  0.2544309

# Final sampling variance σ²(θ̂ₘ) = average across M plausible values (Rubin's step 3)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef) 
sampling_var_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  43.0022631  42.6149717  18.5878217  22.4753969   0.1928241   0.2600246 

# Imputation variance σ²₍test₎ = variance of θ̂ₘ across M plausible values ( Rubin's step 4)
imputation_var_coef <- colSums((main_coefs - matrix(main_coef, nrow=M, ncol=length(main_coef), byrow=TRUE))^2) / (M - 1)
imputation_var_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 11.65419494  4.45387120  4.37081987  3.54802046  0.01737404  0.08638337  

# Total error variance σ²₍error₎ = sampling variance + adjusted imputation variance (Rubin's step 5)
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
var_final_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  55.8218775  47.5142300  23.3957235  26.3782194   0.2119355   0.3550463

# Final standard error σ₍error₎ = √(σ²₍error₎) (Rubin's step 6)
se_final_coef <- sqrt(var_final_coef)
se_final_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >   7.4714040   6.8930567   4.8369126   5.1359731   0.4603646   0.5958576 

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
# > [1] 0.01461424
se_final_adj_r2 <- sqrt(var_final_adj_r2)
se_final_adj_r2
# > [1] 0.01475505
se_final_sigma  <- sqrt(var_final_sigma)
se_final_sigma
# > [1] 6.349353
se_final_fstat_val  <- sqrt(var_final_fstat_val)
se_final_fstat_val
# > [1] 5.03102

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
# >                  Term     Estimate Std. Error     t value Pr(>|t|) t_Signif     z value Pr(>|z|) z_Signif Pr(>F) F_Signif
# >           (Intercept) 517.37265268  7.4714040 69.24704562  < 2e-16      *** 69.24704562  < 2e-16      ***   <NA>     <NA>
# >               MATHMOT  -4.87047996  6.8930567 -0.70657768 0.479859          -0.70657768 0.479829            <NA>     <NA>
# >              MATHEASE   4.74210553  4.8369126  0.98039926 0.326932           0.98039926 0.326889            <NA>     <NA>
# >              MATHPREF   4.04699301  5.1359731  0.78797006 0.430748           0.78797006 0.430714            <NA>     <NA>
# >              EXERPRAC  -2.99623396  0.4603646 -6.50839386 8.27e-11      *** -6.50839386 7.60e-11      ***   <NA>     <NA>
# >              STUDYHMW  -0.37538037  0.5958576 -0.62998330 0.528732          -0.62998330 0.528706            <NA>     <NA>
# >               WORKPAY  -3.50849948  0.6037847 -5.81084569 6.56e-09      *** -5.81084569 6.22e-09      ***   <NA>     <NA>
# >              WORKHOME  -0.69556012  0.4260996 -1.63238852 0.102655          -1.63238852 0.102598            <NA>     <NA>
# >               HOMEPOS  15.69606408  3.1638035  4.96113749 7.22e-07      ***  4.96113749 7.01e-07      ***   <NA>     <NA>
# >                ICTRES -11.08603696  2.2385844 -4.95225319 7.56e-07      *** -4.95225319 7.34e-07      ***   <NA>     <NA>
# >              INFOSEEK  -2.40792110  1.5020055 -1.60313737 0.108962          -1.60313737 0.108904            <NA>     <NA>
# >               BULLIED  -3.75870589  1.5024528 -2.50171308 0.012388        * -2.50171308 0.012359        *   <NA>     <NA>
# >              FEELSAFE   3.18364444  1.6270456  1.95670260 0.050433        .  1.95670260 0.050382        .   <NA>     <NA>
# >                BELONG  -3.42885662  1.9886907 -1.72417795 0.084732        . -1.72417795 0.084676        .   <NA>     <NA>
# >               GROSAGR   4.97788365  1.4837488  3.35493701 0.000799      ***  3.35493701 0.000794      ***   <NA>     <NA>
# >                ANXMAT  -6.86056550  1.4141838 -4.85125460 1.26e-06      *** -4.85125460 1.23e-06      ***   <NA>     <NA>
# >               MATHEFF  26.94010840  1.8276575 14.74023897  < 2e-16      *** 14.74023897  < 2e-16      ***   <NA>     <NA>
# >              MATHEF21   4.26566366  2.4132561  1.76759676 0.077184        .  1.76759676 0.077128        .   <NA>     <NA>
# >              MATHPERS   2.82035224  1.6996399  1.65938219 0.097096        .  1.65938219 0.097039        .   <NA>     <NA>
# >                FAMCON   5.96532494  1.2157107  4.90686233 9.52e-07      ***  4.90686233 9.25e-07      ***   <NA>     <NA>
# >              ASSERAGR   2.43417629  1.4837964  1.64050554 0.100957           1.64050554 0.100900            <NA>     <NA>
# >               COOPAGR  -1.93341844  1.7309521 -1.11696819 0.264057          -1.11696819 0.264008            <NA>     <NA>
# >              CURIOAGR   6.47822409  1.4705262  4.40537811 1.08e-05      ***  4.40537811 1.06e-05      ***   <NA>     <NA>
# >              EMOCOAGR   2.88192368  1.4809878  1.94594697 0.051712        .  1.94594697 0.051661        .   <NA>     <NA>
# >              EMPATAGR  -0.08277775  1.6634248 -0.04976344 0.960313          -0.04976344 0.960311            <NA>     <NA>
# >             PERSEVAGR   0.98410893  1.5895061  0.61912875 0.535857           0.61912875 0.535832            <NA>     <NA>
# >              STRESAGR  -1.56214903  1.6969642 -0.92055507 0.357323          -0.92055507 0.357283            <NA>     <NA>
# >                EXPOFA   0.64717824  2.0394331  0.31733243 0.751003           0.31733243 0.750991            <NA>     <NA>
# >              EXPO21ST  -1.21229312  2.0235776 -0.59908409 0.549141          -0.59908409 0.549117            <NA>     <NA>
# >              COGACRCO   1.62384735  1.6395465  0.99042471 0.322010           0.99042471 0.321967            <NA>     <NA>
# >              COGACMCO  -5.64747324  1.7635836 -3.20227131 0.001371       ** -3.20227131 0.001363       **   <NA>     <NA>
# >               DISCLIM  -0.93693554  1.5902391 -0.58917903 0.555765          -0.58917903 0.555741            <NA>     <NA>
# >                FAMSUP  -2.77304339  1.7831938 -1.55509925 0.119980          -1.55509925 0.119922            <NA>     <NA>
# >              CREATFAM  -0.53249495  1.7223094 -0.30917497 0.757200          -0.30917497 0.757188            <NA>     <NA>
# >              CREATSCH  -5.41156934  1.6441670 -3.29137451 0.001003       ** -3.29137451 0.000997      ***   <NA>     <NA>
# >              CREATEFF  -7.39673159  2.0885100 -3.54163094 0.000401      *** -3.54163094 0.000398      ***   <NA>     <NA>
# >               CREATOP   6.24716759  2.0060228  3.11420570 0.001854       **  3.11420570 0.001844       **   <NA>     <NA>
# >               IMAGINE  -1.43393802  1.6249540 -0.88244836 0.377573          -0.88244836 0.377534            <NA>     <NA>
# >               OPENART  -4.11549239  1.8128021 -2.27023804 0.023232        * -2.27023804 0.023193        *   <NA>     <NA>
# >               CREATAS   0.13718325  1.7470715  0.07852183 0.937416           0.07852183 0.937413            <NA>     <NA>
# >              CREATOOS  -4.07887954  1.7816536 -2.28937849 0.022095        * -2.28937849 0.022057        *   <NA>     <NA>
# >              FAMSUPSL -10.91591228  1.5896268 -6.86696523 7.28e-12      *** -6.86696523 6.56e-12      ***   <NA>     <NA>
# >               FEELLAH  -2.49452407  1.8340562 -1.36011319 0.173850          -1.36011319 0.173794            <NA>     <NA>
# >              PROBSELF  -1.07043731  1.6266425 -0.65806549 0.510524          -0.65806549 0.510496            <NA>     <NA>
# >                SDLEFF   1.60979444  1.7087964  0.94206337 0.346202           0.94206337 0.346160            <NA>     <NA>
# >               SCHSUST   4.13237905  1.7309046  2.38741004 0.017001        *  2.38741004 0.016968        *   <NA>     <NA>
# >               LEARRES   2.04198222  1.8930171  1.07869191 0.280772           1.07869191 0.280725            <NA>     <NA>
# >                  ESCS  18.80491234  2.9246385  6.42982461 1.39e-10      ***  6.42982461 1.28e-10      ***   <NA>     <NA>
# >                MACTIV   1.73211277  1.0916392  1.58670809 0.112636           1.58670809 0.112579            <NA>     <NA>
# >               ABGMATH  -1.31779729  2.7225571 -0.48402926 0.628384          -0.48402926 0.628365            <NA>     <NA>
# >               MTTRAIN   1.36560583  1.6457263  0.82978913 0.406694           0.82978913 0.406658            <NA>     <NA>
# >              CREENVSC  -4.30997868  1.9096453 -2.25695251 0.024050        * -2.25695251 0.024011        *   <NA>     <NA>
# >               OPENCUL   4.22824555  1.9405325  2.17890990 0.029381        *  2.17890990 0.029338        *   <NA>     <NA>
# >               DIGPREP   1.68860877  1.6360307  1.03213758 0.302053           1.03213758 0.302008            <NA>     <NA>
# >             R-squared   0.45000000  0.0100000          NA     <NA>     <NA>          NA     <NA>     <NA>   <NA>     <NA>
# >    Adjusted R-squared   0.44000000  0.0100000          NA     <NA>     <NA>          NA     <NA>     <NA>   <NA>     <NA>
# >   Residual Std. Error 266.55000000  6.3500000          NA     <NA>     <NA>          NA     <NA>     <NA>   <NA>     <NA>
# >           F-statistic  84.64000000  5.0300000          NA     <NA>     <NA>          NA     <NA>     <NA> <2e-16      ***

### ---- Compare results with intsvy package  ----
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=voi_num, data=train_data)
# >             Estimate Std. Error t value
# > (Intercept)   517.37       7.47   69.25
# > MATHMOT        -4.87       6.89   -0.71
# > MATHEASE        4.74       4.84    0.98
# > MATHPREF        4.05       5.14    0.79
# > EXERPRAC       -3.00       0.46   -6.51
# > STUDYHMW       -0.38       0.60   -0.63
# > WORKPAY        -3.51       0.60   -5.81
# > WORKHOME       -0.70       0.43   -1.63
# > HOMEPOS        15.70       3.16    4.96
# > ICTRES        -11.09       2.24   -4.95
# > INFOSEEK       -2.41       1.50   -1.60
# > BULLIED        -3.76       1.50   -2.50
# > FEELSAFE        3.18       1.63    1.96
# > BELONG         -3.43       1.99   -1.72
# > GROSAGR         4.98       1.48    3.35
# > ANXMAT         -6.86       1.41   -4.85
# > MATHEFF        26.94       1.83   14.74
# > MATHEF21        4.27       2.41    1.77
# > MATHPERS        2.82       1.70    1.66
# > FAMCON          5.97       1.22    4.91
# > ASSERAGR        2.43       1.48    1.64
# > COOPAGR        -1.93       1.73   -1.12
# > CURIOAGR        6.48       1.47    4.41
# > EMOCOAGR        2.88       1.48    1.95
# > EMPATAGR       -0.08       1.66   -0.05
# > PERSEVAGR       0.98       1.59    0.62
# > STRESAGR       -1.56       1.70   -0.92
# > EXPOFA          0.65       2.04    0.32
# > EXPO21ST       -1.21       2.02   -0.60
# > COGACRCO        1.62       1.64    0.99
# > COGACMCO       -5.65       1.76   -3.20
# > DISCLIM        -0.94       1.59   -0.59
# > FAMSUP         -2.77       1.78   -1.56
# > CREATFAM       -0.53       1.72   -0.31
# > CREATSCH       -5.41       1.64   -3.29
# > CREATEFF       -7.40       2.09   -3.54
# > CREATOP         6.25       2.01    3.11
# > IMAGINE        -1.43       1.62   -0.88
# > OPENART        -4.12       1.81   -2.27
# > CREATAS         0.14       1.75    0.08
# > CREATOOS       -4.08       1.78   -2.29
# > FAMSUPSL      -10.92       1.59   -6.87
# > FEELLAH        -2.49       1.83   -1.36
# > PROBSELF       -1.07       1.63   -0.66
# > SDLEFF          1.61       1.71    0.94
# > SCHSUST         4.13       1.73    2.39
# > LEARRES         2.04       1.89    1.08
# > ESCS           18.80       2.92    6.43
# > MACTIV          1.73       1.09    1.59
# > ABGMATH        -1.32       2.72   -0.48
# > MTTRAIN         1.37       1.65    0.83
# > CREENVSC       -4.31       1.91   -2.26
# > OPENCUL         4.23       1.94    2.18
# > DIGPREP         1.69       1.64    1.03
# > R-squared       0.45       0.01   30.73

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
# > 1  67.69654 53.66004  4.422382e-13 1.874873 0.4382615
# > 2  66.27312 52.74389  1.338934e-13 1.797124 0.4502053
# > 3  66.68276 53.00499  2.399037e-15 1.833021 0.4526300
# > 4  66.03928 52.62570  7.712781e-13 1.756346 0.4497259
# > 5  66.25288 52.56226  4.470618e-13 1.772508 0.4441551
# > 6  66.15939 52.27914 -1.179695e-12 1.773385 0.4465673
# > 7  66.08653 52.50757 -1.154309e-12 1.783424 0.4430294
# > 8  66.11262 52.24913  1.180381e-12 1.798444 0.4593376
# > 9  66.03315 52.68046 -1.120442e-12 1.776969 0.4502474
# > 10 65.36934 52.20617  9.521786e-13 1.724624 0.4570792

# Obtain final mean estimate 
main_metrics <- colMeans(train_metrics_main)
main_metrics
# >         rmse          mae         bias     bias_pct           r2 
# > 6.627056e+01 5.265194e+01 4.749850e-14 1.789072e+00 4.491239e-01 

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
# > rmse     7.792197e-01 7.323701e-01 8.158813e-01 8.064431e-01 9.204944e-01 9.254673e-01 8.996825e-01 7.817850e-01 6.553996e-01 8.018123e-01
# > mae      5.082356e-01 4.747511e-01 5.465016e-01 5.573449e-01 6.798820e-01 6.488355e-01 6.121872e-01 5.190624e-01 5.596853e-01 6.559872e-01
# > bias     1.426856e-23 1.856205e-23 1.015227e-23 1.056000e-23 1.213982e-23 1.981483e-23 1.891396e-23 1.790780e-23 2.181225e-23 1.271988e-23
# > bias_pct 3.207411e-03 2.827536e-03 3.671255e-03 2.539914e-03 3.489080e-03 3.258583e-03 3.045752e-03 3.267157e-03 2.440557e-03 3.006752e-03
# > r2       1.810837e-04 1.674553e-04 1.463504e-04 1.844490e-04 1.375154e-04 1.746233e-04 1.786360e-04 1.781514e-04 1.666999e-04 1.697326e-04

# <=> Equivalent codes
# sampling_var_matrix <- sapply(1:M, function(m) {
#   sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |> colMeans() / (1 - k)^2
# }) 
# sampling_var_matrix

sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >         rmse          mae         bias     bias_pct           r2 
# > 8.118555e-01 5.762473e-01 1.568514e-23 3.075400e-03 1.684697e-04 

# Imputation variance
imputation_var <- colSums((train_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 3.552056e-01 1.866330e-01 8.072360e-25 1.707700e-03 4.100586e-05

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 1.202582e+00 7.815435e-01 1.657310e-23 4.953869e-03 2.135762e-04 

# Final standard error
se_final <- sqrt(var_final)
se_final
# >         rmse          mae         bias     bias_pct           r2 
# > 1.096623e+00 8.840495e-01 4.071008e-12 7.038373e-02 1.461424e-02 

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
# >    Metric         Point_estimate       Standard_error             CI_lower              CI_upper           CI_length
# >      RMSE 66.2705611873825262137 1.096622879032783215 64.12121983985564100 68.419902534909411429 4.29868269505377043
# >       MAE 52.6519350911014498706 0.884049505171120464 50.91922990041560126 54.384640281787298477 3.46541038137169721
# >      Bias  0.0000000000000474985 0.000000000004071008 -0.00000000000793153  0.000000000008026527 0.00000000001595806
# >     Bias%  1.7890719564144030862 0.070383729154191971  1.65112238217456508  1.927021530654241088 0.27589914847967600
# > R-squared  0.4491238599532508813 0.014614244854575548  0.42048046637703301  0.477767253529468749 0.05728678715243574

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=voi_num, data=train_data)

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
# > 1  66.23222 51.33328  1.7454092 1.893364 0.3776325
# > 2  67.90534 54.09068 -0.1008823 1.642605 0.3712624
# > 3  65.95679 52.65382  0.7172755 1.706535 0.4062460
# > 4  66.26064 52.98360  1.6537472 1.940456 0.3981834
# > 5  65.12941 51.68504  2.3883751 2.002100 0.4120840
# > 6  68.34922 53.99857  3.3955968 2.400483 0.3721709
# > 7  70.13624 55.74269  3.3997819 2.501985 0.3650383
# > 8  65.59383 52.03436  1.6149108 1.923803 0.4157759
# > 9  69.06966 54.53929  2.7618269 2.308879 0.3792312
# > 10 70.66971 55.81559  0.1374989 1.887175 0.3545364

# Obtain final mean estimate 
main_metrics <- colMeans(valid_metrics_main)
main_metrics
# >       rmse        mae       bias   bias_pct         r2 
# > 67.5303048 53.4876910  1.7713540  2.0207385  0.3852161 

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
# > rmse      6.151149820  6.911496940  5.783045844  5.227373887  4.992754583  6.416820130  9.455258400  7.009668399  7.807898560  6.992694776
# > mae       4.533362686  4.629081524  4.172166523  3.491673843  3.454494579  4.737269495  6.196976693  4.757489390  5.626079127  4.274292823
# > bias     12.224339291 13.952685973 12.085048059 10.452148961 12.115253675 14.890684614 14.334243770 13.002027528 14.340298019 11.666120361
# > bias_pct  0.483451620  0.596601056  0.504925736  0.431910598  0.475371958  0.618759222  0.574573758  0.526761619  0.593197340  0.564743788
# > r2        0.001867713  0.001711545  0.001601942  0.001483295  0.001636143  0.001772484  0.001815972  0.001520363  0.001719599  0.001754212
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >        rmse          mae         bias     bias_pct          r2 
# > 6.674816134  4.587288668 12.906285025  0.537029670  0.00168832  

# Imputation variance
imputation_var <- colSums((valid_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 3.9016523100 2.5763579176 1.5529450636 0.0834211806 0.0004525329

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2 
# > 10.966633675  7.421282378 14.614524595  0.628792968  0.002186113 

# Final standard error
se_final <- sqrt(var_final)
se_final
# >       rmse        mae       bias   bias_pct         r2 
# > 3.31159081 2.72420307 3.82289479 0.79296467 0.04675589 

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
# >     Metric Point_estimate Standard_error   CI_lower  CI_upper  CI_length
# >       RMSE     67.5303048     3.31159081 61.0397061 74.020903 12.9811974
# >        MAE     53.4876910     2.72420307 48.1483510 58.827031 10.6786798
# >       Bias      1.7713540     3.82289479 -5.7213821  9.264090 14.9854722
# >      Bias%      2.0207385     0.79296467  0.4665564  3.574921  3.1083644
# >  R-squared      0.3852161     0.04675589  0.2935762  0.476856  0.1832797 

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
# > 1  67.99977 52.74835 -5.404481 0.7124773 0.4777749
# > 2  67.42839 52.44755 -1.966007 1.3639101 0.4615067
# > 3  69.02353 54.05778 -3.754662 1.1105013 0.4670697
# > 4  66.83389 52.52522 -3.619525 1.0734735 0.4738581
# > 5  68.72721 53.74164 -4.708640 0.9484182 0.4709359
# > 6  68.83426 54.39365 -2.749913 1.3233128 0.4506770
# > 7  68.27947 53.60083 -5.178364 0.8405961 0.4695848
# > 8  65.12104 50.92379 -4.046781 0.8621685 0.5035157
# > 9  70.61825 54.32368 -6.266793 0.8289580 0.4847775
# > 10 66.50349 52.27324 -5.695914 0.6358795 0.4791817

# Obtain final mean estimate 
main_metrics <- colMeans(test_metrics_main)
main_metrics
# >       rmse        mae       bias   bias_pct         r2 
# > 67.9369295 53.1035724 -4.3391080  0.9699695  0.4738882

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
# >                  [,1]         [,2]         [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]        [,10]
# > rmse      9.373295477  9.727821059  8.881910314  7.630985574  9.733941631  7.737898452  7.345338690  7.319875427 12.275417515  6.971502612
# > mae       5.603916363  6.238639041  5.733037945  5.339503025  6.373635817  5.478797816  5.476371720  5.203158753  6.288090033  4.266022687
# > bias     13.132588220 13.364180488 13.461582223 12.160738826 13.992640072 12.806980325 13.830046361 14.471605338 15.494266736 11.815107126
# > bias_pct  0.466112096  0.515335441  0.550076033  0.486903634  0.513520449  0.499427349  0.524115261  0.568749745  0.640625144  0.472751606
# > r2        0.001307244  0.001662401  0.001532179  0.001445917  0.001740799  0.001846358  0.001501883  0.001469333  0.001148968  0.001246618
sampling_var <- rowMeans(sampling_var_matrix)
sampling_var
# >       rmse         mae        bias    bias_pct          r2 
# > 8.69979867  5.60011732 13.45297357  0.52376168  0.00149017 

# Imputation variance
imputation_var <- colSums((test_metrics_main - matrix(main_metrics, nrow = M, ncol = 5, byrow = TRUE))^2) / (M - 1)
imputation_var
# >         rmse          mae         bias     bias_pct           r2 
# > 2.3756124571 1.2277571454 1.8483108516 0.0596809521 0.0002005872 

# Final error variance
var_final <- sampling_var + (1 + 1/M) * imputation_var
var_final
# >         rmse          mae         bias     bias_pct           r2
# > 11.312972378  6.950650180 15.486115508  0.589410723  0.001710816 

# Final standard error
se_final <- sqrt(var_final)
se_final
# >       rmse        mae       bias   bias_pct         r2 
# > 3.36347623 2.63640858 3.93524021 0.76773089 0.04136201 

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
# >      RMSE     67.9369295     3.36347623  61.3446372 74.5292218 13.1845846
# >       MAE     53.1035724     2.63640858  47.9363066 58.2708383 10.3345317
# >      Bias     -4.3391080     3.93524021 -12.0520371  3.3738211 15.4258582
# >     Bias%      0.9699695     0.76773089  -0.5347554  2.4746944  3.0094498
# > R-squared      0.4738882     0.04136201   0.3928202  0.5549563  0.1621361 

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
# >    Metric         Point_estimate       Standard_error             CI_lower              CI_upper           CI_length
# >      RMSE 66.2705611873825262137 1.096622879032783215 64.12121983985564100 68.419902534909411429 4.29868269505377043
# >       MAE 52.6519350911014498706 0.884049505171120464 50.91922990041560126 54.384640281787298477 3.46541038137169721
# >      Bias  0.0000000000000474985 0.000000000004071008 -0.00000000000793153  0.000000000008026527 0.00000000001595806
# >     Bias%  1.7890719564144030862 0.070383729154191971  1.65112238217456508  1.927021530654241088 0.27589914847967600
# > R-squared  0.4491238599532508813 0.014614244854575548  0.42048046637703301  0.477767253529468749 0.05728678715243574

print(as.data.frame(valid_eval), row.names = FALSE)
# >     Metric Point_estimate Standard_error   CI_lower  CI_upper  CI_length
# >       RMSE     67.5303048     3.31159081 61.0397061 74.020903 12.9811974
# >        MAE     53.4876910     2.72420307 48.1483510 58.827031 10.6786798
# >       Bias      1.7713540     3.82289479 -5.7213821  9.264090 14.9854722
# >      Bias%      2.0207385     0.79296467  0.4665564  3.574921  3.1083644
# >  R-squared      0.3852161     0.04675589  0.2935762  0.476856  0.1832797 

print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error    CI_lower   CI_upper  CI_length
# >      RMSE     67.9369295     3.36347623  61.3446372 74.5292218 13.1845846
# >       MAE     53.1035724     2.63640858  47.9363066 58.2708383 10.3345317
# >      Bias     -4.3391080     3.93524021 -12.0520371  3.3738211 15.4258582
# >     Bias%      0.9699695     0.76773089  -0.5347554  2.4746944  3.0094498
# > R-squared      0.4738882     0.04136201   0.3928202  0.5549563  0.1621361

