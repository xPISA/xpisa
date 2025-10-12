# ---- Script Decription ----
#
# Explore variables of interest and data preprocessing
#
# ---- Set-up ----

setwd("~/projects/pisa")

# Load required libraries
library(haven)      # For reading .sav (SPSS) files
library(tidyverse)  # Data manipulation and visualization; 
# Includes dplyr, tibble, purrr, ggplot2, readr, tidyr, etc.
#library(dplyr)     # Included in tidyverse; for data manipulation (e.g., summarise, mutate)
#library(tibble)    # Included in tidyverse; for creating and handling tidy data frames
#library(purrr)     # Included in tidyverse; for functional programming and iteration
library(intsvy)     # For analyzing PISA/IEA data (plausible values, BRR, etc.)

# # --- Different Ways to Load PISA 2022 Student Data: Canada subset ---
# # Preserve but not showing user-defined missing values (e.g., 95, 97, 99)
# pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE)
# 
# # Convert only labelled variables to factors using their value labels 
# # Caution! ESCS, numeric, is converted to factor, which is not correct
# pisa_2022_student_canada1 <- haven::as_factor(pisa_2022_student_canada, only_labelled = TRUE)
# 
# # Strip 'labelled' class to reveal raw numeric or character values (e.g., 95, 97, 99)
# pisa_2022_student_canada2 <- pisa_2022_student_canada %>%
#   mutate(across(where(is.labelled), ~ unclass(.)))

# Load Canada-only PISA 2022 student, school, and merged data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE) 
pisa_2022_school_canada <- read_sav("data/pisa2022/CY08MSP_SCH_QQQ_CAN.SAV", user_na = TRUE) 
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE)

pisa_2022_student_canada1 <- haven::as_factor(pisa_2022_student_canada, only_labelled = TRUE)
pisa_2022_student_canada2 <- pisa_2022_student_canada %>% 
  mutate(across(where(is.labelled), ~ unclass(.)))

# Load metadata
metadata_student <- read_csv("data/pisa2022/metadata_student.csv", show_col_types = FALSE)
metadata_school <- read_csv("data/pisa2022/metadata_school.csv", show_col_types = FALSE)

# Load missing summary
missing_summary_student <- read_csv("data/pisa2022/missing_summary_student.csv", show_col_types = FALSE)
missing_summary_school <- read_csv("data/pisa2022/missing_summary_school.csv", show_col_types = FALSE)

# Load combined data: metadata + missing summary
metadata_missing_student <- read_csv("data/pisa2022/metadata_missing_student.csv", show_col_types = FALSE)
metadata_missing_school <- read_csv("data/pisa2022/metadata_missing_school.csv", show_col_types = FALSE)

# ---- Explore and Identify Variables of Interests (VOI) ----

## ---- Student Questionnaire Derived Variables (DVs) ----

### Simple DVs (43)
stuq_simple_dvs <- c(
  # Basic demographics (Module 1)
  "AGE", "GRADE", "ST004D01T",
  "ST003D02T", "ST003D03T", "ST001D01T",   # extra variables listed in Table 19.A7 (PISA 2022 Technical Report)
  # Economic, social and cultural status (Module 2)
  "MISCED", "FISCED", "HISCED", "PAREDINT",
  "OCOD1", "OCOD2", "BMMJ1", "BFMJ2", "HISEI",
  # Educational pathways and post-secondary aspirations (Module 3)
  "DURECEC", "ISCEDP", "PROGN", "REPEAT", "MISSSC", "SKIPPING", "TARDYSD",
  "EXPECEDU", "OCOD3", "BSMJ", "SISCO",
  # Migration and language exposure (Module 4)
  "COBN_S", "COBN_M", "COBN_F", "IMMIG", "LANGN",
  # Subject-specific beliefs, attitudes, feelings and behaviours (Module 7)
  "MATHMOT", "MATHEASE", "MATHPREF",
  # Out-of-school experiences (Module 10)
  "EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME",
  # Country-specific home items + career info (Table 19.A7)
  "ST250D06JA", "ST250D07JA", "ST251D08JA", "ST251D09JA", "ST330D10WA"
)
# Sanity check
# stopifnot(length(stuq_simple_dvs) == 43)
# setdiff(stuq_simple_dvs, names(pisa_2022_student_canada))
# setdiff(stuq_simple_dvs, names(pisa_2022_canada_merged))
# stopifnot(all(stuq_simple_dvs %in% names(pisa_2022_student_canada)))
# stopifnot(all(stuq_simple_dvs %in% names(pisa_2022_canada_merged)))

### IRT-scaled DVs (42)
stuq_irt_scales <- c(
  # Economic, social and cultural status (Module 2)
  "HOMEPOS", "ICTRES",
  # Educational pathways and post-secondary aspirations (Module 3)
  "INFOSEEK",
  # School culture and climate (Module 6)
  "BULLIED", "FEELSAFE", "TEACHSUP", "RELATST", "SCHRISK", "BELONG",
  # Subject-specific beliefs, attitudes, feelings, and behaviours (Module 7)
  "GROSAGR", "ANXMAT", "MATHEFF", "MATHEF21", "MATHPERS", "FAMCON",
  # General social and emotional characteristics (Module 8)
  "ASSERAGR", "COOPAGR", "CURIOAGR", "EMOCOAGR", "EMPATAGR", "PERSEVAGR", "STRESAGR",
  # Exposure to mathematics content (Module 15)
  "EXPOFA", "EXPO21ST",
  # Mathematics teacher behaviour (Module 16)
  "COGACRCO", "COGACMCO", "DISCLIM",
  # Parental/guardian involvement and support (Module 19)
  "FAMSUP",
  # Creative thinking (Module 20)
  "CREATFAM", "CREATSCH", "CREATEFF", "CREATOP", "IMAGINE", "OPENART", "CREATAS", "CREATOOS",
  # Global crises (Module 21)
  "FAMSUPSL", "FEELLAH", "PROBSELF", "SDLEFF", "SCHSUST", "LEARRES"
)
# Sanity Check
# stopifnot(length(stuq_irt_scales) == 42)
# setdiff(stuq_irt_scales, names(pisa_2022_student_canada))
# setdiff(stuq_irt_scales, names(pisa_2022_canada_merged))
# stopifnot(all(stuq_irt_scales %in% names(pisa_2022_student_canada)))
# stopifnot(all(stuq_irt_scales %in% names(pisa_2022_canada_merged)))

### Complex composite (1)
stuq_composite <- c("ESCS")   # Index of economic, social and cultural status 

### Combined list (all 86)
stuq_derived_vars <- c(stuq_simple_dvs, stuq_irt_scales, stuq_composite)
# Sanity check 
# stopifnot(length(stuq_derived_vars) == 86)
# setdiff(stuq_derived_vars, names(pisa_2022_student_canada))
# setdiff(stuq_derived_vars, names(pisa_2022_canada_merged))
# stopifnot(all(stuq_derived_vars %in% names(pisa_2022_student_canada)))
# stopifnot(all(stuq_derived_vars %in% names(pisa_2022_canada_merged)))

### ---- Student DVs: metadata, missing, combined ----

# # Quick Structural checks
# stopifnot(
#   all(c("variable","label","value_labels","type") %in% names(metadata_student)),
#   all(c("variable","missing_count","missing_percent") %in% names(missing_summary_student))
# )
# 
# # Metadata for the 86 Student DVs 
# stuq_dvs_metadata_student <- metadata_student %>%
#   dplyr::filter(variable %in% stuq_derived_vars) %>%
#   dplyr::mutate(variable = factor(variable, levels = stuq_derived_vars)) %>%
#   dplyr::arrange(variable) %>%
#   dplyr::mutate(variable = as.character(variable)) %>%
#   dplyr::select(variable, label, value_labels, type)
# 
# # Missing summary for the 86 Student DVs
# stuq_dvs_missing_summary_student <- missing_summary_student %>%
#   dplyr::filter(variable %in% stuq_derived_vars) %>%
#   dplyr::mutate(variable = factor(variable, levels = stuq_derived_vars)) %>%
#   dplyr::arrange(variable) %>%
#   dplyr::mutate(variable = as.character(variable)) %>%
#   dplyr::select(variable, missing_count, missing_percent)
# 
# # Combined (metadata + missing)
# stuq_dvs_metadata_missing_student <- stuq_dvs_metadata_student %>%
#   dplyr::left_join(stuq_dvs_missing_summary_student, by = "variable") %>%
#   dplyr::select(variable, label, value_labels, type, missing_count, missing_percent)
# 
# # Sanity checks on outputs
# stopifnot(
#   nrow(stuq_dvs_metadata_student) == 86,
#   nrow(stuq_dvs_missing_summary_student) == 86,
#   nrow(stuq_dvs_metadata_missing_student) == 86
# )
# 
# # Quick console status  (fix: use *_student objects)
# cat("Student DVs (metadata/missing/combined):",
#     nrow(stuq_dvs_metadata_student),
#     nrow(stuq_dvs_missing_summary_student),
#     nrow(stuq_dvs_metadata_missing_student), "\n")
# 
# # Save outputs
# readr::write_csv(stuq_dvs_metadata_student,
#                  "data/pisa2022/stuq_dvs_metadata_student.csv")
# readr::write_csv(stuq_dvs_missing_summary_student,
#                  "data/pisa2022/stuq_dvs_missing_summary_student.csv")
# readr::write_csv(stuq_dvs_metadata_missing_student,
#                  "data/pisa2022/stuq_dvs_metadata_missing_student.csv")

# Load combined data (fix: read the *_student files you just wrote)
# stuq_dvs_metadata_student <- readr::read_csv("data/pisa2022/stuq_dvs_metadata_student.csv", show_col_types = FALSE)
# stuq_dvs_missing_summary_student <- readr::read_csv("data/pisa2022/stuq_dvs_missing_summary_student.csv", show_col_types = FALSE)
stuq_dvs_metadata_missing_student <- readr::read_csv("data/pisa2022/stuq_dvs_metadata_missing_student.csv", show_col_types = FALSE)

## ---- School Questionnaire Derived Variables (DVs)----

### Simple DVs (32) 
schq_simple_dvs <- c(
  # Module 10
  "CREACTIV", "SC053D11TA", "MATHEXC", "MACTIV",
  # Module 11
  "RATCMP1", "RATTAB", "RATCMP2", "PROPSUPP", "PROADMIN", "PROMGMT", "PROOSTAF", "SCHSIZE",
  "SCHLTYPE", "PRIVATESCH", "STRATIO", "SMRATIO", "TOTMATH", "TOTSTAFF", "TOTAT",
  # Module 12
  "SCHSEL",
  # Module 13
  "SRESPCUR", "SRESPRES",
  # Module 14
  "ABGMATH", "CLSIZE", "MCLSIZE",
  # Module 17
  "PROATCE", "PROPAT6", "PROPAT7", "PROPAT8", "PROPMATH",
  # Module 21
  "SCSUPRTED", "SCSUPRT"
)
# Sanity check
# stopifnot(length(schq_simple_dvs) == 32)
# setdiff(schq_simple_dvs, names(pisa_2022_school_canada))
# setdiff(schq_simple_dvs, names(pisa_2022_canada_merged))
# stopifnot(all(schq_simple_dvs %in% names(pisa_2022_school_canada)))
# stopifnot(all(schq_simple_dvs %in% names(pisa_2022_canada_merged)))

### IRT-scaled DVs (25)
schq_irt_scales <- c(
  # Module 6
  "NEGSCLIM", "DMCVIEWS", "STUBEHA", "TEACHBEHA",
  # Module 10
  "ALLACTIV",
  # Module 11
  "EDUSHORT", "STAFFSHORT",
  # Module 13
  "EDULEAD", "INSTLEAD", "SCHAUTO", "TCHPART",
  # Module 14
  "DIGDVPOL",
  # Module 17
  "MTTRAIN",
  # Module 18
  "TEAFDBK", "STDTEST", "TDTEST",
  # Module 19
  "ENCOURPG",
  # Module 20
  "BCREATSC", "ACTCRESC", "CREENVSC", "OPENCUL",
  # Module 21
  "PROBSCRI", "SCPREPBP", "SCPREPAP", "DIGPREP"
)
# Sanity check
# stopifnot(length(schq_irt_scales) == 25)
# setdiff(schq_irt_scales, names(pisa_2022_school_canada))
# setdiff(schq_irt_scales, names(pisa_2022_canada_merged))
# stopifnot(all(schq_irt_scales %in% names(pisa_2022_school_canada)))
# stopifnot(all(schq_irt_scales %in% names(pisa_2022_canada_merged)))

# Combined list (all 57) 
schq_derived_vars <- c(schq_simple_dvs, schq_irt_scales)
# Sanity check 
# stopifnot(length(schq_derived_vars) == 57)
# setdiff(schq_derived_vars, names(pisa_2022_school_canada))
# setdiff(schq_derived_vars, names(pisa_2022_canada_merged))
# stopifnot(all(schq_derived_vars %in% names(pisa_2022_school_canada)))
# stopifnot(all(schq_derived_vars %in% names(pisa_2022_canada_merged)))

### ---- School DVs: metadata, missing summary, and combined ----

# # Quick structural checks
# stopifnot(
#   all(c("variable", "label", "value_labels", "type") %in% names(metadata_school)),
#   all(c("variable", "missing_count", "missing_percent") %in% names(missing_summary_school)),
#   length(schq_derived_vars) == 57
# )
# 
# # Metadata for the 57 School DVs 
# schq_dvs_metadata_school <- metadata_school %>%
#   dplyr::filter(variable %in% schq_derived_vars) %>%
#   dplyr::mutate(variable = factor(variable, levels = schq_derived_vars)) %>%
#   dplyr::arrange(variable) %>%
#   dplyr::mutate(variable = as.character(variable)) %>%
#   dplyr::select(variable, label, value_labels, type)
# 
# # Missing summary for the 57 School DVs (reuse the precomputed table; preserve order)
# schq_dvs_missing_summary_school <- missing_summary_school %>%
#   dplyr::filter(variable %in% schq_derived_vars) %>%
#   dplyr::mutate(variable = factor(variable, levels = schq_derived_vars)) %>%
#   dplyr::arrange(variable) %>%
#   dplyr::mutate(variable = as.character(variable)) %>%
#   dplyr::select(variable, missing_count, missing_percent)
# 
# # Combined data: metadata + missing summary
# schq_dvs_metadata_missing_school <- schq_dvs_metadata_school %>%
#   dplyr::left_join(schq_dvs_missing_summary_school, by = "variable") %>%
#   dplyr::select(variable, label, value_labels, type, missing_count, missing_percent)
# 
# # Sanity checks on outputs
# stopifnot(
#   nrow(schq_dvs_metadata_school) == 57,
#   nrow(schq_dvs_missing_summary_school) == 57,
#   nrow(schq_dvs_metadata_missing_school) == 57
# )
# 
# # quick console status
# cat("School DVs (metadata/missing/combined):",
#     nrow(schq_dvs_metadata_school), nrow(schq_dvs_missing_summary_school), nrow(schq_dvs_metadata_missing_school), "\n")
# 
# # Save outputs
# readr::write_csv(schq_dvs_metadata_school,
#                  "data/pisa2022/schq_dvs_metadata_school.csv")
# readr::write_csv(schq_dvs_missing_summary_school,
#                  "data/pisa2022/schq_dvs_missing_summary_school.csv")
# readr::write_csv(schq_dvs_metadata_missing_school,
#                  "data/pisa2022/schq_dvs_metadata_missing_school.csv")


# Load combined data
# schq_dvs_metadata_school <- read_csv("data/pisa2022/schq_dvs_metadata_school.csv", show_col_types = FALSE)
# schq_dvs_missing_summary_school <- read_csv("data/pisa2022/schq_dvs_missing_summary_school.csv",  show_col_types = FALSE)
schq_dvs_metadata_missing_school <- read_csv("data/pisa2022/schq_dvs_metadata_missing_school.csv",  show_col_types = FALSE)

## ---- Temp Data ----

# Constants
M <- 10                  # Number of plausible values
G <- 80                  # Number of BRR replicate weights

# Target varaible
pvmaths  <- paste0("PV", 1:M, "MATH")   # PV1MATH to PV10MATH

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)   # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                 # Final student weight

### ---- oos (Out-of-school experiences): EXERPRAC ----

# Predictors
oos <- c("EXERPRAC")  

# Class/Type 
class(pisa_2022_canada_merged$EXERPRAC)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double"   

## Remark:
# - Two kinds of missing: system missing (true NA) and SPSS user-defined missing.
# - haven_labelled_spss = numeric (double) with labels + SPSS missing metadata (contrast with LANGN).
# - With user_na = TRUE, SPSS user-missings are imported as NA.
# - filter(if_all(..., ~ !is.na(.))) drops both system and user-defined missings.
# - EXERPRAC is, by nature, an ordinal categorical variable.

# Value distribution (codes)
table(pisa_2022_canada_merged$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912    0 
# Value distribution (labels; drop unused levels)
table(droplevels(haven::as_factor(pisa_2022_canada_merged$EXERPRAC), useNA = "always"))
# >                             No exercise or sports 
# >                                              3807 
# >           1 time of exercising or sports per week 
# >                                               984 
# >          2 times of exercising or sports per week 
# >                                              2103 
# >          3 times of exercising or sports per week 
# >                                              1833 
# >          4 times of exercising or sports per week 
# >                                              2129 
# >          5 times of exercising or sports per week 
# >                                              2222 
# >          6 times of exercising or sports per week 
# >                                              1673 
# >          7 times of exercising or sports per week 
# >                                               705 
# >          8 times of exercising or sports per week 
# >                                              1264 
# >          9 times of exercising or sports per week 
# >                                               391 
# > 10 or more times of exercising or sports per week 
# >                                              3050 
# >                                       No Response 
# >                                              2912 
# Codes + labels + counts
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
# > v13  NA                                              <NA>    0

# Missing values
colSums(is.na(pisa_2022_canada_merged[oos]))
# > EXERPRAC 
# >     2912 

# --- Prepare modeling data ---
temp_data0 <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                  # IDs
         all_of(final_wt), all_of(rep_wts),   # Weights
         all_of(pvmaths), all_of(oos)) %>%    # PVs + predictors
  filter(if_all(all_of(oos), ~ !is.na(.)))    # Listwise deletion for predictors

temp_data0i <- pisa_2022_canada_merged %>%   # <=> temp_data0
  filter(if_all(all_of(oos), ~ !is.na(.))) %>%  
  select(CNTSCHID, CNTSTUID,                 
         all_of(final_wt), all_of(rep_wts),   
         all_of(pvmaths), all_of(oos))     

temp_data1 <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                  # IDs
         all_of(final_wt), all_of(rep_wts),   # Weights
         all_of(pvmaths), all_of(oos)) 

# --- Explore effects ---
dim(pisa_2022_student_canada); dim(pisa_2022_school_canada); dim(pisa_2022_canada_merged)
# > [1] 23073  1278
# > [1] 863 431
# > [1] 23073  1699
dim(temp_data0); dim(temp_data0i); dim(temp_data1)
# > [1] 20161    94
# > [1] 20161    94
# > [1] 23073    94

table(temp_data0$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050    0 
table(temp_data0i$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050    0 
table(temp_data1$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912    0  
table(pisa_2022_canada_merged$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912    0  

sum(temp_data0$EXERPRAC == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0$EXERPRAC)) 
# > [1] 0
sum(temp_data0i$EXERPRAC == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0i$EXERPRAC)) 
# > [1] 0
sum(temp_data1$EXERPRAC == 99, na.rm = TRUE)
# > [1] 2912
sum(is.na(temp_data1$EXERPRAC)) 
# > [1] 2912
sum(pisa_2022_canada_merged$EXERPRAC == 99, na.rm = TRUE)
# > [1] 2912
sum(is.na(pisa_2022_canada_merged$EXERPRAC)) 
# > [1] 2912

# Remark: Results are the same if we use pisa_2022_canada_merged instead of pisa_2022_canada_merged. 

### ---- oos (Out-of-school experiences): ALL ----

# Predictors
oos <- c("EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME")  

# Missing values
colSums(is.na(pisa_2022_canada_merged[oos]))
# > EXERPRAC STUDYHMW  WORKPAY WORKHOME 
# >     2912     2887     2998     2944

# --- Prepare modeling data ---
temp_data0 <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                  # IDs
         all_of(final_wt), all_of(rep_wts),   # Weights
         all_of(pvmaths), all_of(oos)) %>%    # PVs + predictors
  filter(if_all(all_of(oos), ~ !is.na(.)))    # Listwise deletion for predictors

temp_data0i <- pisa_2022_canada_merged %>%
  filter(if_all(all_of(oos), ~ !is.na(.))) %>%  
  select(CNTSCHID, CNTSTUID,                 
         all_of(final_wt), all_of(rep_wts),   
         all_of(pvmaths), all_of(oos))     

temp_data1 <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                  # IDs
         all_of(final_wt), all_of(rep_wts),   # Weights
         all_of(pvmaths), all_of(oos)) 

# --- Explore effects ---
dim(pisa_2022_student_canada); dim(pisa_2022_school_canada); dim(pisa_2022_canada_merged)
# > [1] 23073  1278
# > [1] 863 431
# > [1] 23073  1699
dim(temp_data0); dim(temp_data0i); dim(temp_data1)
# > [1] 20003    97
# > [1] 20003    97
# > [1] 23073    94

## EXERPRAC
table(temp_data0$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 3781  979 2088 1815 2111 2199 1666  698 1252  391 3023    0 
table(temp_data0i$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 3781  979 2088 1815 2111 2199 1666  698 1252  391 3023    0  
table(temp_data1$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912    0  
table(pisa_2022_canada_merged$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912    0  
sum(temp_data0$EXERPRAC == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0$EXERPRAC)) 
# > [1] 0
sum(temp_data0i$EXERPRAC == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0i$EXERPRAC)) 
# > [1] 0
sum(temp_data1$EXERPRAC == 99, na.rm = TRUE)
# > [1] 2912
sum(is.na(temp_data1$EXERPRAC)) 
# > [1] 2912
sum(pisa_2022_canada_merged$EXERPRAC == 99, na.rm = TRUE)
# > [1] 2912
sum(is.na(pisa_2022_canada_merged$EXERPRAC)) 
# > [1] 2912

## STUDYHMW
table(temp_data0$STUDYHMW, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 2510 1282 2406 2199 2535 2480 2016  942 1114  408 2111    0 
table(temp_data0i$STUDYHMW, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 2510 1282 2406 2199 2535 2480 2016  942 1114  408 2111    0 
table(temp_data1$STUDYHMW, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 2535 1295 2426 2224 2553 2511 2028  953 1125  411 2125 2887    0  
table(pisa_2022_canada_merged$STUDYHMW, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 2535 1295 2426 2224 2553 2511 2028  953 1125  411 2125 2887    0 
sum(temp_data0$STUDYHMW == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0$STUDYHMW)) 
# > [1] 0
sum(temp_data0i$STUDYHMW == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0i$STUDYHMW)) 
# > [1] 0
sum(temp_data1$STUDYHMW == 99, na.rm = TRUE)
# > [1] 2887
sum(is.na(temp_data1$STUDYHMW)) 
# > [1] 2887
sum(pisa_2022_canada_merged$STUDYHMW == 99, na.rm = TRUE)
# > [1] 2887
sum(is.na(pisa_2022_canada_merged$STUDYHMW)) 
# > [1] 2887

## WORKPAY
table(temp_data0$WORKPAY, useNA="always")
# >    0     1     2     3     4     5     6     7     8     9    10  <NA> 
# > 11133  1017  1771  1307  1276   810   913   250   539   138   849     0
table(temp_data0i$WORKPAY, useNA="always")
# >    0     1     2     3     4     5     6     7     8     9    10  <NA> 
# > 11133  1017  1771  1307  1276   810   913   250   539   138   849     0
table(temp_data1$WORKPAY, useNA="always")
# >     0     1     2     3     4     5     6     7     8     9    10    99  <NA> 
# > 11157  1021  1777  1319  1287   814   915   252   542   140   851  2998     0 
table(pisa_2022_canada_merged$WORKPAY, useNA="always")
# >     0     1     2     3     4     5     6     7     8     9    10    99  <NA> 
# > 11157  1021  1777  1319  1287   814   915   252   542   140   851  2998     0 
sum(temp_data0$WORKPAY == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0$WORKPAY)) 
# > [1] 0
sum(temp_data0i$WORKPAY == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0i$WORKPAY)) 
# > [1] 0
sum(temp_data1$WORKPAY == 99, na.rm = TRUE)
# > [1] 2998
sum(is.na(temp_data1$WORKPAY)) 
# > [1] 2998
sum(pisa_2022_canada_merged$WORKPAY == 99, na.rm = TRUE)
# > [1] 2998
sum(is.na(pisa_2022_canada_merged$WORKPAY)) 
# > [1] 2998

## WORKHOME
table(temp_data0$WORKHOME, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 3287 1290 2103 1733 1864 2238 1579  879 1160  553 3317    0  
table(temp_data0i$WORKHOME, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 3287 1290 2103 1733 1864 2238 1579  879 1160  553 3317    0  
table(temp_data1$WORKHOME, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3297 1298 2119 1748 1880 2256 1589  885 1165  556 3336 2944    0 
table(pisa_2022_canada_merged$WORKHOME, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3297 1298 2119 1748 1880 2256 1589  885 1165  556 3336 2944    0 
sum(temp_data0$WORKHOME == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0$WORKHOME)) 
# > [1] 0
sum(temp_data0i$WORKHOME == 99, na.rm = TRUE)
# > [1] 0
sum(is.na(temp_data0i$WORKHOME)) 
# > [1] 0
sum(temp_data1$WORKHOME == 99, na.rm = TRUE)
# > [1] 2944
sum(is.na(temp_data1$WORKHOME)) 
# > [1] 2944
sum(pisa_2022_canada_merged$WORKHOME == 99, na.rm = TRUE)
# > [1] 2944
sum(is.na(pisa_2022_canada_merged$WORKHOME)) 
# > [1] 2944


### ---- LANGN: Language spoken at home----

# Predictors
voi_cat <- c("LANGN")                                        # nominal

# Class/Type
class(pisa_2022_canada_merged$LANGN)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 

## Remark:
# - haven_labelled = numeric with value labels, but no SPSS user-missing metadata (contrast with EXERPRAC).
# - Special codes like 999 remain as values, not auto-converted to NA (recode if needed).
# - With user_na = TRUE, only system missings are NA.
# - LANGN is, by nature, a nominal categorical variable.

# Missing values
colSums(is.na(pisa_2022_canada_merged[voi_cat]))
# > LANGN 
# >     0

# Value distribution (codes)
table(pisa_2022_canada_merged$LANGN, useNA="always")
# >   313   493   807   999  <NA> 
# > 14818  3553  3094  1608     0 
# Value distribution (labels; drop unused levels)
table(droplevels(haven::as_factor(pisa_2022_canada_merged$LANGN), useNA = "always"))
# > English                 French Another language (CAN)                Missing 
# >   14818                   3553                   3094                   1608
# Codes + labels + counts
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

# --- Prepare modeling data ---

# filter
temp_data0 <- pisa_2022_canada_merged %>%
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
  ) %>%
  filter(if_all(all_of(voi_cat), ~ !is.na(.)))

# do not filter
temp_data1 <- pisa_2022_canada_merged %>%
  select(                      
    CNTSCHID, CNTSTUID,                                       # IDs
    all_of(final_wt), all_of(rep_wts),                        # Weights
    all_of(pvmaths), all_of(voi_cat)                          # PVs + predictors
  ) %>%
  mutate(
    across(                                                   # label → factor for categorical VOIs
      all_of(voi_cat),
      ~ if (inherits(.x, "haven_labelled"))
        haven::as_factor(.x, levels = "labels") else as.factor(.x)
    )
  ) 

# --- Explore effects ---

dim(pisa_2022_student_canada); dim(pisa_2022_school_canada); dim(pisa_2022_canada_merged)
# > [1] 23073  1278
# > [1] 863 431
# > [1] 23073  1699

dim(temp_data0); dim(temp_data1)
# > [1] 21465    94
# > [1] 23073    94

# Class/Type
class(temp_data0$LANGN)
# > [1] "factor"

# Missing values
colSums(is.na(temp_data0[voi_cat]))
# > LANGN 
# >     0

# Value distribution (codes)
#table(temp_data0$LANGN, useNA="always")
# Value distribution (labels; drop unused levels)
table(droplevels(haven::as_factor(temp_data0$LANGN), useNA = "always"))
# > English                 French Another language (CAN)                
# >   14818                   3553                   3094                  
# Codes + labels + counts
data.frame(code=as.numeric(names(table(temp_data0$LANGN, useNA="always"))),
           label=names(attr(temp_data0$LANGN,"labels"))[
             match(as.numeric(names(table(temp_data0$LANGN, useNA="always"))),
                   unname(attr(temp_data0$LANGN,"labels")))],
           n=as.integer(table(temp_data0$LANGN, useNA="always")),
           row.names=NULL)

# (Unfinished) To continue...



### ---- SCHLTYPE: Language spoken at home----

# Predictors
voi_cat <- c("SCHLTYPE")                                        # nominal

# Class/Type
class(pisa_2022_canada_merged$SCHLTYPE)
# > [1] "haven_labelled" "vctrs_vctr"     "double" 

## Remark:
# - haven_labelled = numeric with value labels, but no SPSS user-missing metadata (contrast with EXERPRAC).
# - Special codes like 999 remain as values, not auto-converted to NA (recode if needed).
# - With user_na = TRUE, only system missings are NA.
# - SCHLTYPE is, by nature, a nominal categorical variable.

# Missing values
colSums(is.na(pisa_2022_canada_merged[voi_cat]))
# > SCHLTYPE 
# >     0

# Value distribution (codes)
table(pisa_2022_canada_merged$SCHLTYPE, useNA="always")
# >   313   493   807   999  <NA> 
# > 14818  3553  3094  1608     0 
# Value distribution (labels; drop unused levels)
table(droplevels(haven::as_factor(pisa_2022_canada_merged$SCHLTYPE), useNA = "always"))
# > English                 French Another language (CAN)                Missing 
# >   14818                   3553                   3094                   1608
# Codes + labels + counts
data.frame(code=as.numeric(names(table(pisa_2022_canada_merged$SCHLTYPE, useNA="always"))),
           label=names(attr(pisa_2022_canada_merged$SCHLTYPE,"labels"))[
             match(as.numeric(names(table(pisa_2022_canada_merged$SCHLTYPE, useNA="always"))),
                   unname(attr(pisa_2022_canada_merged$SCHLTYPE,"labels")))],
           n=as.integer(table(pisa_2022_canada_merged$SCHLTYPE, useNA="always")),
           row.names=NULL)
# >   code                  label     n
# > 1  313                English 14818
# > 2  493                 French  3553
# > 3  807 Another language (CAN)  3094
# > 4  999                Missing  1608
# > 5   NA                   <NA>     0

# --- Prepare modeling data ---

# filter
temp_data0 <- pisa_2022_canada_merged %>%
  select(                      
    CNTSCHID, CNTSTUID,                                       # IDs
    all_of(final_wt), all_of(rep_wts),                        # Weights
    all_of(pvmaths), all_of(voi_cat)                          # PVs + predictors
  ) %>%
  mutate(
    SCHLTYPE = na_if(SCHLTYPE, 999),                                # Missing 999 -> NA
    across(                                                   # label → factor for categorical VOIs
      all_of(voi_cat),
      ~ if (inherits(.x, "haven_labelled"))
        haven::as_factor(.x, levels = "labels") else as.factor(.x)
    )
  ) %>%
  filter(if_all(all_of(voi_cat), ~ !is.na(.)))

# do not filter
temp_data1 <- pisa_2022_canada_merged %>%
  select(                      
    CNTSCHID, CNTSTUID,                                       # IDs
    all_of(final_wt), all_of(rep_wts),                        # Weights
    all_of(pvmaths), all_of(voi_cat)                          # PVs + predictors
  ) %>%
  mutate(
    across(                                                   # label → factor for categorical VOIs
      all_of(voi_cat),
      ~ if (inherits(.x, "haven_labelled"))
        haven::as_factor(.x, levels = "labels") else as.factor(.x)
    )
  ) 

# --- Explore effects ---

dim(pisa_2022_student_canada); dim(pisa_2022_school_canada); dim(pisa_2022_canada_merged)
# > [1] 23073  1278
# > [1] 863 431
# > [1] 23073  1699

dim(temp_data0); dim(temp_data1)
# > [1] 21465    94
# > [1] 23073    94

# Class/Type
class(temp_data0$SCHLTYPE)
# > [1] "factor"

# Missing values
colSums(is.na(temp_data0[voi_cat]))
# > SCHLTYPE 
# >     0

# Value distribution (codes)
#table(temp_data0$SCHLTYPE, useNA="always")
# Value distribution (labels; drop unused levels)
table(droplevels(haven::as_factor(temp_data0$SCHLTYPE), useNA = "always"))
# > English                 French Another language (CAN)                
# >   14818                   3553                   3094                  
# Codes + labels + counts
data.frame(code=as.numeric(names(table(temp_data0$SCHLTYPE, useNA="always"))),
           label=names(attr(temp_data0$SCHLTYPE,"labels"))[
             match(as.numeric(names(table(temp_data0$SCHLTYPE, useNA="always"))),
                   unname(attr(temp_data0$SCHLTYPE,"labels")))],
           n=as.integer(table(temp_data0$SCHLTYPE, useNA="always")),
           row.names=NULL)

# (Unfinished) To continue...



### ---- oos + LANGN----

# Predictors (VOI = variables of interest)
oos <- c("EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME")  
voi_num <- c("EXERPRAC", "STUDYHMW", "WORKPAY", "WORKHOME")  # numeric/ordinal
voi_cat <- c("LANGN")                                        # nominal
voi_all <- c(voi_num, voi_cat)

# Prepare modeling data
temp_data0 <- pisa_2022_student_canada %>%
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
  ) %>%
  filter(if_all(all_of(voi_all), ~ !is.na(.)))

temp_data1 <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                      # IDs
         all_of(final_wt), all_of(rep_wts),       # Weights
         all_of(pvmaths), all_of(voi_all)) %>%    # PVs + predictors
  mutate(across(
    all_of(voi_cat),
    ~ if (inherits(.x, "haven_labelled")) haven::as_factor(.x, levels = "labels") else as.factor(.x)
  )) 

temp_data2 <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                      
         all_of(final_wt), all_of(rep_wts),       
         all_of(pvmaths), all_of(voi_all)) %>%   
  mutate(across(
    all_of(voi_cat),
    ~ if (inherits(.x, "haven_labelled")) haven::as_factor(.x, levels = "labels") else as.factor(.x)
  )) %>%
  filter(if_all(all_of(voi_all), ~ !is.na(.)))     

temp_data3 <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                      
         all_of(final_wt), all_of(rep_wts),       
         all_of(pvmaths), all_of(voi_all))  

temp_data4 <- pisa_2022_student_canada %>%
  select(CNTSCHID, CNTSTUID,                      
         all_of(final_wt), all_of(rep_wts),       
         all_of(pvmaths), all_of(voi_all)) %>%
  filter(if_all(all_of(voi_all), ~ !is.na(.))) 


dim(temp_data0); dim(temp_data1); dim(temp_data2); dim(temp_data3); dim(temp_data4)
# > [1] 20003    98
# > [1] 23073    98
# > [1] 20003    98
# > [1] 23073    98
# > [1] 20003    98

table(temp_data0$LANGN, useNA="always")
table(temp_data1$LANGN, useNA="always")
table(temp_data2$LANGN, useNA="always")

table(temp_data3$LANGN, useNA="always")
# >   313   493   807   999  <NA> 
# > 14818  3553  3094  1608     0 
table(temp_data4$LANGN, useNA="always")
# >   313   493   807   999  <NA> 
# > 13786  3316  2880    21     0 

sum(pisa_2022_student_canada$EXERPRAC == 99, na.rm = TRUE)
# > [1] 2912
sum(is.na(pisa_2022_student_canada$EXERPRAC)) 
# > [1] 2912


### ---- derived_vars_selected ----
derived_vars_all <- c(stuq_derived_vars, schq_derived_vars)
length(derived_vars_all) # 143
derived_vars_all

derived_vars_selected <- names(dplyr::select(pisa_2022_canada_merged, dplyr::any_of(derived_vars_all)))[
  colMeans(is.na(dplyr::select(pisa_2022_canada_merged, dplyr::any_of(derived_vars_all)))) < 0.40
]
length(derived_vars_selected) # 119
derived_vars_selected

# Prepare modeling data
temp_data0 <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                                    # IDs
         all_of(final_wt), all_of(rep_wts),                     # Weights
         all_of(pvmaths), all_of(derived_vars_selected)) %>%    # PVs + predictors
  filter(if_all(all_of(derived_vars_selected), ~ !is.na(.)))    # Listwise deletion for predictors

temp_data1 <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                  # IDs
         all_of(final_wt), all_of(rep_wts),   # Weights
         all_of(pvmaths), all_of(derived_vars_selected)) 

dim(temp_data0); dim(temp_data1)
# > [1] 3524  212
# > [1] 23073   212

table(temp_data0$EXERPRAC, useNA="always")
# >   0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 605  181  399  410  418  403  292  137  207   58  414    0 
table(temp_data1$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912    0 
table(pisa_2022_canada_merged$EXERPRAC, useNA="always")
# >    0    1    2    3    4    5    6    7    8    9   10   99 <NA> 
# > 3807  984 2103 1833 2129 2222 1673  705 1264  391 3050 2912    0 

sum(pisa_2022_canada_merged$EXERPRAC == 99, na.rm = TRUE)
# > [1] 2912
sum(is.na(pisa_2022_canada_merged$EXERPRAC)) 
# > [1] 2912

## ---- Variable-by-Variable Investigation ----

# Constants
M <- 10                                   # Number of plausible values
G <- 80                                   # Number of BRR replicate weights

# Variable names
pvmaths  <- paste0("PV", 1:M, "MATH")     # PV1MATH to PV10MATH
final_wt <- "W_FSTUWT"                    # Final student weight
rep_wts  <- paste0("W_FSTURWT", 1:G)      # BRR replicate weights: W_FSTURWT1 to W_FSTURWT80


#### ---- ST004D01T: Student (Standardized) Gender + REGION: REGION----

# Define predictor variables of interest 
stuq_dvs <- c("ST004D01T", "REGION") 

sapply(pisa_2022_canada_merged[stuq_dvs], class)
colSums(is.na(pisa_2022_canada_merged[stuq_dvs]))

# Subset data for analysis
temp_data0 <- pisa_2022_student_canada %>%
  filter(!is.na(ST004D01T), !is.na(REGION)) %>%                 
  mutate(
    ST004D01T = if (inherits(ST004D01T, "haven_labelled"))
      haven::as_factor(ST004D01T, levels = "labels") else as.factor(ST004D01T),
    REGION    = if (inherits(REGION, "haven_labelled"))
      haven::as_factor(REGION, levels = "labels") else as.factor(REGION)
  ) %>%
  select(
    CNTSCHID, CNTSTUID, ST004D01T, REGION, W_FSTUWT,
    all_of(rep_wts), all_of(pvmaths)
  ) 

temp_data1 <- pisa_2022_student_canada %>%
  mutate(
    ST004D01T = if (inherits(ST004D01T, "haven_labelled"))
      haven::as_factor(ST004D01T, levels = "labels") else as.factor(ST004D01T),
    REGION    = if (inherits(REGION, "haven_labelled"))
      haven::as_factor(REGION, levels = "labels") else as.factor(REGION)
  ) %>%
  select(
    CNTSCHID, CNTSTUID, ST004D01T, REGION, W_FSTUWT,
    all_of(rep_wts), all_of(pvmaths)
  )

dim(temp_data0); dim(temp_data1)
# > [1] 23031    95
# > [1] 23073    95

# Mean performance for Mathematics
pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), data = temp_data0)
# >    Freq   Mean s.e.    SD  s.e
# > 1 23031 496.99 1.56 94.02 0.85
pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), data = temp_data1)
# >    Freq   Mean s.e.    SD  s.e
# > 1 23073 496.95 1.56 94.01 0.85

# Means by gender for Mathematics
pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), by = "ST004D01T", data = temp_data0)
# >   ST004D01T  Freq   Mean s.e.    SD  s.e
# > 1    Female 11377 490.71 1.68 88.08 0.97
# > 2      Male 11654 503.04 1.89 99.03 1.22
pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), by = "ST004D01T", data = temp_data1)
# >        ST004D01T  Freq   Mean  s.e.    SD   s.e
# > 1         Female 11377 490.71  1.68 88.08  0.97
# > 2           Male 11654 503.04  1.89 99.03  1.22
# > 3 Not Applicable    42 454.07 20.95 67.85 11.85

# Percentiles by gender for Mathematics
pisa.per.pv(pvlabel=paste0("PV",1:10,"MATH"), per=c(10, 25, 50, 75, 90), by="ST004D01T", data=temp_data0)
# >    ST004D01T Percentiles  Score Std. err.
# > 1     Female          10 377.48      2.44
# > 2     Female          25 428.58      2.01
# > 3     Female          50 489.63      1.96
# > 4     Female          75 551.30      2.59
# > 5     Female          90 605.83      2.67
# > 6       Male          10 372.79      3.13
# > 7       Male          25 433.09      2.47
# > 8       Male          50 503.51      2.17
# > 9       Male          75 571.77      2.88
# > 10      Male          90 631.01      3.10
pisa.per.pv(pvlabel=paste0("PV",1:10,"MATH"), per=c(10, 25, 50, 75, 90), by="ST004D01T", data=temp_data1)
# >         ST004D01T Percentiles  Score Std. err.
# > 1          Female          10 377.48      2.44
# > 2          Female          25 428.58      2.01
# > 3          Female          50 489.63      1.96
# > 4          Female          75 551.30      2.59
# > 5          Female          90 605.83      2.67
# > 6            Male          10 372.79      3.13
# > 7            Male          25 433.09      2.47
# > 8            Male          50 503.51      2.17
# > 9            Male          75 571.77      2.88
# > 10           Male          90 631.01      3.10
# > 11 Not Applicable          10 359.15     37.98
# > 12 Not Applicable          25 397.74     42.14
# > 13 Not Applicable          50 471.38     46.84
# > 14 Not Applicable          75 494.41     38.73
# > 15 Not Applicable          90 531.03     31.80

pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), by = "REGION", data = temp_data0)
# >                               REGION Freq   Mean s.e.    SD  s.e
# > 1  Canada: Newfoundland and Labrador 1053 458.54 5.54 86.19 2.44
# > 2       Canada: Prince Edward Island  356 478.28 6.58 88.34 3.95
# > 3                Canada: Nova Scotia 1578 470.48 3.65 91.15 2.44
# > 4              Canada: New Brunswick 1639 467.74 3.12 90.15 2.17
# > 5                     Canada: Quebec 4135 513.63 3.89 93.57 1.90
# > 6                    Canada: Ontario 5914 495.28 2.97 92.55 1.62
# > 7                   Canada: Manitoba 2628 470.45 2.68 85.60 1.72
# > 8               Canada: Saskatchewan 2270 467.60 2.64 86.44 1.94
# > 9                    Canada: Alberta 1329 503.56 5.70 98.36 2.64
# > 10          Canada: British Columbia 2129 496.30 4.42 93.04 1.91

pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), by = "REGION", data = temp_data1)
# >                               REGION Freq   Mean s.e.    SD  s.e
# > 1  Canada: Newfoundland and Labrador 1053 458.54 5.54 86.19 2.44
# > 2       Canada: Prince Edward Island  357 477.70 6.64 88.47 3.93
# > 3                Canada: Nova Scotia 1590 470.32 3.60 91.17 2.43
# > 4              Canada: New Brunswick 1653 467.67 3.09 90.02 2.17
# > 5                     Canada: Quebec 4137 513.62 3.89 93.56 1.90
# > 6                    Canada: Ontario 5918 495.22 2.97 92.56 1.62
# > 7                   Canada: Manitoba 2629 470.47 2.68 85.59 1.72
# > 8               Canada: Saskatchewan 2276 467.65 2.63 86.40 1.94
# > 9                    Canada: Alberta 1330 503.54 5.70 98.29 2.62
# > 10          Canada: British Columbia 2130 496.30 4.42 93.04 1.91

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="ST004D01T", data=temp_data0)
# >             Estimate Std. Error t value
# > (Intercept)   490.71       1.68  292.93
# > ST004D01T2     12.33       1.74    7.09
# > R-squared       0.00       0.00    3.52
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="ST004D01T", data=temp_data1)
# >             Estimate Std. Error t value
# > (Intercept)   490.71       1.68  292.93
# > ST004D01T2     12.33       1.74    7.09
# > ST004D01T7    -36.64      21.11   -1.74
# > R-squared       0.00       0.00    3.56

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="REGION", data=temp_data0)
# >                                    Estimate Std. Error t value
# > (Intercept)                          458.54       5.54   82.83
# > REGIONCanada: Prince Edward Island    19.75       9.09    2.17
# > REGIONCanada: Nova Scotia             11.94       6.62    1.80
# > REGIONCanada: New Brunswick            9.20       6.22    1.48
# > REGIONCanada: Quebec                  55.10       6.89    8.00
# > REGIONCanada: Ontario                 36.74       6.36    5.78
# > REGIONCanada: Manitoba                11.92       5.53    2.16
# > REGIONCanada: Saskatchewan             9.07       6.34    1.43
# > REGIONCanada: Alberta                 45.02       7.51    5.99
# > REGIONCanada: British Columbia        37.77       7.13    5.29
# > R-squared                              0.02       0.00    5.60
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="REGION", data=temp_data1)
# >                                    Estimate Std. Error t value
# > (Intercept)                          458.54       5.54   82.83
# > REGIONCanada: Prince Edward Island    19.17       9.12    2.10
# > REGIONCanada: Nova Scotia             11.78       6.59    1.79
# > REGIONCanada: New Brunswick            9.14       6.18    1.48
# > REGIONCanada: Quebec                  55.09       6.88    8.00
# > REGIONCanada: Ontario                 36.69       6.36    5.77
# > REGIONCanada: Manitoba                11.93       5.53    2.16
# > REGIONCanada: Saskatchewan             9.12       6.34    1.44
# > REGIONCanada: Alberta                 45.00       7.51    6.00
# > REGIONCanada: British Columbia        37.77       7.13    5.29
# > R-squared                              0.02       0.00    5.61


pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="ST004D01T", by = "REGION", data=temp_data0)
# > $`Canada: Newfoundland and Labrador`
# >               Estimate Std. Error t value
# > (Intercept)     457.34       6.07   75.36
# > ST004D01TMale     2.28       6.39    0.36
# > R-squared         0.00       0.00    0.20

# > $`Canada: Prince Edward Island`
# >               Estimate Std. Error t value
# > (Intercept)     466.73       7.80   59.86
# > ST004D01TMale    22.55      10.24    2.20
# > R-squared         0.02       0.01    1.11

# > $`Canada: Nova Scotia`
# >             Estimate Std. Error t value
# > (Intercept)     466.71       4.48  104.28
# > ST004D01TMale     7.30       5.64    1.29
# > R-squared         0.00       0.00    0.73

# > $`Canada: New Brunswick`
# >             Estimate Std. Error t value
# > (Intercept)     463.43       4.29  108.05
# > ST004D01TMale     8.28       5.92    1.40
# > R-squared         0.00       0.00    0.71

# > $`Canada: Quebec`
# >             Estimate Std. Error t value
# > (Intercept)     509.13       4.27  119.29
# > ST004D01TMale     8.89       3.66    2.43
# > R-squared         0.00       0.00    1.22

# > $`Canada: Ontario`
# >               Estimate Std. Error t value
# > (Intercept)     488.36       3.01  162.03
# > ST004D01TMale    13.39       2.98    4.49
# > R-squared         0.01       0.00    2.23

# > $`Canada: Manitoba`
# >               Estimate Std. Error t value
# > (Intercept)     466.77       3.71  125.85
# > ST004D01TMale     7.28       4.49    1.62
# > R-squared         0.00       0.00    0.81

# > $`Canada: Saskatchewan`
# >               Estimate Std. Error t value
# > (Intercept)     460.80       3.33  138.34
# > ST004D01TMale    12.98       4.47    2.90
# > R-squared         0.01       0.00    1.39

# > $`Canada: Alberta`
# >               Estimate Std. Error t value
# > (Intercept)     495.42       6.13   80.88
# > ST004D01TMale    16.48       5.98    2.76
# > R-squared         0.01       0.00    1.42

# > $`Canada: British Columbia`
# >               Estimate Std. Error t value
# > (Intercept)     488.18       5.22   93.46
# > ST004D01TMale    16.13       6.17    2.62
# > R-squared         0.01       0.01    1.23

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="ST004D01T", by = "REGION", data=temp_data1)
# > $`Canada: Newfoundland and Labrador`
# >                          Estimate Std. Error t value
# > (Intercept)     457.34       6.07   75.36
# > ST004D01TMale     2.28       6.39    0.36
# > R-squared         0.00       0.00    0.20

# > $`Canada: Prince Edward Island`
# >                         Estimate Std. Error t value
# > (Intercept)               466.73       7.80   59.86
# > ST004D01TMale              22.55      10.24    2.20
# > ST004D01TNot Applicable   -96.17      25.91   -3.71
# > R-squared                   0.02       0.02    1.39

# > $`Canada: Nova Scotia`
# >                         Estimate Std. Error t value
# > (Intercept)               466.71       4.48  104.28
# > ST004D01TMale               7.30       5.64    1.29
# > ST004D01TNot Applicable   -24.88      31.39   -0.79
# > R-squared                   0.00       0.00    0.83

# > $`Canada: New Brunswick`
# >                         Estimate Std. Error t value
# > (Intercept)               463.43       4.29  108.05
# > ST004D01TMale               8.28       5.92    1.40
# > ST004D01TNot Applicable    -2.75      24.24   -0.11
# > R-squared                   0.00       0.00    0.72

# > $`Canada: Quebec`
# >                         Estimate Std. Error t value
# > (Intercept)               509.13       4.27  119.29
# > ST004D01TMale               8.89       3.66    2.43
# > ST004D01TNot Applicable   -35.48      20.47   -1.73
# > R-squared                   0.00       0.00    1.24

# > $`Canada: Ontario`
# > Estimate Std. Error t value
# > (Intercept)               488.36       3.01  162.03
# > ST004D01TMale              13.39       2.98    4.49
# > ST004D01TNot Applicable   -79.26      33.99   -2.33
# > R-squared                   0.01       0.00    2.38

# > $`Canada: Manitoba`
# >                         Estimate Std. Error t value
# > (Intercept)               466.77       3.71  125.85
# > ST004D01TMale               7.28       4.49    1.62
# > ST004D01TNot Applicable    73.65      13.40    5.50
# > R-squared                   0.00       0.00    0.88

# > $`Canada: Saskatchewan`
# >                         Estimate Std. Error t value
# > (Intercept)               460.80       3.33  138.34
# > ST004D01TMale              12.98       4.47    2.90
# > ST004D01TNot Applicable    27.45      28.36    0.97
# > R-squared                   0.01       0.00    1.44

# > $`Canada: Alberta`
# >                         Estimate Std. Error t value
# > (Intercept)               495.42       6.13   80.88
# > ST004D01TMale              16.48       5.98    2.76
# > ST004D01TNot Applicable    -4.70      29.46   -0.16
# > R-squared                   0.01       0.00    1.47

# > $`Canada: British Columbia`
# >                         Estimate Std. Error t value
# > (Intercept)               488.18       5.22   93.46
# > ST004D01TMale              16.13       6.17    2.62
# > ST004D01TNot Applicable    -7.17      23.14   -0.31
# > R-squared                   0.01       0.01    1.23

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="REGION", by = "ST004D01T", data=temp_data0)
# > $Female
# >                                    Estimate Std. Error t value
# > (Intercept)                          457.34       6.07   75.36
# > REGIONCanada: Prince Edward Island     9.40       9.82    0.96
# > REGIONCanada: Nova Scotia              9.37       7.14    1.31
# > REGIONCanada: New Brunswick            6.09       7.18    0.85
# > REGIONCanada: Quebec                  51.80       6.84    7.57
# > REGIONCanada: Ontario                 31.03       6.83    4.54
# > REGIONCanada: Manitoba                 9.44       6.70    1.41
# > REGIONCanada: Saskatchewan             3.47       6.78    0.51
# > REGIONCanada: Alberta                 38.09       8.76    4.35
# > REGIONCanada: British Columbia        30.84       7.63    4.04
# > R-squared                              0.02       0.00    4.73

# > $Male
# >                                    Estimate Std. Error t value
# > (Intercept)                          459.61       6.66   69.03
# > REGIONCanada: Prince Edward Island    29.67      12.17    2.44
# > REGIONCanada: Nova Scotia             14.40       8.49    1.70
# > REGIONCanada: New Brunswick           12.10       8.13    1.49
# > REGIONCanada: Quebec                  58.42       8.67    6.74
# > REGIONCanada: Ontario                 42.14       7.78    5.42
# > REGIONCanada: Manitoba                14.44       6.59    2.19
# > REGIONCanada: Saskatchewan            14.17       7.88    1.80
# > REGIONCanada: Alberta                 52.29       8.76    5.97
# > REGIONCanada: British Columbia        44.70       9.03    4.95
# > R-squared                              0.02       0.00    5.14
pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="REGION", by = "ST004D01T", data=temp_data1)
# > $Female
# >                                    Estimate Std. Error t value
# > (Intercept)                          457.34       6.07   75.36
# > REGIONCanada: Prince Edward Island     9.40       9.82    0.96
# > REGIONCanada: Nova Scotia              9.37       7.14    1.31
# > REGIONCanada: New Brunswick            6.09       7.18    0.85
# > REGIONCanada: Quebec                  51.80       6.84    7.57
# > REGIONCanada: Ontario                 31.03       6.83    4.54
# > REGIONCanada: Manitoba                 9.44       6.70    1.41
# > REGIONCanada: Saskatchewan             3.47       6.78    0.51
# > REGIONCanada: Alberta                 38.09       8.76    4.35
# > REGIONCanada: British Columbia        30.84       7.63    4.04
# > R-squared                              0.02       0.00    4.73

# > $Male
# >                                    Estimate Std. Error t value
# > (Intercept)                          459.61       6.66   69.03
# > REGIONCanada: Prince Edward Island    29.67      12.17    2.44
# > REGIONCanada: Nova Scotia             14.40       8.49    1.70
# > REGIONCanada: New Brunswick           12.10       8.13    1.49
# > REGIONCanada: Quebec                  58.42       8.67    6.74
# > REGIONCanada: Ontario                 42.14       7.78    5.42
# > REGIONCanada: Manitoba                14.44       6.59    2.19
# > REGIONCanada: Saskatchewan            14.17       7.88    1.80
# > REGIONCanada: Alberta                 52.29       8.76    5.97
# > REGIONCanada: British Columbia        44.70       9.03    4.95
# > R-squared                              0.02       0.00    5.14

# > $`Not Applicable`
# >                                Estimate Std. Error t value
# > (Intercept)                      370.56      26.26   14.11
# > REGIONCanada: Nova Scotia         71.26      42.77    1.67
# > REGIONCanada: New Brunswick       90.11      31.65    2.85``
# > REGIONCanada: Quebec             103.09      24.23    4.26
# > REGIONCanada: Ontario             38.54      39.97    0.96
# > REGIONCanada: Manitoba           169.86      31.60    5.38
# > REGIONCanada: Saskatchewan       117.69      38.88    3.03
# > REGIONCanada: Alberta            120.16      32.07    3.75
# > REGIONCanada: British Columbia   110.45      19.85    5.56
# > R-squared                          0.30       0.18    1.67

#### ---- SCHLTYPE: School type + REGION: REGION----

# Define predictor variables of interest 
stuq_schq_dvs <- c("SCHLTYPE", "REGION") 

sapply(pisa_2022_canada_merged[stuq_schq_dvs], class)
colSums(is.na(pisa_2022_canada_merged[stuq_schq_dvs]))

# Subset data for analysis
temp_data <- pisa_2022_canada_merged %>%
  #filter(!is.na(SCHLTYPE), !is.na(REGION)) %>%                 
  mutate(
    SCHLTYPE = if (inherits(SCHLTYPE, "haven_labelled"))
      haven::as_factor(SCHLTYPE, levels = "labels") else as.factor(SCHLTYPE),
    REGION    = if (inherits(REGION, "haven_labelled"))
      haven::as_factor(REGION, levels = "labels") else as.factor(REGION)
  ) %>%
  select(
    CNTSCHID, CNTSTUID, REGION, SCHLTYPE,W_FSTUWT,
    all_of(rep_wts), all_of(pvmaths)
  ) 

dim(temp_data)
# > [1] 23073    95

# Mean performance for Mathematics
pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), data = temp_data)
# >    Freq   Mean s.e.    SD  s.e
# > 1 23031 496.99 1.56 94.02 0.85

# Means by school type for Mathematics
pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), by = "SCHLTYPE", data = temp_data)
# >                       SCHLTYPE  Freq   Mean  s.e.    SD  s.e
# > 1          Private independent   980 543.95 12.25 87.33 3.99
# > 2 Private Government-dependent   566 548.84  9.06 82.99 3.77
# > 3                       Public 21527 493.06  1.50 93.54 0.88

# Percentiles by gender for Mathematics
pisa.per.pv(pvlabel=paste0("PV",1:10,"MATH"), per=c(10, 25, 50, 75, 90), by="SCHLTYPE", data=temp_data)
# >                        SCHLTYPE Percentiles  Score Std. err.
# > 1           Private independent          10 426.37     28.18
# > 2           Private independent          25 482.38     16.79
# > 3           Private independent          50 545.04     14.47
# > 4           Private independent          75 604.54     14.45
# > 5           Private independent          90 658.39     11.24
# > 6  Private Government-dependent          10 439.14     11.47
# > 7  Private Government-dependent          25 491.76     11.06
# > 8  Private Government-dependent          50 552.11     10.99
# > 9  Private Government-dependent          75 605.24     10.29
# > 10 Private Government-dependent          90 652.59     12.80
# > 11                       Public          10 372.19      2.31
# > 12                       Public          25 426.89      1.81
# > 13                       Public          50 492.08      1.75
# > 14                       Public          75 557.09      2.10
# > 15                       Public          90 614.50      2.15

# Means by region for Mathematics
pisa.mean.pv(pvlabel = paste0("PV", 1:10, "MATH"), by = "REGION", data = temp_data)
# >                               REGION Freq   Mean s.e.    SD  s.e
# > 1  Canada: Newfoundland and Labrador 1053 458.54 5.54 86.19 2.44
# > 2       Canada: Prince Edward Island  357 477.70 6.64 88.47 3.93
# > 3                Canada: Nova Scotia 1590 470.32 3.60 91.17 2.43
# > 4              Canada: New Brunswick 1653 467.67 3.09 90.02 2.17
# > 5                     Canada: Quebec 4137 513.62 3.89 93.56 1.90
# > 6                    Canada: Ontario 5918 495.22 2.97 92.56 1.62
# > 7                   Canada: Manitoba 2629 470.47 2.68 85.59 1.72
# > 8               Canada: Saskatchewan 2276 467.65 2.63 86.40 1.94
# > 9                    Canada: Alberta 1330 503.54 5.70 98.29 2.62
# > 10          Canada: British Columbia 2130 496.30 4.42 93.04 1.91

# Percentiles by region for Mathematics
pisa.per.pv(pvlabel=paste0("PV",1:10,"MATH"), per=c(10, 25, 50, 75, 90), by="REGION", data=temp_data)
# >                               REGION Percentiles  Score Std. err.
# > 1  Canada: Newfoundland and Labrador          10 349.02      7.37
# > 2  Canada: Newfoundland and Labrador          25 397.85      7.06
# > 3  Canada: Newfoundland and Labrador          50 457.64      6.54
# > 4  Canada: Newfoundland and Labrador          75 516.87      7.14
# > 5  Canada: Newfoundland and Labrador          90 572.67      7.97
# > 6       Canada: Prince Edward Island          10 363.33     11.45
# > 7       Canada: Prince Edward Island          25 412.85      9.30
# > 8       Canada: Prince Edward Island          50 477.81     10.98
# > 9       Canada: Prince Edward Island          75 542.22      9.27
# > 10      Canada: Prince Edward Island          90 591.16     10.95
# > 11               Canada: Nova Scotia          10 355.06      5.64
# > 12               Canada: Nova Scotia          25 402.76      5.09
# > 13               Canada: Nova Scotia          50 467.36      5.07
# > 14               Canada: Nova Scotia          75 532.71      5.42
# > 15               Canada: Nova Scotia          90 590.04      5.78
# > 16             Canada: New Brunswick          10 355.00      5.17
# > 17             Canada: New Brunswick          25 403.60      4.30
# > 18             Canada: New Brunswick          50 465.99      4.66
# > 19             Canada: New Brunswick          75 529.34      4.04
# > 20             Canada: New Brunswick          90 584.73      6.31
# > 21                    Canada: Quebec          10 390.01      5.25
# > 22                    Canada: Quebec          25 449.75      4.85
# > 23                    Canada: Quebec          50 517.44      4.68
# > 24                    Canada: Quebec          75 581.25      4.58
# > 25                    Canada: Quebec          90 630.88      4.34
# > 26                   Canada: Ontario          10 376.23      3.55
# > 27                   Canada: Ontario          25 430.89      3.08
# > 28                   Canada: Ontario          50 493.32      3.44
# > 29                   Canada: Ontario          75 556.21      4.34
# > 30                   Canada: Ontario          90 615.84      4.69
# > 31                  Canada: Manitoba          10 360.26      4.61
# > 32                  Canada: Manitoba          25 411.02      3.42
# > 33                  Canada: Manitoba          50 469.83      3.06
# > 34                  Canada: Manitoba          75 529.98      2.98
# > 35                  Canada: Manitoba          90 581.85      4.25
# > 36              Canada: Saskatchewan          10 358.10      4.84
# > 37              Canada: Saskatchewan          25 406.97      3.89
# > 38              Canada: Saskatchewan          50 465.73      3.03
# > 39              Canada: Saskatchewan          75 526.57      4.35
# > 40              Canada: Saskatchewan          90 580.64      5.22
# > 41                   Canada: Alberta          10 376.10      6.47
# > 42                   Canada: Alberta          25 431.57      6.91
# > 43                   Canada: Alberta          50 501.91      6.59
# > 44                   Canada: Alberta          75 571.00      7.42
# > 45                   Canada: Alberta          90 632.71      9.51
# > 46          Canada: British Columbia          10 376.88      6.47
# > 47          Canada: British Columbia          25 430.64      5.46
# > 48          Canada: British Columbia          50 495.06      5.54
# > 49          Canada: British Columbia          75 560.20      5.03
# > 50          Canada: British Columbia          90 616.55      5.21

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="SCHLTYPE", data=temp_data)
# >                                      Estimate Std. Error t value
# > (Intercept)                            543.95      12.25   44.40
# > SCHLTYPEPrivate Government-dependent     4.88      15.10    0.32
# > SCHLTYPEPublic                         -50.89      12.25   -4.15
# > R-squared                                0.02       0.01    3.44

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="REGION", data=temp_data)
# >                                    Estimate Std. Error t value
# > (Intercept)                          458.54       5.54   82.83
# > REGIONCanada: Prince Edward Island    19.17       9.12    2.10
# > REGIONCanada: Nova Scotia             11.78       6.59    1.79
# > REGIONCanada: New Brunswick            9.14       6.18    1.48
# > REGIONCanada: Quebec                  55.09       6.88    8.00
# > REGIONCanada: Ontario                 36.69       6.36    5.77
# > REGIONCanada: Manitoba                11.93       5.53    2.16
# > REGIONCanada: Saskatchewan             9.12       6.34    1.44
# > REGIONCanada: Alberta                 45.00       7.51    6.00
# > REGIONCanada: British Columbia        37.77       7.13    5.29
# > R-squared                              0.02       0.00    5.61

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x=stuq_schq_dvs, data=temp_data)
# >                                      Estimate Std. Error t value
# > (Intercept)                            504.35      13.44   37.51
# > SCHLTYPEPrivate Government-dependent     4.25      14.76    0.29
# > SCHLTYPEPublic                         -45.81      12.51   -3.66
# > REGIONCanada: Prince Edward Island      18.00       9.03    1.99
# > REGIONCanada: Nova Scotia               11.78       6.59    1.79
# > REGIONCanada: New Brunswick              9.14       6.18    1.48
# > REGIONCanada: Quebec                    45.37       7.04    6.44
# > REGIONCanada: Ontario                   35.19       6.35    5.54
# > REGIONCanada: Manitoba                   8.02       5.56    1.44
# > REGIONCanada: Saskatchewan               8.38       6.33    1.33
# > REGIONCanada: Alberta                   44.62       7.37    6.06
# > REGIONCanada: British Columbia          33.41       7.02    4.76
# > R-squared                                0.04       0.01    5.39

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="SCHLTYPE", by = "REGION", data=temp_data)
# Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]) : 
#   contrasts can be applied only to factors with 2 or more levels

pisa.reg.pv(pvlabel=paste0("PV",1:10,"MATH"), x="REGION", by = "SCHLTYPE", data=temp_data)
# > $`Private independent`
# >                                Estimate Std. Error t value
# > (Intercept)                      535.97      11.92   44.98
# > REGIONCanada: Quebec              20.76      18.17    1.14
# > REGIONCanada: Ontario            -23.74      21.72   -1.09
# > REGIONCanada: Manitoba           -23.50      13.07   -1.80
# > REGIONCanada: Saskatchewan       -63.88      18.63   -3.43
# > REGIONCanada: Alberta             91.57      21.31    4.30
# > REGIONCanada: British Columbia    15.10      30.73    0.49
# > R-squared                          0.08       0.06    1.33

# > $`Private Government-dependent`
# >                                Estimate Std. Error t value
# > (Intercept)                      560.49      11.59   48.34
# > REGIONCanada: Manitoba           -61.21      14.04   -4.36
# > REGIONCanada: Saskatchewan       -92.05      48.82   -1.89
# > REGIONCanada: British Columbia   -33.00      14.20   -2.32
# > R-squared                          0.06       0.03    1.73

# > $Public
# >                                    Estimate Std. Error t value
# > (Intercept)                          458.54       5.54   82.83
# > REGIONCanada: Prince Edward Island    17.64       9.11    1.94
# > REGIONCanada: Nova Scotia             11.78       6.59    1.79
# > REGIONCanada: New Brunswick            9.14       6.18    1.48
# > REGIONCanada: Quebec                  43.63       7.33    5.95
# > REGIONCanada: Ontario                 36.11       6.27    5.76
# > REGIONCanada: Manitoba                 8.79       5.50    1.60
# > REGIONCanada: Saskatchewan             9.09       6.30    1.44
# > REGIONCanada: Alberta                 43.94       7.27    6.04
# > REGIONCanada: British Columbia        33.53       7.12    4.71
# > R-squared                              0.01       0.00    5.20

