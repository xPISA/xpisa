# ---- III. Predictive Modelling: Version 3.4 ----
#
# Nested Cross-Validation (glmnet):
#   inner  - cv.glmnet with MANUAL K-folds (grouped = TRUE, keep = TRUE), select lambda.min AND lambda.1se across α-grid
#   outer  - glmnet exact refit on OUTER-TRAIN at the inner winner’s (alpha, lambda.*); evaluate on OUTER-HOLDOUT
#
# This script runs both rules side-by-side and reports per-fold and aggregated metrics
# for:   Rule ∈ { "lambda.min", "lambda.1se" }.
#
# Remark: glmnet cannot handle NAs?

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
library(doParallel)
# library(Matrix)     # (optional) if use sparse model matrices
options(contrasts = c("contr.treatment", "contr.poly"))

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("tidyverse","glmnet","Matrix","haven","broom","tictoc","caret", "doParallel"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         tidyverse            glmnet            Matrix             haven             broom            tictoc            caret          doParallel  
# > "tidyverse 2.0.0"   "glmnet 4.1.10"    "Matrix 1.7.3"     "haven 2.5.4"     "broom 1.0.8"    "tictoc 1.2.1"     "caret 7.0.1" "doParallel 1.0.17" 

# Load data
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE)
dim(pisa_2022_canada_merged)   # 23073 x 1699

# Load metadata + missing summary: full variables
metadata_missing_student <- read_csv("data/pisa2022/metadata_missing_student.csv", show_col_types = FALSE)
metadata_missing_school <- read_csv("data/pisa2022/metadata_missing_school.csv", show_col_types = FALSE)

# Load metadata + missing summary: student/school questionnaire derived variables
stuq_dvs_metadata_missing_student <- readr::read_csv("data/pisa2022/stuq_dvs_metadata_missing_student.csv", show_col_types = FALSE)
schq_dvs_metadata_missing_school <- read_csv("data/pisa2022/schq_dvs_metadata_missing_school.csv",  show_col_types = FALSE)

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
  "MATHMOT","MATHEASE","MATHPREF",
  ### Out-of-school experiences (Module 10)
  "EXERPRAC","STUDYHMW","WORKPAY","WORKHOME",
  ## Derived variables based on IRT scaling
  ### Economic, social and cultural status (Module 2)
  "HOMEPOS","ICTRES",
  ### Educational pathways and post-secondary aspirations (Module 3)
  "INFOSEEK",
  ### School culture and climate (Module 6)
  "BULLIED","FEELSAFE","BELONG",
  ### Subject-specific beliefs, attitudes, feelings, and behaviours (Module 7)
  "GROSAGR","ANXMAT","MATHEFF","MATHEF21","MATHPERS","FAMCON",
  ### General social and emotional characteristics (Module 8)
  "ASSERAGR","COOPAGR","CURIOAGR","EMOCOAGR","EMPATAGR","PERSEVAGR","STRESAGR",
  ### Exposure to mathematics content (Module 15)
  "EXPOFA","EXPO21ST",
  ### Mathematics teacher behaviour (Module 16)
  "COGACRCO","COGACMCO","DISCLIM",
  ### Parental/guardian involvement and support (Module 19)
  "FAMSUP",
  ### Creative thinking (Module 20)
  "CREATFAM","CREATSCH","CREATEFF","CREATOP","IMAGINE","OPENART","CREATAS","CREATOOS",
  ### Global crises (Module 21)
  "FAMSUPSL","FEELLAH","PROBSELF","SDLEFF","SCHSUST","LEARRES",
  ## Complex composite index
  "ESCS",
  # --- School Questionnaire Derived Variables ---
  "MACTIV","ABGMATH","MTTRAIN","CREENVSC","OPENCUL","DIGPREP"
)                                                                  # numeric predictors
stopifnot(!anyDuplicated(voi_num))  # Fail if any duplicates 

voi_cat <- c("REGION", "ST004D01T", "IMMIG", "LANGN", "SCHLTYPE")  # categorical predictors
stopifnot(!anyDuplicated(voi_cat))

voi_all <- c(voi_num, voi_cat)  
stopifnot(!anyDuplicated(voi_all))# all predictors

length(voi_num);length(voi_cat);length(voi_all)
# > [1] 53
# > [1] 5
# > [1] 58

# Weights
rep_wts  <- paste0("W_FSTURWT", 1:G)    # W_FSTURWT1 to W_FSTURWT80
final_wt <- "W_FSTUWT"                  # Final student weight

length(rep_wts); length(final_wt)
# > [1] 80
# > [1] 1

### ---- Prepare modeling data ----
temp_data <- pisa_2022_canada_merged %>%
  select(CNTSCHID, CNTSTUID,                            # IDs
         all_of(final_wt), all_of(rep_wts),             # Weights
         all_of(pvmaths), all_of(voi_all)) %>%          # PVs + predictors (num + cat)
  mutate(
    LANGN = na_if(LANGN, 999),                          # code 999 ->NA
    across(all_of(voi_cat),
           ~ if (inherits(.x, "haven_labelled"))
             haven::as_factor(.x, levels = "labels")
           else as.factor(.x))
  ) %>%
  filter(IMMIG != "No Response") %>%                    # drop "No Response"
  filter(if_all(all_of(voi_all), ~ !is.na(.))) %>%      # listwise deletion
  droplevels()                                          # drop levels not present  

dim(temp_data)  
# > [1] 6755  151

# Quick invariants for weights & PVs after filtering
stopifnot(all(!is.na(temp_data[[final_wt]])))
stopifnot(all(colSums(is.na(temp_data[, pvmaths, drop = FALSE])) == 0))

### ---- Sanity check (post-prep) ----

## Levels of voi_cat
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

## Missingness
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
sapply(pisa_2022_canada_merged[, voi_cat, drop=FALSE], \(x) sum(is.na(x)))                                
# > REGION ST004D01T     IMMIG     LANGN  SCHLTYPE 
# >      0        42      2610         0         0 
#                                     ⚠️
stopifnot(all(colSums(is.na(temp_data[, voi_all])) == 0))     # no NA in predictors used
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

## Data type/class 
sapply(temp_data[, voi_num, drop=FALSE], is.numeric)          # numeric predictors are truly numeric in temp_data
# > MATHMOT  MATHEASE  MATHPREF  EXERPRAC  STUDYHMW   WORKPAY  WORKHOME   HOMEPOS    ICTRES  INFOSEEK   BULLIED  FEELSAFE    BELONG   GROSAGR    ANXMAT   MATHEFF  MATHEF21  MATHPERS 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 
# >  FAMCON  ASSERAGR   COOPAGR  CURIOAGR  EMOCOAGR  EMPATAGR PERSEVAGR  STRESAGR    EXPOFA  EXPO21ST  COGACRCO  COGACMCO   DISCLIM    FAMSUP  CREATFAM  CREATSCH  CREATEFF   CREATOP 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 
# > IMAGINE   OPENART   CREATAS  CREATOOS  FAMSUPSL   FEELLAH  PROBSELF    SDLEFF   SCHSUST   LEARRES      ESCS    MACTIV   ABGMATH   MTTRAIN  CREENVSC   OPENCUL   DIGPREP 
# >    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE 
sapply(temp_data[, voi_cat, drop=FALSE], is.factor)           # categorical predictors are truly factors in temp_data
# > REGION ST004D01T     IMMIG     LANGN  SCHLTYPE 
# >   TRUE      TRUE      TRUE      TRUE      TRUE

## Check dimension
dim(temp_data); dim(pisa_2022_canada_merged)  
# > [1] 6755  151
# > [1] 23073  1278

## MATHMOT 
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
# > 6499  256    0
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

## MATHEASE 
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
# > 5828  927    0 
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
# > 1    0 No perception of mathematics as easier than other subjects 5828
# > 2    1    Perception of mathematics as easier than other subjects  927
# > 3   NA                                                       <NA>    0

## MATHPREF 
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
# > 16310  2595  4168     0 
table(temp_data$MATHPREF, useNA = "always") 
# >    0    1 <NA> 
# > 5750 1005    0 
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
# > 1    0 No preference for mathematics over other subjects 5750
# > 2    1    Preference for mathematics over other subjects 1005
# > 3   NA                                              <NA>    0

## EXERPRAC 
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
table(temp_data$EXERPRAC, useNA = "always") 
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 1276  330  772  690  749  774  571  251  394  119  829    0 
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
# > 1     0                             No exercise or sports 1276
# > 2     1           1 time of exercising or sports per week  330
# > 3     2          2 times of exercising or sports per week  772
# > 4     3          3 times of exercising or sports per week  690
# > 5     4          4 times of exercising or sports per week  749
# > 6     5          5 times of exercising or sports per week  774
# > 7     6          6 times of exercising or sports per week  571
# > 8     7          7 times of exercising or sports per week  251
# > 9     8          8 times of exercising or sports per week  394
# > 10    9          9 times of exercising or sports per week  119
# > 11   10 10 or more times of exercising or sports per week  829
# > 12   NA                                              <NA>    0

## STUDYHMW 
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
table(temp_data$STUDYHMW, useNA = "always") 
# >   0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 700  418  836  787  919  886  722  335  391  135  626    0
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
# > 1     0                        No studying 700
# > 2     1        1 time of studying per week 418
# > 3     2       2 times of studying per week 836
# > 4     3       3 times of studying per week 787
# > 5     4       4 times of studying per week 919
# > 6     5       5 times of studying per week 886
# > 7     6       6 times of studying per week 722
# > 8     7       7 times of studying per week 335
# > 9     8       8 times of studying per week 391
# > 10    9       9 times of studying per week 135
# > 11   10 10 or more times of study per week 626
# > 12   NA                               <NA>   0

## WORKPAY 
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
table(temp_data$WORKPAY, useNA = "always") 
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 3914  390  647  471  429  228  247   67  159   32  171    0
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
# > 1     0                              No work for pay 3914
# > 2     1           1 time of working for pay per week  390
# > 3     2          2 times of working for pay per week  647
# > 4     3          3 times of working for pay per week  471
# > 5     4          4 times of working for pay per week  429
# > 6     5          5 times of working for pay per week  228
# > 7     6          6 times of working for pay per week  247
# > 8     7          7 times of working for pay per week   67
# > 9     8          8 times of working for pay per week  159
# > 10    9          9 times of working for pay per week   32
# > 11   10 10 or more times of working for pay per week  171
# > 12   NA                                         <NA>    0

## WORKHOME 
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
table(temp_data$WORKHOME, useNA = "always") 
# >    0    1    2    3    4    5    6    7    8    9   10 <NA> 
# > 1054  495  752  650  662  780  506  290  374  188 1004    0 
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
# > 1     0                                 No work in household or care of family members 1054
# > 2     1           1 time of working in household or caring for family members per week  495
# > 3     2          2 times of working in household or caring for family members per week  752
# > 4     3          3 times of working in household or caring for family members per week  650
# > 5     4          4 times of working in household or caring for family members per week  662
# > 6     5          5 times of working in household or caring for family members per week  780
# > 7     6          6 times of working in household or caring for family members per week  506
# > 8     7          7 times of working in household or caring for family members per week  290
# > 9     8          8 times of working in household or caring for family members per week  374
# > 10    9          9 times of working in household or caring for family members per week  188
# > 11   10 10 or more times of working in household or caring for family members per week 1004
# > 12   NA                                                                           <NA>    0

## HOMEPOS 
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
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -6.5856 -0.0948  0.4465  0.4378  0.9494  4.3468 

## ICTRES 
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
# > -5.0283 -0.1954  0.4010  0.4193  0.9490  5.2480 

## INFOSEEK 
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
# > -2.42110 -0.50290  0.04260 -0.06512  0.45570  2.61410 

## BULLIED 
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
# > -1.2280 -1.2280 -0.3264 -0.2766  0.5504  4.6939

## FEELSAFE
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
# > -2.7886 -0.7560  0.4014  0.1074  1.1246  1.1246

## BELONG
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
# > -3.2583 -0.7334 -0.3261 -0.1571  0.2163  2.7562 

## GROSAGR
class(pisa_2022_canada_merged$GROSAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
class(temp_data$GROSAGR)
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double" 
sum(is.na(pisa_2022_canada_merged$GROSAGR))
# > [1] 4067
sum(is.na(temp_data$GROSAGR))
# > [1] 0
summary(haven::zap_missing(pisa_2022_canada_merged$GROSAGR), na.rm = TRUE)
# >     Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# > -3.6086 -0.3061 -0.0144  0.0214  0.6926  3.3724    4067 
#summary(pisa_2022_canada_merged$GROSAGR[!is.na(pisa_2022_canada_merged$GROSAGR)])
summary(temp_data$GROSAGR) 
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -3.59940 -0.30610 -0.01440  0.08146  0.77980  2.02940

## ANXMAT
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
# > -2.39450 -0.50800  0.10310  0.08486  0.77740  2.63500 

## MATHEFF
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
# > -3.49350 -0.68750 -0.16950 -0.05133  0.54605  2.35560 

## MATHEF21
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
# > -2.4607 -0.2601  0.3264  0.3074  0.7507  2.7911

## MATHPERS
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
# > -3.0955 -0.3004  0.1861  0.2821  0.7299  2.8342 

## FAMCON
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
# > -3.9827  0.0577  0.7463  0.9682  1.5729  4.8382

## ASSERAGR
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
# > -8.22560 -0.36050  0.04140  0.09489  0.50370  7.25770 

## COOPAGR
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
# > -6.45170 -0.66225 -0.15640 -0.04843  0.36215  6.12650 

## CURIOAGR
class(pisa_2022_canada_merged$CURIOAGR)
class(temp_data$CURIOAGR)
sum(is.na(pisa_2022_canada_merged$CURIOAGR))
sum(is.na(temp_data$CURIOAGR))
summary(haven::zap_missing(pisa_2022_canada_merged$CURIOAGR), na.rm = TRUE)
#summary(pisa_2022_canada_merged$CURIOAGR[!is.na(pisa_2022_canada_merged$CURIOAGR)])
summary(temp_data$CURIOAGR) 

## EMOCOAGR
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
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -5.18920 -0.42020  0.04950  0.07677  0.59365  5.58720 

## EMPATAGR
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
# > -6.07510 -0.60280 -0.09700  0.04129  0.45385  4.68970 

## PERSEVAGR
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
# > -5.90880 -0.61520 -0.13540  0.03361  0.42885  4.77720 

## STRESAGR
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
# > -5.2609 -0.5948 -0.0213 -0.1160  0.4315  5.6189 
## EXPOFA
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
# >    Min. 1st Qu.  Median    Mean  3rd Qu. Max. 
# > -3.0867 -0.3470  0.0885  0.1096  0.4687  2.64

## EXPO21ST
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
# > -2.6364 -0.2433  0.2844  0.2754  0.7217  3.2705

## COGACRCO
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
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > -2.9785 -0.3199  0.0945  0.1629  0.5653  3.7198

## COGACMCO
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
# > -2.1598 -0.3375  0.2239  0.2418  0.7999  2.6158

## DISCLIM
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
# > -2.49310 -0.55020 -0.09030 -0.03403  0.47680  1.85140 

## FAMSUP
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
# > -2.975100 -0.613750 -0.122400  0.008908  0.492950  1.958300

## CREATFAM
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
# > -2.7890 -0.4924 -0.0374  0.1540  0.7099  2.2394

## CREATSCH
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
# > -2.6234 -0.3960  0.2139  0.2259  0.5352  2.8139 

## CREATEFF
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
# > -2.9210 -0.4709  0.0978  0.1216  0.5741  2.5641

## CREATOP
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
# > -2.9230 -0.5404  0.0235  0.1038  0.5381  2.9234 

## IMAGINE
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
# > -3.39820 -0.49970 -0.09280  0.08993  0.58595  2.51100 

## OPENART
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
# > -2.8185 -0.3483  0.1775  0.1698  0.6258  1.8261 

## CREATAS
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
# > -1.12050 -0.69910  0.01420 -0.02239  0.39500  4.33120 

## CREATOOS
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
# > -0.8205 -0.8105 -0.3947 -0.1173  0.3890  4.6466 

## FAMSUPSL
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
# > -2.38010 -0.71550 -0.02460 -0.09023  0.53640  2.43760

## FEELLAH
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
# > -2.6455000 -0.5889000 -0.0535000 -0.0008298  0.5970000  3.1517000 

## PROBSELF
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
# > -2.23060 -0.51680  0.17330  0.04083  0.64935  3.09760

## SDLEFF
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
# > -2.58520 -0.59310  0.11170  0.01592  0.40985  2.08170 

## SCHSUST
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
# >    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# > -2.7592 -0.4420  0.0466  0.1148  0.5866  2.7623

## LEARRES
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
# > -3.31330 -0.45350 -0.04230 -0.07991  0.37440  3.04650

## ESCS
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
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# > -5.50100 -0.09785  0.53800  0.42441  0.99305  2.74580 

## MACTIV
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
# > 776 1336 1671 1372  969  631    0 
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
# > 1    0 No mathematics-related extra-curricular activities offered  776
# > 2    1    1 mathematics-related extra-curricular activity offered 1336
# > 3    2  2 mathematics-related extra-curricular activities offered 1671
# > 4    3  3 mathematics-related extra-curricular activities offered 1372
# > 5    4  4 mathematics-related extra-curricular activities offered  969
# > 6    5  5 mathematics-related extra-curricular activities offered  631
# > 7   NA                                                       <NA>    0

## ABGMATH
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

## MTTRAIN
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
# > -1.6751 -0.0622  0.6955  0.3508  1.0982  1.0982

## CREENVSC
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
# >     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > -3.22370 -0.76410 -0.26860 -0.09237  0.41180  2.16310

## OPENCUL
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
# > -2.0055 -0.3274  0.4217  0.2895  0.5263  3.3059

## DIGPREP
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
# > -3.11940 -0.67490 -0.16110 -0.08873  0.31050  2.96100 

## REGION
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

## ST004D01T
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

## IMMIG
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

## LANGN
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

## SCHLTYPE
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

# Check summaries
summary(temp_data[[final_wt]])
summary(temp_data[, pvmaths])
sapply(temp_data[, pvmaths], sd)
summary(temp_data[, voi_num])
sapply(temp_data[, voi_num], sd)
summary(temp_data[, voi_cat])


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
n <- nrow(temp_data)   # 6755
indices <- sample(n)   # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.80 * n)         # 5404
n_valid <- floor(0.10 * n)         # 675
n_test  <- n - n_train - n_valid   # 676
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
# >                  train    valid     test
# > n              5404.00   675.00   676.00
# > w_mean           16.19    15.84    16.34
# > w_sum         87466.12 10694.53 11042.92
# > w_min             1.05     1.05     1.05
# > w_max           132.19    88.54    88.54
# > w_sd             13.84    13.83    13.73
# > n_schools       667.00   413.00   407.00
# > pv1math_mean    521.77   522.72   529.01
# > pv1math_sd       89.80    92.48    88.53
# > exerprac_mean     4.19     3.93     3.99
# > exerprac_sd       3.22     3.18     3.33

### ---- Build design matrices for (voi_num + voi_cat) with treatment dummies ----
# One model.matrix on the full filtered data to lock column set; drop intercept.
X_all <- model.matrix(
  as.formula(paste("~", paste(voi_all, collapse = " + "))),
  data = temp_data
)
dim(X_all)
# > [1] 6755   71
if ("(Intercept)" %in% colnames(X_all)) {
  X_all <- X_all[, colnames(X_all) != "(Intercept)", drop = FALSE]
}
dim(X_all)
# > [1] 6755   70

# Split-aligned matrices
X_train <- X_all[train_idx, , drop = FALSE]
X_valid <- X_all[valid_idx, , drop = FALSE]
X_test  <- X_all[test_idx,  , drop = FALSE]

class(X_train); class(X_valid); class(X_test)
# > [1] "matrix" "array" 
# > [1] "matrix" "array" 
# > [1] "matrix" "array" 
dim(X_train); dim(X_valid); dim(X_test)
# > [1] 5404   70
# > [1] 675  70
# > [1] 676  70

## ---- 2. PV1MATH only ----

# --- Remark ---
# 1) Repeat the same process for PV2MATH - PV10MATH.
# 2) Apply best results from PV1MATH to all plausible values in mathematics. 

## ---- Main model using final student weights (W_FSTUWT) ----

### ---- Nested Cross-Validation for PV1MATH (grouped=TRUE, keep=TRUE; lambda.min & lambda.1se) ----

##### ---- Prepare ----
# Target plausible value
pv1math <- pvmaths[1]

# Training/Validation/Test targets & weights (X_* already built above)
y_train <- train_data[[pv1math]]; w_train <- train_data[[final_wt]]
y_valid <- valid_data[[pv1math]]; w_valid <- valid_data[[final_wt]]
y_test  <- test_data[[pv1math]];  w_test  <- test_data[[final_wt]]

length(y_train); length(w_train)
# > [1] 5404
# > [1] 5404
length(y_valid); length(w_valid)
# > [1] 675
# > [1] 675
length(y_test);  length(w_test)
# > [1] 676
# > [1] 676

# --- Helpers: folds and diagnostics ---

# Build K manual folds (indices) with a fixed seed over N rows
make_folds <- function(n_cv, num_folds = 5L, seed = 123L) {
  set.seed(seed)
  cv_order <- sample.int(n_cv)
  bounds <- floor(seq(0, n_cv, length.out = num_folds + 1))
  stopifnot(all(diff(bounds) > 0))
  folds <- vector("list", num_folds)
  for (k in seq_len(num_folds)) folds[[k]] <- cv_order[(bounds[k] + 1):bounds[k + 1]]
  stopifnot(identical(sort(unlist(folds)), seq_len(n_cv)))
  folds
}

# Weight diagnostics per fold
print_weight_balance <- function(w, folds, title = "Weight balance") {
  s <- tibble::tibble(
    fold   = seq_along(folds),
    n      = vapply(folds, length, integer(1)),
    w_sum  = vapply(folds, function(idx) sum(w[idx]), numeric(1)),
    w_mean = vapply(folds, function(idx) mean(w[idx]), numeric(1)),
    w_med  = vapply(folds, function(idx) median(w[idx]), numeric(1)),
    w_effn = vapply(folds, function(idx) { wi <- w[idx]; (sum(wi)^2) / sum(wi^2) }, numeric(1))
  ) |>
    dplyr::mutate(w_share = w_sum / sum(w_sum))
  message(title)
  print(s, n = Inf, width = Inf)
  cat(sprintf("Weight share range: [%.3f, %.3f]\n", min(s$w_share), max(s$w_share)))
  cat(sprintf("Max/Min share ratio: %.3f\n", max(s$w_share) / min(s$w_share)))
  cat(sprintf("Coeff. of variation of shares: %.3f\n\n", stats::sd(s$w_share) / mean(s$w_share)))
}

# α-grid
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso

# --- Outer folds over TRAIN ---
num_folds_outer <- 5L
n_cv_outer <- nrow(X_train)
outer_folds <- make_folds(n_cv_outer, num_folds_outer, seed = 123L)

print_weight_balance(w_train, outer_folds, title = "Outer CV: weight balance over TRAIN")
# > Outer CV: weight balance over TRAIN
# > # A tibble: 5 × 7
# > fold     n  w_sum w_mean w_med w_effn w_share
# > <int> <int>  <dbl>  <dbl> <dbl>  <dbl>   <dbl>
# > 1     1  1080 17657.   16.3 10.2    628.   0.202
# > 2     2  1081 16982.   15.7  9.45   620.   0.194
# > 3     3  1081 17816.   16.5 10.5    621.   0.204
# > 4     4  1081 17197.   15.9  9.85   625.   0.197
# > 5     5  1081 17814.   16.5 10.5    629.   0.204
# > Weight share range: [0.194, 0.204]
# > Max/Min share ratio: 1.049
# > Coeff. of variation of shares: 0.022

# Storage (separate containers for each rule)
outer_oof_pred_min      <- rep(NA_real_, n_cv_outer)
outer_oof_pred_1se      <- rep(NA_real_, n_cv_outer)

outer_metrics_min       <- vector("list", num_folds_outer)
outer_metrics_1se       <- vector("list", num_folds_outer)

inner_winners_rows_min  <- vector("list", num_folds_outer)
inner_winners_rows_1se  <- vector("list", num_folds_outer)

##### ---- Tune: Nested Cross-Validation ----

# Parallel backend for cv.glmnet (inner loop)
doParallel::registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Nested Cross-Validation (glmnet; grouped=TRUE, keep=TRUE; lambda.min & lambda.1se)")
for (o in seq_len(num_folds_outer)) {
  message(sprintf("\n==== Outer fold %d/%d ====", o, num_folds_outer))
  outer_hold_idx  <- outer_folds[[o]]
  outer_train_idx <- setdiff(seq_len(n_cv_outer), outer_hold_idx)
  
  # OUTER-TRAIN / OUTER-HOLDOUT splits
  X_train_outer <- X_train[outer_train_idx, , drop = FALSE]
  y_train_outer <- y_train[outer_train_idx]
  w_train_outer <- w_train[outer_train_idx]
  
  X_hold_outer  <- X_train[outer_hold_idx, , drop = FALSE]
  y_hold_outer  <- y_train[outer_hold_idx]
  w_hold_outer  <- w_train[outer_hold_idx]
  
  # Inner folds (on OUTER-TRAIN)
  n_cv_inner      <- length(outer_train_idx)
  num_folds_inner <- 5L
  inner_folds     <- make_folds(n_cv_inner, num_folds_inner, seed = 123L)
  
  # foldid (1..K for each OUTER-TRAIN row)
  foldid_inner <- integer(n_cv_inner)
  for (k in seq_along(inner_folds)) foldid_inner[inner_folds[[k]]] <- k
  stopifnot(all(foldid_inner %in% seq_len(num_folds_inner)), length(foldid_inner) == n_cv_inner)
  
  print_weight_balance(w_train_outer, inner_folds,
                       title = sprintf("Inner CV (outer fold %d): weight balance over OUTER-TRAIN", o))
  
  # --- Inner tuner over α-grid using cv.glmnet (grouped=TRUE, keep=TRUE) ---
  per_alpha_list_min <- vector("list", length(alpha_grid))   # one row per α at lambda.min
  per_alpha_list_1se <- vector("list", length(alpha_grid))   # one row per α at lambda.1se
  
  tic(sprintf("Inner tuning (outer fold %d)", o))
  for (i in seq_along(alpha_grid)) {
    alpha <- alpha_grid[i]
    message(sprintf("  [Inner α %d/%d] alpha = %.1f  (manual 5-fold on OUTER-TRAIN)", i, length(alpha_grid), alpha))
    
    set.seed(123)
    cvmod <- cv.glmnet(
      x = X_train_outer,
      y = y_train_outer,
      weights = w_train_outer,
      type.measure = "mse",
      foldid = foldid_inner,      # manual inner folds
      grouped = TRUE,             # as requested
      keep = TRUE,                # keep OOF preds (cvmod$fit.preval)
      parallel = TRUE,
      trace.it = 0,
      alpha = alpha,
      family = "gaussian",
      standardize = TRUE,
      intercept = TRUE
    )
    
    # ---- lambda.min row (per α) ----
    idx_min     <- cvmod$index["min", 1]
    per_alpha_list_min[[i]] <- tibble::tibble(
      alpha       = alpha,
      alpha_idx   = i,
      s           = "lambda.min",
      lambda      = cvmod$lambda[idx_min],
      lambda_idx  = idx_min,
      nzero       = cvmod$nzero[idx_min],
      dev_ratio   = as.numeric(cvmod$glmnet.fit$dev.ratio[idx_min]),
      cvm         = as.numeric(cvmod$cvm[idx_min]),
      cvsd        = as.numeric(cvmod$cvsd[idx_min]),
      cvlo        = as.numeric(cvmod$cvlo[idx_min]),
      cvup        = as.numeric(cvmod$cvup[idx_min])
    )
    
    # ---- lambda.1se row (per α) ----
    idx_1se     <- cvmod$index["1se", 1]
    per_alpha_list_1se[[i]] <- tibble::tibble(
      alpha       = alpha,
      alpha_idx   = i,
      s           = "lambda.1se",
      lambda      = cvmod$lambda[idx_1se],
      lambda_idx  = idx_1se,
      nzero       = cvmod$nzero[idx_1se],
      dev_ratio   = as.numeric(cvmod$glmnet.fit$dev.ratio[idx_1se]),
      cvm         = as.numeric(cvmod$cvm[idx_1se]),
      cvsd        = as.numeric(cvmod$cvsd[idx_1se]),
      cvlo        = as.numeric(cvmod$cvlo[idx_1se]),
      cvup        = as.numeric(cvmod$cvup[idx_1se])
    )
  } # α loop
  toc()
  
  # Collate inner tuning results (one row per α per rule)
  tuning_results_min <- dplyr::bind_rows(per_alpha_list_min)
  tuning_results_1se <- dplyr::bind_rows(per_alpha_list_1se)
  
  # Winners across α (tie-breaks: fewer nzero → larger λ → smaller α)
  best_min <- tuning_results_min %>%
    dplyr::arrange(cvm, nzero, dplyr::desc(lambda), alpha) %>%
    dplyr::slice(1)
  
  best_1se <- tuning_results_1se %>%
    dplyr::arrange(cvm, nzero, dplyr::desc(lambda), alpha) %>%
    dplyr::slice(1)
  
  print(as.data.frame(best_min),  row.names = FALSE)
  print(as.data.frame(best_1se),  row.names = FALSE)
  message(sprintf("Inner winner @ lambda.min (outer %d): alpha=%.2f | lambda=%.6f | nzero=%d | CVM=%.5f (± %.5f)",
                  o, best_min$alpha, best_min$lambda, best_min$nzero, best_min$cvm, best_min$cvsd))
  message(sprintf("Inner winner @ lambda.1se (outer %d): alpha=%.2f | lambda=%.6f | nzero=%d | CVM=%.5f (± %.5f)",
                  o, best_1se$alpha, best_1se$lambda, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
  
  # Record winners (for audit)
  inner_winners_rows_min[[o]] <- tibble::tibble(
    outer_fold   = o,
    alpha        = best_min$alpha,
    alpha_idx    = best_min$alpha_idx,
    s            = best_min$s,
    lambda       = best_min$lambda,
    lambda_idx   = best_min$lambda_idx,
    nzero        = best_min$nzero,
    dev_ratio    = best_min$dev_ratio,
    cvm          = best_min$cvm,
    cvsd         = best_min$cvsd
  )
  inner_winners_rows_1se[[o]] <- tibble::tibble(
    outer_fold   = o,
    alpha        = best_1se$alpha,
    alpha_idx    = best_1se$alpha_idx,
    s            = best_1se$s,
    lambda       = best_1se$lambda,
    lambda_idx   = best_1se$lambda_idx,
    nzero        = best_1se$nzero,
    dev_ratio    = best_1se$dev_ratio,
    cvm          = best_1se$cvm,
    cvsd         = best_1se$cvsd
  )
  
  # --- Outer refit at the inner winners; evaluate on OUTER‑HOLDOUT ---
  
  # lambda.min
  fit_outer_min <- glmnet(
    x = X_train_outer, y = y_train_outer, weights = w_train_outer,
    family = "gaussian",
    alpha = best_min$alpha,
    lambda = best_min$lambda,
    standardize = TRUE, intercept = TRUE
  )
  pred_outer_hold_min <- as.numeric(predict(fit_outer_min, newx = X_hold_outer))
  fold_metrics_min <- compute_metrics(y_true = y_hold_outer, y_pred = pred_outer_hold_min, w = w_hold_outer)
  outer_metrics_min[[o]] <- fold_metrics_min
  outer_oof_pred_min[outer_hold_idx] <- pred_outer_hold_min
  message(sprintf("Outer %d @ lambda.min | RMSE=%.5f | MAE=%.5f | Bias=%.5f | Bias%%=%.3f | R2=%.5f",
                  o, fold_metrics_min["rmse"], fold_metrics_min["mae"], fold_metrics_min["bias"],
                  fold_metrics_min["bias_pct"], fold_metrics_min["r2"]))
  
  # lambda.1se
  fit_outer_1se <- glmnet(
    x = X_train_outer, y = y_train_outer, weights = w_train_outer,
    family = "gaussian",
    alpha = best_1se$alpha,
    lambda = best_1se$lambda,
    standardize = TRUE, intercept = TRUE
  )
  pred_outer_hold_1se <- as.numeric(predict(fit_outer_1se, newx = X_hold_outer))
  fold_metrics_1se <- compute_metrics(y_true = y_hold_outer, y_pred = pred_outer_hold_1se, w = w_hold_outer)
  outer_metrics_1se[[o]] <- fold_metrics_1se
  outer_oof_pred_1se[outer_hold_idx] <- pred_outer_hold_1se
  message(sprintf("Outer %d @ lambda.1se | RMSE=%.5f | MAE=%.5f | Bias=%.5f | Bias%%=%.3f | R2=%.5f",
                  o, fold_metrics_1se["rmse"], fold_metrics_1se["mae"], fold_metrics_1se["bias"],
                  fold_metrics_1se["bias_pct"], fold_metrics_1se["r2"]))
}
toc()
# > Nested Cross-Validation (glmnet; grouped=TRUE, keep=TRUE; lambda.min & lambda.1se): 7.325 sec elapsed

# Stop the parallel backend:
# doParallel::stopImplicitCluster()

##### ---- Aggregations over outer holds (both rules) ----

# Helpers to aggregate & print for a given rule
aggregate_rule <- function(outer_metrics_list, outer_oof_pred_vec, rule_label) {
  outer_metrics_matrix <- do.call(rbind, outer_metrics_list)
  metrics_outer_mean <- tibble::tibble(
    Rule = rule_label,
    Aggregation = "Simple-mean (outer folds)",
    RMSE = mean(outer_metrics_matrix[, "rmse"]),
    MAE  = mean(outer_metrics_matrix[, "mae"]),
    Bias = mean(outer_metrics_matrix[, "bias"]),
    `Bias%` = mean(outer_metrics_matrix[, "bias_pct"]),
    R2   = mean(outer_metrics_matrix[, "r2"])
  )
  stopifnot(all(!is.na(outer_oof_pred_vec)))
  pooled_vec <- compute_metrics(y_true = y_train, y_pred = outer_oof_pred_vec, w = w_train)
  metrics_outer_pooled <- tibble::tibble(
    Rule = rule_label,
    Aggregation = "Pooled-weighted OOF",
    RMSE = pooled_vec["rmse"],
    MAE  = pooled_vec["mae"],
    Bias = pooled_vec["bias"],
    `Bias%` = pooled_vec["bias_pct"],
    R2   = pooled_vec["r2"]
  )
  list(mean = metrics_outer_mean, pooled = metrics_outer_pooled,
       fold_matrix = outer_metrics_matrix)
}

agg_min  <- aggregate_rule(outer_metrics_min, outer_oof_pred_min, "lambda.min")
agg_1se  <- aggregate_rule(outer_metrics_1se, outer_oof_pred_1se, "lambda.1se")

# Report both rules side-by-side
message("\n==== Nested-CV aggregated performance (both rules) ====")
print(as.data.frame(dplyr::bind_rows(agg_min$mean, agg_min$pooled,
                                     agg_1se$mean, agg_1se$pooled)),
      row.names = FALSE)
# >       Rule               Aggregation     RMSE      MAE      Bias    Bias%        R2
# > lambda.min Simple-mean (outer folds) 67.59788 53.36624 0.2494544 1.906435 0.4307354
# > lambda.min       Pooled-weighted OOF 67.60672 53.36879 0.2261304 1.902468 0.4331984
# > lambda.1se Simple-mean (outer folds) 68.66041 54.27123 0.2148399 2.076718 0.4127045
# > lambda.1se       Pooled-weighted OOF 68.66404 54.27271 0.1846418 2.070881 0.4153309

# Inner winners tables (for audit)
inner_winners_table_min <- dplyr::bind_rows(inner_winners_rows_min)
inner_winners_table_1se <- dplyr::bind_rows(inner_winners_rows_1se)

message("\n==== Inner winners by outer fold — lambda.min ====")
print(as.data.frame(inner_winners_table_min), row.names = FALSE)
# > outer_fold alpha alpha_idx          s   lambda lambda_idx nzero dev_ratio      cvm      cvsd
# >          1   0.4         5 lambda.min 1.5531558         48    51 0.4644311 4560.886 136.5595
# >          2   0.3         4 lambda.min 2.0210007         48    56 0.4520706 4628.503 210.8245
# >          3   1.0        11 lambda.min 0.8095524         45    47 0.4504401 4628.023  95.3939
# >          4   0.3         4 lambda.min 1.9727367         48    59 0.4529092 4575.846 121.1258
# >          5   0.4         5 lambda.min 1.5062609         48    52 0.4538654 4565.869 148.6076
message("\n==== Inner winners by outer fold — lambda.1se ====")
print(as.data.frame(inner_winners_table_1se), row.names = FALSE)
# > outer_fold alpha alpha_idx          s    lambda lambda_idx nzero  dev_ratio      cvm     cvsd
# >          1   0.1         2 lambda.1se 17.286993         37    40 0.4405396 4679.416 126.6796
# >          2   0.3         4 lambda.1se  9.827321         31    30 0.4178003 4815.625 192.2251
# >          3   0.8         9 lambda.1se  2.565638         35    36 0.4326741 4703.399 105.6432
# >          4   0.1         2 lambda.1se 16.467771         37    45 0.4298981 4678.444 138.9949
# >          5   0.1         2 lambda.1se 16.765041         37    42 0.4305848 4689.556 150.5067

# Per-fold outer holdout metrics tables
outer_folds_table_min <- tibble::tibble(
  outer_fold = seq_len(num_folds_outer),
  RMSE   = agg_min$fold_matrix[, "rmse"],
  MAE    = agg_min$fold_matrix[, "mae"],
  Bias   = agg_min$fold_matrix[, "bias"],
  `Bias%`= agg_min$fold_matrix[, "bias_pct"],
  R2     = agg_min$fold_matrix[, "r2"]
)
outer_folds_table_1se <- tibble::tibble(
  outer_fold = seq_len(num_folds_outer),
  RMSE   = agg_1se$fold_matrix[, "rmse"],
  MAE    = agg_1se$fold_matrix[, "mae"],
  Bias   = agg_1se$fold_matrix[, "bias"],
  `Bias%`= agg_1se$fold_matrix[, "bias_pct"],
  R2     = agg_1se$fold_matrix[, "r2"]
)

message("\n==== Per-fold outer holdout metrics — lambda.min ====")
print(as.data.frame(outer_folds_table_min), row.names = FALSE)
# > outer_fold     RMSE      MAE       Bias    Bias%         R2
# >          1 68.04441 52.65585 -2.2608815 1.3283305 0.3968444
# >          2 66.33173 52.93543  6.6767340 3.1312922 0.4337122
# >          3 67.49419 53.72033 -0.4442964 1.8621643 0.4375659
# >          4 68.46345 53.76726 -4.4676535 0.9654724 0.4446459
# >          5 67.65562 53.75232  1.7433693 2.2449171 0.4409085
message("\n==== Per-fold outer holdout metrics — lambda.1se ====")
print(as.data.frame(outer_folds_table_1se), row.names = FALSE)
# > outer_fold     RMSE      MAE       Bias    Bias%         R2
# >          1 68.40624 53.52523 -2.75788571 1.411618 0.3904127
# >          2 68.13219 54.39845  7.53470535 3.525404 0.4025532
# >          3 69.03867 54.70103 -0.02954573 2.061513 0.4115309
# >          4 69.05386 54.08090 -4.61211931 1.121337 0.4350262
# >          5 68.67109 54.65053  0.93904494 2.263720 0.4239993

# Consolidated: inner winners + outer holdout performance — both rules
message("\n==== Consolidated (winners + outer performance) — both rules ====")
consolidated_min <- inner_winners_table_min %>%
  dplyr::left_join(outer_folds_table_min, by = "outer_fold") %>%
  dplyr::mutate(Rule = "lambda.min")

consolidated_1se <- inner_winners_table_1se %>%
  dplyr::left_join(outer_folds_table_1se, by = "outer_fold") %>%
  dplyr::mutate(Rule = "lambda.1se")

consolidated_both <- dplyr::bind_rows(consolidated_min, consolidated_1se) %>%
  dplyr::relocate(Rule, outer_fold)

print(as.data.frame(consolidated_both), row.names = FALSE)
# >       Rule outer_fold alpha alpha_idx          s    lambda lambda_idx nzero dev_ratio      cvm      cvsd     RMSE      MAE      Bias     Bias%        R2
# > lambda.min          1   0.4         5 lambda.min  1.5531558         48    51 0.4644311 4560.886 136.5595 68.04441 52.65585 -2.26088150 1.3283305 0.3968444
# > lambda.min          2   0.3         4 lambda.min  2.0210007         48    56 0.4520706 4628.503 210.8245 66.33173 52.93543  6.67673396 3.1312922 0.4337122
# > lambda.min          3   1.0        11 lambda.min  0.8095524         45    47 0.4504401 4628.023  95.3939 67.49419 53.72033 -0.44429636 1.8621643 0.4375659
# > lambda.min          4   0.3         4 lambda.min  1.9727367         48    59 0.4529092 4575.846 121.1258 68.46345 53.76726 -4.46765352 0.9654724 0.4446459
# > lambda.min          5   0.4         5 lambda.min  1.5062609         48    52 0.4538654 4565.869 148.6076 67.65562 53.75232  1.74336926 2.2449171 0.4409085
# > lambda.1se          1   0.1         2 lambda.1se 17.2869928         37    40 0.4405396 4679.416 126.6796 68.40624 53.52523 -2.75788571 1.4116177 0.3904127
# > lambda.1se          2   0.3         4 lambda.1se  9.8273213         31    30 0.4178003 4815.625 192.2251 68.13219 54.39845  7.53470535 3.5254041 0.4025532
# > lambda.1se          3   0.8         9 lambda.1se  2.5656379         35    36 0.4326741 4703.399 105.6432 69.03867 54.70103 -0.02954573 2.0615133 0.4115309
# > lambda.1se          4   0.1         2 lambda.1se 16.4677709         37    45 0.4298981 4678.444 138.9949 69.05386 54.08090 -4.61211931 1.1213371 0.4350262
# > lambda.1se          5   0.1         2 lambda.1se 16.7650412         37    42 0.4305848 4689.556 150.5067 68.67109 54.65053  0.93904494 2.2637198 0.4239993

## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# > Majority Rule
# > Repeat Part 2 and 3 in elasticNet_pvmath_vs_voi_v2_4.r
# -> Apply best results from PV1MATH (best_alpha_min, best_lambda_min) to all plausible values in mathematics.