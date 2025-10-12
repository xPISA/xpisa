# ---- II. Predictive Modelling: Version 2.4----

# Tune hyperparameters (alpha α and lambda λ) using cv.glmnet, with manual K-folds + tracks out-of-fold (OOF) predictions/metrics.

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

### ---- Main model using final student weights (W_FSTUWT) ---- 

#### ---- Tuning Elastic Net for PV1MATH only: cv.glmnet ----

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
# > 17657.22 16982.38 17815.81 17196.57 17814.13
table(foldid)
# > foldid
# >    1    2    3    4    5 
# > 1080 1081 1081 1081 1081

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
# > # A tibble: 5 × 7
# >    fold     n  w_sum w_mean w_med w_effn w_share
# >   <int> <int>  <dbl>  <dbl> <dbl>  <dbl>   <dbl>
# > 1     1  1080 17657.   16.3 10.2    628.   0.202
# > 2     2  1081 16982.   15.7  9.45   620.   0.194
# > 3     3  1081 17816.   16.5 10.5    621.   0.204
# > 4     4  1081 17197.   15.9  9.85   625.   0.197
# > 5     5  1081 17814.   16.5 10.5    629.   0.204
# > Weight share range: [0.194, 0.204]
# > Max/Min share ratio: 1.049
# > Coeff. of variation of shares: 0.022

# Outcome distribution by fold
print(do.call(rbind, tapply(y_train, foldid, function(v) c(mean = mean(v), sd = sd(v)))))
# >       mean       sd
# > 1 512.6369 88.73558
# > 2 504.7149 88.03631
# > 3 510.2549 88.14554
# > 4 511.9522 90.84217
# > 5 510.3559 90.78014

# Grid over elastic‑net mixing parameter
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso
# options: alpha_grid <- seq(0, 1, by = 0.05), alpha_grid <- sort(unique(c(seq(0, 1, by = 0.1), 0.001, 0.005, 0.01, 0.05))), fine tuning etc. 

# Storage
cv_list        <- vector("list", length(alpha_grid))  # cv.glmnet fits per alpha
per_alpha_list <- vector("list", length(alpha_grid))  # two rows per alpha: lambda.min & lambda.1se

##### ---- Tune: grouped = TRUE (Default), keep = TRUE ----
#
# Remark: change grouped = FALSE to compare performance
#

# Parallel backend (for cv.glmnet parallelization)
registerDoParallel(cores = max(1L, parallel::detectCores() - 1L))

tic("Grid over alpha (cv.glmnet, manual 5-fold)")
for (i in seq_along(alpha_grid)) {
  alpha <- alpha_grid[i]
  message(sprintf("Fitting cv.glmnet for alpha = %.1f (manual 5-fold CV on TRAIN)", alpha))
  
  # Fix fold randomness 
  set.seed(123)
  
  cvmod <- cv.glmnet(
    x = X_train,
    y = y_train,
    weights = w_train,
    type.measure = "mse", 
    foldid = foldid,           # <-
    grouped = TRUE,            # <-                              
    keep = TRUE,               # <-                               
    parallel = TRUE,          
    trace.it = 0,
    alpha = alpha,
    family = "gaussian",
    standardize = TRUE,
    intercept = TRUE
  )
  cv_list[[i]] <- cvmod
  
  # Indices
  idx_min <- cvmod$index["min", 1]
  idx_1se <- cvmod$index["1se", 1]
  
  # Lambdas and path metadata at those indices
  lambda_min <- cvmod$lambda[idx_min]      # == cvmod$lambda.min
  lambda_1se <- cvmod$lambda[idx_1se]      # == cvmod$lambda.1se
  
  nzero_min  <- cvmod$nzero[idx_min]
  nzero_1se  <- cvmod$nzero[idx_1se]
  
  dev_min    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_min])
  dev_1se    <- as.numeric(cvmod$glmnet.fit$dev.ratio[idx_1se])
  
  # CV summaries (in-built)
  cvm_min    <- as.numeric(cvmod$cvm[idx_min])
  cvsd_min   <- as.numeric(cvmod$cvsd[idx_min])
  cvup_min   <- as.numeric(cvmod$cvup[idx_min])
  cvlo_min   <- as.numeric(cvmod$cvlo[idx_min])
  
  cvm_1se    <- as.numeric(cvmod$cvm[idx_1se])
  cvsd_1se   <- as.numeric(cvmod$cvsd[idx_1se])
  cvup_1se   <- as.numeric(cvmod$cvup[idx_1se])
  cvlo_1se   <- as.numeric(cvmod$cvlo[idx_1se])
  
  # Two rows per α: one for lambda.min, one for lambda.1se
  per_alpha_list[[i]] <- tibble::tibble(
    alpha       = alpha,
    alpha_idx   = i,
    s           = c("lambda.min", "lambda.1se"),
    lambda      = c(lambda_min,   lambda_1se),
    lambda_idx  = c(idx_min,      idx_1se),
    nzero       = c(nzero_min,    nzero_1se),
    dev_ratio   = c(dev_min,      dev_1se),
    cvm         = c(cvm_min,      cvm_1se),
    cvsd        = c(cvsd_min,     cvsd_1se),
    cvlo        = c(cvlo_min,     cvlo_1se),
    cvup        = c(cvup_min,     cvup_1se)
  )
}
toc()
# > Grid over alpha (cv.glmnet, manual 5-fold): 2.001 sec elapsed

##### ---- Explore tuning results ----
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by CV MSE (lower is better) irrespective of rule
tuning_results %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  #head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
# > alpha alpha_idx          s     lambda lambda_idx nzero dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.4         5 lambda.min  1.1479061         51    56 0.4541241 4565.030 51.89823 4513.132 4616.928
# >   0.3         4 lambda.min  1.5305415         51    56 0.4540050 4565.058 51.72718 4513.331 4616.785
# >   0.5         6 lambda.min  0.9183249         51    56 0.4541911 4565.104 52.00973 4513.094 4617.113
# >   0.6         7 lambda.min  0.7652708         51    56 0.4542336 4565.181 52.08729 4513.093 4617.268
# >   0.7         8 lambda.min  0.6559464         51    56 0.4542626 4565.203 52.13586 4513.067 4617.339
# >   0.8         9 lambda.min  0.5739531         51    56 0.4542839 4565.220 52.17493 4513.045 4617.395
# >   0.9        10 lambda.min  0.5101805         51    56 0.4543004 4565.226 52.21123 4513.015 4617.437
# >   1.0        11 lambda.min  0.5039302         50    56 0.4539600 4565.233 51.06299 4514.170 4616.296
# >   0.2         3 lambda.min  2.0918586         52    57 0.4541221 4565.283 52.50684 4512.776 4617.790
# >   0.1         2 lambda.min  3.4733956         54    59 0.4541195 4566.340 53.64777 4512.692 4619.988
# >   0.0         1 lambda.min  4.8102582        100    70 0.4554361 4572.254 60.31204 4511.942 4632.566
# >   1.0        11 lambda.1se  1.4022158         39    41 0.4438261 4607.125 36.13267 4570.992 4643.258
# >   0.9        10 lambda.1se  1.5580176         39    41 0.4437711 4607.489 36.09172 4571.397 4643.581
# >   0.8         9 lambda.1se  1.7527697         39    41 0.4437007 4607.933 36.01006 4571.923 4643.943
# >   0.7         8 lambda.1se  2.0031654         39    41 0.4436066 4608.501 35.91920 4572.582 4644.420
# >   0.1         2 lambda.1se 10.6072483         42    45 0.4431544 4609.157 38.96493 4570.192 4648.122
# >   0.6         7 lambda.1se  2.3370263         39    41 0.4434772 4609.282 35.82182 4573.460 4645.104
# >   0.2         3 lambda.1se  6.3882341         40    43 0.4430909 4610.156 36.04921 4574.107 4646.205
# >   0.5         6 lambda.1se  2.8044316         39    41 0.4432832 4610.461 35.68968 4574.771 4646.151
# >   0.4         5 lambda.1se  3.5055395         39    41 0.4429729 4612.179 35.46141 4576.718 4647.640
# >   0.3         4 lambda.1se  4.6740527         39    41 0.4424096 4615.094 35.05890 4580.035 4650.153
# >   0.0         1 lambda.1se 25.6709014         82    70 0.4436406 4627.791 60.18381 4567.608 4687.975

# Choose winners under each rule separately 
best_min  <- tuning_results %>%
  filter(s == "lambda.min") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%   # tie-breaks: fewer nonzeros (nzero), larger λ, smaller α
  slice(1)

best_1se  <- tuning_results %>%
  filter(s == "lambda.1se") %>%
  arrange(cvm, nzero, desc(lambda), alpha) %>%
  slice(1)

print(as.data.frame(best_min),  row.names = FALSE)
# > alpha alpha_idx          s  lambda lambda_idx nzero dev_ratio      cvm     cvsd     cvlo     cvup
# >   0.4         5 lambda.min 1.147906         51    56 0.4541241 4565.03 51.89823 4513.132 4616.928
print(as.data.frame(best_1se), row.names = FALSE)
# > alpha alpha_idx          s   lambda lambda_idx nzero dev_ratio      cvm   cvsd     cvlo     cvup
# >     1        11 lambda.1se 1.402216         39    41 0.4438261 4607.125 36.13267 4570.992 4643.258

best_alpha_min     <- best_min$alpha
best_lambda_min    <- best_min$lambda
best_alpha_idx_min <- best_min$alpha_idx

best_alpha_1se     <- best_1se$alpha
best_lambda_1se    <- best_1se$lambda
best_alpha_idx_1se <- best_1se$alpha_idx

message(sprintf("Winner @ lambda.min : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_min, best_lambda_min, best_min$nzero, best_min$cvm, best_min$cvsd))
# > Winner @ lambda.min : alpha = 0.40 | lambda = 1.147906 | nzero = 56 | CVM = 4565.02984 (± 51.89823)
message(sprintf("Winner @ lambda.1se : alpha = %.2f | lambda = %.6f | nzero = %d | CVM = %.5f (± %.5f)",
                best_alpha_1se, best_lambda_1se, best_1se$nzero, best_1se$cvm, best_1se$cvsd))
# > Winner @ lambda.1se : alpha = 1.00 | lambda = 1.402216 | nzero = 41 | CVM = 4607.12501 (± 36.13267)

#### ---- Predict and evaluate on TRAIN / VALID / TEST for both winners ----
cv_best_min  <- cv_list[[best_alpha_idx_min]]
cv_best_min
cv_best_1se  <- cv_list[[best_alpha_idx_1se]]
cv_best_1se

# Coefficients & importances
coef(cv_best_min,  s = best_lambda_min) %>% head()
# > 6 x 1 sparse Matrix of class "dgCMatrix"
# >              s=1.147906
# > (Intercept) 505.8196829
# > MATHMOT       0.9068032
# > MATHEASE      3.3634469
# > MATHPREF      .        
# > EXERPRAC     -2.7448350
# > STUDYHMW     -0.5129674
coef(cv_best_1se,  s = best_lambda_1se) %>% head()
# > 6 x 1 sparse Matrix of class "dgCMatrix"
# >              s=1.402216
# > (Intercept) 509.3406773
# > MATHMOT       .        
# > MATHEASE      1.7816637
# > MATHPREF      .        
# > EXERPRAC     -2.5857207
# > STUDYHMW     -0.1080458

varImp(cv_best_min$glmnet.fit,  lambda = best_lambda_min) %>% head()
# >            Overall
# > MATHMOT  0.9068032
# > MATHEASE 3.3634469
# > MATHPREF 0.0000000
# > EXERPRAC 2.7448350
# > STUDYHMW 0.5129674
# > WORKPAY  3.2352938
varImp(cv_best_1se$glmnet.fit,  lambda = best_lambda_1se) %>% head()
# >            Overall
# > MATHMOT  0.0000000
# > MATHEASE 1.7816637
# > MATHPREF 0.0000000
# > EXERPRAC 2.5857207
# > STUDYHMW 0.1080458
# > WORKPAY  3.1923497

# Predictions
pred_train_min <- as.numeric(predict(cv_best_min, newx = X_train, s = best_lambda_min))
pred_valid_min <- as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min))
pred_test_min  <- as.numeric(predict(cv_best_min,  newx = X_test,  s = best_lambda_min))

pred_train_1se <- as.numeric(predict(cv_best_1se, newx = X_train, s = best_lambda_1se))
pred_valid_1se <- as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se))
pred_test_1se  <- as.numeric(predict(cv_best_1se,  newx = X_test,  s = best_lambda_1se))

# Performance Metrics
metrics_train_min <- compute_metrics(y_train, pred_train_min, w_train)
metrics_valid_min <- compute_metrics(y_valid, pred_valid_min, w_valid)
metrics_test_min  <- compute_metrics(y_test,  pred_test_min,  w_test)

metrics_train_1se <- compute_metrics(y_train, pred_train_1se, w_train)
metrics_valid_1se <- compute_metrics(y_valid, pred_valid_1se, w_valid)
metrics_test_1se  <- compute_metrics(y_test,  pred_test_1se,  w_test)

# Results tables
metric_results_min <- tibble::tibble(
  Rule    = "lambda.min",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_min["rmse"], metrics_valid_min["rmse"], metrics_test_min["rmse"]),
  MAE     = c(metrics_train_min["mae"],  metrics_valid_min["mae"],  metrics_test_min["mae"]),
  Bias    = c(metrics_train_min["bias"], metrics_valid_min["bias"], metrics_test_min["bias"]),
  `Bias%` = c(metrics_train_min["bias_pct"], metrics_valid_min["bias_pct"], metrics_test_min["bias_pct"]),
  R2      = c(metrics_train_min["r2"],   metrics_valid_min["r2"],   metrics_test_min["r2"])
)

metric_results_1se <- tibble::tibble(
  Rule    = "lambda.1se",
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_1se["rmse"], metrics_valid_1se["rmse"], metrics_test_1se["rmse"]),
  MAE     = c(metrics_train_1se["mae"],  metrics_valid_1se["mae"],  metrics_test_1se["mae"]),
  Bias    = c(metrics_train_1se["bias"], metrics_valid_1se["bias"], metrics_test_1se["bias"]),
  `Bias%` = c(metrics_train_1se["bias_pct"], metrics_valid_1se["bias_pct"], metrics_test_1se["bias_pct"]),
  R2      = c(metrics_train_1se["r2"],   metrics_valid_1se["r2"],   metrics_test_1se["r2"])
)

print(as.data.frame(metric_results_min),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias     Bias%        R2
# > lambda.min   Training 66.34700 52.33720  3.271511e-13 1.8091748 0.4541241
# > lambda.min Validation 68.71135 53.55590  2.368783e+00 2.4926513 0.4480028
# > lambda.min       Test 68.33551 54.86742 -4.765675e+00 0.8370998 0.4042167

print(as.data.frame(metric_results_1se),  row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias     Bias%        R2
# > lambda.1se   Training 66.96989 52.82406  2.828188e-13 1.8910523 0.4438261
# > lambda.1se Validation 69.25369 53.94957  2.522156e+00 2.6009392 0.4392546
# > lambda.1se       Test 68.50576 55.28534 -4.954567e+00 0.8767793 0.4012444
print(as.data.frame(bind_rows(metric_results_min, metric_results_1se)), row.names = FALSE)
# >       Rule    Dataset     RMSE      MAE          Bias     Bias%        R2
# > lambda.min   Training 66.34700 52.33720  3.271511e-13 1.8091748 0.4541241
# > lambda.min Validation 68.71135 53.55590  2.368783e+00 2.4926513 0.4480028
# > lambda.min       Test 68.33551 54.86742 -4.765675e+00 0.8370998 0.4042167
# > lambda.1se   Training 66.96989 52.82406  2.828188e-13 1.8910523 0.4438261
# > lambda.1se Validation 69.25369 53.94957  2.522156e+00 2.6009392 0.4392546
# > lambda.1se       Test 68.50576 55.28534 -4.954567e+00 0.8767793 0.4012444

# Sanity check: wrapper vs underlying at the same λ
# all.equal(as.numeric(predict(cv_best_min, newx = X_valid, s = best_lambda_min)),
#           as.numeric(predict(cv_best_min$glmnet.fit, newx = X_valid, s = best_lambda_min)))
# # > [1] TRUE
# all.equal(as.numeric(predict(cv_best_1se, newx = X_valid, s = best_lambda_1se)),
#           as.numeric(predict(cv_best_1se$glmnet.fit, newx = X_valid, s = best_lambda_1se)))
# # > [1] TRUE

## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH (best_alpha_min, best_lambda_min) to all plausible values in mathematics.

### ---- 1) Use best min results ----

#### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)
tic("Fitting main glmnet models (fixed best_alpha_min, best_lambda_min)")
main_models <- lapply(pvmaths, function(pv) {
  
  # TRAIN (final weights)
  y_train <- train_data[[pv]]
  w_train <- train_data[[final_wt]]
  
  # Fit glmnet at chosen alpha + lambda
  mod <- glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    weights = w_train,
    alpha = best_alpha_min,       # best_alpha_min = 0.4
    lambda = best_lambda_min,     # best_lambda_min = 1.147906
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda_min))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(voi_all, collapse = " + "))),
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha_min, best_lambda_min): 0.673 sec elapsed

# Quick look
main_models[[1]]$formula
main_models[[1]]$coefs [1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 505.7806003   0.9107029   3.3605805   0.0000000  -2.7458035  -0.5143385 

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coefs[, 1:6]
# >      (Intercept)   MATHMOT MATHEASE   MATHPREF  EXERPRAC   STUDYHMW
# > [1,]    505.7806 0.9107029 3.360581  0.0000000 -2.745804 -0.5143385
# > [2,]    519.7629 0.0000000 2.997469  1.7854877 -2.826625 -0.9586356
# > [3,]    512.5181 0.0000000 4.775763  0.0000000 -2.888669 -0.6380516
# > [4,]    513.1711 1.8083833 3.720267  1.3112919 -3.310317 -0.6930333
# > [5,]    518.7447 0.0000000 3.969892  0.5612383 -2.843491 -0.8844946
# > [6,]    508.1058 0.0000000 4.492334  0.0000000 -2.875391 -0.5047636
# > [7,]    506.5129 3.1595728 1.460615  1.0791172 -2.834522 -0.9554744
# > [8,]    514.6640 0.0000000 4.536978  4.5043357 -2.917696 -0.4925189
# > [9,]    506.6848 0.0000000 6.032882 -0.8867484 -3.185929 -0.3089529
# > [10,]   506.7047 2.0096751 3.927750  0.0000000 -3.101913 -0.2077948

main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 511.2649662   0.7888334   3.9274530   0.8354722  -2.9530359  -0.6158058 

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
# >                                    Term Estimate |Estimate|
# > 1                               MATHEFF   25.052     25.052
# > 2                                  ESCS   17.159     17.159
# > 3                           LANGNFrench   15.629     15.629
# > 4                               HOMEPOS   14.591     14.591
# > 5                         ST004D01TMale   12.397     12.397
# > 6                REGIONCanada: Manitoba  -12.057     12.057
# > 7           REGIONCanada: New Brunswick  -12.032     12.032
# > 8                              FAMSUPSL  -11.155     11.155
# > 9           LANGNAnother language (CAN)    9.365      9.365
# > 10                               ICTRES   -8.983      8.983
# > 11           REGIONCanada: Saskatchewan   -7.247      7.247
# > 12                       SCHLTYPEPublic   -7.228      7.228
# > 13                               ANXMAT   -6.920      6.920
# > 14                               FAMCON    6.693      6.693
# > 15                REGIONCanada: Ontario    6.685      6.685
# > 16                             CREATEFF   -6.307      6.307
# > 17                             CURIOAGR    5.954      5.954
# > 18            REGIONCanada: Nova Scotia   -5.557      5.557
# > 19        IMMIGFirst-Generation student   -4.619      4.619
# > 20                             CREATOOS   -4.299      4.299
# > 21                              GROSAGR    4.244      4.244
# > 22                              BULLIED   -4.240      4.240
# > 23                              OPENCUL    4.165      4.165
# > 24                             MATHEASE    3.927      3.927
# > 25                              SCHSUST    3.879      3.879
# > 26                             COGACMCO   -3.786      3.786
# > 27                              CREATOP    3.746      3.746
# > 28 SCHLTYPEPrivate Government-dependent    3.613      3.613
# > 29                              WORKPAY   -3.574      3.574
# > 30                               BELONG   -3.530      3.530
# > 31                             CREATSCH   -3.295      3.295
# > 32                               FAMSUP   -3.171      3.171
# > 33                              FEELLAH   -3.065      3.065
# > 34                             EXERPRAC   -2.953      2.953
# > 35                             EMOCOAGR    2.822      2.822
# > 36                             MATHPERS    2.742      2.742
# > 37                             ASSERAGR    2.710      2.710
# > 38                             MATHEF21    2.631      2.631
# > 39       IMMIGSecond-Generation student    2.129      2.129
# > 40                               MACTIV    2.039      2.039
# > 41                 REGIONCanada: Quebec    1.694      1.694
# > 42                             FEELSAFE    1.557      1.557
# > 43       REGIONCanada: British Columbia    1.207      1.207
# > 44                             STRESAGR   -0.989      0.989
# > 45                             MATHPREF    0.835      0.835
# > 46                              OPENART   -0.789      0.789
# > 47                              MATHMOT    0.789      0.789
# > 48                              ABGMATH   -0.762      0.762
# > 49   REGIONCanada: Prince Edward Island    0.710      0.710
# > 50                             CREATFAM   -0.675      0.675
# > 51                             CREENVSC   -0.648      0.648
# > 52                             STUDYHMW   -0.616      0.616
# > 53                              DIGPREP    0.596      0.596
# > 54                               SDLEFF    0.576      0.576
# > 55                              COOPAGR   -0.566      0.566
# > 56                             WORKHOME   -0.563      0.563
# > 57                             PROBSELF   -0.422      0.422
# > 58                             EMPATAGR   -0.380      0.380
# > 59                             COGACRCO    0.281      0.281
# > 60                               EXPOFA   -0.183      0.183
# > 61                              CREATAS    0.179      0.179
# > 62                              DISCLIM    0.126      0.126
# > 63                              IMAGINE   -0.123      0.123
# > 64                              MTTRAIN    0.099      0.099
# > 65                            PERSEVAGR    0.082      0.082
# > 66                             INFOSEEK    0.076      0.076
# > 67                             EXPO21ST   -0.064      0.064
# > 68                              LEARRES    0.034      0.034

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

# --- Weighted R² on TRAIN (point estimates per PV) ---
main_r2s_weighted <- sapply(1:M, function(i) {
  model  <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w      <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.4657868

#### ---- Replicate models using BRR replicate weights ----

set.seed(123)

tic("Fitting replicate glmnet models (BRR weights)")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    mod <- glmnet(
      x = X_train,
      y = y_train,
      family = "gaussian",
      weights = w_train,
      alpha = best_alpha_min,
      lambda = best_lambda_min,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda_min))
    coefs  <- coefs_matrix[, 1]
    names(coefs) <- rownames(coefs_matrix)
    
    list(
      formula = as.formula(paste(pv, "~", paste(voi_all, collapse = " + "))),
      mod     = mod,
      coefs   = coefs
    )
  })
})
toc()
# > Fitting replicate glmnet models (BRR weights): 5.812 sec elapsed

# Example inspect
replicate_models[[1]][[1]]$formula
replicate_models[[1]][[1]]$coefs

# --- Replicate weighted R² on TRAIN (G x M) (optional diagnostic) ---
rep_r2_weighted <- matrix(NA_real_, nrow = G, ncol = M)
for (m in 1:M) {
  y_train <- train_data[[pvmaths[m]]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)  
# > [1] 80 10

#### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

# Organize replicate coefficients: rep_coefs[[m]] is G x (p+1) matrix for PV m
rep_coefs <- lapply(replicate_models, function(m) {
  do.call(rbind, lapply(m, function(g) g$coefs))
})

# BRR sampling variance per PV then average across PVs
sampling_var_matrix_coef <- sapply(1:M, function(m) {
  sweep(rep_coefs[[m]], 2, main_coefs[m, ])^2 |>
    colSums() / (G * (1 - k)^2)
})
# Average sampling variance across PVs (Rubin)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef)

# Imputation variance across PVs
imputation_var_coef <- colSums(
  (main_coefs - matrix(main_coef, nrow = M, ncol = length(main_coef), byrow = TRUE))^2
) / (M - 1)

# Total variance & SE for coefficients
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
se_final_coef  <- sqrt(var_final_coef)

# Z-tests (large-sample normal) for coefficients
Estimate   <- main_coef
`Std. Error` <- se_final_coef
`z value`  <- Estimate / `Std. Error`
p_z        <- 2 * pnorm(-abs(`z value`))
`Pr(>|z|)` <- format.pval(p_z, digits = 3, eps = .Machine$double.eps)
z_Signif   <- symnum(p_z, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols = c("***", "**", "*", ".", ""))

# t-test
# library(survey)
# repw <- train_data[, rep_wts]          # matrix/data.frame of replicate weights
# des <- svrepdesign(
#   weights           = ~ W_FSTUWT,
#   repweights        = repw,
#   type              = "Fay",
#   rho               = 0.5,             # Fay’s factor k = 0.5 in PISA
#   combined.weights  = TRUE,
#   mse               = TRUE,
#   data              = train_data
# )
# dof_t <- degf(des)                             # design-based df
# dof_t
# # > [1] 79

dof_t <- G-1
`t value`  <- `z value`                         # same ratio; distribution differs
p_t        <- 2 * pt(-abs(`t value`), df = dof_t)
`Pr(>|t|)` <- format.pval(p_t, digits = 3, eps = .Machine$double.eps)
t_Signif   <- symnum(p_t, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols   = c("***","**","*",".",""))

##### ---- Final Outputs ----
# Coefficient table (glmnet): pooled estimates with BRR+Rubin SE
coef_table <- tibble::tibble(
  Term         = names(Estimate),
  Estimate     = as.numeric(Estimate),
  `Std. Error` = as.numeric(`Std. Error`),
  `z value`    = as.numeric(`z value`),
  `Pr(>|z|)`   = `Pr(>|z|)`,
  z_Signif     = as.character(z_Signif),
  `t value`    = as.numeric(`t value`),
  `Pr(>|t|)`   = `Pr(>|t|)`,
  t_Signif     = as.character(t_Signif),
)
as.data.frame(coef_table) %>% head() %>% print(row.names = FALSE)
# >        Term    Estimate Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# > (Intercept) 511.2649662  5.6931283  89.8038721  < 2e-16      ***  89.8038721  < 2e-16      ***
# >     MATHMOT   0.7888334  1.2970177   0.6081902 0.543061            0.6081902 0.544806         
# >    MATHEASE   3.9274530  1.3670581   2.8729232 0.004067       **   2.8729232 0.005221       **
# >    MATHPREF   0.8354722  1.6230600   0.5147513 0.606727            0.5147513 0.608164         
# >    EXERPRAC  -2.9530359  0.1978020 -14.9292543  < 2e-16      *** -14.9292543  < 2e-16      ***
# >    STUDYHMW  -0.6158058  0.2787132  -2.2094603 0.027143        *  -2.2094603 0.030038        *
as.data.frame(coef_table) %>% print(row.names = FALSE)                                 
# >                                 Term     Estimate Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# >                          (Intercept) 511.26496622  5.6931283  89.8038721  < 2e-16      ***  89.8038721  < 2e-16      ***
# >                              MATHMOT   0.78883342  1.2970177   0.6081902 0.543061            0.6081902 0.544806         
# >                             MATHEASE   3.92745300  1.3670581   2.8729232 0.004067       **   2.8729232 0.005221       **
# >                             MATHPREF   0.83547223  1.6230600   0.5147513 0.606727            0.5147513 0.608164         
# >                             EXERPRAC  -2.95303585  0.1978020 -14.9292543  < 2e-16      *** -14.9292543  < 2e-16      ***
# >                             STUDYHMW  -0.61580581  0.2787132  -2.2094603 0.027143        *  -2.2094603 0.030038        *
# >                              WORKPAY  -3.57447918  0.2631430 -13.5837921  < 2e-16      *** -13.5837921  < 2e-16      ***
# >                             WORKHOME  -0.56333653  0.1734413  -3.2479955 0.001162       **  -3.2479955 0.001707       **
# >                              HOMEPOS  14.59113471  1.5126823   9.6458685  < 2e-16      ***   9.6458685 5.36e-15      ***
# >                               ICTRES  -8.98288188  0.9035839  -9.9413918  < 2e-16      ***  -9.9413918 1.43e-15      ***
# >                             INFOSEEK   0.07638026  0.4543868   0.1680953 0.866508            0.1680953 0.866938         
# >                              BULLIED  -4.24038079  0.7472513  -5.6746382 1.39e-08      ***  -5.6746382 2.21e-07      ***
# >                             FEELSAFE   1.55663636  0.8186135   1.9015522 0.057230        .   1.9015522 0.060876        .
# >                               BELONG  -3.52989968  1.1166546  -3.1611383 0.001572       **  -3.1611383 0.002229       **
# >                              GROSAGR   4.24375924  0.5045649   8.4107305  < 2e-16      ***   8.4107305 1.38e-12      ***
# >                               ANXMAT  -6.91952655  0.7633105  -9.0651532  < 2e-16      ***  -9.0651532 7.26e-14      ***
# >                              MATHEFF  25.05185708  0.7812714  32.0655008  < 2e-16      ***  32.0655008  < 2e-16      ***
# >                             MATHEF21   2.63130332  1.1771621   2.2352940 0.025398        *   2.2352940 0.028222        *
# >                             MATHPERS   2.74215720  0.8465239   3.2393147 0.001198       **   3.2393147 0.001754       **
# >                               FAMCON   6.69273849  0.5012621  13.3517741  < 2e-16      ***  13.3517741  < 2e-16      ***
# >                             ASSERAGR   2.70987980  0.6150176   4.4061822 1.05e-05      ***   4.4061822 3.27e-05      ***
# >                              COOPAGR  -0.56608847  0.5888886  -0.9612828 0.336410           -0.9612828 0.339343         
# >                             CURIOAGR   5.95422073  0.8994864   6.6195783 3.60e-11      ***   6.6195783 3.95e-09      ***
# >                             EMOCOAGR   2.82197020  0.6896983   4.0916008 4.28e-05      ***   4.0916008 0.000103      ***
# >                             EMPATAGR  -0.37960427  0.6816405  -0.5568981 0.577597           -0.5568981 0.579171         
# >                            PERSEVAGR   0.08167440  0.2915239   0.2801637 0.779352            0.2801637 0.780084         
# >                             STRESAGR  -0.98887453  0.6491901  -1.5232435 0.127698           -1.5232435 0.131691         
# >                               EXPOFA  -0.18342492  0.4648120  -0.3946218 0.693122           -0.3946218 0.694185         
# >                             EXPO21ST  -0.06378324  0.5364913  -0.1188896 0.905363           -0.1188896 0.905665         
# >                             COGACRCO   0.28088486  0.4933922   0.5692933 0.569157            0.5692933 0.570772         
# >                             COGACMCO  -3.78609168  1.0410454  -3.6368172 0.000276      ***  -3.6368172 0.000490      ***
# >                              DISCLIM   0.12614238  0.3521204   0.3582365 0.720166            0.3582365 0.721122         
# >                               FAMSUP  -3.17099501  0.9257890  -3.4251812 0.000614      ***  -3.4251812 0.000977      ***
# >                             CREATFAM  -0.67536218  1.0132015  -0.6665626 0.505052           -0.6665626 0.506994         
# >                             CREATSCH  -3.29495034  1.0158856  -3.2434265 0.001181       **  -3.2434265 0.001732       **
# >                             CREATEFF  -6.30733700  0.8273244  -7.6237770 2.46e-14      ***  -7.6237770 4.70e-11      ***
# >                              CREATOP   3.74588946  0.7116609   5.2635872 1.41e-07      ***   5.2635872 1.18e-06      ***
# >                              IMAGINE  -0.12261914  0.2924219  -0.4193227 0.674980           -0.4193227 0.676118         
# >                              OPENART  -0.78890648  0.5559442  -1.4190389 0.155888           -1.4190389 0.159821         
# >                              CREATAS   0.17909722  0.4581615   0.3909041 0.695868            0.3909041 0.696920         
# >                             CREATOOS  -4.29892569  0.8168401  -5.2628728 1.42e-07      ***  -5.2628728 1.19e-06      ***
# >                             FAMSUPSL -11.15460467  0.6200333 -17.9903324  < 2e-16      *** -17.9903324  < 2e-16      ***
# >                              FEELLAH  -3.06496346  0.8392914  -3.6518468 0.000260      ***  -3.6518468 0.000466      ***
# >                             PROBSELF  -0.42225534  0.5839246  -0.7231333 0.469598           -0.7231333 0.471734         
# >                               SDLEFF   0.57600111  0.7964818   0.7231818 0.469568            0.7231818 0.471704         
# >                              SCHSUST   3.87926467  0.6199170   6.2577159 3.91e-10      ***   6.2577159 1.88e-08      ***
# >                              LEARRES   0.03445623  0.2377419   0.1449313 0.884765            0.1449313 0.885134         
# >                                 ESCS  17.15863373  1.5213687  11.2784193  < 2e-16      ***  11.2784193  < 2e-16      ***
# >                               MACTIV   2.03908061  0.3746877   5.4420806 5.27e-08      ***   5.4420806 5.74e-07      ***
# >                              ABGMATH  -0.76166115  1.0251354  -0.7429859 0.457490           -0.7429859 0.459694         
# >                              MTTRAIN   0.09862054  0.3261603   0.3023683 0.762371            0.3023683 0.763166         
# >                             CREENVSC  -0.64849058  0.5888799  -1.1012273 0.270798           -1.1012273 0.274142         
# >                              OPENCUL   4.16454182  0.7076454   5.8850685 3.98e-09      ***   5.8850685 9.16e-08      ***
# >                              DIGPREP   0.59611130  0.5952831   1.0013913 0.316638            1.0013913 0.319695         
# >   REGIONCanada: Prince Edward Island   0.70967902  2.8614559   0.2480133 0.804124            0.2480133 0.804768         
# >            REGIONCanada: Nova Scotia  -5.55661485  3.5816235  -1.5514235 0.120800           -1.5514235 0.124797         
# >          REGIONCanada: New Brunswick -12.03171098  2.4037161  -5.0054626 5.57e-07      ***  -5.0054626 3.31e-06      ***
# >                 REGIONCanada: Quebec   1.69374333  2.3271984   0.7278036 0.466734            0.7278036 0.468885         
# >                REGIONCanada: Ontario   6.68467363  2.0109119   3.3242002 0.000887      ***   3.3242002 0.001346       **
# >               REGIONCanada: Manitoba -12.05701256  3.1165500  -3.8687049 0.000109      ***  -3.8687049 0.000224      ***
# >           REGIONCanada: Saskatchewan  -7.24663620  3.6832161  -1.9674751 0.049128        *  -1.9674751 0.052638        .
# >                REGIONCanada: Alberta   0.00000000  0.2602412   0.0000000 1.000000            0.0000000 1.000000         
# >       REGIONCanada: British Columbia   1.20736402  1.7210525   0.7015266 0.482974            0.7015266 0.485036         
# >                        ST004D01TMale  12.39684384  0.9268989  13.3745379  < 2e-16      ***  13.3745379  < 2e-16      ***
# >              ST004D01TNot Applicable   0.00000000  0.2060401   0.0000000 1.000000            0.0000000 1.000000         
# >       IMMIGSecond-Generation student   2.12928628  1.2772733   1.6670562 0.095503        .   1.6670562 0.099463        .
# >        IMMIGFirst-Generation student  -4.61893872  2.4478058  -1.8869711 0.059164        .  -1.8869711 0.062838        .
# >                          LANGNFrench  15.62850532  2.5289787   6.1797694 6.42e-10      ***   6.1797694 2.63e-08      ***
# >          LANGNAnother language (CAN)   9.36535473  3.2475642   2.8838089 0.003929       **   2.8838089 0.005061       **
# > SCHLTYPEPrivate Government-dependent   3.61285612  2.6749062   1.3506478 0.176808            1.3506478 0.180665         
# >                       SCHLTYPEPublic  -7.22845235  4.1583195  -1.7383110 0.082156        .  -1.7383110 0.086052        .

# --- Sorted table by |estimate|---
coef_table %>%
  mutate(abs_est = abs(Estimate)) %>%
  arrange(desc(abs_est)) %>%
  select(-abs_est) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >                                 Term     Estimate Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# >                          (Intercept) 511.26496622  5.6931283  89.8038721  < 2e-16      ***  89.8038721  < 2e-16      ***
# >                              MATHEFF  25.05185708  0.7812714  32.0655008  < 2e-16      ***  32.0655008  < 2e-16      ***
# >                                 ESCS  17.15863373  1.5213687  11.2784193  < 2e-16      ***  11.2784193  < 2e-16      ***
# >                          LANGNFrench  15.62850532  2.5289787   6.1797694 6.42e-10      ***   6.1797694 2.63e-08      ***
# >                              HOMEPOS  14.59113471  1.5126823   9.6458685  < 2e-16      ***   9.6458685 5.36e-15      ***
# >                        ST004D01TMale  12.39684384  0.9268989  13.3745379  < 2e-16      ***  13.3745379  < 2e-16      ***
# >               REGIONCanada: Manitoba -12.05701256  3.1165500  -3.8687049 0.000109      ***  -3.8687049 0.000224      ***
# >          REGIONCanada: New Brunswick -12.03171098  2.4037161  -5.0054626 5.57e-07      ***  -5.0054626 3.31e-06      ***
# >                             FAMSUPSL -11.15460467  0.6200333 -17.9903324  < 2e-16      *** -17.9903324  < 2e-16      ***
# >          LANGNAnother language (CAN)   9.36535473  3.2475642   2.8838089 0.003929       **   2.8838089 0.005061       **
# >                               ICTRES  -8.98288188  0.9035839  -9.9413918  < 2e-16      ***  -9.9413918 1.43e-15      ***
# >           REGIONCanada: Saskatchewan  -7.24663620  3.6832161  -1.9674751 0.049128        *  -1.9674751 0.052638        .
# >                       SCHLTYPEPublic  -7.22845235  4.1583195  -1.7383110 0.082156        .  -1.7383110 0.086052        .
# >                               ANXMAT  -6.91952655  0.7633105  -9.0651532  < 2e-16      ***  -9.0651532 7.26e-14      ***
# >                               FAMCON   6.69273849  0.5012621  13.3517741  < 2e-16      ***  13.3517741  < 2e-16      ***
# >                REGIONCanada: Ontario   6.68467363  2.0109119   3.3242002 0.000887      ***   3.3242002 0.001346       **
# >                             CREATEFF  -6.30733700  0.8273244  -7.6237770 2.46e-14      ***  -7.6237770 4.70e-11      ***
# >                             CURIOAGR   5.95422073  0.8994864   6.6195783 3.60e-11      ***   6.6195783 3.95e-09      ***
# >            REGIONCanada: Nova Scotia  -5.55661485  3.5816235  -1.5514235 0.120800           -1.5514235 0.124797         
# >        IMMIGFirst-Generation student  -4.61893872  2.4478058  -1.8869711 0.059164        .  -1.8869711 0.062838        .
# >                             CREATOOS  -4.29892569  0.8168401  -5.2628728 1.42e-07      ***  -5.2628728 1.19e-06      ***
# >                              GROSAGR   4.24375924  0.5045649   8.4107305  < 2e-16      ***   8.4107305 1.38e-12      ***
# >                              BULLIED  -4.24038079  0.7472513  -5.6746382 1.39e-08      ***  -5.6746382 2.21e-07      ***
# >                              OPENCUL   4.16454182  0.7076454   5.8850685 3.98e-09      ***   5.8850685 9.16e-08      ***
# >                             MATHEASE   3.92745300  1.3670581   2.8729232 0.004067       **   2.8729232 0.005221       **
# >                              SCHSUST   3.87926467  0.6199170   6.2577159 3.91e-10      ***   6.2577159 1.88e-08      ***
# >                             COGACMCO  -3.78609168  1.0410454  -3.6368172 0.000276      ***  -3.6368172 0.000490      ***
# >                              CREATOP   3.74588946  0.7116609   5.2635872 1.41e-07      ***   5.2635872 1.18e-06      ***
# > SCHLTYPEPrivate Government-dependent   3.61285612  2.6749062   1.3506478 0.176808            1.3506478 0.180665         
# >                              WORKPAY  -3.57447918  0.2631430 -13.5837921  < 2e-16      *** -13.5837921  < 2e-16      ***
# >                               BELONG  -3.52989968  1.1166546  -3.1611383 0.001572       **  -3.1611383 0.002229       **
# >                             CREATSCH  -3.29495034  1.0158856  -3.2434265 0.001181       **  -3.2434265 0.001732       **
# >                               FAMSUP  -3.17099501  0.9257890  -3.4251812 0.000614      ***  -3.4251812 0.000977      ***
# >                              FEELLAH  -3.06496346  0.8392914  -3.6518468 0.000260      ***  -3.6518468 0.000466      ***
# >                             EXERPRAC  -2.95303585  0.1978020 -14.9292543  < 2e-16      *** -14.9292543  < 2e-16      ***
# >                             EMOCOAGR   2.82197020  0.6896983   4.0916008 4.28e-05      ***   4.0916008 0.000103      ***
# >                             MATHPERS   2.74215720  0.8465239   3.2393147 0.001198       **   3.2393147 0.001754       **
# >                             ASSERAGR   2.70987980  0.6150176   4.4061822 1.05e-05      ***   4.4061822 3.27e-05      ***
# >                             MATHEF21   2.63130332  1.1771621   2.2352940 0.025398        *   2.2352940 0.028222        *
# >       IMMIGSecond-Generation student   2.12928628  1.2772733   1.6670562 0.095503        .   1.6670562 0.099463        .
# >                               MACTIV   2.03908061  0.3746877   5.4420806 5.27e-08      ***   5.4420806 5.74e-07      ***
# >                 REGIONCanada: Quebec   1.69374333  2.3271984   0.7278036 0.466734            0.7278036 0.468885         
# >                             FEELSAFE   1.55663636  0.8186135   1.9015522 0.057230        .   1.9015522 0.060876        .
# >       REGIONCanada: British Columbia   1.20736402  1.7210525   0.7015266 0.482974            0.7015266 0.485036         
# >                             STRESAGR  -0.98887453  0.6491901  -1.5232435 0.127698           -1.5232435 0.131691         
# >                             MATHPREF   0.83547223  1.6230600   0.5147513 0.606727            0.5147513 0.608164         
# >                              OPENART  -0.78890648  0.5559442  -1.4190389 0.155888           -1.4190389 0.159821         
# >                              MATHMOT   0.78883342  1.2970177   0.6081902 0.543061            0.6081902 0.544806         
# >                              ABGMATH  -0.76166115  1.0251354  -0.7429859 0.457490           -0.7429859 0.459694         
# >   REGIONCanada: Prince Edward Island   0.70967902  2.8614559   0.2480133 0.804124            0.2480133 0.804768         
# >                             CREATFAM  -0.67536218  1.0132015  -0.6665626 0.505052           -0.6665626 0.506994         
# >                             CREENVSC  -0.64849058  0.5888799  -1.1012273 0.270798           -1.1012273 0.274142         
# >                             STUDYHMW  -0.61580581  0.2787132  -2.2094603 0.027143        *  -2.2094603 0.030038        *
# >                              DIGPREP   0.59611130  0.5952831   1.0013913 0.316638            1.0013913 0.319695         
# >                               SDLEFF   0.57600111  0.7964818   0.7231818 0.469568            0.7231818 0.471704         
# >                              COOPAGR  -0.56608847  0.5888886  -0.9612828 0.336410           -0.9612828 0.339343         
# >                             WORKHOME  -0.56333653  0.1734413  -3.2479955 0.001162       **  -3.2479955 0.001707       **
# >                             PROBSELF  -0.42225534  0.5839246  -0.7231333 0.469598           -0.7231333 0.471734         
# >                             EMPATAGR  -0.37960427  0.6816405  -0.5568981 0.577597           -0.5568981 0.579171         
# >                             COGACRCO   0.28088486  0.4933922   0.5692933 0.569157            0.5692933 0.570772         
# >                               EXPOFA  -0.18342492  0.4648120  -0.3946218 0.693122           -0.3946218 0.694185         
# >                              CREATAS   0.17909722  0.4581615   0.3909041 0.695868            0.3909041 0.696920         
# >                              DISCLIM   0.12614238  0.3521204   0.3582365 0.720166            0.3582365 0.721122         
# >                              IMAGINE  -0.12261914  0.2924219  -0.4193227 0.674980           -0.4193227 0.676118         
# >                              MTTRAIN   0.09862054  0.3261603   0.3023683 0.762371            0.3023683 0.763166         
# >                            PERSEVAGR   0.08167440  0.2915239   0.2801637 0.779352            0.2801637 0.780084         
# >                             INFOSEEK   0.07638026  0.4543868   0.1680953 0.866508            0.1680953 0.866938         
# >                             EXPO21ST  -0.06378324  0.5364913  -0.1188896 0.905363           -0.1188896 0.905665         
# >                              LEARRES   0.03445623  0.2377419   0.1449313 0.884765            0.1449313 0.885134         
# >                REGIONCanada: Alberta   0.00000000  0.2602412   0.0000000 1.000000            0.0000000 1.000000         
# >              ST004D01TNot Applicable   0.00000000  0.2060401   0.0000000 1.000000            0.0000000 1.000000         

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

# R-squared (TRAIN) SE via BRR + Rubin 
sampling_var_r2s_weighted <- sapply(1:M, function(m) {
  sum((rep_r2_weighted[, m] - main_r2s_weighted[m])^2) / (G * (1 - k)^2)
})
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
se_final_r2_weighted  <- sqrt(var_final_r2_weighted)
r2_weighted_table <- tibble::tibble(
  Metric       = "R-squared (Weighted, Train)",
  Estimate     = main_r2_weighted,
  `Std. Error` = se_final_r2_weighted
)
print(as.data.frame(r2_weighted_table), row.names = FALSE)
# >                       Metric  Estimate  Std. Error
# >  R-squared (Weighted, Train) 0.4657868 0.007582851

#### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
train_metrics_main

# Combine across plausible values
train_metric_main <- colMeans(train_metrics_main)

# --- Replicate predictions for training data ---
tic("Computing train_metrics_replicates (glmnet)")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- train_data[[pvmaths[m]]]
    w <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing train_metrics_replicates (glmnet): 1.498 sec elapsed
class(train_metrics_replicates[[1]]); dim(train_metrics_replicates[[1]])
# > [1] "matrix" "array" 
# > [1] 80  6

# BRR sampling variance on TRAIN (vectorized across metrics)
sampling_var_matrix_train <- sapply(1:M, function(m) {
  sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
    colSums() / (G * (1 - k)^2)
})
sampling_var_train <- rowMeans(sampling_var_matrix_train)

# Rubin imputation variance on TRAIN
imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)

# Total variance and SE
var_final_train <- sampling_var_train + (1 + 1/M) * imputation_var_train
se_final_train  <- sqrt(var_final_train)

# Confidence intervals
ci_lower_train  <- train_metric_main - z_crit * se_final_train
ci_upper_train  <- train_metric_main + z_crit * se_final_train
ci_length_train <- ci_upper_train - ci_lower_train

# Format results
train_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train,   scientific = FALSE),
  CI_lower       = format(ci_lower_train,   scientific = FALSE),
  CI_upper       = format(ci_upper_train,   scientific = FALSE),
  CI_length      = format(ci_length_train,  scientific = FALSE)
)
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                 CI_lower               CI_upper             CI_length
# >       MSE 4203.4001354012052615872 82.8189427041150736386 4041.0779904634537160746 4365.72228033895680710 324.64428987550309103
# >      RMSE   64.8311788433813944721  0.6341992375792793180   63.5881711787032486427   66.07418650805954030   2.48601532935629166
# >       MAE   51.4099047495506269456  0.4722465290874555688   50.4843185607151667682   52.33549093838608712   1.85117237767092035
# >      Bias    0.0000000000001570997  0.0000000000004544778   -0.0000000000007336603    0.00000000000104786   0.00000000000178152
# >     Bias%    1.7333696150209252362  0.0338522390592912448    1.6670204456686743555    1.79971878437317612   0.13269833870450176
# > R-squared    0.4657867642436187561  0.0075828510975950742    0.4509246491922023758    0.48064887929503514   0.02972423010283276

#### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
valid_metric_main <- colMeans(valid_metrics_main)

# Replicate predictions on validation set
tic("Computing valid_metrics_replicates (glmnet)")
valid_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- valid_data[[pvmaths[m]]]
    w <- valid_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 0.875 sec elapsed

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
valid_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(valid_metric_main, scientific = FALSE),
  Standard_error = format(se_final_valid,   scientific = FALSE),
  CI_lower       = format(ci_lower_valid,   scientific = FALSE),
  CI_upper       = format(ci_upper_valid,   scientific = FALSE),
  CI_length      = format(ci_length_valid,  scientific = FALSE)
)
print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE    4690.463870   167.52842997 4362.1141809 5018.8135592 656.69937824
# >      RMSE      68.478657     1.22300343   66.0816142   70.8756995   4.79408537
# >       MAE      54.007528     1.22362248   51.6092720   56.4057840   4.79651198
# >      Bias       1.919696     1.93833620   -1.8793734    5.7187649   7.59813829
# >     Bias%       2.352714     0.40879224    1.5514957    3.1539318   1.60243612
# > R-squared       0.453672     0.01283276    0.4285203    0.4788238   0.05030349

#### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_min))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
test_metric_main <- colMeans(test_metrics_main)

# Replicate predictions on test set
tic("Computing test_metrics_replicates (glmnet)")
test_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- test_data[[pvmaths[m]]]
    w <- test_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_min))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates (glmnet): 0.965 sec elapsed

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
test_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(test_metric_main, scientific = FALSE),
  Standard_error = format(se_final_test,   scientific = FALSE),
  CI_lower       = format(ci_lower_test,   scientific = FALSE),
  CI_upper       = format(ci_upper_test,   scientific = FALSE),
  CI_length      = format(ci_length_test,  scientific = FALSE)
)
print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   4872.3089840   283.61616568 4316.4315138 5428.1864542 1111.75494034
# >      RMSE     69.7783848     2.03496834   65.7899201   73.7668494    7.97692931
# >       MAE     55.3654575     1.35747174   52.7048618   58.0260532    5.32119145
# >      Bias     -2.4257738     1.59180798   -5.5456601    0.6941125    6.23977261
# >     Bias%      1.4382762     0.42644177    0.6024657    2.2740867    1.67162100
# > R-squared      0.4023685     0.01766422    0.3677472    0.4369897    0.06924247

#### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.

# Helper
evaluate_split <- function(X, split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           pvmaths, s_value) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model  <- main_models[[i]]$mod
    y      <- split_data[[pvmaths[i]]]
    w      <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X, s = s_value))
    compute_metrics(y_true = y, y_pred = y_pred, w = w)
  }) |> t() |> as.data.frame()
  main_point <- colMeans(main_metrics_df)
  
  # Replicate metrics across PVs
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model  <- replicate_models[[m]][[g]]$mod
      y      <- split_data[[pvmaths[m]]]
      w      <- split_data[[rep_wts[g]]]
      y_pred <- as.numeric(predict(model, newx = X, s = s_value))
      compute_metrics(y_true = y, y_pred = y_pred, w = w)
    }) |> t()
  })
  
  # BRR sampling var (average across PVs)
  sampling_var_matrix <- sapply(1:M, function(m) {
    sweep(replicate_metrics[[m]], 2, unlist(main_metrics_df[m, ]))^2 |>
      colSums() / (G * (1 - k)^2)
  })
  sampling_var <- rowMeans(sampling_var_matrix)
  
  # Rubin imputation var across PVs
  imputation_var <- colSums((main_metrics_df - matrix(main_point, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
  
  # Totals + CIs
  var_final <- sampling_var + (1 + 1/M) * imputation_var
  se_final  <- sqrt(var_final)
  ci_lower  <- main_point - z_crit * se_final
  ci_upper  <- main_point + z_crit * se_final
  ci_length <- ci_upper - ci_lower
  
  tibble::tibble(
    Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
    Point_estimate = format(main_point, scientific = FALSE),
    Standard_error = format(se_final,  scientific = FALSE),
    CI_lower       = format(ci_lower,  scientific = FALSE),
    CI_upper       = format(ci_upper,  scientific = FALSE),
    CI_length      = format(ci_length, scientific = FALSE)
  )
}

# Evaluate on each split
train_eval <- evaluate_split(X_train, train_data, main_models, replicate_models,
                             final_wt, rep_wts, M, G, k, z_crit, pvmaths, best_lambda_min)
valid_eval <- evaluate_split(X_valid, valid_data, main_models, replicate_models,
                             final_wt, rep_wts, M, G, k, z_crit, pvmaths, best_lambda_min)
test_eval  <- evaluate_split(X_test,  test_data,  main_models, replicate_models,
                             final_wt, rep_wts, M, G, k, z_crit, pvmaths, best_lambda_min)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

#### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                 CI_lower               CI_upper             CI_length
# >       MSE 4203.4001354012052615872 82.8189427041150736386 4041.0779904634537160746 4365.72228033895680710 324.64428987550309103
# >      RMSE   64.8311788433813944721  0.6341992375792793180   63.5881711787032486427   66.07418650805954030   2.48601532935629166
# >       MAE   51.4099047495506269456  0.4722465290874555688   50.4843185607151667682   52.33549093838608712   1.85117237767092035
# >      Bias    0.0000000000001570997  0.0000000000004544778   -0.0000000000007336603    0.00000000000104786   0.00000000000178152
# >     Bias%    1.7333696150209252362  0.0338522390592912448    1.6670204456686743555    1.79971878437317612   0.13269833870450176
# > R-squared    0.4657867642436187561  0.0075828510975950742    0.4509246491922023758    0.48064887929503514   0.02972423010283276

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE    4690.463870   167.52842997 4362.1141809 5018.8135592 656.69937824
# >      RMSE      68.478657     1.22300343   66.0816142   70.8756995   4.79408537
# >       MAE      54.007528     1.22362248   51.6092720   56.4057840   4.79651198
# >      Bias       1.919696     1.93833620   -1.8793734    5.7187649   7.59813829
# >     Bias%       2.352714     0.40879224    1.5514957    3.1539318   1.60243612
# > R-squared       0.453672     0.01283276    0.4285203    0.4788238   0.05030349

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   4872.3089840   283.61616568 4316.4315138 5428.1864542 1111.75494034
# >      RMSE     69.7783848     2.03496834   65.7899201   73.7668494    7.97692931
# >       MAE     55.3654575     1.35747174   52.7048618   58.0260532    5.32119145
# >      Bias     -2.4257738     1.59180798   -5.5456601    0.6941125    6.23977261
# >     Bias%      1.4382762     0.42644177    0.6024657    2.2740867    1.67162100
# > R-squared      0.4023685     0.01766422    0.3677472    0.4369897    0.06924247

# --- Keep only four decimals ---
# library(dplyr)
# library(readr)   # for parse_number

train_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        64.8312         0.6342  63.5882  66.0742    2.4860
# >       MAE        51.4099         0.4722  50.4843  52.3355    1.8512
# >      Bias         0.0000         0.0000  -0.0000   0.0000    0.0000
# >     Bias%         1.7334         0.0339   1.6670   1.7997    0.1327
# > R-squared         0.4658         0.0076   0.4509   0.4806    0.0297

valid_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        68.4787         1.2230  66.0816  70.8757    4.7941
# >       MAE        54.0075         1.2236  51.6093  56.4058    4.7965
# >      Bias         1.9197         1.9383  -1.8794   5.7188    7.5981
# >     Bias%         2.3527         0.4088   1.5515   3.1539    1.6024
# > R-squared         0.4537         0.0128   0.4285   0.4788    0.0503

test_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        69.7784         2.0350  65.7899  73.7668    7.9769
# >       MAE        55.3655         1.3575  52.7049  58.0261    5.3212
# >      Bias        -2.4258         1.5918  -5.5457   0.6941    6.2398
# >     Bias%         1.4383         0.4264   0.6025   2.2741    1.6716
# > R-squared         0.4024         0.0177   0.3677   0.4370    0.0692

# --- Keep only two decimals ---
train_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          64.83           0.63    63.59    66.07      2.49
# >       MAE          51.41           0.47    50.48    52.34      1.85
# >      Bias           0.00           0.00    -0.00     0.00      0.00
# >     Bias%           1.73           0.03     1.67     1.80      0.13
# > R-squared           0.47           0.01     0.45     0.48      0.03

valid_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          68.48           1.22    66.08    70.88      4.79
# >       MAE          54.01           1.22    51.61    56.41      4.80
# >      Bias           1.92           1.94    -1.88     5.72      7.60
# >     Bias%           2.35           0.41     1.55     3.15      1.60
# > R-squared           0.45           0.01     0.43     0.48      0.05

test_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          69.78           2.03    65.79    73.77      7.98
# >       MAE          55.37           1.36    52.70    58.03      5.32
# >      Bias          -2.43           1.59    -5.55     0.69      6.24
# >     Bias%           1.44           0.43     0.60     2.27      1.67
# > R-squared           0.40           0.02     0.37     0.44      0.07

# --- Export to word---
library(officer); library(flextable)

fmt2 <- \(df) df %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  )

performance_tables_glmnet_voi_all_v2p4_bestmin <- read_docx() %>%
  body_add_par("Performance - Elastic Net (glmnet, voi_all, v2.4, best min)", style = "Normal") %>%  
  body_add_par("Training performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(train_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Validation performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(valid_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Test performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(test_eval))  %>% align(j = 1:6, align = "right", part = "all") %>% autofit())

print(performance_tables_glmnet_voi_all_v2p4_bestmin, target = "performance_tables_glmnet_voi_all_v2p4_bestmin.docx")

### ---- 2) Use best 1se results ----

#### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)
tic("Fitting main glmnet models (fixed best_alpha_1se, best_lambda_1se)")
main_models <- lapply(pvmaths, function(pv) {
  
  # TRAIN (final weights)
  y_train <- train_data[[pv]] 
  w_train <- train_data[[final_wt]]
  
  # Fit glmnet at chosen alpha + lambda
  mod <- glmnet(
    x = X_train,
    y = y_train,
    family = "gaussian",
    weights = w_train,
    alpha = best_alpha_1se,       # best_alpha_1se = 1
    lambda = best_lambda_1se,     # best_lambda_1se = 1.402216
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda_1se))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(voi_all, collapse = " + "))),
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha_1se, best_lambda_1se): 1.325 sec elapsed

# Quick look
main_models[[1]]$formula
main_models[[1]]$coefs[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 509.2986166   0.0000000   1.7819257   0.0000000  -2.5836317  -0.1081094

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coefs[,1:6]
# >      (Intercept) MATHMOT MATHEASE   MATHPREF  EXERPRAC    STUDYHMW
# > [1,]    509.2986       0 1.781926 0.00000000 -2.583632 -0.10810939
# > [2,]    518.0490       0 1.835997 0.47134806 -2.617615 -0.59551961
# > [3,]    513.7316       0 3.165284 0.00000000 -2.720159 -0.22964531
# > [4,]    515.5368       0 2.848503 0.05396263 -3.107249 -0.29993782
# > [5,]    519.6220       0 2.455145 0.00000000 -2.659769 -0.45809804
# > [6,]    511.2459       0 2.642095 0.00000000 -2.725886  0.00000000
# > [7,]    510.0197       0 0.268711 0.04845318 -2.701631 -0.47867327
# > [8,]    514.3921       0 3.308925 3.17365362 -2.746558 -0.06701187
# > [9,]    507.8455       0 3.788770 0.00000000 -2.970903  0.00000000
# > [10,]   509.9780       0 2.444743 0.00000000 -2.883408  0.00000000

main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef[1:6]
# >  (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# >  512.9719075   0.0000000   2.4540097   0.3747417  -2.7716809  -0.2236995 

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
# >                                    Term Estimate |Estimate|
# > 1                               MATHEFF   26.472     26.472
# > 2                                  ESCS   17.819     17.819
# > 3                           LANGNFrench   11.220     11.220
# > 4                         ST004D01TMale   10.981     10.981
# > 5                              FAMSUPSL  -10.770     10.770
# > 6                               HOMEPOS    9.447      9.447
# > 7                REGIONCanada: Manitoba   -8.641      8.641
# > 8           REGIONCanada: New Brunswick   -8.508      8.508
# > 9                                ANXMAT   -6.936      6.936
# > 10                       SCHLTYPEPublic   -6.496      6.496
# > 11                               FAMCON    6.398      6.398
# > 12                             CURIOAGR    5.385      5.385
# > 13                               ICTRES   -4.818      4.818
# > 14           REGIONCanada: Saskatchewan   -4.594      4.594
# > 15                             CREATEFF   -4.486      4.486
# > 16                REGIONCanada: Ontario    4.124      4.124
# > 17          LANGNAnother language (CAN)    3.797      3.797
# > 18                              GROSAGR    3.648      3.648
# > 19                             CREATOOS   -3.585      3.585
# > 20                              WORKPAY   -3.553      3.553
# > 21                              OPENCUL    3.293      3.293
# > 22                              BULLIED   -3.061      3.061
# > 23                             EXERPRAC   -2.772      2.772
# > 24                             COGACMCO   -2.686      2.686
# > 25            REGIONCanada: Nova Scotia   -2.578      2.578
# > 26                             CREATSCH   -2.497      2.497
# > 27                             MATHEASE    2.454      2.454
# > 28                               FAMSUP   -2.300      2.300
# > 29                              SCHSUST    2.294      2.294
# > 30       IMMIGSecond-Generation student    1.755      1.755
# > 31                               MACTIV    1.743      1.743
# > 32                              FEELLAH   -1.729      1.729
# > 33                             EMOCOAGR    1.720      1.720
# > 34                             MATHEF21    1.668      1.668
# > 35 SCHLTYPEPrivate Government-dependent    1.633      1.633
# > 36        IMMIGFirst-Generation student   -1.543      1.543
# > 37                              CREATOP    1.389      1.389
# > 38                             ASSERAGR    1.326      1.326
# > 39                             MATHPERS    1.298      1.298
# > 40                               BELONG   -1.187      1.187
# > 41                 REGIONCanada: Quebec    0.940      0.940
# > 42                             FEELSAFE    0.600      0.600
# > 43                             WORKHOME   -0.447      0.447
# > 44                             MATHPREF    0.375      0.375
# > 45                             CREATFAM   -0.270      0.270
# > 46                             STUDYHMW   -0.224      0.224
# > 47                              DIGPREP    0.219      0.219
# > 48                              ABGMATH   -0.195      0.195
# > 49                             PROBSELF   -0.176      0.176
# > 50                             EMPATAGR   -0.047      0.047
# > 51                               SDLEFF    0.043      0.043
# > 52                               EXPOFA   -0.037      0.037
# > 53                             EXPO21ST   -0.032      0.032
# > 54                              OPENART   -0.027      0.027
# > 55                            PERSEVAGR    0.017      0.017
# > 56                              COOPAGR   -0.010      0.010

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

# --- Weighted R² on TRAIN (point estimates per PV) ---
main_r2s_weighted <- sapply(1:M, function(i) {
  model  <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w      <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.4547629

#### ---- Replicate models using BRR replicate weights ----

set.seed(123)
tic("Fitting replicate glmnet models (BRR weights)")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    
    y_train <- train_data[[pv]]
    w_train <- train_data[[w]]
    
    mod <- glmnet(
      x = X_train,
      y = y_train,
      family = "gaussian",
      weights = w_train,
      alpha = best_alpha_1se,
      lambda = best_lambda_1se,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda_1se))
    coefs  <- coefs_matrix[, 1]
    names(coefs) <- rownames(coefs_matrix)
    
    list(
      formula = as.formula(paste(pv, "~", paste(voi_all, collapse = " + "))),
      mod     = mod,
      coefs   = coefs
    )
  })
})
toc()
# > Fitting replicate glmnet models (BRR weights): 5.47 sec elapsed

# Example inspect
replicate_models[[1]][[1]]$formula
replicate_models[[1]][[1]]$coefs

# --- Replicate weighted R² on TRAIN (G x M) (optional diagnostic) ---
rep_r2_weighted <- matrix(NA_real_, nrow = G, ncol = M)
for (m in 1:M) {
  
  y_train <- train_data[[pvmaths[m]]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)  # 80 x 10

#### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

# Organize replicate coefficients: rep_coefs[[m]] is G x (p+1) matrix for PV m
rep_coefs <- lapply(replicate_models, function(m) {
  do.call(rbind, lapply(m, function(g) g$coefs))
})

# BRR sampling variance per PV then average across PVs
sampling_var_matrix_coef <- sapply(1:M, function(m) {
  sweep(rep_coefs[[m]], 2, main_coefs[m, ])^2 |>
    colSums() / (G * (1 - k)^2)
})
# Average sampling variance across PVs (Rubin)
sampling_var_coef <- rowMeans(sampling_var_matrix_coef)

# Imputation variance across PVs
imputation_var_coef <- colSums(
  (main_coefs - matrix(main_coef, nrow = M, ncol = length(main_coef), byrow = TRUE))^2
) / (M - 1)

# Total variance & SE for coefficients
var_final_coef <- sampling_var_coef + (1 + 1/M) * imputation_var_coef
se_final_coef  <- sqrt(var_final_coef)

# Z-tests (large-sample normal) for coefficients
Estimate   <- main_coef
`Std. Error` <- se_final_coef
`z value`  <- Estimate / `Std. Error`
p_z        <- 2 * pnorm(-abs(`z value`))
`Pr(>|z|)` <- format.pval(p_z, digits = 3, eps = .Machine$double.eps)
z_Signif   <- symnum(p_z, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols = c("***", "**", "*", ".", ""))

# t-test
# library(survey)
# repw <- train_data[, rep_wts]          # matrix/data.frame of replicate weights
# des <- svrepdesign(
#   weights           = ~ W_FSTUWT,
#   repweights        = repw,
#   type              = "Fay",
#   rho               = 0.5,             # Fay’s factor k = 0.5 in PISA
#   combined.weights  = TRUE,
#   mse               = TRUE,
#   data              = train_data
# )
# dof_t <- degf(des)                             # design-based df
# dof_t
# # > [1] 79

dof_t <- G-1
`t value`  <- `z value`                         # same ratio; distribution differs
p_t        <- 2 * pt(-abs(`t value`), df = dof_t)
`Pr(>|t|)` <- format.pval(p_t, digits = 3, eps = .Machine$double.eps)
t_Signif   <- symnum(p_t, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols   = c("***","**","*",".",""))

##### ---- Final Outputs ----
# Coefficient table (glmnet): pooled estimates with BRR+Rubin SE
coef_table <- tibble::tibble(
  Term         = names(Estimate),
  Estimate     = as.numeric(Estimate),
  `Std. Error` = as.numeric(`Std. Error`),
  `z value`    = as.numeric(`z value`),
  `Pr(>|z|)`   = `Pr(>|z|)`,
  z_Signif     = as.character(z_Signif),
  `t value`    = as.numeric(`t value`),
  `Pr(>|t|)`   = `Pr(>|t|)`,
  t_Signif     = as.character(t_Signif),
)
as.data.frame(coef_table) %>% head() %>% print(row.names = FALSE)
as.data.frame(coef_table) %>% print(row.names = FALSE)
# >                                 Term      Estimate Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# >                          (Intercept) 512.971907494 4.30842982 119.0623797  < 2e-16      *** 119.0623797  < 2e-16      ***
# >                              MATHMOT   0.000000000 0.18696160   0.0000000  1.00000            0.0000000 1.000000         
# >                             MATHEASE   2.454009655 1.13890681   2.1547063  0.03118        *   2.1547063 0.034231        *
# >                             MATHPREF   0.374741749 1.07295291   0.3492621  0.72689            0.3492621 0.727822         
# >                             EXERPRAC  -2.771680923 0.18069192 -15.3392630  < 2e-16      *** -15.3392630  < 2e-16      ***
# >                             STUDYHMW  -0.223699531 0.23952010  -0.9339489  0.35033           -0.9339489 0.353176         
# >                              WORKPAY  -3.553103473 0.26445746 -13.4354443  < 2e-16      *** -13.4354443  < 2e-16      ***
# >                             WORKHOME  -0.446921987 0.18821646  -2.3745107  0.01757        *  -2.3745107 0.019999        *
# >                              HOMEPOS   9.446988043 1.51240177   6.2463482 4.20e-10      ***   6.2463482 1.98e-08      ***
# >                               ICTRES  -4.817777480 0.91564635  -5.2616138 1.43e-07      ***  -5.2616138 1.19e-06      ***
# >                             INFOSEEK   0.000000000 0.04364172   0.0000000  1.00000            0.0000000 1.000000         
# >                              BULLIED  -3.061453682 0.76736691  -3.9895565 6.62e-05      ***  -3.9895565 0.000147      ***
# >                             FEELSAFE   0.600216115 0.63191506   0.9498367  0.34220            0.9498367 0.345092         
# >                               BELONG  -1.186895211 1.02563670  -1.1572277  0.24718           -1.1572277 0.250666         
# >                              GROSAGR   3.647737933 0.52063492   7.0063259 2.45e-12      ***   7.0063259 7.27e-10      ***
# >                               ANXMAT  -6.935507467 0.77396205  -8.9610433  < 2e-16      ***  -8.9610433 1.16e-13      ***
# >                              MATHEFF  26.471752386 0.76467229  34.6184279  < 2e-16      ***  34.6184279  < 2e-16      ***
# >                             MATHEF21   1.668069764 1.16437222   1.4325915  0.15197            1.4325915 0.155920         
# >                             MATHPERS   1.297621343 0.92394580   1.4044345  0.16019            1.4044345 0.164109         
# >                               FAMCON   6.397865306 0.50556074  12.6549884  < 2e-16      ***  12.6549884  < 2e-16      ***
# >                             ASSERAGR   1.326176524 0.61867769   2.1435661  0.03207        *   2.1435661 0.035144        *
# >                              COOPAGR  -0.009848235 0.06866508  -0.1434242  0.88596           -0.1434242 0.886320         
# >                             CURIOAGR   5.384672517 0.89574332   6.0114012 1.84e-09      ***   6.0114012 5.38e-08      ***
# >                             EMOCOAGR   1.720109571 0.57302238   3.0018192  0.00268       **   3.0018192 0.003591       **
# >                             EMPATAGR  -0.047346626 0.15539818  -0.3046794  0.76061           -0.3046794 0.761412         
# >                            PERSEVAGR   0.017115253 0.07877258   0.2172742  0.82799            0.2172742 0.828555         
# >                             STRESAGR   0.000000000 0.02954658   0.0000000  1.00000            0.0000000 1.000000         
# >                               EXPOFA  -0.036630553 0.13114548  -0.2793124  0.78001           -0.2793124 0.780735         
# >                             EXPO21ST  -0.031628319 0.12191817  -0.2594225  0.79531           -0.2594225 0.795984         
# >                             COGACRCO   0.000000000 0.01774209   0.0000000  1.00000            0.0000000 1.000000         
# >                             COGACMCO  -2.685608313 0.99211685  -2.7069476  0.00679       **  -2.7069476 0.008319       **
# >                              DISCLIM   0.000000000 0.05385509   0.0000000  1.00000            0.0000000 1.000000         
# >                               FAMSUP  -2.300293849 0.94430545  -2.4359638  0.01485        *  -2.4359638 0.017103        *
# >                             CREATFAM  -0.269655285 0.52442644  -0.5141909  0.60712           -0.5141909 0.608554         
# >                             CREATSCH  -2.497260454 0.99307862  -2.5146654  0.01191        *  -2.5146654 0.013945        *
# >                             CREATEFF  -4.486306198 0.76438413  -5.8691776 4.38e-09      ***  -5.8691776 9.79e-08      ***
# >                              CREATOP   1.388663739 0.61774285   2.2479641  0.02458        *   2.2479641 0.027366        *
# >                              IMAGINE   0.000000000 0.01228416   0.0000000  1.00000            0.0000000 1.000000         
# >                              OPENART  -0.026738175 0.12126404  -0.2204955  0.82549           -0.2204955 0.826054         
# >                              CREATAS   0.000000000 0.05461238   0.0000000  1.00000            0.0000000 1.000000         
# >                             CREATOOS  -3.585222503 0.69858292  -5.1321360 2.86e-07      ***  -5.1321360 2.00e-06      ***
# >                             FAMSUPSL -10.769790400 0.63434554 -16.9777979  < 2e-16      *** -16.9777979  < 2e-16      ***
# >                              FEELLAH  -1.729442475 0.93750878  -1.8447214  0.06508        .  -1.8447214 0.068826        .
# >                             PROBSELF  -0.175651952 0.40139557  -0.4376031  0.66167           -0.4376031 0.662868         
# >                               SDLEFF   0.042692769 0.11027267   0.3871564  0.69864            0.3871564 0.699681         
# >                              SCHSUST   2.293679923 0.55296306   4.1479803 3.35e-05      ***   4.1479803 8.38e-05      ***
# >                              LEARRES   0.000000000 0.02518135   0.0000000  1.00000            0.0000000 1.000000         
# >                                 ESCS  17.819261934 1.51970049  11.7255091  < 2e-16      ***  11.7255091  < 2e-16      ***
# >                               MACTIV   1.743087719 0.35159164   4.9577053 7.13e-07      ***   4.9577053 3.99e-06      ***
# >                              ABGMATH  -0.194923902 0.60116921  -0.3242413  0.74576           -0.3242413 0.746612         
# >                              MTTRAIN   0.000000000 0.06162747   0.0000000  1.00000            0.0000000 1.000000         
# >                             CREENVSC   0.000000000 0.08384552   0.0000000  1.00000            0.0000000 1.000000         
# >                              OPENCUL   3.293013689 0.68914239   4.7784228 1.77e-06      ***   4.7784228 8.02e-06      ***
# >                              DIGPREP   0.219142281 0.37502365   0.5843426  0.55899            0.5843426 0.560655         
# >   REGIONCanada: Prince Edward Island   0.000000000 0.02371217   0.0000000  1.00000            0.0000000 1.000000         
# >            REGIONCanada: Nova Scotia  -2.578211838 2.70925269  -0.9516321  0.34128           -0.9516321 0.344186         
# >          REGIONCanada: New Brunswick  -8.507557702 2.78799121  -3.0515009  0.00228       **  -3.0515009 0.003100       **
# >                 REGIONCanada: Quebec   0.940398544 1.80566990   0.5208031  0.60250            0.5208031 0.603961         
# >                REGIONCanada: Ontario   4.124223146 1.86668208   2.2093870  0.02715        *   2.2093870 0.030044        *
# >               REGIONCanada: Manitoba  -8.641255056 3.40153374  -2.5403996  0.01107        *  -2.5403996 0.013033        *
# >           REGIONCanada: Saskatchewan  -4.594424639 3.68871366  -1.2455357  0.21293           -1.2455357 0.216616         
# >                REGIONCanada: Alberta   0.000000000 0.06688082   0.0000000  1.00000            0.0000000 1.000000         
# >       REGIONCanada: British Columbia   0.000000000 0.19652585   0.0000000  1.00000            0.0000000 1.000000         
# >                        ST004D01TMale  10.981230544 0.83481387  13.1541065  < 2e-16      ***  13.1541065  < 2e-16      ***
# >              ST004D01TNot Applicable   0.000000000 0.00000000         NaN       NA                  NaN       NA         
# >       IMMIGSecond-Generation student   1.755420932 1.17146262   1.4984865  0.13401            1.4984865 0.137992         
# >        IMMIGFirst-Generation student  -1.542587064 1.69124891  -0.9120994  0.36172           -0.9120994 0.364491         
# >                          LANGNFrench  11.219757853 2.33642081   4.8021135 1.57e-06      ***   4.8021135 7.32e-06      ***
# >          LANGNAnother language (CAN)   3.797093584 2.37036628   1.6019016  0.10918            1.6019016 0.113169         
# > SCHLTYPEPrivate Government-dependent   1.632771348 1.74439849   0.9360082  0.34927            0.9360082 0.352121         
# >                       SCHLTYPEPublic  -6.496107847 3.70850905  -1.7516764  0.07983        .  -1.7516764 0.083710        .

# --- Sorted table by |estimate|---
coef_table %>%
  mutate(abs_est = abs(Estimate)) %>%
  arrange(desc(abs_est)) %>%
  select(-abs_est) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >                                 Term      Estimate Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# >                          (Intercept) 512.971907494 4.30842982 119.0623797  < 2e-16      *** 119.0623797  < 2e-16      ***
# >                              MATHEFF  26.471752386 0.76467229  34.6184279  < 2e-16      ***  34.6184279  < 2e-16      ***
# >                                 ESCS  17.819261934 1.51970049  11.7255091  < 2e-16      ***  11.7255091  < 2e-16      ***
# >                          LANGNFrench  11.219757853 2.33642081   4.8021135 1.57e-06      ***   4.8021135 7.32e-06      ***
# >                        ST004D01TMale  10.981230544 0.83481387  13.1541065  < 2e-16      ***  13.1541065  < 2e-16      ***
# >                             FAMSUPSL -10.769790400 0.63434554 -16.9777979  < 2e-16      *** -16.9777979  < 2e-16      ***
# >                              HOMEPOS   9.446988043 1.51240177   6.2463482 4.20e-10      ***   6.2463482 1.98e-08      ***
# >               REGIONCanada: Manitoba  -8.641255056 3.40153374  -2.5403996  0.01107        *  -2.5403996 0.013033        *
# >          REGIONCanada: New Brunswick  -8.507557702 2.78799121  -3.0515009  0.00228       **  -3.0515009 0.003100       **
# >                               ANXMAT  -6.935507467 0.77396205  -8.9610433  < 2e-16      ***  -8.9610433 1.16e-13      ***
# >                       SCHLTYPEPublic  -6.496107847 3.70850905  -1.7516764  0.07983        .  -1.7516764 0.083710        .
# >                               FAMCON   6.397865306 0.50556074  12.6549884  < 2e-16      ***  12.6549884  < 2e-16      ***
# >                             CURIOAGR   5.384672517 0.89574332   6.0114012 1.84e-09      ***   6.0114012 5.38e-08      ***
# >                               ICTRES  -4.817777480 0.91564635  -5.2616138 1.43e-07      ***  -5.2616138 1.19e-06      ***
# >           REGIONCanada: Saskatchewan  -4.594424639 3.68871366  -1.2455357  0.21293           -1.2455357 0.216616         
# >                             CREATEFF  -4.486306198 0.76438413  -5.8691776 4.38e-09      ***  -5.8691776 9.79e-08      ***
# >                REGIONCanada: Ontario   4.124223146 1.86668208   2.2093870  0.02715        *   2.2093870 0.030044        *
# >          LANGNAnother language (CAN)   3.797093584 2.37036628   1.6019016  0.10918            1.6019016 0.113169         
# >                              GROSAGR   3.647737933 0.52063492   7.0063259 2.45e-12      ***   7.0063259 7.27e-10      ***
# >                             CREATOOS  -3.585222503 0.69858292  -5.1321360 2.86e-07      ***  -5.1321360 2.00e-06      ***
# >                              WORKPAY  -3.553103473 0.26445746 -13.4354443  < 2e-16      *** -13.4354443  < 2e-16      ***
# >                              OPENCUL   3.293013689 0.68914239   4.7784228 1.77e-06      ***   4.7784228 8.02e-06      ***
# >                              BULLIED  -3.061453682 0.76736691  -3.9895565 6.62e-05      ***  -3.9895565 0.000147      ***
# >                             EXERPRAC  -2.771680923 0.18069192 -15.3392630  < 2e-16      *** -15.3392630  < 2e-16      ***
# >                             COGACMCO  -2.685608313 0.99211685  -2.7069476  0.00679       **  -2.7069476 0.008319       **
# >            REGIONCanada: Nova Scotia  -2.578211838 2.70925269  -0.9516321  0.34128           -0.9516321 0.344186         
# >                             CREATSCH  -2.497260454 0.99307862  -2.5146654  0.01191        *  -2.5146654 0.013945        *
# >                             MATHEASE   2.454009655 1.13890681   2.1547063  0.03118        *   2.1547063 0.034231        *
# >                               FAMSUP  -2.300293849 0.94430545  -2.4359638  0.01485        *  -2.4359638 0.017103        *
# >                              SCHSUST   2.293679923 0.55296306   4.1479803 3.35e-05      ***   4.1479803 8.38e-05      ***
# >       IMMIGSecond-Generation student   1.755420932 1.17146262   1.4984865  0.13401            1.4984865 0.137992         
# >                               MACTIV   1.743087719 0.35159164   4.9577053 7.13e-07      ***   4.9577053 3.99e-06      ***
# >                              FEELLAH  -1.729442475 0.93750878  -1.8447214  0.06508        .  -1.8447214 0.068826        .
# >                             EMOCOAGR   1.720109571 0.57302238   3.0018192  0.00268       **   3.0018192 0.003591       **
# >                             MATHEF21   1.668069764 1.16437222   1.4325915  0.15197            1.4325915 0.155920         
# > SCHLTYPEPrivate Government-dependent   1.632771348 1.74439849   0.9360082  0.34927            0.9360082 0.352121         
# >        IMMIGFirst-Generation student  -1.542587064 1.69124891  -0.9120994  0.36172           -0.9120994 0.364491         
# >                              CREATOP   1.388663739 0.61774285   2.2479641  0.02458        *   2.2479641 0.027366        *
# >                             ASSERAGR   1.326176524 0.61867769   2.1435661  0.03207        *   2.1435661 0.035144        *
# >                             MATHPERS   1.297621343 0.92394580   1.4044345  0.16019            1.4044345 0.164109         
# >                               BELONG  -1.186895211 1.02563670  -1.1572277  0.24718           -1.1572277 0.250666         
# >                 REGIONCanada: Quebec   0.940398544 1.80566990   0.5208031  0.60250            0.5208031 0.603961         
# >                             FEELSAFE   0.600216115 0.63191506   0.9498367  0.34220            0.9498367 0.345092         
# >                             WORKHOME  -0.446921987 0.18821646  -2.3745107  0.01757        *  -2.3745107 0.019999        *
# >                             MATHPREF   0.374741749 1.07295291   0.3492621  0.72689            0.3492621 0.727822         
# >                             CREATFAM  -0.269655285 0.52442644  -0.5141909  0.60712           -0.5141909 0.608554         
# >                             STUDYHMW  -0.223699531 0.23952010  -0.9339489  0.35033           -0.9339489 0.353176         
# >                              DIGPREP   0.219142281 0.37502365   0.5843426  0.55899            0.5843426 0.560655         
# >                              ABGMATH  -0.194923902 0.60116921  -0.3242413  0.74576           -0.3242413 0.746612         
# >                             PROBSELF  -0.175651952 0.40139557  -0.4376031  0.66167           -0.4376031 0.662868         
# >                             EMPATAGR  -0.047346626 0.15539818  -0.3046794  0.76061           -0.3046794 0.761412         
# >                               SDLEFF   0.042692769 0.11027267   0.3871564  0.69864            0.3871564 0.699681         
# >                               EXPOFA  -0.036630553 0.13114548  -0.2793124  0.78001           -0.2793124 0.780735         
# >                             EXPO21ST  -0.031628319 0.12191817  -0.2594225  0.79531           -0.2594225 0.795984         
# >                              OPENART  -0.026738175 0.12126404  -0.2204955  0.82549           -0.2204955 0.826054         
# >                            PERSEVAGR   0.017115253 0.07877258   0.2172742  0.82799            0.2172742 0.828555         
# >                              COOPAGR  -0.009848235 0.06866508  -0.1434242  0.88596           -0.1434242 0.886320         
# >                              MATHMOT   0.000000000 0.18696160   0.0000000  1.00000            0.0000000 1.000000         
# >                             INFOSEEK   0.000000000 0.04364172   0.0000000  1.00000            0.0000000 1.000000         
# >                             STRESAGR   0.000000000 0.02954658   0.0000000  1.00000            0.0000000 1.000000         
# >                             COGACRCO   0.000000000 0.01774209   0.0000000  1.00000            0.0000000 1.000000         
# >                              DISCLIM   0.000000000 0.05385509   0.0000000  1.00000            0.0000000 1.000000         
# >                              IMAGINE   0.000000000 0.01228416   0.0000000  1.00000            0.0000000 1.000000         
# >                              CREATAS   0.000000000 0.05461238   0.0000000  1.00000            0.0000000 1.000000         
# >                              LEARRES   0.000000000 0.02518135   0.0000000  1.00000            0.0000000 1.000000         
# >                              MTTRAIN   0.000000000 0.06162747   0.0000000  1.00000            0.0000000 1.000000         
# >                             CREENVSC   0.000000000 0.08384552   0.0000000  1.00000            0.0000000 1.000000         
# >   REGIONCanada: Prince Edward Island   0.000000000 0.02371217   0.0000000  1.00000            0.0000000 1.000000         
# >                REGIONCanada: Alberta   0.000000000 0.06688082   0.0000000  1.00000            0.0000000 1.000000         
# >       REGIONCanada: British Columbia   0.000000000 0.19652585   0.0000000  1.00000            0.0000000 1.000000         
# >              ST004D01TNot Applicable   0.000000000 0.00000000         NaN       NA                  NaN       NA   

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

# R-squared (TRAIN) SE via BRR + Rubin 
sampling_var_r2s_weighted <- sapply(1:M, function(m) {
  sum((rep_r2_weighted[, m] - main_r2s_weighted[m])^2) / (G * (1 - k)^2)
})
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
se_final_r2_weighted  <- sqrt(var_final_r2_weighted)
r2_weighted_table <- tibble::tibble(
  Metric       = "R-squared (Weighted, Train)",
  Estimate     = main_r2_weighted,
  `Std. Error` = se_final_r2_weighted
)
print(as.data.frame(r2_weighted_table), row.names = FALSE)
# >                      Metric    Estimate  Std. Error
# >  R-squared (Weighted, Train)  0.4547629 0.007679188

#### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
train_metrics_main

# Combine across plausible values
train_metric_main <- colMeans(train_metrics_main)

# --- Replicate predictions for training data ---
tic("Computing train_metrics_replicates (glmnet)")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- train_data[[pvmaths[m]]]
    w <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda_1se))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing train_metrics_replicates (glmnet): 1.319 sec elapsed
class(train_metrics_replicates[[1]]); dim(train_metrics_replicates[[1]])
# > [1] "matrix" "array" 
# > [1] 80  6

# BRR sampling variance on TRAIN (vectorized across metrics)
sampling_var_matrix_train <- sapply(1:M, function(m) {
  sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
    colSums() / (G * (1 - k)^2)
})
sampling_var_train <- rowMeans(sampling_var_matrix_train)

# Rubin imputation variance on TRAIN
imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)

# Total variance and SE
var_final_train <- sampling_var_train + (1 + 1/M) * imputation_var_train
se_final_train  <- sqrt(var_final_train)

# Confidence intervals
ci_lower_train  <- train_metric_main - z_crit * se_final_train
ci_upper_train  <- train_metric_main + z_crit * se_final_train
ci_length_train <- ci_upper_train - ci_lower_train

# Format results
train_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train,   scientific = FALSE),
  CI_lower       = format(ci_lower_train,   scientific = FALSE),
  CI_upper       = format(ci_upper_train,   scientific = FALSE),
  CI_length      = format(ci_length_train,  scientific = FALSE)
)
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                 CI_lower                CI_upper              CI_length
# >       MSE 4290.1024480906589815277 81.6647637082046742307 4130.0424524166046467144 4450.162443764713316341 320.119991348108669627
# >      RMSE   65.4965552301389379863  0.6192486330295776664   64.2828502119253073488   66.710260248352568624   2.427410036427261275
# >       MAE   51.9063855763418899869  0.4433717921459921496   51.0373928319747633964   52.775378320709016577   1.737985488734253181
# >      Bias    0.0000000000001913704  0.0000000000004477077   -0.0000000000006861205    0.000000000001068861   0.000000000001754982
# >     Bias%    1.8170243635841469843  0.0332506528520287795    1.7518542815317266204    1.882194445636567348   0.130340164104840728
# > R-squared    0.4547628842165439278  0.0076791876241337790    0.4397119530427160417    0.469813815390371814   0.030101862347655772

#### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
valid_metric_main <- colMeans(valid_metrics_main)

# Replicate predictions on validation set
tic("Computing valid_metrics_replicates (glmnet)")
valid_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- valid_data[[pvmaths[m]]]
    w <- valid_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda_1se))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 0.988 sec elapsed

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
valid_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(valid_metric_main, scientific = FALSE),
  Standard_error = format(se_final_valid,   scientific = FALSE),
  CI_lower       = format(ci_lower_valid,   scientific = FALSE),
  CI_upper       = format(ci_upper_valid,   scientific = FALSE),
  CI_length      = format(ci_length_valid,  scientific = FALSE)
)
print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower    CI_upper    CI_length
# >       MSE   4789.3420941    157.3199939 4481.000572 5097.6836161 616.6830441
# >      RMSE     69.1980609      1.1354399   66.972640   71.4234822   4.4508426
# >       MAE     54.4642247      1.1959626   52.120181   56.8082684   4.6880873
# >      Bias      1.8485896      1.9388487   -1.951484    5.6486632   7.6001472
# >     Bias%      2.4302760      0.4079395    1.630729    3.2298228   1.5990935
# > R-squared      0.4421204      0.0124739    0.417672    0.4665688   0.0488968

#### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_1se))
  compute_metrics(y_train, y_pred, w)
}) |> t() |> as.data.frame()
test_metric_main <- colMeans(test_metrics_main)

# Replicate predictions on test set
tic("Computing test_metrics_replicates (glmnet)")
test_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y_train <- test_data[[pvmaths[m]]]
    w <- test_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda_1se))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates (glmnet): 1.211 sec elapsed

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
test_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(test_metric_main, scientific = FALSE),
  Standard_error = format(se_final_test,   scientific = FALSE),
  CI_lower       = format(ci_lower_test,   scientific = FALSE),
  CI_upper       = format(ci_upper_test,   scientific = FALSE),
  CI_length      = format(ci_length_test,  scientific = FALSE)
)
print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   4908.9229270   284.52459948 4351.2649593 5466.5808947 1115.31593539
# >      RMSE     70.0402936     2.03316151   66.0553702   74.0252169    7.96984666
# >       MAE     55.6306792     1.36820261   52.9490514   58.3123071    5.36325569
# >      Bias     -2.6129670     1.62389825   -5.7957490    0.5698151    6.36556418
# >     Bias%      1.4850816     0.43545463    0.6316062    2.3385570    1.70695078
# > R-squared      0.3978922     0.01682968    0.3649067    0.4308778    0.06597112

#### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.

# Helper
evaluate_split <- function(X, split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           pvmaths, s_value) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model  <- main_models[[i]]$mod
    y      <- split_data[[pvmaths[i]]]
    w      <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X, s = s_value))
    compute_metrics(y_true = y, y_pred = y_pred, w = w)
  }) |> t() |> as.data.frame()
  main_point <- colMeans(main_metrics_df)
  
  # Replicate metrics across PVs
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model  <- replicate_models[[m]][[g]]$mod
      y      <- split_data[[pvmaths[m]]]
      w      <- split_data[[rep_wts[g]]]
      y_pred <- as.numeric(predict(model, newx = X, s = s_value))
      compute_metrics(y_true = y, y_pred = y_pred, w = w)
    }) |> t()
  })
  
  # BRR sampling var (average across PVs)
  sampling_var_matrix <- sapply(1:M, function(m) {
    sweep(replicate_metrics[[m]], 2, unlist(main_metrics_df[m, ]))^2 |>
      colSums() / (G * (1 - k)^2)
  })
  sampling_var <- rowMeans(sampling_var_matrix)
  
  # Rubin imputation var across PVs
  imputation_var <- colSums((main_metrics_df - matrix(main_point, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
  
  # Totals + CIs
  var_final <- sampling_var + (1 + 1/M) * imputation_var
  se_final  <- sqrt(var_final)
  ci_lower  <- main_point - z_crit * se_final
  ci_upper  <- main_point + z_crit * se_final
  ci_length <- ci_upper - ci_lower
  
  tibble::tibble(
    Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
    Point_estimate = format(main_point, scientific = FALSE),
    Standard_error = format(se_final,  scientific = FALSE),
    CI_lower       = format(ci_lower,  scientific = FALSE),
    CI_upper       = format(ci_upper,  scientific = FALSE),
    CI_length      = format(ci_length, scientific = FALSE)
  )
}

# Evaluate on each split
train_eval <- evaluate_split(X_train, train_data, main_models, replicate_models,
                             final_wt, rep_wts, M, G, k, z_crit, pvmaths, best_lambda_1se)
valid_eval <- evaluate_split(X_valid, valid_data, main_models, replicate_models,
                             final_wt, rep_wts, M, G, k, z_crit, pvmaths, best_lambda_1se)
test_eval  <- evaluate_split(X_test,  test_data,  main_models, replicate_models,
                             final_wt, rep_wts, M, G, k, z_crit, pvmaths, best_lambda_1se)

print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

#### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                 CI_lower                CI_upper              CI_length
# >       MSE 4290.1024480906589815277 81.6647637082046742307 4130.0424524166046467144 4450.162443764713316341 320.119991348108669627
# >      RMSE   65.4965552301389379863  0.6192486330295776664   64.2828502119253073488   66.710260248352568624   2.427410036427261275
# >       MAE   51.9063855763418899869  0.4433717921459921496   51.0373928319747633964   52.775378320709016577   1.737985488734253181
# >      Bias    0.0000000000001913704  0.0000000000004477077   -0.0000000000006861205    0.000000000001068861   0.000000000001754982
# >     Bias%    1.8170243635841469843  0.0332506528520287795    1.7518542815317266204    1.882194445636567348   0.130340164104840728
# > R-squared    0.4547628842165439278  0.0076791876241337790    0.4397119530427160417    0.469813815390371814   0.030101862347655772

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower    CI_upper    CI_length
# >       MSE   4789.3420941    157.3199939 4481.000572 5097.6836161 616.6830441
# >      RMSE     69.1980609      1.1354399   66.972640   71.4234822   4.4508426
# >       MAE     54.4642247      1.1959626   52.120181   56.8082684   4.6880873
# >      Bias      1.8485896      1.9388487   -1.951484    5.6486632   7.6001472
# >     Bias%      2.4302760      0.4079395    1.630729    3.2298228   1.5990935
# > R-squared      0.4421204      0.0124739    0.417672    0.4665688   0.0488968

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper     CI_length
# >       MSE   4908.9229270   284.52459948 4351.2649593 5466.5808947 1115.31593539
# >      RMSE     70.0402936     2.03316151   66.0553702   74.0252169    7.96984666
# >       MAE     55.6306792     1.36820261   52.9490514   58.3123071    5.36325569
# >      Bias     -2.6129670     1.62389825   -5.7957490    0.5698151    6.36556418
# >     Bias%      1.4850816     0.43545463    0.6316062    2.3385570    1.70695078
# > R-squared      0.3978922     0.01682968    0.3649067    0.4308778    0.06597112

# --- Keep only four decimals ---
# library(dplyr)
# library(readr)   # for parse_number

train_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        65.4966         0.6192  64.2829  66.7103    2.4274
# >       MAE        51.9064         0.4434  51.0374  52.7754    1.7380
# >      Bias         0.0000         0.0000  -0.0000   0.0000    0.0000
# >     Bias%         1.8170         0.0333   1.7519   1.8822    0.1303
# > R-squared         0.4548         0.0077   0.4397   0.4698    0.0301

valid_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        69.1981         1.1354  66.9726  71.4235    4.4508
# >       MAE        54.4642         1.1960  52.1202  56.8083    4.6881
# >      Bias         1.8486         1.9388  -1.9515   5.6487    7.6001
# >     Bias%         2.4303         0.4079   1.6307   3.2298    1.5991
# > R-squared         0.4421         0.0125   0.4177   0.4666    0.0489

test_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        70.0403         2.0332  66.0554  74.0252    7.9698
# >       MAE        55.6307         1.3682  52.9491  58.3123    5.3633
# >      Bias        -2.6130         1.6239  -5.7957   0.5698    6.3656
# >     Bias%         1.4851         0.4355   0.6316   2.3386    1.7070
# > R-squared         0.3979         0.0168   0.3649   0.4309    0.0660

# --- Keep only two decimals ---
train_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          65.50           0.62    64.28    66.71      2.43
# >       MAE          51.91           0.44    51.04    52.78      1.74
# >      Bias           0.00           0.00    -0.00     0.00      0.00
# >     Bias%           1.82           0.03     1.75     1.88      0.13
# > R-squared           0.45           0.01     0.44     0.47      0.03

valid_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          69.20           1.14    66.97    71.42      4.45
# >       MAE          54.46           1.20    52.12    56.81      4.69
# >      Bias           1.85           1.94    -1.95     5.65      7.60
# >     Bias%           2.43           0.41     1.63     3.23      1.60
# > R-squared           0.44           0.01     0.42     0.47      0.05

test_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          70.04           2.03    66.06    74.03      7.97
# >       MAE          55.63           1.37    52.95    58.31      5.36
# >      Bias          -2.61           1.62    -5.80     0.57      6.37
# >     Bias%           1.49           0.44     0.63     2.34      1.71
# > R-squared           0.40           0.02     0.36     0.43      0.07

# --- Export to word---
library(officer); library(flextable)

fmt2 <- \(df) df %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  )

performance_tables_glmnet_voi_all_v2p4_best1se <- read_docx() %>%
  body_add_par("Performance - Elastic Net (glmnet, voi_all, v2.4, best 1se)", style = "Normal") %>%  
  body_add_par("Training performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(train_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Validation performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(valid_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Test performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(test_eval))  %>% align(j = 1:6, align = "right", part = "all") %>% autofit())

print(performance_tables_glmnet_voi_all_v2p4_best1se, target = "performance_tables_glmnet_voi_all_v2p4_best1se.docx")



