# ---- II. Predictive Modelling: Version 2.1 ----

# Tune hyperparameters (alpha α and lambda λ) using glmnet only

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
# library(Matrix)     # (optional) if use sparse model matrices
options(contrasts = c("contr.treatment", "contr.poly"))

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("tidyverse","glmnet","Matrix","haven","broom","tictoc","caret"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         tidyverse            glmnet            Matrix             haven             broom            tictoc            caret 
# > "tidyverse 2.0.0"   "glmnet 4.1.10"    "Matrix 1.7.3"     "haven 2.5.4"     "broom 1.0.8"    "tictoc 1.2.1"     "caret 7.0.1" 

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

### ---- Fit main model using final student weights (W_FSTUWT) on the training data ---- 

#### ---- Tuning for PV1MATH only: glmnet ----

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

# Grid over elastic‑net mixing parameter
alpha_grid <- seq(0, 1, by = 0.1)   # α ∈ [0, 1]; 0=ridge, 1=lasso
# options: alpha_grid <- seq(0, 1, by = 0.05), alpha_grid <- sort(unique(c(seq(0, 1, by = 0.1), 0.001, 0.005, 0.01, 0.05))), fine tuning etc. 

# Storage
mod_list        <- vector("list", length(alpha_grid))      # glmnet fits per alpha
per_alpha_list  <- vector("list", length(alpha_grid))      # per‑lambda metrics per alpha

##### ---- Tune ----

tic("Grid over alpha (glmnet)")
for (i in seq_along(alpha_grid)) {
  alpha   <- alpha_grid[i]
  message(sprintf("Fitting glmnet path for alpha = %.1f", alpha))
  
  # Fit the whole lambda path at this alpha on TRAIN
  mod <- glmnet(
    x           = X_train, 
    y           = y_train,
    family      = "gaussian",
    weights     = w_train,
    alpha       = alpha,
    standardize = TRUE,
    intercept   = TRUE
  )
  mod_list[[i]] <- mod
  
  # Predictions across the path
  pred_valid_matrix <- predict(mod, newx = X_valid, s = mod$lambda)
  pred_train_matrix <- predict(mod, newx = X_train, s = mod$lambda)
  
  # RMSE curves (weighted)
  rmse_valid_vec <- apply(pred_valid_matrix, 2, function(p) w_rmse(y_valid, p, w_valid))
  rmse_train_vec <- apply(pred_train_matrix, 2, function(p) w_rmse(y_train, p, w_train))
  
  # Collect tidy rows per alpha (one row per lambda on this path)
  per_alpha_list[[i]] <- tibble(
    alpha       = alpha,
    alpha_idx   = i,
    lambda      = as.numeric(mod$lambda),
    lambda_idx  = seq_along(mod$lambda),
    df          = mod$df,
    dev_ratio   = as.numeric(mod$dev.ratio),
    rmse_valid  = as.numeric(rmse_valid_vec),
    rmse_train  = as.numeric(rmse_train_vec)
  )
}
toc()
# > Grid over alpha (glmnet): 1.51 sec elapsed

##### ---- Explore tuning results ----

# Stack all alpha–lambda candidates
tuning_results <- bind_rows(per_alpha_list)

# Top candidates by Validation RMSE
tuning_results %>%
  arrange(rmse_valid) %>%
  head(10) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

# Choose the model with the lowest weighted Validation RMSE (proxy for out‑of‑sample error).
# Tie‑breaks favor simpler and stabler fits:
# 1) fewer non‑zero coefficients (smaller df) → sparser, more interpretable, lower variance;
# 2) if df ties, pick the larger λ → same support but stronger shrinkage (more conservative;
#    akin to a “1‑SE” preference to guard against overfitting);
# 3) if still tied, pick the smaller α → more L2 component, which is numerically stabler
#    under multicollinearity and survey reweighting (grouping effect).
best_row <- tuning_results %>%
  arrange(rmse_valid, df, desc(lambda), alpha) %>%
  slice(1)
print(as.data.frame(best_row), row.names = FALSE)
# > alpha alpha_idx    lambda lambda_idx df dev_ratio rmse_valid rmse_train
# >     1        11 0.1811031         61 64 0.4560385   68.67159   66.23055

best_alpha      <- best_row$alpha
best_lambda     <- best_row$lambda
best_alpha_idx  <- best_row$alpha_idx
best_df         <- best_row$df
best_rmse_valid <- best_row$rmse_valid

message(sprintf(
  "Selected: alpha = %.2f | lambda = %.6f | df = %d | Valid RMSE = %.5f",
  best_alpha, best_lambda, best_df, best_rmse_valid
))
# > Selected: alpha = 1.00 | lambda = 0.181103 | df = 64 | Valid RMSE = 68.67159

#### ---- Predict and evaluate performance on training/validation/test datasets ----
best_mod <- mod_list[[best_alpha_idx]]

coef(best_mod, s = best_lambda) %>% head()
# > 6 x 1 sparse Matrix of class "dgCMatrix"
# >             s=0.1811031
# > (Intercept) 502.2965429
# > MATHMOT       2.3786823
# > MATHEASE      3.9701596
# > MATHPREF     -0.8553248
# > EXERPRAC     -2.8208663
# > STUDYHMW     -0.6696485
varImp(best_mod, lambda = best_lambda) %>% head()
# >            Overall
# > MATHMOT  2.3786823
# > MATHEASE 3.9701596
# > MATHPREF 0.8553248
# > EXERPRAC 2.8208663
# > STUDYHMW 0.6696485
# > WORKPAY  3.2555537

pred_train_best <- as.numeric(predict(best_mod, newx = X_train, s = best_lambda))
pred_valid_best <- as.numeric(predict(best_mod, newx = X_valid, s = best_lambda))
pred_test_best  <- as.numeric(predict(best_mod,  newx = X_test,  s = best_lambda))

metrics_train_best <- compute_metrics(y_true = y_train, y_pred = pred_train_best, w = w_train)
metrics_valid_best <- compute_metrics(y_true = y_valid, y_pred = pred_valid_best, w = w_valid)
metrics_test_best  <- compute_metrics(y_true = y_test,  y_pred = pred_test_best,  w = w_test)

metric_results <- tibble(
  Dataset = c("Training", "Validation", "Test"),
  RMSE    = c(metrics_train_best["rmse"], metrics_valid_best["rmse"], metrics_test_best["rmse"]),
  MAE     = c(metrics_train_best["mae"],  metrics_valid_best["mae"],  metrics_test_best["mae"]),
  Bias    = c(metrics_train_best["bias"], metrics_valid_best["bias"], metrics_test_best["bias"]),
  `Bias%` = c(metrics_train_best["bias_pct"], metrics_valid_best["bias_pct"], metrics_test_best["bias_pct"]),
  R2      = c(metrics_train_best["r2"],   metrics_valid_best["r2"],   metrics_test_best["r2"])
)

print(as.data.frame(metric_results), row.names = FALSE)
# >    Dataset     RMSE      MAE          Bias     Bias%        R2
# >   Training 66.23055 52.28508  2.642306e-13 1.7753372 0.4560385
# > Validation 68.67159 53.56136  2.502716e+00 2.4890062 0.4486414
# >       Test 68.40452 54.88196 -4.731734e+00 0.8132671 0.4030128

## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH (best_alpha, best_lambda) to all plausible values in mathematics.

### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

tic("Fitting main glmnet models (fixed best_alpha, best_lambda)")
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
    alpha = best_alpha,     # best_alpha = 1   
    lambda = best_lambda,   # best_lambda = 0.1811031
    standardize = TRUE,
    intercept = TRUE
  )
  
  # Extract coefficients like linear regression (including intercept)
  coefs_matrix <- as.matrix(coef(mod, s = best_lambda))
  coefs  <- coefs_matrix[, 1]
  names(coefs) <- rownames(coefs_matrix)
  
  list(
    formula = as.formula(paste(pv, "~", paste(voi_all, collapse = " + "))),  # include cats in formula label
    mod     = mod,
    coefs   = coefs
  )
})
toc()
# > Fitting main glmnet models (fixed best_alpha, best_lambda): 1.063 sec elapsed

# Quick look
main_models[[1]]
main_models[[1]]$formula
main_models[[1]]$coefs[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 502.2797312   2.3785168   3.9723710  -0.8568416  -2.8211464  -0.6693094 

# --- Aggregate coefficients across PVs (Rubin Step 2: θ̂) ---
main_coefs <- do.call(rbind, lapply(main_models, function(m) m$coefs))  # M x (p+1)
main_coef  <- colMeans(main_coefs)                                      # pooled coefficients
main_coef
main_coef[1:6]
# > (Intercept)     MATHMOT    MATHEASE    MATHPREF    EXERPRAC    STUDYHMW 
# > 507.7448077   1.4012341   4.3656729   0.6805284  -3.0333387  -0.7687970

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
# >   Term                        Estimate `|Estimate|`
# >   <chr>                          <dbl>        <dbl>
# > 1 MATHEFF                         25.1         25.1
# > 2 LANGNFrench                     17.5         17.5
# > 3 ESCS                            16.9         16.9
# > 4 HOMEPOS                         16.5         16.5
# > 5 ST004D01TMale                   13.0         13.0
# > 6 LANGNAnother language (CAN)     11.4         11.4
# > 7 FAMSUPSL                       -11.4         11.4
# > 8 ICTRES                         -10.6         10.6
# > 9 REGIONCanada: Ontario           10.4         10.4
# > 10 REGIONCanada: New Brunswick   -10.3         10.3

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
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
  compute_metrics(y_train, y_pred, w)["r2"]
}) |> as.numeric()
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted

### ---- Replicate models using BRR replicate weights ----

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
      alpha = best_alpha,
      lambda = best_lambda,
      standardize = TRUE,
      intercept = TRUE
    )
    
    coefs_matrix <- as.matrix(coef(mod, s = best_lambda))
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
# > Fitting replicate glmnet models (BRR weights): 5.979 sec elapsed

# Example inspect
replicate_models[[1]][[1]]
replicate_models[[1]][[1]]$formula
replicate_models[[1]][[1]]$coefs

# --- Replicate weighted R² on TRAIN (G x M) (optional diagnostic) ---
rep_r2_weighted <- matrix(NA_real_, nrow = G, ncol = M)
for (m in 1:M) {
  y_train <- train_data[[pvmaths[m]]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
    rep_r2_weighted[g, m] <- compute_metrics(y_train, y_pred, w)["r2"]
  }
}
dim(rep_r2_weighted)
# > [1] 80 10

### ---- Rubin + BRR for Standard Errors (SEs): Coefficients (Intercept + predictors) ----

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
dof_t <- G-1
`t value`  <- `z value`                         # same ratio; distribution differs
p_t        <- 2 * pt(-abs(`t value`), df = dof_t)
`Pr(>|t|)` <- format.pval(p_t, digits = 3, eps = .Machine$double.eps)
t_Signif   <- symnum(p_t, corr = FALSE, na = FALSE,
                     cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                     symbols   = c("***","**","*",".",""))

#### ---- Final Outputs ----
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
# >                                   Term     Estimate Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# >                            (Intercept) 507.74480768 12.9644230 39.16447396  < 2e-16      *** 39.16447396  < 2e-16      ***
# >                                MATHMOT   1.40123406  5.3828261  0.26031569 0.794620           0.26031569 0.795298         
# >                               MATHEASE   4.36567292  4.5505933  0.95936346 0.337376           0.95936346 0.340303         
# >                               MATHPREF   0.68052837  4.1251202  0.16497176 0.868966           0.16497176 0.869388         
# >                               EXERPRAC  -3.03333868  0.4408246 -6.88105505 5.94e-12      *** -6.88105505 1.26e-09      ***
# >                               STUDYHMW  -0.76879699  0.5391769 -1.42587152 0.153905          -1.42587152 0.157845         
# >                                WORKPAY  -3.58213269  0.5816185 -6.15890400 7.33e-10      *** -6.15890400 2.87e-08      ***
# >                               WORKHOME  -0.58192739  0.4176571 -1.39331388 0.163525          -1.39331388 0.167433         
# >                                HOMEPOS  16.52780483  3.2539285  5.07933866 3.79e-07      ***  5.07933866 2.47e-06      ***
# >                                 ICTRES -10.57359603  2.3238972 -4.54994137 5.37e-06      *** -4.54994137 1.91e-05      ***
# >                               INFOSEEK   0.16684591  1.4931634  0.11173988 0.911030           0.11173988 0.911313         
# >                                BULLIED  -4.64014508  1.5256045 -3.04151250 0.002354       ** -3.04151250 0.003194       **
# >                               FEELSAFE   1.88356627  1.6538149  1.13892209 0.254736           1.13892209 0.258178         
# >                                 BELONG  -4.20761112  1.9687433 -2.13720655 0.032581        * -2.13720655 0.035674        *
# >                                GROSAGR   4.40179261  1.4113414  3.11887162 0.001815       **  3.11887162 0.002534       **
# >                                 ANXMAT  -6.94980368  1.2700710 -5.47198064 4.45e-08      *** -5.47198064 5.08e-07      ***
# >                                MATHEFF  25.08499426  1.8631018 13.46410291  < 2e-16      *** 13.46410291  < 2e-16      ***
# >                               MATHEF21   2.58627843  2.3663474  1.09294113 0.274420           1.09294113 0.277741         
# >                               MATHPERS   3.13131458  1.6098541  1.94509221 0.051764        .  1.94509221 0.055322        .
# >                                 FAMCON   6.77984967  1.1469869  5.91100866 3.40e-09      ***  5.91100866 8.21e-08      ***
# >                               ASSERAGR   3.19102528  1.4249858  2.23933826 0.025134        *  2.23933826 0.027946        *
# >                                COOPAGR  -0.83382075  1.3679743 -0.60952957 0.542173          -0.60952957 0.543923         
# >                               CURIOAGR   6.13950388  1.5779829  3.89072912 9.99e-05      ***  3.89072912 0.000207      ***
# >                               EMOCOAGR   3.33311121  1.5880314  2.09889506 0.035826        *  2.09889506 0.039020        *
# >                               EMPATAGR  -0.58558562  1.4137350 -0.41421173 0.678719          -0.41421173 0.679842         
# >                              PERSEVAGR   0.10641789  1.2289513  0.08659244 0.930995           0.08659244 0.931215         
# >                               STRESAGR  -1.83519540  1.6312898 -1.12499656 0.260590          -1.12499656 0.263997         
# >                                 EXPOFA  -0.27451991  1.4974954 -0.18331937 0.854547          -0.18331937 0.855017         
# >                               EXPO21ST  -0.06887309  1.6448418 -0.04187217 0.966601          -0.04187217 0.966706         
# >                               COGACRCO   0.54745511  1.3905195  0.39370546 0.693799           0.39370546 0.694859         
# >                               COGACMCO  -4.24955180  1.7076497 -2.48853840 0.012827        * -2.48853840 0.014930        *
# >                                DISCLIM   0.21314982  1.3330768  0.15989313 0.872965           0.15989313 0.873373         
# >                                 FAMSUP  -3.38380790  1.7602520 -1.92234288 0.054563        . -1.92234288 0.058168        .
# >                               CREATFAM  -0.92996689  1.7106166 -0.54364424 0.586686          -0.54364424 0.588217         
# >                               CREATSCH  -3.55211339  1.7891341 -1.98538134 0.047102        * -1.98538134 0.050572        .
# >                               CREATEFF  -6.84190981  1.7766060 -3.85111256 0.000118      *** -3.85111256 0.000238      ***
# >                                CREATOP   4.58671251  1.9808138  2.31556978 0.020582        *  2.31556978 0.023177        *
# >                                IMAGINE  -0.27142916  1.2552163 -0.21624094 0.828800          -0.21624094 0.829357         
# >                                OPENART  -1.21749868  1.5382498 -0.79148306 0.428662          -0.79148306 0.431032         
# >                                CREATAS   0.62225717  1.6046847  0.38777534 0.698182           0.38777534 0.699225         
# >                               CREATOOS  -4.72898231  1.7843563 -2.65024548 0.008043       ** -2.65024548 0.009713       **
# >                               FAMSUPSL -11.40495849  1.5225014 -7.49093485 6.84e-14      *** -7.49093485 8.49e-11      ***
# >                                FEELLAH  -3.59274230  1.6382349 -2.19305683 0.028303        * -2.19305683 0.031245        *
# >                               PROBSELF  -0.55143185  1.3422085 -0.41083919 0.681190          -0.41083919 0.682303         
# >                                 SDLEFF   0.94628375  1.5952156  0.59320118 0.553047           0.59320118 0.554741         
# >                                SCHSUST   4.34214673  1.4972563  2.90006919 0.003731       **  2.90006919 0.004830       **
# >                                LEARRES   0.20298870  1.3838801  0.14668084 0.883384           0.14668084 0.883758         
# >                                   ESCS  16.92755301  2.6397339  6.41259838 1.43e-10      ***  6.41259838 9.68e-09      ***
# >                                 MACTIV   2.19121088  1.1838082  1.85098468 0.064172        .  1.85098468 0.067909        .
# >                                ABGMATH  -0.99989230  2.4703382 -0.40475927 0.685654          -0.40475927 0.686748         
# >                                MTTRAIN   0.20440227  1.4958059  0.13665026 0.891307           0.13665026 0.891655         
# >                               CREENVSC  -1.15586921  1.8524854 -0.62395589 0.532657          -0.62395589 0.534454         
# >                                OPENCUL   4.52028259  1.8477591  2.44635933 0.014431        *  2.44635933 0.016652        *
# >                                DIGPREP   0.78038516  1.5119396  0.51614836 0.605751           0.51614836 0.607193         
# >     REGIONCanada: Prince Edward Island   3.52193991  9.6896547  0.36347424 0.716251           0.36347424 0.717222         
# >              REGIONCanada: Nova Scotia  -3.79161376  6.2491086 -0.60674474 0.544020          -0.60674474 0.545760         
# >            REGIONCanada: New Brunswick -10.25822419  5.1662901 -1.98560745 0.047077        * -1.98560745 0.050547        .
# >                   REGIONCanada: Quebec   4.12116210  5.8186317  0.70826998 0.478778           0.70826998 0.480862         
# >                  REGIONCanada: Ontario  10.40206759  4.7524611  2.18877490 0.028613        *  2.18877490 0.031566        *
# >                 REGIONCanada: Manitoba -10.07789052  5.2806088 -1.90847133 0.056330        . -1.90847133 0.059963        .
# >             REGIONCanada: Saskatchewan  -4.97651003  5.6226698 -0.88507955 0.376114          -0.88507955 0.378800         
# >                  REGIONCanada: Alberta   3.08637377  5.1179664  0.60304690 0.546477           0.60304690 0.548205         
# >         REGIONCanada: British Columbia   4.99359687  5.8395273  0.85513718 0.392475           0.85513718 0.395061         
# >                          ST004D01TMale  12.98995255  3.0855410  4.20994332 2.55e-05      ***  4.20994332 6.71e-05      ***
# >                ST004D01TNot Applicable   0.58976744  6.6771836  0.08832578 0.929618           0.08832578 0.929841         
# >         IMMIGSecond-Generation student   2.06688709  4.0279291  0.51313889 0.607854           0.51313889 0.609286         
# >          IMMIGFirst-Generation student  -5.90063931  4.9272071 -1.19756269 0.231087          -1.19756269 0.234668         
# >                            LANGNFrench  17.49219985  5.7049879  3.06612390 0.002169       **  3.06612390 0.002968       **
# >            LANGNAnother language (CAN)  11.42596306  5.7327236  1.99311249 0.046249        *  1.99311249 0.049702        *
# >   SCHLTYPEPrivate Government-dependent   4.19933887  8.6324086  0.48646201 0.626640           0.48646201 0.627986         
# >                         SCHLTYPEPublic  -7.23315406  9.5527497 -0.75718032 0.448942          -0.75718032 0.451194         

# --- Sorted table by |estimate|---
coef_table %>%
  mutate(abs_est = abs(Estimate)) %>%
  arrange(desc(abs_est)) %>%
  select(-abs_est) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >                                 Term     Estimate Std. Error     z value Pr(>|z|) z_Signif     t value Pr(>|t|) t_Signif
# >                          (Intercept) 507.74480768 12.9644230 39.16447396  < 2e-16      *** 39.16447396  < 2e-16      ***
# >                              MATHEFF  25.08499426  1.8631018 13.46410291  < 2e-16      *** 13.46410291  < 2e-16      ***
# >                          LANGNFrench  17.49219985  5.7049879  3.06612390 0.002169       **  3.06612390 0.002968       **
# >                                 ESCS  16.92755301  2.6397339  6.41259838 1.43e-10      ***  6.41259838 9.68e-09      ***
# >                              HOMEPOS  16.52780483  3.2539285  5.07933866 3.79e-07      ***  5.07933866 2.47e-06      ***
# >                        ST004D01TMale  12.98995255  3.0855410  4.20994332 2.55e-05      ***  4.20994332 6.71e-05      ***
# >          LANGNAnother language (CAN)  11.42596306  5.7327236  1.99311249 0.046249        *  1.99311249 0.049702        *
# >                             FAMSUPSL -11.40495849  1.5225014 -7.49093485 6.84e-14      *** -7.49093485 8.49e-11      ***
# >                               ICTRES -10.57359603  2.3238972 -4.54994137 5.37e-06      *** -4.54994137 1.91e-05      ***
# >                REGIONCanada: Ontario  10.40206759  4.7524611  2.18877490 0.028613        *  2.18877490 0.031566        *
# >          REGIONCanada: New Brunswick -10.25822419  5.1662901 -1.98560745 0.047077        * -1.98560745 0.050547        .
# >               REGIONCanada: Manitoba -10.07789052  5.2806088 -1.90847133 0.056330        . -1.90847133 0.059963        .
# >                       SCHLTYPEPublic  -7.23315406  9.5527497 -0.75718032 0.448942          -0.75718032 0.451194         
# >                               ANXMAT  -6.94980368  1.2700710 -5.47198064 4.45e-08      *** -5.47198064 5.08e-07      ***
# >                             CREATEFF  -6.84190981  1.7766060 -3.85111256 0.000118      *** -3.85111256 0.000238      ***
# >                               FAMCON   6.77984967  1.1469869  5.91100866 3.40e-09      ***  5.91100866 8.21e-08      ***
# >                             CURIOAGR   6.13950388  1.5779829  3.89072912 9.99e-05      ***  3.89072912 0.000207      ***
# >        IMMIGFirst-Generation student  -5.90063931  4.9272071 -1.19756269 0.231087          -1.19756269 0.234668         
# >       REGIONCanada: British Columbia   4.99359687  5.8395273  0.85513718 0.392475           0.85513718 0.395061         
# >           REGIONCanada: Saskatchewan  -4.97651003  5.6226698 -0.88507955 0.376114          -0.88507955 0.378800         
# >                             CREATOOS  -4.72898231  1.7843563 -2.65024548 0.008043       ** -2.65024548 0.009713       **
# >                              BULLIED  -4.64014508  1.5256045 -3.04151250 0.002354       ** -3.04151250 0.003194       **
# >                              CREATOP   4.58671251  1.9808138  2.31556978 0.020582        *  2.31556978 0.023177        *
# >                              OPENCUL   4.52028259  1.8477591  2.44635933 0.014431        *  2.44635933 0.016652        *
# >                              GROSAGR   4.40179261  1.4113414  3.11887162 0.001815       **  3.11887162 0.002534       **
# >                             MATHEASE   4.36567292  4.5505933  0.95936346 0.337376           0.95936346 0.340303         
# >                              SCHSUST   4.34214673  1.4972563  2.90006919 0.003731       **  2.90006919 0.004830       **
# >                             COGACMCO  -4.24955180  1.7076497 -2.48853840 0.012827        * -2.48853840 0.014930        *
# >                               BELONG  -4.20761112  1.9687433 -2.13720655 0.032581        * -2.13720655 0.035674        *
# > SCHLTYPEPrivate Government-dependent   4.19933887  8.6324086  0.48646201 0.626640           0.48646201 0.627986         
# >                 REGIONCanada: Quebec   4.12116210  5.8186317  0.70826998 0.478778           0.70826998 0.480862         
# >            REGIONCanada: Nova Scotia  -3.79161376  6.2491086 -0.60674474 0.544020          -0.60674474 0.545760         
# >                              FEELLAH  -3.59274230  1.6382349 -2.19305683 0.028303        * -2.19305683 0.031245        *
# >                              WORKPAY  -3.58213269  0.5816185 -6.15890400 7.33e-10      *** -6.15890400 2.87e-08      ***
# >                             CREATSCH  -3.55211339  1.7891341 -1.98538134 0.047102        * -1.98538134 0.050572        .
# >   REGIONCanada: Prince Edward Island   3.52193991  9.6896547  0.36347424 0.716251           0.36347424 0.717222         
# >                               FAMSUP  -3.38380790  1.7602520 -1.92234288 0.054563        . -1.92234288 0.058168        .
# >                             EMOCOAGR   3.33311121  1.5880314  2.09889506 0.035826        *  2.09889506 0.039020        *
# >                             ASSERAGR   3.19102528  1.4249858  2.23933826 0.025134        *  2.23933826 0.027946        *
# >                             MATHPERS   3.13131458  1.6098541  1.94509221 0.051764        .  1.94509221 0.055322        .
# >                REGIONCanada: Alberta   3.08637377  5.1179664  0.60304690 0.546477           0.60304690 0.548205         
# >                             EXERPRAC  -3.03333868  0.4408246 -6.88105505 5.94e-12      *** -6.88105505 1.26e-09      ***
# >                             MATHEF21   2.58627843  2.3663474  1.09294113 0.274420           1.09294113 0.277741         
# >                               MACTIV   2.19121088  1.1838082  1.85098468 0.064172        .  1.85098468 0.067909        .
# >       IMMIGSecond-Generation student   2.06688709  4.0279291  0.51313889 0.607854           0.51313889 0.609286         
# >                             FEELSAFE   1.88356627  1.6538149  1.13892209 0.254736           1.13892209 0.258178         
# >                             STRESAGR  -1.83519540  1.6312898 -1.12499656 0.260590          -1.12499656 0.263997         
# >                              MATHMOT   1.40123406  5.3828261  0.26031569 0.794620           0.26031569 0.795298         
# >                              OPENART  -1.21749868  1.5382498 -0.79148306 0.428662          -0.79148306 0.431032         
# >                             CREENVSC  -1.15586921  1.8524854 -0.62395589 0.532657          -0.62395589 0.534454         
# >                              ABGMATH  -0.99989230  2.4703382 -0.40475927 0.685654          -0.40475927 0.686748         
# >                               SDLEFF   0.94628375  1.5952156  0.59320118 0.553047           0.59320118 0.554741         
# >                             CREATFAM  -0.92996689  1.7106166 -0.54364424 0.586686          -0.54364424 0.588217         
# >                              COOPAGR  -0.83382075  1.3679743 -0.60952957 0.542173          -0.60952957 0.543923         
# >                              DIGPREP   0.78038516  1.5119396  0.51614836 0.605751           0.51614836 0.607193         
# >                             STUDYHMW  -0.76879699  0.5391769 -1.42587152 0.153905          -1.42587152 0.157845         
# >                             MATHPREF   0.68052837  4.1251202  0.16497176 0.868966           0.16497176 0.869388         
# >                              CREATAS   0.62225717  1.6046847  0.38777534 0.698182           0.38777534 0.699225         
# >              ST004D01TNot Applicable   0.58976744  6.6771836  0.08832578 0.929618           0.08832578 0.929841         
# >                             EMPATAGR  -0.58558562  1.4137350 -0.41421173 0.678719          -0.41421173 0.679842         
# >                             WORKHOME  -0.58192739  0.4176571 -1.39331388 0.163525          -1.39331388 0.167433         
# >                             PROBSELF  -0.55143185  1.3422085 -0.41083919 0.681190          -0.41083919 0.682303         
# >                             COGACRCO   0.54745511  1.3905195  0.39370546 0.693799           0.39370546 0.694859         
# >                               EXPOFA  -0.27451991  1.4974954 -0.18331937 0.854547          -0.18331937 0.855017         
# >                              IMAGINE  -0.27142916  1.2552163 -0.21624094 0.828800          -0.21624094 0.829357         
# >                              DISCLIM   0.21314982  1.3330768  0.15989313 0.872965           0.15989313 0.873373         
# >                              MTTRAIN   0.20440227  1.4958059  0.13665026 0.891307           0.13665026 0.891655         
# >                              LEARRES   0.20298870  1.3838801  0.14668084 0.883384           0.14668084 0.883758         
# >                             INFOSEEK   0.16684591  1.4931634  0.11173988 0.911030           0.11173988 0.911313         
# >                            PERSEVAGR   0.10641789  1.2289513  0.08659244 0.930995           0.08659244 0.931215         
# >                             EXPO21ST  -0.06887309  1.6448418 -0.04187217 0.966601          -0.04187217 0.966706         

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
# >                      Metric  Estimate Std. Error
# > R-squared (Weighted, Train) 0.4676698 0.01501686

### ---- Predict and Evaluate Performance on Training Data ----

# --- Main model predictions for training data ---
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- train_data[[pvmaths[i]]]
  w <- train_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
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
    y_pred <- as.numeric(predict(model, newx = X_train, s = best_lambda))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing train_metrics_replicates (glmnet): 1.931 sec elapsed
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
# >    Metric           Point_estimate         Standard_error                CI_lower                CI_upper              CI_length
# >       MSE 4188.5818953603484260384 143.942759233441222477 3906.459271427483599837 4470.704519293213706987 564.245247865730107151
# >      RMSE   64.7168021722897748305   1.112285901596945514   62.536761864648099163   66.896842479931450498   4.360080615283351335
# >       MAE   51.3344407406146174822   0.906023838713432283   49.558666647601562261   53.110214833627672704   3.551548186026110443
# >      Bias    0.0000000000001553808   0.000000000001306137   -0.000000000002404601    0.000000000002715363   0.000000000005119964
# >     Bias%    1.6990573023032229383   0.062276829405713365    1.576996959596679737    1.821117645009766139   0.244120685413086402
# > R-squared    0.4676697598332424377   0.015016860454982480    0.438237254180612990    0.497102265485871886   0.058865011305258896

### ---- Predict and Evaluate Performance on Validation Data ----

# Main model predictions on validation data
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- valid_data[[pvmaths[i]]]
  w <- valid_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda))
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
    y_pred <- as.numeric(predict(model, newx = X_valid, s = best_lambda))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates (glmnet): 0.747 sec elapsed

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
# >       MSE   4675.3872095   397.03739845 3897.2082081 5453.5662110 1556.3580030
# >      RMSE     68.3680662     2.89582224   62.6923589   74.0437735   11.3514146
# >       MAE     53.9529056     2.53536497   48.9836816   58.9221296    9.9384480
# >      Bias      2.0443757     4.36140757   -6.5038261   10.5925775   17.0964035
# >     Bias%      2.3420485     0.89972387    0.5786221    4.1054749    3.5268527
# > R-squared      0.4554313     0.03702362    0.3828663    0.5279962    0.1451299

### ---- Predict and Evaluate Performance on test Data ----

# Main model predictions on test data
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y_train <- test_data[[pvmaths[i]]]
  w <- test_data[[final_wt]]
  y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda))
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
    y_pred <- as.numeric(predict(model, newx = X_test, s = best_lambda))
    compute_metrics(y_train, y_pred, w)
  }) |> t()
})
toc()

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
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4871.6743054   487.61356490 3915.9692798 5827.3793309 1911.4100511
# >      RMSE     69.7734730     3.48703591   62.9390082   76.6079378   13.6689296
# >       MAE     55.3863042     2.77740315   49.9426941   60.8299144   10.8872203
# >      Bias     -2.4100299     3.89181864  -10.0378543    5.2177945   15.2556488
# >     Bias%      1.4068862     0.85327176   -0.2654957    3.0792681    3.3447638
# > R-squared      0.4024477     0.04177113    0.3205778    0.4843176    0.1637398

### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# --- Remark ---
# This block consolidates the three previously separate prediction/evaluation sections into a single unified process.

evaluate_split <- function(split_data, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit,
                           X_split, pvmaths, best_lambda) {
  # Point estimates across PVs
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    y     <- split_data[[pvmaths[i]]]
    w     <- split_data[[final_wt]]
    y_pred <- as.numeric(predict(model, newx = X_split, s = best_lambda))
    compute_metrics(y_true = y, y_pred = y_pred, w = w)
  }) |> t() |> as.data.frame()
  main_point <- colMeans(main_metrics_df)   # length 6: mse, rmse, mae, bias, bias_pct, r2
  
  # Replicate metrics across PVs
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      y     <- split_data[[pvmaths[m]]]
      w     <- split_data[[rep_wts[g]]]
      y_pred <- as.numeric(predict(model, newx = X_split, s = best_lambda))
      compute_metrics(y_true = y, y_pred = y_pred, w = w)
    }) |> t()
  })
  
  # BRR sampling variance, averaged across PVs
  sampling_var_matrix <- sapply(1:M, function(m) {
    sweep(replicate_metrics[[m]], 2, unlist(main_metrics_df[m, ]))^2 |>
      colSums() / (G * (1 - k)^2)
  })
  sampling_var <- rowMeans(sampling_var_matrix)
  
  # Rubin imputation variance across PVs
  imputation_var <- colSums((main_metrics_df - matrix(main_point, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
  
  # Total variance and CIs
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

# Evaluate on each split (pass the prebuilt matrices)
train_eval <- evaluate_split(train_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, X_train, pvmaths, best_lambda)
valid_eval <- evaluate_split(valid_data, main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, X_valid, pvmaths, best_lambda)
test_eval  <- evaluate_split(test_data,  main_models, replicate_models, final_wt, rep_wts,
                             M, G, k, z_crit, X_test,  pvmaths, best_lambda)

### ---- summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric           Point_estimate         Standard_error                CI_lower                CI_upper              CI_length
# >       MSE 4188.5818953603484260384 143.942759233441222477 3906.459271427483599837 4470.704519293213706987 564.245247865730107151
# >      RMSE   64.7168021722897748305   1.112285901596945514   62.536761864648099163   66.896842479931450498   4.360080615283351335
# >       MAE   51.3344407406146174822   0.906023838713432283   49.558666647601562261   53.110214833627672704   3.551548186026110443
# >      Bias    0.0000000000001553808   0.000000000001306137   -0.000000000002404601    0.000000000002715363   0.000000000005119964
# >     Bias%    1.6990573023032229383   0.062276829405713365    1.576996959596679737    1.821117645009766139   0.244120685413086402
# > R-squared    0.4676697598332424377   0.015016860454982480    0.438237254180612990    0.497102265485871886   0.058865011305258896

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4675.3872095   397.03739845 3897.2082081 5453.5662110 1556.3580030
# >      RMSE     68.3680662     2.89582224   62.6923589   74.0437735   11.3514146
# >       MAE     53.9529056     2.53536497   48.9836816   58.9221296    9.9384480
# >      Bias      2.0443757     4.36140757   -6.5038261   10.5925775   17.0964035
# >     Bias%      2.3420485     0.89972387    0.5786221    4.1054749    3.5268527
# > R-squared      0.4554313     0.03702362    0.3828663    0.5279962    0.1451299

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4871.6743054   487.61356490 3915.9692798 5827.3793309 1911.4100511
# >      RMSE     69.7734730     3.48703591   62.9390082   76.6079378   13.6689296
# >       MAE     55.3863042     2.77740315   49.9426941   60.8299144   10.8872203
# >      Bias     -2.4100299     3.89181864  -10.0378543    5.2177945   15.2556488
# >     Bias%      1.4068862     0.85327176   -0.2654957    3.0792681    3.3447638
# > R-squared      0.4024477     0.04177113    0.3205778    0.4843176    0.1637398

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
# >      RMSE        64.7168         1.1123  62.5368  66.8968    4.3601
# >       MAE        51.3344         0.9060  49.5587  53.1102    3.5515
# >      Bias         0.0000         0.0000  -0.0000   0.0000    0.0000
# >     Bias%         1.6991         0.0623   1.5770   1.8211    0.2441
# > R-squared         0.4677         0.0150   0.4382   0.4971    0.0589

valid_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        68.3681         2.8958  62.6924  74.0438   11.3514
# >       MAE        53.9529         2.5354  48.9837  58.9221    9.9384
# >      Bias         2.0444         4.3614  -6.5038  10.5926   17.0964
# >     Bias%         2.3420         0.8997   0.5786   4.1055    3.5269
# > R-squared         0.4554         0.0370   0.3829   0.5280    0.1451

test_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        69.7735         3.4870  62.9390  76.6079   13.6689
# >       MAE        55.3863         2.7774  49.9427  60.8299   10.8872
# >      Bias        -2.4100         3.8918 -10.0379   5.2178   15.2556
# >     Bias%         1.4069         0.8533  -0.2655   3.0793    3.3448
# > R-squared         0.4024         0.0418   0.3206   0.4843    0.1637

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
# >      RMSE          64.72           1.11    62.54    66.90      4.36
# >       MAE          51.33           0.91    49.56    53.11      3.55
# >      Bias           0.00           0.00    -0.00     0.00      0.00
# >     Bias%           1.70           0.06     1.58     1.82      0.24
# > R-squared           0.47           0.02     0.44     0.50      0.06

valid_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          68.37           2.90    62.69    74.04     11.35
# >       MAE          53.95           2.54    48.98    58.92      9.94
# >      Bias           2.04           4.36    -6.50    10.59     17.10
# >     Bias%           2.34           0.90     0.58     4.11      3.53
# > R-squared           0.46           0.04     0.38     0.53      0.15

test_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          69.77           3.49    62.94    76.61     13.67
# >       MAE          55.39           2.78    49.94    60.83     10.89
# >      Bias          -2.41           3.89   -10.04     5.22     15.26
# >     Bias%           1.41           0.85    -0.27     3.08      3.34
# > R-squared           0.40           0.04     0.32     0.48      0.16

# --- Export to word---
library(officer); library(flextable)

fmt2 <- \(df) df %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  )

performance_tables_glmnet_voi_all_v2p1<- read_docx() %>%
  body_add_par("Performance - Elastic Net (glmnet, voi_all, v2.1)", style = "Normal") %>%  
  body_add_par("Training performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(train_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Validation performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(valid_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Test performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(test_eval))  %>% align(j = 1:6, align = "right", part = "all") %>% autofit())

print(performance_tables_glmnet_voi_all_v2p1, target = "performance_tables_glmnet_voi_all_v2p1.docx")
