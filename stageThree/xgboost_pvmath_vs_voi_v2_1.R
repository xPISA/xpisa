# ---- I. Predictive Modelling: Version 2.1 ----
#
# xgb.train with hyperparameter tuning
#
# Using Matrix::sparse.model.matrix instead of model.matrix on voi_cat, tuning is much faster. 
# 
## ---- 1. Setup ----

# Set working directory
setwd("~/projects/pisa")

# Load libraries
library(haven)        # Read SPSS .sav files
library(tidyverse)    # Includes dplyr, tidyr, purrr, ggplot2, tibble, etc.
library(broom)        # For tidying model output
library(tictoc)       # For timing code execution
library(xgboost)      # For implementing XGBoost model 
library(DiagrammeR)

# Check versions
R.version.string
# > [1] "R version 4.5.1 (2025-06-13)"
sapply(c("tidyverse","xgboost","Matrix","haven","broom","tictoc","DiagrammeR"),
       \(p) paste(p, as.character(packageVersion(p))))
# >         tidyverse             xgboost              Matrix               haven               broom              tictoc          DiagrammeR 
# > "tidyverse 2.0.0"  "xgboost 1.7.11.1"      "Matrix 1.7.3"       "haven 2.5.4"       "broom 1.0.8"      "tictoc 1.2.1" "DiagrammeR 1.0.11"

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
z_crit <- qnorm(0.975)   # 95% CI z-critical value

# Target varaible
pvmaths  <- paste0("PV", 1:M, "MATH")   # PV1MATH to PV10MATH

# Predictors (VOI = variables of interest): numeric (continuous + ordinal) + categorical (nominal)
voi_num <- c(# --- Student Questionnaire Derived Variables ---
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
  "CREATEFF",    # Creative thinking self-efficacy 
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

voi_cat <- c(# --- Student Questionnaire Variables ---
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

### ---- Prepare modeling data ----
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
  ) %>%
  filter(IMMIG != "No Response") %>%                          # Treat "No Response" in `IMMIG` as missing values and drop
  filter(if_all(all_of(voi_all), ~ !is.na(.))) %>%            # Drop missing values; ST004D01T: User-defined NA `Not Applicable` is kept as a category
  droplevels()                                                # Drop levels not present   
# To compare performance without listwise deletion 
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

### ---- Random Train/Validation/Test (80/10/10) split ----
set.seed(123)          # Ensure reproducibility
n <- nrow(temp_data)   # 6755
indices <- sample(n)   # Randomly shuffle row indices

# Compute sizes
n_train <- floor(0.80 * n)         # 5404
n_valid <- floor(0.10 * n)         # 675
n_test  <- n - n_train - n_valid   # 676

# Assign indices
train_idx <- indices[1:n_train]
valid_idx <- indices[(n_train + 1):(n_train + n_valid)]
test_idx  <- indices[(n_train + n_valid + 1):n]

# Subset the data
train_data <- temp_data[train_idx, ]
valid_data <- temp_data[valid_idx, ]
test_data  <- temp_data[test_idx, ]

# --- Check weight shares ---
for (d in c("train","valid","test")) {
  df <- get(paste0(d,"_data"))
  w  <- df[[final_wt]]
  fac <- factor(df$REGION, levels = levels(temp_data$REGION))  
  tab <- tapply(w, fac, sum)              
  share <- tab / sum(w)
  nz <- !is.na(share) & share > 0         
  cat("\nWeighted share in", d, "\n")
  print(round(share[nz], 3))
}

for (d in c("train","valid","test")) {
  df <- get(paste0(d,"_data"))
  w  <- df[[final_wt]]
  fac <- factor(df$ST004D01T, levels = levels(temp_data$ST004D01T))  
  tab <- tapply(w, fac, sum)              
  share <- tab / sum(w)
  nz <- !is.na(share) & share > 0         
  cat("\nWeighted share in", d, "\n")
  print(round(share[nz], 3))
}

for (d in c("train","valid","test")) {
  df <- get(paste0(d,"_data"))
  w  <- df[[final_wt]]
  fac <- factor(df$IMMIG, levels = levels(temp_data$IMMIG))  
  tab <- tapply(w, fac, sum)              
  share <- tab / sum(w)
  nz <- !is.na(share) & share > 0         
  cat("\nWeighted share in", d, "\n")
  print(round(share[nz], 3))
}

for (d in c("train","valid","test")) {
  df <- get(paste0(d,"_data"))
  w  <- df[[final_wt]]
  fac <- factor(df$LANGN, levels = levels(temp_data$LANGN))  
  tab <- tapply(w, fac, sum)              
  share <- tab / sum(w)
  nz <- !is.na(share) & share > 0         
  cat("\nWeighted share in", d, "\n")
  print(round(share[nz], 3))
}

for (d in c("train","valid","test")) {
  df <- get(paste0(d,"_data"))
  w  <- df[[final_wt]]
  fac <- factor(df$SCHLTYPE, levels = levels(temp_data$SCHLTYPE))  
  tab <- tapply(w, fac, sum)              
  share <- tab / sum(w)
  nz <- !is.na(share) & share > 0         
  cat("\nWeighted share in", d, "\n")
  print(round(share[nz], 3))
}

## ---- 2. PV1MATH only ----

# --- Remark ---
# 1) Repeat the same process for PV2MATH - PV10MATH.
# 2) Apply best results from PV1MATH to all plausible values in mathematics. 

### ---- Fit main model using final student weights (W_FSTUWT) on the training data----

#### ---- Tune XGBoost model for PV1MATH only: xgb.train ----

##### ---- Prepare ----
# Define target plausible value
pv1math <- pvmaths[1]   # "PV1MATH"

# Freeze factor levels so one-hot columns align across splits
voi_levels <- lapply(temp_data[, voi_cat, drop = FALSE], levels)
for (voi in names(voi_levels)) {
  train_data[[voi]] <- factor(train_data[[voi]], levels = voi_levels[[voi]])
  valid_data[[voi]] <- factor(valid_data[[voi]], levels = voi_levels[[voi]])
  test_data[[voi]]  <- factor(test_data[[voi]],  levels = voi_levels[[voi]])
}
voi_levels
# > $REGION
# > [1] "Canada: Newfoundland and Labrador" "Canada: Prince Edward Island"      "Canada: Nova Scotia"               "Canada: New Brunswick"            
# > [5] "Canada: Quebec"                    "Canada: Ontario"                   "Canada: Manitoba"                  "Canada: Saskatchewan"             
# > [9] "Canada: Alberta"                   "Canada: British Columbia"         

# > $ST004D01T
# > [1] "Female"         "Male"           "Not Applicable"

# > $IMMIG
# > [1] "Native student"            "Second-Generation student" "First-Generation student" 

# > $LANGN
# > [1] "English"                "French"                 "Another language (CAN)"

# > $SCHLTYPE
# > [1] "Private independent"          "Private Government-dependent" "Public" 

# Targets & weights 
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

# Numeric block
X_train_num <- as.matrix(train_data[, voi_num, drop = FALSE])
X_valid_num <- as.matrix(valid_data[, voi_num, drop = FALSE])
X_test_num  <- as.matrix(test_data[,  voi_num, drop = FALSE])

dim(X_train_num); dim(X_valid_num); dim(X_test_num)
# > [1] 5404   53
# > [1] 675  53
# > [1] 676  53

# One-hot as sparse
contrast_list <- lapply(temp_data[, voi_cat, drop = FALSE], function(f) contrasts(f, contrasts = FALSE))
contrast_list
# > $REGION
# >                                   Canada: Newfoundland and Labrador Canada: Prince Edward Island Canada: Nova Scotia Canada: New Brunswick Canada: Quebec Canada: Ontario Canada: Manitoba Canada: Saskatchewan Canada: Alberta Canada: British Columbia
# > Canada: Newfoundland and Labrador                                 1                            0                   0                     0              0               0                0                    0               0                        0
# > Canada: Prince Edward Island                                      0                            1                   0                     0              0               0                0                    0               0                        0
# > Canada: Nova Scotia                                               0                            0                   1                     0              0               0                0                    0               0                        0
# > Canada: New Brunswick                                             0                            0                   0                     1              0               0                0                    0               0                        0
# > Canada: Quebec                                                    0                            0                   0                     0              1               0                0                    0               0                        0
# > Canada: Ontario                                                   0                            0                   0                     0              0               1                0                    0               0                        0
# > Canada: Manitoba                                                  0                            0                   0                     0              0               0                1                    0               0                        0
# > Canada: Saskatchewan                                              0                            0                   0                     0              0               0                0                    1               0                        0
# > Canada: Alberta                                                   0                            0                   0                     0              0               0                0                    0               1                        0
# > Canada: British Columbia                                          0                            0                   0                     0              0               0                0                    0               0                        1

# > $ST004D01T
# > Female Male Not Applicable
# > Female              1    0              0
# > Male                0    1              0
# > Not Applicable      0    0              1

# > $IMMIG
# > Native student Second-Generation student First-Generation student
# > Native student                         1                         0                        0
# > Second-Generation student              0                         1                        0
# > First-Generation student               0                         0                        1

# > $LANGN
# > English French Another language (CAN)
# > English                      1      0                      0
# > French                       0      1                      0
# > Another language (CAN)       0      0                      1

# > $SCHLTYPE
# > Private independent Private Government-dependent Public
# > Private independent                            1                            0      0
# > Private Government-dependent                   0                            1      0
# > Public                                         0                            0      1

X_train_cat <- Matrix::sparse.model.matrix(
  ~ . - 1,
  data = train_data[, voi_cat, drop = FALSE],
  contrasts.arg = contrast_list,
  xlev = voi_levels
)

X_valid_cat <- Matrix::sparse.model.matrix(
  ~ . - 1,
  data = valid_data[, voi_cat, drop = FALSE],
  contrasts.arg = contrast_list,
  xlev = voi_levels
)

X_test_cat <- Matrix::sparse.model.matrix(
  ~ . - 1,
  data = test_data[, voi_cat, drop = FALSE],
  contrasts.arg = contrast_list,
  xlev = voi_levels
)

dim(X_train_cat); dim(X_valid_cat); dim(X_test_cat)
# > [1] 5404   22
# > [1] 675  22
# > [1] 676  22

# Bind numeric as sparse too (or keep dense and cbind2)
X_train <- cbind(Matrix::Matrix(as.matrix(train_data[, voi_num, drop=FALSE]), sparse=TRUE), X_train_cat)
X_valid <- cbind(Matrix::Matrix(as.matrix(valid_data[, voi_num, drop=FALSE]), sparse=TRUE), X_valid_cat)
X_test  <- cbind(Matrix::Matrix(as.matrix(test_data[,  voi_num, drop=FALSE]), sparse=TRUE),  X_test_cat)

dim(X_train); dim(X_valid); dim(X_test)
# > [1] 5404   75
# > [1] 675  75
# > [1] 676  75

# Model matrix invariants
stopifnot(ncol(X_train) == length(voi_num) + sum(lengths(voi_levels)))
stopifnot(identical(colnames(X_train), colnames(X_valid)))
stopifnot(identical(colnames(X_train), colnames(X_test)))

# CHECK: levels
grep("^EXERPRAC", colnames(X_train), value = TRUE)
# > [1] "EXERPRAC"
grep("^REGION",colnames(X_train), value = TRUE)
# > [1] "REGIONCanada: Newfoundland and Labrador" "REGIONCanada: Prince Edward Island"      "REGIONCanada: Nova Scotia"               "REGIONCanada: New Brunswick"            
# > [5] "REGIONCanada: Quebec"                    "REGIONCanada: Ontario"                   "REGIONCanada: Manitoba"                  "REGIONCanada: Saskatchewan"             
# > [9] "REGIONCanada: Alberta"                   "REGIONCanada: British Columbia"    
grep("^ST004D01T",colnames(X_train), value = TRUE)
# > [1] "ST004D01TFemale"         "ST004D01TMale"           "ST004D01TNot Applicable"
grep("^IMMIG",colnames(X_train), value = TRUE)
# > [1] "IMMIGNative student"            "IMMIGSecond-Generation student" "IMMIGFirst-Generation student" 
grep("^LANGN",    colnames(X_train), value = TRUE)
# > [1] "LANGNEnglish"                "LANGNFrench"                 "LANGNAnother language (CAN)"
grep("^SCHLTYPE", colnames(X_train), value = TRUE)
# > [1] "SCHLTYPEPrivate independent"          "SCHLTYPEPrivate Government-dependent" "SCHLTYPEPublic"

grep("^EXERPRAC", colnames(X_valid), value = TRUE)
# > [1] "EXERPRAC"
grep("^REGION",colnames(X_valid), value = TRUE)
# > [1] "REGIONCanada: Newfoundland and Labrador" "REGIONCanada: Prince Edward Island"      "REGIONCanada: Nova Scotia"               "REGIONCanada: New Brunswick"            
# > [5] "REGIONCanada: Quebec"                    "REGIONCanada: Ontario"                   "REGIONCanada: Manitoba"                  "REGIONCanada: Saskatchewan"             
# > [9] "REGIONCanada: Alberta"                   "REGIONCanada: British Columbia"    
grep("^ST004D01T",colnames(X_valid), value = TRUE)
# > [1] "ST004D01TFemale"         "ST004D01TMale"           "ST004D01TNot Applicable"
grep("^IMMIG",colnames(X_valid), value = TRUE)
# > [1] "IMMIGNative student"            "IMMIGSecond-Generation student" "IMMIGFirst-Generation student" 
grep("^LANGN",    colnames(X_valid), value = TRUE)
# > [1] "LANGNEnglish"                "LANGNFrench"                 "LANGNAnother language (CAN)"
grep("^SCHLTYPE", colnames(X_valid), value = TRUE)
# > [1] "SCHLTYPEPrivate independent"          "SCHLTYPEPrivate Government-dependent" "SCHLTYPEPublic"

grep("^EXERPRAC", colnames(X_test), value = TRUE)
# > [1] "EXERPRAC"
grep("^REGION",colnames(X_test), value = TRUE)
# > [1] "REGIONCanada: Newfoundland and Labrador" "REGIONCanada: Prince Edward Island"      "REGIONCanada: Nova Scotia"               "REGIONCanada: New Brunswick"            
# > [5] "REGIONCanada: Quebec"                    "REGIONCanada: Ontario"                   "REGIONCanada: Manitoba"                  "REGIONCanada: Saskatchewan"             
# > [9] "REGIONCanada: Alberta"                   "REGIONCanada: British Columbia"    
grep("^ST004D01T",colnames(X_test), value = TRUE)
# > [1] "ST004D01TFemale"         "ST004D01TMale"           "ST004D01TNot Applicable"
grep("^IMMIG",colnames(X_test), value = TRUE)
# > [1] "IMMIGNative student"            "IMMIGSecond-Generation student" "IMMIGFirst-Generation student" 
grep("^LANGN",    colnames(X_test), value = TRUE)
# > [1] "LANGNEnglish"                "LANGNFrench"                 "LANGNAnother language (CAN)"
grep("^SCHLTYPE", colnames(X_test), value = TRUE)
# > [1] "SCHLTYPEPrivate independent"          "SCHLTYPEPrivate Government-dependent" "SCHLTYPEPublic"

# Inspect non-zero counts
Matrix::nnzero(X_train) / (nrow(X_train)*ncol(X_train))  # overall sparsity ratio
# > [1] 0.7224303
Matrix::nnzero(X_valid) / (nrow(X_valid)*ncol(X_valid))
# > [1] 0.7218963
Matrix::nnzero(X_test) / (nrow(X_test)*ncol(X_test))
# > [1] 0.7226627

# CHECK: matrices have no NA and aligned columns
stopifnot(!anyNA(X_train), !anyNA(X_valid), !anyNA(X_test))
stopifnot(identical(colnames(X_train), colnames(X_valid)))
stopifnot(identical(colnames(X_train), colnames(X_test)))

# Create DMatrix 
dtrain <- xgb.DMatrix(X_train, label=y_train, weight=w_train)
dvalid <- xgb.DMatrix(X_valid, label=y_valid, weight=w_valid)
dtest  <- xgb.DMatrix(X_test,  label=y_test,  weight=w_test)

dim(dtrain); dim(dvalid); dim(dtest)
# > [1] 5404   75
# > [1] 675  75
# > [1] 676  75

# Sanity on shapes
stopifnot(nrow(X_train) == length(y_train),
          nrow(X_valid) == length(y_valid),
          nrow(X_test)  == length(y_test))

# Define hyperparameter grid (3 × 3 × 4 = 36 combinations)
grid <- expand.grid(
  nrounds = c(100, 200, 300),       # Max number of boosting iterations.
  max_depth = c(4, 6, 8),           # Maximum depth of a tree. Default: 6.
  eta = c(0.01, 0.05, 0.1, 0.3),    # eta control the learning rate: 0 < eta < 1.
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

# Initialize results table to track tuning performance
tuning_results <- tibble(
  grid_id = integer(),
  param_name = character(),
  nrounds = integer(),
  max_depth = integer(),
  eta = double(),
  best_iter_in_grid = integer(),     # Best iteration (boosting round) with lowest valid RMSE
  train_rmse = double(),             # Train RMSE at best iteration
  valid_rmse = double()              # Validation RMSE at best iteration
)

# Store all trained models
model_list <- vector("list", nrow(grid))  

# Store evaluation logs for all runs (for visualization later)
eval_log_list <- list()

# Track best model and its metrics
best_rmse <- Inf        # best valid rmse
best_model <- NULL      # model with best valid rmse
best_params <- list()
best_eval_log <- NULL
best_iter <- NULL
best_grid_id <- NULL

##### ---- Tune ----
set.seed(123)                                         
tic("Tuning (xgb.train)")
for (i in seq_len(nrow(grid))) {                           # nrow(grid) = 36
  row <- grid[i, ]
  
  # Print the current hyperparameter combination
  message(sprintf("Fitting model %d/%d: nrounds = %d, max_depth = %d, eta = %.3f", 
                  i, nrow(grid), row$nrounds, row$max_depth, row$eta))
  
  # Set model parameters
  params <- list(
    objective = "reg:squarederror",                         # Specify the learning task and the corresponding learning objective: reg:squarederror Regression with squared loss (Default)
    max_depth = row$max_depth,                          
    eta = row$eta,                                      
    eval_metric = "rmse",                                   # Default: metric will be assigned according to objective(rmse for regression) 
    nthread = max(1, parallel::detectCores() - 1)           # Manually specify number of thread
    #seed = 123                                             # Random number seed for reproducibility (e.g. tune subsample, colsample_bytree for regularization)
  )
  
  # Train model on current combination of hyperparameters
  mod <- xgb.train(                                         # xgb.train, ?xgb.train
    params = params,
    data = dtrain,
    nrounds = row$nrounds,
    watchlist = list(train = dtrain, valid = dvalid),       # Fit on TRAIN and evaluate on VALID
    verbose = 1,                                            # If 0, xgboost will stay silent. If 1 (default), it will print information about performance. If 2, some additional information will be printed out.
    early_stopping_rounds = NULL  
  )
  
  # Save current model
  model_list[[i]] <- mod  
  
  # Extract log of RMSE over boosting rounds
  eval_log_i <- mod$evaluation_log
  best_iter_i <- which.min(eval_log_i$valid_rmse)           # Best iteration (boosting round) with lowest VALID rmse
  best_train_rmse_i <- eval_log_i$train_rmse[best_iter_i]   # TRAIN rmse at best iteration (not the lowest) 
  best_valid_rmse_i <- eval_log_i$valid_rmse[best_iter_i]   # Lowest VALID rmse
  
  # # --- (Optional) Verify: logged RMSE is truly *weighted* (VALID & TRAIN) ---
  # # Predict at the selected iteration for this config
  # pred_valid_i <- predict(mod, dvalid, iterationrange = c(1, best_iter_i + 1))
  # pred_train_i <- predict(mod, dtrain, iterationrange = c(1, best_iter_i + 1))
  # 
  # # sanity: iterationrange vs ntreelimit equivalence
  # stopifnot(all.equal(
  #   predict(mod, dvalid, iterationrange = c(1, best_iter_i + 1)),
  #   predict(mod, dvalid, ntreelimit = best_iter_i)
  # ))
  # 
  # # Manual weighted RMSE (use your compute_metrics helper for consistency)
  # wrmse_valid_i <- compute_metrics(y_true = y_valid, y_pred = pred_valid_i, w = w_valid)["rmse"]
  # wrmse_train_i <- compute_metrics(y_true = y_train, y_pred = pred_train_i, w = w_train)["rmse"]
  # 
  # # Compare with xgboost's evaluation_log at best_iter_i
  # if (!isTRUE(all.equal(best_valid_rmse_i, wrmse_valid_i))) {
  #   warning(sprintf(
  #     "Weighted VALID RMSE mismatch (grid_id=%d, iter=%d): log=%.8f vs manual=%.8f",
  #     i, best_iter_i, best_valid_rmse_i, wrmse_valid_i
  #   ))
  # }
  # if (!isTRUE(all.equal(best_train_rmse_i, wrmse_train_i))) {
  #   warning(sprintf(
  #     "Weighted TRAIN RMSE mismatch (grid_id=%d, iter=%d): log=%.8f vs manual=%.8f",
  #     i, best_iter_i, best_train_rmse_i, wrmse_train_i
  #   ))
  # }
  # # --- end verification ---
  
  # Store log with grid ID for plotting later
  eval_log_list[[i]] <- eval_log_i |> mutate(grid_id = i)
  
  # Save current result summary to tuning_results
  tuning_results <- add_row(tuning_results,
                            grid_id = i,
                            param_name = paste0("nrounds=", row$nrounds, ", max_depth=", row$max_depth, ", eta=", row$eta),
                            nrounds = row$nrounds,
                            max_depth = row$max_depth,
                            eta = row$eta,
                            best_iter_in_grid = best_iter_i,
                            train_rmse = best_train_rmse_i,
                            valid_rmse = best_valid_rmse_i
  )
  
  # Update best model tracking if current RMSE is better
  # NOTE: strict '<' means exact ties keep the first encountered config.
  # With expand.grid ordering (nrounds slowest varying: 100 <- 200 <- 300),
  # ties favor smaller nrounds (simpler), which is desirable.
  if (best_valid_rmse_i < best_rmse) {
    best_rmse <- best_valid_rmse_i
    best_model <- mod
    best_params <- params
    best_eval_log <- eval_log_i
    best_iter <- best_iter_i
    best_grid_id <- i
  }
}
toc()
# > Tuning (xgb.train): 215.523 sec elapsed

# Sanity: best_iter uses only trees that exist
stopifnot(best_iter <= grid$nrounds[best_grid_id])

##### ---- Explore tuning output ----
length(model_list)  #36
model_list[[1]]
print(as.data.frame(model_list[[1]]$evaluation_log))
length(eval_log_list)  #36
print(as.data.frame(eval_log_list[[1]]))

tuning_results
#tuning_results <- tuning_results %>% mutate(gap = valid_rmse - train_rmse)
print(head(as.data.frame(tuning_results %>% arrange(valid_rmse))), row.names = FALSE)   # Tie break: e.g. grid_id 11 vs 12, smaller full nrounds is chosen in tuning
# > grid_id                         param_name nrounds max_depth  eta best_iter_in_grid train_rmse valid_rmse
# >      12 nrounds=300, max_depth=4, eta=0.05     300         4 0.05               223   45.99244   66.55130
# >      11 nrounds=200, max_depth=4, eta=0.05     200         4 0.05               200   47.23885   66.74407
# >      14 nrounds=200, max_depth=6, eta=0.05     200         6 0.05               122   35.74260   66.80707
# >      15 nrounds=300, max_depth=6, eta=0.05     300         6 0.05               122   35.74260   66.80707
# >      20  nrounds=200, max_depth=4, eta=0.1     200         4 0.10               111   45.71828   66.83304
# >      21  nrounds=300, max_depth=4, eta=0.1     300         4 0.10               111   45.71828   66.83304
print(head(as.data.frame(tuning_results %>% arrange(valid_rmse, best_iter_in_grid, nrounds))), row.names = FALSE)  # Tie-break
print(as.data.frame(tuning_results %>% arrange(valid_rmse)), row.names = FALSE)  # Print all
#print(tuning_results %>% arrange(valid_rmse), n = Inf)

# underfitting/overfitting glance
tuning_results %>% 
  mutate(gap = valid_rmse - train_rmse) %>% 
  arrange(desc(gap)) %>% 
  as.data.frame() %>%
  #head() %>%
  print(row.names=FALSE)

# --- Explore best_model ---
best_model                  # str(best_model)                 
best_model$evaluation_log   # <=> best_eval_log
best_rmse
# > [1] 66.5513
best_params
best_params$max_depth; best_params$eta
# > [1] 4
# > [1] 0.05
best_iter
# > [1] 223
best_grid_id
# > [1] 12
grid[best_grid_id, ]
# >    nrounds max_depth  eta
# > 12     300         4 0.05

print(as.data.frame(best_model$evaluation_log))  # grid_id=12 nrounds=300, max_depth=4, eta=0.05
best_model$evaluation_log |>                     # <=> model_list[[best_grid_id]]$evaluation_log |>
  pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = best_iter, linetype = 2, alpha = 0.4) +
  annotate("text", x = best_iter, y = max(best_model$evaluation_log$valid_rmse) + 5,
           label = paste("Best iter =", best_iter), size = 3, vjust = 0) + 
  labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal()  + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))   # Visualizing the best path

print(as.data.frame(model_list[[11]]$evaluation_log))  # Comparing a close configuration: grid_id=11 nrounds=200, max_depth=4, eta=0.05
model_list[[11]]$evaluation_log |>
  pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = which.min(model_list[[11]]$evaluation_log$valid_rmse), linetype = 2, alpha = 0.4) +
  annotate("text", x = which.min(model_list[[11]]$evaluation_log$valid_rmse), y = max(model_list[[11]]$evaluation_log$valid_rmse) + 5,
           label = paste("Best iter =", which.min(model_list[[11]]$evaluation_log$valid_rmse)), size = 3, vjust = 0) + 
  labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

xgb.importance(model = best_model) %>% head()                              #?xgb.importance
#xgb.importance(model = best_model, trees = 0:(best_iter - 1)) 
xgb.importance(model = best_model, trees = 0:(best_iter - 1)) %>% head(10) # 0-based indices
# >     Feature       Gain      Cover  Frequency
# >      <char>      <num>      <num>      <num>
# >   1:  MATHEFF 0.40323124 0.06693567 0.04151515
# >   2: FAMSUPSL 0.05032721 0.03434541 0.04303030
# >   3:     ESCS 0.04773720 0.03275999 0.03424242
# >   4:   ANXMAT 0.03699428 0.02623820 0.02181818
# >   5:   FAMCON 0.02777742 0.02935347 0.02848485
# >   6:  WORKPAY 0.02670850 0.01907748 0.01939394
# >   7: EXERPRAC 0.02393390 0.02716222 0.03030303
# >   8:  HOMEPOS 0.02258822 0.02722482 0.03454545
# >   9: CREATOOS 0.01664286 0.01093829 0.01606061
# >   10: MATHEF21 0.01617483 0.02845825 0.02030303
sum(xgb.importance(model = best_model, trees = 0:(best_iter - 1))$Gain) # Sanity: sum of gains equals total
# > [1] 1

# Aggregate one‑hot dummies back to the parent variable 
xgb.importance(model = best_model, trees = 0:(best_iter - 1)) |>
  tibble::as_tibble() |>
  dplyr::mutate(
    Feature = purrr::map_chr(Feature, \(f) {
      # If it's one of the numeric VOIs, keep the name
      if (f %in% voi_num) return(f)
      # Otherwise, find which categorical variable name is a prefix of the dummy
      hit <- voi_cat[purrr::map_lgl(voi_cat, ~ startsWith(f, .x))]
      # If multiple match (rare; overlapping names), take the longest prefix; else fallback to original
      if (length(hit)) hit[[which.max(nchar(hit))]] else f
    })
  ) |>
  dplyr::group_by(Feature) |>
  dplyr::summarise(
    Gain      = sum(Gain),
    Cover     = sum(Cover),
    Frequency = sum(Frequency),
    .groups   = "drop"
  ) |>
  dplyr::arrange(dplyr::desc(Gain)) |>
  as.data.frame() |>
  head() |> # Comment it out to display the full
  print(row.names=FALSE) 
# >  Feature       Gain      Cover  Frequency
# >  MATHEFF 0.40323124 0.06693567 0.04151515
# > FAMSUPSL 0.05032721 0.03434541 0.04303030
# >     ESCS 0.04773720 0.03275999 0.03424242
# >   ANXMAT 0.03699428 0.02623820 0.02181818
# >   FAMCON 0.02777742 0.02935347 0.02848485
# >  WORKPAY 0.02670850 0.01907748 0.01939394

# Relative importance 
xgb.plot.importance(importance_matrix = xgb.importance(model = best_model, trees = 0:(best_iter - 1)),
                    top_n = NULL,                              # Top n features (you only have 4 in oos)
                    measure = "Gain",                          # Can also use "Cover" or "Frequency"
                    rel_to_first = TRUE,
                    xlab = "Relative Importance")

xgb.plot.importance(
  importance_matrix = data.table::as.data.table(
    xgb.importance(model = best_model, trees = 0:(best_iter - 1)) |>
      tibble::as_tibble() |>
      dplyr::mutate(
        # Collapse OHE dummies to their parent variable
        Feature = purrr::map_chr(Feature, \(f) {
          if (f %in% voi_num) return(f)
          hits <- voi_cat[startsWith(f, voi_cat)]
          if (length(hits)) hits[[which.max(nchar(hits))]] else f
        })
      ) |>
      dplyr::group_by(Feature) |>
      dplyr::summarise(
        Gain      = sum(Gain),
        Cover     = sum(Cover),
        Frequency = sum(Frequency),
        .groups   = "drop"
      ) |>
      # <<< Make it truly "relative importance" (top feature = 1)
      dplyr::mutate(Gain = Gain / ifelse(max(Gain) > 0, max(Gain), 1)) |>
      dplyr::arrange(dplyr::desc(Gain))
  ),
  top_n = NULL,
  measure = "Gain",
  rel_to_first = TRUE,   # already normalized to [0, 1]
  xlab = "Relative Importance"
)

# Absolute Importance
# no collapse one-hot dummies back to parent variable
xgb.plot.importance(
  importance_matrix = data.table::as.data.table(
    xgb.importance(model = best_model, trees = 0:(best_iter - 1)) |>
      tibble::as_tibble() |>
      dplyr::arrange(dplyr::desc(Gain))
  ),
  top_n        = NULL,               # show all features (including each dummy)
  measure      = "Gain",             # plot Gain
  rel_to_first = FALSE,              # raw (absolute) values, not scaled to top
  xlab         = "Absolute Importance (Gain)"
)

xgb.plot.importance(
  importance_matrix = data.table::as.data.table(
    xgb.importance(model = best_model, trees = 0:(best_iter - 1)) |>
      tibble::as_tibble() |>
      dplyr::mutate(
        # collapse one-hot dummies back to parent variable
        Feature = purrr::map_chr(Feature, \(f) {
          if (f %in% voi_num) return(f)
          hits <- voi_cat[startsWith(f, voi_cat)]
          if (length(hits)) hits[[which.max(nchar(hits))]] else f
        })
      ) |>
      dplyr::group_by(Feature) |>
      dplyr::summarise(
        Gain      = sum(Gain),
        Cover     = sum(Cover),
        Frequency = sum(Frequency),
        .groups   = "drop"
      ) |>
      dplyr::arrange(dplyr::desc(Gain))
  ),
  top_n        = NULL,              # show all variables
  measure      = "Gain",            # plot Gain
  rel_to_first = FALSE,             # <-- raw (absolute) Gain, not scaled to top
  xlab         = "Absolute Importance (Gain)"
)

xgb.importance(model = best_model, trees = 0:(best_iter - 1)) |>
  tibble::as_tibble() |>
  dplyr::mutate(
    Feature = purrr::map_chr(Feature, \(f) {
      if (f %in% voi_num) return(f)
      hit <- voi_cat[purrr::map_lgl(voi_cat, ~ startsWith(f, .x))]
      if (length(hit)) hit[[which.max(nchar(hit))]] else f
    })
  ) |>
  dplyr::group_by(Feature) |>
  dplyr::summarise(Gain = sum(Gain), .groups = "drop") |>
  dplyr::arrange(dplyr::desc(Gain)) |>
  dplyr::mutate(cum_gain = cumsum(Gain), cum_gain_pct = cum_gain / sum(Gain)) |> 
  filter(cum_gain_pct <= 0.90) |> 
  pull(Feature)
# > [1] "MATHEFF"  "FAMSUPSL" "ESCS"     "ANXMAT"   "FAMCON"   "WORKPAY"  "EXERPRAC" "HOMEPOS"  "CREATOOS" "MATHEF21" "CREATAS"  "CREATEFF" "GROSAGR"  "LEARRES"  "CURIOAGR" "INFOSEEK"
# > [17] "SCHSUST"  "BULLIED"  "ICTRES"   "OPENCUL"  "FEELLAH"  "CREATOP"  "EMOCOAGR" "WORKHOME" "COGACRCO" "CREATSCH" "PROBSELF" "COGACMCO" "ASSERAGR" "DIGPREP"  "EXPOFA"

xgb.plot.tree(model = best_model, trees = 0)               # or trees = 1, 2, etc. 
xgb.plot.tree(model = best_model, trees = best_iter-1)     # trees = best_iter-1 as trees = 0 may not be representative
#xgb.dump(best_model, with_stats=TRUE)                     # text alternative

### ---- Predict and evaluate performance on training/validation/test datasets ----

# Predict using best_model at best_iter 
pred_train <- predict(best_model, dtrain, iterationrange = c(1, best_iter + 1))   # ?predict.xgb.Booster; Can try predinteraction = TRUE or predcontrib = TRUE (SHAP)
pred_valid <- predict(best_model, dvalid, iterationrange = c(1, best_iter + 1))   # iterationrange = c(1, best_iter + 1) <=> ntreelimit = best_iter (depreciated)  
pred_test  <- predict(best_model, dtest, iterationrange = c(1, best_iter + 1))

# Compute weighted metrics
metrics_train <- compute_metrics(y_true = y_train, y_pred = pred_train, w = w_train)
metrics_valid <- compute_metrics(y_true = y_valid, y_pred = pred_valid, w = w_valid)
metrics_test  <- compute_metrics(y_true = y_test,  y_pred = pred_test,  w = w_test)

# Combine and display results 
metric_results <- tibble::tibble(
  Dataset = c("Training", "Validation", "Test"),
  #MSE     = c(metrics_train["mse"],      metrics_valid["mse"],      metrics_test["mse"]),       # optional
  RMSE     = c(metrics_train["rmse"],     metrics_valid["rmse"],     metrics_test["rmse"]),
  MAE      = c(metrics_train["mae"],      metrics_valid["mae"],      metrics_test["mae"]),
  Bias     = c(metrics_train["bias"],     metrics_valid["bias"],     metrics_test["bias"]),
  `Bias%`  = c(metrics_train["bias_pct"], metrics_valid["bias_pct"], metrics_test["bias_pct"]),
  R2       = c(metrics_train["r2"],       metrics_valid["r2"],       metrics_test["r2"])
)

print(as.data.frame(metric_results), row.names = FALSE)
# >    Dataset     RMSE      MAE         Bias     Bias%        R2
# >   Training 45.99244 36.03781 -0.005572079 1.1442700 0.7376843
# > Validation 66.55130 51.46321 -0.546652315 1.7848337 0.4821631
# >       Test 68.92864 55.27636 -4.871303511 0.7623835 0.3938294

# Compare to a weighted baseline (predicting weighted mean of training y)
ybar <- sum(w_train*y_train)/sum(w_train)
baseline_train <- compute_metrics(y_train, rep(ybar, length(y_train)), w_train)["rmse"]
baseline_train
# >     rmse 
# > 89.79958 
baseline_valid <- compute_metrics(y_valid, rep(ybar, length(y_valid)), w_valid)["rmse"]  
baseline_valid
# >     rmse 
# > 92.48749
baseline_test  <- compute_metrics(y_test,  rep(ybar, length(y_test)),  w_test)["rmse"]  
baseline_test
# >     rmse 
# > 88.82814


## ---- 3. PV1MATH - PV10MATH (all plausible values in mathematics) ----

# --- Remark ---
# -> Apply best results from PV1MATH to all plausible values in mathematics
#    using BOTH numeric (voi_num) and categorical (voi_cat; one-hot) predictors.

# Reuse objects from Part 1/2; ensure they exist
stopifnot(
  exists("X_train"), exists("X_valid"), exists("X_test"),
  exists("best_params"), exists("best_iter"),
  exists("voi_num"), exists("voi_cat"), exists("voi_all"),
  exists("train_data"), exists("valid_data"), exists("test_data"),
  exists("final_wt"), exists("rep_wts"), exists("pvmaths"),
  exists("compute_metrics")
)

# --- Small helpers (define only if not already defined) ---

if (!exists("collapse_feature")) {
  collapse_feature <- function(f) {
    if (f %in% voi_num) return(f)
    hits <- voi_cat[startsWith(f, voi_cat)]
    if (length(hits)) hits[[which.max(nchar(hits))]] else f
  }
}

if (!exists("collapse_importance_to_voi")) {
  collapse_importance_to_voi <- function(imp_df) {
    if (is.null(imp_df) || nrow(imp_df) == 0) {
      return(setNames(numeric(0), character(0)))
    }
    imp_df |>
      tibble::as_tibble() |>
      dplyr::mutate(Feature = vapply(Feature, collapse_feature, "")) |>
      dplyr::group_by(Feature) |>
      dplyr::summarise(Gain = sum(Gain), .groups = "drop") |>
      (\(x) setNames(x$Gain, x$Feature))()
  }
}

### ---- Fit main models using final student weight (W_FSTUWT) on the training data ----

set.seed(123)

tic("Fitting main models")
main_models <- lapply(pvmaths, function(pv) {
  
  # Labels + weights by PV
  y_train <- train_data[[pv]]
  y_valid <- valid_data[[pv]]
  w_train <- train_data[[final_wt]]
  w_valid <- valid_data[[final_wt]]
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = w_train)
  dvalid <- xgb.DMatrix(data = X_valid, label = y_valid, weight = w_valid)
  
  mod <- xgb.train(
    params = list(
      objective   = "reg:squarederror",
      max_depth   = best_params$max_depth,                # best_params$max_depth = 4
      eta         = best_params$eta,                      # best_params$eta = 0.05
      eval_metric = "rmse",
      nthread     = max(1, parallel::detectCores() - 1)
    ),
    data  = dtrain,
    nrounds = best_iter,                                  # reuse tuned iteration: best_iter = 223
    watchlist = list(train = dtrain, valid = dvalid),     # 'valid_rmse' in evaluation_log <=> 'eval_rmse' if set watchlist = list(train = dtrain, eval = dvalid)
    verbose = 1,
    early_stopping_rounds = NULL
  )
  
  imp_raw <- xgb.importance(model = mod, trees = 0:(best_iter - 1))
  imp_agg <- collapse_importance_to_voi(imp_raw)          # named vector over voi_all
  
  list(
    mod            = mod,
    formula        = as.formula(paste(pv, "~", paste(voi_all, collapse = " + "))),
    importance_raw = imp_raw,
    importance     = imp_agg
  )
})
toc()
# > Fitting main models: 43.519 sec elapsed

# Light inspection (using 'valid_rmse')
main_models[[1]]$formula
main_models[[1]]$mod$evaluation_log
which.min(main_models[[1]]$mod$evaluation_log$valid_rmse)
# > 1] 223
min(main_models[[1]]$mod$evaluation_log$valid_rmse)
# > [1] 66.5513

main_models[[1]]$mod$evaluation_log |>
  pivot_longer(cols = c(train_rmse, valid_rmse), names_to = "Dataset", values_to = "RMSE") |>
  ggplot(aes(x = iter, y = RMSE, color = Dataset)) +
  geom_line(linewidth = 1) +
  labs(
    title = "XGBoost RMSE over Boosting Rounds",
    x = "Boosting Round",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal()

tibble::tibble(
  pv = pvmaths,
  best_nround     = sapply(main_models, \(m) which.min(m$mod$evaluation_log$valid_rmse)),
  best_valid_rmse = sapply(main_models, \(m) min(m$mod$evaluation_log$valid_rmse))
)
# > # A tibble: 10 × 3
# >   pv       best_nround best_valid_rmse
# >   <chr>          <int>           <dbl>
# > 1 PV1MATH          223            66.6
# > 2 PV2MATH          220            66.6
# > 3 PV3MATH          223            66.5
# > 4 PV4MATH          193            65.0
# > 5 PV5MATH          223            64.8
# > 6 PV6MATH          223            66.4
# > 7 PV7MATH          210            68.2
# > 8 PV8MATH          223            66.3
# > 9 PV9MATH          220            64.8
# > 10 PV10MATH        220            65.1

# --- Variable Importance (main): M × p over voi_all, fill 0 if absent ---
feature_order <- voi_all
main_importance_matrix <- do.call(rbind, lapply(main_models, function(m) {
  vec <- m$importance[feature_order]
  vec[is.na(vec)] <- 0
  vec
}))
dimnames(main_importance_matrix) <- list(pvmaths, feature_order)

# quick checks
stopifnot(all(dim(main_importance_matrix) == c(M, length(feature_order))))
stopifnot(!anyNA(main_importance_matrix))
if (any(abs(rowSums(main_importance_matrix) - 1) > 1e-6))
  warning("Row sums not ~1 for some PVs (importance by Gain).")

# Mean importance across PVs
main_importance <- colMeans(main_importance_matrix)
main_importance[1: 6]
# >      MATHMOT     MATHEASE     MATHPREF     EXERPRAC     STUDYHMW      WORKPAY 
# > 5.430098e-05 3.468697e-04 6.748619e-04 2.716011e-02 5.022705e-03 2.989222e-02 

# Display ranked importance (top few)
tibble(
  Variable   = names(main_importance),
  Importance = main_importance
) |> arrange(desc(Importance)) |> head()
# > # A tibble: 6 × 2
# > Variable Importance
# > <chr>         <dbl>
# > 1 MATHEFF      0.400 
# > 2 ESCS         0.0513
# > 3 FAMSUPSL     0.0497
# > 4 ANXMAT       0.0402
# > 5 WORKPAY      0.0299
# > 6 FAMCON       0.0279

tibble(
  Variable   = names(main_importance),
  Importance = main_importance
) |>
  arrange(desc(Importance)) |>
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    x = "Variable",
    y = "Absolute Importance (Gain)"
  ) +
  theme_minimal(base_size = 14)

# --- Estimates of Manual weighted R² (XGBoost models on training data) ---
dtrain_pred_only <- xgb.DMatrix(data = X_train)  # for prediction only
main_r2s_weighted <- sapply(1:M, function(i) {
  pv    <- pvmaths[i]
  model <- main_models[[i]]$mod
  y     <- train_data[[pv]]
  w     <- train_data[[final_wt]]
  y_pred<- predict(model, dtrain_pred_only)
  y_bar <- sum(w * y) / sum(w)
  sse  <- sum(w * (y - y_pred)^2)
  sst  <- sum(w * (y - y_bar)^2)
  1 - sse / sst
})
main_r2_weighted <- mean(main_r2s_weighted)
main_r2_weighted
# > [1] 0.7415924

# --- All five metrics on TRAIN for each PV ---
main_metrics <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y     <- train_data[[pvmaths[i]]]
  w     <- train_data[[final_wt]]
  y_pred <- predict(model, dtrain_pred_only)
  compute_metrics(y, y_pred, w)
}) |> t() |> as.data.frame()
main_metrics
# >         mse     rmse      mae         bias bias_pct        r2
# > 1  2115.305 45.99244 36.03781 -0.005572079 1.144270 0.7376843
# > 2  2033.709 45.09666 35.33106 -0.005799341 1.117773 0.7408180
# > 3  2051.483 45.29330 35.70930 -0.004949265 1.123391 0.7417481
# > 4  2025.573 45.00636 35.28998 -0.004475151 1.091546 0.7396585
# > 5  2087.444 45.68855 35.80580 -0.005448221 1.113790 0.7321170
# > 6  1996.794 44.68550 34.94763 -0.005342161 1.090493 0.7431311
# > 7  1991.805 44.62964 34.91524 -0.004997830 1.086542 0.7428319
# > 8  1919.628 43.81356 34.28401 -0.004833498 1.076101 0.7570260
# > 9  2098.267 45.80685 35.71022 -0.005159242 1.136701 0.7370247
# > 10 2012.792 44.86415 35.18491 -0.005132335 1.095051 0.7438839

### ---- Replicate models using BRR replicate weights ----

set.seed(123)

tic("Fitting replicate models")
replicate_models <- lapply(pvmaths, function(pv) {
  lapply(rep_wts, function(w) {
    y_train <- train_data[[pv]]
    y_valid <- valid_data[[pv]]
    w_train    <- train_data[[w]]
    w_valid   <- valid_data[[w]]
    
    dtrain <- xgb.DMatrix(data = X_train, label = y_train, weight = w_train)
    dvalid <- xgb.DMatrix(data = X_valid, label = y_valid, weight = w_valid)
    
    mod <- xgb.train(
      params = list(
        objective   = "reg:squarederror",
        max_depth   = best_params$max_depth,
        eta         = best_params$eta,
        eval_metric = "rmse",
        nthread     = max(1, parallel::detectCores() - 1)
      ),
      data  = dtrain,
      nrounds = best_iter,
      watchlist = list(train = dtrain, valid = dvalid),
      verbose = 1,
      early_stopping_rounds = NULL
    )
    
    list(
      mod            = mod,
      importance_raw = xgb.importance(model = mod, trees = 0:(best_iter - 1))
    )
  })
})
toc()
# > Fitting replicate models: 3199.628 sec elapsed

# --- Gain-Based Variable Importance Array for Replicates: (M × G × p over voi_all) ---
rep_importance_array <- array(NA_real_, dim = c(M, G, length(feature_order)),
                              dimnames = list(pvmaths, rep_wts, feature_order))

for (m in 1:M) {
  for (g in 1:G) {
    imp_raw <- replicate_models[[m]][[g]]$importance_raw
    gain_map <- collapse_importance_to_voi(imp_raw)  # <- imp_agg: aggregated importance
    vec <- gain_map[feature_order]
    vec[is.na(vec)] <- 0
    rep_importance_array[m, g, ] <- vec
  }
}

# checks
stopifnot(all(dim(rep_importance_array) == c(M, G, length(feature_order))))
stopifnot(!anyNA(rep_importance_array))
if (any(abs(apply(rep_importance_array, c(1, 2), sum) - 1) > 1e-6))
  warning("Some (PV, replicate) gain rows do not sum to ~1.")

# --- Weighted R² across (M × G) on TRAIN ---
rep_r2_weighted <- matrix(NA_real_, nrow = G, ncol = M,
                          dimnames = list(rep_wts, pvmaths))
for (m in 1:M) {
  pv <- pvmaths[m]
  y  <- train_data[[pv]]
  for (g in 1:G) {
    model <- replicate_models[[m]][[g]]$mod
    w     <- train_data[[rep_wts[g]]]
    y_pred <- predict(model, dtrain_pred_only)
    y_bar  <- sum(w * y) / sum(w)
    sse   <- sum(w * (y - y_pred)^2)
    sst   <- sum(w * (y - y_bar)^2)
    rep_r2_weighted[g, m] <- 1 - sse / sst
  }
}

### ---- Rubin + BRR for Standard Errors (SEs) ----

# --- Rubin + BRR: Gain-Based Variable Importance over voi_all ---
sampling_var_importance   <- setNames(numeric(length(feature_order)), feature_order)
imputation_var_importance <- setNames(numeric(length(feature_order)), feature_order)
var_final_importance      <- setNames(numeric(length(feature_order)), feature_order)
se_final_importance       <- setNames(numeric(length(feature_order)), feature_order)
cv_final_importance       <- setNames(numeric(length(feature_order)), feature_order)

for (var in feature_order) {
  rep_vals <- t(rep_importance_array[, , var])  # G × M
  sampling_var_importance[var] <- mean(sapply(1:M, function(m) {
    sum((rep_vals[, m] - main_importance_matrix[m, var])^2) / (G * (1 - k)^2)
  }))
  imputation_var_importance[var] <- sum((main_importance_matrix[, var] - main_importance[var])^2) / (M - 1)
  var_final_importance[var] <- sampling_var_importance[var] + (1 + 1/M) * imputation_var_importance[var]
  se_final_importance[var]  <- sqrt(var_final_importance[var])
  cv_final_importance[var]  <- ifelse(main_importance[var] != 0, se_final_importance[var] / main_importance[var], NA_real_)
}

# --- Rubin + BRR: Weighted R² (TRAIN) ---
sampling_var_r2s_weighted <- sapply(1:M, function(m) {
  sum((rep_r2_weighted[, m] - main_r2s_weighted[m])^2) / (G * (1 - k)^2)
})
sampling_var_r2_weighted <- mean(sampling_var_r2s_weighted)
imputation_var_r2_weighted <- sum((main_r2s_weighted - main_r2_weighted)^2) / (M - 1)
var_final_r2_weighted <- sampling_var_r2_weighted + (1 + 1/M) * imputation_var_r2_weighted
se_final_r2_weighted  <- sqrt(var_final_r2_weighted)

#### ---- Final Outputs ----

# --- Variable Importance Table (mean ± SE) over all VOIs (numeric + categorical parents) ---
importance_table <- tibble::tibble(
  Variable     = feature_order,
  Importance   = main_importance[feature_order],
  `Std. Error` = se_final_importance[feature_order],
  `CV`         = cv_final_importance[feature_order]
) |> dplyr::arrange(dplyr::desc(Importance))

# --- R-squared Output Table (Weighted, TRAIN) ---
r2_weighted_table <- tibble::tibble(
  Metric       = "R-squared (Weighted)",
  Estimate     = mean(main_r2s_weighted),
  `Std. Error` = se_final_r2_weighted
)

print(as.data.frame(r2_weighted_table), row.names = FALSE)
# >               Metric  Estimate Std. Error
# > R-squared (Weighted) 0.7415924 0.03927326

as.data.frame(importance_table) %>% 
  head() %>% 
  print(row.names = FALSE)
# > Variable Importance  Std. Error         CV
# >  MATHEFF 0.40006178 0.027759703 0.06938854
# >     ESCS 0.05128339 0.009832926 0.19173707
# > FAMSUPSL 0.04974540 0.008690705 0.17470368
# >   ANXMAT 0.04021190 0.008610660 0.21413214
# >  WORKPAY 0.02989222 0.007514243 0.25137789
# >   FAMCON 0.02790364 0.007393697 0.26497246
as.data.frame(importance_table) %>% 
  #head() %>%                  # show full
  print(row.names = FALSE)
# >  Variable   Importance   Std. Error         CV
# >   MATHEFF 4.000618e-01 0.0277597033 0.06938854
# >      ESCS 5.128339e-02 0.0098329259 0.19173707
# >  FAMSUPSL 4.974540e-02 0.0086907054 0.17470368
# >    ANXMAT 4.021190e-02 0.0086106602 0.21413214
# >   WORKPAY 2.989222e-02 0.0075142428 0.25137789
# >    FAMCON 2.790364e-02 0.0073936966 0.26497246
# >  EXERPRAC 2.716011e-02 0.0077226803 0.28433910
# >   HOMEPOS 2.089784e-02 0.0072143274 0.34521877
# >  C REATAS 1.807007e-02 0.0075119989 0.41571494
# >  MATHEF21 1.633296e-02 0.0083264577 0.50979487
# >  CREATEFF 1.567345e-02 0.0054460706 0.34747101
# >  CREATOOS 1.558863e-02 0.0075369869 0.48349264
# >  CURIOAGR 1.200460e-02 0.0049189644 0.40975647
# >   LEARRES 1.133266e-02 0.0046705683 0.41213336
# >   SCHSUST 1.117195e-02 0.0050753692 0.45429592
# >  INFOSEEK 1.099196e-02 0.0043920443 0.39956874
# >   GROSAGR 1.073957e-02 0.0047059293 0.43818585
# >  EMOCOAGR 1.028837e-02 0.0054752433 0.53217798
# >  PROBSELF 1.000709e-02 0.0050846145 0.50810138
# >    ICTRES 9.981093e-03 0.0034592431 0.34657957
# >   OPENCUL 9.976758e-03 0.0049723174 0.49839007
# >  CREATSCH 9.610063e-03 0.0037125464 0.38631862
# >   BULLIED 9.501648e-03 0.0039749352 0.41834164
# >   CREATOP 9.228033e-03 0.0038493853 0.41714039
# >  COGACMCO 8.984963e-03 0.0046677913 0.51951147
# > ST004D01T 7.982069e-03 0.0039275737 0.49204956
# >   FEELLAH 7.943897e-03 0.0042635640 0.53670935
# >  CREATFAM 7.658262e-03 0.0044287070 0.57829141
# >  ASSERAGR 7.586382e-03 0.0034926476 0.46038380
# >  WORKHOME 7.196881e-03 0.0040735177 0.56601157
# >   DIGPREP 6.720723e-03 0.0037187231 0.55332188
# >    FAMSUP 6.617519e-03 0.0039904182 0.60300815
# >    EXPOFA 6.586581e-03 0.0034299118 0.52074234
# >  COGACRCO 6.519952e-03 0.0036916622 0.56621007
# > PERSEVAGR 6.289785e-03 0.0041065848 0.65289747
# >    BELONG 6.170409e-03 0.0028388840 0.46008037
# >    REGION 6.129063e-03 0.0030174864 0.49232423
# >   IMAGINE 5.557377e-03 0.0034020648 0.61217091
# >  MATHPERS 5.535891e-03 0.0038200561 0.69005261
# >   COOPAGR 5.427823e-03 0.0029843155 0.54981810
# >  EXPO21ST 5.380942e-03 0.0037518659 0.69725072
# >  STRESAGR 5.079604e-03 0.0026896277 0.52949551
# >  STUDYHMW 5.022705e-03 0.0043249476 0.86107940
# >     LANGN 4.595496e-03 0.0031710774 0.69004028
# >  EMPATAGR 4.507072e-03 0.0036482402 0.80944792
# >  CREENVSC 4.336859e-03 0.0031251873 0.72061075
# >   DISCLIM 4.229946e-03 0.0025114997 0.59374272
# >   OPENART 3.711441e-03 0.0030700197 0.82717734
# >    MACTIV 3.262266e-03 0.0030162531 0.92458825
# >    SDLEFF 3.057313e-03 0.0023320602 0.76278088
# >  FEELSAFE 3.037100e-03 0.0024638890 0.81126372
# >     IMMIG 1.901431e-03 0.0020667837 1.08696223
# >   MTTRAIN 1.809952e-03 0.0018390935 1.01610067
# >  SCHLTYPE 1.780431e-03 0.0024098232 1.35350551
# >  MATHPREF 6.748619e-04 0.0012978789 1.92317698
# >   ABGMATH 6.486356e-04 0.0012040108 1.85622046
# >  MATHEASE 3.468697e-04 0.0008421257 2.42778686
# >   MATHMOT 5.430098e-05 0.0003072701 5.65864695

# Extract the smallest set of top-ranked variables whose cumulative importance (Gain) reaches at least 90% of the total
cum_gain90 <- importance_table %>%
  arrange(desc(Importance)) %>%
  mutate(cum = cumsum(Importance)) %>%
  { k <- match(TRUE, .$cum >= 0.90)
  .$Variable[seq_len(ifelse(is.na(k), nrow(.), k))]
  } 
length(cum_gain90)
# > [1] 33
cum_gain90
# >  [1] "MATHEFF"   "ESCS"      "FAMSUPSL"  "ANXMAT"    "WORKPAY"   "FAMCON"    "EXERPRAC"  "HOMEPOS"   "CREATAS"   "MATHEF21"  "CREATEFF"  "CREATOOS"  "CURIOAGR"  "LEARRES"   "SCHSUST"   "INFOSEEK"  "GROSAGR"   "EMOCOAGR"  "PROBSELF" 
# > [20] "ICTRES"    "OPENCUL"   "CREATSCH"  "BULLIED"   "CREATOP"   "COGACMCO"  "ST004D01T" "FEELLAH"   "CREATFAM"  "ASSERAGR"  "WORKHOME"  "DIGPREP"   "FAMSUP"    "EXPOFA" 


# --- Plot: Importance with CV gradient ---
ggplot(importance_table, aes(x = reorder(Variable, Importance), y = Importance, fill = CV)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  labs(title = "XGBoost Variable Importance (Gain) — Coefficient of Variation",  # Coefficient of Variation = SE/Importance: A small CV (e.g., 0.07 for MATHEFF) means the estimate is precise relative to its magnitude.
       x = "Variable", y = "Mean Importance", fill = "CV") +
  theme_minimal()

# --- Plot: Importance with SE (simpler version)---
ggplot(importance_table, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(aes(ymin = pmax(0, Importance - `Std. Error`),
                    ymax = pmin(1, Importance + `Std. Error`)), width = 0.2) +
  coord_flip() +
  labs(title = "XGBoost Variable Importance (Gain) with Standard Errors",
       y = "Importance (Mean ± SE)", x = "Variable") +
  theme_minimal()

# --- Plot: XGBoost variable importance (Gain) with 95% confidence intervals (another version)---
# Bars show mean importance across plausible values; error bars reflect
# Rubin + BRR–based standard errors. Variables are ordered by mean importance.
ggplot(
  importance_table %>%
    mutate(
      lo95 = pmax(0, Importance - z_crit * `Std. Error`),
      hi95 = pmin(1, Importance + `Std. Error` * z_crit),
      Variable = fct_reorder(Variable, Importance)
    ) %>%
    arrange(desc(Importance)),
  aes(x = Variable, y = Importance)
) +
  geom_col(fill = "steelblue", width = 0.75) +
  geom_errorbar(aes(ymin = lo95, ymax = hi95), width = 0.25, linewidth = 0.4) +
  coord_flip() +
  labs(
    title = "XGBoost Variable Importance (Gain) with 95% CIs",
    x = NULL, y = "Importance (mean ± 1.96×SE)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


### ---- Predict and Evaluate Performance on Training Data ----

# Main model predictions on TRAIN
train_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y     <- train_data[[pvmaths[i]]]
  w     <- train_data[[final_wt]]
  y_pred <- predict(model, dtrain_pred_only)
  compute_metrics(y, y_pred, w)
}) |> t() |> as.data.frame()

# Replicate predictions on TRAIN
tic("Computing train_metrics_replicates")
train_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y     <- train_data[[pvmaths[m]]]
    w     <- train_data[[rep_wts[g]]]
    y_pred <- predict(model, dtrain_pred_only)
    compute_metrics(y, y_pred, w)
  }) |> t()
})
toc()
# > Computing train_metrics_replicates: 1.365 sec elapsed

# Combine BRR + Rubin on TRAIN
sampling_var_matrix_train <- sapply(1:M, function(m) {
  sweep(train_metrics_replicates[[m]], 2, unlist(train_metrics_main[m, ]))^2 |>
    colSums() / (G * (1 - k)^2)
})
sampling_var_train   <- rowMeans(sampling_var_matrix_train)
train_metric_main    <- colMeans(train_metrics_main)
imputation_var_train <- colSums((train_metrics_main - matrix(train_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
var_final_train      <- sampling_var_train + (1 + 1/M) * imputation_var_train
se_final_train       <- sqrt(var_final_train)

ci_lower_train  <- train_metric_main - z_crit * se_final_train
ci_upper_train  <- train_metric_main + z_crit * se_final_train
ci_length_train <- ci_upper_train - ci_lower_train

train_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(train_metric_main, scientific = FALSE),
  Standard_error = format(se_final_train, scientific = FALSE),
  CI_lower       = format(ci_lower_train, scientific = FALSE),
  CI_upper       = format(ci_upper_train, scientific = FALSE),
  CI_length      = format(ci_length_train, scientific = FALSE)
)
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error       CI_lower       CI_upper      CI_length
# >       MSE 2033.279920359 315.2911449237 1415.320631664 2651.239209053 1235.918577390
# >      RMSE   45.087701759   3.5590110870   38.112168208   52.063235310   13.951067102
# >       MAE   35.321594976   3.6102765552   28.245582954   42.397606999   14.152024045
# >      Bias   -0.005170912   0.0009470752   -0.007027146   -0.003314679    0.003712466
# >     Bias%    1.107565814   0.1288618291    0.855001270    1.360130358    0.505129088
# > R-squared    0.741592361   0.0392732646    0.664618176    0.818566545    0.153948368

### ---- Predict and Evaluate Performance on Validation Data ----

dvalid_pred_only <- xgb.DMatrix(data = X_valid)

# Main (VALID)
valid_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y     <- valid_data[[pvmaths[i]]]
  w     <- valid_data[[final_wt]]
  y_pred <- predict(model, dvalid_pred_only)
  compute_metrics(y, y_pred, w)
}) |> t() |> as.data.frame()

# Replicates (VALID)
tic("Computing valid_metrics_replicates")
valid_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y     <- valid_data[[pvmaths[m]]]
    w     <- valid_data[[rep_wts[g]]]
    y_pred <- predict(model, dvalid_pred_only)
    compute_metrics(y, y_pred, w)
  }) |> t()
})
toc()
# > Computing valid_metrics_replicates: 2.021 sec elapsed

# BRR + Rubin (VALID)
sampling_var_matrix_valid <- sapply(1:M, function(m) {
  sweep(valid_metrics_replicates[[m]], 2, unlist(valid_metrics_main[m, ]))^2 |>
    colSums() / (G * (1 - k)^2)
})
sampling_var_valid   <- rowMeans(sampling_var_matrix_valid)
valid_metric_main    <- colMeans(valid_metrics_main)
imputation_var_valid <- colSums((valid_metrics_main - matrix(valid_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
var_final_valid      <- sampling_var_valid + (1 + 1/M) * imputation_var_valid
se_final_valid       <- sqrt(var_final_valid)

ci_lower_valid  <- valid_metric_main - z_crit * se_final_valid
ci_upper_valid  <- valid_metric_main + z_crit * se_final_valid
ci_length_valid <- ci_upper_valid - ci_lower_valid

valid_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(valid_metric_main, scientific = FALSE),
  Standard_error = format(se_final_valid, scientific = FALSE),
  CI_lower       = format(ci_lower_valid, scientific = FALSE),
  CI_upper       = format(ci_upper_valid, scientific = FALSE),
  CI_length      = format(ci_length_valid, scientific = FALSE)
)
print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper   CI_length
# >       MSE   4363.7369827   431.82064637 3517.3840680 5210.0898973 1692.705829
# >      RMSE     66.0501718     3.25433877   59.6717850   72.4285586   12.756774
# >       MAE     51.4521882     2.86752554   45.8319414   57.0724350   11.240494
# >      Bias     -1.6269804     4.36448091  -10.1812058    6.9272450   17.108451
# >     Bias%      1.5268215     0.87713988   -0.1923411    3.2459840    3.438325
# > R-squared      0.4916461     0.04261125    0.4081296    0.5751626    0.167033

### ---- Predict and Evaluate Performance on Test Data ----

dtest_pred_only <- xgb.DMatrix(data = X_test)

# Main (TEST)
test_metrics_main <- sapply(1:M, function(i) {
  model <- main_models[[i]]$mod
  y     <- test_data[[pvmaths[i]]]
  w     <- test_data[[final_wt]]
  y_pred <- predict(model, dtest_pred_only)
  compute_metrics(y, y_pred, w)
}) |> t() |> as.data.frame()

# Replicates (TEST)
tic("Computing test_metrics_replicates")
test_metrics_replicates <- lapply(1:M, function(m) {
  sapply(1:G, function(g) {
    model <- replicate_models[[m]][[g]]$mod
    y     <- test_data[[pvmaths[m]]]
    w     <- test_data[[rep_wts[g]]]
    y_pred <- predict(model, dtest_pred_only)
    compute_metrics(y, y_pred, w)
  }) |> t()
})
toc()
# > Computing test_metrics_replicates: 1.918 sec elapsed

# BRR + Rubin (TEST)
sampling_var_matrix_test <- sapply(1:M, function(m) {
  sweep(test_metrics_replicates[[m]], 2, unlist(test_metrics_main[m, ]))^2 |>
    colSums() / (G * (1 - k)^2)
})
sampling_var_test   <- rowMeans(sampling_var_matrix_test)
test_metric_main    <- colMeans(test_metrics_main)
imputation_var_test <- colSums((test_metrics_main - matrix(test_metric_main, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
var_final_test      <- sampling_var_test + (1 + 1/M) * imputation_var_test
se_final_test       <- sqrt(var_final_test)

ci_lower_test  <- test_metric_main - z_crit * se_final_test
ci_upper_test  <- test_metric_main + z_crit * se_final_test
ci_length_test <- ci_upper_test - ci_lower_test

test_eval <- tibble::tibble(
  Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
  Point_estimate = format(test_metric_main, scientific = FALSE),
  Standard_error = format(se_final_test, scientific = FALSE),
  CI_lower       = format(ci_lower_test, scientific = FALSE),
  CI_upper       = format(ci_upper_test, scientific = FALSE),
  CI_length      = format(ci_length_test, scientific = FALSE)
)
print(as.data.frame(test_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4806.5991121   560.79796722 3707.4552938 5905.7429305 2198.2876367
# >      RMSE     69.2937642     4.03046980   61.3941886   77.1933399   15.7991513
# >       MAE     55.0039124     3.14685517   48.8361896   61.1716352   12.3354456
# >      Bias     -2.2966304     4.28344282  -10.6920241    6.0987633   16.7907873
# >     Bias%      1.3898401     0.90134897   -0.3767714    3.1564516    3.5332230
# > R-squared      0.4106218     0.05332882    0.3060992    0.5151443    0.2090451

### ---- ** Predictive Performance on the training/validation/test datasets (Weighted, Rubin + BRR) ** ----

# Unified evaluator (reuses precomputed matrices; does NOT rebuild design)
evaluate_split <- function(split_data, X_split, main_models, replicate_models,
                           final_wt, rep_wts, M, G, k, z_crit, pvmaths) {
  
  dmat <- xgb.DMatrix(data = X_split)
  
  # Main PV loop
  main_metrics_df <- sapply(1:M, function(i) {
    model <- main_models[[i]]$mod
    y     <- split_data[[pvmaths[i]]]
    w     <- split_data[[final_wt]]
    y_pred <- predict(model, dmat)
    compute_metrics(y, y_pred, w)
  }) |> t() |> as.data.frame()
  
  main_point <- colMeans(main_metrics_df)
  
  # Replicate loop
  replicate_metrics <- lapply(1:M, function(m) {
    sapply(1:G, function(g) {
      model <- replicate_models[[m]][[g]]$mod
      y     <- split_data[[pvmaths[m]]]
      w     <- split_data[[rep_wts[g]]]
      y_pred <- predict(model, dmat)
      compute_metrics(y, y_pred, w)
    }) |> t()
  })
  
  # BRR sampling variance
  sampling_var_matrix <- sapply(1:M, function(m) {
    sweep(replicate_metrics[[m]], 2, unlist(main_metrics_df[m, ]))^2 |>
      colSums() / (G * (1 - k)^2)
  })
  sampling_var <- rowMeans(sampling_var_matrix)
  
  # Rubin's imputation variance
  imputation_var <- colSums((main_metrics_df - matrix(main_point, nrow = M, ncol = 6, byrow = TRUE))^2) / (M - 1)
  
  # Total variance and confidence intervals
  var_final <- sampling_var + (1 + 1/M) * imputation_var
  se_final  <- sqrt(var_final)
  ci_lower  <- main_point - z_crit * se_final
  ci_upper  <- main_point + z_crit * se_final
  ci_length <- ci_upper - ci_lower
  
  tibble::tibble(
    Metric         = c("MSE", "RMSE", "MAE", "Bias", "Bias%", "R-squared"),
    Point_estimate = format(main_point, scientific = FALSE),
    Standard_error = format(se_final, scientific = FALSE),
    CI_lower       = format(ci_lower, scientific = FALSE),
    CI_upper       = format(ci_upper, scientific = FALSE),
    CI_length      = format(ci_length, scientific = FALSE)
  )
}

# Evaluate for each split using existing matrices
train_eval <- evaluate_split(train_data, X_train, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, pvmaths)
valid_eval <- evaluate_split(valid_data, X_valid, main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, pvmaths)
test_eval  <- evaluate_split(test_data,  X_test,  main_models, replicate_models, final_wt, rep_wts, M, G, k, z_crit, pvmaths)

# Display
print(as.data.frame(train_eval), row.names = FALSE)
print(as.data.frame(valid_eval), row.names = FALSE)
print(as.data.frame(test_eval),  row.names = FALSE)

### ---- Summary ----
print(as.data.frame(train_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error       CI_lower       CI_upper      CI_length
# >       MSE 2033.279920359 315.2911449237 1415.320631664 2651.239209053 1235.918577390
# >      RMSE   45.087701759   3.5590110870   38.112168208   52.063235310   13.951067102
# >       MAE   35.321594976   3.6102765552   28.245582954   42.397606999   14.152024045
# >      Bias   -0.005170912   0.0009470752   -0.007027146   -0.003314679    0.003712466
# >     Bias%    1.107565814   0.1288618291    0.855001270    1.360130358    0.505129088
# > R-squared    0.741592361   0.0392732646    0.664618176    0.818566545    0.153948368

print(as.data.frame(valid_eval), row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper   CI_length
# >       MSE   4363.7369827   431.82064637 3517.3840680 5210.0898973 1692.705829
# >      RMSE     66.0501718     3.25433877   59.6717850   72.4285586   12.756774
# >       MAE     51.4521882     2.86752554   45.8319414   57.0724350   11.240494
# >      Bias     -1.6269804     4.36448091  -10.1812058    6.9272450   17.108451
# >     Bias%      1.5268215     0.87713988   -0.1923411    3.2459840    3.438325
# > R-squared      0.4916461     0.04261125    0.4081296    0.5751626    0.167033

print(as.data.frame(test_eval),  row.names = FALSE)
# >    Metric Point_estimate Standard_error     CI_lower     CI_upper    CI_length
# >       MSE   4806.5991121   560.79796722 3707.4552938 5905.7429305 2198.2876367
# >      RMSE     69.2937642     4.03046980   61.3941886   77.1933399   15.7991513
# >       MAE     55.0039124     3.14685517   48.8361896   61.1716352   12.3354456
# >      Bias     -2.2966304     4.28344282  -10.6920241    6.0987633   16.7907873
# >     Bias%      1.3898401     0.90134897   -0.3767714    3.1564516    3.5332230
# > R-squared      0.4106218     0.05332882    0.3060992    0.5151443    0.2090451

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
# >      RMSE        45.0877         3.5590  38.1122  52.0632   13.9511
# >       MAE        35.3216         3.6103  28.2456  42.3976   14.1520
# >      Bias        -0.0052         0.0009  -0.0070  -0.0033    0.0037
# >     Bias%         1.1076         0.1289   0.8550   1.3601    0.5051
# > R-squared         0.7416         0.0393   0.6646   0.8186    0.1539

valid_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        66.0502         3.2543  59.6718  72.4286   12.7568
# >       MAE        51.4522         2.8675  45.8319  57.0724   11.2405
# >      Bias        -1.6270         4.3645 -10.1812   6.9272   17.1085
# >     Bias%         1.5268         0.8771  -0.1923   3.2460    3.4383
# > R-squared         0.4916         0.0426   0.4081   0.5752    0.1670

test_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 4))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE        69.2938         4.0305  61.3942  77.1933   15.7992
# >       MAE        55.0039         3.1469  48.8362  61.1716   12.3354
# >      Bias        -2.2966         4.2834 -10.6920   6.0988   16.7908
# >     Bias%         1.3898         0.9013  -0.3768   3.1565    3.5332
# > R-squared         0.4106         0.0533   0.3061   0.5151    0.2090

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
# >      RMSE          45.09           3.56    38.11    52.06     13.95
# >       MAE          35.32           3.61    28.25    42.40     14.15
# >      Bias          -0.01           0.00    -0.01    -0.00      0.00
# >     Bias%           1.11           0.13     0.86     1.36      0.51
# > R-squared           0.74           0.04     0.66     0.82      0.15

valid_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          66.05           3.25    59.67    72.43     12.76
# >       MAE          51.45           2.87    45.83    57.07     11.24
# >      Bias          -1.63           4.36   -10.18     6.93     17.11
# >     Bias%           1.53           0.88    -0.19     3.25      3.44
# > R-squared           0.49           0.04     0.41     0.58      0.17

test_eval %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  ) %>%
  as.data.frame() %>%
  print(row.names=FALSE)
# >    Metric Point_estimate Standard_error CI_lower CI_upper CI_length
# >      RMSE          69.29           4.03    61.39    77.19     15.80
# >       MAE          55.00           3.15    48.84    61.17     12.34
# >      Bias          -2.30           4.28   -10.69     6.10     16.79
# >     Bias%           1.39           0.90    -0.38     3.16      3.53
# > R-squared           0.41           0.05     0.31     0.52      0.21

# --- Export to word---
library(officer); library(flextable)

fmt2 <- \(df) df %>%
  filter(Metric != "MSE") %>%   
  mutate(
    across(-Metric, ~ parse_number(.x)),
    across(where(is.numeric), ~ formatC(.x, format = "f", digits = 2))
  )

performance_tables_xgboost_voi_all_v2p1<- read_docx() %>%
  body_add_par("Performance - Extreme Gradient Boosting (xgboost, voi_all, v2.1)", style = "Normal") %>%  
  body_add_par("Training performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(train_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Validation performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(valid_eval)) %>% align(j = 1:6, align = "right", part = "all") %>% autofit()) %>%
  body_add_par("Test performance", style = "Normal") %>%
  body_add_flextable(flextable(fmt2(test_eval))  %>% align(j = 1:6, align = "right", part = "all") %>% autofit())

print(performance_tables_xgboost_voi_all_v2p1, target = "performance_tables_xgboost_voi_all_v2p1.docx")

