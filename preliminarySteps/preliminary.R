# ---- Script Description ----
#
# Purpose:
#   Prepare the PISA 2022 Canada student-level dataset for analysis by:
#     1. Loading the raw international SPSS data file and filtering records for Canada only.
#     2. Extracting and saving variable metadata (names, labels, value labels, data types).
#     3. Inspecting variable attributes and structures to understand coding and formats.
#     4. Summarizing missing values across all variables.
#
# Data Sources:
#   - Raw data: CY08MSP_STU_QQQ.SAV (PISA 2022 student-level, international version, SPSS format).
#   - Filtered dataset: CY08MSP_STU_QQQ_CAN.SAV (Canada-only records).
#   - Metadata: metadata_student.csv (extracted variable names, labels, formats, and value labels).
#
# Key Notes:
#   - Initial attempt to process SAS (.sas7bdat) files in R was not recommended due to memory constraints
#     and format limitations; SPSS (.sav) format is used instead.
#   - Metadata extraction uses a custom function (`extract_metadata`) with `haven` to capture variable-level
#     attributes for documentation and analysis.
#   - Missing value analysis identifies variables with 100% missingness and ranks all variables by missing percentage.

# ---- I. Data Preparation ---- 

## ---- SAS Files ----

# # Set working directory
# setwd("~/projects/pisa")

# # Load required libraries
# library(haven)      # for reading/writing .sas7bdat files
# library(dplyr)      # for data manipulation
# 
# # Read the full dataset 
# pisa_2022_student <- read_sas("data/pisa2022/SAS/STU_QQQ_SAS/CY08MSP_STU_QQQ.SAS7BDAT") 
# # > Error: vector memory limit of 16.0 Gb reached, see mem.maxVSize()
# 
# # Filter for Canada only
# pisa_2022_student_canada <- pisa_2022_student %>% filter(CNT == "CAN")
# 
# # Save the filtered dataset to new .sas7bdat file
# write_sas(pisa_2022_student_canada, "data/pisa2022/SAS/STU_QQQ_SAS/CY08MSP_STU_QQQ_CAN.SAS7BDAT")
# # > Error: Failed to insert value [1, 55]: A provided tag value was outside the range of allowed values in the specified file format.

#  --- Remark ---
#   Not recommended to deal with SAS directly in R

## ---- SPSS Files----- 

# Set working directory
setwd("~/projects/pisa")

# Install required libraries if not already installed

# Load required libraries
library(haven)      # for reading .sav (SPSS) files
library(tidyverse)  # or library(tidyr) + library(dplyr) + library(ggplot2)
library(tibble)     # For tidy table outputs
library(intsvy)     # for PISA and IEA survey analysis (plausible values, BRR, PVs)
library(purrr)      # for functional iteration (already in tidyverse, used in extract_metadata)

# Load PISA 2022 Data: Full student datasets (international version)
#pisa_2022_student <- read_sav("~/projects/pisa/data/pisa2022/CY08MSP_STU_QQQ.SAV")   # 613744 x 1278
#pisa_2022_school <- read_sav("~/projects/pisa/data/pisa2022/CY08MSP_SCH_QQQ.SAV")    # 21629 x 431

# Filter datasets for Canada only and save for local use
#pisa_2022_student_canada <- pisa_2022_student %>% filter(CNT == "CAN")
#write_sav(pisa_2022_student_canada, "data/pisa2022/CY08MSP_STU_QQQ_CAN.sav")
#pisa_2022_school_canada <- pisa_2022_school %>% filter(CNT == "CAN")
#write_sav(pisa_2022_school_canada, "data/pisa2022/CY08MSP_SCH_QQQ_CAN.sav")
# Remark: alternative, use SPSS Statistics for data preparation.

# Load PISA 2022 Data: Canada
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE)  # 23073 x 1278
pisa_2022_school_canada <- read_sav("data/pisa2022/CY08MSP_SCH_QQQ_CAN.SAV", user_na = TRUE)   # 863 x 431

# # View data structure 
# str(pisa_2022_student_canada)
# str(pisa_2022_school_canada)

# ---- II. Metadata ----
## ---- Extract and Save Metadata for PISA Canada Datasets ----

# library(dplyr)
# library(tibble)

# # Function: Extract variable name, label, value labels, and class type
# extract_metadata <- function(data) {
#   tibble::tibble(
#     variable = names(data),
#     label = sapply(data, function(x) attr(x, "label") %||% NA_character_),
#     value_labels = sapply(data, function(x) {
#       lbls <- attr(x, "labels")
#       if (is.null(lbls)) return(NA_character_)
#       paste(paste0(lbls, " = ", names(lbls)), collapse = "; ")
#     }),
#     type = sapply(data, function(x) paste(class(x), collapse = ", "))
#   )
# }
# 
# # Save metadata (only if not already saved)
# student_meta_path <- "data/pisa2022/metadata_student.csv"
# school_meta_path <- "data/pisa2022/metadata_school.csv"
# 
# if (!file.exists(student_meta_path)) {
#   write_csv(extract_metadata(pisa_2022_student_canada), student_meta_path)
# }
# 
# if (!file.exists(school_meta_path)) {
#   write_csv(extract_metadata(pisa_2022_school_canada), school_meta_path)
# }

## ---- Inspect Metadata ----

# Function: Load saved metadata (student or school)
load_metadata <- function(type = c("student", "school")) {
  type <- match.arg(type)
  read_csv(paste0("data/pisa2022/metadata_", type, ".csv"), show_col_types = FALSE) |> tibble::as_tibble()
}

# Load student metadata
metadata_student <- load_metadata("student")
metadata_school <- load_metadata("school")

### ---- Student ----
names(pisa_2022_student_canada)  # Column names in the dataset
unique(sapply(pisa_2022_student_canada, class))       # All unique variable types in the dataset
# > [[1]]
# > [1] "haven_labelled" "vctrs_vctr"     "character"     

# > [[2]]
# > [1] "haven_labelled" "vctrs_vctr"     "double"        

# > [[3]]
# > [1] "numeric"

# > [[4]]
# > [1] "character"

# > [[5]]
# > [1] "haven_labelled_spss" "haven_labelled"      "vctrs_vctr"          "double"  
length(unique(sapply(pisa_2022_student_canada, class)))
# > [1] 5

# --- Example Metadata Access (per variable) ---
attributes(pisa_2022_student_canada$CNT)
attr(pisa_2022_student_canada$CNT, "label")           # Variable label: Human-readable description
attr(pisa_2022_student_canada$CNT, "labels")          # Value labels: Coded values and their descriptions (named vector)
attr(pisa_2022_student_canada$CNT, "format.spss")     # SPSS display format (e.g., "F2.0", "A10")
attr(pisa_2022_student_canada$CNT, "display_width")   # Display width (SPSS-specific)
attr(pisa_2022_student_canada$CNT, "na_values")
attr(pisa_2022_student_canada$CNT, "na_range")
attr(pisa_2022_student_canada$CNT, "is_na")

attributes(pisa_2022_student_canada$ST004D01T)
attr(pisa_2022_student_canada$ST004D01T, "label")           # Variable label: Human-readable description
attr(pisa_2022_student_canada$ST004D01T, "labels")          # Value labels: Coded values and their descriptions (named vector)
attr(pisa_2022_student_canada$ST004D01T, "format.spss")     # SPSS display format (e.g., "F2.0", "A10")
attr(pisa_2022_student_canada$ST004D01T, "display_width")   # Display width (SPSS-specific)
attr(pisa_2022_student_canada$ST004D01T, "na_values")
attr(pisa_2022_student_canada$ST004D01T, "na_range")
attr(pisa_2022_student_canada$ST004D01T, "is_na")

table(pisa_2022_student_canada$ST251Q07JA, useNA = "ifany")
# >    1     2     3     4    99  <NA> 
# > 3450  1921  2209 13894   273  1326
table(pisa_2022_student_canada$ST251Q07JA)
# >    1     2     3     4    99 
# > 3450  1921  2209 13894   273 
sum(is.na(pisa_2022_student_canada$ST251Q07JA))
# > 1599

var_classes <- sapply(pisa_2022_student_canada, function(x) paste(class(x), collapse = ", "))
table(var_classes)
# >                                               character 
# >                                                       2
# > haven_labelled_spss, haven_labelled, vctrs_vctr, double 
# >                                                    1046
# >                   haven_labelled, vctrs_vctr, character
# >                                                      16 
# >                      haven_labelled, vctrs_vctr, double 
# >                                                      19 
# >                                                 numeric 
# >                                                     195

names(var_classes[var_classes == "character"])
# > [1] "CYC"     "VER_DAT"
names(var_classes[var_classes == "haven_labelled_spss, haven_labelled, vctrs_vctr, double"])
# > [1] "ST003D02T"  "ST003D03T"  "ST004D01T"  "ST250Q01JA" "ST250Q02JA" "ST250Q03JA" "ST250Q04JA" "ST250Q05JA"
# > [9] "ST251Q01JA" "ST251Q02JA" "ST251Q03JA" "ST251Q04JA" "ST251Q06JA" "ST251Q07JA" "ST253Q01JA" "ST254Q01JA"
# > ... ...
names(var_classes[var_classes == "haven_labelled, vctrs_vctr, character"])
# > [1] "CNT"        "NatCen"     "STRATUM"    "SUBNATIO"   "ST250D06JA" "ST250D07JA" "ST251D08JA" "ST251D09JA"
# > [9] "ST330D10WA" "OCOD1"      "OCOD2"      "OCOD3"      "PROGN"      "COBN_S"     "COBN_M"     "COBN_F"  
names(var_classes[var_classes == "haven_labelled, vctrs_vctr, double"])
# > [1] "CNTRYID"      "REGION"       "OECD"         "ADMINMODE"    "LANGTEST_QQQ" "LANGTEST_COG" "LANGTEST_PAQ"
# > [8] "Option_CT"    "Option_FL"    "Option_ICTQ"  "Option_WBQ"   "Option_PQ"    "Option_TQ"    "Option_UH"   
# > [15] "BOOKID"       "ST001D01T"    "ISCEDP"       "LANGN"        "UNIT"  
names(var_classes[var_classes == "numeric"])
# > [1] "CNTSCHID"    "CNTSTUID"    "W_FSTUWT"    "W_FSTURWT1"  "W_FSTURWT2"  "W_FSTURWT3"  "W_FSTURWT4" 
# > [8] "W_FSTURWT5"  "W_FSTURWT6"  "W_FSTURWT7"  "W_FSTURWT8"  "W_FSTURWT9"  "W_FSTURWT10" "W_FSTURWT11"
# > ... ...


### ---- School ----
names(pisa_2022_school_canada)  # Column names in the dataset
unique(sapply(pisa_2022_school_canada, class))       # All unique variable types in the dataset
# > [[1]]
# > [1] "haven_labelled" "vctrs_vctr"     "character"     

# > [[2]]
# > [1] "haven_labelled" "vctrs_vctr"     "double"        

# > [[3]]
# > [1] "numeric"

# > [[4]]
# > [1] "character"

length(unique(sapply(pisa_2022_school_canada, class)))
# > [1] 4

# --- Example Metadata Access (per variable) ---
attributes(pisa_2022_school_canada$CNT)
attr(pisa_2022_school_canada$CNT, "label")           # Variable label: Human-readable description
attr(pisa_2022_school_canada$CNT, "labels")          # Value labels: Coded values and their descriptions (named vector)
attr(pisa_2022_school_canada$CNT, "format.spss")     # SPSS display format (e.g., "F2.0", "A10")
attr(pisa_2022_school_canada$CNT, "display_width")   # Display width (SPSS-specific)
attr(pisa_2022_school_canada$CNT, "na_values")
attr(pisa_2022_school_canada$CNT, "na_range")
attr(pisa_2022_school_canada$CNT, "is_na")


attributes(pisa_2022_school_canada$SC013Q01TA)
attr(pisa_2022_school_canada$SC013Q01TA, "label")           # Variable label: Human-readable description
attr(pisa_2022_school_canada$SC013Q01TA, "labels")          # Value labels: Coded values and their descriptions (named vector)
attr(pisa_2022_school_canada$SC013Q01TA, "format.spss")     # SPSS display format (e.g., "F2.0", "A10")
attr(pisa_2022_school_canada$SC013Q01TA, "display_width")   # Display width (SPSS-specific)
attr(pisa_2022_school_canada$SC013Q01TA, "na_values")
attr(pisa_2022_school_canada$SC013Q01TA, "na_range")
attr(pisa_2022_school_canada$SC013Q01TA, "is_na")

table(pisa_2022_school_canada$SC013Q01TA, useNA = "ifany")
# >   1    2 <NA> 
# > 750   56   57 
table(pisa_2022_school_canada$SC013Q01TA)
# >   1   2 
# > 750  56
sum(is.na(pisa_2022_school_canada$SC013Q01TA))
# > 57

var_classes <- sapply(pisa_2022_school_canada, function(x) paste(class(x), collapse = ", "))
table(var_classes)
# > var_classes
# >                             character 
# >                                     3 
# > haven_labelled, vctrs_vctr, character 
# >                                     5 
# >    haven_labelled, vctrs_vctr, double 
# >                                   418 
# >                               numeric 
# >                                     5 


names(var_classes[var_classes == "character"])
# > [1] "CYC"        "PRIVATESCH" "VER_DAT" 
names(var_classes[var_classes == "haven_labelled, vctrs_vctr, character"])
# > [1] "CNT"        "NatCen"     "STRATUM"    "SUBNATIO"   "SC053D11TA" 
names(var_classes[var_classes == "haven_labelled, vctrs_vctr, double"])
# >  [1] "CNTRYID"      "REGION"       "OECD"         "ADMINMODE"    "LANGTEST_QQQ" "SC001Q01TA"   "SC013Q01TA"  
# >  [8] "SC014Q01TA"   "SC016Q01TA"   "SC016Q02TA"   "SC016Q03TA"   "SC016Q04TA"   "SC011Q01TA"   "SC002Q01TA" 
# > [15] "SC002Q02TA"   "SC211Q01JA"   "SC211Q02JA"   "SC211Q03JA"   "SC211Q04JA"   "SC211Q05JA"   "SC211Q06JA" 
# > ......
names(var_classes[var_classes == "numeric"])
# > [1] "CNTSCHID"         "W_SCHGRNRABWT"    "W_FSTUWT_SCH_SUM" "W_FSTUWT_SCH_N"   "SENWT" 


# ---- III. Missing Values ----

# ✓ Columns: Checked for variables with missing values  
# → Rows: Identify rows with a lot of missing entries

## ---- Canada ----

# Check missing values for target variable
colSums(is.na(pisa_2022_student_canada[paste0("PV", 1:10, "MATH")]))
# > PV1MATH  PV2MATH  PV3MATH  PV4MATH  PV5MATH  PV6MATH  PV7MATH  PV8MATH  PV9MATH PV10MATH 
# >       0        0        0        0        0        0        0        0        0        0 
colSums(is.na(pisa_2022_student_canada[paste0("PV", 1:10, "READ")]))
# > PV1READ  PV2READ  PV3READ  PV4READ  PV5READ  PV6READ  PV7READ  PV8READ  PV9READ PV10READ 
# >       0        0        0        0        0        0        0        0        0        0 
colSums(is.na(pisa_2022_student_canada[paste0("PV", 1:10, "SCIE")]))
# > PV1SCIE  PV2SCIE  PV3SCIE  PV4SCIE  PV5SCIE  PV6SCIE  PV7SCIE  PV8SCIE  PV9SCIE PV10SCIE 
# >       0        0        0        0        0        0        0        0        0        0

# Check missing values for weights
sum(is.na(pisa_2022_student_canada$W_FSTUWT))
# > 0

#colSums(is.na(pisa_2022_student_canada[paste0("W_FSTURWT", 1:80)]))
all(colSums(is.na(pisa_2022_student_canada[paste0("W_FSTURWT", 1:80)])) == 0)
# > TRUE

all(
  sum(is.na(pisa_2022_student_canada$W_FSTUWT)) == 0,
  colSums(is.na(pisa_2022_student_canada[paste0("W_FSTURWT", 1:80)])) == 0
)

stopifnot(
  sum(is.na(pisa_2022_student_canada$W_FSTUWT)) == 0,
  all(colSums(is.na(pisa_2022_student_canada[paste0("W_FSTURWT", 1:80)])) == 0)
)

# Count missing values per column in student dataset (Canada only)
missing_counts_student <- colSums(is.na(pisa_2022_student_canada))
missing_counts_school <- colSums(is.na(pisa_2022_school_canada))

# Get number of rows in each dataset
n_student <- nrow(pisa_2022_student_canada)
n_school <- nrow(pisa_2022_school_canada)

# Find variables with 100% missing values
vars_all_missing_student <- names(missing_counts_student[missing_counts_student == n_student])
vars_all_missing_school <- names(missing_counts_school[missing_counts_school == n_school])

# Count how many variables are fully missing
length(vars_all_missing_student)  # 396
length(vars_all_missing_school)   # 48


## ---- Missing Summary ----

### ---- Student ----

# # Create summary of missing values (student)
# missing_summary_student <- tibble(
#   variable = names(missing_counts_student),
#   missing_count = missing_counts_student,
#   missing_percent = round(missing_counts_student / n_student * 100, 2)
# ) %>%
#   arrange(desc(missing_count))
# 
# # View summary of missing values (student)
# missing_summary_student
# 
# # Save student missing summary
# write_csv(missing_summary_student, "data/pisa2022/missing_summary_student.csv")

# Load student missing summary
missing_summary_student <- read_csv("data/pisa2022/missing_summary_student.csv",show_col_types = FALSE)

# Explore student missing summary
sum(missing_summary_student$missing_percent == 100)
# > [1] 396

sum(missing_summary_student$missing_percent <= 50)
# > [1] 605
605/1278
# > [1] 0.4733959

sum(missing_summary_student$missing_percent <= 40)
# > [1] 486
486/1278
# > 0.3802817

sum(missing_summary_student$missing_percent <= 30)
# > [1] 447
447/1278
# > 0.3497653

sum(missing_summary_student$missing_percent <= 25)
# > [1] 417
417/1278
# > 0.3262911

sum(missing_summary_student$missing_percent <= 20)
# > [1] 373
373/1278
# > 0.2918623

sum(missing_summary_student$missing_percent <= 15)
# > [1] 329
329/1278
# > [1] 0.2574335
sum(missing_summary_student$missing_percent <= 10)
# > [1] 274
274/1278
# > [1] 0.2143975

sum(missing_summary_student$missing_percent <= 5)
# > [1] 236
236/1278
# > [1] 0.1846635

sum(missing_summary_student$missing_percent == 0)
# > [1] 234
234/1278
# > [1] 0.1830986

### ---- School ----

# # Create summary of missing values (school)
# missing_summary_school <- tibble(
#   variable = names(missing_counts_school),
#   missing_count = missing_counts_school,
#   missing_percent = round(missing_counts_school / n_school * 100, 2)
# ) %>%
#   arrange(desc(missing_count))
# 
# # View summary of missing values(school)
# missing_summary_school
# 
# # Save school missing summary
# write_csv(missing_summary_school,  "data/pisa2022/missing_summary_school.csv")

# Load school missing summary
missing_summary_school <- read_csv("data/pisa2022/missing_summary_school.csv", show_col_types = FALSE)

# Explore school missing summary
sum(missing_summary_school$missing_percent == 100)
# > [1] 48

sum(missing_summary_school$missing_percent <= 50)
# > [1] 378
378/431
# > [1] 0.8770302

sum(missing_summary_school$missing_percent <= 40)
# > [1] 378
378/431
# >  0.8770302

sum(missing_summary_school$missing_percent <= 30)
# > [1] 378
378/431
# > 0.8770302

sum(missing_summary_school$missing_percent <= 25)
# > [1] 376
376/431
# > 0.8723898

sum(missing_summary_school$missing_percent <= 20)
# > [1] 321
321/431
# > 0.7447796

sum(missing_summary_school$missing_percent <= 15)
# > [1] 298
298/431
# > [1] 0.6914153
sum(missing_summary_school$missing_percent <= 10)
# > [1] 58
58/431
# > [1] 0.1345708

sum(missing_summary_school$missing_percent <= 5)
# > [1] 18
18/431
# > [1] 0.04176334

sum(missing_summary_school$missing_percent == 0)
# > [1] 18
18/431
# > [1] 0.04176334


# ---- IV. Metadata + Missing Summary----

# Student: join and save combined data - metadata + missing summary
metadata_missing_student <- metadata_student %>%
  dplyr::left_join(missing_summary_student, by = "variable") %>%
  dplyr::select(variable, label, value_labels, type, missing_count, missing_percent)

write_csv(
  metadata_missing_student,
  "data/pisa2022/metadata_missing_student.csv"
)

# School: join and save combined data - metadata + missing summary
metadata_missing_school <- metadata_school %>%
  dplyr::left_join(missing_summary_school, by = "variable") %>%
  dplyr::select(variable, label, value_labels, type, missing_count, missing_percent)

write_csv(
  metadata_missing_school,
  "data/pisa2022/metadata_missing_school.csv"
)

# Load combined data
metadata_missing_student <- read_csv("data/pisa2022/metadata_missing_student.csv", show_col_types = FALSE)
metadata_missing_school <- read_csv("data/pisa2022/metadata_missing_school.csv",  show_col_types = FALSE)

