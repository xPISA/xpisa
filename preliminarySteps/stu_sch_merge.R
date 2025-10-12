# ---- Script Description ----
# Purpose: Merge Canada-only PISA 2022 student and school data
# Merge key: CNTSCHID (school ID)
# Notes: 13 common variables; 3 inconsistent across files (LANGTEST_QQQ, SENWT, VER_DAT)
# Strategy:
# - Left join (student ⟵ school) on CNTSCHID; preserves all 23,073 student rows (863 schools).
# - Drop consistent common vars from school: CNT, CNTRYID, CYC, NatCen, STRATUM, SUBNATIO, REGION, OECD, ADMINMODE.
# - Keep both copies of inconsistent vars with suffixes: *_stu (student), *_sch (school).
# Reference: https://www.oecd.org/en/about/programmes/pisa/how-to-prepare-and-analyse-the-pisa-database.html

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


# Load Canada-only PISA 2022 student and school data
pisa_2022_student_canada <- read_sav("data/pisa2022/CY08MSP_STU_QQQ_CAN.SAV", user_na = TRUE) 
pisa_2022_school_canada <- read_sav("data/pisa2022/CY08MSP_SCH_QQQ_CAN.SAV", user_na = TRUE) 

# Check dimensions and class
dim(pisa_2022_student_canada)  # 23,073 × 1,278
class(pisa_2022_student_canada)  # "tbl_df", "tbl", "data.frame"

dim(pisa_2022_school_canada)  # 863 × 431
class(pisa_2022_school_canada)  # "tbl_df", "tbl", "data.frame"

# Identify common variables (TODO: ensure consistency)
intersect(names(pisa_2022_student_canada), names(pisa_2022_school_canada))
# > "CNT"          "CNTRYID"      "CNTSCHID"     "CYC"          "NatCen"       "STRATUM"      "SUBNATIO"  
# > "REGION"       "OECD"         "ADMINMODE"    "LANGTEST_QQQ" "SENWT"        "VER_DAT" 
length(intersect(names(pisa_2022_student_canada), names(pisa_2022_school_canada)))
# > 13

# Check Unique values or Duplicates: `CNTSCHID`: Intl. School ID; `CNTSTUID`: Intl. Student ID.
length(unique(pisa_2022_student_canada$CNTSCHID))                     # 863
length(unique(pisa_2022_school_canada$CNTSCHID))                      # 863
sum(duplicated(pisa_2022_student_canada$CNTSCHID))                    # 22210
sum(duplicated(pisa_2022_school_canada$CNTSCHID))                     # 0

length(unique(pisa_2022_student_canada$CNTSTUID))                     # 23073
# `CNTSTUID` is not in pisa_2022_school_canada
sum(duplicated(pisa_2022_student_canada$CNTSTUID))                    # 0

sum(duplicated(pisa_2022_student_canada[c("CNTSCHID", "CNTSTUID")]))  # 0

# Load metadata CSVs (pre-generated; see pisa2022.R)
metadata_student <- read.csv("data/pisa2022/metadata_student.csv")
metadata_school  <- read.csv("data/pisa2022/metadata_school.csv")


# ---- Consistency Checks Before Merging ----

## Each block below checks one common variable across both datasets.
## Variables flagged with * indicate inconsistencies or require attention.


## ---- CNT: Country Code ----

### Student
class(pisa_2022_student_canada$CNT)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_student_canada$CNT, useNA = "ifany")
# >   CAN 
# > 23073
length(unique(pisa_2022_student_canada$CNT))
# > 1

attr(pisa_2022_student_canada$CNT, "labels")[attr(pisa_2022_student_canada$CNT, "labels") == "CAN"]
# > Canada 
# >  "CAN"

### School
class(pisa_2022_school_canada$CNT)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_school_canada$CNT, useNA = "ifany")
# > CAN 
# > 863
length(unique(pisa_2022_school_canada$CNT))
# > 1

attr(pisa_2022_school_canada$CNT, "labels")[attr(pisa_2022_school_canada$CNT, "labels") == "CAN"]
# > Canada 
# >  "CAN"

### Consistency Check Summary
# All 23,073 students and 863 schools are coded as "CAN" (Canada).
# Each file contains exactly one unique country code.
# No missing or unexpected values are present.
# "CAN" is correctly labeled as "Canada" in both datasets.
# Consistent across files and safe to use as a merge key component.


## ---- CNTRYID: Country Identifier ----

### Student
class(pisa_2022_student_canada$CNTRYID)
# > "haven_labelled" "vctrs_vctr"     "double" 

table(pisa_2022_student_canada$CNTRYID, useNA = "ifany")
# >   124 
# > 23073
length(unique(pisa_2022_student_canada$CNTRYID))
# > 1

attr(pisa_2022_student_canada$CNTRYID, "labels")[attr(pisa_2022_student_canada$CNTRYID, "labels") == 124]
# > Canada 
# >    124

### School
class(pisa_2022_school_canada$CNTRYID)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_school_canada$CNTRYID, useNA = "ifany")
# > 124 
# > 863
length(unique(pisa_2022_school_canada$CNTRYID))
# > 1

attr(pisa_2022_school_canada$CNTRYID, "labels")[attr(pisa_2022_school_canada$CNTRYID, "labels") == 124]
# > Canada 
# >    124

### Consistency Check Summary
# All 23,073 students and 863 schools are coded as "124" (Canada).
# Each file contains exactly one unique country identifier.
# No missing or unexpected values are present.
# "124" is correctly labeled as "Canada" in both datasets.
# Consistent across files and valid for verification or filtering.


## ---- CNTSCHID: Intl. School ID ----

### Student
class(pisa_2022_student_canada$CNTSCHID)
# > "numeric" 

length(unique(pisa_2022_student_canada$CNTSCHID))
# > 863

sum(is.na(pisa_2022_student_canada$CNTSCHID))
# > 0

pisa_2022_student_canada %>%
  group_by(CNTSCHID) %>%
  summarise(
    n_students = n(),
    n_weights      = sum(W_FSTUWT),            # estimated pop. count represented by that school
  ) %>%
  arrange(desc(n_students)) %>%
  as.data.frame() %>%
  head(10) %>%
  print(row.names = FALSE)

### School
class(pisa_2022_school_canada$CNTSCHID)
# > "numeric"  

length(unique(pisa_2022_school_canada$CNTSCHID))
# > 863

sum(is.na(pisa_2022_school_canada$CNTSCHID))
# > 0

# Check if school IDs match between student and school datasets
setequal(
  unique(pisa_2022_student_canada$CNTSCHID),
  unique(pisa_2022_school_canada$CNTSCHID)
)
# > TRUE 

# If FALSE, inspect mismatches:
# setdiff(unique(pisa_2022_student_canada$CNTSCHID), unique(pisa_2022_school_canada$CNTSCHID))
# setdiff(unique(pisa_2022_school_canada$CNTSCHID), unique(pisa_2022_student_canada$CNTSCHID))

### Consistency Check Summary
# All 863 school IDs are numeric in both datasets.
# Each file contains exactly 863 unique values.
# No missing or unexpected values are present.
# All values match exactly between student and school files.
# Consistent across files and safe to use as the merge key.


## ---- CYC: PISA Assessment Cycle (2 digits + 2 character Assessment type - MS/FT)----

### Student
class(pisa_2022_student_canada$CYC)
# > "character" 

table(pisa_2022_student_canada$CYC, useNA = "ifany")
# >  08MS
# > 23073
length(unique(pisa_2022_student_canada$CYC))
# > 1

unique(pisa_2022_student_canada$CYC)
# > "08MS"

### School
class(pisa_2022_school_canada$CYC)
# > "character" 

table(pisa_2022_school_canada$CYC, useNA = "ifany")
# > 08MS
# >  863
length(unique(pisa_2022_school_canada$CYC))
# > 1

unique(pisa_2022_school_canada$CYC)
# > "08MS"

# Check if PISA Assessment Cycle match between student and school datasets
setequal(
  unique(pisa_2022_student_canada$CYC),
  unique(pisa_2022_school_canada$CYC)
)
# > TRUE 

### Consistency Check Summary
# CYC is stored as a character variable in both datasets.
# Each file contains exactly one unique value: "08MS".
# No missing or unexpected values are detected.
# Values match exactly between student and school datasets.
# CYC is consistent and valid across files.


## ---- NatCen: National Centre 6-digit Code ----

### Student
class(pisa_2022_student_canada$NatCen)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_student_canada$NatCen, useNA = "ifany")
# > 012400
# >  23073
length(unique(pisa_2022_student_canada$NatCen))
# > 1

attr(pisa_2022_student_canada$NatCen, "labels")[attr(pisa_2022_student_canada$NatCen, "labels") == "012400"]
# >   Canada 
# > "012400"

### School
class(pisa_2022_school_canada$NatCen)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_school_canada$NatCen, useNA = "ifany")
# > 012400 
# >    863
length(unique(pisa_2022_school_canada$NatCen))
# > 1

attr(pisa_2022_school_canada$NatCen, "labels")[attr(pisa_2022_school_canada$NatCen, "labels") == "012400"]
# >   Canada 
# > "012400"

### Consistency Check Summary
# NatCen is a labeled character variable in both datasets.
# Each file contains exactly one unique value: "012400".
# No missing or unexpected values are present.
# The value "012400" is correctly labeled as "Canada" in both datasets.
# NatCen is consistent and valid across student and school files.


## ---- STRATUM: Stratum ID 5-character (cnt + original stratum ID) ----

### Student
class(pisa_2022_student_canada$STRATUM)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_student_canada$STRATUM, useNA = "ifany")
# > CAN99
# > 23073
length(unique(pisa_2022_student_canada$STRATUM))
# > 1

attr(pisa_2022_student_canada$STRATUM, "labels")[attr(pisa_2022_student_canada$STRATUM, "labels") == "CAN99"]
# > Undisclosed STRATUM - Canada 
# >                      "CAN99"

### School
class(pisa_2022_school_canada$STRATUM)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_school_canada$STRATUM, useNA = "ifany")
# > CAN99
# > 23073
length(unique(pisa_2022_school_canada$STRATUM))
# > 1

attr(pisa_2022_school_canada$STRATUM, "labels")[attr(pisa_2022_school_canada$STRATUM, "labels") == "CAN99"]
# > Undisclosed STRATUM - Canada 
# >                      "CAN99"

### Consistency Check Summary
# STRATUM is a labeled character variable in both datasets.
# Each file contains exactly one unique value: "CAN99".
# No missing or unexpected values are present.
# The value "CAN99" is labeled as "Undisclosed STRATUM - Canada" in both datasets.
# STRATUM is consistent and valid across student and school files.


## ---- SUBNATIO: Adjudicated sub-region code 7-digit code (3-digit country code + region ID + stratum ID) ----

### Student
class(pisa_2022_student_canada$SUBNATIO)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_student_canada$SUBNATIO, useNA = "ifany")
# >  1240000
# >    23073
length(unique(pisa_2022_student_canada$SUBNATIO))
# > 1

attr(pisa_2022_student_canada$SUBNATIO, "labels")[attr(pisa_2022_student_canada$SUBNATIO, "labels") == "1240000"]
# >    Canada 
# > "1240000"

### School
class(pisa_2022_school_canada$SUBNATIO)
# > "haven_labelled" "vctrs_vctr"     "character" 

table(pisa_2022_school_canada$SUBNATIO, useNA = "ifany")
# >  1240000
# >      863
length(unique(pisa_2022_school_canada$SUBNATIO))
# > 1

attr(pisa_2022_school_canada$SUBNATIO, "labels")[attr(pisa_2022_school_canada$SUBNATIO, "labels") == "1240000"]
# >    Canada 
# > "1240000"

### Consistency Check Summary
# SUBNATIO is a labeled character variable in both datasets.
# Each file contains exactly one unique value: "1240000".
# No missing or unexpected values are present.
# The value "1240000" is correctly labeled as "Canada" in both datasets.
# SUBNATIO is consistent and valid across student and school files.


## ---- OECD: OECD country ----

### Student
class(pisa_2022_student_canada$OECD)
# > "haven_labelled" "vctrs_vctr"     "double" 

table(pisa_2022_student_canada$OECD, useNA = "ifany")
# >     1
# > 23073
length(unique(pisa_2022_student_canada$OECD))
# > 1

attr(pisa_2022_student_canada$OECD, "labels")[attr(pisa_2022_student_canada$OECD, "labels") == 1]
# > Yes 
# >   1

### School
class(pisa_2022_school_canada$OECD)
# > "haven_labelled" "vctrs_vctr"     "double" 

table(pisa_2022_school_canada$OECD, useNA = "ifany")
# >   1
# > 863
length(unique(pisa_2022_school_canada$OECD))
# > 1

attr(pisa_2022_school_canada$OECD, "labels")[attr(pisa_2022_school_canada$OECD, "labels") == 1]
# > Yes
# >   1

### Consistency Check Summary
# OECD is a labeled double in both student and school datasets.
# Each file contains exactly one unique value: 1 ("Yes").
# No missing or unexpected values are present.
# The value 1 is correctly labeled as "Yes" in both datasets.
# OECD is consistent and valid across student and school files.


## ---- ADMINMODE: Mode of Respondent ----

### Student
class(pisa_2022_student_canada$ADMINMODE)
# > "haven_labelled" "vctrs_vctr"     "double" 

table(pisa_2022_student_canada$ADMINMODE, useNA = "ifany")
# >     2
# > 23073
length(unique(pisa_2022_student_canada$ADMINMODE))
# > 1

attr(pisa_2022_student_canada$ADMINMODE, "labels")[attr(pisa_2022_student_canada$ADMINMODE, "labels") == 2] 
# > Computer 
# >        2

### School
class(pisa_2022_school_canada$ADMINMODE)
# > "haven_labelled" "vctrs_vctr"     "double" 

table(pisa_2022_school_canada$ADMINMODE, useNA = "ifany")
# >   2
# > 863
length(unique(pisa_2022_school_canada$ADMINMODE))
# > 1

attr(pisa_2022_school_canada$ADMINMODE, "labels")[attr(pisa_2022_school_canada$ADMINMODE, "labels") == 2]
# > Computer
# >        2

### Consistency Check Summary
# ADMINMODE is a labeled double in both student and school datasets.
# Each file contains exactly one unique value: 2 ("Computer").
# No missing or unexpected values are present.
# The value 2 is correctly labeled as "Computer" in both datasets.
# ADMINMODE is consistent and valid across student and school files.


## ---- *LANGTEST_QQQ: Language of Questionnaire ----

### Student
class(pisa_2022_student_canada$LANGTEST_QQQ)
# > "haven_labelled" "vctrs_vctr"     "double" 

table(pisa_2022_student_canada$LANGTEST_QQQ, useNA = "ifany")
# >   313   493  <NA> 
# > 16014  6149   910
prop.table(table(pisa_2022_student_canada$LANGTEST_QQQ, useNA = "ifany")) * 100
# >       313       493      <NA> 
# > 69.405799 26.650197  3.944004
length(unique(pisa_2022_student_canada$LANGTEST_QQQ))
# > 3

attr(pisa_2022_student_canada$LANGTEST_QQQ, "labels")[attr(pisa_2022_student_canada$LANGTEST_QQQ, "labels") == 313]
# > English
# >     313
attr(pisa_2022_student_canada$LANGTEST_QQQ, "labels")[attr(pisa_2022_student_canada$LANGTEST_QQQ, "labels") == 493]
# > French
# >    493
attr(pisa_2022_student_canada$LANGTEST_QQQ, "labels")[attr(pisa_2022_student_canada$LANGTEST_QQQ, "labels") == NA]
# > <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> 
# >   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA 
# > <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> 
# >   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA 
# > <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> 
# >   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA 

### School
class(pisa_2022_school_canada$LANGTEST_QQQ)
# > "haven_labelled" "vctrs_vctr"     "double" 

table(pisa_2022_school_canada$LANGTEST_QQQ, useNA = "ifany")
# > 313  493 <NA>
# > 596  216   51
prop.table(table(pisa_2022_school_canada$LANGTEST_QQQ, useNA = "ifany")) * 100
# >       313       493      <NA> 
# > 69.061414 25.028969  5.909618
length(unique(pisa_2022_school_canada$LANGTEST_QQQ))
# > 3

attr(pisa_2022_school_canada$LANGTEST_QQQ, "labels")[attr(pisa_2022_school_canada$LANGTEST_QQQ, "labels") == 313]
# > English
# >     313
attr(pisa_2022_school_canada$LANGTEST_QQQ, "labels")[attr(pisa_2022_school_canada$LANGTEST_QQQ, "labels") == 493]
# > French
# >    493
attr(pisa_2022_school_canada$LANGTEST_QQQ, "labels")[attr(pisa_2022_school_canada$LANGTEST_QQQ, "labels") == NA]
# > <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> 
# >   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA 
# > <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> 
# >   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA 
# > <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> <NA> 
# >   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA   NA 
# > <NA> <NA> 
# >   NA   NA 

### Check Student- and School-Level Missing IDs and Overlap
unique(pisa_2022_student_canada$CNTSCHID[is.na(pisa_2022_student_canada$LANGTEST_QQQ)])
# > 12400524 12400216 12400466 12400886 12400017 12400816 12400741 12400652 12400562 12400926 12400725 12400483
# > 12400191 12400391 12400878 12400740 12400818 12400641 12400249 12400901 12400653 12400008 12400493 12400251
# > 12400810 12400475 12400875 12400308 12400361 12400363 12400089 12400876 12400906 12400186 12400861 12400120
# > 12400257 12400679 12400359 12400245 12400620 12400525 12400799 12400868 12400093 12400556 12400171 12400927
# > 12400177 12400105 12400266 12400127 12400134 12400207 12400419 12400922 12400795 12400132 12400498 12400259
# > 12400768 12400903 12400444 12400224 12400377 12400225 12400246 12400327 12400316 12400683 12400088 12400615
# > 12400912 12400213 12400689 12400115 12400815 12400012 12400595 12400585 12400417 12400006 12400375 12400510
# > 12400239 12400445 12400126 12400080 12400602 12400826 12400372 12400596 12400022 12400109 12400770 12400339
# > 12400226 12400837 12400365 12400264 12400457 12400675 12400471 12400590 12400803 12400346 12400173 12400866
# > 12400518 12400342 12400711 12400412 12400197 12400576 12400220 12400082 12400113 12400427 12400098 12400344
# > 12400894 12400685 12400410 12400133 12400210 12400011 12400569 12400178 12400383 12400411 12400592 12400644
# > 12400661 12400446 12400610 12400870 12400094 12400727 12400189 12400732 12400905 12400209 12400273 12400116
# > 12400864 12400736 12400600 12400364 12400597 12400436 12400341 12400481 12400203 12400916 12400547 12400728
# > 12400269 12400533 12400368 12400074 12400500 12400464 12400696 12400310 12400260 12400382 12400106 12400016
# > 12400004 12400544 12400050 12400621 12400284 12400508 12400827 12400418 12400840 12400142 12400772 12400401
# > 12400433 12400895 12400676 12400374 12400102 12400467 12400397 12400846 12400097 12400757 12400897 12400814
# > 12400198 12400627 12400628 12400320 12400185 12400715 12400009 12400645 12400898 12400721 12400110 12400130
# > 12400730 12400691 12400538 12400612 12400487 12400669 12400270 12400424 12400151 12400381 12400278 12400159
# > 12400231 12400314 12400174 12400925 12400069 12400581 12400407 12400100 12400890 12400018 12400456 12400149
# > 12400492 12400303 12400754 12400291 12400802 12400545 12400892 12400422 12400531 12400092 12400528 12400283
# > 12400007 12400480 12400053 12400575 12400340 12400839 12400779 12400413 12400143 12400700 12400654 12400014
# > 12400746 12400784 12400829 12400604 12400853 12400077 12400402 12400527 12400350 12400268 12400734 12400869
# > 12400896 12400775 12400305 12400282 12400041 12400104 12400733 12400394 12400568 12400276 12400254 12400451
# > 12400040 12400300 12400399 12400468 12400384 12400392 12400720 12400769 12400240 12400699 12400881 12400804
# > 12400792 12400554 12400302 12400429 12400137 12400438 12400306 12400825 12400453 12400479 12400657 12400443
# > 12400718 12400099 12400386 12400514 12400506 12400194 12400376 12400613 12400297 12400078
length(unique(pisa_2022_student_canada$CNTSCHID[is.na(pisa_2022_student_canada$LANGTEST_QQQ)]))
# > 310

unique(pisa_2022_school_canada$CNTSCHID[is.na(pisa_2022_school_canada$LANGTEST_QQQ)])
# > 12400004 12400006 12400013 12400029 12400030 12400037 12400040 12400077 12400078 12400092 12400095 12400108
# > 12400142 12400180 12400196 12400197 12400232 12400233 12400239 12400247 12400249 12400269 12400282 12400294
# > 12400310 12400326 12400344 12400383 12400397 12400450 12400458 12400475 12400496 12400534 12400538 12400542
# > 12400555 12400572 12400599 12400610 12400612 12400627 12400635 12400675 12400730 12400811 12400824 12400847
# > 12400863 12400871 12400900
length(unique(pisa_2022_school_canada$CNTSCHID[is.na(pisa_2022_school_canada$LANGTEST_QQQ)]))
# > 51

intersect(unique(pisa_2022_student_canada$CNTSCHID[is.na(pisa_2022_student_canada$LANGTEST_QQQ)]), unique(pisa_2022_school_canada$CNTSCHID[is.na(pisa_2022_school_canada$LANGTEST_QQQ)]))
# >  12400249 12400475 12400006 12400239 12400675 12400197 12400344 12400383 12400610 12400269 12400310 12400004
# >  12400142 12400397 12400627 12400730 12400538 12400612 12400092 12400077 12400282 12400040 12400078
length(intersect(unique(pisa_2022_student_canada$CNTSCHID[is.na(pisa_2022_student_canada$LANGTEST_QQQ)]), unique(pisa_2022_school_canada$CNTSCHID[is.na(pisa_2022_school_canada$LANGTEST_QQQ)])))
# > 23


### Cross-tab: student LANGTEST_QQQ by school LANGTEST_QQQ
pisa_2022_student_canada %>%
  filter(CNTSCHID %in% (
    pisa_2022_school_canada %>%
      filter(LANGTEST_QQQ == 313) %>%
      pull(CNTSCHID)
  )) %>%
  count(LANGTEST_QQQ)
# > LANGTEST_QQQ      n
# > <dbl+lbl>     <int>
# > 313 [English] 14883
# > 493 [French]    192
# > NA              604

pisa_2022_student_canada %>%
  filter(CNTSCHID %in% (
    pisa_2022_school_canada %>%
      filter(LANGTEST_QQQ == 493) %>%
      pull(CNTSCHID)
  )) %>%
  count(LANGTEST_QQQ)
# > LANGTEST_QQQ      n
# > <dbl+lbl>     <int>
# > 313 [English]    67
# > 493 [French]   5702
# > NA              227

pisa_2022_student_canada %>%
  filter(CNTSCHID %in% (
    pisa_2022_school_canada %>%
      filter(is.na(LANGTEST_QQQ)) %>%
      pull(CNTSCHID)
  )) %>%
  count(LANGTEST_QQQ)
# > LANGTEST_QQQ      n
# > <dbl+lbl>     <int>
# > 313 [English]  1064
# > 493 [French]    255
# > NA               79


### Consistency Check Summary
# LANGTEST_QQQ is a labeled double in both student and school datasets.
# Three values: 313 ("English"), 493 ("French"), and NA (missing).
# Student data: 69.4% English, 26.7% French, 3.9% missing.
# School data: 69.1% English, 25.0% French, 5.9% missing.
# Cross-tab reveals some inconsistencies.
# TODO: Investigate causes of mismatch.
# TODO: Decide merge strategy—retain both columns, prioritize student data values, or impute where missing.
# TODO: Use SPSS or SAS to explore metadata or missing labels for NA cases.


## ---- *SENWT: Senate Weight (sum of 5000 per country) ----

### Student
class(pisa_2022_student_canada$SENWT)
# > "numeric" 

length(unique(pisa_2022_student_canada$SENWT))
# > 2119

summary(pisa_2022_student_canada$SENWT)
# > Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# > 0.01418  0.05228  0.12376  0.21670  0.36284 11.64358

sum(is.na(pisa_2022_student_canada$SENWT))
# > 0

### School
class(pisa_2022_school_canada$SENWT)
# > "numeric"  

length(unique(pisa_2022_school_canada$SENWT))
# > 392

summary(pisa_2022_school_canada$SENWT)
# > Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# > 1.238   1.263   2.509   5.794   5.156 799.064

sum(is.na(pisa_2022_school_canada$SENWT))
# > 0

### Histogram: 2x2 layout, top = student, bottom = school
par(mfrow = c(2, 2))

hist(pisa_2022_student_canada$SENWT,
     breaks = 50,
     col = "skyblue",
     main = "Student (Original Scale)",
     xlab = "SENWT")

hist(log1p(pisa_2022_student_canada$SENWT),
     breaks = 50,
     col = "skyblue",
     main = "Student (Log Scale)",
     xlab = "log(1 + SENWT)")

hist(pisa_2022_school_canada$SENWT,
     breaks = 50,
     col = "lightgreen",
     main = "School (Original Scale)",
     xlab = "SENWT")

hist(log1p(pisa_2022_school_canada$SENWT),
     breaks = 50,
     col = "lightgreen",
     main = "School (Log Scale)",
     xlab = "log(1 + SENWT)")

### Consistency Check Summary
# SENWT is a numeric variable in both student and school datasets.
# No missing values in either dataset.
# Student data: 2,119 unique values, range 0.01–11.64.
# School data: 392 unique values, range 1.24–799.06.
# Distributions are highly skewed.
# TODO: Investigate causes of mismatch.
# TODO: Decide merge strategy—retain both columns or prioritize one.


## ---- *VER_DAT: Date of the database creation ----

### Student
class(pisa_2022_student_canada$VER_DAT)
# > "character"

length(pisa_2022_student_canada$VER_DAT)
# > 23073

length(unique(pisa_2022_student_canada$VER_DAT))
# > 10


### School
class(pisa_2022_school_canada$VER_DAT)
# > "character"

length(pisa_2022_school_canada$VER_DAT)
# > 23073

length(unique(pisa_2022_school_canada$VER_DAT))
# 1

### Consistency Check Summary
# VER_DAT is a character variable in both student and school datasets.
# Student data: 10 unique values across 23,073 records.
# School data: 1 unique value across 23,073 records.


# ---- Merging PISA 2022 Canada student + school by CNTSCHID  ----

#  Merge Canada-only PISA 2022 student + school by CNTSCHID 
pisa_2022_canada_merged <-
  pisa_2022_student_canada %>%
  # Keep both versions for the 3 inconsistent variables on the student side
  dplyr::rename_with(~ paste0(.x, "_stu"),
                     .cols = dplyr::any_of(c("LANGTEST_QQQ", "SENWT", "VER_DAT"))) %>%
  dplyr::left_join(
    pisa_2022_school_canada %>%
      # Drop duplicated constants from the school file, but keep key + the 3 to preserve
      dplyr::select(
        -dplyr::all_of(setdiff(
          intersect(names(pisa_2022_student_canada), names(pisa_2022_school_canada)),
          c("CNTSCHID", "LANGTEST_QQQ", "SENWT", "VER_DAT")
        ))
      ) %>%
      # Keep school versions of the 3 inconsistent variables
      dplyr::rename_with(~ paste0(.x, "_sch"),
                         .cols = dplyr::any_of(c("LANGTEST_QQQ", "SENWT", "VER_DAT"))),
    by = "CNTSCHID"
  )

# A simpler version
# pisa_2022_canada_merged <-
#   pisa_2022_student_canada %>%
#   # Keep both versions for the 3 inconsistent variables on the student side
#   rename_with(~ paste0(.x, "_stu"),
#               c("LANGTEST_QQQ", "SENWT", "VER_DAT")) %>%
#   left_join(
#     pisa_2022_school_canada %>%
#       # Drop duplicated constants from the school file, but keep key + the 3 to preserve
#       select(
#         -all_of(setdiff(
#           intersect(names(pisa_2022_student_canada), names(pisa_2022_school_canada)),
#           c("CNTSCHID", "LANGTEST_QQQ", "SENWT", "VER_DAT")
#         ))
#       ) %>%
#       # Keep school versions of the 3 inconsistent variables
#       rename_with(~ paste0(.x, "_sch"),
#                   c("LANGTEST_QQQ", "SENWT", "VER_DAT")),
#     by = "CNTSCHID"
#   )

# Sanity checks 
stopifnot(nrow(pisa_2022_canada_merged) == nrow(pisa_2022_student_canada))  
stopifnot(all(pisa_2022_canada_merged$CNTSCHID %in% pisa_2022_student_canada$CNTSCHID))
stopifnot(all(pisa_2022_canada_merged$CNTSCHID %in% pisa_2022_school_canada$CNTSCHID))
stopifnot(!anyDuplicated(names(pisa_2022_canada_merged)))

# Save merged data
write_sav(pisa_2022_canada_merged, "data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav")

# Load merged data
pisa_2022_canada_merged <- read_sav("data/pisa2022/CY08MSP_STU_SCH_QQQ_CAN.sav", user_na = TRUE)

# Explore merged dataset
dim(pisa_2022_canada_merged)    # 23,073 × 1699
class(pisa_2022_canada_merged)  # "tbl_df", "tbl", "data.frame"


