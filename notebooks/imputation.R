# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# # Spaceship Titanic: Data Imputation
# ## Imports

# %%
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(missForest))

# %% [markdown]
# ## Read data

# %%
# Path to data directory
data_dir <- here("input", "spaceship-titanic")
stopifnot(dir.exists(data_dir))

# %%
# Path to training data
train_path <- file.path(data_dir, "train_prep.csv")
stopifnot(file.exists(train_path))

# %%
# Read training data
df_train <- read_csv(
  train_path,
  show_col_types = FALSE,
  col_types = list(
    PassengerId = col_character(),
    Alone = col_logical(),
    CompCntReduced = col_factor(levels = c("0", "1", "2", "3+"), ordered = TRUE),
    HomePlanetOrd = col_factor(levels = c("0.0", "1.0", "2.0"), ordered = FALSE),
    CryoSleep = col_logical(),
    CabinDeckOrd = col_factor(
      levels = c("0.0", "1.0", "2.0", "3.0", "4.0", "5.0"),
      ordered = FALSE
    ),
    CabinPort = col_logical(),
    DestinationOrd = col_factor(levels = c("0.0", "1.0", "2.0"), ordered = FALSE),
    DiscretizedAge4 = col_factor(levels = c("0.0", "1.0", "2.0", "3.0"), ordered = TRUE),
    DiscretizedAge5 = col_factor(levels = c("0.0", "1.0", "2.0", "3.0", "4.0"), ordered = TRUE),
    VIP = col_logical(),
    PosRoomService = col_logical(),
    PosFoodCourt = col_logical(),
    PosShoppingMall = col_logical(),
    PosSpa = col_logical(),
    PosVRDeck = col_logical(),
    PTTotalSpent = col_double(),
    Transported = col_skip()
  )
) %>%
  column_to_rownames("PassengerId") %>%
  as.data.frame()
str(df_train)

# %%
head(df_train, n = 10L)

# %%
# Path to test data
test_path <- file.path(data_dir, "test_prep.csv")
stopifnot(file.exists(test_path))

# %%
# Read test data
df_test <- read_csv(
  test_path,
  show_col_types = FALSE,
  col_types = list(
    PassengerId = col_character(),
    Alone = col_logical(),
    CompCntReduced = col_factor(levels = c("0", "1", "2", "3+"), ordered = TRUE),
    HomePlanetOrd = col_factor(levels = c("0.0", "1.0", "2.0"), ordered = FALSE),
    CryoSleep = col_logical(),
    CabinDeckOrd = col_factor(
      levels = c("0.0", "1.0", "2.0", "3.0", "4.0", "5.0"),
      ordered = FALSE
    ),
    CabinPort = col_logical(),
    DestinationOrd = col_factor(levels = c("0.0", "1.0", "2.0"), ordered = FALSE),
    DiscretizedAge4 = col_factor(levels = c("0.0", "1.0", "2.0", "3.0"), ordered = TRUE),
    DiscretizedAge5 = col_factor(levels = c("0.0", "1.0", "2.0", "3.0", "4.0"), ordered = TRUE),
    VIP = col_logical(),
    PosRoomService = col_logical(),
    PosFoodCourt = col_logical(),
    PosShoppingMall = col_logical(),
    PosSpa = col_logical(),
    PosVRDeck = col_logical(),
    PTTotalSpent = col_double()
  )
) %>%
  column_to_rownames("PassengerId") %>%
  as.data.frame()
str(df_test)

# %%
head(df_test, n = 10L)
