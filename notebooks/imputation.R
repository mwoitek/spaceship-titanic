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
    Alone = col_factor(levels = c("True", "False"), ordered = FALSE),
    CompCntReduced = col_factor(levels = c("0", "1", "2", "3+"), ordered = TRUE),
    HomePlanetOrd = col_factor(levels = c("0.0", "1.0", "2.0"), ordered = FALSE),
    CryoSleep = col_factor(levels = c("True", "False"), ordered = FALSE),
    CabinDeckOrd = col_factor(
      levels = c("0.0", "1.0", "2.0", "3.0", "4.0", "5.0"),
      ordered = FALSE
    ),
    CabinPort = col_factor(levels = c("True", "False"), ordered = FALSE),
    DestinationOrd = col_factor(levels = c("0.0", "1.0", "2.0"), ordered = FALSE),
    DiscretizedAge4 = col_factor(levels = c("0.0", "1.0", "2.0", "3.0"), ordered = TRUE),
    DiscretizedAge5 = col_factor(levels = c("0.0", "1.0", "2.0", "3.0", "4.0"), ordered = TRUE),
    VIP = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosRoomService = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosFoodCourt = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosShoppingMall = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosSpa = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosVRDeck = col_factor(levels = c("True", "False"), ordered = FALSE),
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
    Alone = col_factor(levels = c("True", "False"), ordered = FALSE),
    CompCntReduced = col_factor(levels = c("0", "1", "2", "3+"), ordered = TRUE),
    HomePlanetOrd = col_factor(levels = c("0.0", "1.0", "2.0"), ordered = FALSE),
    CryoSleep = col_factor(levels = c("True", "False"), ordered = FALSE),
    CabinDeckOrd = col_factor(
      levels = c("0.0", "1.0", "2.0", "3.0", "4.0", "5.0"),
      ordered = FALSE
    ),
    CabinPort = col_factor(levels = c("True", "False"), ordered = FALSE),
    DestinationOrd = col_factor(levels = c("0.0", "1.0", "2.0"), ordered = FALSE),
    DiscretizedAge4 = col_factor(levels = c("0.0", "1.0", "2.0", "3.0"), ordered = TRUE),
    DiscretizedAge5 = col_factor(levels = c("0.0", "1.0", "2.0", "3.0", "4.0"), ordered = TRUE),
    VIP = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosRoomService = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosFoodCourt = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosShoppingMall = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosSpa = col_factor(levels = c("True", "False"), ordered = FALSE),
    PosVRDeck = col_factor(levels = c("True", "False"), ordered = FALSE),
    PTTotalSpent = col_double()
  )
) %>%
  column_to_rownames("PassengerId") %>%
  as.data.frame()
str(df_test)

# %%
head(df_test, n = 10L)

# %% [markdown]
# ## Imputation with `missForest`

# %%
# Training data
train_imp <- missForest(
  df_train,
  maxiter = 10,
  ntree = 100,
  variablewise = TRUE,
  verbose = TRUE
)

# %%
head(train_imp$ximp, n = 10L)

# %%
err <- train_imp$OOBerror
num_miss <- colSums(is.na(df_train))
mask <- num_miss > 0

tibble(
  Feature = names(num_miss[mask]),
  MissingValues = num_miss[mask],
  PFC = err[mask] # PFC = Proportion of falsely classified
) %>% column_to_rownames("Feature")

# %%
# Test data
test_imp <- missForest(
  df_test,
  maxiter = 10,
  ntree = 100,
  variablewise = TRUE,
  verbose = TRUE
)

# %%
head(test_imp$ximp, n = 10L)

# %%
err <- test_imp$OOBerror
num_miss <- colSums(is.na(df_test))
mask <- num_miss > 0

tibble(
  Feature = names(num_miss[mask]),
  MissingValues = num_miss[mask],
  PFC = err[mask] # PFC = Proportion of falsely classified
) %>% column_to_rownames("Feature")
