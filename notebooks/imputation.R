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
    Transported = col_logical()
  )
) %>%
  column_to_rownames("PassengerId") %>%
  as.data.frame()

# %%
target <- df_train$Transported
df_train$Transported <- NULL
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
maxiter <- 10
ntree <- 100

# %%
# Training data
train_imp <- missForest(
  df_train,
  maxiter = maxiter,
  ntree = ntree,
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
  maxiter = maxiter,
  ntree = ntree,
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

# %% [markdown]
# ## Save imputed datasets

# %%
# Training data
tbl_train <- as_tibble(train_imp$ximp, rownames = NA) %>%
  rownames_to_column(var = "PassengerId") %>%
  mutate(
    Alone = Alone == "True",
    CompCntReduced = as.character(CompCntReduced),
    HomePlanetOrd = HomePlanetOrd %>% as.character() %>% as.integer(),
    CryoSleep = CryoSleep == "True",
    CabinDeckOrd = CabinDeckOrd %>% as.character() %>% as.integer(),
    CabinPort = CabinPort == "True",
    DestinationOrd = DestinationOrd %>% as.character() %>% as.integer(),
    DiscretizedAge4 = DiscretizedAge4 %>% as.character() %>% as.integer(),
    DiscretizedAge5 = DiscretizedAge5 %>% as.character() %>% as.integer(),
    VIP = VIP == "True",
    PosRoomService = PosRoomService == "True",
    PosFoodCourt = PosFoodCourt == "True",
    PosShoppingMall = PosShoppingMall == "True",
    PosSpa = PosSpa == "True",
    PosVRDeck = PosVRDeck == "True"
  ) %>%
  add_column(Transported = target)
str(tbl_train)

# %%
head(tbl_train, n = 10L)

# %%
write_csv(
  tbl_train,
  file.path(data_dir, "train_imputed.csv"),
  quote = "needed",
  escape = "backslash",
  progress = FALSE
)

# %%
# Test data
tbl_test <- as_tibble(test_imp$ximp, rownames = NA) %>%
  rownames_to_column(var = "PassengerId") %>%
  mutate(
    Alone = Alone == "True",
    CompCntReduced = as.character(CompCntReduced),
    HomePlanetOrd = HomePlanetOrd %>% as.character() %>% as.integer(),
    CryoSleep = CryoSleep == "True",
    CabinDeckOrd = CabinDeckOrd %>% as.character() %>% as.integer(),
    CabinPort = CabinPort == "True",
    DestinationOrd = DestinationOrd %>% as.character() %>% as.integer(),
    DiscretizedAge4 = DiscretizedAge4 %>% as.character() %>% as.integer(),
    DiscretizedAge5 = DiscretizedAge5 %>% as.character() %>% as.integer(),
    VIP = VIP == "True",
    PosRoomService = PosRoomService == "True",
    PosFoodCourt = PosFoodCourt == "True",
    PosShoppingMall = PosShoppingMall == "True",
    PosSpa = PosSpa == "True",
    PosVRDeck = PosVRDeck == "True"
  )
str(tbl_test)

# %%
head(tbl_test, n = 10L)

# %%
write_csv(
  tbl_test,
  file.path(data_dir, "test_imputed.csv"),
  quote = "needed",
  escape = "backslash",
  progress = FALSE
)
