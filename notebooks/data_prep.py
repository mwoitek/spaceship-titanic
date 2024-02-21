# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spaceship Titanic: Data Preparation
# ## Imports

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

# %%
warnings.simplefilter(action="ignore", category=FutureWarning)

# %% [markdown]
# ## Read data

# %%
data_dir = Path.cwd().parent / "input" / "spaceship-titanic"
assert data_dir.exists(), f"directory doesn't exist: {data_dir}"

# %%
# Training data
df_train = pd.read_csv(data_dir / "train.csv")
df_train.head(10)

# %%
# Test data
df_test = pd.read_csv(data_dir / "test.csv")
df_test.head(10)

# %% [markdown]
# ## New features from `PassengerId`

# %%
# Group
df_train["Group"] = df_train.PassengerId.str.split("_", expand=True).iloc[:, 0].astype("category")
df_test["Group"] = df_test.PassengerId.str.split("_", expand=True).iloc[:, 0].astype("category")

# %%
# Alone and CompanionCount

# Training data
df_train = (
    df_train.join(
        df_train.groupby(by="Group").PassengerId.count().rename("GroupSize"),
        on="Group",
    )
    .assign(
        Alone=lambda x: x.GroupSize == 1,
        CompanionCount=lambda x: x.GroupSize - 1,
    )
    .drop(columns="GroupSize")
)

# %%
# Test data
df_test = (
    df_test.join(
        df_test.groupby(by="Group").PassengerId.count().rename("GroupSize"),
        on="Group",
    )
    .assign(
        Alone=lambda x: x.GroupSize == 1,
        CompanionCount=lambda x: x.GroupSize - 1,
    )
    .drop(columns="GroupSize")
)

# %%
# Set indexes
df_train = df_train.set_index("PassengerId", verify_integrity=True)
df_test = df_test.set_index("PassengerId", verify_integrity=True)

# %%
# Combine infrequent values of CompanionCount
df_train = df_train.assign(
    CompCntReduced=df_train.CompanionCount.transform(lambda x: np.where(x > 2, "3+", str(x)))
).drop(columns="CompanionCount")

df_test = df_test.assign(
    CompCntReduced=df_test.CompanionCount.transform(lambda x: np.where(x > 2, "3+", str(x)))
).drop(columns="CompanionCount")

# %% [markdown]
# ## Impute some missing values of `HomePlanet`

# %%
# Training data
df_1 = (
    df_train.loc[(df_train.Alone == False) & df_train.HomePlanet.notna(), ["Group", "HomePlanet"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
df_2 = df_train.loc[
    (df_train.Alone == False) & df_train.Group.isin(df_1.Group) & df_train.HomePlanet.isna(), ["Group"]
].reset_index(drop=False)

df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
# display(df_3.head(20))

df_train.loc[df_3.index, "HomePlanet"] = df_3.HomePlanet
# display(df_train.head(20))

del df_1, df_2, df_3

# %%
# Test data
df_1 = (
    df_test.loc[(df_test.Alone == False) & df_test.HomePlanet.notna(), ["Group", "HomePlanet"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
df_2 = df_test.loc[
    (df_test.Alone == False) & df_test.Group.isin(df_1.Group) & df_test.HomePlanet.isna(), ["Group"]
].reset_index(drop=False)

df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
# display(df_3.head(20))

df_test.loc[df_3.index, "HomePlanet"] = df_3.HomePlanet
# display(df_test.head(20))

del df_1, df_2, df_3

# %%
# Convert to ordinal integers
enc = OrdinalEncoder().fit(df_train[["HomePlanet"]])
# display(enc.categories_)

df_train["HomePlanetOrd"] = enc.transform(df_train[["HomePlanet"]]).flatten()
df_test["HomePlanetOrd"] = enc.transform(df_test[["HomePlanet"]]).flatten()

del enc

# %% [markdown]
# ## More simple data imputation

# %%
# TotalSpent
cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df_train["TotalSpent"] = df_train[cols].agg(np.nansum, axis=1)
df_test["TotalSpent"] = df_test[cols].agg(np.nansum, axis=1)

# %%
# Fill some missing CryoSleep values based on TotalSpent
df_train.loc[df_train.CryoSleep.isna() & df_train.TotalSpent.gt(0.0), "CryoSleep"] = False
df_test.loc[df_test.CryoSleep.isna() & df_test.TotalSpent.gt(0.0), "CryoSleep"] = False

# %%
# Passengers who were in cryo sleep spent NO MONEY

# Training data
df_1 = df_train.loc[df_train.CryoSleep.notna() & (df_train.CryoSleep == True), cols].fillna(0.0)
df_train.loc[df_1.index, cols] = df_1
del df_1

# %%
# Test data
df_1 = df_test.loc[df_test.CryoSleep.notna() & (df_test.CryoSleep == True), cols].fillna(0.0)
df_test.loc[df_1.index, cols] = df_1
del df_1, cols

# %% [markdown]
# ## New features from `Cabin`

# %%
# CabinDeck, CabinNum and CabinSide
df_train = df_train.join(
    df_train.Cabin.str.split("/", expand=True).rename(columns={0: "CabinDeck", 1: "CabinNum", 2: "CabinSide"})
).drop(columns="Cabin")

df_test = df_test.join(
    df_test.Cabin.str.split("/", expand=True).rename(columns={0: "CabinDeck", 1: "CabinNum", 2: "CabinSide"})
).drop(columns="Cabin")

# %% [markdown]
# ## Discretize `Age`

# %%
# Discretize using quantiles and 4 bins
discretizer = KBinsDiscretizer(n_bins=4, strategy="quantile", encode="ordinal", random_state=333).fit(
    df_train.loc[df_train.Age.notna(), ["Age"]]
)
# display(discretizer.bin_edges_)

df_train["DiscretizedAge4"] = np.nan
df_train.loc[df_train.Age.notna(), "DiscretizedAge4"] = discretizer.transform(
    df_train.loc[df_train.Age.notna(), ["Age"]]
)

df_test["DiscretizedAge4"] = np.nan
df_test.loc[df_test.Age.notna(), "DiscretizedAge4"] = discretizer.transform(
    df_test.loc[df_test.Age.notna(), ["Age"]]
)

# %%
# Discretize using quantiles and 5 bins
discretizer = KBinsDiscretizer(n_bins=5, strategy="quantile", encode="ordinal", random_state=333).fit(
    df_train.loc[df_train.Age.notna(), ["Age"]]
)
# display(discretizer.bin_edges_)

df_train["DiscretizedAge5"] = np.nan
df_train.loc[df_train.Age.notna(), "DiscretizedAge5"] = discretizer.transform(
    df_train.loc[df_train.Age.notna(), ["Age"]]
)

df_test["DiscretizedAge5"] = np.nan
df_test.loc[df_test.Age.notna(), "DiscretizedAge5"] = discretizer.transform(
    df_test.loc[df_test.Age.notna(), ["Age"]]
)

# %%
df_train = df_train.drop(columns="Age")
df_test = df_test.drop(columns="Age")

# %%
