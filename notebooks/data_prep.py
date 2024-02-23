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
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, PowerTransformer

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
# Passengers who belong to the same group also come from the same home planet
df_test.loc[df_test.HomePlanet.notna(), ["Group", "HomePlanet"]].groupby(
    "Group", observed=True
).HomePlanet.nunique().eq(1).all()

# %%
# Number of missing values BEFORE
print(f"Training data: {df_train.HomePlanet.isna().sum()}")
print(f"Test data: {df_test.HomePlanet.isna().sum()}")

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
# Number of missing values AFTER
print(f"Training data: {df_train.HomePlanet.isna().sum()}")
print(f"Test data: {df_test.HomePlanet.isna().sum()}")

# %% [markdown]
# ## Using the `Name` column

# %%
# Add Surname column
df_train = df_train.assign(Surname=df_train.Name.str.split(" ", expand=True).iloc[:, 1]).drop(columns="Name")
df_test = df_test.assign(Surname=df_test.Name.str.split(" ", expand=True).iloc[:, 1]).drop(columns="Name")

# %%
# Passengers with the same surname are from the same planet
df_test[["Surname", "HomePlanet"]].dropna().groupby("Surname").HomePlanet.nunique().eq(1).all()

# %%
# Use Surname to fill more missing HomePlanet values

# Number of missing values BEFORE
print(f"Training data: {df_train.HomePlanet.isna().sum()}")
print(f"Test data: {df_test.HomePlanet.isna().sum()}")

# %%
# Training data
df_sur_1 = (
    df_train[["Surname", "HomePlanet"]]
    .dropna()
    .groupby("Surname")
    .HomePlanet.first()
    .to_frame()
    .reset_index(drop=False)
)

# %%
df_1 = df_train.loc[
    df_train.Surname.notna() & df_train.Surname.isin(df_sur_1.Surname) & df_train.HomePlanet.isna(),
    ["Surname"],
].reset_index(drop=False)
df_2 = df_1.merge(df_sur_1, on="Surname").drop(columns="Surname").set_index("PassengerId")
df_train.loc[df_2.index, "HomePlanet"] = df_2.HomePlanet
del df_1, df_2

# %%
# Test data
df_sur_2 = (
    df_test[["Surname", "HomePlanet"]]
    .dropna()
    .groupby("Surname")
    .HomePlanet.first()
    .to_frame()
    .reset_index(drop=False)
)

# %%
# Consistency check
assert_frame_equal(
    df_sur_1.loc[df_sur_1.Surname.isin(df_sur_2.Surname), :].sort_values("Surname").reset_index(drop=True),
    df_sur_2.loc[df_sur_2.Surname.isin(df_sur_1.Surname), :].sort_values("Surname").reset_index(drop=True),
)

# %%
# To fix test data, I'll also use some training data. Combine all relevant data:
df_sur = pd.concat(
    [df_sur_1, df_sur_2.loc[~df_sur_2.Surname.isin(df_sur_1.Surname), :]],
    ignore_index=True,
)
del df_sur_1, df_sur_2
assert df_sur.Surname.nunique() == df_sur.shape[0]

# %%
df_1 = df_test.loc[
    df_test.Surname.notna() & df_test.Surname.isin(df_sur.Surname) & df_test.HomePlanet.isna(),
    ["Surname"],
].reset_index(drop=False)
df_2 = df_1.merge(df_sur, on="Surname").drop(columns="Surname").set_index("PassengerId")
df_test.loc[df_2.index, "HomePlanet"] = df_2.HomePlanet
del df_1, df_2, df_sur

# %%
# Number of missing values AFTER
print(f"Training data: {df_train.HomePlanet.isna().sum()}")
print(f"Test data: {df_test.HomePlanet.isna().sum()}")

# %%
# Convert to ordinal integers
enc = OrdinalEncoder().fit(df_train[["HomePlanet"]])
# display(enc.categories_)

df_train["HomePlanetOrd"] = enc.transform(df_train[["HomePlanet"]]).flatten()
df_test["HomePlanetOrd"] = enc.transform(df_test[["HomePlanet"]]).flatten()

del enc

# %%
# Consistency checks
assert df_train.loc[df_train.HomePlanet.isna(), "HomePlanetOrd"].isna().all()
assert df_train.loc[df_train.HomePlanet.notna(), "HomePlanetOrd"].notna().all()

# %%
assert df_test.loc[df_test.HomePlanet.isna(), "HomePlanetOrd"].isna().all()
assert df_test.loc[df_test.HomePlanet.notna(), "HomePlanetOrd"].notna().all()

# %% [markdown]
# ## More simple data imputation

# %%
# The "money features" are dominated by zeros. Then it's reasonable to fill all
# of their missing values with zero.
cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Number of missing values BEFORE
print("Training data:")
display(df_train[cols].isna().sum())

print("Test data:")
display(df_test[cols].isna().sum())

# %%
df_train.loc[:, cols] = df_train[cols].fillna(0.0)
df_test.loc[:, cols] = df_test[cols].fillna(0.0)

# %%
# TotalSpent
df_train["TotalSpent"] = df_train[cols].agg("sum", axis=1)
df_test["TotalSpent"] = df_test[cols].agg("sum", axis=1)

# %%
# Fill some missing CryoSleep values based on TotalSpent

# Number of missing values BEFORE
print(f"Training data: {df_train.CryoSleep.isna().sum()}")
print(f"Test data: {df_test.CryoSleep.isna().sum()}")

# %%
df_train.loc[df_train.CryoSleep.isna() & df_train.TotalSpent.gt(0.0), "CryoSleep"] = False
df_test.loc[df_test.CryoSleep.isna() & df_test.TotalSpent.gt(0.0), "CryoSleep"] = False

# %%
# Number of missing values AFTER
print(f"Training data: {df_train.CryoSleep.isna().sum()}")
print(f"Test data: {df_test.CryoSleep.isna().sum()}")

# %%
# Passengers who were in cryo sleep spent NO MONEY
assert df_train.loc[df_train.CryoSleep.notna() & (df_train.CryoSleep == True), cols].eq(0.0).all(axis=None)
assert df_test.loc[df_test.CryoSleep.notna() & (df_test.CryoSleep == True), cols].eq(0.0).all(axis=None)

# %% [markdown]
# ## New features from "money variables"

# %%
# Original money variables will be replaced with binary features. They indicate
# when the original variables were strictly positive.
df_train = df_train.join(df_train[cols].gt(0.0).rename(columns={col: f"Pos{col}" for col in cols})).drop(
    columns=cols
)
df_test = df_test.join(df_test[cols].gt(0.0).rename(columns={col: f"Pos{col}" for col in cols})).drop(
    columns=cols
)
del cols

# %%
# Power transformation of TotalSpent
transformer = PowerTransformer().fit(df_train[["TotalSpent"]])
print(transformer.lambdas_[0])

# %%
df_train = df_train.assign(PTTotalSpent=transformer.transform(df_train[["TotalSpent"]]).flatten()).drop(
    columns="TotalSpent"
)
df_test = df_test.assign(PTTotalSpent=transformer.transform(df_test[["TotalSpent"]]).flatten()).drop(
    columns="TotalSpent"
)
del transformer

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

# %%
# CabinDeck: Combine three categories into one
df_train.loc[df_train.CabinDeck.notna() & df_train.CabinDeck.isin(["D", "A", "T"]), "CabinDeck"] = "Other"
df_test.loc[df_test.CabinDeck.notna() & df_test.CabinDeck.isin(["D", "A", "T"]), "CabinDeck"] = "Other"

# %%
# Convert to ordinal integers
enc = OrdinalEncoder().fit(df_train[["CabinDeck"]])
# display(enc.categories_)

df_train["CabinDeckOrd"] = enc.transform(df_train[["CabinDeck"]]).flatten()
df_test["CabinDeckOrd"] = enc.transform(df_test[["CabinDeck"]]).flatten()

del enc

# %%
# Consistency checks
assert df_train.loc[df_train.CabinDeck.isna(), "CabinDeckOrd"].isna().all()
assert df_train.loc[df_train.CabinDeck.notna(), "CabinDeckOrd"].notna().all()

# %%
assert df_test.loc[df_test.CabinDeck.isna(), "CabinDeckOrd"].isna().all()
assert df_test.loc[df_test.CabinDeck.notna(), "CabinDeckOrd"].notna().all()

# %%
df_train = df_train.drop(columns="CabinDeck")
df_test = df_test.drop(columns="CabinDeck")

# %%
# Fill some missing CabinSide values using group data
# Passengers that belong to the same group were on the same side of the spaceship

# Number of missing values BEFORE
print(f"Training data: {df_train.CabinSide.isna().sum()}")
print(f"Test data: {df_test.CabinSide.isna().sum()}")

# %%
# Training data
df_1 = (
    df_train.loc[(df_train.Alone == False) & df_train.CabinSide.notna(), ["CabinSide", "Group"]]
    .groupby("Group", observed=True)
    .CabinSide.first()
    .to_frame()
    .reset_index(drop=False)
)
df_2 = df_train.loc[(df_train.Alone == False) & df_train.CabinSide.isna(), ["Group"]].reset_index(drop=False)
df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
df_train.loc[df_3.index, "CabinSide"] = df_3.CabinSide
del df_1, df_2, df_3

# %%
# Test data
df_1 = (
    df_test.loc[(df_test.Alone == False) & df_test.CabinSide.notna(), ["CabinSide", "Group"]]
    .groupby("Group", observed=True)
    .CabinSide.first()
    .to_frame()
    .reset_index(drop=False)
)
df_2 = df_test.loc[(df_test.Alone == False) & df_test.CabinSide.isna(), ["Group"]].reset_index(drop=False)
df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
df_test.loc[df_3.index, "CabinSide"] = df_3.CabinSide
del df_1, df_2, df_3

# %%
# Number of missing values AFTER
print(f"Training data: {df_train.CabinSide.isna().sum()}")
print(f"Test data: {df_test.CabinSide.isna().sum()}")

# %%
# Convert CabinSide to a boolean feature
df_train["CabinPort"] = np.nan
df_train.loc[df_train.CabinSide.notna(), "CabinPort"] = (
    df_train.loc[df_train.CabinSide.notna(), "CabinSide"] == "P"
)
df_train = df_train.drop(columns="CabinSide")

df_test["CabinPort"] = np.nan
df_test.loc[df_test.CabinSide.notna(), "CabinPort"] = (
    df_test.loc[df_test.CabinSide.notna(), "CabinSide"] == "P"
)
df_test = df_test.drop(columns="CabinSide")

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

del discretizer

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

del discretizer

# %%
df_train = df_train.drop(columns="Age")
df_test = df_test.drop(columns="Age")

# %%

# %%
df_train.info()

# %%
df_train.isna().sum()

# %%
df_test.info()

# %%
df_test.isna().sum()
