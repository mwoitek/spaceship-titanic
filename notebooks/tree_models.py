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
# # Spaceship Titanic: Tree Models
# ## Imports

# %%
import warnings
from pathlib import Path

import pandas as pd

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
# ## Create features from `PassengerId`

# %%
# Group
df_train["Group"] = df_train["PassengerId"].str.split("_", expand=True).iloc[:, 0]
df_test["Group"] = df_test["PassengerId"].str.split("_", expand=True).iloc[:, 0]

# %%
# GroupSize
df_train = df_train.join(
    df_train.groupby(by="Group").agg(GroupSize=pd.NamedAgg(column="PassengerId", aggfunc="count")),
    on="Group",
)
df_test = df_test.join(
    df_test.groupby(by="Group").agg(GroupSize=pd.NamedAgg(column="PassengerId", aggfunc="count")),
    on="Group",
)

# %%
# Set indexes
df_train = df_train.set_index("PassengerId", verify_integrity=True)
df_test = df_test.set_index("PassengerId", verify_integrity=True)

# %% [markdown]
# ## Using the `Name` column

# %%
# Add Surname column
df_train = df_train.assign(Surname=df_train["Name"].str.split(" ", expand=True).iloc[:, 1])
df_test = df_test.assign(Surname=df_test["Name"].str.split(" ", expand=True).iloc[:, 1])

# %% [markdown]
# ## Impute some missing values
# Passengers who belong to the same group also come from the same home planet:

# %%
assert (
    df_train[df_train["HomePlanet"].notna()]
    .groupby("Group")
    .agg({"HomePlanet": "nunique"})
    .eq(1)
    .all(axis=None)
)
assert (
    df_test[df_test["HomePlanet"].notna()]
    .groupby("Group")
    .agg({"HomePlanet": "nunique"})
    .eq(1)
    .all(axis=None)
)

# %% [markdown]
# Using group data to impute some missing `HomePlanet` values:

# %%
# Training data
df_1 = (
    df_train.loc[df_train["GroupSize"].gt(1) & df_train["HomePlanet"].notna(), ["Group", "HomePlanet"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
query = "GroupSize > 1 and Group in @df_1.Group and HomePlanet.isna()"
df_2 = df_train.query(query).loc[:, ["Group"]].reset_index()
df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
df_train.loc[df_3.index, "HomePlanet"] = df_3["HomePlanet"]
del df_1, df_2, df_3

# %%
# Test data
df_1 = (
    df_test.loc[df_test["GroupSize"].gt(1) & df_test["HomePlanet"].notna(), ["Group", "HomePlanet"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
df_2 = df_test.query(query).loc[:, ["Group"]].reset_index()
df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
df_test.loc[df_3.index, "HomePlanet"] = df_3["HomePlanet"]
del df_1, df_2, df_3, query

# %% [markdown]
# Passengers with the same surname are from the same planet:

# %%
assert (
    df_train[["Surname", "HomePlanet"]]
    .dropna()
    .groupby("Surname")
    .agg({"HomePlanet": "nunique"})
    .eq(1)
    .all(axis=None)
)
assert (
    df_test[["Surname", "HomePlanet"]]
    .dropna()
    .groupby("Surname")
    .agg({"HomePlanet": "nunique"})
    .eq(1)
    .all(axis=None)
)

# %% [markdown]
# Use `Surname` to fill more missing `HomePlanet` values:

# %%
# Training data
df_sur_1 = (
    df_train[["Surname", "HomePlanet"]].dropna().groupby("Surname").agg({"HomePlanet": "first"}).reset_index()
)
query = "Surname.notna() and Surname in @df_sur_1.Surname and HomePlanet.isna()"
df_1 = df_train.query(query).loc[:, ["Surname"]].reset_index()
df_2 = df_1.merge(df_sur_1, on="Surname").drop(columns="Surname").set_index("PassengerId")
df_train.loc[df_2.index, "HomePlanet"] = df_2["HomePlanet"]
del df_1, df_2

# %%
# Test data

# To fix test data, I'll also use some training data. Combine all relevant data:
df_sur_2 = (
    df_test[["Surname", "HomePlanet"]].dropna().groupby("Surname").agg({"HomePlanet": "first"}).reset_index()
)
df_sur = pd.concat([df_sur_1, df_sur_2.query("Surname not in @df_sur_1.Surname")], ignore_index=True)
del df_sur_1, df_sur_2

# %%
query = query.replace("df_sur_1", "df_sur")
df_1 = df_test.query(query).loc[:, ["Surname"]].reset_index()
df_2 = df_1.merge(df_sur, on="Surname").drop(columns="Surname").set_index("PassengerId")
df_test.loc[df_2.index, "HomePlanet"] = df_2["HomePlanet"]
del df_1, df_2, df_sur, query

# %%
