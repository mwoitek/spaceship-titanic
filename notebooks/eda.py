# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Spaceship Titanic: Exploratory Data Analysis
# ## Imports

# %%
import warnings
from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
import seaborn.objects as so
from IPython.display import display
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer
from statsmodels.graphics.mosaicplot import mosaic

# %%
pl.Config.set_tbl_cols(14)
warnings.simplefilter(action="ignore", category=FutureWarning)

# %% [markdown]
# ## Read data

# %%
data_dir = Path.cwd().parent / "input" / "spaceship-titanic"
assert data_dir.exists(), f"directory doesn't exist: {data_dir}"

# %%
# Training data
df_train = pl.read_csv(data_dir / "train.csv")
df_train.head(10)

# %%
# Test data
df_test = pl.read_csv(data_dir / "test.csv")
df_test.head(10)

# %% [markdown]
# ## Number of observations

# %%
# Training data
df_train.height

# %%
# Test data
df_test.height

# %% [markdown]
# ## Missing values

# %%
# Training data
df_train.null_count()

# %%
# Test data
df_test.null_count()

# %% [markdown]
# ## `PassengerId`

# %%
# Extract groups
groups = df_train.get_column("PassengerId").str.split("_").list.first()
groups.head()

# %%
# Number of unique groups
groups.n_unique()

# %%
# Add as a DataFrame column
df_train = df_train.with_columns((pl.col("PassengerId").str.split("_").list.first()).alias("Group"))
df_train.head(10)

# %%
# Create a couple of features from `Group`
df_groups = (
    df_train.group_by("Group")
    .len()
    .rename({"len": "GroupSize"})
    .with_columns(
        [
            (pl.col("GroupSize") - 1).alias("CompanionCount"),
            (pl.col("GroupSize") == 1).alias("Alone"),
        ]
    )
    .drop("GroupSize")
)
df_groups.head(10)

# %%
df_train = df_train.join(df_groups, on="Group", how="left")
df_train.head(10)

# %%
# Number of people traveling alone
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(x=df_train.get_column("Alone").to_numpy(), order=[True, False], ax=ax)
ax.set_title("Traveling alone?")
ax.set_xticks(ax.get_xticks())  # seems useless but silences a warning
ax.set_xticklabels(["Yes", "No"])
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.show()

# %%
df_train.get_column("Alone").value_counts(sort=True)

# %%
# Relationship with target variable
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(df_train, x="Alone", hue="Transported", order=[True, False], ax=ax)
ax.set_title("Relationship between Alone and Transported")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.show()

# %%
df_crosstab = df_train.select(["Alone", "Transported"]).to_pandas()
pd.crosstab(df_crosstab.Alone, df_crosstab.Transported)

# %%
del df_crosstab

# %%
# Unique values of `CompanionCount`
companion_count = df_train.get_column("CompanionCount")
companion_count.unique()

# %%
# Visualizing number of companions
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

pos_counts = companion_count.filter(companion_count.gt(0)).to_numpy()
sns.countplot(x=pos_counts, order=list(range(1, 8)), ax=ax)
ax.set_title("Number of companions for those who had company")
ax.set_xlabel("Number of companions")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.show()

# %%
companion_count.value_counts().sort(by="CompanionCount")

# %%
# Identifying infrequent counts
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(
    df_train.select("CompanionCount"),
    x="CompanionCount",
    order=list(range(8)),
    stat="percent",
    ax=ax,
)
ax.axhline(y=5, color="red", linestyle="--")
ax.set_xlabel("Number of companions")
ax.set_ylabel("Percentage")
ax.yaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

plt.show()

# %%
# Combine infrequent counts into a single category
df_train = df_train.with_columns(
    CompCntReduced=pl.when(pl.col("CompanionCount").gt(2))
    .then(pl.lit("3+"))
    .otherwise(pl.col("CompanionCount").cast(pl.String))
)

# %%
# Relationship with target variable
tmp_df = df_train.select(["CompanionCount", "Transported"]).filter(pl.col("CompanionCount").gt(0))

fig = plt.figure(figsize=(8.0, 6.0))
ax = fig.add_subplot()

p = (
    so.Plot(tmp_df, x="CompanionCount", color="Transported")
    .add(so.Bar(), so.Count(), so.Stack())
    .on(ax)
    .label(
        title="Relationship between CompanionCount and Transported",
        x="Number of companions",
        y="Count",
    )
    .layout(engine="constrained")
)
p.show()

# %%
tmp_df = tmp_df.to_pandas()
pd.crosstab(
    tmp_df.Transported,
    tmp_df.CompanionCount,
    margins=True,
    margins_name="Total",
)

# %%
del tmp_df

# %%
# Relationship between CompCntReduced and Transported
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(
    df_train,
    x="CompCntReduced",
    hue="Transported",
    order=["0", "1", "2", "3+"],
    ax=ax,
)
ax.set_xlabel("Number of companions")
ax.set_ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

plt.show()

# %% [markdown]
# ## `HomePlanet`

# %%
# Unique values
df_train.get_column("HomePlanet").unique()

# %%
# Do passengers who belong to the same group also come from the same home planet?
df_train.select(["Group", "HomePlanet"]).drop_nulls().group_by("Group").agg(
    pl.col("HomePlanet").n_unique().alias("UniquePlanets")
).get_column("UniquePlanets").eq(1).all()

# %%
# Fix some of the missing values of HomePlanet

# Missing values BEFORE
df_train.get_column("HomePlanet").is_null().sum()

# %%
# Some of the rows that can be fixed:
df_train.filter(pl.col("Group").str.starts_with("044"), pl.col("Alone").not_()).select(
    ["PassengerId", "HomePlanet"]
)

# %%
# Identify rows that can be fixed, and the new values of HomePlanet
df_1 = df_train.filter(pl.col("HomePlanet").is_null(), pl.col("Alone").not_()).select(
    ["PassengerId", "Group"]
)
df_2 = (
    df_train.filter(
        pl.col("HomePlanet").is_not_null(),
        pl.col("Group").is_in(df_1.get_column("Group").unique()),
    )
    .select(["HomePlanet", "Group"])
    .unique()
)
df_3 = df_1.join(df_2, on="Group", how="inner").select(["PassengerId", "HomePlanet"])
del df_1
del df_2

# %%
# Update DataFrame with new values of HomePlanet
df_train = (
    df_train.join(df_3, on="PassengerId", how="left")
    .with_columns(pl.col("HomePlanet_right").fill_null(pl.col("HomePlanet")))
    .drop("HomePlanet")
    .rename({"HomePlanet_right": "HomePlanet"})
)
del df_3

# %%
# Quick check
df_train.filter(pl.col("Group").str.starts_with("044"), pl.col("Alone").not_()).select(
    ["PassengerId", "HomePlanet"]
)

# %%
# Missing values AFTER
df_train.get_column("HomePlanet").is_null().sum()

# %%
# For the moment, ignore missing values
home_planet = df_train.get_column("HomePlanet").drop_nulls()

# %%
# Visualizing number of passengers by home planet
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(x=home_planet.to_numpy(), order=["Earth", "Europa", "Mars"], ax=ax)
ax.set_title("Number of passengers by home planet")
ax.set_ylabel("Number of passengers")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.show()

# %%
home_planet.value_counts(sort=True)

# %%
# Relationship with the target variable

# For this, I need a Pandas DataFrame
tmp_df = (
    df_train.select(
        pl.col("HomePlanet"),
        pl.col("Transported"),
    )
    .filter(pl.col("HomePlanet").is_not_null())
    .to_pandas()
)
# tmp_df.head(10)

# %%
# Visualization
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(
    tmp_df,
    x="HomePlanet",
    hue="Transported",
    order=["Earth", "Europa", "Mars"],
    ax=ax,
)
ax.set_title("Relationship between home planet and target")
ax.set_xlabel("")
ax.set_ylabel("Number of passengers")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.show()

# %%
# Unfortunately, this function doesn't exist in Polars
pd.crosstab(tmp_df.Transported, tmp_df.HomePlanet)

# %%
del tmp_df

# %% [markdown]
# ## `CryoSleep`

# %%
# First, I'm going to do some consistency tests. To do so, I need to know the
# total amount spent by each passenger:
df_total = (
    df_train.select(["CryoSleep", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"])
    .filter(pl.col("CryoSleep").is_not_null())
    .with_columns(TotalSpent=pl.sum_horizontal(pl.col(pl.FLOAT_DTYPES)))
    .select(["CryoSleep", "TotalSpent"])
)
df_total.head(10)

# %%
# Passengers who spent money were NOT in cryo sleep
assert df_total.filter(pl.col("TotalSpent").gt(0.0)).get_column("CryoSleep").not_().all()

# %%
# Passengers who were in cryo sleep spent NO MONEY
assert df_total.filter(pl.col("CryoSleep")).get_column("TotalSpent").eq(0.0).all()

# %%
# The converse is NOT true: Some passengers who spent no money were awake
df_total.filter(pl.col("TotalSpent").eq(0.0)).get_column("CryoSleep").value_counts(sort=True)

# %%
del df_total

# %%
# Add TotalSpent column to DataFrame
df_train = df_train.with_columns(
    TotalSpent=pl.sum_horizontal("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
)
assert df_train.get_column("TotalSpent").ge(0.0).all()
df_train.select(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpent"]).head(10)

# %%
# Missing values BEFORE
df_train.get_column("CryoSleep").is_null().sum()

# %%
# Fill some missing CryoSleep values based on TotalSpent
df_cryo = (
    df_train.filter(pl.col("CryoSleep").is_null(), pl.col("TotalSpent").gt(0.0))
    .select(["PassengerId", "CryoSleep"])
    .with_columns(pl.col("CryoSleep").fill_null(False))  # noqa: FBT003
)
df_train = (
    df_train.join(df_cryo, on="PassengerId", how="left")
    .with_columns(pl.col("CryoSleep_right").fill_null(pl.col("CryoSleep")))
    .drop("CryoSleep")
    .rename({"CryoSleep_right": "CryoSleep"})
)
del df_cryo
assert df_train.filter(pl.col("TotalSpent").gt(0.0)).get_column("CryoSleep").not_().all()
df_train.head(10)

# %%
# Missing values AFTER
df_train.get_column("CryoSleep").is_null().sum()

# %%
# For the moment, ignore missing values that remain
cryo = df_train.get_column("CryoSleep").drop_nulls()

# %%
# Visualize number of passengers in cryo sleep
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(x=cryo.to_numpy(), order=[False, True], ax=ax)
ax.set_title("In cryo sleep?")
ax.set_xticks(ax.get_xticks())  # seems useless but silences a warning
ax.set_xticklabels(["No", "Yes"])
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.show()

# %%
cryo.value_counts(sort=True)

# %%
# Relationship between CryoSleep and other variables

# Get Pandas DataFrame
tmp_df = df_train.select(["CryoSleep", "Alone", "Transported"]).drop_nulls().to_pandas()
# tmp_df.head(10)

# %%
# Relationship between CryoSleep and Alone
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(tmp_df, x="CryoSleep", hue="Alone", order=[False, True], ax=ax)
ax.set_title("Relationship between CryoSleep and Alone")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.show()

# %%
pd.crosstab(tmp_df.CryoSleep, tmp_df.Alone)

# %%
# Relationship between CryoSleep and Transported
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(tmp_df, x="CryoSleep", hue="Transported", order=[False, True], ax=ax)
ax.set_title("Relationship between CryoSleep and Transported")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.show()

# %%
pd.crosstab(tmp_df.CryoSleep, tmp_df.Transported)

# %%
del tmp_df

# %% [markdown]
# ## `TotalSpent`

# %%
# Summary statistics
df_train.get_column("TotalSpent").describe()

# %%
# Power transformation of TotalSpent
pt_total_spent = PowerTransformer().fit_transform(df_train.select("TotalSpent").to_numpy()).flatten()

# %%
# Check that new feature is standardized
print(np.mean(pt_total_spent))
print(np.var(pt_total_spent))

# %%
# Visualize distribution of new feature
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.histplot(x=pt_total_spent, bins=10, stat="density", kde=True, ax=ax)
ax.set_xlabel("Power-Transformed Total Spent")
plt.show()

# %%
# Relationship with target variable
tmp_df = df_train.select("Transported").with_columns(
    pl.Series(name="PTTotalSpent", values=pt_total_spent, dtype=pl.Float64)
)
del pt_total_spent

fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.histplot(
    tmp_df,
    x="PTTotalSpent",
    hue="Transported",
    bins=10,
    stat="density",
    element="step",
    ax=ax,
)
ax.set_xlabel("Power-Transformed Total Spent")

del tmp_df
plt.show()

# %% [markdown]
# ## `Cabin`

# %%
# Split this feature to create 3 features
df_cabin = (
    df_train.select(["PassengerId", "Cabin"])
    .drop_nulls()
    .with_columns(pl.col("Cabin").str.split("/").list.to_struct().alias("CabinParts"))
    .drop("Cabin")
    .unnest("CabinParts")
    .rename(
        {
            "field_0": "CabinDeck",
            "field_1": "CabinNum",
            "field_2": "CabinSide",
        }
    )
    .with_columns(pl.col("CabinNum").str.to_integer(strict=True))
)
df_cabin.head(10)

# %%
# Quick check
assert df_cabin.get_column("CabinSide").is_in(["P", "S"]).all()

# %%
# Add new features to DataFrame
df_train = df_train.join(df_cabin, on="PassengerId", how="left")
del df_cabin
df_train.head(10)

# %%
# Unique values of CabinDeck
df_train.get_column("CabinDeck").unique()

# %%
df_train.get_column("CabinDeck").drop_nulls().value_counts(sort=True)

# %%
# Identify infrequent categories
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(
    df_train.select("CabinDeck").drop_nulls(),
    x="CabinDeck",
    order=["F", "G", "E", "B", "C", "D", "A", "T"],
    stat="percent",
    ax=ax,
)
ax.axhline(y=5, color="red", linestyle="--")
ax.set_xlabel("Deck")
ax.set_ylabel("Percentage")
ax.yaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

plt.show()

# %%
# Relationship between CabinDeck and Transported

# Get Pandas DataFrame
df_mosaic = (
    df_train.select(["CabinDeck", "Transported"])
    .drop_nulls()
    .filter(pl.col("CabinDeck").ne("T"))
    .sort("CabinDeck")
    .with_columns(Transported=pl.when(pl.col("Transported")).then(pl.lit("T")).otherwise(pl.lit("F")))
    .to_pandas()
)
df_mosaic.head(10)

# %%
pd.crosstab(df_mosaic.CabinDeck, df_mosaic.Transported, margins=True, margins_name="Total")

# %%
# Create mosaic plot
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

mosaic(
    df_mosaic,
    ["CabinDeck", "Transported"],
    title="Relationship between CabinDeck and Transported",
    labelizer=lambda _: "",
    ax=ax,
)
ax.set_xlabel("CabinDeck")
ax.set_ylabel("Transported")

del df_mosaic
plt.show()

# %%
# Combine two categories into one
tmp_df = (
    df_train.select(["CabinDeck", "Transported"])
    .drop_nulls()
    .with_columns(
        CabinDeck=pl.when(pl.col("CabinDeck").is_in(["A", "T"]))
        .then(pl.lit("Other"))
        .otherwise(pl.col("CabinDeck"))
    )
)

fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(
    tmp_df,
    x="CabinDeck",
    hue="Transported",
    order=["F", "G", "E", "B", "C", "D", "Other"],
    ax=ax,
)
ax.set_xlabel("Deck")
ax.set_ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %%
# Combine three categories into one
tmp_df = (
    df_train.select(["CabinDeck", "Transported"])
    .drop_nulls()
    .with_columns(
        CabinDeck=pl.when(pl.col("CabinDeck").is_in(["D", "A", "T"]))
        .then(pl.lit("Other"))
        .otherwise(pl.col("CabinDeck"))
    )
)

fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(
    tmp_df,
    x="CabinDeck",
    hue="Transported",
    order=["F", "G", "E", "B", "C", "Other"],
    ax=ax,
)
ax.set_xlabel("Deck")
ax.set_ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %%
# CabinNum: Number of unique values
df_train.get_column("CabinNum").drop_nulls().n_unique()

# %%
# Count encoding for CabinNum
cabin_counts = df_train.get_column("CabinNum").drop_nulls().value_counts().rename({"count": "CabinNumCount"})
df_train = df_train.join(cabin_counts, on="CabinNum", how="left")
del cabin_counts
# df_train.head(10)

# %%
# Visualize count encoding
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.violinplot(
    df_train.select(["CabinNumCount", "Transported"]).drop_nulls(),
    x="Transported",
    y="CabinNumCount",
    ax=ax,
)
ax.set_title("Violinplots of CabinNumCount")
plt.show()

# %%
# Passengers that belong to the same group were on the same side of the
# spaceship
df_train.select(["CabinSide", "Group"]).drop_nulls().group_by("Group").agg(
    pl.col("CabinSide").n_unique().alias("UniqueSides")
).get_column("UniqueSides").eq(1).all()

# %%
# Missing values BEFORE
df_train.get_column("CabinSide").is_null().sum()

# %%
# Fill some missing CabinSide values using group data
df_1 = df_train.select(["CabinSide", "Group"]).drop_nulls().group_by("Group").agg(pl.col("CabinSide").first())
df_2 = df_train.filter(
    pl.col("Alone").not_(), pl.col("CabinSide").is_null(), pl.col("Group").is_in(df_1.get_column("Group"))
).select(["PassengerId", "Group"])
df_3 = df_2.join(df_1, on="Group", how="inner").drop("Group")
df_train = (
    df_train.join(df_3, on="PassengerId", how="left")
    .with_columns(CabinSide=pl.col("CabinSide_right").fill_null(pl.col("CabinSide")))
    .drop("CabinSide_right")
)
del df_1, df_2, df_3

# %%
# Missing values AFTER
df_train.get_column("CabinSide").is_null().sum()

# %%
# Relationship between CabinSide and Transported
tmp_df = df_train.select(["CabinSide", "Transported"]).drop_nulls().to_pandas()
df_crosstab = pd.crosstab(tmp_df.CabinSide, tmp_df.Transported, margins=True, margins_name="Total")
del tmp_df
df_crosstab

# %%
# Plot passenger count
target_counts = {
    "False": df_crosstab.iloc[:-1, 0].to_numpy(),
    "True": df_crosstab.iloc[:-1, 1].to_numpy(),
}
del df_crosstab

width = 0.6
bottom = np.zeros(2)

fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

for target, count in target_counts.items():
    p = ax.bar(("P", "S"), count, width, label=target, bottom=bottom)
    bottom += count
    ax.bar_label(p, label_type="center")

ax.set_title("Relationship between CabinSide and Transported")
ax.legend(title="Transported")

plt.show()

# %% [markdown]
# ## `Destination`

# %%
# Unique values of Destination
df_train.get_column("Destination").drop_nulls().unique()

# %%
# Most of the time, passengers that belong to the same group have the same
# destination. But sometimes there are 2 or 3 different destinations:
df_dest = (
    df_train.select(["Group", "Destination"])
    .drop_nulls()
    .group_by("Group")
    .agg(pl.col("Destination").n_unique().alias("UniqueDestinations"))
)
df_dest.get_column("UniqueDestinations").value_counts(sort=True)

# %%
# Use Group data to fill some missing Destination values

# Missing values BEFORE
df_train.get_column("Destination").is_null().sum()

# %%
groups = df_dest.filter(pl.col("UniqueDestinations").eq(1)).get_column("Group")
del df_dest

df_1 = df_train.filter(pl.col("Destination").is_null(), pl.col("Group").is_in(groups)).select(
    ["PassengerId", "Group"]
)
df_2 = (
    df_train.filter(pl.col("Destination").is_not_null(), pl.col("Group").is_in(groups))
    .select(["Group", "Destination"])
    .unique()
)
df_3 = df_1.join(df_2, on="Group", how="inner").drop("Group")

del groups
del df_1
del df_2

df_train = (
    df_train.join(df_3, on="PassengerId", how="left")
    .with_columns(Destination=pl.col("Destination_right").fill_null(pl.col("Destination")))
    .drop("Destination_right")
)
del df_3

df_train.head(10)

# %%
# Missing values AFTER
df_train.get_column("Destination").is_null().sum()

# %%
# Number of passengers by destination
df_train.get_column("Destination").drop_nulls().value_counts(sort=True)

# %%
# Visualization
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(
    df_train.select("Destination").drop_nulls(),
    x="Destination",
    order=["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
    ax=ax,
)
ax.bar_label(ax.containers[0])  # pyright: ignore [reportArgumentType]
ax.set_title("Number of passengers by destination")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.show()

# %%
# Relationship with target variable
tmp_df = df_train.select(["Destination", "Transported"]).drop_nulls().to_pandas()
pd.crosstab(tmp_df.Destination, tmp_df.Transported, margins=True, margins_name="Total")

# %%
# Visualization
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(
    tmp_df,
    x="Destination",
    hue="Transported",
    order=["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
    ax=ax,
)
ax.set_title("Relationship between Destination and Transported")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %%
# Relationship between Destination and HomePlanet
df_mosaic = df_train.select(["HomePlanet", "Destination"]).drop_nulls().to_pandas()
pd.crosstab(df_mosaic.HomePlanet, df_mosaic.Destination, margins=True, margins_name="Total")

# %%
# Create mosaic plot
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

mosaic(
    df_mosaic,
    ["Destination", "HomePlanet"],
    title="Relationship between Destination and HomePlanet",
    labelizer=lambda _: "",
    ax=ax,
)

del df_mosaic
plt.show()

# %% [markdown]
# ## `Age`

# %%
# For now, drop missing values
age = df_train.get_column("Age").drop_nulls()
age.head()

# %%
# Consistency checks
assert age.ge(0.0).all()
assert (age - age.cast(pl.UInt32)).eq(0.0).all()

# %%
# Convert to integer
age = age.cast(pl.UInt32)
age.head()

# %%
# Summary statistics
age.describe()

# %%
# Create boxplot
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.boxplot(y=age.to_numpy(), ax=ax)
ax.set_title("Boxplot of Age")
ax.set_xticks([])
ax.set_ylabel("Age")

plt.show()

# %%
# Histogram and KDE
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.histplot(x=age.to_numpy(), bins=20, stat="density", kde=True, ax=ax)
ax.set_title("Histogram and KDE for Age")
ax.set_xlabel("Age")

plt.show()

# %%
# Relationship with the target variable

# Get Pandas DataFrame
tmp_df = (
    df_train.select(pl.col("Age"), pl.col("Transported"))
    .filter(pl.col("Age").is_not_null())
    .with_columns(pl.col("Age").cast(pl.UInt32))
    .to_pandas()
)
# tmp_df.head(10)

# %%
# Create boxplots
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.boxplot(tmp_df, x="Transported", y="Age", ax=ax)
ax.set_title("Boxplots of Age")

plt.show()

# %%
# Create histograms
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.histplot(
    tmp_df,
    x="Age",
    hue="Transported",
    bins=20,
    stat="density",
    element="step",
    ax=ax,
)
ax.set_title("Histograms of Age")

plt.show()

# %%
del tmp_df

# %%
# Comparing approaches to Age discretization
for n_bins, strategy in product([3, 4, 5], ["uniform", "quantile", "kmeans"]):
    print(f"Number of bins = {n_bins}")
    print(f"Strategy = {strategy}")

    discretizer = KBinsDiscretizer(n_bins=n_bins, strategy=strategy, encode="ordinal", random_state=333)
    disc_ages = discretizer.fit_transform(df_train.select("Age").drop_nulls().to_numpy()).flatten()

    print("Bin edges:")
    print(discretizer.bin_edges_)

    tmp_df = (
        df_train.filter(pl.col("Age").is_not_null())
        .select("Transported")
        .with_columns(pl.Series(name="DiscretizedAge", values=disc_ages, dtype=pl.UInt32))
    )

    fig = plt.figure(figsize=(10.0, 6.0), layout="tight")
    ax = fig.add_subplot()

    sns.countplot(tmp_df, x="DiscretizedAge", hue="Transported", ax=ax)
    ax.set_xlabel("Discretized age")
    ax.set_ylabel("Count")

    for container in ax.containers:
        ax.bar_label(container)  # pyright: ignore [reportArgumentType]

    plt.show()

# %% [markdown]
# ## `VIP`

# %%
# Most passengers don't have VIP status
df_train.get_column("VIP").drop_nulls().value_counts(sort=True)

# %%
# No VIP passenger is from Earth
df_train.select(["VIP", "HomePlanet"]).drop_nulls().filter(pl.col("VIP")).get_column("HomePlanet").ne(
    "Earth"
).all()

# %%
df_test.select(["VIP", "HomePlanet"]).drop_nulls().filter(pl.col("VIP")).get_column("HomePlanet").ne(
    "Earth"
).all()

# %%
# Fix some of the missing values of VIP

# Missing values BEFORE
df_train.get_column("VIP").is_null().sum()

# %%
tmp_df = (
    df_train.filter(pl.col("VIP").is_null(), pl.col("HomePlanet").eq("Earth"))
    .select("PassengerId")
    .with_columns(VIP=pl.lit(False))  # noqa: FBT003
)
df_train = (
    df_train.join(tmp_df, on="PassengerId", how="left")
    .with_columns(VIP=pl.col("VIP_right").fill_null(pl.col("VIP")))
    .drop("VIP_right")
)
del tmp_df

# %%
# Missing values AFTER
df_train.get_column("VIP").is_null().sum()

# %%
# Number of VIP passengers
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(df_train.select("VIP").drop_nulls(), x="VIP", ax=ax)
ax.bar_label(ax.containers[0])  # pyright: ignore [reportArgumentType]
ax.set_title("Number of VIP passengers")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.show()

# %%
# Relationship with target variable
tmp_df = df_train.select(["VIP", "Transported"]).drop_nulls().to_pandas()
pd.crosstab(tmp_df.VIP, tmp_df.Transported)

# %%
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(tmp_df, x="VIP", hue="Transported", ax=ax)
ax.set_title("Relationship between VIP and Transported")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %%
# Most VIP passengers were awake (not in cryo sleep)
df_train.select(["VIP", "CryoSleep"]).drop_nulls().filter(pl.col("VIP")).get_column("CryoSleep").value_counts(
    sort=True
)

# %%
# Relationship between VIP and TotalSpent
tmp_df = df_train.select(["VIP", "TotalSpent"]).drop_nulls()
fig = plt.figure(figsize=(9.0, 6.0), layout="tight")

ax_1 = fig.add_subplot(121)
sns.boxplot(tmp_df, x="VIP", y="TotalSpent", hue="VIP", ax=ax_1)
ax_1.set_title("Boxplot of Total Spent")
ax_1.set_ylabel("Total Spent")
ax_1.get_legend().remove()

ax_2 = fig.add_subplot(122, sharey=ax_1)
sns.violinplot(tmp_df, x="VIP", y="TotalSpent", hue="VIP", ax=ax_2)
ax_2.set_title("Violinplot of Total Spent")
ax_2.set_ylabel("")
plt.setp(ax_2.get_yticklabels(), visible=False)
ax_2.get_legend().remove()

del tmp_df
fig.suptitle("Relationship between VIP status and total amount spent")

plt.show()

# %% [markdown]
# ## `RoomService`

# %%
# Missing values BEFORE
df_train.get_column("RoomService").is_null().sum()

# %%
# Fill some missing values based on CryoSleep
df_1 = (
    df_train.filter(pl.col("CryoSleep"), pl.col("RoomService").is_null())
    .select("PassengerId")
    .with_columns(RoomService=pl.lit(0.0))
)
df_train = (
    df_train.join(df_1, on="PassengerId", how="left")
    .with_columns(RoomService=pl.col("RoomService_right").fill_null(pl.col("RoomService")))
    .drop("RoomService_right")
)
del df_1
df_train.head(10)

# %%
# Missing values AFTER
df_train.get_column("RoomService").is_null().sum()

# %%
# Summary statistics
df_train.get_column("RoomService").drop_nulls().describe()

# %%
# Clearly, this feature is dominated by zeros
df_train.get_column("RoomService").drop_nulls().eq(0.0).rename("EqualToZero").value_counts(sort=True)

# %%
# Power transformation of RoomService
pt_room_service = (
    PowerTransformer().fit_transform(df_train.select("RoomService").drop_nulls().to_numpy()).flatten()
)

# %%
# Visualize distribution of new feature
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.histplot(x=pt_room_service, bins=10, ax=ax)
ax.set_xlabel("Power-Transformed Room Service")
plt.show()

# %%
del pt_room_service

# %%
# Derive a binary feature from RoomService
tmp_df = (
    df_train.select(["RoomService", "Transported"])
    .drop_nulls()
    .with_columns(Spent=pl.col("RoomService").gt(0.0))
    .drop("RoomService")
)

fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(tmp_df, x="Spent", hue="Transported", ax=ax)
ax.set_ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %% [markdown]
# ## `FoodCourt`

# %%
# Missing values BEFORE
df_train.get_column("FoodCourt").is_null().sum()

# %%
# Fill some missing values based on CryoSleep
df_1 = (
    df_train.filter(pl.col("CryoSleep"), pl.col("FoodCourt").is_null())
    .select("PassengerId")
    .with_columns(FoodCourt=pl.lit(0.0))
)
df_train = (
    df_train.join(df_1, on="PassengerId", how="left")
    .with_columns(FoodCourt=pl.col("FoodCourt_right").fill_null(pl.col("FoodCourt")))
    .drop("FoodCourt_right")
)
del df_1
df_train.head(10)

# %%
# Missing values AFTER
df_train.get_column("FoodCourt").is_null().sum()

# %%
# Summary statistics
df_train.get_column("FoodCourt").drop_nulls().describe()

# %%
# Power transformation of FoodCourt
pt_food_court = (
    PowerTransformer().fit_transform(df_train.select("FoodCourt").drop_nulls().to_numpy()).flatten()
)

# %%
# Visualize distribution of new feature
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.histplot(x=pt_food_court, bins=10, ax=ax)
ax.set_xlabel("Power-Transformed Food Court")
plt.show()

# %%
del pt_food_court

# %%
# Derive a binary feature from FoodCourt
tmp_df = (
    df_train.select(["FoodCourt", "Transported"])
    .drop_nulls()
    .with_columns(Spent=pl.col("FoodCourt").gt(0.0))
    .drop("FoodCourt")
)

fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(tmp_df, x="Spent", hue="Transported", ax=ax)
ax.set_ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %% [markdown]
# ## `ShoppingMall`

# %%
# Missing values BEFORE
df_train.get_column("ShoppingMall").is_null().sum()

# %%
# Fill some missing values based on CryoSleep
df_1 = (
    df_train.filter(pl.col("CryoSleep"), pl.col("ShoppingMall").is_null())
    .select("PassengerId")
    .with_columns(ShoppingMall=pl.lit(0.0))
)
df_train = (
    df_train.join(df_1, on="PassengerId", how="left")
    .with_columns(ShoppingMall=pl.col("ShoppingMall_right").fill_null(pl.col("ShoppingMall")))
    .drop("ShoppingMall_right")
)
del df_1
df_train.head(10)

# %%
# Missing values AFTER
df_train.get_column("ShoppingMall").is_null().sum()

# %%
# Summary statistics
df_train.get_column("ShoppingMall").drop_nulls().describe()

# %%
# Power transformation of ShoppingMall
pt_shopping_mall = (
    PowerTransformer().fit_transform(df_train.select("ShoppingMall").drop_nulls().to_numpy()).flatten()
)

# %%
# Visualize distribution of new feature
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.histplot(x=pt_shopping_mall, bins=10, ax=ax)
ax.set_xlabel("Power-Transformed Shopping Mall")
plt.show()

# %%
del pt_shopping_mall

# %%
# Derive a binary feature from ShoppingMall
tmp_df = (
    df_train.select(["ShoppingMall", "Transported"])
    .drop_nulls()
    .with_columns(Spent=pl.col("ShoppingMall").gt(0.0))
    .drop("ShoppingMall")
)

fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(tmp_df, x="Spent", hue="Transported", ax=ax)
ax.set_ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %% [markdown]
# ## `Spa`

# %%
# Missing values BEFORE
df_train.get_column("Spa").is_null().sum()

# %%
# Fill some missing values based on CryoSleep
df_1 = (
    df_train.filter(pl.col("CryoSleep"), pl.col("Spa").is_null())
    .select("PassengerId")
    .with_columns(Spa=pl.lit(0.0))
)
df_train = (
    df_train.join(df_1, on="PassengerId", how="left")
    .with_columns(Spa=pl.col("Spa_right").fill_null(pl.col("Spa")))
    .drop("Spa_right")
)
del df_1
df_train.head(10)

# %%
# Missing values AFTER
df_train.get_column("Spa").is_null().sum()

# %%
# Summary statistics
df_train.get_column("Spa").drop_nulls().describe()

# %%
# Power transformation of Spa
pt_spa = PowerTransformer().fit_transform(df_train.select("Spa").drop_nulls().to_numpy()).flatten()

# %%
# Visualize distribution of new feature
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.histplot(x=pt_spa, bins=10, ax=ax)
ax.set_xlabel("Power-Transformed Spa")
plt.show()

# %%
del pt_spa

# %%
# Derive a binary feature from Spa
tmp_df = (
    df_train.select(["Spa", "Transported"]).drop_nulls().with_columns(Spent=pl.col("Spa").gt(0.0)).drop("Spa")
)

fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(tmp_df, x="Spent", hue="Transported", ax=ax)
ax.set_ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %% [markdown]
# ## `VRDeck`

# %%
# Missing values BEFORE
df_train.get_column("VRDeck").is_null().sum()

# %%
# Fill some missing values based on CryoSleep
df_1 = (
    df_train.filter(pl.col("CryoSleep"), pl.col("VRDeck").is_null())
    .select("PassengerId")
    .with_columns(VRDeck=pl.lit(0.0))
)
df_train = (
    df_train.join(df_1, on="PassengerId", how="left")
    .with_columns(VRDeck=pl.col("VRDeck_right").fill_null(pl.col("VRDeck")))
    .drop("VRDeck_right")
)
del df_1
df_train.head(10)

# %%
# Missing values AFTER
df_train.get_column("VRDeck").is_null().sum()

# %%
# Quick check
assert (
    df_train.filter(pl.col("CryoSleep"))
    .select(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpent"])
    .to_pandas()
    .eq(0.0)
    .all()
    .all()  # pyright: ignore [reportAttributeAccessIssue]
)

# %%
# Summary statistics
df_train.get_column("VRDeck").drop_nulls().describe()

# %%
# Power transformation of VRDeck
pt_vr_deck = PowerTransformer().fit_transform(df_train.select("VRDeck").drop_nulls().to_numpy()).flatten()

# %%
# Visualize distribution of new feature
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.histplot(x=pt_vr_deck, bins=10, ax=ax)
ax.set_xlabel("Power-Transformed VR Deck")
plt.show()

# %%
del pt_vr_deck

# %%
# Derive a binary feature from VRDeck
tmp_df = (
    df_train.select(["VRDeck", "Transported"])
    .drop_nulls()
    .with_columns(Spent=pl.col("VRDeck").gt(0.0))
    .drop("VRDeck")
)

fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()

sns.countplot(tmp_df, x="Spent", hue="Transported", ax=ax)
ax.set_ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)  # pyright: ignore [reportArgumentType]

del tmp_df
plt.show()

# %% [markdown]
# ## All numeric variables

# %%
# Correlation
df_numeric = df_train.select(cs.numeric().exclude("CabinNum")).drop_nulls()
corr = df_numeric.corr()
corr

# %%
# Create heatmap
fig = plt.figure(figsize=(8.0, 8.0), layout="tight")
ax = fig.add_subplot()

sns.heatmap(
    corr,
    xticklabels=corr.columns,
    yticklabels=corr.columns,
    annot=True,
    cmap=mpl.colormaps["coolwarm"],
    ax=ax,
)
ax.set_title("Heatmap of Correlations")

del corr
plt.show()

# %%
del df_numeric

# %% [markdown]
# ## `Name`

# %%
names = df_train.get_column("Name").drop_nulls()
names.head()

# %%
# Every name has 2 parts: the first name and 1 surname
assert names.str.split(" ").list.len().eq(2).all()

# %%
# Number of unique surnames
names.str.split(" ").list.last().n_unique()

# %%
# Add Surname column to DataFrame
df_surnames = (
    df_train.filter(pl.col("Name").is_not_null())
    .select(["PassengerId", "Name"])
    .with_columns(pl.col("Name").str.split(" ").list.last().alias("Surname"))
    .drop("Name")
)
df_train = df_train.join(df_surnames, on="PassengerId", how="left")
del df_surnames
df_train.select(["PassengerId", "Name", "Surname"]).head(10)

# %%
# Most of the time, passengers who belong to the same group are also part of
# the same family
df_train.select(["Group", "Surname"]).drop_nulls().group_by("Group").agg(
    pl.col("Surname").n_unique().alias("UniqueSurnames")
).with_columns(pl.col("UniqueSurnames").eq(1).alias("OnlyOneSurname")).get_column(
    "OnlyOneSurname"
).value_counts(sort=True)

# %%
# Count encoding for Surname
surname_counts = df_train.get_column("Surname").drop_nulls().value_counts().rename({"count": "SurnameCount"})
df_train = df_train.join(surname_counts, on="Surname", how="left")
del surname_counts
# df_train.head(20)

# %%
# Visualize count encoding
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.violinplot(
    df_train.select(["SurnameCount", "Transported"]).drop_nulls(),
    x="Transported",
    y="SurnameCount",
    ax=ax,
)
ax.set_title("Violinplots of SurnameCount")
plt.show()

# %%
# Passengers with the same surname are from the same planet
df_train.select(["Surname", "HomePlanet"]).drop_nulls().group_by("Surname").agg(
    pl.col("HomePlanet").n_unique().alias("UniqueHomePlanets")
).get_column("UniqueHomePlanets").eq(1).all()

# %%

# %%
with pl.Config(tbl_cols=df_train.width):
    display(df_train.null_count())
