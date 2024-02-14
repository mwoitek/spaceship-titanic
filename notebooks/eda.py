# %% [markdown]
# # Spaceship Titanic: Exploratory Data Analysis
# ## Imports

# %%
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

# %%
pl.Config.set_tbl_cols(14)

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
fig = cast(Figure, fig)

ax = fig.add_subplot()
ax = cast(Axes, ax)

sns.countplot(x=df_train.get_column("Alone").to_numpy(), order=[True, False], ax=ax)
ax.set_title("Traveling alone?")
ax.set_xticklabels(["Yes", "No"])
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.show()

# %%
df_train.get_column("Alone").value_counts(sort=True)

# %%
# Unique values of `CompanionCount`
companion_count = df_train.get_column("CompanionCount")
companion_count.unique()

# %%
# Visualizing number of companions
fig = plt.figure(figsize=(8.0, 6.0), layout="tight")
fig = cast(Figure, fig)

ax = fig.add_subplot()
ax = cast(Axes, ax)

pos_counts = companion_count.filter(companion_count.gt(0)).to_numpy()
sns.countplot(x=pos_counts, order=list(range(1, 8)), ax=ax)
ax.set_title("Number of companions for those who had company")
ax.set_xlabel("Number of companions")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.show()

# %%
companion_count.value_counts().sort(by="CompanionCount")

# %% [markdown]
# ## `HomePlanet`

# %%
# Unique values
home_planet = df_train.get_column("HomePlanet")
home_planet.unique()

# %%
# For the moment, ignore missing values
home_planet = home_planet.drop_nulls()

# %%
# Visualizing number of passengers by home planet
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
fig = cast(Figure, fig)

ax = fig.add_subplot()
ax = cast(Axes, ax)

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
fig = cast(Figure, fig)

ax = fig.add_subplot()
ax = cast(Axes, ax)

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
