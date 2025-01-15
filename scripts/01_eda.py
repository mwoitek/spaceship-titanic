# %% [markdown]
# # Spaceship Titanic: Exploratory Data Analysis
# ## Imports

# %%
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from IPython.display import display
from matplotlib.container import BarContainer
from matplotlib.ticker import AutoMinorLocator, PercentFormatter

# %%
# matplotlib config
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["figure.figsize"] = (6.0, 6.0)

# %% [markdown]
# ## Read data

# %%
data_dir = Path.cwd() / "data"
assert data_dir.exists(), f"Directory doesn't exist: {data_dir}"

# %%
# Training data
df_train = pl.read_csv(data_dir / "train.csv")
display(df_train.head(10))

# %% [markdown]
# ## Basic information

# %%
# Number of observations
print(df_train.height)

# %%
# Missing values
with pl.Config(tbl_cols=df_train.width):
    display(df_train.null_count())

# %% [markdown]
# ## `Transported` (target variable)

# %%
# It's important to know if we have a balanced dataset or not
fig, ax = plt.subplots()
sns.countplot(df_train, x="Transported", order=[True, False], ax=ax)
container = cast(BarContainer, ax.containers[0])
ax.bar_label(container)
ax.set_title("Transported: Is the dataset balanced?")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.show()

# %%
# Same thing, but show percentages
fig, ax = plt.subplots()
sns.countplot(df_train, x="Transported", order=[True, False], stat="percent", ax=ax)
container = cast(BarContainer, ax.containers[0])
ax.bar_label(container, fmt="%.2f%%")
ax.set_title("Transported: Is the dataset balanced?")
ax.set_ylabel("Percentage")
ax.yaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.show()

# %% [markdown]
# ## `PassengerId`
# ### Groups

# %%
# Extract groups
df_train = df_train.with_columns(Group=pl.col("PassengerId").str.split("_").list.first())
cols = df_train.columns
cols.insert(1, cols.pop())
df_train = df_train.select(cols)
# %xdel cols
display(df_train.select(["PassengerId", "Group"]).head(10))

# %%
# Number of unique groups
print(df_train.get_column("Group").n_unique())

# %% [markdown]
# ### New features: `CompanionCount` and `Alone`

# %%
# Create a couple of features from `Group`
df_groups = (
    df_train.group_by("Group")
    .len()
    .rename({"len": "GroupSize"})
    .with_columns(
        CompanionCount=pl.col("GroupSize").sub(1),
        Alone=pl.col("GroupSize").eq(1).cast(pl.UInt8),
    )
    .drop("GroupSize")
)
df_train = df_train.join(df_groups, on="Group", how="left")
# %xdel df_groups
cols = df_train.columns
cols.insert(2, cols.pop())
cols.insert(2, cols.pop())
df_train = df_train.select(cols)
# %xdel cols
display(df_train.select(["PassengerId", "Group", "CompanionCount", "Alone"]).head(10))

# %% [markdown]
# ### Visualizing `Alone`

# %%
# Number of people traveling alone
fig, ax = plt.subplots()
sns.countplot(df_train, x="Alone", order=[1, 0], ax=ax)
container = cast(BarContainer, ax.containers[0])
ax.bar_label(container)
ax.set_title("Traveling alone?")
ax.set_xticks(ax.get_xticks())  # seems useless but silences a warning
ax.set_xticklabels(["Yes", "No"])
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.show()

# %%
# Relationship with the target variable
fig, ax = plt.subplots(figsize=(8.0, 6.0))
sns.countplot(df_train, x="Alone", order=[1, 0], hue="Transported", ax=ax)
for container in ax.containers:
    container = cast(BarContainer, container)
    ax.bar_label(container)
ax.set_title("Relationship between Alone and Transported")
ax.set_xticks(ax.get_xticks())  # seems useless but silences a warning
ax.set_xticklabels(["Yes", "No"])
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.show()

# %% [markdown]
# ### Visualizing `CompanionCount`

# %%
# Unique values
display(df_train.get_column("CompanionCount").unique())

# %%
# Number of companions for those who had company
fig, ax = plt.subplots(figsize=(8.0, 6.0))
sns.countplot(
    df_train.filter(pl.col("CompanionCount").gt(0)),
    x="CompanionCount",
    order=list(range(1, 8)),
    ax=ax,
)
container = cast(BarContainer, ax.containers[0])
ax.bar_label(container)
ax.set_title("Number of companions for those who had company")
ax.set_xlabel("Number of companions")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.show()

# %%
# Relationship with the target variable
fig, ax = plt.subplots(figsize=(10.0, 6.0))
sns.countplot(
    df_train,
    x="CompanionCount",
    order=list(range(8)),
    hue="Transported",
    ax=ax,
)
for container in ax.containers:
    container = cast(BarContainer, container)
    ax.bar_label(container)
ax.set_title("Relationship between CompanionCount and Transported")
ax.set_xlabel("Number of companions")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
plt.show()

# %%
# Identifying infrequent counts
fig, ax = plt.subplots(figsize=(8.0, 6.0))
sns.countplot(
    df_train,
    x="CompanionCount",
    order=list(range(8)),
    stat="percent",
    ax=ax,
)
container = cast(BarContainer, ax.containers[0])
ax.bar_label(container, fmt="%.2f%%")
ax.axhline(y=5, color="red", linestyle="--")
ax.set_title("Identifying infrequent companion counts")
ax.set_xlabel("Number of companions")
ax.set_ylabel("Percentage")
ax.yaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.show()

# %%
# plt.close("all")
