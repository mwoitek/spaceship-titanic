# %% [markdown]
# # Spaceship Titanic: Exploratory Data Analysis
# ## Imports

# %%
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from pprint import pprint
from typing import cast

import matplotlib.pyplot as plt
import mpmath
import numpy as np
import polars as pl
import seaborn as sns
from IPython.display import display
from matplotlib.container import BarContainer
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, PercentFormatter
from scipy.stats import chi2_contingency
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.graphics.mosaicplot import mosaic

# %%
# Configuring imports
mpmath.mp.dps = 50
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["figure.figsize"] = (6.0, 6.0)

# %% [markdown]
# ## Read data

# %%
# Directory containing the data
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
cols = df_train.columns
cols.insert(2, cols.pop())
cols.insert(2, cols.pop())
df_train = df_train.select(cols)
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

# %% [markdown]
# ### `CompanionCount` and `Transported`: Independence test


# %%
# Create a contingency table with Polars
def contingency_table(df: pl.DataFrame, row: str, col: str) -> pl.DataFrame:
    if row not in df.columns:
        msg = f"Column {row} does not exist"
        raise ValueError(msg)
    if col not in df.columns:
        msg = f"Column {col} does not exist"
        raise ValueError(msg)
    tbl = (
        df.select(["PassengerId", row, col])
        .drop_nulls()
        .pivot(col, index=row, aggregate_function="len")
        .sort(by=row)
    )
    cols = sorted(tbl.columns[1:])
    cols.insert(0, row)
    return tbl.select(cols)


# %%
# Compute contingency table
ct = contingency_table(df_train, row="Transported", col="CompanionCount")
with pl.Config(tbl_cols=ct.width):
    display(ct)

# %%
# Compute expected frequencies
obs_vals = ct.select(ct.columns[1:]).to_numpy()
row_sum = np.sum(obs_vals, keepdims=True, axis=1)
row_sum = np.tile(row_sum, (1, obs_vals.shape[1]))
col_sum = np.sum(obs_vals, keepdims=True, axis=0)
col_sum = np.tile(col_sum, (obs_vals.shape[0], 1))
tbl_sum = np.sum(obs_vals)
exp_vals = row_sum * col_sum / tbl_sum
print(exp_vals)


# %%
# CDF for the chi-squared distribution
def chi2_cdf(x: int | float, dof: int) -> Decimal:
    x, dof = mpmath.mpf(x), mpmath.mpf(dof)
    cdf = mpmath.gammainc(dof / 2, 0, x / 2, regularized=True)
    return Decimal(str(cdf))


# %%
# Compute test statistic and p-value
test_stat = np.sum((obs_vals - exp_vals) ** 2 / exp_vals)
print(f"Test statistic: {test_stat}")
dof = (obs_vals.shape[0] - 1) * (obs_vals.shape[1] - 1)
pvalue = 1 - chi2_cdf(test_stat, dof)
print(f"p-value: {pvalue}")

# %%
# Checking the above calculations
res = chi2_contingency(obs_vals)
print(f"Test statistic: {res.statistic}")
print(f"p-value: {res.pvalue}")


# %%
# Chi-square test of independence
def chi2_independence_test(df: pl.DataFrame, var1: str, var2: str, alpha: float = 0.05) -> None:
    ct = contingency_table(df, row=var1, col=var2)
    obs_vals = ct.select(ct.columns[1:]).to_numpy()
    res = chi2_contingency(obs_vals)
    print("Chi-square test of independence")
    print(f"Null hypothesis: {var1} and {var2} are independent")
    print(f"Test statistic: {res.statistic}")
    print(f"p-value: {res.pvalue}")
    decision = "REJECT" if res.pvalue <= alpha else "FAIL TO REJECT"
    print(f"{decision} the null hypothesis (Significance level: {alpha})")


# %%
# Test the above function
chi2_independence_test(df_train, "Transported", "CompanionCount")

# %% [markdown]
# ### `CompanionCount`: Dealing with infrequent counts

# %%
# Identifying infrequent counts
percent_cutoff = 5
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
ax.axhline(y=percent_cutoff, color="red", linestyle="--")
ax.set_title("Identifying infrequent companion counts")
ax.set_xlabel("Number of companions")
ax.set_ylabel("Percentage")
ax.yaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.show()

# %%
# Compute frequencies related to `CompanionCount`
companion_freq = (
    df_train.get_column("CompanionCount").value_counts(normalize=True).sort(by="CompanionCount")
)
display(companion_freq)

# %%
# Select rare values
companion_rare = (
    companion_freq.filter(pl.col("proportion") < percent_cutoff / 100)
    .get_column("CompanionCount")
    .to_list()
)
print(companion_rare)

# %%
# Combine infrequent counts into a single category
df_train = df_train.with_columns(
    CompCntReduced=pl.when(pl.col("CompanionCount").gt(2))
    .then(pl.lit("3+"))
    .otherwise(pl.col("CompanionCount").cast(pl.String))
)
cols = df_train.columns
cols.insert(3, cols.pop())
df_train = df_train.select(cols)

# Checking
display(df_train.select(["CompanionCount", "CompCntReduced"]).head(10))
display(
    df_train.filter(pl.col("CompanionCount") > 2)
    .select(["CompanionCount", "CompCntReduced"])
    .head(10)
)

# %%
# Relationship between `CompCntReduced` and `Transported`
fig, ax = plt.subplots(figsize=(8.0, 6.0))
sns.countplot(
    df_train,
    x="CompCntReduced",
    order=["0", "1", "2", "3+"],
    hue="Transported",
    ax=ax,
)
for container in ax.containers:
    container = cast(BarContainer, container)
    ax.bar_label(container)
ax.set_title("Relationship between CompCntReduced and Transported")
ax.set_xlabel("Number of companions")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
plt.show()

# %% [markdown]
# ## `HomePlanet`

# %%
# Unique values
display(df_train.get_column("HomePlanet").unique())

# %%
# Do passengers who belong to the same group also come from the same home planet?
df_train.select(["Group", "HomePlanet"]).drop_nulls().group_by("Group").agg(
    UniquePlanets=pl.col("HomePlanet").n_unique()
).get_column("UniquePlanets").eq(1).all()

# %%
# Identify rows that can be fixed
# - `HomePlanet` is missing;
# - Person is part of a group.
df_1 = df_train.filter(pl.col("HomePlanet").is_null(), pl.col("Alone").eq(0)).select(
    ["PassengerId", "Group"]
)
df_2 = (
    df_train.filter(
        pl.col("HomePlanet").is_not_null(),
        pl.col("Group").is_in(df_1.get_column("Group").unique()),
    )
    .select(["Group", "HomePlanet"])
    .unique()
)
df_3 = df_1.join(df_2, on="Group", how="inner").select(["PassengerId", "HomePlanet"])
display(df_3.head(10))

# %%
# Update DataFrame with new values of `HomePlanet`
print(f"Current number of missing values: {df_train.get_column('HomePlanet').null_count()}")
print(f"Number of rows that will be fixed: {df_3.height}")
col_idx = df_train.columns.index("HomePlanet")
df_train = (
    df_train.join(df_3, on="PassengerId", how="left")
    .with_columns(pl.col("HomePlanet_right").fill_null(pl.col("HomePlanet")))
    .drop("HomePlanet")
    .rename({"HomePlanet_right": "HomePlanet"})
)
print(f"Current number of missing values: {df_train.get_column('HomePlanet').null_count()}")
cols = df_train.columns
cols.insert(col_idx, cols.pop())
df_train = df_train.select(cols)

# %% [markdown]
# ### Visualizing `HomePlanet`

# %%
# Number of passengers by home planet
fig, ax = plt.subplots()
sns.countplot(
    df_train.filter(pl.col("HomePlanet").is_not_null()),
    x="HomePlanet",
    order=["Earth", "Europa", "Mars"],
    ax=ax,
)
container = cast(BarContainer, ax.containers[0])
ax.bar_label(container)
ax.set_title("Number of passengers by home planet")
ax.set_xlabel("")
ax.set_ylabel("Number of passengers")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.show()

# %%
# Relationship with the target variable
fig, ax = plt.subplots(figsize=(8.0, 6.0))
sns.countplot(
    df_train.filter(pl.col("HomePlanet").is_not_null()),
    x="HomePlanet",
    order=["Earth", "Europa", "Mars"],
    hue="Transported",
    ax=ax,
)
for container in ax.containers:
    container = cast(BarContainer, container)
    ax.bar_label(container)
ax.set_title("Relationship between HomePlanet and Transported")
ax.set_xlabel("")
ax.set_ylabel("Number of passengers")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
plt.show()

# %%
chi2_independence_test(df_train, "Transported", "HomePlanet")

# %% [markdown]
# ## `TotalSpent`

# %%
# Add `TotalSpent` column to DataFrame
money_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df_train = df_train.with_columns(TotalSpent=pl.sum_horizontal(*money_cols))
cols = df_train.columns
cols.insert(cols.index("Name"), cols.pop())
df_train = df_train.select(cols)
assert df_train.get_column("TotalSpent").ge(0.0).all()
display(df_train.select([*money_cols, "TotalSpent"]).head(10))

# %% [markdown]
# ## `CryoSleep`
# ### Consistency tests

# %%
# For these tests, ignore missing values
df_cryo = df_train.select(["CryoSleep", "TotalSpent"]).drop_nulls()
display(df_cryo.head(10))

# %%
# Passengers who spent money were NOT in cryo sleep
assert df_cryo.filter(pl.col("TotalSpent").gt(0.0)).get_column("CryoSleep").not_().all()

# %%
# Passengers who were in cryo sleep spent NO MONEY
assert df_cryo.filter(pl.col("CryoSleep")).get_column("TotalSpent").eq(0.0).all()

# %%
# The converse is NOT true: Some passengers who spent no money were awake
df_cryo.filter(pl.col("TotalSpent").eq(0.0)).get_column("CryoSleep").value_counts(sort=True)

# %% [markdown]
# ### Deal with some of the missing values

# %%
# Identify rows that can be fixed
# Idea: A passenger is NOT in cryo sleep if he is spending money
df_cryo = (
    df_train.filter(pl.col("CryoSleep").is_null(), pl.col("TotalSpent").gt(0.0))
    .with_columns(CryoSleep=pl.lit(False))  # noqa: FBT003
    .select(["PassengerId", "CryoSleep", "TotalSpent"])
)
display(df_cryo.head(10))
df_cryo = df_cryo.drop("TotalSpent")

# %%
# Fill some missing `CryoSleep` values based on `TotalSpent`
print(f"Current number of missing values: {df_train.get_column('CryoSleep').null_count()}")
print(f"Number of rows that will be fixed: {df_cryo.height}")
df_train = (
    df_train.join(df_cryo, on="PassengerId", how="left")
    .with_columns(pl.col("CryoSleep_right").fill_null(pl.col("CryoSleep")))
    .drop("CryoSleep")
    .rename({"CryoSleep_right": "CryoSleep"})
)
print(f"Current number of missing values: {df_train.get_column('CryoSleep').null_count()}")
cols = df_train.columns
cols.insert(cols.index("Cabin"), cols.pop())
df_train = df_train.select(cols)

# %%
# Check: Data is still consistent
assert df_train.filter(pl.col("TotalSpent").gt(0.0)).get_column("CryoSleep").not_().all()

# %% [markdown]
# ### Visualizing `CryoSleep`

# %%
# Number of passengers in cryo sleep
fig, ax = plt.subplots()
sns.countplot(
    df_train.filter(pl.col("CryoSleep").is_not_null()),
    x="CryoSleep",
    order=[False, True],
    ax=ax,
)
container = cast(BarContainer, ax.containers[0])
ax.bar_label(container)
ax.set_title("In cryo sleep?")
ax.set_xticks(ax.get_xticks())  # seems useless but silences a warning
ax.set_xticklabels(["No", "Yes"])
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.show()

# %%
# Relationship between `CryoSleep` and `Transported`
fig, ax = plt.subplots()
sns.countplot(
    df_train.filter(pl.col("CryoSleep").is_not_null()),
    x="CryoSleep",
    order=[False, True],
    hue="Transported",
    ax=ax,
)
for container in ax.containers:
    container = cast(BarContainer, container)
    ax.bar_label(container)
ax.set_title("Relationship between CryoSleep and Transported")
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(["No", "Yes"])
ax.set_xlabel("In cryo sleep?")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
plt.show()

# %% [markdown]
# ### Relationship with other categorical variables
# #### `Alone`

# %%
# Prepare data to create a mosaic plot
ct = contingency_table(df_train, "CryoSleep", "Alone")
ct_np = ct.select(ct.columns[1:]).to_numpy()
x_labels = ["Accompanied", "Alone"]
y_labels = ["Awake", "Asleep"]
data = {(x, y): ct_np[j, i] for j, y in enumerate(y_labels) for i, x in enumerate(x_labels)}
pprint(data)

# %%
# Create the mosaic plot
fig, ax = plt.subplots(figsize=(7.0, 7.0))
mosaic(data, labelizer=lambda k: data[k], ax=ax)
text_iter = filter(lambda c: isinstance(c, Text) and c.get_text() != "", ax.get_children())
text_iter = cast(Iterator[Text], text_iter)
for text in text_iter:
    text.set_color("w")
    text.set_fontsize(20)
ax.set_title("Relationship between CryoSleep and Alone")
plt.show()

# %%
# Interaction plot: CryoSleep and Alone
# NOTE: Parallel lines indicate NO INTERACTION
fig, ax = plt.subplots()
df_plot = df_train.filter(pl.col("CryoSleep").is_not_null()).select(
    ["Alone", "CryoSleep", "Transported"]
)
interaction_plot(
    x=df_plot.get_column("CryoSleep"),
    trace=df_plot.get_column("Alone"),
    response=df_plot.get_column("Transported"),
    xlabel="In cryo sleep?",
    ax=ax,
)
ax.set_title("Interaction plot: CryoSleep and Alone")
ax.set_xticks([0, 1], labels=["No", "Yes"])
ax.set_ylabel("Proportion of positive cases")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["No", "Yes"], title="Alone?")
plt.show()

# %% [markdown]
# #### `HomePlanet`

# %%
# Mosaic plot
ct = contingency_table(df_train, "CryoSleep", "HomePlanet")
ct_np = ct.select(ct.columns[1:]).to_numpy()
x_labels = ct.columns[1:]
y_labels = ["Awake", "Asleep"]
data = {(x, y): ct_np[j, i] for j, y in enumerate(y_labels) for i, x in enumerate(x_labels)}
fig, ax = plt.subplots(figsize=(7.0, 7.0))
mosaic(data, labelizer=lambda k: data[k], ax=ax)
text_iter = filter(lambda c: isinstance(c, Text) and c.get_text() != "", ax.get_children())
text_iter = cast(Iterator[Text], text_iter)
for text in text_iter:
    text.set_color("w")
    text.set_fontsize(20)
ax.set_title("Relationship between CryoSleep and HomePlanet")
plt.show()

# %%
# Interaction plot
fig, ax = plt.subplots()
df_plot = df_train.select(["HomePlanet", "CryoSleep", "Transported"]).drop_nulls()
interaction_plot(
    x=df_plot.get_column("CryoSleep"),
    trace=df_plot.get_column("HomePlanet"),
    response=df_plot.get_column("Transported"),
    xlabel="In cryo sleep?",
    ax=ax,
)
ax.set_title("Interaction plot: CryoSleep and HomePlanet")
ax.set_xticks([0, 1], labels=["No", "Yes"])
ax.set_ylabel("Proportion of positive cases")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Home planet")
plt.show()

# %%

# %%
