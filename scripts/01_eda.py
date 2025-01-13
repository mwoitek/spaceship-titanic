# %% [markdown]
# # Spaceship Titanic: Exploratory Data Analysis
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
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
df_train.head(10)

# %% [markdown]
# ## Basic information

# %%
# Number of observations
df_train.height

# %%
# Missing values
with pl.Config(tbl_cols=df_train.width):
    print(df_train.null_count())

# %% [markdown]
# ## `Transported` (target variable)

# %%
# It's important to know if we have a balanced dataset or not
fig, ax = plt.subplots()
sns.countplot(df_train, x="Transported", order=[True, False], ax=ax)
ax.bar_label(ax.containers[0])  # pyright: ignore [reportArgumentType]
ax.set_title("Transported: Is the dataset balanced?")
ax.set_ylabel("Count")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.show()

# %%
# Same thing, but show percentages
fig, ax = plt.subplots()
sns.countplot(df_train, x="Transported", order=[True, False], stat="percent", ax=ax)
ax.bar_label(ax.containers[0], fmt="%.2f%%")  # pyright: ignore [reportArgumentType]
ax.set_title("Transported: Is the dataset balanced?")
ax.set_ylabel("Percentage")
ax.yaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.show()

# HERE
