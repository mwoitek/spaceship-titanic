# %% [markdown]
# # Spaceship Titanic: Data Preparation
# ## Imports

# %%
from pathlib import Path

import polars as pl
from IPython.display import display

# %% [markdown]
# ## Read data

# %%
data_dir = Path.cwd() / "data"
assert data_dir.exists(), f"Directory doesn't exist: {data_dir}"

# %%
# Training data
df_train = pl.read_csv(data_dir / "train.csv")
display(df_train.head(10))

# %%
# Test data
df_test = pl.read_csv(data_dir / "test.csv")
display(df_test.head(10))

# %% [markdown]
# ## `Transported` (target variable)

# %%
# Convert target into an integer
assert df_train.get_column("Transported").null_count() == 0
df_train = df_train.with_columns(Transported=pl.col("Transported").cast(pl.Int8))

# %%
# HERE
