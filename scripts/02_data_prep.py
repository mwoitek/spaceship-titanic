# %% [markdown]
# # Spaceship Titanic: Data Preparation
# ## Imports

# %%
import os
from pathlib import Path

import polars as pl
from IPython.display import display

# %%
VERBOSE = bool(int(os.environ.get("VERBOSE", "1")))

# %% [markdown]
# ## Read data

# %%
data_dir = Path.cwd() / "data"
assert data_dir.exists(), f"Directory doesn't exist: {data_dir}"

# %%
# Training data
df_train = pl.read_csv(data_dir / "train.csv")
if VERBOSE:
    display(df_train.head(10))

# %%
# Test data
df_test = pl.read_csv(data_dir / "test.csv")
if VERBOSE:
    display(df_test.head(10))

# %% [markdown]
# ## `Transported` (target variable)

# %%
# Convert target into an integer
assert df_train.get_column("Transported").null_count() == 0
df_train = df_train.with_columns(Transported=pl.col("Transported").cast(pl.UInt8))

# %% [markdown]
# ## `PassengerId`

# %%
assert df_train.get_column("PassengerId").null_count() == 0
assert df_test.get_column("PassengerId").null_count() == 0


# %%
def add_passengerid_features(df: pl.DataFrame) -> pl.DataFrame:
    if all(col in df.columns for col in ["Group", "CompanionCount", "Alone"]):
        return df

    # Group
    df = df.with_columns(Group=pl.col("PassengerId").str.split("_").list.first())
    cols = df.columns
    cols.insert(1, cols.pop())
    df = df.select(cols)

    # CompanionCount and Alone
    df_groups = (
        df.group_by("Group")
        .len()
        .rename({"len": "GroupSize"})
        .with_columns(
            CompanionCount=pl.col("GroupSize").sub(1),
            Alone=pl.col("GroupSize").eq(1).cast(pl.UInt8),
        )
        .drop("GroupSize")
    )
    df = df.join(df_groups, on="Group", how="left")
    cols = df.columns
    cols.insert(2, cols.pop())
    cols.insert(2, cols.pop())
    df = df.select(cols)

    return df


# %%
df_train = add_passengerid_features(df_train)
if VERBOSE:
    display(df_train.head(10))

# %%
df_test = add_passengerid_features(df_test)
if VERBOSE:
    display(df_test.head(10))

# %% [markdown]
# ## Encoding `CompanionCount`

# %%
cc_vals = list(range(8))
assert df_train.get_column("CompanionCount").is_in(cc_vals).all()
assert df_test.get_column("CompanionCount").is_in(cc_vals).all()
# %xdel cc_vals


# %%
# Every count above 2 is considered to be the same
def reduce_companioncount(df: pl.DataFrame) -> pl.DataFrame:
    if "CompCntReduced" in df.columns:
        return df

    if "CompanionCount" not in df.columns:
        msg = "Column `CompanionCount` has to be created first"
        raise ValueError(msg)

    df = df.with_columns(
        CompCntReduced=pl.when(pl.col("CompanionCount").gt(2))
        .then(pl.lit(3))
        .otherwise(pl.col("CompanionCount"))
        .cast(pl.UInt8)
    )
    cols = df.columns
    cols.insert(3, cols.pop())
    df = df.select(cols)

    return df


# %%
# Frequency encoding
def frequency_encode_companioncount(
    df: pl.DataFrame,
    freq: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    if "CompCntFreq" in df.columns:
        return df, freq

    if "CompanionCount" not in df.columns:
        msg = "Column `CompanionCount` has to be created first"
        raise ValueError(msg)

    if freq is None:
        freq = (
            df.get_column("CompanionCount")
            .value_counts(normalize=True, name="CompCntFreq")
            .sort(by="CompanionCount")
        )

    df = df.join(freq, on="CompanionCount", how="left")
    cols = df.columns
    cols.insert(4, cols.pop())
    df = df.select(cols)

    return df, freq


# %%
# Training data: Run all encoding functions
df_train = reduce_companioncount(df_train)
df_train, companion_freq = frequency_encode_companioncount(df_train)

# %%
# Test data: Run all encoding functions
df_test = reduce_companioncount(df_test)
df_test, _ = frequency_encode_companioncount(df_test, companion_freq)

# %%
# %xdel companion_freq

# %%

# %%

# %%

# %%
