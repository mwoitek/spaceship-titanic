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
