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
# # Spaceship Titanic: Feature Selection
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
# Training data (Imputed)
df_train = pd.read_csv(
    data_dir / "train_imputed.csv",
    index_col="PassengerId",
    dtype={
        "CompCntReduced": pd.CategoricalDtype(categories=["0", "1", "2", "3+"], ordered=True),
        "HomePlanetOrd": pd.CategoricalDtype(categories=["0", "1", "2"]),
        "CabinDeckOrd": pd.CategoricalDtype(categories=["0", "1", "2", "3", "4", "5"]),
        "DestinationOrd": pd.CategoricalDtype(categories=["0", "1", "2"]),
        "DiscretizedAge4": pd.CategoricalDtype(categories=["0", "1", "2", "3"], ordered=True),
        "DiscretizedAge5": pd.CategoricalDtype(categories=["0", "1", "2", "3", "4"], ordered=True),
    },
)
df_train.head(10)

# %%
df_train.info()

# %%
assert df_train.isna().sum().eq(0).all()

# %%
