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
# # Spaceship Titanic: Logistic Regression
# ## Imports

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# %%
warnings.simplefilter(action="ignore", category=FutureWarning)

# %% [markdown]
# ## Read data

# %%
data_dir = Path.cwd().parent / "input" / "spaceship-titanic"
assert data_dir.exists(), f"directory doesn't exist: {data_dir}"

# %%
df = pd.read_csv(
    data_dir / "train_imputed.csv",
    index_col="PassengerId",
    dtype={"CompCntReduced": pd.CategoricalDtype(categories=["0", "1", "2", "3+"], ordered=True)},
).assign(CompCntReduced=lambda x: x.CompCntReduced.cat.codes)
df.head(10)

# %%
df.info()

# %% [markdown]
# ## Categorical features: One-hot encoding
# Figuring out how I'm going to do this:

# %%
feat = "HomePlanetOrd"
uniq_vals: list[int] = np.sort(df[feat].unique()).tolist()

# %%
encoder = OneHotEncoder(
    categories=[uniq_vals],  # pyright: ignore [reportArgumentType]
    sparse_output=False,
    dtype=np.int8,
)
one_hot_encoded = encoder.fit_transform(df[[feat]])

# %%
new_cols = [f"{feat}_{val}" for val in uniq_vals]
df_one_hot = pd.DataFrame(one_hot_encoded, columns=new_cols, index=df.index)

# %%
df = df.merge(df_one_hot, left_index=True, right_index=True)
df[[feat, *new_cols]].head(20)

# %%
df = df.drop(columns=feat)

# %% [markdown]
# Turn the above code into a function:


# %%
def one_hot_encode(df: pd.DataFrame, feat: str) -> pd.DataFrame:
    uniq_vals = np.sort(df[feat].unique()).tolist()

    encoder = OneHotEncoder(
        categories=[uniq_vals],  # pyright: ignore [reportArgumentType]
        sparse_output=False,
        dtype=np.int8,
    )
    df_one_hot = pd.DataFrame(
        encoder.fit_transform(df[[feat]]),
        columns=[f"{feat}_{val}" for val in uniq_vals],
        index=df.index,
    )

    df = df.merge(df_one_hot, left_index=True, right_index=True).drop(columns=feat)
    return df


# %%
cat_cols = [
    "CompCntReduced",
    "CabinDeckOrd",
    "DestinationOrd",
    "DiscretizedAge4",
    "DiscretizedAge5",
]
for col in cat_cols:
    df = one_hot_encode(df, col)

# %%
# Re-order DataFrame columns
cols = [
    "Alone",
    "CompCntReduced_0",
    "CompCntReduced_1",
    "CompCntReduced_2",
    "CompCntReduced_3",
    "HomePlanetOrd_0",
    "HomePlanetOrd_1",
    "HomePlanetOrd_2",
    "CryoSleep",
    "CabinDeckOrd_0",
    "CabinDeckOrd_1",
    "CabinDeckOrd_2",
    "CabinDeckOrd_3",
    "CabinDeckOrd_4",
    "CabinDeckOrd_5",
    "CabinPort",
    "DestinationOrd_0",
    "DestinationOrd_1",
    "DestinationOrd_2",
    "DiscretizedAge4_0",
    "DiscretizedAge4_1",
    "DiscretizedAge4_2",
    "DiscretizedAge4_3",
    "DiscretizedAge5_0",
    "DiscretizedAge5_1",
    "DiscretizedAge5_2",
    "DiscretizedAge5_3",
    "DiscretizedAge5_4",
    "VIP",
    "PosRoomService",
    "PosFoodCourt",
    "PosShoppingMall",
    "PosSpa",
    "PosVRDeck",
    "PTTotalSpent",
    "Transported",
]
df = df[cols]
df.info()

# %%
# Convert boolean columns
bool_cols = df.select_dtypes(include=[bool]).columns.values.tolist()  # pyright: ignore [reportArgumentType]
df.loc[:, bool_cols] = df[bool_cols].astype(np.int8)
df.info()

# %%
df = df.astype(np.float_)
df.info()

# %%
df.head(20)

# %%
# feat_cols
# X = df[feat_cols]
# y = df["Transported"]
