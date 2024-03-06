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
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from IPython.display import display
from sklearn.feature_selection import SelectKBest, chi2

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
    dtype={"CompCntReduced": pd.CategoricalDtype(categories=["0", "1", "2", "3+"], ordered=True)},
).assign(CompCntReduced=lambda x: x.CompCntReduced.cat.codes)
df_train.head(10)

# %%
df_train.info()

# %%
assert df_train.isna().sum().eq(0).all()

# %%
y_train = df_train.pop("Transported")
X_train = df_train

# %% [markdown]
# ## Univariate feature selection
#
# Feature selection with chi-square for scoring:

# %%
# This makes sense only for categorical variables. This is why `PTTotalSpent`
# is excluded.
selector = SelectKBest(
    chi2,
    k="all",  # pyright: ignore [reportArgumentType]
).fit(X_train.drop(columns="PTTotalSpent"), y_train)
selector = cast(SelectKBest, selector)

# %%
# Compute scores
pvalues = cast(npt.ArrayLike, selector.pvalues_)
scores = -np.log10(pvalues)
scores /= scores.max()

# %%
# Visualize results

# Plot scores
fig = plt.figure(figsize=(8.0, 7.0), layout="tight")
ax = fig.add_subplot()
x = np.arange(scores.size)
ax.bar(x, scores)
ax.set_xticks(x, labels=selector.feature_names_in_)
ax.tick_params(axis="x", labelrotation=90.0)
ax.set_xlabel("Feature")
ax.set_ylabel("Normalized univariate score")
ax.set_title("Feature selection with chi-square scoring")
plt.show()

# Display scores table
scores_df = pd.DataFrame(
    data={
        "Feature": selector.feature_names_in_,
        "PValue": pvalues,
        "Score": scores,
    }
).set_index("Feature")
display(scores_df)

# %%
# Same thing, but order by score
scores_df = scores_df.sort_values(by="Score", ascending=False)

fig = plt.figure(figsize=(8.0, 7.0), layout="tight")
ax = fig.add_subplot()
ax.bar(scores_df.index, scores_df["Score"])
ax.tick_params(axis="x", labelrotation=90.0)
ax.set_xlabel("Feature")
ax.set_ylabel("Normalized univariate score")
ax.set_title("Feature selection with chi-square scoring")
plt.show()

display(scores_df)

# %%
