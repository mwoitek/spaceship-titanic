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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

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
X_train = df_train.drop(columns="Transported")
y_train = df_train["Transported"]

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

# %% [markdown]
# ## Tree-based feature selection

# %%
# Prepare data
X = df_train.drop(columns=["Alone", "DiscretizedAge4", "Transported"])
y = df_train["Transported"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=333)

# %%
# Train classifier
clf = RandomForestClassifier(random_state=333).fit(X_train, y_train)
clf = cast(RandomForestClassifier, clf)

# %% [markdown]
# ### Feature importance based on mean decrease in impurity

# %%
# Get feature importances and the corresponding errors
importances = cast(npt.ArrayLike, clf.feature_importances_)
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
std = cast(npt.ArrayLike, std)

# %%
# Visualize results
# NOTE: Impurity-based feature importances can be misleading for high
# cardinality features (many unique values).

# Plot feature importances
fig = plt.figure(figsize=(8.0, 7.0), layout="tight")
ax = fig.add_subplot()
ax.bar(clf.feature_names_in_, importances, yerr=std)
ax.tick_params(axis="x", labelrotation=90.0)
ax.set_xlabel("Feature")
ax.set_ylabel("Mean decrease in impurity")
ax.set_title("Feature importances using MDI")
plt.show()

# Display results table
importances_df = pd.DataFrame(
    data={
        "Feature": clf.feature_names_in_,
        "Importance": importances,
        "Error": std,
    }
).set_index("Feature")
display(importances_df)

# %%
# Same thing, but order by importance
importances_df = importances_df.sort_values(by="Importance", ascending=False)

fig = plt.figure(figsize=(8.0, 7.0), layout="tight")
ax = fig.add_subplot()
ax.bar(
    importances_df.index,
    importances_df["Importance"],
    yerr=importances_df["Error"],
)
ax.tick_params(axis="x", labelrotation=90.0)
ax.set_xlabel("Feature")
ax.set_ylabel("Mean decrease in impurity")
ax.set_title("Feature importances using MDI")
plt.show()

display(importances_df)

# %% [markdown]
# ### Feature importance based on feature permutation

# %%
result = permutation_importance(
    clf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=333,
    n_jobs=2,
)

# %%
fig = plt.figure(figsize=(8.0, 7.0), layout="tight")
ax = fig.add_subplot()
idx = np.argsort(result.importances_mean)[::-1]
ax.bar(
    clf.feature_names_in_[idx],
    result.importances_mean[idx],
    yerr=result.importances_std[idx],
)
ax.tick_params(axis="x", labelrotation=90.0)
ax.set_xlabel("Feature")
ax.set_ylabel("Mean accuracy decrease")
ax.set_title("Feature importances using permutation on full model")
plt.show()
