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
from pprint import pprint
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from IPython.display import display
from scipy.stats import chisquare
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
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
# ### Feature selection with chi-square for scoring

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
# How to reproduce the above results using scipy
# Perform the same calculation for `Alone`
observed = df_train.loc[df_train["Alone"], "Transported"].value_counts().sort_index()
expected = df_train["Transported"].value_counts(normalize=True).sort_index() * observed.sum()
chi2_res = chisquare(observed, f_exp=expected)

# %%
# p-value
print(f"scipy       : {chi2_res.pvalue}")
print(f"scikit-learn: {selector.pvalues_[0]}")  # pyright: ignore [reportOptionalSubscript]

# %%
# Test statistic
print(f"scipy       : {chi2_res.statistic}")
print(f"scikit-learn: {selector.scores_[0]}")

# %% [markdown]
# ### Feature selection with mutual information for scoring

# %%
selector = SelectKBest(
    mutual_info_classif,
    k="all",  # pyright: ignore [reportArgumentType]
).fit(X_train, y_train)
selector = cast(SelectKBest, selector)

# %%
# Normalized mutual information
mi = cast(npt.ArrayLike, selector.scores_)
mi /= np.max(mi)

mi_df = (
    pd.DataFrame(data={"Feature": selector.feature_names_in_, "MI": mi})
    .set_index("Feature")
    .sort_values(by="MI", ascending=False)
)

# %%
# Visualize results

# Plot mutual information
fig = plt.figure(figsize=(8.0, 7.0), layout="tight")
ax = fig.add_subplot()
ax.bar(mi_df.index, mi_df["MI"])
ax.tick_params(axis="x", labelrotation=90.0)
ax.set_xlabel("Feature")
ax.set_ylabel("Normalized mutual information")
ax.set_title("Feature selection with mutual information scoring")
plt.show()

# Display mutual information table
display(mi_df)

# %% [markdown]
# ### Compare model performance

# %%
feature_names = [
    # "Alone",
    "CompCntReduced",
    "HomePlanetOrd",
    "CryoSleep",
    "CabinDeckOrd",
    "CabinPort",
    "DestinationOrd",
    # "DiscretizedAge4",
    "DiscretizedAge5",
    "VIP",
    "PosRoomService",
    "PosFoodCourt",
    "PosShoppingMall",
    "PosSpa",
    "PosVRDeck",
    "PTTotalSpent",
]
X = df_train[feature_names]
y = df_train["Transported"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=333)

# %%
feature_sets = []
accs = []
f1_scores = []

max_features = len(feature_names)
for num_features in range(1, max_features + 1):
    selector = SelectKBest(mutual_info_classif, k=num_features).fit(X_train, y_train)
    selector = cast(SelectKBest, selector)

    idx = selector.get_support(indices=True)
    feature_set = selector.feature_names_in_[idx]
    feature_sets.append(feature_set)

    X_train_new = selector.transform(X_train)
    X_test_new = selector.transform(X_test)

    clf = GradientBoostingClassifier(random_state=333).fit(X_train_new, y_train)
    clf = cast(GradientBoostingClassifier, clf)

    acc = clf.score(X_test_new, y_test)
    accs.append(acc)

    y_pred = clf.predict(X_test_new)
    f1 = f1_score(y_test, y_pred, average="weighted")
    f1_scores.append(f1)

# %%
metrics_df = pd.DataFrame(
    data={
        "NumFeatures": np.arange(1, max_features + 1),
        "Accuracy": accs,
        "F1Score": f1_scores,
    },
).set_index("NumFeatures")

# %%
fig = plt.figure(figsize=(8.0, 7.0), layout="tight")
ax = fig.add_subplot()
ks = np.arange(1, max_features + 1)
ax.bar(ks, accs)
ax.set_xticks(ks)
ax.set_xlabel("Number of features")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy as a function of the number of features")
plt.show()

display(metrics_df[["Accuracy"]])

# %%
fig = plt.figure(figsize=(8.0, 7.0), layout="tight")
ax = fig.add_subplot()
ax.bar(ks, f1_scores)
ax.set_xticks(ks)
ax.set_xlabel("Number of features")
ax.set_ylabel("F1 score")
ax.set_title("F1 score as a function of the number of features")
plt.show()

display(metrics_df[["F1Score"]])

# %%
metrics_df = metrics_df.sort_values(by=["Accuracy", "F1Score"], ascending=[False, False])
metrics_df

# %%
top_5 = metrics_df.iloc[:5, :]
for i in top_5.index.to_numpy() - 1:
    print(f"Number of features: {i + 1}")
    print("Feature set:")
    pprint(feature_sets[i].tolist())

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
