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
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
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
cols_to_drop = {
    "Alone",
    "DiscretizedAge4_0",
    "DiscretizedAge4_1",
    "DiscretizedAge4_2",
    "DiscretizedAge4_3",
    "VIP",
    "Transported",
}
feat_cols = [col for col in cols if col not in cols_to_drop]
feat_cols

# %%
X = df[feat_cols]
y = df["Transported"]

# %% [markdown]
# ## Dummy Classifier

# %%
dummy_clf = DummyClassifier(strategy="uniform", random_state=333).fit(X, y)
y_pred = dummy_clf.predict(X)

# %%
print(classification_report(y, y_pred))

# %% [markdown]
# ## Logistic Regression
# ### L1 Regularization
# Find optimal model:

# %%
logistic_model = LogisticRegression(penalty="l1", random_state=33, solver="liblinear", max_iter=1000)
param_grid = {"C": np.logspace(-2, 1, 50), "class_weight": [None, "balanced"]}

# %%
grid_search = GridSearchCV(logistic_model, param_grid=param_grid, scoring="accuracy", n_jobs=2).fit(X, y)

# %%
cv_results = (
    pd.DataFrame(grid_search.cv_results_)
    .drop(columns=["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "params"])
    .sort_values(by="rank_test_score")
    .reset_index(drop=True)
)
cv_results.head(10)

# %%
grid_search.best_params_

# %%
best_model = grid_search.best_estimator_
best_model = cast(LogisticRegression, best_model)

# %% [markdown]
# Inspect model coefficients:

# %%
coef = best_model.coef_.flatten()
coef

# %%
# Features that have been "eliminated"
feature_names = best_model.feature_names_in_
feature_names[coef == 0].tolist()

# %%
df_coef = (
    pd.DataFrame(data={"Feature": feature_names, "Coefficient": coef})
    .set_index("Feature")
    .assign(AbsCoef=lambda x: np.abs(x.Coefficient))
)
df_coef

# %%
df_coef = df_coef[df_coef.AbsCoef > 0].sort_values(by="AbsCoef", ascending=False)
df_coef["Color"] = np.where(df_coef.Coefficient > 0, "#228B22", "#DE0030")

# %%
fig = plt.figure(figsize=(6.0, 6.0), layout="tight")
ax = fig.add_subplot()
sns.barplot(
    x=df_coef.AbsCoef,
    y=df_coef.index,
    palette=df_coef.Color.tolist(),
    orient="h",
    ax=ax,
)
ax.set_title("Absolute value of model coefficients")
ax.set_xlabel("Absolute value of coefficient")
plt.show()

# %%
