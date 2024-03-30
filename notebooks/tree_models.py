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
# # Spaceship Titanic: Tree Models
# ## Imports

# %%
import warnings
from pathlib import Path
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

# %%
warnings.simplefilter(action="ignore", category=FutureWarning)

# %% [markdown]
# ## Download data (if necessary)

# %%
data_dir = Path.cwd().parent / "input" / "spaceship-titanic"
if not data_dir.exists():
    data_dir.mkdir(parents=True)


# %%
def download_file(base_url: str, file_name: str) -> None:
    file_path = data_dir / file_name
    if file_path.exists():
        print(f"File {file_name} already exists. Nothing to do.")
        return

    url = urljoin(base_url, file_name)
    response = requests.get(url, stream=True)  # noqa: S113
    if response.status_code != 200:
        print(f"Failed to download file {file_name}")
        return

    with file_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)


# %%
# Training data
base_url = "https://raw.githubusercontent.com/mwoitek/spaceship-titanic/master/input/spaceship-titanic"
download_file(base_url, "train.csv")

# %%
# Test data
download_file(base_url, "test.csv")

# %% [markdown]
# ## Read data

# %%
# Training data
df_train = pd.read_csv(data_dir / "train.csv")
df_train.head(10)

# %%
# Test data
df_test = pd.read_csv(data_dir / "test.csv")
df_test.head(10)

# %% [markdown]
# ## Create features from `PassengerId`

# %%
# Group
df_train["Group"] = df_train["PassengerId"].str.split("_", expand=True).iloc[:, 0]
df_test["Group"] = df_test["PassengerId"].str.split("_", expand=True).iloc[:, 0]

# %%
# GroupSize
df_train = df_train.join(
    df_train.groupby(by="Group").agg(GroupSize=pd.NamedAgg(column="PassengerId", aggfunc="count")),
    on="Group",
)
df_test = df_test.join(
    df_test.groupby(by="Group").agg(GroupSize=pd.NamedAgg(column="PassengerId", aggfunc="count")),
    on="Group",
)

# %%
# Set indexes
df_train = df_train.set_index("PassengerId", verify_integrity=True)
df_test = df_test.set_index("PassengerId", verify_integrity=True)

# %% [markdown]
# ## New features from `Cabin`

# %%
# CabinDeck, CabinNum and CabinSide
df_train = df_train.join(
    df_train["Cabin"]
    .str.split("/", expand=True)
    .rename(columns={0: "CabinDeck", 1: "CabinNum", 2: "CabinSide"})
)
df_test = df_test.join(
    df_test["Cabin"]
    .str.split("/", expand=True)
    .rename(columns={0: "CabinDeck", 1: "CabinNum", 2: "CabinSide"})
)

# %% [markdown]
# ## Using the `Name` column

# %%
# Add Surname column
df_train = df_train.assign(Surname=df_train["Name"].str.split(" ", expand=True).iloc[:, 1])
df_test = df_test.assign(Surname=df_test["Name"].str.split(" ", expand=True).iloc[:, 1])

# %% [markdown]
# ## Impute some missing values
# Passengers who belong to the same group also come from the same home planet:

# %%
assert (
    df_train[df_train["HomePlanet"].notna()]
    .groupby("Group")
    .agg({"HomePlanet": "nunique"})
    .eq(1)
    .all(axis=None)
)
assert (
    df_test[df_test["HomePlanet"].notna()]
    .groupby("Group")
    .agg({"HomePlanet": "nunique"})
    .eq(1)
    .all(axis=None)
)

# %% [markdown]
# Using group data to impute some missing `HomePlanet` values:

# %%
# Training data
df_1 = (
    df_train.loc[df_train["GroupSize"].gt(1) & df_train["HomePlanet"].notna(), ["Group", "HomePlanet"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
query = "GroupSize > 1 and Group in @df_1.Group and HomePlanet.isna()"
df_2 = df_train.query(query).loc[:, ["Group"]].reset_index()
df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
df_train.loc[df_3.index, "HomePlanet"] = df_3["HomePlanet"]
del df_1, df_2, df_3

# %%
# Test data
df_1 = (
    df_test.loc[df_test["GroupSize"].gt(1) & df_test["HomePlanet"].notna(), ["Group", "HomePlanet"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
df_2 = df_test.query(query).loc[:, ["Group"]].reset_index()
df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
df_test.loc[df_3.index, "HomePlanet"] = df_3["HomePlanet"]
del df_1, df_2, df_3, query

# %% [markdown]
# Passengers that belong to the same group were on the same side of the spaceship:

# %%
assert (
    df_train[df_train["CabinSide"].notna()]
    .groupby(by="Group")
    .agg({"CabinSide": "nunique"})
    .eq(1)
    .all(axis=None)
)
assert (
    df_test[df_test["CabinSide"].notna()]
    .groupby(by="Group")
    .agg({"CabinSide": "nunique"})
    .eq(1)
    .all(axis=None)
)

# %% [markdown]
# Fill some missing `CabinSide` values using group data:

# %%
# Training data
df_1 = (
    df_train.query("GroupSize > 1 and CabinSide.notna()")
    .groupby("Group")
    .agg({"CabinSide": "first"})
    .reset_index()
)
query = "GroupSize > 1 and Group in @df_1.Group and CabinSide.isna()"
df_2 = df_train.query(query).loc[:, ["Group"]].reset_index()
df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
df_train.loc[df_3.index, "CabinSide"] = df_3["CabinSide"]
del df_1, df_2, df_3

# %%
# Test data
df_1 = (
    df_test.query("GroupSize > 1 and CabinSide.notna()")
    .groupby("Group")
    .agg({"CabinSide": "first"})
    .reset_index()
)
df_2 = df_test.query(query).loc[:, ["Group"]].reset_index()
df_3 = df_2.merge(df_1, on="Group").drop(columns="Group").set_index("PassengerId")
df_test.loc[df_3.index, "CabinSide"] = df_3["CabinSide"]
del df_1, df_2, df_3, query

# %% [markdown]
# Passengers with the same surname are from the same planet:

# %%
assert (
    df_train[["Surname", "HomePlanet"]]
    .dropna()
    .groupby("Surname")
    .agg({"HomePlanet": "nunique"})
    .eq(1)
    .all(axis=None)
)
assert (
    df_test[["Surname", "HomePlanet"]]
    .dropna()
    .groupby("Surname")
    .agg({"HomePlanet": "nunique"})
    .eq(1)
    .all(axis=None)
)

# %% [markdown]
# Use `Surname` to fill more missing `HomePlanet` values:

# %%
# Training data
df_sur_1 = (
    df_train[["Surname", "HomePlanet"]].dropna().groupby("Surname").agg({"HomePlanet": "first"}).reset_index()
)
query = "Surname.notna() and Surname in @df_sur_1.Surname and HomePlanet.isna()"
df_1 = df_train.query(query).loc[:, ["Surname"]].reset_index()
df_2 = df_1.merge(df_sur_1, on="Surname").drop(columns="Surname").set_index("PassengerId")
df_train.loc[df_2.index, "HomePlanet"] = df_2["HomePlanet"]
del df_1, df_2

# %%
# Test data

# To fix test data, I'll also use some training data. Combine all relevant data:
df_sur_2 = (
    df_test[["Surname", "HomePlanet"]].dropna().groupby("Surname").agg({"HomePlanet": "first"}).reset_index()
)
df_sur = pd.concat([df_sur_1, df_sur_2.query("Surname not in @df_sur_1.Surname")], ignore_index=True)
del df_sur_1, df_sur_2

# %%
query = query.replace("df_sur_1", "df_sur")
df_1 = df_test.query(query).loc[:, ["Surname"]].reset_index()
df_2 = df_1.merge(df_sur, on="Surname").drop(columns="Surname").set_index("PassengerId")
df_test.loc[df_2.index, "HomePlanet"] = df_2["HomePlanet"]
del df_1, df_2, df_sur, query

# %% [markdown]
# No VIP passenger is from Earth:

# %%
query = "VIP.notna() and VIP == True and HomePlanet.notna()"
assert df_train.query(query).HomePlanet.ne("Earth").all()
assert df_test.query(query).HomePlanet.ne("Earth").all()
del query

# %% [markdown]
# Impute some missing values of `VIP`:

# %%
# Training data
query = "VIP.isna() and HomePlanet.notna() and HomePlanet == 'Earth'"
idx = df_train.query(query).index
df_train.loc[idx, "VIP"] = False

# %%
# Test data
idx = df_test.query(query).index
df_test.loc[idx, "VIP"] = False
del idx, query

# %% [markdown]
# Dealing with the "money columns":

# %%
# All medians equal zero
money_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
assert df_train[money_cols].median().eq(0.0).all()
assert df_test[money_cols].median().eq(0.0).all()

# %%
# Fill missing values with zeros (medians)
df_train.loc[:, money_cols] = df_train[money_cols].fillna(0.0)
df_test.loc[:, money_cols] = df_test[money_cols].fillna(0.0)

# %%
# Add `TotalSpent` column
df_train["TotalSpent"] = df_train[money_cols].agg("sum", axis=1)
df_test["TotalSpent"] = df_test[money_cols].agg("sum", axis=1)
del money_cols

# %% [markdown]
# Passengers who spent money were NOT in cryo sleep:

# %%
assert not df_train.query("TotalSpent > 0 and CryoSleep.notna()").CryoSleep.any()
assert not df_test.query("TotalSpent > 0 and CryoSleep.notna()").CryoSleep.any()

# %% [markdown]
# Fill some missing `CryoSleep` values based on `TotalSpent`:

# %%
df_train.loc[df_train["CryoSleep"].isna() & df_train["TotalSpent"].gt(0.0), "CryoSleep"] = False
df_test.loc[df_test["CryoSleep"].isna() & df_test["TotalSpent"].gt(0.0), "CryoSleep"] = False

# %% [markdown]
# ## Missing values that remain

# %%
feats = ["HomePlanet", "CryoSleep", "Destination", "Age", "VIP", "CabinDeck", "CabinSide"]
df_miss = df_train.isna().sum().rename("Number").to_frame().rename_axis("Feature", axis=0)
df_miss = df_miss[df_miss["Number"] > 0].loc[feats, :]
df_miss = df_miss.assign(Percentage=(100.0 * df_miss["Number"] / df_train.shape[0]).round(2)).sort_values(
    by="Percentage", ascending=False
)
df_miss

# %%
df_miss = df_test.isna().sum().rename("Number").to_frame().rename_axis("Feature", axis=0)
df_miss = df_miss[df_miss["Number"] > 0].loc[feats, :]
df_miss = df_miss.assign(Percentage=(100.0 * df_miss["Number"] / df_test.shape[0]).round(2)).sort_values(
    by="Percentage", ascending=False
)
df_miss

# %%
del df_miss, feats

# %% [markdown]
# ## Encode categorical features

# %%
cat_feats = ["HomePlanet", "CabinDeck", "CabinSide", "Destination"]
enc = OrdinalEncoder().fit(df_train[cat_feats])
df_train.loc[:, cat_feats] = enc.transform(df_train[cat_feats])
df_test.loc[:, cat_feats] = enc.transform(df_test[cat_feats])
del cat_feats

# %% [markdown]
# ## Feature selection

# %%
feature_names = [
    "GroupSize",
    "HomePlanet",
    "CryoSleep",
    "CabinDeck",
    "CabinSide",
    "Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpent",
]
X = df_train[feature_names]
y = df_train["Transported"]

# %%
max_features = len(feature_names)
score_func = lambda X, y: mutual_info_classif(X, y, random_state=0)
feature_sets = []

for num_features in range(1, max_features + 1):
    idx_1 = X[X.notna().all(axis=1)].index
    selector = SelectKBest(score_func=score_func, k=num_features).fit(X.loc[idx_1, :], y.loc[idx_1])

    idx_2 = selector.get_support(indices=True)
    feature_set = selector.feature_names_in_[idx_2].tolist()
    feature_sets.append(feature_set)

# %% [markdown]
# ## Tree models
# ### Random Forest
# Accuracy as a function of the number of features:

# %%
accs = []
cv = StratifiedKFold(shuffle=True, random_state=0)
rf_params = {
    "n_estimators": 200,
    "criterion": "entropy",
    "class_weight": "balanced_subsample",
    "random_state": 0,
}

for feature_set in feature_sets:
    cv_accs = []
    X_new = X[feature_set]

    for train_idx, test_idx in cv.split(X_new, y):
        X_train, X_test = X_new.iloc[train_idx, :], X_new.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestClassifier(**rf_params).fit(X_train, y_train)
        acc = rf.score(X_test, y_test)
        cv_accs.append(acc)

    mean_acc = np.mean(cv_accs)
    accs.append(mean_acc)

df_accs = pd.DataFrame(
    data={
        "NumFeatures": np.arange(1, max_features + 1),
        "Accuracy": accs,
        "FeatureSet": feature_sets,
    }
).set_index("NumFeatures")

# %%
idx_max = int(df_accs["Accuracy"].idxmax())

# %%
fig = plt.figure(figsize=(9.0, 6.0), layout="tight")
ax = fig.add_subplot()
colors = ["#2f4f4f"] * df_accs.shape[0]
colors[idx_max - 1] = "#6039b2"
ax.bar(df_accs.index, df_accs["Accuracy"], color=colors)
ax.bar_label(ax.containers[0], fmt="%.3f")  # pyright: ignore [reportArgumentType]
ax.set_xticks(df_accs.index)
ax.set_xlabel("Number of features")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy as a function of the number of features")
plt.show()

with pd.option_context("display.max_colwidth", None):
    display(df_accs)

# %% [markdown]
# Find optimal model:

# %%
feature_names = df_accs.loc[idx_max, "FeatureSet"]
feature_names

# %%
X = df_train[feature_names]

# %%
rf = RandomForestClassifier(**rf_params)
param_grid = {
    "max_depth": [8, 11, 12, 13, 14],
    "min_samples_split": [13, 17, 18, 19, 20],
    "min_samples_leaf": [5, 6, 7, 14, 20],
}

# %%
grid_search = GridSearchCV(rf, param_grid=param_grid, scoring="accuracy", n_jobs=3).fit(X, y)

# %%
cv_results = (
    pd.DataFrame(grid_search.cv_results_)
    .drop(columns=["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "params"])
    .set_index("rank_test_score")
    .sort_index()
)
cv_results.head(10)

# %%
grid_search.best_params_

# %%
