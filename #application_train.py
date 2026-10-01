from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("application_train.csv")
print(df["TARGET"].value_counts())
print(df["TARGET"].value_counts(normalize=True))
print(df.dtypes.value_counts())
print(df.info())

missing = df.isnull().sum().sort_values(ascending=False)
print(missing.head(20))
missing_pct = df.isnull().mean().sort_values(ascending=False)*100
print(missing_pct.head(20))


for col in df.columns:
    print(col)

df[
    [
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "CNT_CHILDREN",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED"
    ]
].describe()


# age analysis
df["AGE_YEARS"] = -df["DAYS_BIRTH"]/365
"""
print(df.groupby("TARGET")["AGE_YEARS"].mean())

print(
    df.groupby("TARGET")["AGE_YEARS"].agg(["mean", "median", "min", "max"])
)


plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="AGE_YEARS",
    hue="TARGET",
    bins=30,
    kde=True,
    stat="density",
    common_norm=False
)
plt.title("Age Distribution by Default Status")
plt.xlabel("Age(Years)")
plt.ylabel("Density")
plt.show()

# income analysis
print(
    df.groupby("TARGET")["AMT_INCOME_TOTAL"].agg(
        ["mean", "median", "min", "max"])
)

print(df["AMT_INCOME_TOTAL"].describe())
print(df["AMT_INCOME_TOTAL"].quantile(
    [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
))
"""

df["INCOME_LOG"] = np.log1p(df["AMT_INCOME_TOTAL"])
df[["AMT_INCOME_TOTAL", "INCOME_LOG"]].head()

"""
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="AMT_INCOME_TOTAL",
    hue="TARGET",
    bins=50,
    kde=True,
    stat="density",
    common_norm=False
)
plt.xlim(0, df["AMT_INCOME_TOTAL"].quantile(0.99))

plt.title("Income Distribution by Default Status")
plt.xlabel("Income")
plt.ylabel("Density")
plt.show()


sns.countplot(data=df, x="TARGET")
plt.title("Target Distribution")
plt.xlabel("Default")
plt.ylabel("Number of Applicants")
plt.show()

"""
# Baseline Model
X = df.drop(columns=["TARGET", "SK_ID_CURR"])
Y = df["TARGET"]
print(X.shape)
print(Y.shape)

cat_cols = X.select_dtypes(include=["object", "str"]).columns
print(len(cat_cols))
print(cat_cols.tolist())


X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

"""
missing = X.isnull().sum()
print("Total missing values:", missing.sum())
print("Columns with missing values", sum(missing > 0))
print(missing[missing > 0].sort_values(ascending=False).head(10))


imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
print(X_imputed.shape)
print(np.isnan(X_imputed).sum())

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_imputed,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
model.fit(X_train, Y_train)

Y_valid_pred = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(Y_valid, Y_valid_pred)
print("Validation ROC-AUC:", auc)

gb_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

gb_model.fit(X_train, Y_train)
Y_valid_gb = gb_model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(Y_valid, Y_valid_gb)
print("Gradient Boosting Validation ROC-AUC:", auc)

result = permutation_importance(
    gb_model,
    X_valid,
    Y_valid,
    scoring="roc_auc",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

feature_importance = pd.Series(
    result.importances_mean,
    index=X.columns
).sort_values(ascending=False)
print(feature_importance.head(20))

gb_model_2 = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=63,
    random_state=42
)
gb_model_2.fit(X_train, Y_train)

Y_valid_gb2 = gb_model_2.predict_proba(X_valid)[:, 1]
gb_auc_2 = roc_auc_score(Y_valid, Y_valid_gb2)
print("Tuned Gradient Boosting Roc-Auc", gb_auc_2)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

gb_cv = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

cv_scores = cross_val_score(
    gb_cv,
    X,
    Y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)
print("CV ROC-AUC:", cv_scores)
print("Mean ROC-AUC:", cv_scores.mean())

"""


gb_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores_clean = cross_val_score(
    gb_pipeline,
    X,
    Y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)
print("Clean CV ROC-AUC:", cv_scores_clean)
print("Mean ROC-AUC:", cv_scores_clean.mean())

gb_tuned = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42

    ))
])

tunes_scores = cross_val_score(
    gb_tuned,
    X,
    Y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)
print("Tuned CV ROC-AUC:", cv_scores_clean)
print("Tuned Mean ROC-AUC:", cv_scores_clean.mean())
