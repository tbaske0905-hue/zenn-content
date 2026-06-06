---
title: "scikit-learnパイプラインで作る、再現性のある機械学習モデル"
emoji: "🔧"
type: "tech"
topics: ["python", "machinelearning", "datascience", "scikit-learn"]
published: true
---

## はじめに

「ローカルでは動くのに、本番環境でエラーが出る」
「前処理の順番を間違えてデータリークが起きた」

機械学習モデルを実装していると、こういった問題に頻繁に遭遇します。原因の多くは、前処理とモデル訓練が**バラバラに管理されている**ことにあります。

scikit-learn の `Pipeline` を使うと、前処理からモデル訓練・推論までを一つの流れとして管理できます。本記事では、Pipelineの基本から本番運用まで使える実践的な設計方法を解説します。

## Pipelineを使わないと何が起きるか

まず、問題のある典型的なコードを見てみましょう。

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ❌ 問題のある実装
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # ← ここが問題！

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
model.score(X_test_scaled, y_test)
```

`X_test_scaled = scaler.fit_transform(X_test)` の部分でテストデータにもfitしてしまっています。これは **データリーク** です。テストデータの統計量（平均・分散）が前処理に混入し、評価指標が過剰に良く見えてしまいます。

## Pipelineの基本

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Pipelineを使った正しい実装
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipeline.fit(X_train, y_train)         # fit はtrainのみ
score = pipeline.score(X_test, y_test) # testには transform のみが適用される
print(f"Accuracy: {score:.4f}")
```

`Pipeline` はステップをリストで受け取ります。`fit()` を呼ぶと、最後のステップ以外は `fit_transform()`、最後のステップは `fit()` が順番に実行されます。`predict()` や `score()` 時は、最後以外は `transform()` のみが呼ばれるため、データリークが防げます。

## 数値・カテゴリ変数の混在を扱う

実務では、数値変数とカテゴリ変数が混在するデータを扱うことがほとんどです。`ColumnTransformer` を使って列ごとに異なる処理を適用できます。

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# サンプルデータ
data = pd.DataFrame({
    "age":        [25, 30, None, 45, 22],
    "salary":     [300, 500, 400, None, 250],
    "department": ["engineer", "sales", "engineer", "manager", None],
    "resigned":   [0, 1, 0, 1, 0],
})
X = data.drop("resigned", axis=1)
y = data["resigned"]

# 列の種類ごとに分類
numeric_cols = ["age", "salary"]
categorical_cols = ["department"]

# 数値列の前処理：欠損補完 → 標準化
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# カテゴリ列の前処理：欠損補完 → One-Hot Encoding
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ColumnTransformer で列ごとに適用
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# 前処理 + モデルを一つのPipelineに
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

pipeline.fit(X, y)
print(pipeline.predict(X))
```

## ハイパーパラメータのチューニング

Pipelineの中のパラメータを `GridSearchCV` でチューニングできます。ステップ名と `__` でパラメータを指定します。

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [3, 5, None],
    "preprocessor__num__imputer__strategy": ["mean", "median"],
}

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best CV score: {search.best_score_:.4f}")
```

Pipelineのおかげで、CVの各foldでも正しく train/test が分離されます。

## カスタム変換器を作る

標準の変換器では対応できない処理は、`BaseEstimator` と `TransformerMixin` を継承してカスタム変換器を作れます。

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    """正の数値列に対数変換を適用する。"""
    
    def __init__(self, offset: float = 1.0):
        self.offset = offset
    
    def fit(self, X, y=None):
        return self  # fit不要な変換器はselfを返すだけ
    
    def transform(self, X):
        return np.log(X + self.offset)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """IQRベースで外れ値を上下限にクリップする。"""
    
    def __init__(self, factor: float = 1.5):
        self.factor = factor
    
    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self
    
    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)


# カスタム変換器をPipelineに組み込む
pipeline = Pipeline([
    ("clipper", OutlierClipper(factor=1.5)),
    ("log", LogTransformer()),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
```

`fit()` でtrainデータの統計量を覚え、`transform()` でその統計量を使って変換することで、データリークを防げます。

## モデルの保存と読み込み

学習済みPipelineをそのまま保存・読み込みできます。前処理のパラメータ（スケーラーの平均・分散など）も含めて保存されるため、本番環境でも同じ前処理が再現されます。

```python
import joblib

# 保存
joblib.dump(pipeline, "model_pipeline.pkl")
print("モデルを保存しました")

# 読み込み
loaded_pipeline = joblib.load("model_pipeline.pkl")

# 新しいデータに対してそのまま推論
new_data = pd.DataFrame({
    "age": [28],
    "salary": [350],
    "department": ["engineer"],
})
prediction = loaded_pipeline.predict(new_data)
print(f"予測結果: {prediction}")
```

## Pipelineの可視化

`set_config` を使うと、JupyterノートブックでPipelineを視覚的に確認できます。

```python
from sklearn import set_config
set_config(display="diagram")

# Jupyterでpipelineを表示するだけでHTML図が出る
pipeline
```

## まとめ

| やること | 方法 |
|---------|------|
| 前処理 + モデルを繋げる | `Pipeline` |
| 列ごとに異なる前処理 | `ColumnTransformer` |
| ハイパーパラメータ探索 | `GridSearchCV` + `__` 記法 |
| 独自の変換処理 | `BaseEstimator` + `TransformerMixin` |
| モデル保存・再利用 | `joblib.dump / load` |

Pipelineを使うことで、**データリーク防止・再現性向上・コードのシンプル化** の3つが同時に達成できます。実装が少し複雑に見えますが、一度習慣にすると「Pipelineなしでは書けない」と思えるほど便利です。まずは簡単な前処理から試してみてください。
