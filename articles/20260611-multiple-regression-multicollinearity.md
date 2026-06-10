---
title: "重回帰分析と多重共線性の診断・対処：統計検定DS対策"
emoji: "📐"
type: "tech"
topics: ["python", "statistics", "datascience"]
published: true
---

## はじめに

「重回帰分析を実行したら、どの変数も有意じゃなくなった。でも単回帰だと有意なのに…」

これは多重共線性（マルチコ）の典型的な症状です。変数を増やすほどモデルが不安定になる、統計検定でも頻出のテーマです。本記事では重回帰分析の数学的背景から多重共線性の診断・対処法まで、Pythonコードで実践的に解説します。

---

## 重回帰分析とは

### 数式

$n$ 個の観測値に対して $p$ 個の説明変数を持つ重回帰モデル：

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i$$

行列表記：

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$

最小二乗推定量（OLS）：

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

推定量の分散共分散行列：

$$\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}$$

ここで：
- $\mathbf{X}$：計画行列（$n \times (p+1)$）
- $\boldsymbol{\beta}$：回帰係数ベクトル
- $\sigma^2$：誤差分散

### いつ使うか

- 目的変数（連続値）を複数の説明変数で予測・説明したいとき
- 交絡因子を統制しながら特定の変数の効果を推定したいとき
- 変数間の偏相関係数を求めたいとき

---

## 多重共線性とは

### 数式

説明変数間に強い線形関係が存在する状態。例えば $x_1 \approx a x_2 + b$ のとき、$\mathbf{X}^\top \mathbf{X}$ の行列式が $0$ に近づき逆行列が不安定になります：

$$\mathrm{Var}(\hat{\beta}_j) = \sigma^2 [(\mathbf{X}^\top \mathbf{X})^{-1}]_{jj} = \frac{\sigma^2}{(n-1) S_j^2} \cdot \frac{1}{1 - R_j^2}$$

ここで $R_j^2$ は $x_j$ を他の説明変数で回帰したときの決定係数。

**分散膨張因子（VIF）**：

$$\mathrm{VIF}_j = \frac{1}{1 - R_j^2}$$

- $\mathrm{VIF}_j = 1$：他変数との相関なし
- $\mathrm{VIF}_j > 5$：要注意
- $\mathrm{VIF}_j > 10$：深刻な多重共線性

### いつ問題になるか

| 症状 | 原因 |
|------|------|
| 単回帰では有意なのに重回帰で非有意 | 係数の標準誤差が膨張 |
| 係数の符号が逆転する | 推定が不安定 |
| 変数を1つ追加するだけで係数が大幅に変わる | 行列が不良条件 |

---

## Pythonで実装

### 基本：重回帰分析

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

np.random.seed(42)

# サンプルデータ生成
X, y = make_regression(n_samples=100, n_features=4, noise=20, random_state=42)
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, 5)])
df["y"] = y

# statsmodelsで係数・p値・信頼区間を確認
X_sm = sm.add_constant(df[["x1", "x2", "x3", "x4"]])
model = sm.OLS(df["y"], X_sm).fit()
print(model.summary())
```

出力には係数・標準誤差・t値・p値・95%信頼区間がすべて含まれます。

---

### 多重共線性の診断：VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 多重共線性のあるデータを作成
np.random.seed(0)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = x1 * 0.99 + np.random.normal(0, 0.1, n)  # x1とほぼ同じ
x3 = np.random.normal(0, 1, n)
y = 2 * x1 + 3 * x3 + np.random.normal(0, 1, n)

df_mc = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
X_mc = sm.add_constant(df_mc)

# VIF計算
vif_data = pd.DataFrame({
    "変数": X_mc.columns,
    "VIF": [variance_inflation_factor(X_mc.values, i) for i in range(X_mc.shape[1])]
})
print(vif_data)
```

```
      変数         VIF
0    const    9.214...
1       x1  507.432...  ← 深刻な多重共線性
2       x2  503.891...  ← 深刻な多重共線性
3       x3    1.001...  ← 問題なし
```

---

### 多重共線性の影響を可視化

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

# 多重共線性あり vs なしで係数推定の安定性を比較
n_simulations = 200
coef_with_mc = []
coef_without_mc = []

np.random.seed(42)
for _ in range(n_simulations):
    x1 = np.random.normal(0, 1, 50)
    x2_corr = x1 * 0.95 + np.random.normal(0, 0.3, 50)   # 相関あり
    x2_indep = np.random.normal(0, 1, 50)                   # 独立
    y_sim = 2 * x1 + np.random.normal(0, 1, 50)

    # 多重共線性あり
    X_c = sm.add_constant(np.column_stack([x1, x2_corr]))
    b_c = sm.OLS(y_sim, X_c).fit().params[1]
    coef_with_mc.append(b_c)

    # 多重共線性なし
    X_i = sm.add_constant(np.column_stack([x1, x2_indep]))
    b_i = sm.OLS(y_sim, X_i).fit().params[1]
    coef_without_mc.append(b_i)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, title in zip(axes,
    [coef_without_mc, coef_with_mc],
    ["多重共線性なし（x1係数の分布）", "多重共線性あり（x1係数の分布）"]):
    ax.hist(data, bins=30, edgecolor="white")
    ax.axvline(2.0, color="red", linestyle="--", label="真の値=2")
    ax.set_title(title)
    ax.set_xlabel("推定係数")
    ax.legend()
    ax.text(0.05, 0.9, f"標準偏差: {np.std(data):.3f}",
            transform=ax.transAxes)

plt.tight_layout()
plt.savefig("multicollinearity_comparison.png", dpi=150)
plt.show()
```

多重共線性があると係数の推定値がばらつき、信頼性が大幅に低下することが確認できます。

---

### 対処法①：変数選択（VIFで除外）

```python
# VIFが高い変数を削除
def remove_high_vif(df_X, threshold=5.0):
    while True:
        vif = pd.Series(
            [variance_inflation_factor(df_X.values, i) for i in range(df_X.shape[1])],
            index=df_X.columns
        )
        max_vif = vif.max()
        if max_vif < threshold:
            break
        drop_col = vif.idxmax()
        print(f"VIF={max_vif:.1f} → {drop_col} を除外")
        df_X = df_X.drop(columns=drop_col)
    return df_X

X_clean = remove_high_vif(df_mc.copy())
print("\n残った変数:", X_clean.columns.tolist())
```

```
VIF=507.4 → x1 を除外
残った変数: ['x2', 'x3']
```

---

### 対処法②：リッジ回帰（L2正則化）

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

# データ準備
X_ridge = df_mc[["x1", "x2", "x3"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ridge)

# αを交差検証で選択
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_scaled, y)

print(f"最適α: {ridge_cv.alpha_:.4f}")
print(f"係数: {ridge_cv.coef_}")

# OLSとリッジで係数を比較
ols = sm.OLS(y, sm.add_constant(X_scaled)).fit()
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_scaled, y)

comparison = pd.DataFrame({
    "OLS": ols.params[1:],
    "Ridge": ridge.coef_
}, index=["x1", "x2", "x3"])
print(comparison)
```

リッジ回帰は正則化項 $\lambda\|\boldsymbol{\beta}\|^2$ を加えることで、係数が安定します：

$$\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

---

### 対処法③：主成分回帰（PCR）

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# PCA→回帰のパイプライン
pcr = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),  # 2主成分に削減
    ("regressor", LinearRegression())
])

scores = cross_val_score(pcr, X_ridge, y, cv=5, scoring="r2")
print(f"PCR R²（CV）: {scores.mean():.3f} ± {scores.std():.3f}")

pcr.fit(X_ridge, y)
pca = pcr.named_steps["pca"]
print(f"説明分散比: {pca.explained_variance_ratio_}")
```

---

## モデル診断：回帰の前提確認

```python
# 残差診断
X_diag = sm.add_constant(df_mc[["x3"]])  # 多重共線性なし版
model_diag = sm.OLS(y, X_diag).fit()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 残差 vs 予測値
fitted = model_diag.fittedvalues
residuals = model_diag.resid
axes[0].scatter(fitted, residuals, alpha=0.5)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("予測値")
axes[0].set_ylabel("残差")
axes[0].set_title("残差 vs 予測値（等分散性の確認）")

# QQプロット（正規性の確認）
sm.qqplot(residuals, line="s", ax=axes[1])
axes[1].set_title("QQプロット（残差の正規性）")

plt.tight_layout()
plt.savefig("regression_diagnostics.png", dpi=150)
plt.show()
```

---

## 多重共線性への対処フロー

```
1. VIFを計算する
   ├─ VIF < 5：問題なし → そのままモデル化
   ├─ 5 ≤ VIF < 10：軽度 → 変数の取捨選択を検討
   └─ VIF ≥ 10：深刻 → 以下のいずれかで対処
       ├─ ①相関の高い変数を一方削除
       ├─ ②リッジ回帰（予測精度重視）
       ├─ ③主成分回帰（解釈性を維持したい場合）
       └─ ④ドメイン知識で変数を統合（例：身長＋体重→BMI）
```

---

## まとめ

| ポイント | 内容 |
|---------|------|
| OLS推定量 | $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$ |
| 多重共線性の原因 | 説明変数間の強い線形関係 |
| 診断指標 | VIF（5以上で注意、10以上で深刻） |
| 対処法① | 相関の高い変数を削除 |
| 対処法② | リッジ回帰（L2正則化） |
| 対処法③ | 主成分回帰（PCR） |
| 残差診断 | 等分散性・正規性・独立性を確認 |

多重共線性は「変数が多ければ多いほど良い」という直感に反する落とし穴です。VIFで必ず診断する習慣をつけましょう。
