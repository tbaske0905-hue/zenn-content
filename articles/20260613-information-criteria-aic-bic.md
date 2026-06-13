---
title: "情報量基準（AIC・BIC）でモデル選択を数式から理解する：複雑さと当てはまりのバランス"
emoji: "⚖️"
type: "tech"
topics: ["python", "statistics", "datascience", "machinelearning"]
published: true
---

## はじめに

回帰モデルの次数を上げれば上げるほど、訓練データへの当てはまり（対数尤度）は必ず良くなります。しかし、当てはまりだけでモデルを選ぶと**過学習**を招きます。

「モデルの複雑さ」と「データへの当てはまりの良さ」をどう天秤にかけるか——この問いに答えるのが**情報量基準（Information Criteria）**です。代表的なものが **AIC（赤池情報量基準）** と **BIC（ベイズ情報量基準）** です。

統計検定データサイエンティスト/エキスパートでも頻出のトピックであり、最尤推定（MLE）の延長として理解すると応用が広がります。

---

## AICとは

### 数式

AIC（Akaike's Information Criterion）は以下で定義されます。

$$
\text{AIC} = -2\ell(\hat{\theta}) + 2k
$$

- $\ell(\hat{\theta})$：最大対数尤度（MLEで求めたパラメータでの対数尤度）
- $k$：モデルのパラメータ数（自由度）

第1項 $-2\ell(\hat{\theta})$ は「当てはまりの悪さ」、第2項 $2k$ は「複雑さへのペナルティ」を表します。**AICが小さいモデルほど良い**と判断します。

AICは、推定したモデルと真のモデルとの間の**Kullback-Leibler情報量（KLダイバージェンス）**を最小化する、という考え方に基づいて導出されています。つまり「真の分布にどれだけ近いか」を近似的に評価する指標です。

### いつ使うか

- 複数のモデル（変数の組み合わせ・多項式の次数・ARIMAの次数など）を比較して、**予測性能が良いモデル**を選びたいとき
- サンプルサイズが大きく、「真のモデル」がそもそも候補の中に存在しない・候補が無限にあるような状況（漸近的に**予測誤差最小**のモデルを選ぶ性質がある）
- 時系列モデル（ARIMA）の次数 $(p, d, q)$ 選択

### Pythonで実装

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 真のモデル: y = 1.5 + 2.0*x + 0.5*x^2 + ノイズ
n = 100
x = np.linspace(-3, 3, n)
y_true = 1.5 + 2.0 * x + 0.5 * x**2
y = y_true + rng.normal(scale=1.0, size=n)

def fit_polynomial_aic(x, y, degree):
    """degree次の多項式回帰を当てはめ、AICを計算する"""
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    resid = y - y_pred

    n = len(y)
    k = degree + 2  # 係数の数 + 残差分散1個

    # 正規分布を仮定した最大対数尤度
    sigma2_hat = np.sum(resid**2) / n  # MLEの分散（n割り）
    log_likelihood = -n / 2 * (np.log(2 * np.pi * sigma2_hat) + 1)

    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + k * np.log(n)
    return aic, bic

print(f"{'次数':>4} | {'AIC':>10} | {'BIC':>10}")
for degree in range(1, 6):
    aic, bic = fit_polynomial_aic(x, y, degree)
    print(f"{degree:>4} | {aic:>10.2f} | {bic:>10.2f}")
```

実行すると、真のモデルが2次式であるため、**2次のときにAIC・BICが最小**になることが確認できます。3次以上に上げても対数尤度はわずかに改善しますが、ペナルティ項 $2k$ や $k\log n$ がそれを上回るため、AIC・BICは増加します。

---

## BICとは

### 数式

BIC（Bayesian Information Criterion、Schwarz's Information Criterionとも呼ばれる）は以下で定義されます。

$$
\text{BIC} = -2\ell(\hat{\theta}) + k\log(n)
$$

- $n$：サンプルサイズ

AICとの違いはペナルティ項だけで、AICの $2k$ がBICでは $k\log(n)$ になります。$n \geq 8$ では $\log(n) > 2$ となるため、**BICはAICよりも複雑なモデルに厳しい**（よりシンプルなモデルを選びやすい）という性質があります。

BICはベイズ的な視点から導出され、「真のモデルが候補の中に含まれている」という仮定のもとで、**真のモデルを選ぶ確率が $n\to\infty$ で1に収束する**（一致性, consistency）という性質を持ちます。AICにはこの一致性はありません。

### いつ使うか

- 「真のモデル」が候補集合に含まれていると想定できる、または**解釈性の高い・シンプルなモデル**を優先したいとき
- サンプルサイズが大きいとき（$\log n$ のペナルティがAICより強く働く）
- 回帰分析での変数選択、混合正規分布のコンポーネント数決定など

### Pythonで実装

```python
import statsmodels.api as sm

# 変数選択の例：説明変数を増やしたときのAIC/BICの変化
rng = np.random.default_rng(0)
n = 200
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
x3 = rng.normal(size=n)          # 本当は y と無関係なノイズ変数
y = 1.0 + 2.0 * x1 - 1.0 * x2 + rng.normal(scale=1.0, size=n)

import pandas as pd
df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

models = {
    "x1のみ":        "y ~ x1",
    "x1 + x2":       "y ~ x1 + x2",
    "x1 + x2 + x3":  "y ~ x1 + x2 + x3",
}

print(f"{'モデル':<15} | {'AIC':>10} | {'BIC':>10}")
for name, formula in models.items():
    model = sm.OLS.from_formula(formula, data=df).fit()
    print(f"{name:<15} | {model.aic:>10.2f} | {model.bic:>10.2f}")
```

不要な変数 `x3` を加えても対数尤度はほぼ変わらないため、AIC・BICはともに「x1 + x2」のモデルを最良と判定します。特にBICは、ノイズ変数を含むモデルにより強いペナルティを課す傾向が見られます。

---

## AIC vs BIC：使い分けの整理

| 観点 | AIC | BIC |
|------|-----|-----|
| ペナルティ項 | $2k$ | $k\log(n)$ |
| 目的 | 予測誤差の最小化（KLダイバージェンス最小化） | 真のモデルの特定（一致性） |
| サンプルが大きいとき | 比較的複雑なモデルを選びやすい | よりシンプルなモデルを選びやすい |
| 漸近的な性質 | 一致性なし（真のモデルより複雑なモデルを選ぶ確率が残る） | 一致性あり（$n\to\infty$で真のモデルに収束） |
| 向いている用途 | 予測精度重視のモデリング | 説明変数の絞り込み・解釈重視のモデリング |

実務では、**両方を計算して傾向を確認し、選んだモデルの差が小さい場合は解釈性やドメイン知識で最終判断する**のが現実的です。

---

## 補足：尤度比検定との違い

入れ子になった2つのモデル（片方が他方の特殊ケース）を比較する場合、**尤度比検定**を使うこともできます。

$$
\text{LR} = -2\left(\ell(\hat{\theta}_{\text{小}}) - \ell(\hat{\theta}_{\text{大}})\right) \sim \chi^2_{(k_{\text{大}} - k_{\text{小}})}
$$

尤度比検定は「2つの特定のモデルの優劣」を**仮説検定の枠組み**で判定するのに対し、AIC/BICは入れ子でない複数モデルも含めて**ランキング**として比較できる点が異なります。両者は相互排他的ではなく、状況に応じて使い分けます。

```python
from scipy.stats import chi2

# 例: 2次モデル vs 3次モデルの尤度比検定
def log_likelihood(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    resid = y - y_pred
    n = len(y)
    sigma2_hat = np.sum(resid**2) / n
    return -n / 2 * (np.log(2 * np.pi * sigma2_hat) + 1)

ll_small = log_likelihood(x, y, 2)
ll_large = log_likelihood(x, y, 3)
lr_stat = -2 * (ll_small - ll_large)
p_value = chi2.sf(lr_stat, df=1)  # パラメータ数の差は1

print(f"LR統計量 = {lr_stat:.4f}, p値 = {p_value:.4f}")
# p値が大きい → 3次モデルへの拡張は有意な改善とは言えない
```

---

## まとめ

- AIC・BICはいずれも「**当てはまりの良さ**」と「**モデルの複雑さへのペナルティ**」のバランスでモデルを評価する指標
- $\text{AIC} = -2\ell(\hat{\theta}) + 2k$、$\text{BIC} = -2\ell(\hat{\theta}) + k\log(n)$ で、**値が小さいほど良いモデル**
- AICは予測精度重視・BICはシンプルなモデル・真のモデルの特定を重視
- `statsmodels` の回帰モデルは `.aic` / `.bic` 属性で簡単に取得できる
- 入れ子モデルの比較には尤度比検定も併用すると、統計的な裏付けを持って判断できる

モデル選択は「精度が高ければ良い」という単純な話ではなく、**目的（予測か説明か）に応じて評価軸を選ぶこと**が重要です。
