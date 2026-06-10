---
title: "最尤推定（MLE）の仕組みと実装：パラメータ推定の核心を数式とPythonで理解する"
emoji: "🎯"
type: "tech"
topics: ["python", "statistics", "datascience", "machinelearning"]
published: true
---

## はじめに

「正規分布でデータをフィットするとき、なぜ平均 $\bar{X}$ と分散 $S^2$ を使うのか？」

この問いに答えるのが**最尤推定（Maximum Likelihood Estimation, MLE）**です。「観測データが得られる確率を最大にするパラメータを選ぶ」という原理で、機械学習・統計モデルのパラメータ推定の根幹をなします。

---

## 尤度関数：いつ使うか

### いつ使うか
- 確率分布のパラメータ（$\mu$、$\sigma$、$p$、$\lambda$ など）を観測データから推定するとき
- ロジスティック回帰・線形回帰・混合分布モデルなど、ほぼ全ての統計モデルの訓練
- 情報量基準（AIC・BIC）によるモデル選択

### 数式

$n$ 個の独立な観測値 $x_1, x_2, \ldots, x_n$ が確率密度 $f(x;\theta)$ に従うとき、**尤度関数（Likelihood Function）**：

$$L(\theta) = \prod_{i=1}^{n} f(x_i; \theta)$$

積の形は扱いにくいため、**対数尤度（Log-Likelihood）**を使う：

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log f(x_i; \theta)$$

対数は単調増加なので、$L(\theta)$ を最大化することと $\ell(\theta)$ を最大化することは等価。

**最尤推定量（MLE）**：

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \ell(\theta)$$

---

## 例1：正規分布のMLE

### 数式

$X_i \sim N(\mu, \sigma^2)$ のとき、対数尤度：

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2$$

$\partial \ell / \partial \mu = 0$、$\partial \ell / \partial \sigma^2 = 0$ を解くと：

$$\hat{\mu}_{\text{MLE}} = \bar{X} = \frac{1}{n}\sum_{i=1}^n x_i$$

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{X})^2$$

**注意**：$\hat{\sigma}^2_{\text{MLE}}$ は $n$ で割るため**偏りあり**（不偏推定量は $n-1$ で割る）。

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
data = rng.normal(loc=3.0, scale=2.0, size=100)

# MLEで解析的に求める
mu_mle    = data.mean()
sigma_mle = data.std(ddof=0)   # MLE: n で割る
sigma_unbiased = data.std(ddof=1)  # 不偏推定量: n-1 で割る

print(f"真のパラメータ: μ=3.0, σ=2.0")
print(f"MLE:   μ̂={mu_mle:.4f}, σ̂={sigma_mle:.4f}")
print(f"不偏:  σ̂={sigma_unbiased:.4f}  (n={len(data)})")

# scipy.stats.norm.fit でも同じ結果
mu_fit, sigma_fit = stats.norm.fit(data)
print(f"scipy.fit: μ̂={mu_fit:.4f}, σ̂={sigma_fit:.4f}")
```

---

## 例2：ポアソン分布のMLE

### 数式

$X_i \sim \text{Pois}(\lambda)$ のとき対数尤度：

$$\ell(\lambda) = \sum_{i=1}^n \left( x_i \log\lambda - \lambda - \log(x_i!) \right)$$

$d\ell/d\lambda = 0$ を解くと：

$$\hat{\lambda}_{\text{MLE}} = \bar{X} = \frac{1}{n}\sum_{i=1}^n x_i$$

ポアソン分布の場合、MLEは**標本平均**に一致する。

```python
# 1時間あたりの問い合わせ数（λ=4 の生成データ）
data_pois = rng.poisson(lam=4.0, size=200)

lambda_mle = data_pois.mean()
print(f"真のλ=4.0, MLE: λ̂={lambda_mle:.4f}")

# 尤度曲線を可視化
lambdas = np.linspace(2, 7, 300)
log_likelihoods = np.array([
    np.sum(stats.poisson.logpmf(data_pois, mu=lam)) for lam in lambdas
])

plt.figure(figsize=(8, 4))
plt.plot(lambdas, log_likelihoods, color="steelblue")
plt.axvline(lambda_mle, color="red", linestyle="--", label=f"MLE λ̂={lambda_mle:.2f}")
plt.xlabel("λ"); plt.ylabel("対数尤度 ℓ(λ)")
plt.title("ポアソン分布の尤度曲線")
plt.legend(); plt.tight_layout(); plt.show()
```

---

## 例3：ベルヌーイ分布のMLE（二項比率）

### 数式

$X_i \sim \text{Bernoulli}(p)$ のとき：

$$\ell(p) = k\log p + (n-k)\log(1-p)$$

（$k$：成功回数）

$d\ell/dp = 0$ を解くと：

$$\hat{p}_{\text{MLE}} = \frac{k}{n}$$

→ **標本割合**がMLEになる。

```python
# コインを100回投げて62回表
n, k = 100, 62
p_mle = k / n
print(f"MLE: p̂={p_mle:.4f}")

# 尤度曲線
ps = np.linspace(0.3, 0.9, 300)
log_likelihoods = k * np.log(ps) + (n - k) * np.log(1 - ps)

plt.figure(figsize=(8, 4))
plt.plot(ps, log_likelihoods, color="tomato")
plt.axvline(p_mle, color="navy", linestyle="--", label=f"MLE p̂={p_mle}")
plt.xlabel("p"); plt.ylabel("対数尤度 ℓ(p)")
plt.title("ベルヌーイ分布の尤度曲線")
plt.legend(); plt.tight_layout(); plt.show()
```

---

## 数値最適化によるMLE（任意の分布）

解析解がない場合は `scipy.optimize` で数値的に最大化する。

```python
from scipy.optimize import minimize

# ガンマ分布のパラメータ推定（解析解なし）
true_alpha, true_beta = 3.0, 2.0
data_gamma = rng.gamma(shape=true_alpha, scale=true_beta, size=300)

def neg_log_likelihood(params):
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf
    return -np.sum(stats.gamma.logpdf(data_gamma, a=alpha, scale=beta))

result = minimize(
    neg_log_likelihood,
    x0=[1.0, 1.0],          # 初期値
    method="Nelder-Mead",
)
alpha_mle, beta_mle = result.x
print(f"真のパラメータ: α={true_alpha}, β={true_beta}")
print(f"MLE:            α̂={alpha_mle:.4f}, β̂={beta_mle:.4f}")

# scipy.stats.gamma.fit でも可能
alpha_fit, loc_fit, scale_fit = stats.gamma.fit(data_gamma, floc=0)
print(f"scipy.fit:      α̂={alpha_fit:.4f}, β̂={scale_fit:.4f}")
```

---

## MLEの性質

| 性質 | 内容 |
|------|------|
| **一致性** | $n\to\infty$ で $\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta_{\text{true}}$ |
| **漸近有効性** | 十分大きな $n$ でCramer-Rao下限を達成する（最も分散が小さい） |
| **不変性** | $\hat{\theta}$ が MLE なら $g(\hat{\theta})$ も $g(\theta)$ の MLE |
| **偏りの可能性** | 有限標本では偏りがある場合がある（正規分布の $\hat{\sigma}^2$ など） |

**Cramer-Rao下限（CRLB）**：不偏推定量の分散の下限

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}, \quad I(\theta) = -E\!\left[\frac{\partial^2 \ell}{\partial\theta^2}\right]$$

$I(\theta)$：**Fisher情報量**。データが多いほど $I(\theta)$ が大きくなり、下限が小さくなる。

```python
# Fisher情報量の計算例（正規分布のμに関する情報量）
# I(μ) = n/σ² → サンプルが増えるほど推定精度が上がる
sigma = 2.0
for n in [10, 100, 1000]:
    fisher_info = n / sigma**2
    crlb = 1 / fisher_info
    print(f"n={n:4d}: Fisher情報量={fisher_info:.2f}, CRLB(μの分散下限)={crlb:.4f}")
```

---

## MLEとベイズ推定の関係

| | MLE | MAP（最大事後確率推定） | ベイズ推定 |
|-|-----|---------------------|-----------|
| 式 | $\arg\max L(\theta)$ | $\arg\max P(\theta\|D)$ | $E[\theta\|D]$ |
| 事前分布 | 使わない | 使う | 使う |
| 出力 | 点推定 | 点推定 | 分布 |

**MAP = MLE + 正則化**と解釈できる。L2正則化（Ridge）は正規分布の事前分布に対応する。

---

## まとめ

| 分布 | MLE（解析解） | 数式 |
|------|-------------|------|
| 正規分布 $N(\mu,\sigma^2)$ | $\hat{\mu}=\bar{X}$、$\hat{\sigma}^2=\frac{1}{n}\sum(x_i-\bar{X})^2$ | 偏差の2乗和を最小化 |
| ポアソン分布 $\text{Pois}(\lambda)$ | $\hat{\lambda}=\bar{X}$ | 標本平均 |
| ベルヌーイ分布 $\text{Ber}(p)$ | $\hat{p}=k/n$ | 標本割合 |
| 指数分布 $\text{Exp}(\lambda)$ | $\hat{\lambda}=1/\bar{X}$ | 標本平均の逆数 |

解析解がない分布には `scipy.optimize.minimize` で対数尤度を数値最大化する。
