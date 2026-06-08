---
title: "確率分布の使い分け完全ガイド：データサイエンティストが押さえるべき10の分布"
emoji: "📊"
type: "tech"
topics: ["python", "statistics", "datascience", "machinelearning"]
published: true
---

## はじめに

「どの分布をいつ使うか」は統計・機械学習のつまずきポイントです。本記事では、10の主要分布を**数式（PDF/PMF・期待値・分散）・いつ使うか・Pythonコード**の3点セットで整理します。

---

## 離散型分布

### 1. 二項分布 $\text{Bin}(n, p)$

#### いつ使うか
- $n$ 回の独立な試行で、各回の成功確率が $p$ で一定のとき
- クリック率の推定、合格者数の予測、製品の不良品数など

#### 数式

PMF（確率質量関数）：

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

期待値・分散：

$$E[X] = np, \qquad \text{Var}[X] = np(1-p)$$

モーメント母関数：

$$M_X(t) = (1 - p + pe^t)^n$$

```python
from scipy import stats
import numpy as np

n, p = 20, 0.3
dist = stats.binom(n=n, p=p)

print(f"E[X] = {dist.mean():.2f}  (理論値 np = {n*p})")
print(f"Var[X] = {dist.var():.2f}  (理論値 np(1-p) = {n*p*(1-p):.2f})")
print(f"P(X=7) = {dist.pmf(7):.4f}")
print(f"P(X≥10) = {1-dist.cdf(9):.4f}")
```

---

### 2. ポアソン分布 $\text{Pois}(\lambda)$

#### いつ使うか
- 単位時間・面積あたりの「まれなイベント数」をモデル化するとき
- Webサイトへのアクセス数、コールセンターの着信数、欠陥品の個数など
- 二項分布で $n$ が大きく $p$ が小さい（$\lambda = np$ 一定）ときの近似

#### 数式

PMF：

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

期待値・分散（両方とも $\lambda$）：

$$E[X] = \lambda, \qquad \text{Var}[X] = \lambda$$

**ポアソン近似の条件**：$n \geq 20$、$p \leq 0.05$、$\lambda = np$

```python
lam = 4.0
dist = stats.poisson(mu=lam)

print(f"E[X] = Var[X] = {dist.mean():.2f}")
print(f"P(X=0) = {dist.pmf(0):.4f}  (= e^{{-λ}} = {np.exp(-lam):.4f})")
print(f"P(X≥7) = {1-dist.cdf(6):.4f}")

# 二項分布 → ポアソン近似（n大・p小）
n_large, p_small = 1000, 0.004  # λ = 4
binom_approx = stats.binom(n=n_large, p=p_small)
print(f"\nポアソン近似の精度（P(X=k)比較）:")
for k in range(6):
    print(f"  k={k}: Binom={binom_approx.pmf(k):.5f}, Poisson={dist.pmf(k):.5f}")
```

---

### 3. 幾何分布 $\text{Geom}(p)$

#### いつ使うか
- 初めて成功するまでの試行回数をモデル化するとき
- 無記憶性（Memoryless Property）が成り立つ場合

#### 数式

PMF（$k$ 回目に初めて成功）：

$$P(X = k) = (1-p)^{k-1}p, \quad k = 1, 2, 3, \ldots$$

期待値・分散：

$$E[X] = \frac{1}{p}, \qquad \text{Var}[X] = \frac{1-p}{p^2}$$

無記憶性：

$$P(X > s + t \mid X > s) = P(X > t)$$

```python
p = 0.3
dist = stats.geom(p=p)
print(f"E[X] = {dist.mean():.2f}  (= 1/p = {1/p:.2f})")
print(f"P(X=1) = {dist.pmf(1):.4f}  (= p = {p})")
print(f"P(X≤3) = {dist.cdf(3):.4f}")
```

---

## 連続型分布

### 4. 正規分布 $N(\mu, \sigma^2)$

#### いつ使うか
- 自然現象の多くの測定値（身長・体重・測定誤差など）
- 中心極限定理（CLT）により $n$ が大きいとき標本平均が近似的に従う
- 標準化した量（z得点）の分布

#### 数式

PDF：

$$f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

期待値・分散：

$$E[X] = \mu, \qquad \text{Var}[X] = \sigma^2$$

標準化（z変換）：

$$Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$$

**68-95-99.7 ルール**：

$$P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 0.683$$
$$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 0.954$$
$$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 0.997$$

```python
mu, sigma = 170, 6
dist = stats.norm(loc=mu, scale=sigma)

z = (182 - mu) / sigma
print(f"z得点: {z:.2f}")
print(f"P(X > 182) = {1-dist.cdf(182):.4f} = P(Z > {z:.2f}) = {1-stats.norm.cdf(z):.4f}")

# 正規性の検定（Shapiro-Wilk）
data = np.random.normal(mu, sigma, 100)
stat, p = stats.shapiro(data)
print(f"\nShapiro-Wilk: W={stat:.4f}, p={p:.4f}")
```

---

### 5. t分布 $t(\nu)$

#### いつ使うか
- 母分散 $\sigma^2$ が未知で、小サンプルから推定するとき
- t検定の検定統計量として（標本平均の差の検定）
- 自由度 $\nu = n - 1$（1標本の場合）

#### 数式

PDF：

$$f(t; \nu) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1+\frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

t統計量（1標本）：

$$t = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t(n-1)$$

2標本（等分散）：

$$t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}, \quad S_p^2 = \frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}$$

**正規分布との関係**：$\nu \to \infty$ で $t(\nu) \to N(0,1)$

```python
# 2群の平均差を検定
group_a = [12.5, 13.1, 11.8, 14.2, 12.9]
group_b = [14.1, 15.3, 13.8, 16.0, 14.7]
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t={t_stat:.3f}, p={p_val:.4f}")

# 自由度と裾の重さの比較
import matplotlib.pyplot as plt
x = np.linspace(-4, 4, 300)
plt.figure(figsize=(7, 4))
for df in [1, 5, 30]:
    plt.plot(x, stats.t(df=df).pdf(x), label=f"t({df})")
plt.plot(x, stats.norm.pdf(x), "--k", label="N(0,1)")
plt.legend(); plt.title("自由度と裾の重さ"); plt.tight_layout(); plt.show()
```

---

### 6. カイ二乗分布 $\chi^2(\nu)$

#### いつ使うか
- 正規母集団の**分散**の推定・検定
- カテゴリカルデータの**適合度検定・独立性検定**
- 標準正規変数の2乗和：$Z_1^2 + \cdots + Z_\nu^2 \sim \chi^2(\nu)$

#### 数式

PDF：

$$f(x; \nu) = \frac{x^{\nu/2-1}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}, \quad x > 0$$

期待値・分散：

$$E[X] = \nu, \qquad \text{Var}[X] = 2\nu$$

独立性検定の統計量：

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

```python
# 独立性検定（性別 × 好みのアイス）
observed = np.array([[30, 10, 20], [20, 25, 15]])
chi2, p_val, dof, expected = stats.chi2_contingency(observed)
print(f"χ²={chi2:.3f}, p={p_val:.4f}, 自由度={dof}")
print(f"期待度数:\n{expected.round(1)}")
```

---

### 7. F分布 $F(\nu_1, \nu_2)$

#### いつ使うか
- 2つの独立した正規母集団の**分散比**の検定
- 分散分析（ANOVA）：グループ間分散 ÷ グループ内分散
- 回帰モデルの有意性検定

#### 数式

F統計量（分散比）：

$$F = \frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2} \sim F(\nu_1, \nu_2) \quad \text{（} H_0: \sigma_1^2 = \sigma_2^2 \text{ のとき } F = S_1^2/S_2^2\text{）}$$

一元配置ANOVAのF統計量：

$$F = \frac{MS_{\text{between}}}{MS_{\text{within}}} = \frac{SS_{\text{between}}/(k-1)}{SS_{\text{within}}/(N-k)}$$

```python
group1 = [23, 25, 28, 22, 24]
group2 = [30, 32, 35, 29, 31]
group3 = [22, 24, 26, 21, 23]

f_stat, p_val = stats.f_oneway(group1, group2, group3)
print(f"F={f_stat:.3f}, p={p_val:.6f}")
```

---

### 8. ガンマ分布 $\text{Gamma}(\alpha, \beta)$

#### いつ使うか
- $\alpha$ 番目のイベントが起きるまでの待ち時間（$\alpha=1$ で指数分布）
- ベイズ統計でポアソン分布のレートパラメータの事前分布

#### 数式

PDF（形状パラメータ $\alpha$、スケールパラメータ $\beta$）：

$$f(x; \alpha, \beta) = \frac{x^{\alpha-1}e^{-x/\beta}}{\beta^\alpha \Gamma(\alpha)}, \quad x > 0$$

期待値・分散：

$$E[X] = \alpha\beta, \qquad \text{Var}[X] = \alpha\beta^2$$

指数分布は $\alpha=1$ の特殊ケース：

$$f(x; \beta) = \frac{1}{\beta}e^{-x/\beta}$$

```python
alpha, beta = 3, 2
dist = stats.gamma(a=alpha, scale=beta)
print(f"E[X] = {dist.mean():.1f}  (= αβ = {alpha*beta})")
print(f"Var[X] = {dist.var():.1f}  (= αβ² = {alpha*beta**2})")
```

---

## 分布の選び方：まとめ

| データ型 | 状況 | 分布 | 主な数式の特徴 |
|---------|------|------|--------------|
| 離散 | n回試行の成功数 | $\text{Bin}(n,p)$ | $E=np$ |
| 離散 | 単位時間のイベント数 | $\text{Pois}(\lambda)$ | $E=\text{Var}=\lambda$ |
| 離散 | 初成功までの試行数 | $\text{Geom}(p)$ | $E=1/p$、無記憶性 |
| 連続 | 測定値・CLT | $N(\mu,\sigma^2)$ | 68-95-99.7則 |
| 連続 | 小サンプル平均の検定 | $t(\nu)$ | $\nu\to\infty$ で $N(0,1)$ |
| 連続 | 分散検定・独立性検定 | $\chi^2(\nu)$ | $E=\nu$、$\text{Var}=2\nu$ |
| 連続 | 分散比・ANOVA | $F(\nu_1,\nu_2)$ | $MS_{\text{between}}/MS_{\text{within}}$ |
| 連続 | 待ち時間・信頼性 | $\text{Gamma}(\alpha,\beta)$ | $E=\alpha\beta$ |
| 連続 | 割合パラメータの事前分布 | $\text{Beta}(\alpha,\beta)$ | $E=\alpha/(\alpha+\beta)$ |

「どのデータに・何を知りたいか」から逆算して分布を選ぶ習慣が、統計を使いこなす近道です。
