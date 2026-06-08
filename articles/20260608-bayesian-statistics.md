---
title: "ベイズ統計の実践入門：「事前知識」を使ってデータから正しく学ぶ"
emoji: "🎲"
type: "tech"
topics: ["python", "statistics", "datascience", "machinelearning"]
published: true
---

## はじめに

「コインを10回投げて7回表が出た。このコインは偏っているか？」

頻度論（p値）では「帰無仮説のもとで…」という回りくどい手順が必要です。ベイズ統計なら「表が出る確率の分布」を直接得られます。

本記事では、ベイズ統計の核心的な数式とPythonによる実装を丁寧に解説します。

---

## ベイズの定理：いつ使うか

### いつ使うか
- 過去の知識（事前分布）とデータを組み合わせて推論したいとき
- サンプルが少なく、頻度論の推定が不安定なとき
- 「〜の確率は何%か」という形で答えを直接出したいとき
- A/Bテスト・パラメータ推定・分類モデルの確率校正など

### 数式

**ベイズの定理**：

$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) \cdot P(\theta)}{P(\mathcal{D})}$$

各項の意味：

| 項 | 名称 | 意味 |
|----|------|------|
| $P(\theta \mid \mathcal{D})$ | 事後分布（Posterior） | データを見た後のパラメータの確率分布 |
| $P(\mathcal{D} \mid \theta)$ | 尤度（Likelihood） | パラメータ $\theta$ のもとでデータが得られる確率 |
| $P(\theta)$ | 事前分布（Prior） | データを見る前のパラメータの信念 |
| $P(\mathcal{D})$ | 周辺尤度（Evidence） | 正規化定数（定数なので比例式で扱うことが多い） |

比例式で書くと：

$$P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta) \cdot P(\theta)$$

> **事後分布 ∝ 尤度 × 事前分布**

---

## Beta分布と共役事前分布

### いつ使うか
- $\theta \in [0, 1]$（確率・割合）をパラメータとして推定するとき
- 二項分布の尤度と組み合わせると解析的に更新できる（共役事前分布）

### 数式

**Beta分布の確率密度関数（PDF）**：

$$f(\theta; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1}, \quad \theta \in [0,1]$$

期待値と分散：

$$E[\theta] = \frac{\alpha}{\alpha + \beta}, \quad \text{Var}[\theta] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

**二項尤度 × Beta事前分布 → Beta事後分布**（共役性）：

データ：$n$ 回試行で $k$ 回成功

$$P(\mathcal{D} \mid \theta) = \binom{n}{k}\theta^k(1-\theta)^{n-k}$$

事前分布：$P(\theta) = \text{Beta}(\alpha_0, \beta_0)$

事後分布：

$$P(\theta \mid \mathcal{D}) = \text{Beta}(\alpha_0 + k,\; \beta_0 + n - k)$$

→ 成功回数を $\alpha$ に、失敗回数を $\beta$ に**足すだけ**で更新できる。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# コインを10回投げて7回表
n_trials, n_heads = 10, 7

# 事前分布: Beta(1, 1) = 一様分布（知識なし）
prior_a, prior_b = 1, 1

# ベイズ更新：足し算だけ
post_a = prior_a + n_heads
post_b = prior_b + (n_trials - n_heads)

prior_dist = stats.beta(prior_a, prior_b)
post_dist  = stats.beta(post_a, post_b)

print(f"事後分布: Beta({post_a}, {post_b})")
print(f"事後期待値 E[θ] = {post_a/(post_a+post_b):.3f}")
print(f"事後分散 Var[θ] = {post_a*post_b/((post_a+post_b)**2*(post_a+post_b+1)):.4f}")

theta = np.linspace(0, 1, 300)
plt.figure(figsize=(9, 4))
plt.plot(theta, prior_dist.pdf(theta), "--", color="gray", label="事前 Beta(1,1)")
plt.plot(theta, post_dist.pdf(theta), color="steelblue", lw=2,
         label=f"事後 Beta({post_a},{post_b})")
plt.axvline(0.5, color="red", linestyle=":", label="θ=0.5（公平）")
plt.xlabel("θ（表が出る確率）")
plt.ylabel("確率密度")
plt.legend(); plt.tight_layout(); plt.show()
```

---

## 信用区間（Credible Interval）

### いつ使うか
- 「θ が X〜Y の範囲にある確率は 95%」と直接言いたいとき
- 頻度論の信頼区間（「この手順を繰り返すと 95% のケースで区間が真値を含む」）より解釈が直感的

### 数式

95% 等裾信用区間（Equal-tail Credible Interval）：

$$\text{CI}_{0.95} = \left[\theta_{0.025},\; \theta_{0.975}\right]$$

$\theta_{q}$ は事後分布の $q$ 分位点。Beta分布なら：

$$\theta_q = F^{-1}_{\text{Beta}(\alpha, \beta)}(q)$$

```python
lower = post_dist.ppf(0.025)
upper = post_dist.ppf(0.975)
print(f"95%信用区間: [{lower:.3f}, {upper:.3f}]")

# 「θ > 0.5 の確率」を直接計算
prob_biased = 1 - post_dist.cdf(0.5)
print(f"表が出やすい（θ>0.5）確率: {prob_biased:.1%}")
```

---

## ベイズA/Bテスト

### いつ使うか
- 「BがAより優れている確率は何%か」をビジネス判断に使いたいとき
- サンプルが少なく途中経過でも意思決定が必要なとき

### 数式

A群・B群それぞれの事後分布：

$$\theta_A \mid \mathcal{D}_A \sim \text{Beta}(1+x_A,\; 1+n_A-x_A)$$
$$\theta_B \mid \mathcal{D}_B \sim \text{Beta}(1+x_B,\; 1+n_B-x_B)$$

「BがAより優れている確率」：

$$P(\theta_B > \theta_A \mid \mathcal{D}) = \int_0^1 P(\theta_B > \theta_A \mid \theta_A) \, p(\theta_A)\, d\theta_A$$

解析解もあるが、**モンテカルロ積分**が実用的：

$$\hat{P}(\theta_B > \theta_A) \approx \frac{1}{S}\sum_{s=1}^{S} \mathbf{1}[\theta_B^{(s)} > \theta_A^{(s)}]$$

```python
n_a, x_a = 1000, 50
n_b, x_b = 1000, 65

post_a = stats.beta(1 + x_a, 1 + n_a - x_a)
post_b = stats.beta(1 + x_b, 1 + n_b - x_b)

# モンテカルロ積分
S = 200_000
samp_a = post_a.rvs(S)
samp_b = post_b.rvs(S)

prob_b_better = np.mean(samp_b > samp_a)
expected_lift = np.mean((samp_b - samp_a) / samp_a)

print(f"E[θ_A] = {post_a.mean():.4f}")
print(f"E[θ_B] = {post_b.mean():.4f}")
print(f"P(θ_B > θ_A | D) = {prob_b_better:.1%}")
print(f"期待改善率: {expected_lift:.1%}")
```

---

## データが増えると事後分布はどう変わるか

事前知識の影響は $n \to \infty$ で消え、事後平均は最尤推定量（MLE）に収束する。

$$E[\theta \mid \mathcal{D}] = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n} \xrightarrow{n\to\infty} \frac{k}{n} = \hat{\theta}_{\text{MLE}}$$

```python
experiments = [(0,0,"データなし"), (3,2,"3回中2表"), (10,7,"10回中7表"), (100,70,"100回中70表")]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (n, k, label) in zip(axes, experiments):
    dist = stats.beta(1+k, 1+(n-k))
    theta = np.linspace(0, 1, 300)
    ax.plot(theta, dist.pdf(theta), color="steelblue", lw=2)
    ax.axvline(0.7, color="red", ls="--", alpha=0.5, label="真値")
    ax.set_title(label); ax.set_xlabel("θ"); ax.set_xlim(0, 1)
plt.suptitle("データが増えると事後分布が真値（0.7）に収束する", y=1.02)
plt.tight_layout(); plt.show()
```

---

## まとめ

| 概念 | 数式 | 意味 |
|------|------|------|
| ベイズの定理 | $P(\theta\|D) \propto P(D\|\theta)P(\theta)$ | 事後 ∝ 尤度 × 事前 |
| Beta分布の期待値 | $E[\theta]=\alpha/(\alpha+\beta)$ | 更新後の割合の推定値 |
| ベイズ更新（Beta-Binomial） | $\text{Beta}(\alpha_0+k,\;\beta_0+n-k)$ | 成功・失敗を足すだけ |
| 信用区間 | $[F^{-1}(0.025), F^{-1}(0.975)]$ | 事後分布の分位点 |
