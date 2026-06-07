---
title: "ベイズ統計の実践入門：「事前知識」を使ってデータから正しく学ぶ"
emoji: "🎲"
type: "tech"
topics: ["python", "statistics", "datascience", "machinelearning"]
published: true
---

## はじめに

「コインを10回投げて7回表が出た。このコインは偏っているか？」

頻度論（p値）で答えようとすると、「帰無仮説のもとで…」という回りくどい手順が必要です。ベイズ統計なら、「表が出る確率の分布」を直接得られます。

本記事では、ベイズ統計の考え方をPythonコードと一緒に解説します。難しい数式より「何を計算しているか」を重視します。

## ベイズ統計の核心：3つの概念

### 事前分布（Prior）
実験前にすでに持っている知識や仮定を確率分布で表したもの。

「コインの表が出る確率 θ は 0〜1 の間で、均等に分布していそう」→ 一様分布 `Beta(1, 1)`

### 尤度（Likelihood）
観測データが得られた確率。

「θ のコインを10回投げて7回表が出る確率」→ 二項分布

### 事後分布（Posterior）
事前分布と尤度をかけ合わせたもの。「データを見た後の信念の更新」。

$$P(\theta | data) \propto P(data | \theta) \cdot P(\theta)$$

## Pythonで理解するベイズ更新

`scipy.stats` だけで実装できます。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# コインを10回投げて7回表が出た
n_trials = 10
n_heads = 7

# 事前分布: Beta(1, 1) = 一様分布
prior_alpha, prior_beta = 1, 1

# ベイズ更新: Beta分布の共役事前分布を使う
# 事後分布は Beta(prior_alpha + n_heads, prior_beta + n_tails)
posterior_alpha = prior_alpha + n_heads
posterior_beta = prior_beta + (n_trials - n_heads)

print(f"事後分布: Beta({posterior_alpha}, {posterior_beta})")
print(f"事後平均: {posterior_alpha / (posterior_alpha + posterior_beta):.3f}")

# 可視化
theta = np.linspace(0, 1, 300)
prior_dist = stats.beta(prior_alpha, prior_beta)
posterior_dist = stats.beta(posterior_alpha, posterior_beta)

plt.figure(figsize=(10, 5))
plt.plot(theta, prior_dist.pdf(theta), label="事前分布 Beta(1,1)", linestyle="--", color="gray")
plt.plot(theta, posterior_dist.pdf(theta), label=f"事後分布 Beta({posterior_alpha},{posterior_beta})", color="steelblue", linewidth=2)
plt.axvline(0.5, color="red", linestyle=":", label="θ=0.5（公平なコイン）")
plt.xlabel("θ（表が出る確率）")
plt.ylabel("確率密度")
plt.title("ベイズ更新：コイン投げの例")
plt.legend()
plt.tight_layout()
plt.show()
```

事後分布のピークが 0.7 付近にあり、「このコインは少し表寄りかもしれない」という推論が視覚的に分かります。

## 信用区間（Credible Interval）

ベイズ統計では「θ が 95% の確率でこの区間にある」と直接言えます（頻度論の信頼区間とは意味が異なります）。

```python
# 95% 信用区間（HDI: Highest Density Interval の近似）
lower = posterior_dist.ppf(0.025)
upper = posterior_dist.ppf(0.975)

print(f"95%信用区間: [{lower:.3f}, {upper:.3f}]")
# → [0.397, 0.915] のような結果

# 「θ > 0.5 の確率」を直接計算できる
prob_biased = 1 - posterior_dist.cdf(0.5)
print(f"表が出やすい確率: {prob_biased:.1%}")
```

p値では「表が出やすいかどうか」を直接言えませんが、ベイズ統計では確率として取り出せます。

## 実務例：A/Bテストをベイズで解く

クリック率の比較を例に、ベイズA/Bテストを実装します。

```python
import numpy as np
from scipy import stats

# AパターンとBパターンのデータ
n_a, clicks_a = 1000, 50   # 表示1000回、クリック50回
n_b, clicks_b = 1000, 65   # 表示1000回、クリック65回

# 事後分布（Beta分布の共役性を利用）
posterior_a = stats.beta(1 + clicks_a, 1 + n_a - clicks_a)
posterior_b = stats.beta(1 + clicks_b, 1 + n_b - clicks_b)

print(f"A CTR 事後平均: {posterior_a.mean():.3f}")
print(f"B CTR 事後平均: {posterior_b.mean():.3f}")

# モンテカルロシミュレーションで「BがAより優れている確率」を計算
n_samples = 100_000
samples_a = posterior_a.rvs(n_samples)
samples_b = posterior_b.rvs(n_samples)

prob_b_better = np.mean(samples_b > samples_a)
expected_lift = np.mean((samples_b - samples_a) / samples_a)

print(f"\nBがAより優れている確率: {prob_b_better:.1%}")
print(f"期待改善率: {expected_lift:.1%}")

# 可視化
theta = np.linspace(0, 0.15, 300)
plt.figure(figsize=(10, 5))
plt.plot(theta, posterior_a.pdf(theta), label=f"A（{clicks_a}/{n_a}回）", color="steelblue")
plt.plot(theta, posterior_b.pdf(theta), label=f"B（{clicks_b}/{n_b}回）", color="tomato")
plt.xlabel("クリック率")
plt.ylabel("確率密度")
plt.title(f"A/Bテスト事後分布（BがAより優れている確率: {prob_b_better:.1%}）")
plt.legend()
plt.tight_layout()
plt.show()
```

p値と違い「BはAより X% 改善している確率は Y%」という形でビジネス判断に使いやすい結果が得られます。

## データが増えると事後分布はどう変わるか

「データが少ないうちは事前知識が強く影響し、データが増えるにつれて観測値に収束する」のがベイズ統計の直感的な特徴です。

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

# 徐々にデータを増やしていく
experiments = [
    (0, 0, "データなし"),
    (3, 2, "3回中2回表"),
    (10, 7, "10回中7回表"),
    (100, 70, "100回中70回表"),
]

for ax, (n, k, label) in zip(axes, experiments):
    alpha = 1 + k
    beta_param = 1 + (n - k)
    dist = stats.beta(alpha, beta_param)
    theta = np.linspace(0, 1, 300)
    ax.plot(theta, dist.pdf(theta), color="steelblue", linewidth=2)
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.5)
    ax.set_title(label)
    ax.set_xlabel("θ")
    ax.set_xlim(0, 1)

plt.suptitle("データが増えるにつれて事後分布が真の値（0.7）に収束する", y=1.02)
plt.tight_layout()
plt.show()
```

## 頻度論との使い分け

| | 頻度論（p値） | ベイズ統計 |
|---|---|---|
| 「θ の確率」を直接言える | ❌（パラメータは固定値） | ✅ |
| 事前知識を組み込める | ❌ | ✅ |
| 解釈のしやすさ | p値は誤解されやすい | 「〜の確率」と直接言える |
| 計算コスト | 低い | 複雑なモデルは高い |
| サンプル数が少ない場合 | 信頼性が低い | 事前分布で補完できる |

どちらが優れているかではなく、問いに応じて使い分けるのが実務での正解です。

## まとめ

- **事前分布**：実験前の知識を確率で表す
- **尤度**：データが得られる確率
- **事後分布**：事前 × 尤度で「データを見た後の信念」
- Beta分布は二項分布の共役事前分布で、手計算でも更新できる
- 「〜の確率」を直接言えるので、ビジネス判断に使いやすい

最初の一歩として `scipy.stats.beta` を使ったコインの例をぜひ手元で動かしてみてください。
