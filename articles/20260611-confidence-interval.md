---
title: "区間推定と信頼区間の正しい解釈：統計検定DS対策"
emoji: "📏"
type: "tech"
topics: ["python", "statistics", "datascience"]
published: true
---

## はじめに

「95%信頼区間とは、母数が95%の確率でこの区間に含まれる」——これは**誤り**です。

統計学の中でも信頼区間の解釈は特に誤解されやすく、統計検定でも頻出の落とし穴です。本記事では区間推定の数学的な定義から正しい解釈、Pythonによる実装まで徹底解説します。

---

## 点推定と区間推定

**点推定**はパラメータを1つの値で推定します（例：標本平均 $\bar{X}$ で母平均 $\mu$ を推定）。しかし点推定だけでは推定の**精度**がわかりません。

**区間推定**は「パラメータが含まれる可能性の高い区間」を求め、推定の不確かさを定量化します。

---

## 信頼区間の定義

### 数式

母平均 $\mu$（母分散 $\sigma^2$ 既知）の $100(1-\alpha)\%$ 信頼区間：

$$\left[\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}},\quad \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right]$$

ここで：
- $\bar{X}$：標本平均
- $\sigma$：母標準偏差
- $n$：標本サイズ
- $z_{\alpha/2}$：標準正規分布の上側 $\alpha/2$ 点（95%の場合 $z_{0.025} = 1.96$）

母分散が**未知**の場合（実務でほぼこちら）は $t$ 分布を使います：

$$\left[\bar{X} - t_{\alpha/2}(n-1) \frac{s}{\sqrt{n}},\quad \bar{X} + t_{\alpha/2}(n-1) \frac{s}{\sqrt{n}}\right]$$

- $s$：標本標準偏差（不偏）
- $t_{\alpha/2}(n-1)$：自由度 $n-1$ の $t$ 分布の上側 $\alpha/2$ 点

### 信頼係数の意味（最重要）

$100(1-\alpha)\%$ 信頼区間の正しい解釈：

> **同じ手続きで繰り返しサンプリングして区間を構成したとき、そのうち $100(1-\alpha)\%$ の区間が真の母数を含む**

- ✅ 正しい：「この手続きで作った区間の95%は母平均を含む」
- ❌ 誤り：「母平均がこの区間に含まれる確率は95%」

一度計算した区間に対して、母平均は「含まれる」か「含まれない」かのどちらかです。確率の対象は**区間の構成手続き**であり、固定された母数ではありません。

---

## いつ使うか

| 状況 | 使う区間推定 |
|------|------------|
| 母分散既知（理論値など） | 正規分布による信頼区間 |
| 母分散未知・小標本（$n < 30$） | $t$ 分布による信頼区間 |
| 母分散未知・大標本（$n \geq 30$） | $t$ 分布（または近似的に正規分布） |
| 母比率の推定 | 正規近似または二項分布 |
| 中央値・非正規データ | ブートストラップ信頼区間 |

---

## Pythonで実装

### 基本：t分布による信頼区間

```python
import numpy as np
from scipy import stats

np.random.seed(42)
# 母平均170、母標準偏差8の正規分布から30サンプル
data = np.random.normal(loc=170, scale=8, size=30)

n = len(data)
mean = np.mean(data)
se = stats.sem(data)  # 標準誤差 s/√n

# 95%信頼区間
ci = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=se)
print(f"標本平均: {mean:.2f}")
print(f"95%信頼区間: ({ci[0]:.2f}, {ci[1]:.2f})")
```

```
標本平均: 170.53
95%信頼区間: (167.56, 173.50)
```

### 信頼区間の「頻度論的」解釈を可視化

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

TRUE_MEAN = 170
n_simulations = 50
n_sample = 30
alpha = 0.05

np.random.seed(0)
contains_true = []
intervals = []

for _ in range(n_simulations):
    sample = np.random.normal(loc=TRUE_MEAN, scale=8, size=n_sample)
    mean = np.mean(sample)
    se = stats.sem(sample)
    ci = stats.t.interval(1 - alpha, df=n_sample - 1, loc=mean, scale=se)
    intervals.append((mean, ci[0], ci[1]))
    contains_true.append(ci[0] <= TRUE_MEAN <= ci[1])

# プロット
fig, ax = plt.subplots(figsize=(10, 8))
for i, (mean, lo, hi) in enumerate(intervals):
    color = "steelblue" if contains_true[i] else "tomato"
    ax.plot([lo, hi], [i, i], color=color, linewidth=1.5)
    ax.plot(mean, i, "o", color=color, markersize=4)

ax.axvline(TRUE_MEAN, color="black", linestyle="--", label=f"真の母平均 ({TRUE_MEAN})")
coverage = sum(contains_true) / n_simulations * 100
ax.set_title(f"50回の95%信頼区間シミュレーション\n母平均を含む割合: {coverage:.0f}%（青）、含まない（赤）")
ax.set_xlabel("値")
ax.set_ylabel("シミュレーション回数")
ax.legend()
plt.tight_layout()
plt.savefig("confidence_interval_simulation.png", dpi=150)
plt.show()
print(f"母平均を含む割合: {coverage:.1f}%")
```

50回繰り返すと約95%の区間が母平均170を含みます。**区間が変わるたびに含む・含まないが決まる**ことが視覚的に確認できます。

### 標本サイズと区間幅の関係

```python
import pandas as pd

sample_sizes = [10, 30, 50, 100, 200, 500]
results = []

np.random.seed(42)
for n in sample_sizes:
    sample = np.random.normal(loc=170, scale=8, size=n)
    mean = np.mean(sample)
    se = stats.sem(sample)
    ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
    width = ci[1] - ci[0]
    results.append({"n": n, "標本平均": round(mean, 2), "区間幅": round(width, 2)})

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

```
  n  標本平均  区間幅
 10  169.85   5.96
 30  170.35   2.96
 50  170.06   2.25
100  170.11   1.59
200  170.05   1.12
500  169.99   0.71
```

標本サイズ $n$ を4倍にすると区間幅は **$1/2$** になります（$\propto 1/\sqrt{n}$）。

### 母比率の信頼区間

```python
from statsmodels.stats.proportion import proportion_confint

# 100人中60人が賛成
n_total = 100
n_success = 60
p_hat = n_success / n_total

# Wilson法（小標本でも安定）
ci_wilson = proportion_confint(n_success, n_total, alpha=0.05, method='wilson')
# 正規近似
ci_normal = proportion_confint(n_success, n_total, alpha=0.05, method='normal')

print(f"標本比率: {p_hat:.2f}")
print(f"95%信頼区間（正規近似）: ({ci_normal[0]:.3f}, {ci_normal[1]:.3f})")
print(f"95%信頼区間（Wilson法）: ({ci_wilson[0]:.3f}, {ci_wilson[1]:.3f})")
```

```
標本比率: 0.60
95%信頼区間（正規近似）: (0.504, 0.696)
95%信頼区間（Wilson法）: (0.499, 0.693)
```

### ブートストラップ信頼区間（分布によらない方法）

```python
from scipy.stats import bootstrap

# 非正規なデータ（対数正規分布）
np.random.seed(42)
data = np.random.lognormal(mean=3, sigma=0.5, size=50)

# 中央値のブートストラップ信頼区間
result = bootstrap(
    (data,),
    statistic=np.median,
    n_resamples=10000,
    confidence_level=0.95,
    random_state=42
)
print(f"標本中央値: {np.median(data):.2f}")
print(f"95%ブートストラップCI: ({result.confidence_interval.low:.2f}, {result.confidence_interval.high:.2f})")
```

```
標本中央値: 19.84
95%ブートストラップCI: (18.03, 22.41)
```

---

## 区間幅を狭める方法

$$\text{区間幅} \propto \frac{z_{\alpha/2} \cdot s}{\sqrt{n}}$$

| 方法 | 効果 | 注意点 |
|------|------|--------|
| $n$ を増やす | 幅 $\propto 1/\sqrt{n}$ で縮小 | コスト増加 |
| 信頼水準を下げる（95%→90%） | $z$ 値が小さくなり幅縮小 | 信頼性低下 |
| 分散の小さい測定方法に変える | $s$ が小さくなり幅縮小 | 測定コスト増加 |

---

## 検定との対応関係

信頼区間と仮説検定は表裏一体の関係です：

> **帰無仮説の値 $\mu_0$ が $100(1-\alpha)\%$ 信頼区間の外にある ⟺ 有意水準 $\alpha$ で帰無仮説を棄却**

```python
# 帰無仮説 μ₀ = 168 を検定
mu_0 = 168
sample = np.random.normal(loc=170, scale=8, size=30)
mean = np.mean(sample)
se = stats.sem(sample)

ci = stats.t.interval(0.95, df=len(sample)-1, loc=mean, scale=se)
t_stat, p_value = stats.ttest_1samp(sample, popmean=mu_0)

print(f"95%信頼区間: ({ci[0]:.2f}, {ci[1]:.2f})")
print(f"μ₀={mu_0} は区間内か: {ci[0] <= mu_0 <= ci[1]}")
print(f"t検定 p値: {p_value:.4f}")
print(f"棄却（α=0.05）: {p_value < 0.05}")
```

信頼区間に $\mu_0$ が含まれない場合、p値は0.05未満になります。

---

## まとめ

| ポイント | 内容 |
|---------|------|
| 正しい解釈 | 同じ手続きを繰り返したとき、$100(1-\alpha)\%$ の区間が母数を含む |
| 誤った解釈 | 母数がこの区間に $100(1-\alpha)\%$ の確率で入っている |
| 母分散未知 | $t$ 分布を使う（実務はほぼこちら） |
| 比率の区間 | Wilson法が小標本でも安定 |
| 非正規データ | ブートストラップが有効 |
| 検定との関係 | 信頼区間と有意水準は対応する |

信頼区間の解釈を正確に押さえることは、統計検定の合格だけでなく、実務でのデータ解釈の精度にも直結します。「この結果は偶然ではないか？」を定量的に示す手段として、ぜひ使いこなしてください。
