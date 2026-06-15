---
title: "中心極限定理と大数の法則：「標本平均が正規分布に従う」を数式とシミュレーションで体感する"
emoji: "🎲"
type: "tech"
topics: ["python", "statistics", "datascience"]
published: true
---

## はじめに

「なぜt検定やA/Bテストの信頼区間は、母集団の分布がわからなくても正規分布で近似できるのか？」

この問いに答えるのが**大数の法則（Law of Large Numbers, LLN）**と**中心極限定理（Central Limit Theorem, CLT）**です。この2つは統計的推測・検定のほぼ全ての手法の土台になっており、統計検定でも頻出のテーマです。

本記事では、それぞれの数式・成立条件・実務での使い所を整理し、Pythonのシミュレーションで「収束する様子」を実際に確認します。

---

## 大数の法則（LLN）とは

「サンプルサイズを増やせば、標本平均は母平均に近づく」という直感を数学的に保証する定理です。

### 数式

$X_1, X_2, \ldots, X_n$ を平均 $\mu$、分散 $\sigma^2$ を持つ独立同分布（i.i.d.）の確率変数とする。標本平均を

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$$

とすると、**弱法則（WLLN）**は次のように成り立つ:

$$\bar{X}_n \xrightarrow{p} \mu \quad (n \to \infty)$$

これは「任意の $\varepsilon > 0$ に対して」

$$\lim_{n \to \infty} P(|\bar{X}_n - \mu| > \varepsilon) = 0$$

が成り立つことを意味する（確率収束）。チェビシェフの不等式を使うと、

$$P(|\bar{X}_n - \mu| > \varepsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} = \frac{\sigma^2}{n\varepsilon^2}$$

であり、$n \to \infty$ で右辺が0に収束するため証明できる。

### いつ使うか

- モンテカルロ法で「シミュレーション回数を増やせば真の値に近づく」ことの理論的根拠
- 「サンプルサイズが大きいほど推定値が安定する」という経験則を数式で裏付けたいとき
- A/Bテストで「サンプルサイズを増やせば観測される平均は真の効果量に近づく」と説明するとき

**注意**: LLNは「平均が真の値に近づく」ことのみを保証し、「分布の形」については何も言わない。分布の形を保証するのがCLTである。

### Pythonで実装

サイコロを繰り返し投げ、標本平均が理論値 $3.5$ に収束する様子を確認する。

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n_max = 5000
rolls = rng.integers(1, 7, size=n_max)  # サイコロ: 1〜6
running_mean = np.cumsum(rolls) / np.arange(1, n_max + 1)

plt.figure(figsize=(8, 4))
plt.plot(running_mean, color="steelblue", lw=0.8)
plt.axhline(3.5, color="red", linestyle="--", label="理論値 E[X]=3.5")
plt.xlabel("サイコロを投げた回数 n")
plt.ylabel("標本平均 X̄_n")
plt.title("大数の法則：標本平均は母平均3.5に収束する")
plt.legend(); plt.tight_layout(); plt.show()

print(f"n=10:    平均={rolls[:10].mean():.3f}")
print(f"n=100:   平均={rolls[:100].mean():.3f}")
print(f"n=5000:  平均={rolls.mean():.3f}")
```

`n` が小さいうちは標本平均が大きく揺れるが、`n` が増えるにつれて $3.5$ 付近に収束していく様子が確認できる。

---

## 中心極限定理（CLT）とは

LLNは「標本平均が $\mu$ に収束する」ことしか教えてくれないが、CLTは**収束する際の"揺らぎ"の分布**を教えてくれる。

### 数式

$X_1, \ldots, X_n$ を平均 $\mu$、分散 $\sigma^2$（有限）のi.i.d.確率変数とする。標本平均を標準化した

$$Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}}$$

は、$n \to \infty$ で標準正規分布に**分布収束**する:

$$Z_n \xrightarrow{d} N(0, 1)$$

これを言い換えると、$n$ が十分大きいとき、

$$\bar{X}_n \approx N\left(\mu, \frac{\sigma^2}{n}\right)$$

**ポイント**: 元の確率変数 $X_i$ がどんな分布（一様分布・指数分布・ポアソン分布など）であっても、標本平均 $\bar{X}_n$ は $n$ が大きくなれば正規分布に近づく。

### いつ使うか

- 母集団の分布が不明・非正規でも、$n$ が十分大きければ標本平均を正規分布として扱い、**信頼区間・仮説検定（t検定、z検定）を構成できる**根拠として
- $n \geq 30$ が経験的な目安としてよく使われるが、元の分布の歪みが大きい場合はより大きな $n$ が必要
- ブートストラップ法やA/Bテストの統計的有意性判定の理論的支柱

**CLTが効きにくいケース**:
- 分散が無限大、または極端に歪んだ分布（コーシー分布など）
- $n$ が極端に小さい（数件〜十数件）場合

### Pythonで実装

指数分布（強く右に歪んだ分布）から標本平均を繰り返し計算し、$n$ を増やすとヒストグラムが正規分布に近づく様子を確認する。

```python
from scipy import stats

rng = np.random.default_rng(0)

lam = 1.0          # 指数分布のパラメータ（平均=1, 分散=1）
n_trials = 10000   # シミュレーション回数

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, n in zip(axes, [1, 5, 50]):
    samples = rng.exponential(scale=1/lam, size=(n_trials, n))
    sample_means = samples.mean(axis=1)

    ax.hist(sample_means, bins=40, density=True,
            color="steelblue", alpha=0.7, label="標本平均の分布")

    # CLTが予測する正規分布
    mu, sigma = 1/lam, (1/lam) / np.sqrt(n)
    x = np.linspace(sample_means.min(), sample_means.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="red", lw=2, label="CLTの予測")

    ax.set_title(f"n = {n}")
    ax.legend(fontsize=8)

plt.suptitle("指数分布からの標本平均：nが増えるほど正規分布に近づく")
plt.tight_layout(); plt.show()
```

$n=1$ では元の指数分布そのもの（右に歪んだ形）だが、$n=5$ で歪みが緩和され、$n=50$ ではほぼ正規分布に重なる。

---

## 実務での応用：信頼区間はCLTから導かれる

母平均 $\mu$ の95%信頼区間が

$$\bar{X} \pm 1.96 \times \frac{\sigma}{\sqrt{n}}$$

となるのは、CLTにより $\bar{X} \sim N(\mu, \sigma^2/n)$ と近似できるためである。$\sigma$ が未知の場合は標本標準偏差 $S$ で代用し、自由度 $n-1$ の $t$ 分布を使う。

```python
# 標本から95%信頼区間を計算する例
data = rng.exponential(scale=1/lam, size=50)

x_bar = data.mean()
s = data.std(ddof=1)
n = len(data)

# t分布に基づく信頼区間（小〜中規模サンプルではこちらが正確）
t_crit = stats.t.ppf(0.975, df=n - 1)
ci_t = (x_bar - t_crit * s / np.sqrt(n), x_bar + t_crit * s / np.sqrt(n))

# CLTに基づく正規近似の信頼区間（nが大きいとき）
z_crit = stats.norm.ppf(0.975)
ci_z = (x_bar - z_crit * s / np.sqrt(n), x_bar + z_crit * s / np.sqrt(n))

print(f"標本平均: {x_bar:.3f}")
print(f"t分布による95%CI: ({ci_t[0]:.3f}, {ci_t[1]:.3f})")
print(f"正規近似による95%CI: ({ci_z[0]:.3f}, {ci_z[1]:.3f})")
```

$n$ が大きくなるほど $t$ 分布は正規分布に近づくため、両者の信頼区間の差は縮まっていく。

---

## LLNとCLTの違い

| | 大数の法則（LLN） | 中心極限定理（CLT） |
|---|---|---|
| 主張 | 標本平均は母平均に**収束する** | 標本平均の**分布の形**は正規分布に近づく |
| 必要な仮定 | i.i.d.、平均が有限 | i.i.d.、平均・分散が有限 |
| 教えてくれること | 「中心」がどこか | 「ばらつき」の形と大きさ |
| 主な用途 | モンテカルロ法の妥当性 | 信頼区間・仮説検定の構成 |

---

## まとめ

- **大数の法則**：標本平均はサンプルサイズ $n$ を増やすほど母平均 $\mu$ に確率収束する。「中心」が定まることを保証する定理
- **中心極限定理**：標本平均を標準化すると、元の分布の形に関わらず $n \to \infty$ で標準正規分布に収束する。「ばらつきの形」を保証する定理
- 信頼区間・t検定・z検定など、統計的推測の手法はほぼすべてCLTを土台にしている
- 元の分布が極端に歪んでいる場合や $n$ が小さい場合は、CLTによる正規近似の精度が落ちるため注意が必要

数式だけでは抽象的に感じやすいこの2つの定理も、シミュレーションで「収束していく様子」を可視化すると直感的に理解しやすくなる。
