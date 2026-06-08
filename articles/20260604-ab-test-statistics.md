---
title: "A/Bテストで失敗しない統計設計：p値の罠・サンプルサイズ計算・多重比較を徹底解説"
emoji: "🔬"
type: "tech"
topics: ["statistics", "datascience", "python", "machinelearning"]
published: true
---

## はじめに

「A/Bテストで有意差が出たので施策を展開した。でも結果が全然改善されなかった」——よくある失敗です。

A/Bテストは正しく設計しなければ、**誤った結論を「科学的に」導いてしまう**危険なツールになります。本記事では現場でよく起きる3つの統計的ミスを、数式とPythonコードで解説します。

---

## 仮説検定の基礎：いつ使うか

### いつ使うか
- 2つのグループ（A群・B群）の平均値・割合を比較したいとき
- 「施策の効果がある」という主張を統計的に裏付けたいとき
- ランダム割り当て実験（RCT）の結果を評価するとき

### 数式

帰無仮説 $H_0$ と対立仮説 $H_1$ を設定する。

$$H_0: p_A = p_B \quad \text{（効果なし）}$$
$$H_1: p_A \neq p_B \quad \text{（効果あり、両側検定）}$$

コンバージョン率の差の検定では、**Z検定統計量**を使う。

$$z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A}+\frac{1}{n_B}\right)}}$$

- $\hat{p}_A, \hat{p}_B$：各群の標本割合
- $\hat{p}$：プールされた割合 $= \dfrac{x_A + x_B}{n_A + n_B}$
- $n_A, n_B$：各群のサンプルサイズ

p値は $z$ が標準正規分布に従うとした確率：

$$p = 2 \times \left(1 - \Phi(|z|)\right)$$

```python
from scipy import stats
import numpy as np

# A群：1000人中50回クリック、B群：1000人中65回クリック
n_a, x_a = 1000, 50
n_b, x_b = 1000, 65

p_a = x_a / n_a
p_b = x_b / n_b
p_pool = (x_a + x_b) / (n_a + n_b)

z = (p_b - p_a) / np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"z統計量: {z:.4f}")
print(f"p値: {p_value:.4f}")
print("有意差あり" if p_value < 0.05 else "有意差なし")
```

---

## 失敗1：サンプルサイズを事前に決めない

### いつ使うか
- テスト開始前に「何人集めれば十分か」を決めるとき
- 「10%改善を検出したい」など、最小検出効果量（MDE）を設定するとき

### 数式

必要サンプルサイズは以下の式で求める。

$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot 2\bar{p}(1-\bar{p})}{(p_A - p_B)^2}$$

- $z_{\alpha/2}$：有意水準 $\alpha$ に対応する正規分布の上側分位点（$\alpha=0.05$ なら $z_{0.025} = 1.96$）
- $z_{\beta}$：検出力 $1-\beta$ に対応する分位点（検出力 $80\%$ なら $z_{0.2} = 0.842$）
- $\bar{p} = (p_A + p_B)/2$：プールされた割合
- 第一種過誤：$\alpha$（偽陽性：効果がないのに「あり」と判断する確率）
- 第二種過誤：$\beta$（偽陰性：効果があるのに「なし」と判断する確率）

$$\alpha = P(\text{棄却} \mid H_0 \text{ が真}) \quad \beta = P(\text{採択} \mid H_1 \text{ が真})$$

```python
from scipy import stats
import numpy as np

def calculate_sample_size(
    baseline_rate: float,
    relative_mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """A/Bテストに必要なサンプルサイズを計算する"""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + relative_mde)

    z_alpha = stats.norm.ppf(1 - alpha / 2)   # 両側検定
    z_beta  = stats.norm.ppf(power)

    p_bar = (p1 + p2) / 2
    n = (z_alpha + z_beta) ** 2 * 2 * p_bar * (1 - p_bar) / (p1 - p2) ** 2
    return int(np.ceil(n))


n = calculate_sample_size(baseline_rate=0.05, relative_mde=0.10)
print(f"片群あたり必要サンプル数: {n:,}")   # → 約 14,751
print(f"合計サンプル数: {n * 2:,}")

# 有意水準・検出力のトレードオフ
print("\n--- サンプルサイズのトレードオフ ---")
for power in [0.7, 0.8, 0.9]:
    n = calculate_sample_size(0.05, 0.10, power=power)
    print(f"検出力 {power:.0%} → 片群 {n:,} 人")
```

| 設定 | 記号 | 意味 |
|------|------|------|
| 有意水準 | $\alpha = 0.05$ | 偽陽性を5%以下に抑える |
| 検出力 | $1-\beta = 0.80$ | 真の効果を80%の確率で検出する |
| MDE | $\delta = 10\%$ 改善 | これより小さい効果は無視する |

---

## 失敗2：途中経過を見て止める（ピーキング問題）

### いつ使うか（= 使ってはいけない状況）
- 毎日ダッシュボードを確認して「有意になったら終了」は **NG**
- 事前に決めたサンプルサイズに達する前に判断してはいけない

### 数式

$k$ 回確認する場合、ファミリーワイズエラー率（FWER）は：

$$\text{FWER} = 1 - (1 - \alpha)^k$$

例：$\alpha=0.05$、途中確認が $k=6$ 回なら $\text{FWER} \approx 1 - 0.95^6 \approx 26.5\%$

```python
import numpy as np
from scipy import stats

def simulate_peeking(n_simulations: int = 10_000) -> float:
    """ピーキングによる偽陽性率をシミュレーション（真の効果なし）"""
    false_positive = 0
    check_points = [100, 200, 300, 500, 700, 1000]

    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        control   = rng.binomial(1, 0.10, 1000)
        treatment = rng.binomial(1, 0.10, 1000)  # 効果なし

        for n in check_points:
            _, p = stats.ttest_ind(control[:n], treatment[:n])
            if p < 0.05:
                false_positive += 1
                break

    return false_positive / n_simulations

fpr = simulate_peeking()
print(f"通常（ピーキングなし）: 5.0%")
print(f"ピーキングあり: {fpr:.1%}")  # → 約 20〜26%

# 理論値との比較
k = 6
theoretical_fwer = 1 - (1 - 0.05) ** k
print(f"理論値 FWER（k={k}）: {theoretical_fwer:.1%}")
```

:::note warn
「もう少しで有意になりそう」という理由でテストを延長するのも同じ罠です。サンプルサイズは事前に固定してください。
:::

---

## 失敗3：複数の指標を同時に検定する（多重比較問題）

### いつ使うか
- 複数の指標（CVR・CTR・滞在時間など）を同時に検定するとき
- Bonferroni補正やBH法で誤検出率をコントロールする

### 数式

$m$ 個の検定を同時に行う場合の族別エラー率（FWER）：

$$\text{FWER} = 1 - (1 - \alpha)^m$$

**Bonferroni補正**：各検定の有意水準を $\alpha/m$ に下げる

$$\alpha_{\text{Bonferroni}} = \frac{\alpha}{m}$$

**Benjamini-Hochberg法（FDR制御）**：$p$ 値を昇順に並べ、$k$ 番目の検定の棄却条件：

$$p_{(k)} \leq \frac{k}{m} \cdot q^*$$

- $q^*$：制御したいFDR水準（例：$q^* = 0.05$）

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

alpha = 0.05
n_tests = 5
family_wise_error = 1 - (1 - alpha) ** n_tests
print(f"{n_tests}指標を同時検定 → 誤検出率: {family_wise_error:.1%}")

# 模擬p値（5指標の検定結果）
p_values = np.array([0.001, 0.032, 0.048, 0.072, 0.310])

# Bonferroni補正
reject_bonf, p_bonf, _, _ = multipletests(p_values, alpha=alpha, method="bonferroni")
# Benjamini-Hochberg（FDR制御）
reject_bh, p_bh, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

print("\n指標   | p値  | Bonferroni | BH法")
print("-" * 42)
for i, (p, rb, rbh) in enumerate(zip(p_values, reject_bonf, reject_bh)):
    print(f"指標{i+1}  | {p:.3f} | {'✅' if rb else '❌'}        | {'✅' if rbh else '❌'}")
```

| 手法 | 制御するもの | 特徴 | 使い所 |
|------|------------|------|--------|
| Bonferroni | FWER | 保守的・シンプル | 指標が少ない（〜5個） |
| BH法 | FDR | バランスが良い | 指標が多い（10個〜） |
| 主指標を1つに絞る | — | 最もシンプル | 可能なら一番おすすめ |

---

## 正しいA/Bテスト設計チェックリスト

| フェーズ | チェック項目 |
|---------|------------|
| テスト前 | 主指標（Primary Metric）を1つだけ決めた |
| テスト前 | MDEをビジネス観点で決めた |
| テスト前 | 必要サンプルサイズを事前に計算した（上式を使用） |
| テスト前 | テスト期間を事前に固定した |
| テスト中 | 途中で結果を見て判断しない（ピーキング禁止） |
| テスト後 | 事前に決めたサンプルサイズに達してから判断する |
| テスト後 | 効果量（実際の改善幅）もあわせて報告する |

## まとめ

| 落とし穴 | 問題 | 数式的背景 | 対策 |
|---------|------|-----------|------|
| サンプルサイズ不足 | 偽陰性・偽陽性増加 | $n \propto (z_\alpha + z_\beta)^2$ | 事前に計算して固定 |
| ピーキング | FWER増大 | $\text{FWER} = 1-(1-\alpha)^k$ | 判断タイミングを事前に決める |
| 多重比較 | 偶然の有意差を拾う | $\text{FWER} = 1-(1-\alpha)^m$ | 主指標を1つに絞る or 補正 |
