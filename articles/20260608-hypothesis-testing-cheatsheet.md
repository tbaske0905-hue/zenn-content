---
title: "仮説検定チートシート：どの検定をいつ使うか完全ガイド"
emoji: "📋"
type: "tech"
topics: ["statistics", "python", "datascience", "machinelearning"]
published: true
---

## このチートシートの使い方

「2群の平均を比べたい→正規分布に従う？→t検定 or Mann-Whitney検定」のように、データの種類と知りたいことを答えるだけで検定を選べる構成にしています。数式・使用条件・Pythonコードをセットで掲載します。

---

## 検定の選び方フローチャート

```
データの種類は？
├─ 連続値・量的データ
│   └─ 群の数は？
│       ├─ 1群：母平均の検定
│       │   ├─ 正規分布に従う → 1標本t検定
│       │   └─ 非正規 → Wilcoxon符号順位検定
│       ├─ 2群：2群の比較
│       │   ├─ 対応あり（同一対象の前後比較）
│       │   │   ├─ 正規分布 → 対応t検定
│       │   │   └─ 非正規 → Wilcoxon符号順位検定
│       │   └─ 対応なし（独立した2群）
│       │       ├─ 正規分布・等分散 → 独立t検定
│       │       ├─ 正規分布・不等分散 → Welchのt検定
│       │       └─ 非正規 → Mann-Whitney U検定
│       └─ 3群以上
│           ├─ 正規分布 → 一元配置ANOVA → 多重比較
│           └─ 非正規 → Kruskal-Wallis検定
└─ カテゴリカル・度数データ
    ├─ 1変数の分布検定 → χ²適合度検定
    └─ 2変数の独立性 → χ²独立性検定
```

---

## 1. 1標本t検定

### いつ使うか
- 1つのグループの母平均 $\mu$ が特定の値 $\mu_0$ と等しいか検定する
- **前提**：データが正規分布に従う（$n < 30$ のときは特に重要）

### 数式

検定統計量：

$$t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t(n-1)$$

- $\bar{X}$：標本平均、$S$：不偏標準偏差、$n$：サンプルサイズ
- 帰無仮説 $H_0: \mu = \mu_0$

95%信頼区間：

$$\bar{X} \pm t_{0.025}(n-1) \cdot \frac{S}{\sqrt{n}}$$

```python
from scipy import stats
import numpy as np

data = [172, 168, 175, 171, 169, 174, 170, 173]
mu_0 = 170  # 帰無仮説：母平均は170cm

t_stat, p_val = stats.ttest_1samp(data, popmean=mu_0)
ci = stats.t.interval(0.95, df=len(data)-1,
                      loc=np.mean(data), scale=stats.sem(data))
print(f"t={t_stat:.3f}, p={p_val:.4f}")
print(f"95%信頼区間: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

---

## 2. 独立2標本t検定（Welchのt検定）

### いつ使うか
- **独立した2群**の母平均の差を検定する
- 等分散が保証されない場合は **Welch検定**（デフォルトで推奨）

### 数式

Welchのt統計量：

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\dfrac{S_1^2}{n_1} + \dfrac{S_2^2}{n_2}}}$$

自由度（Welch-Satterthwaite近似）：

$$\nu \approx \frac{\left(\dfrac{S_1^2}{n_1}+\dfrac{S_2^2}{n_2}\right)^2}{\dfrac{(S_1^2/n_1)^2}{n_1-1}+\dfrac{(S_2^2/n_2)^2}{n_2-1}}$$

等分散を仮定する場合（Student）：

$$t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}, \quad S_p^2 = \frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}$$

```python
group_a = [23.1, 25.4, 22.8, 24.6, 23.9]
group_b = [27.3, 29.1, 26.8, 28.5, 27.9]

# Welch検定（equal_var=False がデフォルト）
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"Welch t={t_stat:.3f}, p={p_val:.4f}")

# 等分散検定（まずLevene検定で分散が等しいか確認）
stat, p_levene = stats.levene(group_a, group_b)
print(f"Levene検定（等分散?）: p={p_levene:.4f} → {'等分散' if p_levene>0.05 else '不等分散'}")
```

---

## 3. 対応t検定（Paired t-test）

### いつ使うか
- **同一対象の前後比較**（治療前後・A/Bテストの同一ユーザーなど）
- 対応のある差 $D_i = X_{1i} - X_{2i}$ の平均が0かを検定

### 数式

差 $D_i = X_{1i} - X_{2i}$ を1標本として1標本t検定：

$$t = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t(n-1), \quad H_0: \mu_D = 0$$

```python
before = [82, 79, 75, 88, 85, 90, 83, 78]
after  = [78, 74, 71, 83, 80, 85, 79, 73]

t_stat, p_val = stats.ttest_rel(before, after)
d = np.array(before) - np.array(after)
print(f"平均差: {d.mean():.2f}")
print(f"対応t検定: t={t_stat:.3f}, p={p_val:.4f}")
```

---

## 4. Mann-Whitney U検定（ノンパラメトリック）

### いつ使うか
- 正規分布を仮定できない2群の比較
- 順位データや外れ値が多いデータ
- 独立t検定の代替

### 数式

U統計量：

$$U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$

- $R_1$：グループ1の順位和
- $H_0$：2群の分布は同一（中央値に差がない）

大標本近似（$n_1, n_2 > 8$）：

$$z = \frac{U - \mu_U}{\sigma_U}, \quad \mu_U = \frac{n_1 n_2}{2}, \quad \sigma_U = \sqrt{\frac{n_1 n_2 (n_1+n_2+1)}{12}}$$

```python
group_a = [5, 8, 3, 6, 9, 4, 7]
group_b = [12, 15, 10, 14, 11, 13, 16]

u_stat, p_val = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
print(f"U={u_stat:.1f}, p={p_val:.4f}")
```

---

## 5. 一元配置分散分析（ANOVA）

### いつ使うか
- **3群以上の平均**を同時に比較する
- 前提：各群が正規分布・等分散
- 有意ならPost-hoc多重比較（Tukey HSD等）で差のある群を特定

### 数式

F統計量：

$$F = \frac{MS_{\text{between}}}{MS_{\text{within}}} \sim F(k-1, N-k)$$

$$SS_{\text{between}} = \sum_{j=1}^k n_j(\bar{X}_j - \bar{X})^2, \quad SS_{\text{within}} = \sum_{j=1}^k\sum_{i=1}^{n_j}(X_{ij}-\bar{X}_j)^2$$

$$MS_{\text{between}} = \frac{SS_{\text{between}}}{k-1}, \quad MS_{\text{within}} = \frac{SS_{\text{within}}}{N-k}$$

- $k$：群の数、$N$：全サンプル数

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

g1 = [23, 25, 28, 22, 24]
g2 = [30, 32, 35, 29, 31]
g3 = [22, 24, 26, 21, 23]

f_stat, p_val = stats.f_oneway(g1, g2, g3)
print(f"ANOVA: F={f_stat:.3f}, p={p_val:.6f}")

if p_val < 0.05:
    # Tukey HSD で多重比較
    data = g1 + g2 + g3
    groups = ["G1"]*len(g1) + ["G2"]*len(g2) + ["G3"]*len(g3)
    result = pairwise_tukeyhsd(data, groups, alpha=0.05)
    print(result)
```

---

## 6. χ²検定（独立性・適合度）

### いつ使うか
- **独立性検定**：2つのカテゴリカル変数に関連があるか（例：性別 × 購買行動）
- **適合度検定**：観測度数が期待分布と一致するか

### 数式

χ²統計量：

$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \sim \chi^2\bigl((r-1)(c-1)\bigr)$$

期待度数：

$$E_{ij} = \frac{(\text{行和}_i)(\text{列和}_j)}{N}$$

**注意**：$E_{ij} < 5$ のセルが多い場合はFisherの正確検定を使う。

```python
# 独立性検定
observed = np.array([[40, 20], [30, 50]])
chi2, p, dof, expected = stats.chi2_contingency(observed)
print(f"χ²={chi2:.3f}, p={p:.4f}, 自由度={dof}")
print(f"期待度数:\n{expected}")

# 適合度検定（サイコロが公平か）
observed_dice = [18, 14, 22, 17, 16, 13]  # 6面の出現回数
expected_dice = [100/6] * 6               # 期待値（均等）
chi2, p = stats.chisquare(observed_dice, f_exp=expected_dice)
print(f"\n適合度検定: χ²={chi2:.3f}, p={p:.4f}")
```

---

## 7. 相関係数の検定

### いつ使うか
- **Pearson相関**：両変数が正規分布に従う連続値の線形関係
- **Spearman相関**：順位データ・非正規・単調関係

### 数式

Pearson相関係数：

$$r = \frac{\sum(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum(X_i-\bar{X})^2 \cdot \sum(Y_i-\bar{Y})^2}}$$

帰無仮説 $H_0: \rho = 0$ の検定統計量：

$$t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} \sim t(n-2)$$

```python
x = [2.1, 3.4, 4.2, 5.0, 6.1, 7.3, 8.0]
y = [1.5, 2.8, 3.9, 4.4, 5.8, 6.2, 7.5]

r_pearson, p_pearson = stats.pearsonr(x, y)
r_spearman, p_spearman = stats.spearmanr(x, y)

print(f"Pearson  r={r_pearson:.4f}, p={p_pearson:.4f}")
print(f"Spearman r={r_spearman:.4f}, p={p_spearman:.4f}")
```

---

## 検定まとめ一覧表

| 目的 | 条件 | 検定 | 統計量の分布 |
|------|------|------|------------|
| 1群の平均 | 正規分布 | 1標本t検定 | $t(n-1)$ |
| 1群の平均 | 非正規 | Wilcoxon符号順位 | 正規近似 |
| 2群（独立）の平均 | 正規・等分散 | Student t検定 | $t(n_1+n_2-2)$ |
| 2群（独立）の平均 | 正規・不等分散 | Welch t検定 | $t(\nu_{\text{Welch}})$ |
| 2群（独立）の平均 | 非正規 | Mann-Whitney U | 正規近似 |
| 2群（対応）の平均 | 正規分布 | 対応t検定 | $t(n-1)$ |
| 2群（対応）の平均 | 非正規 | Wilcoxon符号順位 | 正規近似 |
| 3群以上の平均 | 正規・等分散 | 一元配置ANOVA | $F(k-1, N-k)$ |
| 3群以上の平均 | 非正規 | Kruskal-Wallis | $\chi^2(k-1)$ |
| 2変数の独立性 | カテゴリカル | χ²独立性検定 | $\chi^2((r-1)(c-1))$ |
| 分布の当てはまり | カテゴリカル | χ²適合度検定 | $\chi^2(k-1)$ |
| 2群の分散比 | 正規分布 | F検定 | $F(n_1-1, n_2-1)$ |
| 線形相関 | 正規分布 | Pearson相関 | $t(n-2)$ |
| 単調相関 | 順位・非正規 | Spearman相関 | 正規近似 |

## 検定を選ぶ3つの確認事項

1. **データの型**：連続値 or カテゴリカル
2. **群の構造**：1群・対応あり2群・独立2群・3群以上
3. **正規性の確認**：`scipy.stats.shapiro(data)` で p > 0.05 なら正規分布と判断

```python
# 正規性の確認（Shapiro-Wilk）
data = [23.1, 25.4, 22.8, 24.6, 23.9, 26.1, 22.3]
stat, p = stats.shapiro(data)
print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")
print("→ 正規分布と判断" if p > 0.05 else "→ 非正規、ノンパラ検定を使う")

# 等分散の確認（Levene検定）
g1 = [23.1, 25.4, 22.8, 24.6, 23.9]
g2 = [27.3, 29.1, 26.8, 28.5, 27.9]
stat, p = stats.levene(g1, g2)
print(f"\nLevene検定: W={stat:.4f}, p={p:.4f}")
print("→ 等分散と判断（Student t検定可）" if p > 0.05 else "→ 不等分散（Welch t検定を使う）")
```
