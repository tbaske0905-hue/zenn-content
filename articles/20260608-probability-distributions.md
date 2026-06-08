---
title: "確率分布の使い分け完全ガイド：データサイエンティストが押さえるべき10の分布"
emoji: "📊"
type: "tech"
topics: ["python", "statistics", "datascience", "machinelearning"]
published: true
---

## はじめに

統計・機械学習の学習でつまずくポイントの一つが「どの分布をいつ使うか」です。正規分布は知っていても、ポアソン分布やガンマ分布が実務でどう登場するかがわからない——そんな状態を解消するために、10の主要分布を「使う場面」と「Pythonでの扱い方」を中心に整理します。

## 確率分布の全体像

まず、分布を「離散型」と「連続型」に分けて俯瞰します。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle("主要確率分布の一覧", fontsize=14)

# 離散型
discrete_dists = [
    (stats.binom(n=20, p=0.3),    np.arange(0, 21),  "二項分布\nBin(20, 0.3)"),
    (stats.poisson(mu=4),          np.arange(0, 15),  "ポアソン分布\nPois(4)"),
    (stats.geom(p=0.3),            np.arange(1, 15),  "幾何分布\nGeom(0.3)"),
    (stats.hypergeom(M=50,n=10,N=8), np.arange(0, 9), "超幾何分布\nHypergeom"),
    (stats.nbinom(n=5, p=0.4),     np.arange(0, 20),  "負の二項分布\nNBin(5,0.4)"),
]
for ax, (dist, x, title) in zip(axes[0], discrete_dists):
    ax.bar(x, dist.pmf(x), color="steelblue", alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")

# 連続型
x_cont = np.linspace(-4, 8, 300)
continuous_dists = [
    (stats.norm(0, 1),           np.linspace(-4, 4, 300),   "正規分布\nN(0,1)"),
    (stats.t(df=5),              np.linspace(-4, 4, 300),   "t分布\nt(5)"),
    (stats.chi2(df=5),           np.linspace(0, 20, 300),   "カイ二乗分布\nχ²(5)"),
    (stats.f(dfn=5, dfd=10),     np.linspace(0, 6, 300),    "F分布\nF(5,10)"),
    (stats.gamma(a=2, scale=2),  np.linspace(0, 16, 300),   "ガンマ分布\nGamma(2,2)"),
]
for ax, (dist, x, title) in zip(axes[1], continuous_dists):
    ax.plot(x, dist.pdf(x), color="tomato", linewidth=2)
    ax.fill_between(x, dist.pdf(x), alpha=0.3, color="tomato")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")

plt.tight_layout()
plt.show()
```

---

## 離散型分布

### 1. 二項分布 Bin(n, p)

**使う場面：** 成功確率 p の試行を n 回繰り返したときの成功回数

```python
from scipy import stats

# コインを20回投げたとき、表が8回以上出る確率
n, p = 20, 0.5
dist = stats.binom(n=n, p=p)

prob = 1 - dist.cdf(7)  # P(X >= 8)
print(f"P(X >= 8) = {prob:.4f}")
print(f"期待値: {dist.mean():.1f}, 分散: {dist.var():.1f}")

# 例：クリック率0.03の広告を1000人に表示した場合の期待クリック数
ad_dist = stats.binom(n=1000, p=0.03)
print(f"\n広告クリック数の期待値: {ad_dist.mean():.0f}")
print(f"95%信用区間: [{ad_dist.ppf(0.025):.0f}, {ad_dist.ppf(0.975):.0f}]")
```

### 2. ポアソン分布 Pois(λ)

**使う場面：** 単位時間・面積あたりの「まれなイベント数」

```python
# 1時間あたり平均4件のお問い合わせ → 1時間に7件以上来る確率
lam = 4
dist = stats.poisson(mu=lam)

prob = 1 - dist.cdf(6)
print(f"P(X >= 7) = {prob:.4f}")

# 二項分布 → ポアソン分布への収束（n大・p小・np=λ）
n_large, p_small = 10000, 0.0004  # np = 4
binom_dist = stats.binom(n=n_large, p=p_small)
x = np.arange(0, 15)
print("\n二項分布とポアソン分布の比較:")
for k in range(8):
    print(f"  k={k}: Binom={binom_dist.pmf(k):.5f}, Poisson={dist.pmf(k):.5f}")
```

### 3. 幾何分布 Geom(p)

**使う場面：** 初めて成功するまでの試行回数

```python
# 打率0.3のバッターが初ヒットを打つまでの打席数
p = 0.3
dist = stats.geom(p=p)
print(f"初ヒットが1打席目: {dist.pmf(1):.4f}")
print(f"初ヒットが3打席以内: {dist.cdf(3):.4f}")
print(f"期待打席数: {dist.mean():.2f}")  # 1/p = 3.33
```

---

## 連続型分布

### 4. 正規分布 N(μ, σ²)

**使う場面：** 自然現象の多く・中心極限定理の収束先

```python
mu, sigma = 170, 6  # 身長の分布（仮）
dist = stats.norm(loc=mu, scale=sigma)

# 180cm以上の割合
prob_tall = 1 - dist.cdf(180)
print(f"180cm以上: {prob_tall:.2%}")

# 標準化（z得点）
height = 182
z = (height - mu) / sigma
print(f"身長{height}cmのz得点: {z:.2f}")
print(f"上位: {(1 - stats.norm.cdf(z)):.2%}")

# 正規性の検定
from scipy.stats import shapiro
data = np.random.normal(mu, sigma, 100)
stat, p_value = shapiro(data)
print(f"\nShapiro-Wilk検定: p={p_value:.4f} → {'正規分布と一致' if p_value > 0.05 else '正規分布と異なる'}")
```

### 5. t分布 t(ν)

**使う場面：** 小サンプルの平均推定・t検定の検定統計量

```python
# 自由度ν が増えると正規分布に近づく
x = np.linspace(-4, 4, 300)
plt.figure(figsize=(8, 4))
for df, label in [(1, "t(1)"), (5, "t(5)"), (30, "t(30)")]:
    plt.plot(x, stats.t(df=df).pdf(x), label=label)
plt.plot(x, stats.norm.pdf(x), label="N(0,1)", linestyle="--", color="black")
plt.legend()
plt.title("自由度とt分布の形状変化")
plt.show()

# 小サンプルの平均差の検定
group_a = [12.5, 13.1, 11.8, 14.2, 12.9]
group_b = [14.1, 15.3, 13.8, 16.0, 14.7]
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"独立t検定: t={t_stat:.3f}, p={p_val:.4f}")
```

### 6. カイ二乗分布 χ²(ν)

**使う場面：** 適合度検定・独立性検定・分散の推定

```python
# 独立性検定：性別とアイスクリームの好みに関係があるか
observed = np.array([
    [30, 10, 20],  # 男性：バニラ・チョコ・ストロベリー
    [20, 25, 15],  # 女性
])
chi2, p_val, dof, expected = stats.chi2_contingency(observed)
print(f"χ²={chi2:.3f}, p={p_val:.4f}, 自由度={dof}")
print(f"期待度数:\n{expected.round(1)}")
```

### 7. F分布 F(ν₁, ν₂)

**使う場面：** 分散比の検定・分散分析（ANOVA）

```python
# 3グループの平均値に差があるか（一元配置分散分析）
group1 = [23, 25, 28, 22, 24]
group2 = [30, 32, 35, 29, 31]
group3 = [22, 24, 26, 21, 23]

f_stat, p_val = stats.f_oneway(group1, group2, group3)
print(f"一元配置ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
print("→ 少なくとも1グループの平均が異なる" if p_val < 0.05 else "→ グループ間に有意差なし")
```

### 8. ガンマ分布 Gamma(α, β)

**使う場面：** 待ち時間・ポアソン過程のα番目のイベントまでの時間

```python
# 指数分布はガンマ分布の特殊ケース（α=1）
# 平均10分のコールセンター対応時間
scale = 10  # β（スケールパラメータ）

exp_dist = stats.expon(scale=scale)        # 指数分布
gamma_dist = stats.gamma(a=3, scale=scale)  # ガンマ分布（α=3：3件目まで待つ時間）

print(f"1件対応の中央値: {exp_dist.median():.1f}分")
print(f"3件対応の中央値: {gamma_dist.median():.1f}分")
```

---

## 各分布の選び方まとめ

```python
# 実務での選択フローチャート（擬似コード）
def choose_distribution(data_type, situation):
    if data_type == "離散":
        if situation == "n回試行の成功数":          return "二項分布"
        if situation == "単位時間のイベント数":       return "ポアソン分布"
        if situation == "初成功までの試行数":         return "幾何分布"
        if situation == "非復元抽出の成功数":         return "超幾何分布"
    
    elif data_type == "連続":
        if situation == "自然現象・中心極限定理":      return "正規分布"
        if situation == "小サンプル平均の推定":        return "t分布"
        if situation == "カテゴリカルデータの検定":    return "カイ二乗分布"
        if situation == "分散比・ANOVA":             return "F分布"
        if situation == "待ち時間・信頼性工学":        return "指数/ガンマ分布"
        if situation == "0〜1の割合データ":           return "ベータ分布"
```

## まとめ

| 分布 | 型 | 代表的な使用場面 |
|------|----|----|
| 二項分布 | 離散 | n回試行の成功数（クリック率・合格率） |
| ポアソン分布 | 離散 | 単位時間のイベント数（アクセス数・欠陥数） |
| 幾何分布 | 離散 | 初成功までの試行数 |
| 正規分布 | 連続 | 自然現象全般・CLTの収束先 |
| t分布 | 連続 | 小サンプルの平均推定・t検定 |
| カイ二乗分布 | 連続 | 適合度検定・独立性検定 |
| F分布 | 連続 | 分散分析・分散比検定 |
| ガンマ/指数分布 | 連続 | 待ち時間・信頼性 |
| ベータ分布 | 連続 | 割合パラメータの事前分布 |

「どのデータに・何を知りたいか」から逆算して分布を選ぶ習慣が、統計を使いこなす近道です。
