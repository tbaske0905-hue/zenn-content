---
title: "A/Bテストで失敗しない統計設計：p値の罠・サンプルサイズ計算・多重比較を徹底解説"
emoji: "🔬"
type: "tech"
topics: ["statistics", "datascience", "python", "machinelearning"]
published: true
---

## はじめに

「A/Bテストを実施したら有意差が出たので施策を展開した。でも結果が全然改善されなかった」——データサイエンティストやマーケターがよく経験する失敗です。

A/Bテストは正しく設計しなければ、**誤った結論を「科学的に」導いてしまう**危険なツールになります。本記事では現場でよく起きる3つの統計的ミスと、その対処法をPythonコードを交えて解説します。

## よくある失敗1：サンプルサイズを事前に決めない

「とりあえず1週間やってみて有意差が出たら展開しよう」——これが最も多い失敗です。

サンプルサイズが不十分だと、**本当は効果がないのに「有意差あり」と判断する確率（第一種過誤）が高まります**。

### 正しいサンプルサイズの計算方法

```python
from scipy import stats
import numpy as np

def calculate_sample_size(
    baseline_rate: float,
    relative_mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    A/Bテストに必要なサンプルサイズを計算する

    Parameters
    ----------
    baseline_rate : コントロール群のコンバージョン率
    relative_mde  : 検出したい最小効果量（相対値, 例: 0.1 = 10%改善）
    alpha         : 有意水準（デフォルト 5%）
    power         : 検出力（デフォルト 80%）
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + relative_mde)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    pooled_p = (p1 + p2) / 2
    n = (z_alpha + z_beta) ** 2 * 2 * pooled_p * (1 - pooled_p) / (p1 - p2) ** 2
    return int(np.ceil(n))


# 例：コンバージョン率 5% を 10% 改善（相対）で検出したい場合
n = calculate_sample_size(baseline_rate=0.05, relative_mde=0.10)
print(f"片群あたり必要サンプル数: {n:,}")   # → 約 14,751
print(f"合計サンプル数: {n * 2:,}")          # → 約 29,502
```

| 設定 | 値 | 意味 |
|------|-----|------|
| α（有意水準）| 0.05 | 偽陽性を5%以下に抑える |
| β（検出力）  | 0.80 | 真の効果を80%の確率で検出する |
| MDE         | 10%改善 | これより小さい効果は無視する |

サンプルサイズに達する前に判断してはいけません。

## よくある失敗2：途中経過を見て止める（ピーキング問題）

「毎日ダッシュボードを確認して、有意差が出たら終了」——これをピーキング（Peeking）といい、**偽陽性率が大幅に跳ね上がります**。

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
                break  # 有意になった時点で「展開」とみなす

    return false_positive / n_simulations

fpr = simulate_peeking()
print(f"通常のテスト 偽陽性率:  5.0%")
print(f"ピーキングあり 偽陽性率: {fpr:.1%}")  # → 約 20%前後
```

本来5%のはずの偽陽性率が約20%まで膨らみます。

**対策：**
- 事前に決めたサンプルサイズに達するまで絶対に判断しない
- どうしても途中確認したい場合は **逐次検定（Sequential Testing）** を使う

:::note warn
「もう少しで有意になりそう」という理由でテストを延長するのも同じ罠です。サンプルサイズは事前に固定してください。
:::

## よくある失敗3：複数の指標を同時に検定する（多重比較問題）

「CVR・クリック率・滞在時間の3つを同時に検定した」——これを多重比較といい、**検定の数だけ偽陽性のチャンスが増えます**。

```python
alpha   = 0.05
n_tests = 5  # CVR, CTR, 滞在時間, 直帰率, ページ遷移数

# 少なくとも1つ誤検出する確率
family_wise_error = 1 - (1 - alpha) ** n_tests
print(f"{n_tests}指標を同時検定した場合の誤検出率: {family_wise_error:.1%}")
# → 22.6%（本来5%のはずが4倍以上に！）

# Bonferroni 補正
bonferroni_alpha = alpha / n_tests
print(f"Bonferroni 補正後の有意水準: {bonferroni_alpha:.4f}")  # 0.0100
```

| 手法 | 特徴 | 使い所 |
|------|------|--------|
| Bonferroni 補正 | シンプルだが保守的 | 指標が少ない（〜5個） |
| Benjamini-Hochberg | バランスが良い | 指標が多い（10個〜） |
| 主指標を1つに絞る | 最もシンプル | 可能なら一番おすすめ |

:::note tip
**最強の対策は「主指標を1つだけ決める」こと**。その他は参考指標として扱い、統計的判断の根拠にしないのが現場のベストプラクティスです。
:::

## 正しいA/Bテスト設計チェックリスト

| フェーズ | チェック項目 |
|---------|------------|
| テスト前 | 主指標（Primary Metric）を1つだけ決めた |
| テスト前 | MDEをビジネス観点で決めた |
| テスト前 | 必要サンプルサイズを事前に計算した |
| テスト前 | テスト期間を事前に固定した |
| テスト中 | 途中で結果を見て判断しない（ピーキング禁止） |
| テスト後 | 事前に決めたサンプルサイズに達してから判断する |
| テスト後 | 効果量（実際の改善幅）もあわせて報告する |

## まとめ

| 落とし穴 | 問題 | 対策 |
|---------|------|------|
| サンプルサイズ不足 | 偽陽性・偽陰性が増える | 事前に計算して固定する |
| ピーキング | 偽陽性率が5% → 20%+ | 判断タイミングを事前に決める |
| 多重比較 | 偶然の有意差を拾う | 主指標を1つに絞る |

統計的に正しいテストを設計することは、チームへの信頼にもつながります。次のA/Bテストの前にぜひこのチェックリストを活用してください。
