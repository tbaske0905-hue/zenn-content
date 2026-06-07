"""
Zennに記事をpushした後、X投稿用の文言を生成して表示するスクリプト。
【パターンA：問題提起型】で生成する。

ツイート構成:
  1. ドキっとする状況・失敗例（1〜2行）
  2. → 結果・問題点
  3. なぜそうなるかの一言
  4. 記事の価値を示して「👇」
  5. URL
  6. #カテゴリ連番 補助タグ #Zenn

使い方:
    python scripts/post_to_x.py articles/20260604-ab-test-statistics.md
"""

import json
import re
import sys
from pathlib import Path

ZENN_USERNAME = "ai_lab_ds"
COUNTS_FILE = Path(__file__).parent / "category_counts.json"

# カテゴリ判定ルール（上から順にマッチ）
CATEGORY_RULES = [
    (["statistics"], "統計学"),
    (["llm", "ai"], "AI最新情報"),
    (["mlops"], "データサイエンス"),
    (["pandas", "polars", "pytorch", "tensorflow", "scikit"], "Python実装"),
    (["python"], "Python実装"),
    (["datascience", "machinelearning"], "データサイエンス"),
]
DS_GROWTH_KEYWORDS = ["成長", "キャリア", "習慣", "スキル", "マインド"]

CATEGORY_TAG = {
    "AI最新情報":      "#AI最新情報",
    "DS成長":         "#DS成長",
    "統計学":         "#統計学",
    "Python実装":     "#Python実装",
    "データサイエンス": "#データサイエンス",
}

EXTRA_TAGS = {
    "python":         "#Python",
    "ai":             "#AI",
    "machinelearning":"#機械学習",
    "deeplearning":   "#深層学習",
    "datascience":    "#データサイエンス",
    "statistics":     "#統計",
    "nlp":            "#NLP",
    "llm":            "#LLM",
    "pandas":         "#Pandas",
    "polars":         "#Polars",
    "pytorch":        "#PyTorch",
    "mlops":          "#MLOps",
}

# ===== パターンA：カテゴリ別ツイートテンプレート =====
# {numbered_tag} = #カテゴリ名+連番、{extra} = 補助タグ、{url} = 記事URL
TWEET_TEMPLATES = {
    "統計学": """\
「p値が0.05未満だから効果あり」

この判断、実は危険です。

ベイズ統計なら「効果がある確率は何%か」を直接言えます。
Pythonコードで事前分布・事後分布・信用区間を解説👇

{url}

{numbered_tag} {extra} #Zenn""",

    "AI最新情報": """\
「結局、Claude・GPT・Geminiどれを使えばいいの？」

正直、用途によって答えが変わります。

コーディングはClaude、マルチモーダルはGPT-4o、超長文はGemini。
用途別の使い分けをまとめました👇

{url}

{numbered_tag} {extra} #Zenn""",

    "DS成長": """\
「データサイエンティストになったけど、どう成長すればいい？」

Kaggleより業務データ、コードより統計理解、精度より一言説明。

現場で本当に差がつく習慣5つをまとめました👇

{url}

{numbered_tag} {extra} #Zenn""",

    "Python実装": """\
「ローカルでは動くのに本番でエラーになった」
「前処理の順番を間違えてデータリークが起きた」

どちらも scikit-learn の Pipeline を使えば防げます。

前処理からモデル保存まで、再現性のある実装パターンをまとめました👇

{url}

{numbered_tag} {extra} #Zenn""",

    "データサイエンス": """\
データサイエンスの現場でよく聞く声。

「ツールは使えるけど、なぜこの手法なのかがわからない」

理論と実装をセットで理解するために、まとめてみました👇

{url}

{numbered_tag} {extra} #Zenn""",
}


def parse_frontmatter(content: str) -> dict:
    match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}
    fm = match.group(1)
    result = {}
    if m := re.search(r'^title:\s*"?(.+?)"?\s*$', fm, re.MULTILINE):
        result["title"] = m.group(1).strip('"')
    if m := re.search(r"^topics:\s*\[(.+?)\]", fm, re.MULTILINE):
        result["topics"] = [t.strip().strip('"') for t in m.group(1).split(",")]
    if m := re.search(r'^type:\s*"?(\w+)"?', fm, re.MULTILINE):
        result["type"] = m.group(1)
    return result


def detect_category(title: str, topics: list[str], article_type: str) -> str:
    if article_type == "idea" and any(kw in title for kw in DS_GROWTH_KEYWORDS):
        return "DS成長"
    for rule_topics, category in CATEGORY_RULES:
        if any(t in topics for t in rule_topics):
            return category
    return "データサイエンス"


def load_counts() -> dict:
    return json.loads(COUNTS_FILE.read_text(encoding="utf-8"))


def save_counts(counts: dict) -> None:
    COUNTS_FILE.write_text(
        json.dumps(counts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    if len(sys.argv) != 2:
        print("使い方: python scripts/post_to_x.py articles/ファイル名.md")
        sys.exit(1)

    article_path = Path(sys.argv[1])
    if not article_path.exists():
        print(f"ファイルが見つかりません: {article_path}")
        sys.exit(1)

    content = article_path.read_text(encoding="utf-8")
    meta = parse_frontmatter(content)

    title = meta.get("title", article_path.stem)
    topics = meta.get("topics", [])
    article_type = meta.get("type", "tech")
    slug = article_path.stem
    url = f"https://zenn.dev/{ZENN_USERNAME}/articles/{slug}"

    # カテゴリ判定 → 連番更新
    category = detect_category(title, topics, article_type)
    counts = load_counts()
    counts[category] = counts.get(category, 0) + 1
    save_counts(counts)

    numbered_tag = f"{CATEGORY_TAG[category]}{counts[category]}"
    extra = " ".join(EXTRA_TAGS[t] for t in topics if t in EXTRA_TAGS)[:2]
    extra_tags = " ".join(list(dict.fromkeys(
        [EXTRA_TAGS[t] for t in topics if t in EXTRA_TAGS]
    ))[:2])

    tweet = TWEET_TEMPLATES[category].format(
        url=url,
        numbered_tag=numbered_tag,
        extra=extra_tags,
    )

    print(f"\nカテゴリ: {category}（通算{counts[category]}本目）")
    print("\n" + "=" * 50)
    print("📋 X投稿用テキスト（コピーしてXに貼り付け）")
    print("=" * 50)
    print(tweet)
    print("=" * 50)
    print(f"文字数: {len(tweet)} / 280")


if __name__ == "__main__":
    main()
