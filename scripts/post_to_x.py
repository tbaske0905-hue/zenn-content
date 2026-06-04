"""
Zennに記事をpushした後、X投稿用の文言を生成して表示するスクリプト。
カテゴリ別に連番ハッシュタグをつける（例: #統計学1, #AI最新情報2）

使い方:
    python scripts/post_to_x.py articles/20260604-ab-test-statistics.md
"""

import json
import re
import sys
from pathlib import Path

ZENN_USERNAME = "ai_lab_ds"

# scriptsディレクトリのcategory_counts.json
COUNTS_FILE = Path(__file__).parent / "category_counts.json"

# topics → カテゴリ の判定ルール（上から順に最初にマッチしたものを使う）
CATEGORY_RULES = [
    (["statistics"], "統計学"),
    (["llm", "ai"], "AI最新情報"),
    (["mlops"], "データサイエンス"),
    (["pandas", "polars", "pytorch", "tensorflow", "scikit"], "Python実装"),
    (["python"], "Python実装"),
    (["datascience", "machinelearning"], "データサイエンス"),
]

# type: idea でDS成長系の判定
DS_GROWTH_KEYWORDS = ["成長", "キャリア", "習慣", "スキル", "マインド"]

# カテゴリ → Xハッシュタグのプレフィックス
CATEGORY_TAG = {
    "AI最新情報":      "#AI最新情報",
    "DS成長":         "#DS成長",
    "統計学":         "#統計学",
    "Python実装":     "#Python実装",
    "データサイエンス": "#データサイエンス",
}

# topics → 補助ハッシュタグ
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
    """記事のカテゴリを自動判定する。"""
    # タイトルにDS成長系キーワードが含まれていればDS成長
    if article_type == "idea" and any(kw in title for kw in DS_GROWTH_KEYWORDS):
        return "DS成長"

    # topicsで判定
    for rule_topics, category in CATEGORY_RULES:
        if any(t in topics for t in rule_topics):
            return category

    return "データサイエンス"  # デフォルト


def load_counts() -> dict:
    return json.loads(COUNTS_FILE.read_text(encoding="utf-8"))


def save_counts(counts: dict) -> None:
    COUNTS_FILE.write_text(
        json.dumps(counts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generate_tweet(title: str, url: str, topics: list[str], article_type: str) -> str:
    # カテゴリ判定 → 連番取得
    category = detect_category(title, topics, article_type)
    counts = load_counts()
    counts[category] = counts.get(category, 0) + 1
    save_counts(counts)

    # 連番ハッシュタグ（例: #統計学1）
    numbered_tag = f"{CATEGORY_TAG[category]}{counts[category]}"

    # 補助ハッシュタグ（最大2個）
    extra = [EXTRA_TAGS[t] for t in topics if t in EXTRA_TAGS][:2]
    extra_str = " ".join(extra)

    tweet = f"""📝 新記事を公開しました！

{title}

{url}

{numbered_tag} {extra_str} #Zenn"""

    return tweet, category, counts[category]


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

    tweet, category, number = generate_tweet(title, url, topics, article_type)

    print(f"\nカテゴリ: {category}（通算{number}本目）")
    print("\n" + "=" * 50)
    print("📋 X投稿用テキスト（コピーしてXに貼り付け）")
    print("=" * 50)
    print(tweet)
    print("=" * 50)
    print(f"文字数: {len(tweet)} / 280")


if __name__ == "__main__":
    main()
