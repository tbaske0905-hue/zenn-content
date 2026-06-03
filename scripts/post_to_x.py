"""
Zennに記事をpushした後、X投稿用の文言を生成して表示するスクリプト。

使い方:
    python scripts/post_to_x.py articles/20260604-ab-test-statistics.md
"""

import re
import sys
from pathlib import Path

ZENN_USERNAME = "ai_lab_ds"  # ZennのユーザーID

# topicsタグ → Xハッシュタグ の変換マップ
HASHTAG_MAP = {
    "python": "#Python",
    "ai": "#AI",
    "machinelearning": "#機械学習",
    "deeplearning": "#深層学習",
    "datascience": "#データサイエンス",
    "statistics": "#統計学",
    "nlp": "#自然言語処理",
    "llm": "#LLM",
    "mlops": "#MLOps",
    "pandas": "#Pandas",
    "pytorch": "#PyTorch",
    "kaggle": "#Kaggle",
}


def parse_frontmatter(content: str) -> dict:
    """記事のフロントマターからtitleとtopicsを取得する。"""
    match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}

    frontmatter = match.group(1)
    result = {}

    title_match = re.search(r'^title:\s*"?(.+?)"?\s*$', frontmatter, re.MULTILINE)
    if title_match:
        result["title"] = title_match.group(1).strip('"')

    topics_match = re.search(r"^topics:\s*\[(.+?)\]", frontmatter, re.MULTILINE)
    if topics_match:
        raw = topics_match.group(1)
        result["topics"] = [t.strip().strip('"') for t in raw.split(",")]

    return result


def generate_tweet(title: str, url: str, topics: list[str]) -> str:
    """X投稿用のツイート文を生成する。"""
    hashtags = " ".join(
        HASHTAG_MAP.get(t, f"#{t}") for t in topics[:4] if t in HASHTAG_MAP
    )
    # 共通タグ
    hashtags += " #Zenn"

    tweet = f"""📝 新記事を公開しました！

{title}

{url}

{hashtags}"""
    return tweet


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
    slug = article_path.stem
    url = f"https://zenn.dev/{ZENN_USERNAME}/articles/{slug}"

    tweet = generate_tweet(title, url, topics)

    print("\n" + "=" * 50)
    print("📋 X投稿用テキスト（コピーしてXに貼り付け）")
    print("=" * 50)
    print(tweet)
    print("=" * 50)
    print(f"文字数: {len(tweet)} / 280")


if __name__ == "__main__":
    main()
