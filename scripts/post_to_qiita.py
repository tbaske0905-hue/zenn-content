"""
Zenn記事をQiitaにも投稿するスクリプト。

使い方:
    python scripts/post_to_qiita.py articles/20260604-ab-test-statistics.md
"""

import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

QIITA_API = "https://qiita.com/api/v2/items"

# Zenn topics → Qiita タグ名
TAG_MAP = {
    "python":          "Python",
    "ai":              "AI",
    "llm":             "LLM",
    "machinelearning": "機械学習",
    "deeplearning":    "DeepLearning",
    "datascience":     "DataScience",
    "statistics":      "統計学",
    "nlp":             "NLP",
    "pandas":          "Pandas",
    "polars":          "Polars",
    "pytorch":         "PyTorch",
    "mlops":           "MLOps",
    "kaggle":          "Kaggle",
}


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """フロントマターと本文を分離して返す。"""
    match = re.search(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not match:
        return {}, content

    fm = match.group(1)
    body = content[match.end():]
    result = {}

    if m := re.search(r'^title:\s*"?(.+?)"?\s*$', fm, re.MULTILINE):
        result["title"] = m.group(1).strip('"')
    if m := re.search(r"^topics:\s*\[(.+?)\]", fm, re.MULTILINE):
        result["topics"] = [t.strip().strip('"') for t in m.group(1).split(",")]

    return result, body


def convert_body(body: str) -> str:
    """Zenn固有の記法をQiita向けに変換する。"""
    # :::note tip/warn/info → Qiitaのnote記法に変換
    body = re.sub(r":::note tip", ":::note", body)
    body = re.sub(r":::note warn", ":::note alert", body)
    body = re.sub(r":::note info", ":::note", body)
    body = re.sub(r":::note\b(?! alert)", ":::note", body)
    return body.strip()


def post_to_qiita(title: str, body: str, topics: list[str]) -> str:
    """Qiita APIに投稿してURLを返す。"""
    token = os.environ.get("QIITA_TOKEN")
    if not token:
        print("Error: QIITA_TOKEN が設定されていません", file=sys.stderr)
        sys.exit(1)

    tags = [{"name": TAG_MAP.get(t, t)} for t in topics[:5]]
    if not tags:
        tags = [{"name": "DataScience"}]

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "title": title,
        "body": body,
        "tags": tags,
        "private": False,
        "tweet": False,
    }

    response = requests.post(QIITA_API, json=payload, headers=headers)

    if response.status_code == 201:
        url = response.json().get("url", "")
        return url
    else:
        print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("使い方: python scripts/post_to_qiita.py articles/ファイル名.md")
        sys.exit(1)

    article_path = Path(sys.argv[1])
    if not article_path.exists():
        print(f"ファイルが見つかりません: {article_path}")
        sys.exit(1)

    content = article_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(content)

    title = meta.get("title", article_path.stem)
    topics = meta.get("topics", [])
    body = convert_body(body)

    print(f"投稿中: {title}")
    url = post_to_qiita(title, body, topics)

    print(f"\n✅ Qiita投稿完了！")
    print(f"URL: {url}")


if __name__ == "__main__":
    main()
