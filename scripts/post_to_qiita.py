"""
Zenn記事をQiitaに投稿・更新するスクリプト。
- 新規：POST /api/v2/items
- 更新：PATCH /api/v2/items/:id  (qiita_ids.json にIDが登録済みの場合)

使い方:
    python scripts/post_to_qiita.py articles/20260604-ab-test-statistics.md
"""

import json
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

QIITA_API   = "https://qiita.com/api/v2/items"
IDS_FILE    = Path(__file__).parent / "qiita_ids.json"

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
    match = re.search(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not match:
        return {}, content
    fm   = match.group(1)
    body = content[match.end():]
    result = {}
    if m := re.search(r'^title:\s*"?(.+?)"?\s*$', fm, re.MULTILINE):
        result["title"] = m.group(1).strip('"')
    if m := re.search(r"^topics:\s*\[(.+?)\]", fm, re.MULTILINE):
        result["topics"] = [t.strip().strip('"') for t in m.group(1).split(",")]
    return result, body


def convert_body(body: str) -> str:
    body = re.sub(r":::note tip",   ":::note",       body)
    body = re.sub(r":::note warn",  ":::note alert", body)
    body = re.sub(r":::note info",  ":::note",       body)
    body = re.sub(r":::note\b(?! alert)", ":::note", body)
    return body.strip()


def load_ids() -> dict:
    if IDS_FILE.exists():
        return json.loads(IDS_FILE.read_text(encoding="utf-8"))
    return {}


def save_ids(ids: dict) -> None:
    IDS_FILE.write_text(
        json.dumps(ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def call_api(method: str, url: str, payload: dict, token: str) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.request(method, url, json=payload, headers=headers)
    if resp.status_code in (200, 201):
        return resp.json()
    elif resp.status_code == 429:
        print("⚠️  レート制限中です。時間をおいて再実行してください。", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("使い方: python scripts/post_to_qiita.py articles/ファイル名.md")
        sys.exit(1)

    article_path = Path(sys.argv[1])
    if not article_path.exists():
        print(f"ファイルが見つかりません: {article_path}")
        sys.exit(1)

    token = os.environ.get("QIITA_TOKEN")
    if not token:
        print("Error: QIITA_TOKEN が設定されていません", file=sys.stderr)
        sys.exit(1)

    content      = article_path.read_text(encoding="utf-8")
    meta, body   = parse_frontmatter(content)
    title        = meta.get("title", article_path.stem)
    topics       = meta.get("topics", [])
    body         = convert_body(body)
    tags         = [{"name": TAG_MAP.get(t, t)} for t in topics[:5]] or [{"name": "DataScience"}]

    payload = {
        "title":   title,
        "body":    body,
        "tags":    tags,
        "private": False,
        "tweet":   False,
    }

    ids      = load_ids()
    filename = article_path.name
    item_id  = ids.get(filename)

    if item_id:
        print(f"更新中: {title}")
        result = call_api("PATCH", f"{QIITA_API}/{item_id}", payload, token)
        url = result.get("url", f"https://qiita.com/items/{item_id}")
        print(f"\n✅ Qiita更新完了！")
    else:
        print(f"新規投稿中: {title}")
        result = call_api("POST", QIITA_API, payload, token)
        url     = result.get("url", "")
        item_id = result.get("id", "")
        ids[filename] = item_id
        save_ids(ids)
        print(f"\n✅ Qiita投稿完了！")

    print(f"URL: {url}")


if __name__ == "__main__":
    main()
