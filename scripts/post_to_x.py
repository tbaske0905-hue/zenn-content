"""
Zennに記事をpushした後、XにURLを投稿するスクリプト。

使い方:
    python scripts/post_to_x.py "記事タイトル" "https://zenn.dev/..."

環境変数（.env または GitHub Secrets に設定）:
    X_API_KEY
    X_API_SECRET
    X_ACCESS_TOKEN
    X_ACCESS_TOKEN_SECRET
    ZENN_USERNAME  （例: tbaske0905）
"""

import os
import sys
import tweepy


def post_to_x(title: str, url: str) -> None:
    client = tweepy.Client(
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"],
    )

    tweet = f"""📝 新記事を公開しました！

{title}

{url}

#Zenn #データサイエンス #AI #Python"""

    response = client.create_tweet(text=tweet)
    print(f"✅ X投稿完了: https://x.com/i/web/status/{response.data['id']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使い方: python scripts/post_to_x.py \"タイトル\" \"URL\"")
        sys.exit(1)

    post_to_x(title=sys.argv[1], url=sys.argv[2])
