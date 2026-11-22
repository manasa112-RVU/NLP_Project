
import re
from getpass import getpass
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "released_indian_movies.csv"
OUTPUT_FILE = ROOT / "data" / "youtube_comments_verified.csv"

BASE_URL = "https://www.googleapis.com/youtube/v3"

LANGUAGE_CODES = {
    "Kannada": "kn",
    "Telugu": "te",
    "Tamil": "ta",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Odia": "or",
    "Punjabi": "pa",
    "Manipuri": "mni"
}

REVIEW_WORDS = [
    "review", "public talk", "public opinion",
    "reaction", "audience talk", "movie talk",
    "film review"
]

# Skip titles that are too generic to match reliably.
SKIP_TITLES = {"hi", "video"}


def title_pattern(movie_title):
    words = re.findall(r"[a-z0-9]+", str(movie_title).lower())

    if not words:
        return None

    phrase = r"[\W_]+".join(re.escape(word) for word in words)

    return re.compile(
        r"(?<![a-z0-9])" + phrase + r"(?![a-z0-9])",
        re.IGNORECASE
    )


def is_review_video(video_title, movie_title):
    pattern = title_pattern(movie_title)

    if pattern is None:
        return False

    if not pattern.search(video_title):
        return False

    return any(
        word in video_title.lower()
        for word in REVIEW_WORDS
    )


def search_videos(movie, api_key):
    release_date = pd.Timestamp(movie["release_date"])

    if release_date.tzinfo is None:
        release_date = release_date.tz_localize("UTC")
    else:
        release_date = release_date.tz_convert("UTC")

    response = requests.get(
        f"{BASE_URL}/search",
        params={
            "part": "snippet",
            "q": f'"{movie["title"]}" movie review',
            "type": "video",
            "maxResults": 10,
            "regionCode": "IN",
            "relevanceLanguage": LANGUAGE_CODES[movie["language"]],
            "publishedAfter": release_date.isoformat(),
            "key": api_key
        },
        timeout=30
    )

    response.raise_for_status()

    videos = []

    for item in response.json().get("items", []):
        snippet = item.get("snippet", {})
        video_title = snippet.get("title", "")

        if is_review_video(video_title, movie["title"]):
            videos.append({
                "video_id": item["id"]["videoId"],
                "video_title": video_title,
                "channel_title": snippet.get("channelTitle", ""),
                "video_published_at": snippet.get("publishedAt", "")
            })

    return videos[:3]


def fetch_comments(video, api_key):
    response = requests.get(
        f"{BASE_URL}/commentThreads",
        params={
            "part": "snippet",
            "videoId": video["video_id"],
            "maxResults": 100,
            "textFormat": "plainText",
            "key": api_key
        },
        timeout=30
    )

    if response.status_code in (403, 404):
        return []

    response.raise_for_status()

    comments = []

    for item in response.json().get("items", []):
        details = item["snippet"]["topLevelComment"]["snippet"]

        comment_text = re.sub(
            r"\s+", " ",
            details.get("textDisplay", "")
        ).strip()

        if len(comment_text) >= 15:
            comments.append({
                "comment": comment_text,
                "likes": details.get("likeCount", 0),
                "published_at": details.get("publishedAt", "")
            })

    return comments


def main():
    if not INPUT_FILE.exists():
        print("Movie data file not found:", INPUT_FILE)
        return

    api_key = getpass("YouTube API key: ")

    movies = pd.read_csv(INPUT_FILE)

    movies["release_date"] = pd.to_datetime(
        movies["release_date"],
        errors="coerce"
    )

    movies = movies.dropna(
        subset=["release_date", "languages"]
    )

    candidates = []

    for _, row in movies.iterrows():
        language = str(row["languages"]).strip("[]'\" ")
        title = str(row["title"]).strip()

        if language not in LANGUAGE_CODES or not title:
            continue

        if title.lower() in SKIP_TITLES:
            print("Skipping ambiguous title:", title)
            continue

        candidates.append({
            "title": title,
            "language": language,
            "release_date": row["release_date"]
        })

    selected = (
        pd.DataFrame(candidates)
        .drop_duplicates(["title", "language"])
        .groupby("language", group_keys=False)
        .head(3)
    )

    print("Movies selected:", len(selected))

    results = []

    for _, movie in selected.iterrows():

        print(
            f"\nSearching: {movie['title']} "
            f"({movie['language']})"
        )

        try:
            videos = search_videos(movie, api_key)

        except requests.RequestException as error:
            print("Search failed:", error)
            continue

        print("Potential review videos:", len(videos))

        for video in videos:

            try:
                comments = fetch_comments(video, api_key)

            except requests.RequestException as error:
                print("Comment fetch failed:", error)
                continue

            for comment in comments:
                results.append({
                    "movie": movie["title"],
                    "language": movie["language"],
                    "release_date": movie["release_date"].date(),
                    **video,
                    "video_url": (
                        "https://www.youtube.com/watch?v="
                        + video["video_id"]
                    ),
                    **comment
                })

    columns = [
        "movie",
        "language",
        "release_date",
        "video_title",
        "channel_title",
        "video_published_at",
        "video_id",
        "video_url",
        "comment",
        "likes",
        "published_at"
    ]

    df = pd.DataFrame(results, columns=columns)

    if not df.empty:
        df = df.drop_duplicates(
            ["video_id", "comment"]
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCollection completed!")
    print("Comments collected:", len(df))
    print("Saved to:", OUTPUT_FILE)

    print(
        "Note: title matching reduces false matches "
        "but does not guarantee video relevance."
    )


if __name__ == "__main__":
    main()
