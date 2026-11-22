
import pandas as pd
import re
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "youtube_comments_verified.csv"
OUTPUT_FILE = ROOT / "data" / "cleaned_youtube_comments.csv"

def clean_comment(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    print("Loading comments...")

    df = pd.read_csv(INPUT_FILE)

    print("Original comments:", len(df))

    df["cleaned_comment"] = (
        df["comment"].fillna("").apply(clean_comment)
    )

    df = df[df["cleaned_comment"].str.len() > 0]

    df = df.drop_duplicates(
        subset=["video_id", "cleaned_comment"]
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCleaning completed!")
    print("Remaining comments:", len(df))
    print("Movies:", df["movie"].nunique())
    print("Videos:", df["video_id"].nunique())
    print("Saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
