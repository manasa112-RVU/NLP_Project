
import pandas as pd
import re
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent

CALENDAR_FILE = ROOT / "data" / "indian_theatrical_movies.csv"
CATALOGUE_FILE = ROOT / "data" / "indian_movies_2026.csv"
OUTPUT_FILE = ROOT / "data" / "released_indian_movies.csv"


def normalize(title):
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def main():
    print("Filtering released theatrical movies...")

    calendar = pd.read_csv(CALENDAR_FILE)
    calendar["release_date"] = pd.to_datetime(
        calendar["release_date"], errors="coerce"
    )

    today = pd.Timestamp(date.today())

    # Keep movies whose listed release date has arrived.
    released = calendar[
        calendar["release_date"].notna()
        & (calendar["release_date"] <= today)
    ].copy()

    # Match against Film Nagar to identify Indian film records
    # and obtain their available language information.
    catalogue = pd.read_csv(CATALOGUE_FILE)

    catalogue["match_title"] = catalogue["title"].apply(normalize)
    released["match_title"] = released["title"].apply(normalize)

    catalogue = catalogue.drop_duplicates("match_title")

    released = released.merge(
        catalogue[["match_title", "languages", "directors"]],
        on="match_title",
        how="left"
    )

    # Keep only records matched with the Indian film catalogue.
    released = released[released["languages"].notna()].copy()

    released["release_date"] = (
        released["release_date"].dt.strftime("%Y-%m-%d")
    )

    released = released.drop(columns=["match_title"])

    released.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFiltering completed!")
    print("Calendar entries:", len(calendar))
    print("Past-dated entries:", len(
        calendar[calendar["release_date"].notna()
                 & (pd.to_datetime(calendar["release_date"]) <= today)]
    ))
    print("Matched Indian catalogue entries:", len(released))
    print("Saved to:", OUTPUT_FILE)

    print("\nLanguage distribution:")
    print(released["languages"].value_counts().head(20))


if __name__ == "__main__":
    main()
