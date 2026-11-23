"""
CineSense - NLP-based movie review sentiment analysis platform.

Run with:  streamlit run app.py

Pipeline (see the About page for the full explanation):
  YouTube comments / user text -> preprocessing -> script detection ->
  routing to English or multilingual transformer -> 3-class sentiment ->
  rule checks (mixed opinions, emoji conflicts) -> charts and tables.
"""
from __future__ import annotations

import base64
import hashlib
import html
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import emoji
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="CineSense | Movie Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MOVIES_FILE = DATA_DIR / "released_indian_movies.csv"
SENTIMENT_FILE = DATA_DIR / "sentiment_results.csv"
VERIFIED_FILE = DATA_DIR / "youtube_comments_verified.csv"
CLEANED_FILE = DATA_DIR / "cleaned_youtube_comments.csv"
COLLECTED_FILE = DATA_DIR / "youtube_comments_collected.csv"  # written by this app
POSTER_CACHE_FILE = DATA_DIR / "poster_cache_v4.csv"             # written by this app
ANALYSIS_CACHE_FILE = DATA_DIR / "analysis_cache.csv"         # written by this app

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
WIKI_API = "https://en.wikipedia.org/w/api.php"
HTTP_HEADERS = {
    "User-Agent": "CineSense/1.0 (academic NLP student project; poster lookup via Wikipedia API)"
}

ENGLISH_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MULTI_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

# Bump this string whenever the analysis rules change; it invalidates the disk cache.
ANALYSIS_VERSION = "v3"

try:
    ADMIN = str(st.query_params.get("admin", "")) == "1"
except Exception:  # noqa: BLE001
    ADMIN = False

PAGES = ["Home", "Dashboard", "Review Analyzer", "Movie Explorer", "Viewer Reviews", "About"]
ORDER = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#6ee7a8", "Neutral": "#b8a9d9", "Negative": "#ff6f91"}


# =========================================================
# 2. DESIGN (dark cinematic theme, purple accents)
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&display=swap');

:root {
  --bg:#0a0713; --panel:#16101f; --panel2:#1e1530; --purple:#9b6cff; --purple2:#7045d8;
  --text:#f5f1ff; --muted:#bfb0d8; --border:rgba(181,145,255,.22);
  --display:'Fraunces', Georgia, serif; --body:'Manrope', system-ui, sans-serif;
}
.stApp {
  background: radial-gradient(ellipse at 8% 0%, #2a1646 0%, transparent 38%),
              radial-gradient(ellipse at 95% 12%, #1c1233 0%, transparent 36%), var(--bg);
  color: var(--text); font-family: var(--body);
}
.stApp p, .stApp label, .stApp li, .stApp span,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stCaptionContainer"] { color: #e9e0f8; }
h1, h2, h3, h4 { font-family: var(--display) !important; color: #fff !important; letter-spacing: -.01em; }
.block-container { max-width: 1400px; padding-top: 1.6rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }
hr { border-color: var(--border); }
a { color: #cdb4ff !important; }

[data-testid="stSidebar"] { background: linear-gradient(180deg,#171026,#0c0914); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: #f5f1ff; }

[data-testid="stMetric"] { background: linear-gradient(145deg,#251a39,#161020); border: 1px solid var(--border);
  padding: 16px 18px; border-radius: 16px; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .82rem; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: var(--display); }

div.stButton > button, div[data-testid="stDownloadButton"] > button, div[data-testid="stFormSubmitButton"] > button {
  width: 100%; background: linear-gradient(110deg,var(--purple2),#a16cff) !important; color: #fff !important;
  border: none !important; border-radius: 10px; font-weight: 700; padding: .55rem 1rem; }
div.stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover { box-shadow: 0 0 20px rgba(155,108,255,.45); }
div.stButton > button:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }

.stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div, [data-baseweb="input"] {
  background-color: #211832 !important; color: #fff !important; border: 1px solid #7d62a8 !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #b9a8d0 !important; opacity: 1 !important; }
[data-baseweb="popover"], [role="listbox"] { background-color: #211832 !important; }
[data-baseweb="popover"] *, [role="listbox"] *, [role="option"] { color: #fff !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }
.stTabs [data-baseweb="tab"] { color: #d9ccf0; }
.stTabs [aria-selected="true"] { color: #fff !important; }
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 12px; background: #140f1d; }

.hero { background: linear-gradient(115deg,#2c1a4b,#150f24 62%,#2a1642); border: 1px solid var(--border);
  border-radius: 24px; padding: 40px 38px; margin-bottom: 26px; box-shadow: 0 18px 60px rgba(0,0,0,.3); }
.hero-title { font-family: var(--display); font-size: clamp(32px,5vw,56px); font-weight: 800; line-height: 1.08;
  color: #fff; margin: 0 0 14px; }
.hero-sub { color: #e0d5ef; font-size: 1.02rem; line-height: 1.7; max-width: 720px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
.chip { background: rgba(155,108,255,.14); border: 1px solid var(--border); border-radius: 999px;
  padding: 7px 14px; font-size: .85rem; color: #efe6ff; }
.page-title { font-family: var(--display); font-size: 2.2rem; font-weight: 700; color: #fff; margin: 0 0 4px; }
.page-sub { color: var(--muted); margin-bottom: 22px; }

.card { background: linear-gradient(145deg,#211832,#141019); border: 1px solid var(--border);
  border-radius: 16px; padding: 18px 20px; margin-bottom: 14px; }
.card h3, .card h4 { margin-top: 0; }
.small { color: var(--muted) !important; font-size: .85rem; }

.mc-poster { aspect-ratio: 2/3; border-radius: 14px; background-size: cover; background-position: center;
  background-color: #1d1530; position: relative; border: 1px solid var(--border); overflow: hidden; }
.mc-noposter { display: flex; flex-direction: column; justify-content: flex-end; padding: 16px;
  background: linear-gradient(160deg,#3d2368 0%,#1a1228 72%); }
.mc-ptitle { font-family: var(--display); font-size: 1.25rem; font-weight: 700; line-height: 1.15; color: #fff; }
.mc-psub { color: #b9a8d0; font-size: .72rem; margin-top: 8px; }
.mc-blur { position: absolute; inset: -14px; background-size: cover; background-position: center; filter: blur(16px) brightness(.55); }
.mc-img { position: absolute; inset: 0; background-size: cover; background-repeat: no-repeat; background-position: center top; }
.badge-today { z-index: 2; position: absolute; top: 10px; left: 10px; background: linear-gradient(110deg,#ff4d8d,#9b6cff);
  color: #fff; font-weight: 800; font-size: .7rem; letter-spacing: .1em; padding: 5px 11px; border-radius: 999px; }
.mc-title { font-weight: 700; margin-top: 10px; color: #fff; line-height: 1.25; }
.mc-meta { color: var(--muted); font-size: .8rem; margin: 2px 0 8px; }

.pill { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .75rem; font-weight: 800; }
.pill-Positive { background: rgba(110,231,168,.16); color: #6ee7a8; border: 1px solid rgba(110,231,168,.4); }
.pill-Neutral  { background: rgba(184,169,217,.16); color: #cfc3ea; border: 1px solid rgba(184,169,217,.4); }
.pill-Negative { background: rgba(255,111,145,.16); color: #ff8fae; border: 1px solid rgba(255,111,145,.4); }
.cm { background: #17111f; border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; }
.cm-top { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 8px; }
.cm-text { color: #f1eaff; line-height: 1.55; word-break: break-word; }
.cm-src { color: var(--muted); font-size: .78rem; margin-top: 8px; }
.result { background: linear-gradient(135deg,#291c44,#171126); border: 1px solid rgba(181,145,255,.32);
  border-radius: 18px; padding: 24px 26px; margin: 16px 0; }
.result .label { font-family: var(--display); font-size: 2.6rem; font-weight: 800; margin: 4px 0; }
.label-Positive { color: #6ee7a8; } .label-Neutral { color: #cfc3ea; } .label-Negative { color: #ff8fae; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 3. SMALL HELPERS
# =========================================================
def safe(text) -> str:
    """Escape text for HTML. '$' is escaped too so Streamlit does not treat it as LaTeX."""
    return html.escape(str(text)).replace("$", "&#36;").replace("\n", "<br>")


def md_html(markup: str) -> None:
    """Render an HTML block (newlines/indentation collapsed so Markdown never sees code blocks)."""
    st.markdown(re.sub(r"\s*\n\s*", " ", markup.strip()), unsafe_allow_html=True)


def page_heading(title: str, subtitle: str) -> None:
    md_html(f'<div class="page-title">{safe(title)}</div><div class="page-sub">{safe(subtitle)}</div>')


def safe_url(url) -> str:
    url = str(url or "")
    return html.escape(url, quote=True) if url.startswith(("http://", "https://")) else ""


def find_column(df: pd.DataFrame, candidates: list[str]):
    """Return the first column whose normalised name matches a candidate (case/punctuation-insensitive)."""
    if df is None or df.empty:
        return None
    normalised = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in normalised:
            return normalised[key]
    return None


def movie_key(title) -> str:
    """Normalise a movie title for matching: drop '(...)' parts, punctuation and case."""
    text = re.sub(r"\([^)]*\)", " ", str(title).lower())
    return re.sub(r"[^a-z0-9]", "", text)


def text_key(text) -> str:
    """Normalisation used to remove duplicate comments."""
    return re.sub(r"[\W_]+", "", str(text).lower())


def today_ist() -> date:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Kolkata")).date()
    except Exception:
        return date.today()


def fmt_date(ts) -> str:
    return ts.strftime("%d %b %Y") if pd.notna(ts) else ""


def file_stamp(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def read_csv_any(path: Path) -> pd.DataFrame:
    """Read a CSV trying common encodings; return an empty DataFrame on any problem."""
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# =========================================================
# 4. MOVIE CATALOGUE (released_indian_movies.csv)
# =========================================================
def clean_language(value) -> str:
    """Turn "['Telugu']" or "Telugu, Hindi" into a tidy "Telugu, Hindi"."""
    text = re.sub(r"[\[\]'\"]", "", str(value))
    if text.strip().lower() in ("nan", "none"):
        return ""
    return ", ".join(p.strip() for p in text.split(",") if p.strip())


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date strings; pick day-first or month-first, whichever parses more rows."""
    s = series.astype(str).str.strip()
    a = pd.to_datetime(s, errors="coerce", dayfirst=False)
    b = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return b if b.notna().sum() >= a.notna().sum() else a


@st.cache_data(show_spinner=False)
def _build_catalogue(stamp: int):
    raw = read_csv_any(MOVIES_FILE)
    meta = {"rows": len(raw), "columns": list(raw.columns)}
    if raw.empty:
        return pd.DataFrame(), meta
    t_col = find_column(raw, ["title", "movie", "movie_name", "movie_title", "film", "name"])
    d_col = find_column(raw, ["release_date", "releasedate", "date", "release", "released"])
    l_col = find_column(raw, ["language", "languages", "original_language", "lang"])
    p_col = find_column(raw, ["poster_url", "poster", "image_url", "image", "thumbnail", "poster_path", "movie_poster"])
    u_col = find_column(raw, ["url", "link", "source_url", "detail_url", "movie_url", "page_url", "film_url", "details"])
    meta.update(title=t_col, date=d_col, language=l_col, poster=p_col, page_url=u_col)
    if not t_col:
        return pd.DataFrame(), meta

    cat = pd.DataFrame({"title": raw[t_col].fillna("").astype(str).str.strip()})
    cat["language"] = raw[l_col].map(clean_language) if l_col else ""
    cat["page_url"] = raw[u_col].fillna("").astype(str).str.strip() if u_col else ""
    cat["release_date"] = parse_dates(raw[d_col]) if d_col else pd.NaT
    cat["poster_csv"] = raw[p_col].fillna("").astype(str).str.strip() if p_col else ""
    cat = cat[(cat["title"] != "") & (cat["title"].str.lower() != "nan")]
    cat["language"] = cat["language"].replace({"nan": "", "NaN": ""})
    cat["key"] = cat["title"].map(movie_key)
    cat["year"] = cat["release_date"].dt.year
    cat = cat.drop_duplicates(subset=["key", "release_date"]).reset_index(drop=True)
    return cat, meta


# ---- live catalogue: films released now, from Wikipedia year/language categories ----
LIVE_CATALOGUE_FILE = DATA_DIR / "live_catalogue_v5.csv"
MY_MOVIES_FILE = DATA_DIR / "my_movies.csv"   # optional: films YOU add (title, language, release_date, poster)
POSTER_DIR = ROOT / "posters"                 # optional: poster image files for my_movies.csv
LIVE_TTL = 12 * 3600
MIN_COMMENTS = 15          # a released film appears on Home only if it has at least this many audience comments
MAX_NEW_CHECKS = 8        # films checked for comments per Home load (each costs ~100 YouTube quota units)
MIN_GRID = 6              # never leave the main grid with fewer films than this
PINNED_TITLES = ["Brindavihari", "Om", "OM - Operation Mole"]   # always shown in the main grid when they have a poster
UPCOMING_DAYS = 30        # See more: films opening within this many days
MORE_DAYS = 14            # See more: language rows look back this far
RECENT_DAYS = 21          # a film counts as "running now" for this many days after release
TOP_FILMS = 10            # size of the main Latest releases grid
MAIN_LANGUAGES = ["Kannada", "Hindi", "Telugu", "Tamil"]
HOME_LANGUAGE_ORDER = ["Kannada", "Hindi", "Telugu", "Tamil", "Malayalam", "Bengali", "Marathi", "Punjabi",
                       "Gujarati", "Odia", "Assamese", "Bhojpuri", "Tulu", "Konkani", "Haryanvi", "Rajasthani"]
INDIAN_LANGUAGES = ["Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Marathi", "Punjabi",
                    "Gujarati", "Odia", "Assamese", "Bhojpuri", "Tulu", "Konkani", "Haryanvi", "Rajasthani"]
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


SPECIAL_DATE_WORDS = ("premiere", "preview", "festival", "screening", "special", "iffi", "tiff", "paid")


def parse_release_date(wikitext: str):
    """Release date from a film article's infobox. Ignores premieres/previews and prefers the India release."""
    head = wikitext[:15000]
    found = []                                   # (date, is_india, is_special)
    for m in re.finditer(r"\{\{\s*(?:Film date|Start date|Release date|Start-date)\s*\|([^{}]*)\}\}", head, re.I):
        params = m.group(1)
        low = params.lower()
        d = re.search(r"(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})", params)
        if d:
            try:
                found.append((date(int(d.group(1)), int(d.group(2)), int(d.group(3))), "india" in low,
                              any(w in low for w in SPECIAL_DATE_WORDS)))
            except ValueError:
                pass
    if not found:
        m = re.search(rf"released\s*=[^\n]*?(\d{{1,2}})\s+({MONTHS})\s+(20\d\d)", head, re.I)
        m2 = re.search(rf"released\s*=[^\n]*?({MONTHS})\s+(\d{{1,2}}),?\s+(20\d\d)", head, re.I)
        try:
            if m:
                found.append((datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").date(), False, False))
            elif m2:
                found.append((datetime.strptime(f"{m2.group(2)} {m2.group(1)} {m2.group(3)}", "%d %B %Y").date(), False, False))
        except ValueError:
            pass
    if not found:
        return None
    pool = [f for f in found if not f[2]] or found
    india = [d for d, is_india, _ in pool if is_india]
    return min(india) if india else min(d for d, _, _ in pool)


def infobox_image(wikitext: str) -> str:
    """File name of the infobox poster, e.g. 'Ramayana Part 1.jpg' (used when the page-image lookup finds none)."""
    m = re.search(r"\|\s*image\s*=\s*([^\n]+)", wikitext[:15000], re.I)
    if not m:
        return ""
    f = re.search(r"([^\[\]|{}=\n]+?\.(?:jpe?g|png|webp|gif))", m.group(1), re.I)
    return re.sub(r"^(?:file|image):", "", f.group(1).strip(), flags=re.I).replace("_", " ").strip() if f else ""


def _image_urls(names: list[str]) -> dict:
    """{lower-case file name: 500px thumbnail URL} for up to 50 Wikipedia file names."""
    data = wiki_get({"action": "query", "format": "json", "formatversion": 2, "prop": "imageinfo",
                     "titles": "|".join("File:" + n for n in names), "iiprop": "url", "iiurlwidth": 800})
    out = {}
    for pg in data.get("query", {}).get("pages", []):
        info = pg.get("imageinfo")
        if info:
            out[re.sub(r"^file:", "", pg.get("title", ""), flags=re.I).lower()] = info[0].get("thumburl") or info[0].get("url", "")
    return out


def wiki_get(params: dict, timeout: int = 40, api: str = "") -> dict:
    """GET the Wikipedia API with retries for rate limits / temporary server errors."""
    last = ""
    for attempt in range(4):
        try:
            resp = requests.get(api or WIKI_API, params=params, headers=HTTP_HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}"
            time.sleep(1 + attempt)
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {resp.status_code}"
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise ValueError(data["error"].get("info", "API error"))
        return data
    raise RuntimeError(last or "request failed")


def _category_titles(language: str, category: str):
    """Cheap first pass: every article title in a category (no page content)."""
    params = {"action": "query", "format": "json", "formatversion": 2, "list": "categorymembers",
              "cmtitle": category, "cmnamespace": 0, "cmlimit": 500}
    titles: list[str] = []
    for _ in range(40):
        data = wiki_get(params, timeout=30)
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        if "continue" not in data:
            break
        params.update(data["continue"])
    return language, titles


def _fetch_pages(titles: list[str]) -> dict:
    """Second pass: poster thumbnail + opening wikitext (with the infobox) for up to 50 titles."""
    base = {"action": "query", "format": "json", "formatversion": 2, "titles": "|".join(titles),
            "prop": "pageimages|revisions", "piprop": "thumbnail", "pithumbsize": 800,
            "rvprop": "content", "rvslots": "main"}
    for with_section in (True, False):          # rvsection=0 keeps responses small; fall back if refused
        params = dict(base)
        if with_section:
            params["rvsection"] = 0
        pages: dict = {}
        try:
            for _ in range(60):
                data = wiki_get(params)
                for pg in data.get("query", {}).get("pages", []):
                    if pg.get("missing"):
                        continue
                    cur = pages.setdefault(pg.get("title"), {"title": pg.get("title", "")})
                    if pg.get("thumbnail", {}).get("source"):
                        cur["thumb"] = pg["thumbnail"]["source"]
                    revs = pg.get("revisions")
                    if revs:
                        cur["text"] = revs[0].get("slots", {}).get("main", {}).get("content", "")
                if "continue" not in data:
                    break
                params.update(data["continue"])
            return pages
        except Exception:  # noqa: BLE001
            if not with_section:
                raise
    return {}


WIKI_CODES = {"Hindi": "hi", "Telugu": "te", "Tamil": "ta", "Kannada": "kn", "Malayalam": "ml", "Bengali": "bn",
              "Marathi": "mr", "Punjabi": "pa", "Gujarati": "gu", "Odia": "or", "Assamese": "as", "Bhojpuri": "bh"}


def _interwiki_posters(pending: dict) -> dict:
    """For films with no poster on English Wikipedia: look for the film's article in its own language's
    Wikipedia (Kannada, Telugu, Hindi...) and use its lead image. pending = {English title: primary language}."""
    by_code: dict[str, list[str]] = {}
    for title, lang in pending.items():
        if lang in WIKI_CODES:
            by_code.setdefault(WIKI_CODES[lang], []).append(title)
    out: dict[str, str] = {}
    for code, titles in by_code.items():
        local: dict[str, str] = {}
        for i in range(0, len(titles), 50):
            try:
                data = wiki_get({"action": "query", "format": "json", "formatversion": 2, "prop": "langlinks",
                                 "lllang": code, "lllimit": "max", "titles": "|".join(titles[i:i + 50])})
            except Exception:  # noqa: BLE001
                continue
            for pg in data.get("query", {}).get("pages", []):
                if pg.get("langlinks"):
                    local[pg["langlinks"][0]["title"]] = pg["title"]
        names = list(local)
        for i in range(0, len(names), 50):
            try:
                data = wiki_get({"action": "query", "format": "json", "formatversion": 2, "prop": "pageimages",
                                 "piprop": "thumbnail", "pithumbsize": 800, "titles": "|".join(names[i:i + 50])},
                                api=f"https://{code}.wikipedia.org/w/api.php")
            except Exception:  # noqa: BLE001
                continue
            for pg in data.get("query", {}).get("pages", []):
                thumb = pg.get("thumbnail", {}).get("source")
                if thumb and pg.get("title") in local:
                    out[local[pg["title"]]] = thumb
    return out


def fetch_live_rows(years: list[int], today: date, known: dict) -> list[dict]:
    """Every film in the year / 'Upcoming' language categories, so nothing is cut off by an alphabetical limit.
    `known` = rows saved earlier; films released more than 60 days ago are reused instead of downloaded again."""
    categories = [(lang, f"Category:{y} {lang}-language films") for lang in INDIAN_LANGUAGES for y in years]
    categories += [(lang, f"Category:Upcoming {lang}-language films") for lang in INDIAN_LANGUAGES]
    langs_by_title: dict[str, list[str]] = {}
    ok, errors = 0, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for fut in [pool.submit(_category_titles, lang, cat) for lang, cat in categories]:
            try:
                lang, titles = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            ok += 1
            for t in titles:
                if lang not in langs_by_title.setdefault(t, []):
                    langs_by_title[t].append(lang)
    if ok == 0:
        raise RuntimeError(f"no language category could be downloaded ({errors[0] if errors else 'unknown reason'})")

    old_cutoff = (today - timedelta(days=60)).isoformat()
    rows, to_fetch = [], []
    for title, langs in langs_by_title.items():
        old = known.get(title)
        if old and str(old.get("release_date", "9999")) < old_cutoff:
            rows.append({**old, "languages": langs, "language": ", ".join(langs)})
        else:
            to_fetch.append(title)

    def run(batch):
        try:
            return _fetch_pages(batch)
        except Exception as exc:  # noqa: BLE001 - one failed batch must not lose the others
            errors.append(f"{type(exc).__name__}: {exc}")
            return {}

    batches = [to_fetch[i:i + 50] for i in range(0, len(to_fetch), 50)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for pages in pool.map(run, batches):
            for title, pg in pages.items():
                text = pg.get("text", "")
                if "infobox film" not in text[:15000].lower():
                    continue
                released = parse_release_date(text)
                if released is None:
                    continue
                langs = langs_by_title.get(title, [])
                rows.append({"title": re.sub(r"\s*\([^)]*film[^)]*\)\s*$", "", title, flags=re.I).strip(),
                             "wiki_title": title, "languages": langs, "language": ", ".join(langs),
                             "release_date": released.isoformat(), "poster_url": pg.get("thumb", ""),
                             "_img": infobox_image(text)})

    names = sorted({r["_img"] for r in rows if not r.get("poster_url") and r.get("_img")})
    if names:
        def img_batch(chunk):
            try:
                return _image_urls(chunk)
            except Exception:  # noqa: BLE001
                return {}
        found: dict = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            for part in pool.map(img_batch, [names[i:i + 50] for i in range(0, len(names), 50)]):
                found.update(part)
        for r in rows:
            if not r.get("poster_url") and r.get("_img"):
                r["poster_url"] = found.get(r["_img"].lower(), "")
    pending = {r["wiki_title"]: (r.get("languages") or [""])[0] for r in rows if not r.get("poster_url")}
    if pending:
        try:
            local_posters = _interwiki_posters(pending)
        except Exception:  # noqa: BLE001
            local_posters = {}
        for r in rows:
            if not r.get("poster_url"):
                r["poster_url"] = local_posters.get(r["wiki_title"], "")
    if not rows:
        raise RuntimeError(f"film details could not be read ({errors[0] if errors else 'no film had a readable release date'})")
    return rows


@st.cache_data(ttl=3600, show_spinner="Fetching the latest Indian releases in every language (the first load can take up to a minute)...")
def fetch_live_catalogue():
    today = today_ist()
    years = [today.year] + ([today.year - 1] if today.month <= 2 else [])
    saved = read_csv_any(LIVE_CATALOGUE_FILE)
    if not saved.empty and time.time() - LIVE_CATALOGUE_FILE.stat().st_mtime < LIVE_TTL:
        return saved, "Loaded from the copy saved in the last 12 hours (data/live_catalogue_v5.csv)."
    known = {}
    if not saved.empty and {"wiki_title", "release_date"} <= set(saved.columns):
        known = {str(r["wiki_title"]): r for r in saved.fillna("").to_dict("records")}
    try:
        rows = fetch_live_rows(years, today, known)
    except Exception as exc:  # noqa: BLE001
        if saved.empty:
            return saved, f"Could not reach Wikipedia ({exc})."
        return saved, f"Could not reach Wikipedia ({exc}); showing the last saved copy."
    df = pd.DataFrame(rows, columns=["title", "wiki_title", "language", "release_date", "poster_url"])
    try:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(LIVE_CATALOGUE_FILE, index=False)
    except OSError:
        pass
    return df, f"Fetched {len(df)} films from Wikipedia."


def cb_more_lang(lang):
    limits = st.session_state.setdefault("lang_limits", {})
    limits[lang] = limits.get(lang, 4) + 4


def cb_show_less():
    st.session_state.home_extra = 0


def cb_see_more():
    st.session_state.home_extra = st.session_state.get("home_extra", 0) + 1


def cb_more_languages():
    st.session_state.show_more_langs = True


def cb_more_home():
    st.session_state.home_limit = st.session_state.get("home_limit", 12) + 12


def cb_refresh_catalogue():
    fetch_live_catalogue.clear()
    try:
        LIVE_CATALOGUE_FILE.unlink()
    except OSError:
        pass


@st.cache_data(show_spinner=False)
def _load_my_movies(stamp: int):
    raw = read_csv_any(MY_MOVIES_FILE)
    if raw.empty:
        return pd.DataFrame()
    t = find_column(raw, ["title", "movie", "name"])
    d = find_column(raw, ["release_date", "date", "released"])
    if not t or not d:
        return pd.DataFrame()
    l = find_column(raw, ["language", "languages"])
    p = find_column(raw, ["poster", "poster_url", "image"])
    out = pd.DataFrame({"title": raw[t].fillna("").astype(str).str.strip(),
                        "language": raw[l].map(clean_language) if l else "",
                        "release_date": pd.to_datetime(raw[d], errors="coerce", dayfirst=True)})
    posters = []
    for v in (raw[p].fillna("").astype(str).str.strip() if p else [""] * len(raw)):
        if v.startswith(("http://", "https://")):
            posters.append(v)
            continue
        path = next((c for c in (POSTER_DIR / v, ROOT / v) if v and c.is_file()), None)
        if path and path.stat().st_size < 3_000_000:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            posters.append(f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode())
        else:
            posters.append("")
    out["poster_csv"], out["page_url"] = posters, ""
    return out[(out["title"] != "") & out["release_date"].notna()]


# ---- current release calendar (Hindi, Telugu, Tamil, Kannada, Malayalam): public monthly release pages ----
FILMIBEAT_FILE = DATA_DIR / "release_calendar.csv"
FB_PATHS = {"telugu": "Telugu", "tamil": "Tamil", "kannada": "Kannada", "malayalam": "Malayalam", "bollywood": "Hindi"}
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december"]
FB_ANCHOR = re.compile(r"""<a\s[^>]*?href=["'](?:https?://(?:www\.)?filmibeat\.com)?(/(telugu|tamil|kannada|malayalam|bollywood)/movies/([a-z0-9\-]+)\.html)["'][^>]*>""", re.I)
FB_DATE = re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(20\d\d)\b")
FB_MONTH_SLUG = re.compile(r"^(?:" + "|".join(MONTH_NAMES) + r")-20\d\d$")


def parse_release_page(page_html: str, year: int, month: int) -> list[dict]:
    """Films on one monthly release page: title, language (from the film's address) and release date."""
    anchors = list(FB_ANCHOR.finditer(page_html))
    found: dict[str, dict] = {}
    for i, m in enumerate(anchors):
        path, slug = m.group(2).lower(), m.group(3)
        if FB_MONTH_SLUG.match(slug):
            continue
        url = "https://www.filmibeat.com" + m.group(1)
        stop = anchors[i + 1].start() if i + 1 < len(anchors) else len(page_html)
        window = re.sub(r"<[^>]+>", " ", page_html[m.end(): min(stop, m.end() + 1500)])
        title_attr = re.search(r'title=["\']([^"\']+)["\']', m.group(0))
        entry = found.setdefault(url, {"title": html.unescape(title_attr.group(1)).strip() if title_attr
                                       else slug.replace("-", " ").title(),
                                       "language": FB_PATHS[path], "date": None, "page_url": url})
        d = FB_DATE.search(window)
        if d and entry["date"] is None:
            try:
                dt = datetime.strptime(f"{d.group(1)} {d.group(2)} {d.group(3)}", "%d %b %Y").date()
            except ValueError:
                continue
            if (dt.year, dt.month) == (year, month):      # ignores 'latest' / 'upcoming' side lists
                entry["date"] = dt
    return [{**e, "release_date": e["date"].isoformat()} for e in found.values() if e["date"]]


def fetch_release_rows(today: date):
    months, y, m = [], today.year, today.month
    for _ in range(3):                                      # this month and the two before it
        months.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    errors: list[str] = []

    def get(job):
        path, yy, mm = job
        url = f"https://www.filmibeat.com/{path}/movies/{MONTH_NAMES[mm - 1]}-{yy}.html"
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=25)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return parse_release_page(resp.text, yy, mm)
        except requests.RequestException as exc:
            errors.append(f"{type(exc).__name__} {getattr(exc.response, 'status_code', '')}".strip())
            return []

    with ThreadPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(get, [(p, yy, mm) for p in FB_PATHS for yy, mm in months]))
    unique = {r["page_url"]: r for part in parts for r in part}
    return list(unique.values()), errors


@st.cache_data(ttl=3600, show_spinner="Reading this month's release calendar...")
def fetch_release_calendar():
    saved = read_csv_any(FILMIBEAT_FILE)
    if not saved.empty and time.time() - FILMIBEAT_FILE.stat().st_mtime < LIVE_TTL:
        return saved, "Loaded the saved calendar (updated in the last 12 hours)."
    rows, errors = fetch_release_rows(today_ist())
    if not rows:
        reason = errors[0] if errors else "no film with a readable date was found on the pages"
        return saved, f"Release calendar could not be loaded ({reason})."
    df = pd.DataFrame(rows, columns=["title", "language", "release_date", "page_url"])
    try:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(FILMIBEAT_FILE, index=False)
    except OSError:
        pass
    return df, f"Loaded {len(df)} releases from the release calendar."


def load_catalogue():
    """Catalogue = your optional my_movies.csv + live Wikipedia releases (old CSV only as an offline fallback)."""
    live, status = fetch_live_catalogue()
    calendar, cal_status = fetch_release_calendar()
    meta = {"live_status": status, "calendar_status": cal_status}
    frames = []
    mine = _load_my_movies(file_stamp(MY_MOVIES_FILE))
    if not mine.empty:
        frames.append(mine)
        meta["my_movies_rows"] = len(mine)
    if not calendar.empty:
        frames.append(pd.DataFrame({
            "title": calendar["title"].astype(str), "language": calendar["language"].fillna("").astype(str),
            "release_date": pd.to_datetime(calendar["release_date"], errors="coerce"),
            "poster_csv": "", "page_url": calendar["page_url"].fillna("").astype(str)}))
        meta["calendar_rows"] = len(calendar)
    if not live.empty:
        frames.append(pd.DataFrame({
            "title": live["title"].astype(str), "language": live["language"].fillna("").astype(str),
            "release_date": pd.to_datetime(live["release_date"], errors="coerce"),
            "poster_csv": live["poster_url"].fillna("").astype(str), "page_url": ""}))
        meta["live_rows"] = len(live)
    else:
        fetch_live_catalogue.clear()   # do not remember a failure; retry on the next run
        legacy, _ = _build_catalogue(file_stamp(MOVIES_FILE))
        if not legacy.empty:
            frames.append(legacy[["title", "language", "release_date", "poster_csv", "page_url"]])
            meta["fallback"] = "released_indian_movies.csv (offline fallback)"
    if not frames:
        return pd.DataFrame(), meta
    cat = pd.concat(frames, ignore_index=True)
    cat["key"] = cat["title"].map(movie_key)
    cat["year"] = cat["release_date"].dt.year
    first = lambda s: next((x for x in s if isinstance(x, str) and x), "")      # keep any poster / page link found
    for col in ("poster_csv", "page_url"):
        cat[col] = cat.groupby(["key", "year"], dropna=False)[col].transform(first)
    cat = cat[cat["key"] != ""].drop_duplicates(["key", "year"]).reset_index(drop=True)
    return cat, meta


def movie_info(row: dict) -> dict:
    year = row.get("year")
    return {
        "title": row["title"],
        "language": row.get("language", ""),
        "date": row["release_date"] if pd.notna(row.get("release_date")) else None,
        "year": int(year) if pd.notna(year) else None,
        "csv_poster": row.get("poster_csv", ""),
        "page_url": row.get("page_url", ""),
    }


# =========================================================
# 5. POSTERS (Wikipedia / Wikimedia API, cached to CSV)
# =========================================================
POSTER_COLS = ["key", "title", "year", "poster_url", "wiki_title", "status", "checked_ts", "source", "trailer_tried"]
NOT_FOUND_TTL = 24 * 3600   # retry "not found" after a week
ERROR_TTL = 300                # retry network errors after an hour


def poster_key(title, year) -> str:
    return f"{movie_key(title)}|{year or ''}"


def _wiki_search(query: str) -> list[dict]:
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
        "gsrlimit": 6, "gsrnamespace": 0, "prop": "pageimages|extracts", "piprop": "thumbnail",
        "pithumbsize": 800, "exintro": 1, "explaintext": 1, "exsentences": 3, "exlimit": "max",
    }
    data = wiki_get(params, timeout=20)
    pages = data.get("query", {}).get("pages", {})
    return sorted(pages.values(), key=lambda p: p.get("index", 99))


def _valid_film_page(page: dict, title: str, year) -> bool:
    """Accept a Wikipedia page only if it is clearly THIS film (accuracy over coverage)."""
    if movie_key(page.get("title", "")) != movie_key(title):
        return False
    extract = page.get("extract", "").lower()
    if "film" not in extract and "movie" not in extract:
        return False
    if not page.get("thumbnail", {}).get("source"):
        return False
    if year:
        blob = page.get("title", "") + " " + extract
        if not any(str(y) in blob for y in (year - 1, year, year + 1)):
            return False
    return True


def wiki_poster_lookup(title: str, year):
    """Return (poster_url, wiki_title, status). status: found | not_found | error."""
    queries = [f"{title} {year or ''} Indian film".replace("  ", " "), f"{title} (film)"]
    try:
        for query in queries:
            for page in _wiki_search(query):
                if _valid_film_page(page, title, year):
                    return page["thumbnail"]["source"], page.get("title", ""), "found"
        return "", "", "not_found"
    except requests.RequestException:
        return "", "", "error"


OG_PATTERNS = [
    re.compile(r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\'][^>]*content=["\']([^"\']+)["\']', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\']', re.I),
]
BAD_IMAGE_WORDS = ("logo", "default", "placeholder", "favicon", "noimage", "no-image", "blank", "avatar", "icon")


def page_image_candidates(page_url: str) -> list[str]:
    """Poster candidates on a film's own page: social-share image first, then images whose address says 'poster'."""
    if not str(page_url).startswith(("http://", "https://")):
        return []
    try:
        resp = requests.get(page_url, headers=HTTP_HEADERS, timeout=12)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    head = resp.text[:400000]
    urls: list[str] = []
    for pattern in OG_PATTERNS:
        for match in pattern.finditer(head):
            urls.append(match.group(1))
    for match in re.finditer(r"""(?:src|data-src|data-original)=["']([^"']+\.(?:jpe?g|png|webp)[^"']*)["']""", head, re.I):
        if re.search(r"poster|movie_posters|/movies?/", match.group(1), re.I):
            urls.append(match.group(1))
    out: list[str] = []
    for u in urls:
        full = urljoin(page_url, html.unescape(u))
        if full not in out and not any(w in full.lower() for w in BAD_IMAGE_WORDS):
            out.append(full)
    return out[:8]


def page_image_lookup(page_url: str) -> str:
    out = page_image_candidates(page_url)
    return out[0] if out else ""


HIDDEN_FILE = DATA_DIR / "hidden_movies.csv"      # films you never want on the Home page (one title per line)
APP_VERSION = "2026-11-22-e"


def load_hidden() -> set:
    if not HIDDEN_FILE.exists():                   # first run: start with the two films without a proper poster
        try:
            DATA_DIR.mkdir(exist_ok=True)
            pd.DataFrame({"title": ["Mathi", "Chaplin Samandhi"]}).to_csv(HIDDEN_FILE, index=False)
        except OSError:
            return {"mathi", "chaplinsamandhi"}
    df = read_csv_any(HIDDEN_FILE)
    return {movie_key(t) for t in df["title"].astype(str)} if not df.empty and "title" in df.columns else set()


def add_hidden(title: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    keep = sorted(read_csv_any(HIDDEN_FILE)["title"].astype(str).tolist() if HIDDEN_FILE.exists() and not read_csv_any(HIDDEN_FILE).empty else [])
    pd.DataFrame({"title": sorted(set(keep) | {title})}).to_csv(HIDDEN_FILE, index=False)


REJECT_FILE = DATA_DIR / "poster_rejects.csv"


def load_rejects() -> set:
    df = read_csv_any(REJECT_FILE)
    return set(df["url"].astype(str)) if not df.empty and "url" in df.columns else set()


def add_reject(url: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"url": sorted(load_rejects() | {url})}).to_csv(REJECT_FILE, index=False)


try:
    from PIL import ImageFile
except Exception:  # noqa: BLE001
    ImageFile = None


def portrait_ok(url: str) -> bool:
    """True only for sharp, portrait, poster-shaped images (at least 300 px wide, height 1.25 to 1.85 times the width).
    Wide banners, trailer frames, tiny/blurry images and posters you rejected are not used."""
    if str(url).startswith("data:image/"):
        return True
    if url in load_rejects():
        return False
    if ImageFile is None:
        return True
    try:
        with requests.get(url, headers=HTTP_HEADERS, timeout=12, stream=True) as resp:
            resp.raise_for_status()
            parser, got = ImageFile.Parser(), 0
            for chunk in resp.iter_content(8192):
                parser.feed(chunk)
                got += len(chunk)
                if parser.image is not None:
                    w, h = parser.image.size
                    return w >= 300 and 1.25 <= h / w <= 1.85
                if got > 600_000:
                    break
    except Exception:  # noqa: BLE001 - could not measure: do not punish the film
        return True
    return True


def wiki_poster_candidates(title: str, year):
    """All Wikipedia articles that are clearly this film and have an image. Returns (list of (url, title), status)."""
    queries = [f"{title} {year or ''} Indian film".replace("  ", " "), f"{title} (film)"]
    out = []
    try:
        for query in queries:
            for page in _wiki_search(query):
                if _valid_film_page(page, title, year):
                    out.append((page["thumbnail"]["source"], page.get("title", "")))
        return out, ("found" if out else "not_found")
    except (requests.RequestException, RuntimeError, ValueError):
        return out, "error"


def lookup_poster(title: str, year, page_url: str, preset: str = ""):
    """First portrait image from: the catalogue's Wikipedia image, a Wikipedia search, the film's own page.
    Returns (url, label, status, source)."""
    if preset.startswith(("http://", "https://")) and portrait_ok(preset):
        return preset, "", "found", "wikipedia"
    candidates, status = wiki_poster_candidates(title, year)
    for url, label in candidates:
        if url != preset and portrait_ok(url):
            return url, label, "found", "wikipedia"
    for img in page_image_candidates(page_url):
        if portrait_ok(img):
            return img, "", "found", "catalogue page"
    return "", "", status, ""


def _load_poster_cache() -> dict:
    df = read_csv_any(POSTER_CACHE_FILE)
    if df.empty or not set(POSTER_COLS) <= set(df.columns):
        return {}
    df = df.fillna("")
    df = df[df["source"].astype(str) != "official trailer"]      # trailer stills are no longer used as posters
    return {str(r["key"]): r for r in df.to_dict("records")}


def get_posters(items: list[tuple]) -> dict:
    """items: [(title, year, csv_poster_url, page_url)] -> {poster_key: url or ''}. Uses a CSV cache."""
    cache = _load_poster_cache()
    now = time.time()
    result, todo = {}, []
    for title, year, csv_url, page_url in items:
        key = poster_key(title, year)
        if key in result:
            continue
        if isinstance(csv_url, str) and csv_url.startswith("data:image/"):      # poster you uploaded yourself
            result[key] = csv_url
            continue
        preset = csv_url if isinstance(csv_url, str) and csv_url.startswith(("http://", "https://")) else ""
        row = cache.get(key)
        if row:
            status = row.get("status")
            age = now - float(row.get("checked_ts") or 0)
            if status == "found" and row.get("poster_url") and row["poster_url"] not in load_rejects():
                result[key] = row["poster_url"]
                continue
            if (status == "not_found" and age < NOT_FOUND_TTL) or (status == "error" and age < ERROR_TTL):
                result[key] = ""
                continue
        todo.append((title, year, key, page_url, preset))

    if todo:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(lambda it: lookup_poster(it[0], it[1], it[3], it[4]), todo))
        for (title, year, key, _, _), (url, label, status, source) in zip(todo, outcomes):
            cache[key] = {"key": key, "title": title, "year": year or "", "poster_url": url, "wiki_title": label,
                          "status": status, "checked_ts": now, "source": source, "trailer_tried": ""}
        # An image shared by several different films is a site default/logo, not a poster.
        shared = Counter(r["poster_url"] for r in cache.values() if r.get("poster_url") and r.get("source") == "catalogue page")
        for r in cache.values():
            if r.get("source") == "catalogue page" and shared[r["poster_url"]] > 1:
                r.update(poster_url="", status="not_found", source="")
        for _, _, key, _, _ in todo:
            result[key] = cache[key].get("poster_url", "")
        try:
            DATA_DIR.mkdir(exist_ok=True)
            pd.DataFrame(list(cache.values()), columns=POSTER_COLS).to_csv(POSTER_CACHE_FILE, index=False)
        except OSError:
            pass
    return result


def poster_block(title: str, url: str, badge: bool = False) -> str:
    label = badge if isinstance(badge, str) else ("OUT TODAY" if badge else "")
    badge_html = f'<span class="badge-today">{safe(label)}</span>' if label else ""
    safe_link = url if str(url).startswith("data:image/") else safe_url(url).replace("'", "%27")
    if safe_link:
        return (f'<div class="mc-poster" role="img" aria-label="Poster of {safe(title)}">'
                f"<div class=\"mc-img\" style=\"background-image:url('{safe_link}')\"></div>{badge_html}</div>")
    return (f'<div class="mc-poster mc-noposter">{badge_html}<div class="mc-ptitle">{safe(title)}</div>'
            f'<div class="mc-psub">Poster not available</div></div>')


# =========================================================
# 6. SENTIMENT ENGINE
# =========================================================
class ModelError(RuntimeError):
    pass


@st.cache_resource(show_spinner="Loading sentiment models (the first run downloads them)...")
def load_models():
    from transformers import pipeline

    english = pipeline("text-classification", model=ENGLISH_MODEL, device=-1)
    multi = pipeline("text-classification", model=MULTI_MODEL, device=-1)
    return english, multi


def ensure_models():
    try:
        return load_models(), None
    except Exception as exc:  # noqa: BLE001 - show any loading problem to the user
        return None, f"{type(exc).__name__}: {exc}"


# ---- preprocessing ----------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
TAG_RE = re.compile(r"<[^>]+>")


def preprocess(text) -> str:
    """Light clean-up in the style the Twitter-RoBERTa model was trained on. Emojis are KEPT."""
    t = html.unescape(str(text if text is not None else ""))
    t = TAG_RE.sub(" ", t)
    t = URL_RE.sub("http", t)
    t = MENTION_RE.sub("@user", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:1000]


# ---- script detection and routing ------------------------------------------
INDIC_SCRIPTS = [
    ("Devanagari (Hindi / Marathi)", 0x0900, 0x097F), ("Bengali / Assamese", 0x0980, 0x09FF),
    ("Gurmukhi (Punjabi)", 0x0A00, 0x0A7F), ("Gujarati", 0x0A80, 0x0AFF), ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF), ("Telugu", 0x0C00, 0x0C7F), ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
]


def detect_script(text: str) -> dict:
    """Identify the dominant writing system by counting Unicode blocks (script, not language)."""
    counts: Counter = Counter()
    for ch in emoji.replace_emoji(text, ""):
        code = ord(ch)
        for name, lo, hi in INDIC_SCRIPTS:
            if lo <= code <= hi:
                counts[name] += 1
                break
        else:
            if ch.isalpha():
                counts["Latin" if code < 0x250 else "Other non-Latin script"] += 1
    total = sum(counts.values())
    if total == 0:
        return {"script": "No letters", "route": "english"}
    top, top_n = counts.most_common(1)[0]
    non_latin = total - counts.get("Latin", 0)
    if top != "Latin" and top_n / total >= 0.3:
        return {"script": top, "route": "multi"}
    if non_latin / total >= 0.3:
        return {"script": "Mixed / other non-Latin script", "route": "multi"}
    return {"script": "Latin (English or romanised text)", "route": "english"}


# ---- emoji signal ------------------------------------------------------------
POSITIVE_EMOJI = set("😀😃😄😁😆😊😍🥰😘🤩🥳❤💕💖💗💙💜💛💚🧡🔥👏👍✨💯🙌😻🤗😎🎉🥹😇")
NEGATIVE_EMOJI = set("😢😭😞😔☹🙁😡🤬👎💔🤮😠😣😩😫🥱😤😖😒🙄😿🤢")


def emoji_polarity(text: str):
    """Return ('Positive'|'Negative'|'Mixed'|None, n_positive, n_negative)."""
    pos = neg = 0
    for item in emoji.emoji_list(text):
        first = item["emoji"].replace("\ufe0f", "")[:1]
        if first in POSITIVE_EMOJI:
            pos += 1
        elif first in NEGATIVE_EMOJI:
            neg += 1
    if pos and neg:
        return "Mixed", pos, neg
    if pos:
        return "Positive", pos, neg
    if neg:
        return "Negative", pos, neg
    return None, pos, neg


def emoji_breakdown(text: str) -> list[dict]:
    rows = []
    for item in emoji.emoji_list(text):
        first = item["emoji"].replace("\ufe0f", "")[:1]
        signal = "Positive" if first in POSITIVE_EMOJI else "Negative" if first in NEGATIVE_EMOJI else "Not scored"
        rows.append({"Emoji": item["emoji"], "Signal": signal})
    return rows


# ---- contrast clauses (mixed opinions) ----------------------------------------
CONTRAST_RE = re.compile(r"\b(?:but|however|although|though|whereas|nevertheless|except)\b|;", re.I)


def split_contrast(text: str) -> list[str]:
    parts = [p.strip(" ,.-\u2013\u2014!?\"'") for p in CONTRAST_RE.split(text)]
    parts = [p for p in parts if len(re.findall(r"\w+", p)) >= 2]
    return parts[:4] if len(parts) >= 2 else []


def norm_label(label) -> str:
    low = str(label).lower()
    if "pos" in low:
        return "Positive"
    if "neg" in low:
        return "Negative"
    return "Neutral"


ROUTE_NAMES = {"english": "English model (RoBERTa)", "multi": "Multilingual model (DistilBERT)"}


def analyse_core(texts: list[str], models) -> list[dict]:
    """Run the full pipeline on a list of texts (no caching). Returns one dict per text."""
    english, multi = models
    n = len(texts)
    prepared = [preprocess(t) for t in texts]
    infos = [detect_script(p) for p in prepared]
    emos = [emoji_polarity(p)[0] for p in prepared]
    has_words = [any(c.isalnum() for c in emoji.replace_emoji(p, "")) for p in prepared]

    jobs = {"english": [], "multi": []}   # (text index, clause index or -1, text)
    clauses: dict[int, list[str]] = {}
    for i, text in enumerate(prepared):
        if not text:
            continue
        route = infos[i]["route"]
        jobs[route].append((i, -1, text))
        if route == "english" and has_words[i]:
            clauses[i] = split_contrast(text)
            for j, clause in enumerate(clauses[i]):
                jobs["english"].append((i, j, clause))

    predictions: dict[tuple, tuple] = {}
    for route, pipe in (("english", english), ("multi", multi)):
        if not jobs[route]:
            continue
        outputs = pipe([j[2] for j in jobs[route]], truncation=True, max_length=256, batch_size=16)
        for (i, j, _), out in zip(jobs[route], outputs):
            predictions[(i, j)] = (norm_label(out["label"]), float(out["score"]))

    results = []
    for i in range(n):
        if not prepared[i]:
            results.append({"sentiment": "Neutral", "score": 0.0, "model_label": "Neutral", "model_route": "None",
                            "script": "No text", "emoji_signal": "", "reason": "The text is empty.", "parts": []})
            continue
        label, score = predictions[(i, -1)]
        model_label = label
        reasons, parts = [], []
        sig = emos[i]

        for j, clause in enumerate(clauses.get(i, [])):
            c_label, c_score = predictions[(i, j)]
            parts.append({"Part of review": clause, "Sentiment": c_label, "Score %": round(c_score * 100, 1)})
        strong = {p["Sentiment"] for p in parts if p["Sentiment"] != "Neutral" and p["Score %"] >= 60}
        if {"Positive", "Negative"} <= strong:
            pos_part = next(p["Part of review"] for p in parts if p["Sentiment"] == "Positive" and p["Score %"] >= 60)
            neg_part = next(p["Part of review"] for p in parts if p["Sentiment"] == "Negative" and p["Score %"] >= 60)
            label = "Neutral"
            reasons.append(f'Mixed opinion: "{pos_part}" reads positive but "{neg_part}" reads negative, '
                           "so the overall review is treated as Neutral.")

        if not has_words[i]:
            if sig in ("Positive", "Negative"):
                if label in ("Neutral", sig):
                    label = sig
                    reasons.append(f"Emoji-only review: the emojis express a {sig.lower()} reaction.")
                else:
                    label = "Neutral"
                    reasons.append(f"Emoji-only review: the model said {model_label} but the emojis say {sig}; "
                                   "the conflicting signals give Neutral.")
            elif sig == "Mixed":
                label = "Neutral"
                reasons.append("Emoji-only review with both positive and negative emojis.")
            else:
                reasons.append("Emoji-only review: judged by the model alone.")
        else:
            if sig == "Mixed" and label != "Neutral":
                label = "Neutral"
                reasons.append("The emojis contain both positive and negative signals, so the result is Neutral.")
            elif sig in ("Positive", "Negative") and label in ("Positive", "Negative") and label != sig:
                reasons.append(f"The words read {label.lower()} but the emoji read {sig.lower()}; "
                               "conflicting signals are classified as Neutral.")
                label = "Neutral"

        results.append({
            "emojis": emoji_breakdown(prepared[i]),
            "sentiment": label, "score": round(score * 100, 1), "model_label": model_label,
            "model_route": ROUTE_NAMES[infos[i]["route"]], "script": infos[i]["script"],
            "emoji_signal": sig or "", "reason": " ".join(reasons), "parts": parts,
        })
    return results


# ---- disk-cached analysis (so the models run once per unique comment) -------------
CACHE_COLS = ["key", "sentiment", "score", "model_route", "script", "emoji_signal", "reason"]
RESULT_COLS = CACHE_COLS[1:]
BATCH = 48


def _analysis_key(text) -> str:
    return hashlib.md5(f"{ANALYSIS_VERSION}|{text}".encode("utf-8")).hexdigest()[:20]


@st.cache_data(show_spinner=False)
def _read_analysis_cache(stamp: int) -> dict:
    if not ANALYSIS_CACHE_FILE.exists():
        return {}
    try:
        df = pd.read_csv(ANALYSIS_CACHE_FILE, dtype=str, keep_default_na=False)
    except Exception:
        return {}
    if not set(CACHE_COLS) <= set(df.columns):
        return {}
    out = {}
    for row in df.to_dict("records"):
        try:
            row["score"] = float(row["score"])
        except ValueError:
            row["score"] = 0.0
        out[row["key"]] = row
    return out


def pending_count(texts) -> int:
    cache = _read_analysis_cache(file_stamp(ANALYSIS_CACHE_FILE))
    return len({_analysis_key(t) for t in texts} - cache.keys())


def analyse_cached(texts, label: str = "Analysing comments") -> list[dict]:
    """Analyse texts, running the models only for texts not analysed before."""
    texts = ["" if t is None else str(t) for t in texts]
    keys = [_analysis_key(t) for t in texts]
    cache = dict(_read_analysis_cache(file_stamp(ANALYSIS_CACHE_FILE)))
    missing = {}
    for k, t in zip(keys, texts):
        if k not in cache and k not in missing:
            missing[k] = t
    if missing:
        models, err = ensure_models()
        if models is None:
            raise ModelError(err)
        items = list(missing.items())
        bar = st.progress(0.0, text=f"{label} (0/{len(items)})")
        for start in range(0, len(items), BATCH):
            chunk = items[start:start + BATCH]
            outputs = analyse_core([t for _, t in chunk], models)
            rows = []
            for (k, _), out in zip(chunk, outputs):
                row = {"key": k, **{c: out[c] for c in RESULT_COLS}}
                cache[k] = row
                rows.append(row)
            try:
                DATA_DIR.mkdir(exist_ok=True)
                pd.DataFrame(rows, columns=CACHE_COLS).to_csv(
                    ANALYSIS_CACHE_FILE, mode="a", header=not ANALYSIS_CACHE_FILE.exists(), index=False)
            except OSError:
                pass
            done = min(start + BATCH, len(items))
            bar.progress(done / len(items), text=f"{label} ({done}/{len(items)})")
        bar.empty()
    return [{c: cache[k][c] for c in RESULT_COLS} for k in keys]


def attach_results(df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    return df.reset_index(drop=True).join(pd.DataFrame(results, columns=RESULT_COLS))


def count_labels(series) -> dict:
    counts = {k: 0 for k in ORDER}
    for label, n in pd.Series(list(series)).value_counts().items():
        counts[label if label in counts else "Neutral"] += int(n)
    return counts


def audience_summary(counts: dict, title: str) -> str:
    """Plain-language summary computed from the real counts (nothing hardcoded)."""
    total = sum(counts.values())
    if total == 0:
        return "No comments to summarise."
    pct = {k: counts[k] / total * 100 for k in ORDER}
    lead = max(ORDER, key=lambda k: pct[k])
    if pct["Positive"] >= 55:
        tone = "largely positive"
    elif pct["Negative"] >= 55:
        tone = "largely negative"
    elif abs(pct["Positive"] - pct["Negative"]) <= 10:
        tone = "divided"
    else:
        tone = f"leaning {'positive' if pct['Positive'] > pct['Negative'] else 'negative'}"
    note = " The sample is small, so treat this cautiously." if total < 30 else ""
    return (f"Across {total:,} analysed comments, audience reaction to {title} is {tone}: "
            f"{pct['Positive']:.0f}% positive, {pct['Neutral']:.0f}% neutral and {pct['Negative']:.0f}% negative "
            f"({lead.lower()} is the largest group).{note}")


# =========================================================
# 7. CHARTS (Matplotlib, all three classes always shown)
# =========================================================
BG = "#16101f"


def _style_axes(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#e9e0f8")
    for spine in ax.spines.values():
        spine.set_color("#3a2d52")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("#e9e0f8")
    ax.yaxis.label.set_color("#e9e0f8")


def chart_donut(counts: dict, title: str, size: float = 5):
    values = [counts[k] for k in ORDER]
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    fig.patch.set_facecolor(BG)
    ax.pie(values, colors=[COLORS[k] for k in ORDER], startangle=90, counterclock=False,
           autopct=lambda p: f"{p:.0f}%" if p >= 4 else "", pctdistance=0.79,
           wedgeprops={"width": 0.42, "edgecolor": BG, "linewidth": 2},
           textprops={"color": "#120d1c", "fontweight": "bold"})
    ax.text(0, 0.06, f"{sum(values):,}", ha="center", va="center", color="white", fontsize=22, fontweight="bold")
    ax.text(0, -0.2, "comments", ha="center", va="center", color="#bfb0d8", fontsize=10)
    ax.set_title(title, color="white", fontsize=11)
    ax.legend(ORDER, loc="lower center", ncol=3, frameon=False, labelcolor="#e9e0f8", bbox_to_anchor=(0.5, -0.08))
    st.pyplot(fig)
    plt.close(fig)


def chart_bars(counts: dict, title: str = "Comments per sentiment class"):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(ORDER, [counts[k] for k in ORDER], color=[COLORS[k] for k in ORDER])
    ax.bar_label(bars, color="white", padding=3)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Comments")
    ax.grid(axis="y", alpha=0.15)
    _style_axes(fig, ax)
    st.pyplot(fig)
    plt.close(fig)


def chart_movie_stack(table: pd.DataFrame):
    d = table.sort_values("Total comments").tail(15)
    names = [n if len(n) <= 28 else n[:26] + "…" for n in d["Movie"]]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(d) + 1.2)))
    left = np.zeros(len(d))
    for lab in ORDER:
        pct = d[f"{lab} %"].to_numpy(dtype=float)
        ax.barh(names, pct, left=left, color=COLORS[lab], label=lab)
        left += pct
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of comments (%)")
    ax.set_title("Sentiment mix by movie", fontsize=11)
    ax.legend(ncol=3, frameon=False, labelcolor="#e9e0f8", loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style_axes(fig, ax)
    st.pyplot(fig)
    plt.close(fig)


def show_metrics(counts: dict, total_label: str = "Comments analysed"):
    total = sum(counts.values())
    c = st.columns(4)
    c[0].metric(total_label, f"{total:,}")
    for col, lab in zip(c[1:], ["Positive", "Negative", "Neutral"]):
        col.metric(lab, f"{counts[lab] / total * 100:.1f}%" if total else "0.0%", f"{counts[lab]:,} comments", delta_color="off")


def show_charts(counts: dict, title: str):
    if sum(counts.values()) == 0:
        st.info("There are no analysed comments to chart yet.")
        return
    left, right = st.columns(2)
    with left:
        chart_donut(counts, title)
    with right:
        chart_bars(counts)


def comment_card(item: dict) -> str:
    """HTML for one analysed comment. `item` is a dict of row values."""
    date_txt = ""
    if item.get("published_at"):
        ts = pd.to_datetime(item["published_at"], errors="coerce", utc=True)
        date_txt = ts.strftime("%d %b %Y") if pd.notna(ts) else ""
    meta = " · ".join(x for x in [date_txt, f"{int(item['likes'])} likes" if str(item.get("likes", "")).strip() not in ("", "nan") and str(item.get("likes")).replace(".0", "").isdigit() else ""] if x)
    link = safe_url(item.get("comment_url") or item.get("video_url"))
    src = ""
    if item.get("video_title"):
        title_html = safe(item["video_title"])
        src = f'Source: <a href="{link}" target="_blank" rel="noopener">{title_html}</a>' if link else f"Source: {title_html}"
        if item.get("channel"):
            src += f" · {safe(item['channel'])}"
    elif link:
        src = f'<a href="{link}" target="_blank" rel="noopener">Source link</a>'
    reason = f'<div class="cm-src">Why: {safe(item["reason"])}</div>' if item.get("reason") else ""
    return (f'<div class="cm"><div class="cm-top"><span class="pill pill-{item["sentiment"]}">{item["sentiment"]}</span>'
            f'<span class="small">Score {float(item["score"]):.0f}% · {safe(item["model_route"])}</span>'
            f'<span class="small">{safe(meta)}</span></div><div class="cm-text">{safe(item["comment"])}</div>'
            f'{reason}<div class="cm-src">{src}</div></div>')


def show_comment_list(df: pd.DataFrame, key: str):
    """Sentiment filter + limited list of comment cards."""
    c1, c2 = st.columns([2, 1])
    chosen = c1.multiselect("Filter by sentiment", ORDER, default=ORDER, key=f"{key}_sent")
    limit = c2.selectbox("Comments to show", [10, 25, 50, 100], key=f"{key}_limit")
    view = df[df["sentiment"].isin(chosen)]
    st.caption(f"Showing {min(limit, len(view))} of {len(view)} matching comments.")
    for item in view.head(limit).to_dict("records"):
        md_html(comment_card(item))


# =========================================================
# 8. COMMENT CORPUS (existing CSVs + comments saved from YouTube page)
# =========================================================
COMMENT_COLS = ["movie", "comment", "source", "video_title", "channel", "video_url", "comment_url",
                "published_at", "likes", "author"]


@st.cache_data(show_spinner=False)
def _build_corpus(stamps: tuple):
    """Merge every available comment file into one de-duplicated table. Nothing is invented."""
    spec = [(COLLECTED_FILE, False), (VERIFIED_FILE, False), (SENTIMENT_FILE, False), (CLEANED_FILE, True)]
    frames, notes = [], []
    for path, is_fallback in spec:
        if not path.exists():
            notes.append(f"{path.name}: file not found (skipped).")
            continue
        raw = read_csv_any(path)
        if raw.empty:
            notes.append(f"{path.name}: empty or unreadable (skipped).")
            continue
        if is_fallback and frames:
            notes.append(f"{path.name}: skipped - it is a pre-processed copy of comments already loaded "
                         "from other files (avoids double counting).")
            continue
        m_col = find_column(raw, ["movie", "movie_name", "movie_title", "film", "film_name", "searched_movie", "query", "title"])
        c_col = find_column(raw, ["comment", "comment_text", "textdisplay", "original_comment", "raw_comment",
                                  "review", "text", "content", "cleaned_comment", "clean_comment", "clean_text"])
        if not m_col or not c_col:
            notes.append(f"{path.name}: skipped - needs a movie column and a comment-text column "
                         f"(found columns: {', '.join(map(str, raw.columns))}).")
            continue

        def opt(cands):
            col = find_column(raw, cands)
            return raw[col].fillna("").astype(str) if col else ""

        frame = pd.DataFrame({
            "movie": raw[m_col].fillna("").astype(str).str.strip(),
            "comment": raw[c_col].fillna("").astype(str).str.strip(),
            "source": path.name,
            "video_title": opt(["video_title", "videotitle", "video"]),
            "channel": opt(["channel", "channel_title", "channelname", "channel_name"]),
            "video_url": opt(["video_url", "url", "link", "source_url"]),
            "comment_url": opt(["comment_url"]),
            "published_at": opt(["published_at", "publishedat", "comment_date", "date", "timestamp"]),
            "likes": opt(["likes", "like_count", "likecount"]),
            "author": opt(["author", "author_name", "authordisplayname"]),
        })
        frame = frame[(frame["movie"] != "") & (frame["comment"] != "")
                      & (~frame["movie"].str.lower().isin(["nan", "none"]))
                      & (~frame["comment"].str.lower().isin(["nan", "none"]))]
        notes.append(f"{path.name}: {len(frame):,} rows loaded (movie column '{m_col}', text column '{c_col}').")
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=COMMENT_COLS + ["key"]), notes
    corpus = pd.concat(frames, ignore_index=True)
    corpus["key"] = corpus["movie"].map(movie_key)
    corpus["_dup"] = corpus["key"] + "|" + corpus["comment"].map(text_key)
    before = len(corpus)
    corpus = corpus[corpus["key"] != ""].drop_duplicates("_dup").drop(columns="_dup")
    notes.append(f"Removed {before - len(corpus):,} duplicate comments; {len(corpus):,} unique comments remain.")
    names = corpus.groupby("key")["movie"].agg(lambda s: s.value_counts().index[0])
    corpus["movie"] = corpus["key"].map(names)
    return corpus.reset_index(drop=True), notes


def load_corpus():
    files = (COLLECTED_FILE, VERIFIED_FILE, SENTIMENT_FILE, CLEANED_FILE)
    return _build_corpus(tuple(file_stamp(f) for f in files))


def comments_for_movie(corpus: pd.DataFrame, title: str) -> pd.DataFrame:
    if corpus.empty:
        return corpus
    return corpus[corpus["key"] == movie_key(title)].copy()


# =========================================================
# 9. YOUTUBE (Data API v3)
# =========================================================
class YouTubeError(Exception):
    def __init__(self, reasons, message, status):
        super().__init__(message)
        self.reasons = [r for r in reasons if r]
        self.message = message
        self.status = status


SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"


def _clean_key(value) -> str:
    key = str(value or "").strip().strip("\"'\u201c\u201d").strip()
    return "" if (not key or "paste" in key.lower() or "your" in key.lower()) else key


def _key_from_secrets_file():
    """Read the key straight from <project>/.streamlit/secrets.toml (works from any working directory)."""
    if not SECRETS_PATH.exists():
        return "", f"File not found: {SECRETS_PATH}"
    try:
        import tomllib

        data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        return "", f"secrets.toml exists but cannot be read ({type(exc).__name__}). The key must be inside straight quotes."
    for name, value in data.items():
        if str(name).strip().upper() == "YOUTUBE_API_KEY":
            return _clean_key(value), ""
    return "", f"secrets.toml has no YOUTUBE_API_KEY line (it contains: {', '.join(map(str, data)) or 'nothing'})."


def find_youtube_key():
    """Return (key, where_found, problems). Checks Streamlit secrets, the project file, then an env variable."""
    problems = []
    try:
        key = _clean_key(st.secrets["YOUTUBE_API_KEY"])
        if key:
            return key, "Streamlit secrets", problems
    except Exception:  # noqa: BLE001
        pass
    key, problem = _key_from_secrets_file()
    if key:
        return key, "secrets.toml in the project folder", problems
    if problem:
        problems.append(problem)
    key = _clean_key(os.getenv("YOUTUBE_API_KEY", ""))
    if key:
        return key, "environment variable", problems
    problems.append("No YOUTUBE_API_KEY environment variable is set.")
    return "", "", problems


def get_youtube_key() -> str:
    return find_youtube_key()[0]


def _yt_get(endpoint: str, params: dict) -> dict:
    resp = requests.get(f"{YOUTUBE_API}/{endpoint}", params=params, timeout=25)
    if resp.status_code == 200:
        return resp.json()
    try:
        err = resp.json().get("error", {})
    except ValueError:
        err = {}
    reasons = [e.get("reason") for e in err.get("errors", []) if isinstance(e, dict)]
    reasons += [d.get("reason") for d in err.get("details", []) if isinstance(d, dict)]
    raise YouTubeError(reasons, err.get("message", resp.text[:200]), resp.status_code)


def explain_youtube_error(exc) -> str:
    reasons = " ".join(getattr(exc, "reasons", [])).lower()
    msg = getattr(exc, "message", str(exc))
    if "api_key_invalid" in reasons or "keyinvalid" in reasons or "api key not valid" in msg.lower():
        return "The YouTube API key is invalid. Check YOUTUBE_API_KEY in .streamlit/secrets.toml."
    if "quota" in reasons or "dailylimit" in reasons or "ratelimit" in reasons:
        return ("The YouTube API quota is used up for today (it resets at midnight Pacific Time). "
                "Comments you already fetched are cached, so searches you ran before still work.")
    if "commentsdisabled" in reasons:
        return "Comments are turned off for this video."
    if "accessnotconfigured" in reasons or "has not been used" in msg.lower():
        return "YouTube Data API v3 is not enabled for this key's Google Cloud project."
    if "iprefererblocked" in reasons or "referer" in reasons:
        return "This API key has restrictions that block requests from this computer."
    return f"YouTube API error: {msg}"


def render_key_doctor(prefix: str, detail: str = ""):
    """Owner-only help: shows where the key was looked for and tests it with one cheap API call."""
    key, where, problems = find_youtube_key()
    if detail:
        st.write(f"Last error: {detail}")
    if key:
        st.write(f"Key found in: {where}. Length {len(key)} characters, starts with “{key[:4]}”.")
    else:
        st.write("No key was found.")
        for line in problems:
            st.write("• " + line)
        st.write("Create `C:\\CineSense\\.streamlit\\secrets.toml` with one line: "
                 "`YOUTUBE_API_KEY = \"your-key\"` and restart the app.")
    if st.button("Test the key now", key=f"{prefix}_testkey"):
        if not key:
            st.error("There is no key to test.")
        else:
            try:
                _yt_get("videos", {"part": "id", "id": "dQw4w9WgXcQ", "key": key})
                st.success("The key works. If audience response still fails, press Try again.")
            except YouTubeError as exc:
                st.error(explain_youtube_error(exc))
                st.caption(f"Google said: {exc.message[:200]} {exc.reasons}")
            except requests.RequestException as exc:
                st.error(f"Network problem: {exc}")


FATAL_YT = ("quota", "dailylimit", "ratelimit", "keyinvalid", "api_key_invalid", "accessnotconfigured", "iprefererblocked")

REVIEW_WORDS = {"review", "reviews", "reaction", "reactions", "public", "audience", "verdict", "response",
                "opinion", "opinions", "fdfs", "rating", "honest"}
REJECT_WORDS = {"trailer", "teaser", "song", "songs", "jukebox", "lyrical", "shorts", "promo", "glimpse", "making"}
SEQUEL_WORDS = {"chapter", "part", "ii", "iii", "iv", "returns", "reloaded", "begins", "rising", "legacy"}


def video_matches_movie(video_title: str, movie: str, release_ts, published_at: str, relaxed: bool = False) -> bool:
    """Strict title matching to avoid unrelated videos and sequel/original mix-ups."""
    tokens = re.findall(r"[a-z0-9]+", video_title.lower())
    target = re.findall(r"[a-z0-9]+", re.sub(r"\([^)]*\)", " ", movie.lower()))
    if not target or not tokens:
        return False
    n = len(target)
    pos = next((i for i in range(len(tokens) - n + 1) if tokens[i:i + n] == target), None)
    if pos is None:
        return False
    nxt = tokens[pos + n] if pos + n < len(tokens) else ""
    if nxt and (nxt.isdigit() or nxt in SEQUEL_WORDS) and not target[-1].isdigit():
        return False  # e.g. searching "Jailer" must not match "Jailer 2 review"
    if relaxed:      # second pass: reactions / trailers are fine, songs and shorts are not
        if {"song", "songs", "jukebox", "lyrical", "shorts"} & set(tokens):
            return False
    elif not REVIEW_WORDS & set(tokens) or REJECT_WORDS & set(tokens):
        return False
    years = {int(t) for t in tokens if re.fullmatch(r"20\d\d", t)}
    if pd.notna(release_ts) and years and release_ts.year not in years and not (years & {release_ts.year - 1, release_ts.year + 1}):
        return False
    if pd.notna(release_ts) and published_at:
        pub = pd.to_datetime(published_at, errors="coerce", utc=True)
        if pd.notna(pub) and pub.tz_localize(None) < release_ts - timedelta(days=180 if relaxed else 14):
            return False  # video is older than the film's release window
    return True


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def yt_search_videos(movie: str, release_iso: str, _api_key: str, language: str = "", relaxed: bool = False) -> list[dict]:
    release_ts = pd.to_datetime(release_iso, errors="coerce") if release_iso else pd.NaT
    year = f" {release_ts.year}" if pd.notna(release_ts) else ""
    data = _yt_get("search", {"part": "snippet", "q": " ".join(x for x in [movie, language.split(",")[0].strip(), "movie" if relaxed else "movie review", year.strip()] if x), "type": "video",
                              "regionCode": "IN", "order": "relevance", "maxResults": 25, "key": _api_key})
    videos, seen = [], set()
    for item in data.get("items", []):
        snip, vid = item.get("snippet", {}), item.get("id", {}).get("videoId")
        if not vid or vid in seen:
            continue
        if not video_matches_movie(snip.get("title", ""), movie, release_ts, snip.get("publishedAt", ""), relaxed):
            continue
        seen.add(vid)
        videos.append({"video_id": vid, "title": html.unescape(snip.get("title", "")),
                       "channel": snip.get("channelTitle", ""), "published_at": snip.get("publishedAt", ""),
                       "url": f"https://www.youtube.com/watch?v={vid}"})
    return videos[:8]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def yt_fetch_comments(video_id: str, max_comments: int, _api_key: str):
    """Return (comments, status). status is 'ok' or 'comments_disabled'."""
    comments, token = [], None
    while len(comments) < max_comments:
        params = {"part": "snippet", "videoId": video_id, "maxResults": min(100, max_comments - len(comments)),
                  "order": "relevance", "textFormat": "plainText", "key": _api_key}
        if token:
            params["pageToken"] = token
        try:
            data = _yt_get("commentThreads", params)
        except YouTubeError as exc:
            if "commentsDisabled" in exc.reasons:
                return comments, "comments_disabled"
            raise
        for item in data.get("items", []):
            top = item.get("snippet", {}).get("topLevelComment", {})
            snip = top.get("snippet", {})
            text = html.unescape(snip.get("textDisplay", "")).strip()
            if text:
                comments.append({"comment": text, "comment_id": top.get("id", ""),
                                 "author": snip.get("authorDisplayName", ""),
                                 "published_at": snip.get("publishedAt", ""), "likes": snip.get("likeCount", 0)})
        token = data.get("nextPageToken")
        if not token:
            break
    return comments, "ok"


def search_videos_smart(movie: str, release_ts, language: str, key: str):
    """Review videos first; if none match, reaction/trailer videos about the same film."""
    iso = release_ts.isoformat() if pd.notna(release_ts) else ""
    vids = yt_search_videos(movie, iso, key, language, False)
    if vids:
        return vids, "review videos"
    return yt_search_videos(movie, iso, key, language, True), "reaction/trailer videos"


def auto_collect_comments(title: str, release_ts, language: str = "", strict_only: bool = False):
    """One-click collection for a film: find review videos, fetch comments, save to CSV.
    Returns (status, message). status: ok | no_key | no_videos | no_comments | error."""
    key = get_youtube_key()
    if not key:
        return "no_key", "No API key is configured."
    try:
        if strict_only:
            videos, kind = yt_search_videos(title, release_ts.isoformat() if pd.notna(release_ts) else "", key, language, False), "review videos"
        else:
            videos, kind = search_videos_smart(title, release_ts, language, key)
        if not videos:
            return "no_videos", "No videos matching this exact title were found yet (searched reviews, then reactions and trailers)."
        rows = []
        for vid in videos[:3]:
            try:
                comments, _ = yt_fetch_comments(vid["video_id"], 50, key)
            except YouTubeError as exc:
                if any(f in " ".join(exc.reasons).lower() or f in exc.message.lower() for f in FATAL_YT):
                    raise
                continue
            for c in comments:
                rows.append({"movie": title, "comment": c["comment"], "video_title": vid["title"],
                             "channel": vid["channel"], "video_url": vid["url"],
                             "comment_url": f"{vid['url']}&lc={c['comment_id']}" if c["comment_id"] else vid["url"],
                             "published_at": c["published_at"], "likes": c["likes"], "author": c["author"]})
        if not rows:
            return "no_comments", "Review videos exist, but they have no public comments (or comments are disabled)."
        save_collected(pd.DataFrame(rows))
        return "ok", f"{len(rows)} comments collected from {kind}."
    except YouTubeError as exc:
        return "error", explain_youtube_error(exc)
    except requests.RequestException as exc:
        return "error", f"Network problem while contacting YouTube: {exc}"


ATTEMPTS_FILE = DATA_DIR / "comment_attempts.csv"


def load_attempts() -> dict:
    df = read_csv_any(ATTEMPTS_FILE)
    if df.empty or not {"key", "checked_ts"} <= set(df.columns):
        return {}
    return {str(r["key"]): float(r["checked_ts"]) for r in df.to_dict("records")}


def prefetch_comments(frame: pd.DataFrame, counts: dict) -> bool:
    """Collect comments for a few released films that have none saved yet, so Home can show only films people discuss.
    Returns True when new comments were saved."""
    if not get_youtube_key():
        return False
    attempts, now, saved = load_attempts(), time.time(), False
    todo = [r for r in frame.to_dict("records")
            if counts.get(r["key"], 0) < MIN_COMMENTS and now - attempts.get(r["key"], 0) > 12 * 3600][:MAX_NEW_CHECKS]
    for r in todo:
        status, _ = auto_collect_comments(r["title"], r["release_date"], r["language"], strict_only=True)
        if status == "error":                    # quota or network problem: stop and retry later
            break
        attempts[r["key"]] = now
        saved = saved or status == "ok"
    try:
        DATA_DIR.mkdir(exist_ok=True)
        pd.DataFrame({"key": list(attempts), "checked_ts": list(attempts.values())}).to_csv(ATTEMPTS_FILE, index=False)
    except OSError:
        pass
    return saved


def save_collected(df: pd.DataFrame) -> int:
    """Append fetched comments to data/youtube_comments_collected.csv (de-duplicated). Returns rows added."""
    cols = ["movie", "comment", "video_title", "channel", "video_url", "comment_url", "published_at", "likes", "author"]
    new = df[cols].copy()
    old = read_csv_any(COLLECTED_FILE)
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    merged["_k"] = merged["movie"].map(movie_key) + "|" + merged["comment"].map(text_key)
    merged = merged.drop_duplicates("_k").drop(columns="_k")
    DATA_DIR.mkdir(exist_ok=True)
    merged.to_csv(COLLECTED_FILE, index=False)
    return len(merged) - (len(old) if not old.empty else 0)


# =========================================================
# 10. SESSION STATE AND CALLBACKS
# =========================================================
for _k, _v in {"nav": "Home", "selected": None, "review_text": "", "review_result": None, "yt_movie": "",
               "yt_videos": [], "yt_videos_for": "", "yt_df": None, "yt_notes": [], "dash_run": False}.items():
    st.session_state.setdefault(_k, _v)


def cb_select(info):
    st.session_state.selected = info


def save_manual_poster(info: dict, uploaded) -> str:
    """Save an uploaded poster image and register the film in data/my_movies.csv (overrides the automatic entry)."""
    if uploaded.size > 3_000_000:
        return "That image is larger than 3 MB. Please choose a smaller file."
    POSTER_DIR.mkdir(exist_ok=True)
    fname = f"{movie_key(info['title'])}{'.png' if uploaded.name.lower().endswith('.png') else '.jpg'}"
    (POSTER_DIR / fname).write_bytes(uploaded.getvalue())
    cols = ["title", "language", "release_date", "poster"]
    df = read_csv_any(MY_MOVIES_FILE)
    df = df[[c for c in cols if c in df.columns]] if not df.empty else pd.DataFrame(columns=cols)
    if "title" in df.columns and not df.empty:
        df = df[df["title"].map(movie_key) != movie_key(info["title"])]
    date_txt = info["date"].date().isoformat() if info.get("date") is not None else ""
    df = pd.concat([df, pd.DataFrame([{"title": info["title"], "language": info["language"],
                                       "release_date": date_txt, "poster": fname}])], ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(MY_MOVIES_FILE, index=False)
    return ""


def cb_clear_selected():
    st.session_state.selected = None


def cb_open_explorer():
    st.session_state.nav = "Movie Explorer"


def cb_find_comments(title):
    st.session_state.yt_movie = title
    st.session_state.yt_videos, st.session_state.yt_df = [], None
    st.session_state.nav = "Viewer Reviews"


def cb_example(text):
    st.session_state.review_text = text
    st.session_state.review_result = None


def cb_clear_review():
    st.session_state.review_text = ""
    st.session_state.review_result = None


# =========================================================
# 11. SHARED UI: movie cards, selected-movie panel
# =========================================================
catalogue, catalogue_meta = load_catalogue()
corpus, corpus_notes = load_corpus()
TODAY = today_ist()


def render_movie_grid(df: pd.DataFrame, prefix: str, per_row: int = 4):
    records = df.to_dict("records")
    posters = get_posters([(r["title"], int(r["year"]) if pd.notna(r["year"]) else None, r["poster_csv"], r.get("page_url", "")) for r in records])
    for start in range(0, len(records), per_row):
        cols = st.columns(per_row)
        for offset, rec in enumerate(records[start:start + per_row]):
            info = movie_info(rec)
            ts = rec["release_date"]
            url = posters.get(poster_key(info["title"], info["year"]), "")
            badge_label = "OUT TODAY" if (pd.notna(ts) and ts.date() == TODAY) else ("COMING SOON" if (pd.notna(ts) and ts.date() > TODAY) else "")
            meta = " · ".join(x for x in [info["language"], fmt_date(ts)] if x)
            with cols[offset]:
                md_html(poster_block(info["title"], url, badge=badge_label)
                        + f'<div class="mc-title">{safe(info["title"])}</div><div class="mc-meta">{safe(meta)}</div>')
                st.button("Analyse sentiment", key=f"{prefix}_{start + offset}", on_click=cb_select, args=(info,))


def cb_retry_collect(k):
    st.session_state.auto_attempts.pop(k, None)


def render_movie_analysis(title: str, prefix: str, release_ts=None, language: str = ""):
    sub = comments_for_movie(corpus, title)
    if sub.empty:
        attempts = st.session_state.setdefault("auto_attempts", {})
        k = movie_key(title)
        if k not in attempts or (attempts[k][0] == "no_key" and get_youtube_key()):
            with st.spinner("Loading audience response..."):
                attempts[k] = auto_collect_comments(title, release_ts, language)
            if attempts[k][0] == "ok":
                st.rerun()
        status = attempts[k][0]
        text = {"no_key": "Audience response is not available right now.",
                "error": "Audience response could not be loaded right now. Please try again in a moment."}.get(
            status, "There are no public viewer comments for this film yet. Check back after more people have watched it.")
        md_html(f'<div class="card"><h4>Audience response</h4><div class="small">{safe(text)}</div></div>')
        st.button("Try again", key=f"{prefix}_retry", on_click=cb_retry_collect, args=(k,))
        if ADMIN and status not in ("no_key", "error"):
            st.caption(f"Owner view: {attempts[k][1]}")
        if ADMIN and status in ("no_key", "error"):
            with st.expander("Owner view: setup note", expanded=True):
                render_key_doctor(prefix, attempts[k][1])
        return
    try:
        results = analyse_cached(sub["comment"].tolist(), f"Analysing audience response for {title}")
    except ModelError as exc:
        st.error("The sentiment models could not be loaded. Check your internet connection and the install steps.")
        st.code(str(exc))
        return
    sub = attach_results(sub, results)
    counts = count_labels(sub["sentiment"])
    total = sum(counts.values())
    st.markdown("#### Audience response")
    left, right = st.columns([3, 2])
    with left:
        chart_donut(counts, f"{title}", size=5.2)
    with right:
        for lab in ORDER:
            pct = counts[lab] / total * 100 if total else 0
            md_html(f'<div class="card" style="padding:12px 16px;margin-bottom:10px;"><span class="pill pill-{lab}">{lab}</span>'
                    f'<div style="font-size:1.7rem;font-weight:800;color:#fff;margin-top:6px;">{pct:.1f}%</div>'
                    f'<div class="small">{counts[lab]:,} comments</div></div>')
    md_html(f'<div class="card"><h4>Summary</h4><div>{safe(audience_summary(counts, title))}</div></div>')
    st.markdown("#### What viewers are saying")
    show_comment_list(sub.assign(video_title="", channel="", video_url="", comment_url=""), f"{prefix}_cm")


def render_selected_panel(prefix: str, offer_explorer: bool):
    sel = st.session_state.selected
    if not sel:
        return
    url = get_posters([(sel["title"], sel["year"], sel["csv_poster"], sel.get("page_url", ""))]).get(poster_key(sel["title"], sel["year"]), "")
    ts = sel["date"]
    is_today = ts is not None and ts.date() == TODAY
    c1, c2 = st.columns([1, 3])
    with c1:
        md_html(poster_block(sel["title"], url, badge=is_today))
    with c2:
        md_html(f'<div class="page-title">{safe(sel["title"])}</div>'
                f'<div class="page-sub">{safe(" · ".join(x for x in [sel["language"], "Released " + fmt_date(ts) if ts is not None else ""] if x))}</div>')
        b1, b2 = st.columns(2)
        b1.button("Close", key=f"{prefix}_close", on_click=cb_clear_selected)
        if offer_explorer:
            b2.button("Open in Movie Explorer", key=f"{prefix}_explorer", on_click=cb_open_explorer)
    if ADMIN:
        with st.expander("Owner tools: set this film's poster"):
            if st.button("Hide this film from the Home page", key=f"{prefix}_hidefilm"):
                add_hidden(sel["title"])
                st.rerun()
            if url and st.button("Reject the current poster and find another", key=f"{prefix}_rejectposter"):
                add_reject(url)
                st.rerun()
            up = st.file_uploader("Upload the official poster (JPG or PNG, under 3 MB)", type=["jpg", "jpeg", "png"], key=f"{prefix}_posterfile")
            if up is not None and st.button("Save poster", key=f"{prefix}_saveposter"):
                problem = save_manual_poster(sel, up)
                if problem:
                    st.error(problem)
                else:
                    st.success("Poster saved. It now appears on the cards. Press R or refresh the page to see it.")
    st.markdown("---")
    render_movie_analysis(sel["title"], prefix, sel["date"], sel["language"])
    st.markdown("---")


# =========================================================
# 12. SIDEBAR
# =========================================================
with st.sidebar:
    md_html('<div style="padding:8px 4px 18px;"><div style="font-size:30px;">🎬</div>'
            '<div style="font-family:Fraunces,serif;font-size:27px;font-weight:800;color:#fff;">Cine<span style="color:#b794ff;">Sense</span></div>'
            '<div class="small">Movie review sentiment analysis</div></div>')
    page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
    if ADMIN:
        st.markdown("---")
        _rel = catalogue[catalogue["release_date"].dt.date <= TODAY] if not catalogue.empty else catalogue
        latest = _rel["release_date"].max() if not _rel.empty else pd.NaT
        st.caption(f"App version: {APP_VERSION}")
        st.caption(f"Today (IST): {TODAY.strftime('%d %b %Y')}")
        st.caption(f"Catalogue: {len(catalogue):,} movies" + (f", latest released {fmt_date(latest)}" if pd.notna(latest) else ""))
        st.caption(f"Comments available: {len(corpus):,}")
        st.caption("Audience data service: " + ("connected" if get_youtube_key() else "NOT connected"))


# =========================================================
# 13. PAGE 1 - HOME
# =========================================================
if page == "Home":
    n_movies_c = corpus["key"].nunique() if not corpus.empty else 0
    md_html(f"""
    <div class="hero">
      <div class="hero-title">Every movie has a story.<br>Every audience has a voice.</div>
      <div class="hero-sub">CineSense reads what viewers write about Indian films and classifies each opinion as
      positive, neutral or negative using transformer-based Natural Language Processing.</div>
      <div class="chips"><span class="chip">{len(corpus):,} comments collected</span>
      <span class="chip">{n_movies_c:,} movies with comments</span><span class="chip">{len(catalogue):,} movies in the catalogue</span></div>
    </div>""")

    render_selected_panel("home", offer_explorer=True)

    if ADMIN and not catalogue_meta.get("calendar_rows"):
        st.info(f"Owner view: {catalogue_meta.get('calendar_status', '')} Showing Wikipedia data instead, which can be a few weeks behind.")
    if ADMIN and catalogue_meta.get("fallback"):
        st.warning("Owner view: live releases could not be loaded, so older saved data is shown. "
                   f"Reason: {catalogue_meta.get('live_status', 'unknown')} Press Refresh releases to try again.")
    if catalogue.empty or catalogue["release_date"].notna().sum() == 0:
        st.warning("The latest releases are not available right now. Please check back in a few minutes.")
        if ADMIN:
            st.write("Owner view: CineSense needs an internet connection to read releases. Add films by hand in data/my_movies.csv if needed.")
            st.write(catalogue_meta)
    else:
        released = catalogue[catalogue["release_date"].notna()
                             & (catalogue["release_date"].dt.date <= TODAY + timedelta(days=UPCOMING_DAYS))].copy()   # includes upcoming openings
        released["is_today"] = released["release_date"].dt.date == TODAY
        released["is_upcoming"] = released["release_date"].dt.date > TODAY
        released = released.sort_values(["is_today", "is_upcoming", "release_date", "title"], ascending=[False, True, False, True])
        released["primary"] = released["language"].str.split(",").str[0].str.strip().str.title().replace("", "Other")
        newest = released.loc[~released["is_upcoming"], "release_date"].max()
        if ADMIN:
            st.button("Refresh releases", key="home_refresh", on_click=cb_refresh_catalogue)
        if pd.notna(newest) and (TODAY - newest.date()).days > 14:
            st.warning(f"The newest release in the data is {fmt_date(newest)}, so very recent films may be missing. "
                       "The list refreshes automatically.")

        cutoff = TODAY - timedelta(days=RECENT_DAYS)
        hidden = load_hidden()
        released = released[~released["key"].isin(hidden)]
        pool = released[(released["release_date"].dt.date >= cutoff) & (released["release_date"].dt.date <= TODAY + timedelta(days=7))]
        if pool.empty:
            pool = released.head(30)

        # films you always want on the main grid (looked up by title in the whole catalogue)
        pinned_parts = []
        for t in PINNED_TITLES:
            m = catalogue[(catalogue["key"] == movie_key(t)) & catalogue["release_date"].notna()
                          & (catalogue["release_date"].dt.date <= TODAY + timedelta(days=UPCOMING_DAYS))]
            if not m.empty:
                r = m.sort_values("release_date", ascending=False).iloc[[0]].copy()
                r["is_today"] = r["release_date"].dt.date == TODAY
                r["is_upcoming"] = r["release_date"].dt.date > TODAY
                r["primary"] = r["language"].str.split(",").str[0].str.strip().str.title().replace("", "Other")
                pinned_parts.append(r)
        pinned = pd.concat(pinned_parts) if pinned_parts else pool.head(0)

        def keep_with_poster(frame, posters):
            keep = [bool(posters.get(poster_key(t, int(y) if pd.notna(y) else None)))
                    for t, y in zip(frame["title"], frame["year"])]
            return frame[np.array(keep, dtype=bool)]

        def poster_items(frame):
            return [(r["title"], int(r["year"]) if pd.notna(r["year"]) else None, r["poster_csv"], r.get("page_url", ""))
                    for r in frame.to_dict("records")]

        language_order = list(MAIN_LANGUAGES) + [l for l in HOME_LANGUAGE_ORDER if l not in MAIN_LANGUAGES]
        language_order += sorted(set(pool["primary"]) - set(language_order))
        candidates = [pool.head(40)] + [pool[pool["primary"] == lang].head(8) for lang in language_order]
        candidates = pd.concat(candidates + [pinned]).drop_duplicates(["key", "year"])
        with st.spinner("Loading the latest releases and their posters..."):
            posters = get_posters(poster_items(candidates))

        # 1) Latest releases, all languages together: only films with a real poster
        counts = corpus.groupby("key").size().to_dict() if not corpus.empty else {}
        have_key = bool(get_youtube_key())
        if have_key:                                   # check audience response for a few films not checked before
            with st.spinner("Checking audience response for the latest films..."):
                if prefetch_comments(keep_with_poster(pool[~pool["is_upcoming"]], posters), counts):
                    st.rerun()

        def eligible(frame):
            """Released films need real audience comments; films opening soon are exempt."""
            if not have_key:
                return frame
            ok = [bool(up) or counts.get(k, 0) >= MIN_COMMENTS for k, up in zip(frame["key"], frame["is_upcoming"])]
            return frame[np.array(ok, dtype=bool)]

        by_lang = {lang: eligible(keep_with_poster(pool[pool["primary"] == lang], posters)) for lang in language_order}
        mix, used = [], set()

        def take(r):
            k = (r["key"], r["year"])
            if k not in used:
                used.add(k)
                mix.append(r)

        for _, r in keep_with_poster(pinned, posters).iterrows():
            take(r)
        for lang, need in {"Kannada": 2, "Malayalam": 1}.items():        # keep these languages visible (posters only)
            for _, r in by_lang.get(lang, pool.head(0)).head(need).iterrows():
                take(r)
        rank = 0
        while len(mix) < TOP_FILMS and any(len(g) > rank for g in by_lang.values()):
            for lang in language_order:                                  # newest film of every language, then the next...
                g = by_lang[lang]
                if len(g) > rank and len(mix) < TOP_FILMS:
                    take(g.iloc[rank])
            rank += 1
        if len(mix) < MIN_GRID:                                          # not enough discussed films yet: add films with posters
            for _, r in keep_with_poster(pool, posters).iterrows():
                if len(mix) >= MIN_GRID:
                    break
                take(r)
        top = (pd.DataFrame(mix).sort_values(["is_today", "is_upcoming", "release_date", "title"], ascending=[False, True, False, True])
               if mix else pool.head(0))
        st.markdown("## Latest releases")
        st.caption("Now showing and opening this week, from every language.")
        if top.empty:
            st.info("The latest posters are still loading. Please refresh the page in a moment.")
        else:
            render_movie_grid(top, "home_top", per_row=5)

        # See more: more recent films (posters only), one list, not split by language
        if st.session_state.get("home_extra", 0) == 0:
            st.button("See more", key="home_see_more", on_click=cb_see_more)
        else:
            st.markdown("## More new releases")
            top_keys = set(zip(top["key"], top["year"])) if not top.empty else set()
            not_in_top = lambda f: f[np.array([(k, y) not in top_keys for k, y in zip(f["key"], f["year"])], dtype=bool)]
            recent_more = not_in_top(released[(released["release_date"].dt.date >= TODAY - timedelta(days=MORE_DAYS))
                                              & (released["release_date"].dt.date <= TODAY)]).head(30)
            next_up = not_in_top(released[released["release_date"].dt.date > TODAY].sort_values("release_date")).head(30)
            with st.spinner("Loading more films..."):
                posters_more = get_posters(poster_items(pd.concat([recent_more, next_up])))
            recent_show = eligible(keep_with_poster(recent_more, posters_more)).head(10)       # newest first
            next_show = keep_with_poster(next_up, posters_more).head(10)              # soonest first
            more = pd.concat([recent_show, next_show])
            if more.empty:
                st.info("No more films with posters are available right now.")
            else:
                render_movie_grid(more, "home_more", per_row=5)
            st.button("Show less", key="home_show_less", on_click=cb_show_less)

        if ADMIN:
            with st.expander("Owner view: poster and catalogue diagnostics"):
                st.write({k: v for k, v in catalogue_meta.items() if k != "columns"})
                pc = read_csv_any(POSTER_CACHE_FILE)
                st.dataframe(pc[[c for c in ["title", "status", "source", "poster_url"] if c in pc.columns]] if not pc.empty else pc, hide_index=True)


# =========================================================
# 14. PAGE 2 - DASHBOARD
# =========================================================
elif page == "Dashboard":
    page_heading("Dashboard", "Three-class sentiment (Positive, Neutral, Negative) computed by the transformer models "
                              "on the comments in your data folder.")
    if corpus.empty:
        st.warning("No comment data could be loaded, so there is nothing to analyse yet.")
    else:
        f1, f2 = st.columns(2)
        movies_sel = f1.multiselect("Movies", sorted(corpus["movie"].unique()), placeholder="All movies")
        sources_sel = f2.multiselect("Data source", sorted(corpus["source"].unique()), placeholder="All sources")
        view = corpus
        if movies_sel:
            view = view[view["movie"].isin(movies_sel)]
        if sources_sel:
            view = view[view["source"].isin(sources_sel)]

        pending = pending_count(view["comment"].tolist())
        if pending:
            md_html(f'<div class="card"><h4>{pending:,} comments still need analysis</h4><div class="small">'
                    "The models run once per comment and results are saved to data/analysis_cache.csv, so later "
                    "visits are instant. On a CPU this can take a few minutes for large datasets.</div></div>")
            if st.button(f"Analyse {pending:,} comments now", key="dash_go"):
                st.session_state.dash_run = True
        if pending and not st.session_state.dash_run:
            st.info("Press the button above to run the sentiment models on the selected comments.")
        else:
            try:
                results = analyse_cached(view["comment"].tolist(), "Analysing dataset")
            except ModelError as exc:
                st.error("The sentiment models could not be loaded.")
                st.code(str(exc))
                results = None
            st.session_state.dash_run = False
            if results is not None:
                data = attach_results(view, results)
                g1, g2 = st.columns(2)
                routes = g1.multiselect("Model used", sorted(data["model_route"].unique()), placeholder="All models")
                sents = g2.multiselect("Sentiment", ORDER, placeholder="All sentiments")
                if routes:
                    data = data[data["model_route"].isin(routes)]
                if sents:
                    data = data[data["sentiment"].isin(sents)]

                counts = count_labels(data["sentiment"])
                tab1, tab2, tab3 = st.tabs(["Overview", "Movie comparison", "Comments table"])
                with tab1:
                    m1, m2 = st.columns(2)
                    m1.metric("Movies analysed", f"{data['key'].nunique():,}")
                    m2.metric("Comments analysed", f"{len(data):,}")
                    show_metrics(counts)
                    show_charts(counts, "Overall sentiment distribution")
                with tab2:
                    min_n = st.slider("Minimum comments per movie", 1, 100, 5)
                    rows = []
                    for name, grp in data.groupby("movie"):
                        c = count_labels(grp["sentiment"])
                        tot = len(grp)
                        rows.append({"Movie": name, "Total comments": tot, "Positive": c["Positive"],
                                     "Neutral": c["Neutral"], "Negative": c["Negative"],
                                     **{f"{k} %": round(c[k] / tot * 100, 1) for k in ORDER},
                                     "Dominant": max(ORDER, key=lambda k: c[k])})
                    table = pd.DataFrame(rows)
                    table = table[table["Total comments"] >= min_n] if not table.empty else table
                    if table.empty:
                        st.info("No movie has that many comments under the current filters. Lower the minimum.")
                    else:
                        chart_movie_stack(table)
                        st.dataframe(table.sort_values("Total comments", ascending=False), hide_index=True)
                        st.download_button("Download movie summary (CSV)", table.to_csv(index=False).encode("utf-8"),
                                           "cinesense_movie_sentiment_summary.csv", "text/csv")
                with tab3:
                    show_cols = ["movie", "comment", "sentiment", "score", "model_route", "script", "source"]
                    st.dataframe(data[show_cols], hide_index=True)
                    st.download_button("Download analysed comments (CSV)", data[show_cols].to_csv(index=False).encode("utf-8"),
                                       "cinesense_analysed_comments.csv", "text/csv")
    if ADMIN:
        with st.expander("Owner view: data sources used"):
            for note in corpus_notes:
                st.write("• " + note)
            st.caption("Old binary (positive/negative) percentages in movie_sentiment_summary.csv are deliberately NOT used; "
                       "every number here comes from the new three-class models.")


# =========================================================
# 15. PAGE 3 - REVIEW ANALYZER
# =========================================================
elif page == "Review Analyzer":
    page_heading("Review Analyzer", "Write a movie review and see how the NLP models classify it.")
    st.text_area("Your movie review", key="review_text", height=150,
                 placeholder="e.g. The acting was good but the story was boring 😢")
    b1, b2 = st.columns([3, 1])
    go = b1.button("Analyse review", key="rev_go")
    b2.button("Clear", key="rev_clear", on_click=cb_clear_review)

    st.caption("Try an example (the models analyse it; nothing is pre-programmed):")
    examples = ["The movie was amazing!", "The movie was not good.", "The acting was good but the story was boring.",
                "Good movie 😢", "Worst film ever.", "😍🔥👏"]
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        ex_cols[i % 3].button(ex, key=f"ex_{i}", on_click=cb_example, args=(ex,))

    if go:
        text = st.session_state.review_text.strip()
        if not text:
            st.warning("Write a review first, then press Analyse review.")
        else:
            models, err = ensure_models()
            if models is None:
                st.error("The sentiment models could not be loaded.")
                st.code(err)
            else:
                with st.spinner("Analysing..."):
                    st.session_state.review_result = analyse_core([text], models)[0]

    res = st.session_state.review_result
    if res:
        md_html(f'<div class="result"><div class="small">Predicted sentiment</div>'
                f'<div class="label label-{res["sentiment"]}">{res["sentiment"]}</div>'
                f'<div class="small">Model score {res["score"]:.1f}% · {safe(res["model_route"])}</div></div>')
        m1, m2, m3 = st.columns(3)
        m1.metric("Model score", f"{res['score']:.1f}%")
        m2.metric("Script detected", res["script"].split(" (")[0])
        m3.metric("Emoji signal", res["emoji_signal"] or "None")
        if res.get("emojis"):
            st.markdown("**Emoji analysis:** each emoji is read as a positive or negative signal and compared with the words.")
            st.dataframe(pd.DataFrame(res["emojis"]), hide_index=True)
        if res["reason"]:
            st.info(res["reason"])
        if res["model_label"] != res["sentiment"]:
            st.caption(f"The model on its own predicted {res['model_label']}; the final label is {res['sentiment']} "
                       "after the mixed-opinion / emoji checks above.")
        if res["parts"]:
            st.markdown("**Contrast check:** the review was split at words like 'but' and 'however' and each part scored.")
            st.dataframe(pd.DataFrame(res["parts"]), hide_index=True)
        st.caption("The score is the model's confidence in its own prediction, not a measure of accuracy. "
                   "Sarcasm, slang, code-mixed text and romanised Indian languages (e.g. Hindi typed in English letters) "
                   "are often misread, because romanised text is routed to the English model.")


# =========================================================
# 16. PAGE 4 - MOVIE EXPLORER
# =========================================================
elif page == "Movie Explorer":
    page_heading("Movie Explorer", "Search the catalogue, pick a film and explore its audience sentiment.")
    render_selected_panel("exp", offer_explorer=False)

    if catalogue.empty:
        st.warning("The movie catalogue is missing or has no title column (data/released_indian_movies.csv).")
    else:
        languages = sorted({t.strip().title() for v in catalogue["language"] for t in re.split(r"[,/&;]| and ", v) if t.strip()})
        f1, f2, f3 = st.columns([3, 2, 2])
        query = f1.text_input("Search by movie name", placeholder="Type a movie name...")
        lang = f2.selectbox("Language", ["All languages"] + languages)
        order = f3.selectbox("Sort by", ["Newest first", "A to Z"])
        c1, c2 = st.columns(2)
        upcoming = c1.checkbox("Include upcoming", value=False)
        reviewed_only = c2.checkbox("Only films with audience reviews", value=True)

        counts = corpus.groupby("key").size().to_dict() if not corpus.empty else {}
        have_key = bool(get_youtube_key())
        if reviewed_only and have_key and not query.strip():        # each visit checks a few more recent films
            recent = catalogue[catalogue["release_date"].notna() & (catalogue["release_date"].dt.date <= TODAY)
                               & (catalogue["release_date"].dt.date >= TODAY - timedelta(days=90))]
            recent = recent[~recent["key"].isin(load_hidden())].sort_values("release_date", ascending=False)
            with st.spinner("Checking audience reviews for recent films..."):
                if prefetch_comments(recent.head(40), counts):
                    st.rerun()

        df = catalogue.copy()
        if not upcoming:
            df = df[df["release_date"].isna() | (df["release_date"].dt.date <= TODAY)]
        if query.strip():
            df = df[df["title"].str.contains(re.escape(query.strip()), case=False, na=False)]
        if lang != "All languages":
            df = df[df["language"].str.contains(re.escape(lang), case=False, na=False)]
        if reviewed_only and have_key and not query.strip():          # searching by name always looks at every film
            df = df[np.array([counts.get(k, 0) >= MIN_COMMENTS for k in df["key"]], dtype=bool)]
        df = df.sort_values("title") if order == "A to Z" else df.sort_values(["release_date", "title"], ascending=[False, True], na_position="last")
        if reviewed_only and have_key and not query.strip() and not df.empty:      # browsing view: only films that also have a poster
            head = df.head(120)
            with st.spinner("Loading posters..."):
                pst = get_posters([(r["title"], int(r["year"]) if pd.notna(r["year"]) else None, r["poster_csv"], r.get("page_url", ""))
                                   for r in head.to_dict("records")])
            df = head[np.array([bool(pst.get(poster_key(t, int(y) if pd.notna(y) else None)))
                                for t, y in zip(head["title"], head["year"])], dtype=bool)]

        if df.empty:
            st.info("No films match yet. Untick “Only films with audience reviews”, search by name, or refresh in a minute "
                    "while more recent films are checked.")
        else:
            per_page = 12
            pages = max(1, -(-len(df) // per_page))
            page_no = st.number_input(f"Page (1 to {pages})", 1, pages, 1) if pages > 1 else 1
            st.caption(f"{len(df):,} films. " + ("Showing films that have audience reviews and a poster; untick the box or search by name to see every film."
                                                  if reviewed_only and not query.strip() else ""))
            render_movie_grid(df.iloc[(page_no - 1) * per_page: page_no * per_page], "exp")


# =========================================================
# 17. PAGE 5 - VIEWER REVIEWS (YouTube comments)
# =========================================================
elif page == "Viewer Reviews":
    page_heading("Viewer Reviews", "Public viewer comments on movie-review videos, analysed with the NLP models.")
    md_html('<div class="card"><h4>What this page shows</h4><div class="small">These are <b>public comments left on '
            "online videos about the film (reviews, reactions and trailers)</b>, collected through the YouTube Data API. They are not verified reviews "
            "and do not represent everyone who watched the film. Nothing here is generated or sample data.</div></div>")

    api_key = get_youtube_key()
    if not api_key:
        st.info("Live viewer comments are not available right now.")
        if ADMIN:
            with st.expander("Owner view: setup note"):
                render_key_doctor("vr")
    else:
        recent_titles = [""] + (catalogue[catalogue["release_date"].dt.date <= TODAY].sort_values("release_date", ascending=False)["title"].head(60).tolist() if not catalogue.empty else [])
        st.selectbox("Pick a recent release (optional)", recent_titles, key="yt_pick",
                     on_change=lambda: st.session_state.update(yt_movie=st.session_state.yt_pick) if st.session_state.yt_pick else None)
        movie_name = st.text_input("Movie name", key="yt_movie", placeholder="e.g. Kantara Chapter 1")

        release_ts, matched = pd.NaT, None
        if movie_name.strip() and not catalogue.empty:
            hit = catalogue[catalogue["key"] == movie_key(movie_name)]
            if not hit.empty:
                matched = hit.iloc[0]
                release_ts = matched["release_date"]
        if matched is not None:
            st.caption(f"Matched to catalogue: {matched['title']} · {matched['language'] or 'language n/a'} · "
                       f"released {fmt_date(release_ts) or 'date n/a'}. Videos older than the release window are ignored.")

        per_video = st.selectbox("Maximum comments per video", [20, 50, 100, 200], index=1)
        if st.button("Fetch comments", key="yt_fetch"):
            name = movie_name.strip()
            if not name:
                st.warning("Enter a movie name first.")
            else:
                rows, notes, vids = [], [], []
                try:
                    with st.spinner("Fetching comments..."):
                        vids, _kind = search_videos_smart(name, release_ts, str(matched["language"]) if matched is not None else "", api_key)
                        if not vids:
                            notes.append("No matching comments were found. Try the exact film title and check the spelling.")
                        for vid in vids[:3]:
                            try:
                                comments, status = yt_fetch_comments(vid["video_id"], per_video, api_key)
                            except YouTubeError as exc:
                                notes.append(f"{vid['title']}: {explain_youtube_error(exc)}")
                                if any(f in " ".join(exc.reasons).lower() or f in exc.message.lower() for f in FATAL_YT):
                                    break
                                continue
                            if status == "comments_disabled":
                                notes.append(f"Comments are turned off on “{vid['title']}”.")
                            for c in comments:
                                rows.append({"movie": name, "comment": c["comment"], "video_title": vid["title"],
                                             "channel": vid["channel"], "video_url": vid["url"],
                                             "comment_url": f"{vid['url']}&lc={c['comment_id']}" if c["comment_id"] else vid["url"],
                                             "published_at": c["published_at"], "likes": c["likes"], "author": c["author"]})
                    df = pd.DataFrame(rows)
                    if not df.empty:
                        df["_k"] = df["comment"].map(text_key)
                        df = df.drop_duplicates("_k").drop(columns="_k")
                        try:
                            df = attach_results(df, analyse_cached(df["comment"].tolist(), "Analysing comments"))
                        except ModelError as exc:
                            st.error("The sentiment models could not be loaded.")
                            st.code(str(exc))
                            df = pd.DataFrame()
                    st.session_state.yt_videos, st.session_state.yt_videos_for = vids, name
                    st.session_state.yt_df, st.session_state.yt_notes = df, notes
                except YouTubeError as exc:
                    st.error(explain_youtube_error(exc))
                except requests.RequestException as exc:
                    st.error(f"Network problem: {exc}")

        for note in st.session_state.yt_notes:
            st.warning(note)

        df = st.session_state.yt_df
        if df is not None:
            if df.empty:
                st.info("No comments could be analysed. See any messages above.")
            else:
                st.markdown("---")
                st.markdown(f"## Analysis: {st.session_state.yt_videos_for}")
                counts = count_labels(df["sentiment"])
                show_metrics(counts, "Comments collected")
                st.caption(f"Public comments from {df['video_url'].nunique()} selected review video(s). "
                           "Not a complete picture of all viewers.")
                show_charts(counts, "Viewer comment sentiment")
                show_comment_list(df, "yt")
                d1, d2 = st.columns(2)
                d1.download_button("Download analysed comments (CSV)", df.to_csv(index=False).encode("utf-8"),
                                   f"cinesense_{movie_key(st.session_state.yt_videos_for)}_comments.csv", "text/csv")
                if d2.button("Save to project dataset", key="yt_save"):
                    try:
                        added = save_collected(df)
                        st.success(f"Saved {added} new comments to data/youtube_comments_collected.csv. "
                                   "They now count in the Dashboard and Movie Explorer.")
                    except OSError as exc:
                        st.error(f"Could not write the file: {exc}")


# =========================================================
# 18. PAGE 6 - ABOUT
# =========================================================
elif page == "About":
    page_heading("About CineSense", "A movie-review sentiment analysis platform built on Natural Language Processing.")
    md_html("""
    <div class="hero"><div class="hero-title">What do audiences think of a film?</div>
    <div class="hero-sub">Reading thousands of comments by hand is impractical. CineSense collects public viewer
    comments, classifies each one as positive, neutral or negative with transformer models, and shows the overall picture.</div></div>""")

    st.markdown("## The problem and the objectives")
    st.write("Audience opinion about a film is spread across thousands of free-text comments in many languages, "
             "which makes it hard to see the overall mood. Objectives: (1) discover recent Indian releases, "
             "(2) collect real viewer comments, (3) classify them into three sentiment classes, (4) let users analyse "
             "their own reviews, and (5) present results clearly and honestly, including the limitations.")

    st.markdown("## How the pipeline works")
    steps = [
        ("Collect", "Public comments are fetched from videos about the film (reviews, reactions, trailers) with the YouTube Data API v3. "
                    "Existing CSV files in the data folder are loaded too and duplicates removed."),
        ("Pre-process", "HTML tags are removed, links become “http”, @mentions become “@user”, extra spaces are trimmed and "
                        "very long text is cut at 1,000 characters. Emojis are kept because they carry sentiment."),
        ("Detect script", "The app counts characters in Unicode blocks (Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, "
                          "Malayalam, Gurmukhi, Odia, Latin...). This identifies the writing system, not the exact language."),
        ("Route", "Text mostly in an Indian (or other non-Latin) script goes to the multilingual model. Latin-script text goes "
                  "to the English model."),
        ("Classify", "The chosen transformer converts the text into numbers, and a final layer outputs a probability for "
                     "Positive, Neutral and Negative. The highest probability is the label and its value is the score."),
        ("Check", "Two rule checks run on top (below): mixed opinions and emoji conflicts."),
        ("Present", "Counts and percentages for all three classes are shown in charts and tables. Results are saved to "
                    "data/analysis_cache.csv so each comment is analysed only once."),
    ]
    for i, (name, desc) in enumerate(steps, start=1):
        md_html(f'<div class="card"><h4>{i}. {name}</h4><div class="small">{safe(desc)}</div></div>')

    st.markdown("## The two models")
    st.write(f"**English:** `{ENGLISH_MODEL}` is a RoBERTa model trained on a very large collection of tweets and fine-tuned "
             "for 3-class sentiment. It is used for Latin-script text.")
    st.write(f"**Multilingual:** `{MULTI_MODEL}` is a smaller DistilBERT model built on a multilingual base model and trained for "
             "3-class sentiment on multilingual data. It is used for non-Latin scripts. Its fine-tuning data does not cover every "
             "Indian language, so results for some languages (for example Tamil, Telugu, Kannada, Malayalam) are less reliable. "
             "See the model's page on Hugging Face for its supported languages.")

    st.markdown("## Mixed opinions, negation and emoji")
    st.write("**Negation** (“not good”) is handled by the transformers themselves because they read the whole sentence, not single "
             "words. **Mixed opinions:** if a review contains words like “but” or “however”, each part is also scored; when one part "
             "is clearly positive and another clearly negative, the review is labelled Neutral and the app explains why. "
             "**Emoji:** emojis are counted as positive or negative signals. If the words and the emoji clearly disagree "
             "(“good movie 😢”), the result is Neutral. An emoji-only review is judged from its emojis. These are simple rules, "
             "not a perfect understanding of meaning.")

    st.markdown("## Data sources")
    st.write("Release data: the current month's release calendar pages of a public film-news site (Hindi, Telugu, Tamil, Kannada and "
             "Malayalam), plus Wikipedia's yearly Indian-language film categories for other languages (release date from each film's infobox). Viewer comments: public comments left on online movie-review "
             "videos, collected through the YouTube Data API v3 and saved to CSV. Posters: the film's poster from its Wikipedia article "
             "(films whose article has no poster show a title card). Poster images belong to their "
             "copyright holders and are shown for identification.")

    st.markdown("## Limitations")
    for text in [
        "The release list only includes films that have an English Wikipedia article, so it is not a complete list of Indian releases; films with no poster or trailer image show a title card.",
        "YouTube comments come from selected review videos and do not represent every viewer. They are not verified reviews.",
        "Script detection finds the writing system, not the language, and it cannot tell romanised Hindi from English, so romanised text goes to the English model and may be misclassified.",
        "Sarcasm, slang, spelling variation and code-mixed text (e.g. Hinglish) reduce accuracy.",
        "Model scores are confidence values, not accuracy. No accuracy figure is claimed because the comments have no human-labelled ground truth.",
        "The clause-splitting and emoji rules are heuristics and can be wrong.",
    ]:
        st.write("• " + text)

    st.markdown("## Future improvements")
    st.write("Transliterating romanised Indian languages before classification, fine-tuning on human-labelled Indian movie "
             "comments, aspect-based sentiment (acting, story, music), sarcasm detection, and a scheduled job to refresh the catalogue.")

st.markdown("---")
st.caption("CineSense · NLP-based movie review sentiment analysis · Third-year BSc Computer Science project")
