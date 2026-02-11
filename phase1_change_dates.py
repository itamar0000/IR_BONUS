import os
import re
import pandas as pd
from pathlib import Path

# --- keep using YOUR clean_text() as-is ---
# (paste your existing clean_text here, unchanged)
def clean_text(text):
    """
    Cleans the raw text files.
    """
    # 1. Remove tags
    text = text.replace('\\', '')
    # 2. Remove HTML tags (like <pre>, <br>)
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Remove specific noise observed in NBC/Congressional logs
    text = text.replace('Extensions of Remarks', '')
    text = text.replace('Congressional Record Vol.', '')

    # 4. Attempt to remove NBC Cookie Notice
    if "This Cookie Notice" in text:
        text = text.split("This Cookie Notice")[0]

    # 5. Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --- DATE PARSERS (updated to your exact filename formats) ---

FMT_GMT = "%a_%d_%b_%Y_%H_%M_%S_GMT"   # Fri_01_Aug_2025_07_00_00_GMT

def parse_date_uk_politics(stem: str):
    # debates2023-07-03
    m = re.search(r"debates(\d{4}-\d{2}-\d{2})", stem)
    return pd.to_datetime(m.group(1), errors="coerce") if m else pd.NaT

def parse_date_iso(stem: str):
    # 2023-07-10
    m = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    return pd.to_datetime(m.group(1), errors="coerce") if m else pd.NaT

def parse_date_gmt(stem: str):
    # Fri_01_Aug_2025_07_00_00_GMT
    return pd.to_datetime(stem, format=FMT_GMT, errors="coerce")

def get_date_parser(country: str, arena: str):
    # UK debates
    if country == "UK" and arena == "politics":
        return parse_date_uk_politics
    # US congress
    if country == "USA" and arena == "politics":
        return parse_date_iso
    # BBC + NBC
    if arena == "media":
        return parse_date_gmt
    return parse_date_iso  # safe fallback

# --- LOADER THAT RETURNS A DATAFRAME WITH DATES ---

def load_df_from_folder(folder_path: str, country: str, arena: str) -> pd.DataFrame:
    rows = []
    folder = Path(folder_path)
    if not folder.exists():
        print(f"⚠️ Folder not found: {folder_path}")
        return pd.DataFrame(columns=["country","arena","date","file","text"])

    date_parser = get_date_parser(country, arena)

    for p in folder.glob("*.txt"):
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            txt = clean_text(raw)
            if len(txt) < 50:
                continue

            stem = p.stem  # filename without .txt
            dt = date_parser(stem)
            rows.append({
                "country": country,
                "arena": arena,
                "date": dt,
                "file": p.name,
                "text": txt
            })
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

# --- WINDOW BUILDER (needed for the next phase: 14-day windows, 7-day step) ---

def make_windows(df: pd.DataFrame, win_days=14, step_days=7):
    if df.empty:
        return []
    start = df["date"].min().normalize()
    end   = df["date"].max().normalize()

    windows = []
    cur = start
    while cur + pd.Timedelta(days=win_days) <= end + pd.Timedelta(days=1):
        windows.append((cur, cur + pd.Timedelta(days=win_days)))
        cur += pd.Timedelta(days=step_days)
    return windows

# --- MAIN: just load + verify dates + create windows (no BERTopic re-training here) ---

if __name__ == "__main__":
    DATA_PATHS = {
        "UK_Politics": r"C:\Users\olete\OneDrive\Desktop\ihzur\british_debates_text_files_normalize",
        "UK_Media":    r"C:\Users\olete\OneDrive\Desktop\ihzur\BBC_News",
        "US_Politics": r"C:\Users\olete\OneDrive\Desktop\ihzur\US_congressional_speeches_Text_Files",
        "US_Media":    r"C:\Users\olete\OneDrive\Desktop\ihzur\NBC_News",
    }

    uk_pol = load_df_from_folder(DATA_PATHS["UK_Politics"], "UK",  "politics")
    uk_med = load_df_from_folder(DATA_PATHS["UK_Media"],    "UK",  "media")
    us_pol = load_df_from_folder(DATA_PATHS["US_Politics"], "USA", "politics")
    us_med = load_df_from_folder(DATA_PATHS["US_Media"],    "USA", "media")

    def quick_check(df, name):
        print(f"\n{name}: rows={len(df)}")
        if df.empty:
            return
        print("sample:", df.loc[:2, ["file","date"]].to_string(index=False))
        print("min/max:", df["date"].min(), "->", df["date"].max())
        print("windows(14d step7d):", len(make_windows(df)))

    quick_check(uk_pol, "UK Politics (debatesYYYY-MM-DD)")
    quick_check(uk_med, "UK Media (GMT format)")
    quick_check(us_pol, "US Politics (YYYY-MM-DD)")
    quick_check(us_med, "US Media (GMT format)")
    # --- SAVE PREPARED DATAFRAMES (ONE-TIME PREPROCESSING OUTPUT) ---

    OUTPUT_DIR = "prepared_corpora"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    uk_pol.to_csv(f"{OUTPUT_DIR}/uk_politics_prepared.csv", index=False)
    uk_med.to_csv(f"{OUTPUT_DIR}/uk_media_prepared.csv", index=False)
    us_pol.to_csv(f"{OUTPUT_DIR}/us_politics_prepared.csv", index=False)
    us_med.to_csv(f"{OUTPUT_DIR}/us_media_prepared.csv", index=False)

    print("\nPrepared corpora saved:")
    print(" - prepared_corpora/uk_politics_prepared.csv")
    print(" - prepared_corpora/uk_media_prepared.csv")
    print(" - prepared_corpora/us_politics_prepared.csv")
    print(" - prepared_corpora/us_media_prepared.csv")
