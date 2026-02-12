import os
import pandas as pd
from bertopic import BERTopic
from pandas import Timedelta
from sentence_transformers import SentenceTransformer
import torch

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\mordih\Desktop\ihzur"
MODEL_DIR = os.path.join(BASE_DIR, "bertopic_results")
CSV_FOLDER = os.path.join(BASE_DIR, "prepared_corpora")
OUTPUT_DIR = os.path.join(BASE_DIR, "phaseB_method1")

# Use the same model name as Phase 1
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'


def safe_top_words(model: BERTopic, topic_id: int, k: int = 4) -> str:
    if topic_id == -1:
        return "Outliers"
    words = model.get_topic(topic_id)
    if not words:
        return "N/A"
    return ", ".join([w for w, _ in words[:k]])


def make_windows(df: pd.DataFrame, win_days: int = 14, step_days: int = 7):
    if df.empty:
        return []
    start = df["date"].min().normalize()
    end = df["date"].max().normalize()

    windows = []
    cur = start
    while cur + Timedelta(days=win_days) <= end + Timedelta(days=1):
        windows.append((cur, cur + Timedelta(days=win_days)))
        cur += Timedelta(days=step_days)
    return windows


def dominance_for_texts(topic_model: BERTopic, texts: list[str], topic_ids=None) -> dict[int, float]:
    if topic_ids is None:
        topic_ids = list(range(20))

    # This is the line that was failing because it didn't have the embedding model
    assigned, _ = topic_model.transform(texts)

    counts = {t: 0 for t in topic_ids}
    for t in assigned:
        if t in counts:
            counts[t] += 1

    total = len(assigned)
    if total == 0:
        return {t: 0.0 for t in topic_ids}
    return {t: counts[t] / total for t in topic_ids}


def run_method1_all_windows(df: pd.DataFrame, topic_model: BERTopic, country: str, arena: str, out_csv_name: str):
    df = df.sort_values("date").reset_index(drop=True)
    windows = make_windows(df, 14, 7)

    rows = []
    topic_ids = list(range(20))

    print(f"Processing {country} {arena} | Documents: {len(df)}")
    for w0, w1 in windows:
        wdf = df[(df["date"] >= w0) & (df["date"] < w1)]
        if wdf.empty:
            continue

        dom = dominance_for_texts(topic_model, wdf["text"].tolist(), topic_ids=topic_ids)

        for tid in topic_ids:
            rows.append({
                "country": country,
                "arena": arena,
                "window_start": w0,
                "window_end": w1,
                "topic_id": tid,
                "topic_label": safe_top_words(topic_model, tid, k=4),
                "dominance": round(dom.get(tid, 0.0), 6)
            })

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    out_path = os.path.join(OUTPUT_DIR, out_csv_name)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"✅ Saved {out_csv_name} | Windows processed: {len(windows)}")


if __name__ == "__main__":
    # 1. Setup Device and Embedding Model (Crucial for .transform)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    # 2. Load prepared corpora
    uk_pol = pd.read_csv(os.path.join(CSV_FOLDER, "uk_politics_prepared.csv"), parse_dates=["date"])
    uk_med = pd.read_csv(os.path.join(CSV_FOLDER, "uk_media_prepared.csv"), parse_dates=["date"])
    us_pol = pd.read_csv(os.path.join(CSV_FOLDER, "us_politics_prepared.csv"), parse_dates=["date"])
    us_med = pd.read_csv(os.path.join(CSV_FOLDER, "us_media_prepared.csv"), parse_dates=["date"])

    # 3. Load saved topic models AND link the embedding model
    print("Loading BERTopic models...")
    uk_model_path = os.path.join(MODEL_DIR, "UK_model")
    us_model_path = os.path.join(MODEL_DIR, "USA_model")

    uk_model = BERTopic.load(uk_model_path, embedding_model=embedding_model)
    us_model = BERTopic.load(us_model_path, embedding_model=embedding_model)

    # 4. Run Analysis
    run_method1_all_windows(uk_pol, uk_model, "UK", "politics", "uk_politics_method1.csv")
    run_method1_all_windows(uk_med, uk_model, "UK", "media", "uk_media_method1.csv")

    run_method1_all_windows(us_pol, us_model, "USA", "politics", "us_politics_method1.csv")
    run_method1_all_windows(us_med, us_model, "USA", "media", "us_media_method1.csv")