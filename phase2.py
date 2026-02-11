import os
import pandas as pd
from bertopic import BERTopic
from pandas import Timedelta

def safe_top_words(model: BERTopic, topic_id: int, k: int = 4) -> str:
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
        topic_ids = list(range(20))  # 0..19
    assigned, _ = topic_model.transform(texts)

    counts = {t: 0 for t in topic_ids}
    for t in assigned:
        if t in counts:
            counts[t] += 1

    total = sum(counts.values())
    if total == 0:
        return {t: 0.0 for t in topic_ids}
    return {t: counts[t] / total for t in topic_ids}

def run_method1_all_windows(df: pd.DataFrame, topic_model: BERTopic, country: str, arena: str, out_csv: str):
    df = df.sort_values("date").reset_index(drop=True)
    windows = make_windows(df, 14, 7)

    rows = []
    topic_ids = list(range(20))

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

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} | windows={len(windows)} | rows={len(out)}")

if __name__ == "__main__":
    # read prepared corpora you already created
    uk_pol = pd.read_csv("prepared_corpora/uk_politics_prepared.csv", parse_dates=["date"])
    uk_med = pd.read_csv("prepared_corpora/uk_media_prepared.csv", parse_dates=["date"])
    us_pol = pd.read_csv("prepared_corpora/us_politics_prepared.csv", parse_dates=["date"])
    us_med = pd.read_csv("prepared_corpora/us_media_prepared.csv", parse_dates=["date"])

    # load saved topic models you already trained
    uk_model = BERTopic.load("UK_bertopic_model")
    us_model = BERTopic.load("USA_bertopic_model")

    os.makedirs("phaseB_method1", exist_ok=True)

    run_method1_all_windows(uk_pol, uk_model, "UK", "politics", "phaseB_method1/uk_politics_method1.csv")
    run_method1_all_windows(uk_med, uk_model, "UK", "media",   "phaseB_method1/uk_media_method1.csv")

    run_method1_all_windows(us_pol, us_model, "USA", "politics", "phaseB_method1/us_politics_method1.csv")
    run_method1_all_windows(us_med, us_model, "USA", "media",   "phaseB_method1/us_media_method1.csv")
