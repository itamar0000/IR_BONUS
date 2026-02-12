import pandas as pd
import numpy as np

MAX_LAG = 3
TOLERANCE_DAYS = 3
STD_EPS = 1e-6  # if std smaller than this => treat as constant/invalid

def load_method1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["window_start", "window_end"])
    return df[df["topic_id"].between(0, 19)].copy()

def overlap_filter(pol: pd.DataFrame, med: pd.DataFrame):
    start = max(pol["window_start"].min(), med["window_start"].min())
    end   = min(pol["window_start"].max(), med["window_start"].max())
    pol2 = pol[(pol["window_start"] >= start) & (pol["window_start"] <= end)].copy()
    med2 = med[(med["window_start"] >= start) & (med["window_start"] <= end)].copy()
    return pol2, med2, start, end

def build_window_alignment(pol: pd.DataFrame, med: pd.DataFrame) -> pd.DataFrame:
    pol_w = pd.DataFrame({"pol_window_start": sorted(pol["window_start"].unique())}).sort_values("pol_window_start")
    med_w = pd.DataFrame({"med_window_start": sorted(med["window_start"].unique())}).sort_values("med_window_start")

    aligned = pd.merge_asof(
        pol_w,
        med_w,
        left_on="pol_window_start",
        right_on="med_window_start",
        direction="nearest",
        tolerance=pd.Timedelta(days=TOLERANCE_DAYS)
    ).dropna()

    return aligned

def topic_series(df: pd.DataFrame, tid: int) -> pd.Series:
    return df[df["topic_id"] == tid].set_index("window_start")["dominance"].sort_index()

def aligned_vectors(pol: pd.DataFrame, med: pd.DataFrame, aligned: pd.DataFrame, tid: int):
    P = topic_series(pol, tid)
    M = topic_series(med, tid)

    p_list, m_list = [], []
    for _, r in aligned.iterrows():
        pw = r["pol_window_start"]
        mw = r["med_window_start"]
        if pw in P.index and mw in M.index:
            p_list.append(float(P.loc[pw]))
            m_list.append(float(M.loc[mw]))

    return np.array(p_list), np.array(m_list)

def best_lag_by_corr(P_vals: np.ndarray, M_vals: np.ndarray, max_lag: int = 3):
    # variance check
    if np.std(P_vals) < STD_EPS or np.std(M_vals) < STD_EPS:
        return None, None

    results = {}
    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            p = P_vals[:-k]
            m = M_vals[k:]
        elif k < 0:
            p = P_vals[-k:]
            m = M_vals[:k]
        else:
            p = P_vals
            m = M_vals

        if len(p) < 10:
            continue

        corr = np.corrcoef(p, m)[0, 1]
        if np.isnan(corr):
            continue
        results[k] = corr

    if not results:
        return None, None

    best_lag = max(results, key=lambda kk: abs(results[kk]))
    return best_lag, results[best_lag]

def dominant_topics_with_variance(pol: pd.DataFrame, med: pd.DataFrame, aligned: pd.DataFrame, top_k=5):
    # dominance rank
    pol_mean = pol.groupby("topic_id")["dominance"].mean()
    med_mean = med.groupby("topic_id")["dominance"].mean()
    combined = ((pol_mean + med_mean) / 2).sort_values(ascending=False)

    chosen = []
    for tid in combined.index.tolist():
        tid = int(tid)
        P_vals, M_vals = aligned_vectors(pol, med, aligned, tid)
        if len(P_vals) < 10:
            continue
        if np.std(P_vals) < STD_EPS or np.std(M_vals) < STD_EPS:
            continue
        chosen.append(tid)
        if len(chosen) == top_k:
            break

    return chosen

def analyze_country(pol_path: str, med_path: str, country: str) -> pd.DataFrame:
    pol = load_method1(pol_path)
    med = load_method1(med_path)

    pol, med, start, end = overlap_filter(pol, med)
    print(f"\n{country} overlap: {start.date()} -> {end.date()}")
    print(f"{country} windows: pol={pol['window_start'].nunique()} med={med['window_start'].nunique()}")

    aligned = build_window_alignment(pol, med)
    print(f"{country} aligned pairs (±{TOLERANCE_DAYS}d): {len(aligned)}")

    top5 = dominant_topics_with_variance(pol, med, aligned, top_k=5)
    print(f"{country} Top-5 topics (dominant + variance): {top5}")

    rows = []
    for tid in top5:
        P_vals, M_vals = aligned_vectors(pol, med, aligned, tid)
        best_lag, best_corr = best_lag_by_corr(P_vals, M_vals, MAX_LAG)

        if best_lag is None:
            continue

        if best_lag > 0:
            leader = "politics → media"
        elif best_lag < 0:
            leader = "media → politics"
        else:
            leader = "synchronous"

        rows.append({
            "country": country,
            "topic_id": tid,
            "best_lag_windows": int(best_lag),
            "best_corr": round(float(best_corr), 3),
            "leader": leader,
            "matched_points": int(len(P_vals))
        })

    out = pd.DataFrame(rows)
    out.to_csv(f"{country.lower()}_lead_lag_results.csv", index=False)
    return out

if __name__ == "__main__":
    uk = analyze_country("phaseB_method1/uk_politics_method1.csv",
                         "phaseB_method1/uk_media_method1.csv", "UK")
    us = analyze_country("phaseB_method1/us_politics_method1.csv",
                         "phaseB_method1/us_media_method1.csv", "USA")

    print("\nUK results:\n", uk)
    print("\nUSA results:\n", us)
