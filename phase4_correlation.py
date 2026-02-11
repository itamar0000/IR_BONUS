import pandas as pd
import numpy as np

MAX_LAG = 3
TOLERANCE_DAYS = 3
STD_EPS = 1e-8

# -------------------------
# Helpers: Load + Align
# -------------------------
def load_method1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["window_start", "window_end"])
    df = df[df["topic_id"].between(0, 19)].copy()
    return df

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
    P = topic_series(pol, tid)  # politics
    M = topic_series(med, tid)  # media

    p_list, m_list = [], []
    for _, r in aligned.iterrows():
        pw = r["pol_window_start"]
        mw = r["med_window_start"]
        if pw in P.index and mw in M.index:
            p_list.append(float(P.loc[pw]))
            m_list.append(float(M.loc[mw]))

    return np.array(m_list, dtype=float), np.array(p_list, dtype=float)  # (media, politics)

# -------------------------
# Similarities required
# -------------------------
def pearson_corr(x, y):
    if len(x) < 5:
        return np.nan
    if np.std(x) < STD_EPS or np.std(y) < STD_EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def best_lag_correlation(media, politics, max_lag=3):
    best = None  # (abs_corr, lag, corr)
    all_lags = {}

    for lag in range(-max_lag, max_lag + 1):
        # compare politics(t) vs media(t+lag)
        if lag > 0:
            p = politics[:-lag]
            m = media[lag:]
        elif lag < 0:
            p = politics[-lag:]
            m = media[:lag]
        else:
            p = politics
            m = media

        corr = pearson_corr(p, m)
        all_lags[lag] = corr

        if not np.isnan(corr):
            cand = (abs(corr), lag, corr)
            if (best is None) or (cand[0] > best[0]):
                best = cand

    if best is None:
        return None, np.nan, all_lags
    return best[1], best[2], all_lags  # best_lag, best_corr

# --- Real-valued "Edit distance" (DP on sequences of floats) ---
# Cost: substitution = abs(a-b), insertion=deletion=gap
def real_edit_distance(a, b, gap=0.05):
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=float)
    dp[:, 0] = np.arange(n + 1) * gap
    dp[0, :] = np.arange(m + 1) * gap

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            sub = dp[i - 1, j - 1] + abs(ai - bj)
            ins = dp[i, j - 1] + gap
            dele = dp[i - 1, j] + gap
            dp[i, j] = min(sub, ins, dele)

    return float(dp[n, m])

def best_lag_real_edit_similarity(media, politics, max_lag=3, gap=0.05):
    # Convert distance -> similarity in [0,1] using normalization
    best = None  # (sim, lag, dist)
    all_lags = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            p = politics[:-lag]
            m = media[lag:]
        elif lag < 0:
            p = politics[-lag:]
            m = media[:lag]
        else:
            p = politics
            m = media

        if len(p) < 5:
            all_lags[lag] = np.nan
            continue

        dist = real_edit_distance(m, p, gap=gap)
        # normalize by worst-case: gap * length + max possible substitution (<=1 per step because dominance in [0,1])
        norm = gap * (len(m) + len(p)) + 1.0 * min(len(m), len(p))
        sim = 1.0 - (dist / norm)
        all_lags[lag] = sim

        if best is None or sim > best[0]:
            best = (sim, lag, dist)

    if best is None:
        return None, np.nan, all_lags
    return best[1], best[0], all_lags  # best_lag, best_sim

# --- DTW (Dynamic Time Warping) ---
def dtw_distance(x, y):
    n, m = len(x), len(y)
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            yj = y[j - 1]
            cost = abs(xi - yj)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[n, m])

def dtw_similarity(media, politics):
    if len(media) < 5 or len(politics) < 5:
        return np.nan
    dist = dtw_distance(media, politics)
    # normalize similarly
    norm = 1.0 * min(len(media), len(politics)) + 0.05 * (len(media) + len(politics))
    return 1.0 - (dist / norm)

# -------------------------
# Direction change detection
# -------------------------
def split_direction(media, politics):
    mid = len(media) // 2
    if mid < 10:
        return None, None  # too short

    lag1, corr1, _ = best_lag_correlation(media[:mid], politics[:mid], MAX_LAG)
    lag2, corr2, _ = best_lag_correlation(media[mid:], politics[mid:], MAX_LAG)

    def leader_from_lag(lag):
        if lag is None or np.isnan(lag):
            return "unknown"
        if lag > 0: return "politics → media"
        if lag < 0: return "media → politics"
        return "sync"

    return leader_from_lag(lag1), leader_from_lag(lag2)

# -------------------------
# Main analysis
# -------------------------
def analyze_country(country, media_path, politics_path, out_csv):
    med = load_method1(media_path)
    pol = load_method1(politics_path)

    pol, med, start, end = overlap_filter(pol, med)
    aligned = build_window_alignment(pol, med)

    print(f"\n{country} overlap: {start.date()} -> {end.date()}")
    print(f"{country} aligned pairs (±{TOLERANCE_DAYS}d): {len(aligned)}")

    rows = []
    for tid in range(20):
        topic_label = None
        if "topic_label" in med.columns:
            lab = med.loc[med["topic_id"] == tid, "topic_label"]
            if len(lab) > 0:
                topic_label = str(lab.iloc[0])

        media_vals, politics_vals = aligned_vectors(pol, med, aligned, tid)
        if len(media_vals) < 15:
            continue

        # Correlation & lag
        best_lag_c, best_corr, _ = best_lag_correlation(media_vals, politics_vals, MAX_LAG)

        # Real-edit similarity & lag
        best_lag_e, best_sim_e, _ = best_lag_real_edit_similarity(media_vals, politics_vals, MAX_LAG, gap=0.05)

        # DTW similarity (no lag)
        sim_dtw = dtw_similarity(media_vals, politics_vals)

        # dominant direction from correlation lag
        if best_lag_c is None or np.isnan(best_corr):
            direction = "unknown"
        elif best_lag_c > 0:
            direction = "politics → media"
        elif best_lag_c < 0:
            direction = "media → politics"
        else:
            direction = "sync"

        # Direction change
        first_dir, second_dir = split_direction(media_vals, politics_vals)
        changing = (first_dir is not None and second_dir is not None and first_dir != second_dir)

        rows.append({
            "country": country,
            "topic_id": tid,
            "topic_label": topic_label if topic_label else "",
            "matched_points": len(media_vals),

            "best_lag_corr": best_lag_c if best_lag_c is not None else "",
            "best_corr": round(best_corr, 3) if not np.isnan(best_corr) else "",

            "best_lag_edit": best_lag_e if best_lag_e is not None else "",
            "best_sim_edit": round(best_sim_e, 3) if not np.isnan(best_sim_e) else "",

            "sim_dtw": round(sim_dtw, 3) if not np.isnan(sim_dtw) else "",

            "direction": direction,
            "direction_first_half": first_dir if first_dir else "",
            "direction_second_half": second_dir if second_dir else "",
            "direction_changes": bool(changing) if first_dir else ""
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv, "| rows:", len(out))
    return out

if __name__ == "__main__":
    analyze_country("UK",
        "phaseB_method1/uk_media_method1.csv",
        "phaseB_method1/uk_politics_method1.csv",
        "UK_stageD_results.csv"
    )

    analyze_country("USA",
        "phaseB_method1/us_media_method1.csv",
        "phaseB_method1/us_politics_method1.csv",
        "USA_stageD_results.csv"
    )
