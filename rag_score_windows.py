# rag_score_windows.py
# Windowed RAG topic-dominance scoring using:
# - ChromaDB (vector retrieval)
# - BERTopic (topic labels/keywords)
# - Ollama (LLM scorer)
#
# Fixes included:
# 1) Chroma "where" uses $and operator (new Chroma validation)
# 2) Over-fetch then local date filter to avoid empty evidence
# 3) Robust JSON extraction + repair + normalization to sum=100 (no crashes)
# 4) ROBUST CSV LOADING: Auto-detects date columns (case-insensitive, ignores spaces)

import os
import re
import json
import uuid
import subprocess
from datetime import datetime

import pandas as pd
from pandas import Timedelta

import chromadb
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

# -------------------------
# Config (adjust paths if needed)
# -------------------------
CHROMA_DIR = "chroma_store"
COLLECTION = "news_politics_chunks"

MODEL_DIR = "bertopic_results"
UK_MODEL_PATH = f"{MODEL_DIR}/UK_model"
US_MODEL_PATH = f"{MODEL_DIR}/USA_model"

# Keep this consistent with how you built/loaded BERTopic in your pipeline
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Ollama
OLLAMA_MODEL = "llama3.1"  # change to what you have installed: `ollama list`

# Retrieval & context limits
TOP_K = 6  # keep per-topic evidence size
OVERFETCH_MULT = 30  # over-fetch candidates then filter by window (stability)
MIN_OVERFETCH = 80  # minimum candidates to fetch
MAX_CHUNKS_TOTAL = 60  # cap overall evidence in prompt (token safety)
MIN_CHUNK_CHARS = 200  # ignore tiny chunks if they exist

# Sliding window parameters (14 days window, step 7 days)
WINDOW_DAYS = 14
STEP_DAYS = 7

# Inputs (prepared corpora)
INPUTS = [
    ("UK", "media", "prepared_corpora/uk_media_prepared.csv", "phaseC_rag/uk_media_rag.csv"),
    ("USA", "politics", "prepared_corpora/us_politics_prepared.csv", "phaseC_rag/us_politics_rag.csv"),
    ("USA", "media", "prepared_corpora/us_media_prepared.csv", "phaseC_rag/us_media_rag.csv"),
]


# -------------------------
# Windows
# -------------------------
def make_windows(df: pd.DataFrame, win_days: int = WINDOW_DAYS, step_days: int = STEP_DAYS):
    if df.empty:
        return []

    # Ensure date column is strictly datetime (handle mixed formats)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return []

    start = df["date"].min().normalize()
    end = df["date"].max().normalize()
    windows = []
    cur = start
    # inclusive start, exclusive end
    while cur + Timedelta(days=win_days) <= end + Timedelta(days=1):
        windows.append((cur, cur + Timedelta(days=win_days)))
        cur += Timedelta(days=step_days)
    return windows


# -------------------------
# Topic label helpers
# -------------------------
def safe_top_words(model: BERTopic, topic_id: int, k: int = 6) -> str:
    if topic_id == -1:
        return "Outliers"
    words = model.get_topic(topic_id)
    if not words:
        return "N/A"
    return ", ".join([w for w, _ in words[:k]])


# -------------------------
# Ollama call
# -------------------------
def ollama_chat(prompt: str) -> str:
    """
    Simple CLI-based call:
      ollama run <model>
    """
    res = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    out = res.stdout.decode("utf-8", errors="ignore").strip()
    if not out:
        err = res.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Ollama returned empty output. stderr:\n{err}")
    return out


# -------------------------
# Retrieval (Chroma + local window filter)
# -------------------------
def retrieve_for_topic(col, embedder, country, arena, w0, w1, topic_query, top_k: int):
    """
    1) Embed the topic query
    2) Query Chroma with metadata filter (country+arena)
    3) Over-fetch, then locally filter by window date
    4) Return up to top_k chunks
    """
    q_emb = embedder.encode([topic_query], normalize_embeddings=True).tolist()[0]

    n_fetch = max(MIN_OVERFETCH, top_k * OVERFETCH_MULT)

    # IMPORTANT: Chroma requires exactly one operator at top-level in `where`
    where_filter = {
        "$and": [
            {"country": {"$eq": country}},
            {"arena": {"$eq": arena}},
        ]
    }

    res = col.query(
        query_embeddings=[q_emb],
        n_results=n_fetch,
        where=where_filter,
        include=["documents", "metadatas"]
    )

    docs = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []

    # Filter by date window locally (robust across Chroma versions)
    candidates = []
    for d, m in zip(docs, metas):
        if not isinstance(d, str):
            continue
        if len(d.strip()) < MIN_CHUNK_CHARS:
            continue

        dt_str = m.get("date")
        if not dt_str:
            continue
        dt = pd.to_datetime(dt_str, errors="coerce")
        if pd.isna(dt):
            continue

        if (dt >= w0) and (dt < w1):
            candidates.append((d.strip(), m))

        if len(candidates) >= top_k:
            break

    return candidates


# -------------------------
# Prompt
# -------------------------
def build_prompt(country, arena, w0, w1, topics, evidence_by_topic):
    """
    Force JSON-only output. Still, we repair JSON after.
    """
    topic_lines = []
    for tid, label, query in topics:
        topic_lines.append(f"{tid}: {label}")

    # Evidence block (bounded)
    ev_lines = []
    used = 0
    for tid, _, _ in topics:
        evs = evidence_by_topic.get(tid, [])
        if not evs:
            continue
        ev_lines.append(f"\n[T{tid}]")
        for e in evs:
            if used >= MAX_CHUNKS_TOTAL:
                break
            ev_lines.append(f"- {e}")
            used += 1
        if used >= MAX_CHUNKS_TOTAL:
            break

    return f"""
You are scoring topic dominance for a time window based ONLY on the evidence.

Country: {country}
Arena: {arena}
Window: {w0.date()} to {w1.date()} (14-day window)

Topics (0-19):
{chr(10).join(topic_lines)}

Evidence (retrieved chunks):
{chr(10).join(ev_lines)}

Task:
Return a JSON object with exactly these keys:
- "scores": an object mapping topic ids "0".."18" to integer percentages.

Rules:
- Provide ONLY topics 0..18 (do NOT include 19).
- Values must be integers (no decimals).
- The sum of 0..18 must be <= 100.
- Use evidence only. If evidence is weak for a topic, give it a low score.
- Return STRICT JSON with double quotes only, no trailing commas, no extra text.

Return ONLY valid JSON, nothing else.
""".strip()


# -------------------------
# Robust JSON parsing + repair + normalization
# -------------------------
def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON block found in model output.")
    return text[start:end + 1]


def _repair_json(s: str) -> str:
    """
    Repair common LLM JSON mistakes:
    - code fences
    - single quotes -> double quotes
    - unquoted word keys
    - unquoted numeric keys (0: 13 -> "0": 13)
    - trailing commas
    """
    import re

    s = s.strip()

    # Remove code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Convert single quotes to double quotes
    s = s.replace("'", '"')

    # Quote unquoted word keys: {scores: ...} -> {"scores": ...}
    s = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):', r'\1"\2"\3:', s)

    # Quote unquoted numeric keys: {0: 13, 1: 6} -> {"0": 13, "1": 6}
    s = re.sub(r'([{,]\s*)(\d+)(\s*):', r'\1"\2"\3:', s)

    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s


def parse_scores(text: str):
    raw = _extract_json_block(text)
    raw = _repair_json(raw)

    obj = json.loads(raw)
    scores = obj.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Parsed JSON but missing 'scores' dict.")

    out = {}
    for k, v in scores.items():
        m = re.search(r"\d+", str(k))
        if not m:
            continue
        tid = int(m.group(0))
        if tid < 0 or tid > 18:
            continue
        try:
            val = int(round(float(v)))
        except:
            val = 0
        out[tid] = max(0, val)

    # ensure 0..18 exist
    for tid in range(19):
        out.setdefault(tid, 0)

    s = sum(out.values())
    if s > 100:
        # normalize 0..18 down to 100 and set 19 = 0
        scaled = {tid: int(round(out[tid] * 100.0 / s)) for tid in range(19)}
        diff = 100 - sum(scaled.values())
        # fix rounding drift
        order = sorted(range(19), key=lambda t: out[t], reverse=True)
        i = 0
        while diff != 0:
            t = order[i % 19]
            if diff > 0:
                scaled[t] += 1
                diff -= 1
            else:
                if scaled[t] > 0:
                    scaled[t] -= 1
                    diff += 1
            i += 1
        out = scaled
        s = 100

    # compute topic 19 deterministically
    topic19 = 100 - s

    final = {tid: out[tid] for tid in range(19)}
    final[19] = topic19
    return final


# -------------------------
# Helper: Robust CSV Loading
# -------------------------
def load_and_prep_csv(csv_path):
    """
    Robustly loads CSV, finding the date column regardless of case/whitespace.
    Renames found column to 'date'.
    """
    print(f"DEBUG: Reading {csv_path}...")

    # 1. Read just header to inspect columns
    try:
        df_preview = pd.read_csv(csv_path, nrows=0)
    except Exception as e:
        print(f"Error reading CSV header: {e}")
        return pd.DataFrame()  # return empty on failure

    # 2. Normalize columns (strip whitespace, lowercase)
    # Mapping: clean_name -> original_name
    col_map = {c.strip().lower(): c for c in df_preview.columns}

    # 3. Find candidate date column
    # Priority: exact 'date' -> contains 'date' -> contains 'time' -> contains 'created'
    date_col_actual = None

    if "date" in col_map:
        date_col_actual = col_map["date"]
    else:
        # fuzzy search
        for clean, original in col_map.items():
            if "date" in clean:
                date_col_actual = original
                break
        if not date_col_actual:
            for clean, original in col_map.items():
                if "time" in clean or "created" in clean:
                    date_col_actual = original
                    break

    if not date_col_actual:
        print(f"ERROR: Could not find a date-like column in {list(df_preview.columns)}")
        return pd.DataFrame()

    print(f"DEBUG: Detected date column: '{date_col_actual}'")

    # 4. Load full csv with parse_dates
    try:
        df = pd.read_csv(csv_path, parse_dates=[date_col_actual])
    except Exception as e:
        # Fallback: load as string then convert
        print(f"Warning: parse_dates failed ({e}). Loading as string and converting...")
        df = pd.read_csv(csv_path)
        df[date_col_actual] = pd.to_datetime(df[date_col_actual], errors='coerce')

    # 5. Rename to 'date' standard
    df = df.rename(columns={date_col_actual: "date"})

    # 6. Drop rows with invalid dates or missing text
    # Ensure text column exists too (try 'text', 'content', 'body')
    text_col = None
    cols_lower = {c.lower(): c for c in df.columns}
    if 'text' in cols_lower:
        text_col = cols_lower['text']
    elif 'content' in cols_lower:
        text_col = cols_lower['content']
    elif 'body' in cols_lower:
        text_col = cols_lower['body']

    if text_col and text_col != 'text':
        df = df.rename(columns={text_col: "text"})

    if "text" not in df.columns:
        # Create dummy text if missing, just to avoid crash, though logic might fail later
        print("Warning: No 'text' column found. Adding empty string.")
        df["text"] = ""

    df = df.dropna(subset=["date", "text"]).reset_index(drop=True)
    return df


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs("phaseC_rag", exist_ok=True)

    # Embedder (also used for BERTopic load)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Load BERTopic models
    print("Loading BERTopic models...")
    uk_model = BERTopic.load(UK_MODEL_PATH, embedding_model=embedder)
    us_model = BERTopic.load(US_MODEL_PATH, embedding_model=embedder)

    # Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION)

    for country, arena, csv_path, out_path in INPUTS:
        # Use new robust loader
        df = load_and_prep_csv(csv_path)

        if df.empty:
            print(f"Skipping {country} {arena} (empty or invalid CSV).")
            continue

        model = uk_model if country == "UK" else us_model

        windows = make_windows(df, WINDOW_DAYS, STEP_DAYS)
        print(f"\n=== RAG Scoring {country} {arena} | windows={len(windows)} ===")

        # Pre-build topic descriptors (0..19)
        topics = []
        for tid in range(20):
            label = safe_top_words(model, tid, k=6)
            topics.append((tid, label, label))  # query uses label words

        rows = []
        for w0, w1 in windows:
            evidence_by_topic = {}
            for tid, label, query in topics:
                hits = retrieve_for_topic(col, embedder, country, arena, w0, w1, query, TOP_K)
                evidence_by_topic[tid] = [h[0] for h in hits]

            print(f"Scoring window {w0.date()} -> {w1.date()}")
            prompt = build_prompt(country, arena, w0, w1, topics, evidence_by_topic)
            try:
                raw = ollama_chat(prompt)
                scores = parse_scores(raw)
            except Exception as e:
                print("!! ERROR in window:", w0.date(), "->", w1.date())
                print("!! Reason:", repr(e))
                # fallback: uniform
                scores = {i: 5 for i in range(20)}
                scores[0] += 100 - sum(scores.values())

            for tid in range(20):
                rows.append({
                    "country": country,
                    "arena": arena,
                    "window_start": w0,
                    "window_end": w1,
                    "topic_id": tid,
                    "topic_label": safe_top_words(model, tid, k=4),
                    "dominance": round(scores[tid] / 100.0, 6)
                })

        out = pd.DataFrame(rows)
        out.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path} | rows={len(out)}")

    print("\nAll done ✅")


if __name__ == "__main__":
    main()