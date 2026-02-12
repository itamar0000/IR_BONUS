import os
import uuid
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb

# -------------------------
# Config
# -------------------------
EMBED_MODEL = "all-mpnet-base-v2"  # same family as your BERTopic embedding model :contentReference[oaicite:5]{index=5}
CHROMA_DIR = "chroma_store"
CHUNK_SIZE = 900      # chars (simple + stable)
CHUNK_OVERLAP = 150   # chars
MIN_CHUNK_CHARS = 200

INPUTS = [
    ("UK",  "politics", "prepared_corpora/uk_politics_prepared.csv"),
    ("UK",  "media",    "prepared_corpora/uk_media_prepared.csv"),
    ("USA", "politics", "prepared_corpora/us_politics_prepared.csv"),
    ("USA", "media",    "prepared_corpora/us_media_prepared.csv"),
]

# -------------------------
# Chunker (simple, robust)
# -------------------------
def chunk_text(text: str, size: int, overlap: int):
    if not isinstance(text, str):
        return []
    text = text.strip()
    if len(text) < MIN_CHUNK_CHARS:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name="news_politics_chunks")

    embedder = SentenceTransformer(EMBED_MODEL)

    total_added = 0
    for country, arena, csv_path in INPUTS:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.dropna(subset=["date", "text"]).reset_index(drop=True)

        # Create stable doc id
        if "file" in df.columns:
            doc_ids = df["file"].astype(str).tolist()
        else:
            doc_ids = [f"{country}_{arena}_{i}" for i in range(len(df))]

        print(f"\nIndexing: {country} {arena} | docs={len(df)} | {csv_path}")

        for i, row in df.iterrows():
            doc_id = doc_ids[i]
            dt = row["date"]
            text = str(row["text"])

            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                continue

            # Embed in small batches (fast + safe)
            embs = embedder.encode(chunks, normalize_embeddings=True).tolist()

            ids = []
            metadatas = []
            for j, _ in enumerate(chunks):
                ids.append(str(uuid.uuid4()))
                metadatas.append({
                    "country": country,
                    "arena": arena,
                    "date": dt.strftime("%Y-%m-%d"),
                    "doc_id": doc_id,
                    "chunk_id": j
                })

            col.add(
                ids=ids,
                documents=chunks,
                embeddings=embs,
                metadatas=metadatas
            )
            total_added += len(chunks)

    print(f"\n✅ Done. Total chunks added: {total_added}")
    print(f"Chroma DB at: {CHROMA_DIR}")

if __name__ == "__main__":
    main()