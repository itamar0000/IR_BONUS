import os
import re
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import ftfy
import html
import nltk
import torch

# Download stopwords if not already present
nltk.download('stopwords')

# --- CONFIGURATION ---
# Using the specific file paths you provided
DATA_PATHS = {
    'UK_Politics': r"C:\Users\mordih\Desktop\ihzur\prepared_corpora\uk_politics_prepared.csv",
    'UK_Media': r"C:\Users\mordih\Desktop\ihzur\prepared_corpora\uk_media_prepared.csv",
    'US_Politics': r"C:\Users\mordih\Desktop\ihzur\prepared_corpora\us_politics_prepared.csv",
    'US_Media': r"C:\Users\mordih\Desktop\ihzur\prepared_corpora\us_media_prepared.csv"
}

# Define where you want the results to go
OUTPUT_DIR = "bertopic_results"


def clean_text(text):
    if not isinstance(text, str): return ""
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = text.replace('\\', '')
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('Extensions of Remarks', '')
    text = text.replace('Congressional Record Vol.', '')
    if "This Cookie Notice" in text: text = text.split("This Cookie Notice")[0]
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data_from_csv(file_path):
    """Loads text from the 'text' column of a CSV file."""
    texts = []
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        print(f"⚠️ ERROR: File not found: {file_path}")
        return texts

    try:
        # Read the CSV
        df = pd.read_csv(file_path)

        if 'text' not in df.columns:
            print(f"⚠️ ERROR: Column 'text' not found in {file_path}")
            return texts

        # Extract, clean, and filter
        # We drop empty rows and only keep texts longer than 50 characters
        raw_list = df['text'].dropna().astype(str).tolist()
        for raw_content in raw_list:
            cleaned_content = clean_text(raw_content)
            if len(cleaned_content) > 50:
                texts.append(cleaned_content)

        print(f"✅ Loaded {len(texts)} documents.")
    except Exception as e:
        print(f"Error reading CSV: {e}")

    return texts


def run_bertopic(docs, country_name):
    print(f"\n--- Processing {country_name} Corpus ({len(docs)} documents) ---")

    if len(docs) < 20:
        print("⚠️ Not enough documents.")
        return None, None

    # --- 1. SETUP GPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Computation Device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

    # --- 2. CONFIGURE MODELS ---
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=15,
        ngram_range=(1, 2)
    )

    # Embedding Model -> FORCE GPU HERE
    embedding_model = SentenceTransformer('all-mpnet-base-v2', device=device)

    # --- 3. INITIALIZE & RUN BERTopic ---
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        nr_topics=20,
        verbose=True
    )

    # Fit the model
    topics, probs = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()

    # --- 4. SAVE RESULTS ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    csv_filename = os.path.join(OUTPUT_DIR, f"{country_name}_topics.csv")
    topic_info.to_csv(csv_filename, index=False)
    print(f"✅ Saved Topic Info CSV to: {csv_filename}")

    model_path = os.path.join(OUTPUT_DIR, f"{country_name}_model")
    topic_model.save(model_path, serialization="safetensors", save_ctfidf=True)
    print(f"✅ Saved BERTopic Model to: {model_path}")

    return topic_model, topic_info


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data using the NEW CSV loader
    uk_pol = load_data_from_csv(DATA_PATHS['UK_Politics'])
    uk_med = load_data_from_csv(DATA_PATHS['UK_Media'])
    us_pol = load_data_from_csv(DATA_PATHS['US_Politics'])
    us_med = load_data_from_csv(DATA_PATHS['US_Media'])

    # 2. Merge by Country
    uk_corpus = uk_pol + uk_med
    us_corpus = us_pol + us_med

    # 3. Run Analysis
    if uk_corpus:
        run_bertopic(uk_corpus, "UK")
    else:
        print("⚠️ No UK data found. Check your CSV files.")

    if us_corpus:
        run_bertopic(us_corpus, "USA")
    else:
        print("⚠️ No US data found. Check your CSV files.")