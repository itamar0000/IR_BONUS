import os
import re
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already present
nltk.download('stopwords')

# --- CONFIGURATION ---
# These paths match the ones you provided in your code snippet
DATA_PATHS = {
    'UK_Politics': r"C:\Users\olete\OneDrive\Desktop\ihzur\british_debates_text_files_normalize",
    'UK_Media': r"C:\Users\olete\OneDrive\Desktop\ihzur\BBC_News",
    'US_Politics': r"C:\Users\olete\OneDrive\Desktop\ihzur\US_congressional_speeches_Text_Files",
    'US_Media': r"C:\Users\olete\OneDrive\Desktop\ihzur\NBC_News"
}


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


def load_data_from_folder(folder_path, label):
    """Loads all .txt files from a folder into a list."""
    texts = []
    print(f"Loading files from {folder_path}...")

    # Check if path exists first to avoid crashing
    if not os.path.exists(folder_path):
        print(f"⚠️ ERROR: Folder not found: {folder_path}")
        return texts

    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_full_path = os.path.join(folder_path, filename)
                try:
                    with open(file_full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_content = f.read()
                        cleaned_content = clean_text(raw_content)
                        if len(cleaned_content) > 50:  # Skip empty/tiny files
                            texts.append(cleaned_content)
                except Exception as e:
                    print(f"Skipping file {filename} due to error: {e}")
    except Exception as e:
        print(f"Error accessing folder: {e}")

    return texts


def run_bertopic(docs, country_name):
    """
    Runs BERTopic to find exactly 20 topics.
    """
    print(f"\n--- Processing {country_name} Corpus ({len(docs)} documents) ---")

    if len(docs) < 20:
        print("⚠️ Not enough documents to run BERTopic properly. Check data loading.")
        return None, None

    # We use a custom CountVectorizer to remove English stopwords
    vectorizer_model = CountVectorizer(stop_words="english")

    # Initialize BERTopic
    # nr_topics=20 forces the model to merge topics down to exactly 20
    topic_model = BERTopic(
        language="english",
        vectorizer_model=vectorizer_model,
        nr_topics=20,
        verbose=True
    )

    # Fit the model
    topics, probs = topic_model.fit_transform(docs)

    # Get Topic Info
    topic_info = topic_model.get_topic_info()

    # Save the model for later use (Crucial for Phase 2!)
    model_filename = f"{country_name}_bertopic_model"
    try:
        topic_model.save(model_filename)
        print(f"Model saved as {model_filename}")
    except Exception as e:
        print(f"Could not save model: {e}")

    # Export topic list to CSV
    output_csv = f"{country_name}_20_topics.csv"
    topic_info.to_csv(output_csv, index=False)
    print(f"Topics exported to {output_csv}")

    return topic_model, topic_info


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data using the UNIFIED keys defined in DATA_PATHS
    uk_pol = load_data_from_folder(DATA_PATHS['UK_Politics'], "UK_Pol")
    uk_med = load_data_from_folder(DATA_PATHS['UK_Media'], "UK_Media")
    us_pol = load_data_from_folder(DATA_PATHS['US_Politics'], "US_Pol")
    us_med = load_data_from_folder(DATA_PATHS['US_Media'], "US_Media")

    # 2. Merge by Country
    uk_corpus = uk_pol + uk_med
    us_corpus = us_pol + us_med

    # 3. Run Analysis (only if data exists)
    if uk_corpus:
        run_bertopic(uk_corpus, "UK")
    else:
        print("⚠️ No UK data found. Check your folder paths.")

    if us_corpus:
        run_bertopic(us_corpus, "USA")
    else:
        print("⚠️ No US data found. Check your folder paths.")