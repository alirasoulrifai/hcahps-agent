import os
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
INPUT_FILE = r"c:\Users\USER\Documents\streamlit_app\Arabic Poem Comprehensive Dataset (APCD).csv"
OUTPUT_DIR = r"c:\Users\USER\Documents\streamlit_app\arabic_poet_index"
INDEX_PATH = os.path.join(OUTPUT_DIR, "poetry.faiss")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.parquet")

BATCH_SIZE = 1024 # Optimized for GPU
SAMPLE_SIZE = 100000 # Let's start with 100k for the first run

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Checking for GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: paraphrase-multilingual-MiniLM-L12-v2...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)

    print(f"Reading first {SAMPLE_SIZE} rows of the dataset...")
    df = pd.read_csv(INPUT_FILE, nrows=SAMPLE_SIZE)
    df = df.fillna("")

    # Combine metadata into the text for better retrieval
    print("Preparing text for embedding (fusing poet and verse)...")
    df['text_to_embed'] = "الشاعر: " + df['الشاعر'] + " | العصر: " + df['العصر'] + " | البيت: " + df['البيت']
    
    texts = df['text_to_embed'].tolist()
    all_embeddings = []

    print(f"Starting embedding of {len(texts)} verses...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(embeddings)

    embeddings_matrix = np.vstack(all_embeddings).astype('float32')
    dim = embeddings_matrix.shape[1]

    print(f"Creating FAISS index (Dimension: {dim})...")
    index = faiss.IndexFlatIP(dim) # Inner Product for cosine similarity with normalized vectors
    index.add(embeddings_matrix)

    print(f"Saving index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving metadata to {METADATA_PATH}...")
    # Saving specific columns to save space
    df[['العصر', 'الشاعر', 'الديوان', 'البحر', 'البيت']].to_parquet(METADATA_PATH)

    print("\n✅ Success! Sample index created.")
    print(f"Index size: {os.path.getsize(INDEX_PATH) / (1024*1024):.2f} MB")
    print(f"Metadata size: {os.path.getsize(METADATA_PATH) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
