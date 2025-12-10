import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

# --- Config ---
TRAIN_NEWS = 'train/train_news.tsv'
TEST_NEWS = 'test/test_news.tsv'
PROCESSED_PATH = 'processed'
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')
OUTPUT_MATRIX = os.path.join(PROCESSED_PATH, 'news_cat_subcat.npy')

def load_news_manual(file_path):
    """
    Robustly loads news ID, Category, and Subcategory, ignoring broken abstracts.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}"):
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            # We strictly need ID (0), Cat (1), Subcat (2)
            if len(parts) >= 3:
                nid = parts[0]
                cat = parts[1]
                subcat = parts[2]
                data.append({'news_id': nid, 'cat': cat, 'subcat': subcat})
    return pd.DataFrame(data)

def main():
    print("Loading ID Mapping...")
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: {VOCAB_PATH} not found. Run the global Roberta preprocess first.")
        return

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
        nid2idx = vocab['nid2idx'] # The global mapping
        
    print("Reading News (Manual Parse)...")
    # REPLACE pd.read_csv with manual loader
    df_train = load_news_manual(TRAIN_NEWS)
    df_test = load_news_manual(TEST_NEWS)
    
    print("Merging News...")
    all_news = pd.concat([df_train, df_test]).drop_duplicates(subset=['news_id'])
    
    # Build Category Vocabs
    print("Building Category Vocabs...")
    cats = all_news['cat'].unique()
    cat2idx = {k: v+1 for v, k in enumerate(cats)}
    cat2idx['PAD'] = 0
    
    subcats = all_news['subcat'].unique()
    subcat2idx = {k: v+1 for v, k in enumerate(subcats)}
    subcat2idx['PAD'] = 0
    
    # Save vocab for model size
    vocab['cat2idx'] = cat2idx
    vocab['subcat2idx'] = subcat2idx
    
    print(f"Updating {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
        
    # Build Matrix
    print("Building Feature Matrix...")
    # Matrix size: [Num_News + 1 (for PAD), 2 (Cat, Subcat)]
    # Use max index from nid2idx to determine size (safest way)
    max_idx = max(nid2idx.values())
    feature_matrix = np.zeros((max_idx + 1, 2), dtype=np.int32)
    
    hit_count = 0
    for _, row in tqdm(all_news.iterrows(), total=len(all_news), desc="Filling Matrix"):
        nid = row['news_id']
        if nid in nid2idx:
            idx = nid2idx[nid]
            feature_matrix[idx, 0] = cat2idx.get(row['cat'], 0)
            feature_matrix[idx, 1] = subcat2idx.get(row['subcat'], 0)
            hit_count += 1
            
    np.save(OUTPUT_MATRIX, feature_matrix)
    print(f"Saved {feature_matrix.shape} matrix to {OUTPUT_MATRIX}")
    print(f"Mapped {hit_count} news items.")
    print(f"Cat Vocab Size: {len(cat2idx)}, Subcat Vocab Size: {len(subcat2idx)}")

if __name__ == "__main__":
    main()