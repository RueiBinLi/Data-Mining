import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import pickle
import os

# --- Configuration ---
NEWS_TSV_PATH = 'train/train_news.tsv'
BEHAVIORS_TSV_PATH = 'train/train_behaviors.tsv'

PROCESSED_PATH = 'processed'
NEWS_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_features.npy') # The Matrix
USER_DATA_PATH = os.path.join(PROCESSED_PATH, 'user_data.pkl')       # The Behaviors
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')

MAX_TITLE_LEN = 30
MAX_HISTORY_LEN = 50

def main():
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    # ============================
    # 1. Process News -> Matrix
    # ============================
    print("1. Loading News...")
    news_df = pd.read_csv(
        NEWS_TSV_PATH, sep='\t', header=None,
        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
               'url', 'title_entities', 'abstract_entities']
    )
    news_df['title'] = news_df['title'].fillna('')

    print("2. Building Vocabs...")
    # Category Vocab
    cats = news_df['category'].unique()
    cat2idx = {k: v+1 for v, k in enumerate(cats)}
    cat2idx['PAD'] = 0
    
    # Subcategory Vocab
    subcats = news_df['subcategory'].unique()
    subcat2idx = {k: v+1 for v, k in enumerate(subcats)}
    subcat2idx['PAD'] = 0
    
    # Word Vocab
    word_counter = Counter()
    for title in tqdm(news_df['title']):
        word_counter.update(title.lower().split())
    
    word_vocab = [w for w, c in word_counter.items() if c > 1]
    word2idx = {w: i+2 for i, w in enumerate(word_vocab)}
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1

    # --- News ID Mapping ---
    # Map 'N123' -> 1, 'N456' -> 2
    news_ids = news_df['news_id'].values
    nid2idx = {nid: i+1 for i, nid in enumerate(news_ids)}
    nid2idx['PAD'] = 0
    
    # --- Build the News Feature Matrix ---
    # Shape: [Num_News + 1, 30 + 1 + 1] -> (Title + Cat + Subcat)
    # Row 0 is PAD
    print("3. Building News Matrix...")
    num_news = len(news_ids)
    # Matrix columns: 0-29 (Title), 30 (Cat), 31 (Subcat)
    feature_matrix = np.zeros((num_news + 1, MAX_TITLE_LEN + 2), dtype=np.int32)
    
    for i, row in tqdm(news_df.iterrows(), total=len(news_df)):
        idx = nid2idx[row['news_id']]
        
        # Categories
        feature_matrix[idx, MAX_TITLE_LEN] = cat2idx.get(row['category'], 0)
        feature_matrix[idx, MAX_TITLE_LEN+1] = subcat2idx.get(row['subcategory'], 0)
        
        # Title
        tokens = row['title'].lower().split()
        token_ids = [word2idx.get(w, 1) for w in tokens][:MAX_TITLE_LEN]
        feature_matrix[idx, :len(token_ids)] = token_ids

    # ============================
    # 2. Process Behaviors -> Int Arrays
    # ============================
    print("4. Processing Behaviors...")
    behaviors_df = pd.read_csv(
        BEHAVIORS_TSV_PATH, sep='\t', header=None,
        names=['id', 'user_id', 'time', 'clicked_news', 'impressions']
    )
    
    # Helper to map list of NIDs to list of Ints
    def map_history(hist_str):
        if pd.isna(hist_str): return [0] * MAX_HISTORY_LEN
        nids = hist_str.split()
        # Truncate/Pad
        nids = nids[-MAX_HISTORY_LEN:] # Keep recent
        pad_len = MAX_HISTORY_LEN - len(nids)
        
        # Map to Ints
        int_ids = [nid2idx.get(n, 0) for n in nids]
        return [0]*pad_len + int_ids

    # Process History
    print("   - Encoding Histories...")
    # This might take a minute but saves hours of training time
    tqdm.pandas()
    behaviors_df['history_indices'] = behaviors_df['clicked_news'].progress_apply(map_history)
    
    # Process Impressions (Explode)
    print("   - Exploding Impressions...")
    behaviors_df['impressions'] = behaviors_df['impressions'].fillna('').str.split()
    behaviors_df = behaviors_df.explode('impressions').dropna(subset=['impressions'])
    
    # Parse "N123-1" -> N123, 1
    print("   - Parsing Labels...")
    impr_split = behaviors_df['impressions'].str.split('-', expand=True)
    behaviors_df['candidate_nid'] = impr_split[0]
    behaviors_df['label'] = impr_split[1].astype(int)
    
    # Map Candidate to Int
    behaviors_df['candidate_idx'] = behaviors_df['candidate_nid'].map(nid2idx).fillna(0).astype(int)
    
    # ============================
    # 3. Save Optimized Data
    # ============================
    print("5. Saving Optimized Data...")
    
    # Stack history into a 2D numpy array [N_samples, 50]
    history_matrix = np.stack(behaviors_df['history_indices'].values)
    candidate_array = behaviors_df['candidate_idx'].values
    label_array = behaviors_df['label'].values
    
    # Save News Matrix
    np.save(NEWS_MATRIX_PATH, feature_matrix)
    
    # Save Training Arrays
    with open(USER_DATA_PATH, 'wb') as f:
        pickle.dump({
            'history': history_matrix,
            'candidate': candidate_array,
            'label': label_array
        }, f)
        
    # Save Vocabs (for model init)
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({
            'word': word2idx, 'category': cat2idx, 'subcategory': subcat2idx,
            'nid2idx': nid2idx # Useful for inference later
        }, f)

    print("Done! Data is now vectorized.")

if __name__ == '__main__':
    main()