import pandas as pd
import numpy as np
import pickle
import os

# --- Config ---
# Must use the SAME paths as your global RoBERTa preprocess
TRAIN_NEWS = 'train/train_news.tsv'
TEST_NEWS = 'test/test_news.tsv'
PROCESSED_PATH = 'processed'
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')
OUTPUT_MATRIX = os.path.join(PROCESSED_PATH, 'news_cat_subcat.npy')

def main():
    print("Loading ID Mapping...")
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
        nid2idx = vocab['nid2idx'] # The global mapping
        
    print("Reading News...")
    cols = ['news_id', 'cat', 'subcat', 'title', 'abs', 'url', 'te', 'ae']
    df_train = pd.read_csv(TRAIN_NEWS, sep='\t', header=None, names=cols, quoting=3)
    df_test = pd.read_csv(TEST_NEWS, sep='\t', header=None, names=cols, quoting=3)
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
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
        
    # Build Matrix
    print("Building Feature Matrix...")
    num_news = len(nid2idx) # This usually equals len(all_news) + 1 (for PAD)
    # Col 0: Category, Col 1: Subcategory
    feature_matrix = np.zeros((num_news + 1, 2), dtype=np.int32)
    
    for _, row in all_news.iterrows():
        nid = row['news_id']
        if nid in nid2idx:
            idx = nid2idx[nid]
            feature_matrix[idx, 0] = cat2idx.get(row['cat'], 0)
            feature_matrix[idx, 1] = subcat2idx.get(row['subcat'], 0)
            
    np.save(OUTPUT_MATRIX, feature_matrix)
    print(f"Saved {feature_matrix.shape} matrix to {OUTPUT_MATRIX}")
    print(f"Cat Size: {len(cat2idx)}, Subcat Size: {len(subcat2idx)}")

if __name__ == "__main__":
    main()