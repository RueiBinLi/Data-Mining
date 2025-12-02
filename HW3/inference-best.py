import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
import os
import sys

# --- CRITICAL FIX: Import the exact model class used in training ---
# This ensures architecture and state_dict keys match perfectly.
from model import BaselineModel 

# --- Configuration ---
TEST_NEWS_PATH = 'test/test_news.tsv' # Make sure this path exists
TEST_BEHAVIORS_PATH = 'test/test_behaviors.tsv'
VOCAB_PATH = 'processed/vocabs.pkl'
MODEL_PATH = 'baseline_model.pt'
SUBMISSION_PATH = 'submission.csv'

# Must match train.py exactly
MAX_TITLE_LEN = 30
MAX_HISTORY_LEN = 50
EMBED_DIM = 256

def process_test_news(news_path, vocabs, max_title_len=30):
    """
    Loads test news using manual parsing to handle 'bad' lines with extra tabs.
    """
    print(f"Processing test news from {news_path}...")
    
    news_lookup = {}
    word_to_index = vocabs['word']
    cat_to_index = vocabs['category']
    subcat_to_index = vocabs['subcategory']
    
    # Pre-fetch special indices
    unk_idx = word_to_index.get('UNK', 0)
    pad_cat = cat_to_index.get('PAD', 0)
    pad_subcat = subcat_to_index.get('PAD', 0)

    # Use standard python file reading (more robust than pandas for this)
    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Indexing News"):
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            
            # We only strictly need the first 4 columns: ID, Cat, Subcat, Title
            # Even if the Abstract (col 4) has tabs and splits into more parts, 
            # the first 4 parts are usually safe.
            if len(parts) >= 4:
                news_id = parts[0]
                category = parts[1]
                subcategory = parts[2]
                title_text = parts[3]
                
                # Title Processing
                title_tokens = str(title_text).lower().split()
                title_indices = [word_to_index.get(w, unk_idx) for w in title_tokens]
                
                padded_title = np.zeros(max_title_len, dtype=np.int32)
                L = min(len(title_indices), max_title_len)
                if L > 0:
                    padded_title[:L] = title_indices[:L]
                
                news_lookup[news_id] = {
                    'title': padded_title,
                    'category': cat_to_index.get(category, pad_cat),
                    'subcategory': subcat_to_index.get(subcategory, pad_subcat)
                }
        
    # Ensure PAD exists
    news_lookup['PAD'] = {
        'title': np.zeros(max_title_len, dtype=np.int32),
        'category': pad_cat,
        'subcategory': pad_subcat
    }
    
    return news_lookup

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Vocabs
    print(f"Loading vocabs from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'rb') as f:
        vocabs = pickle.load(f)
        
    # 2. Process Test News
    test_news_lookup = process_test_news(TEST_NEWS_PATH, vocabs, MAX_TITLE_LEN)
    
    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    # Initialize the model structure FIRST
    model = BaselineModel(
        len(vocabs['word']), 
        len(vocabs['category']), 
        len(vocabs['subcategory']), 
        EMBED_DIM
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print("\n!!! ERROR LOADING WEIGHTS !!!")
        print("This usually happens if the architecture in model.py doesn't match the saved file.")
        print(e)
        return

    model.eval() # CRITICAL: Disables dropout/batchnorm updates

    # 4. Load Behaviors
    print(f"Loading behaviors from {TEST_BEHAVIORS_PATH}...")
    behaviors_df = pd.read_csv(
        TEST_BEHAVIORS_PATH, sep='\t', header=None,
        names=['id', 'user_id', 'time', 'clicked_news', 'impressions'],
        engine='python'
    )
    
    # Handle Header Row if present
    if str(behaviors_df.iloc[0]['id']).strip().lower() == 'id':
        behaviors_df = behaviors_df.iloc[1:].reset_index(drop=True)

    results = []
    
    print("Starting Inference...")
    with torch.no_grad():
        for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Predicting"):
            impression_id = row['id']
            
            # --- Parse History ---
            history_ids = str(row['clicked_news']).split()
            # Pad/Truncate History
            padded_history = ['PAD'] * MAX_HISTORY_LEN
            L = min(len(history_ids), MAX_HISTORY_LEN)
            if L > 0:
                padded_history[-L:] = history_ids[-L:] # Keep most recent interactions
            
            # --- Parse Candidates ---
            # Impressions format: "NewsID-Label NewsID-Label" (if labeled) or "NewsID NewsID"
            # We only need the NewsID.
            raw_impressions = str(row['impressions']).split()
            candidate_ids = [imp.split('-')[0] for imp in raw_impressions]
            num_candidates = len(candidate_ids)
            
            if num_candidates == 0:
                results.append({'id': impression_id, **{f'p{i+1}': 0.0 for i in range(15)}})
                continue

            # --- Prepare Batch Data ---
            
            # 1. Fetch History Features (List of Arrays)
            h_titles, h_cats, h_subcats = [], [], []
            for nid in padded_history:
                d = test_news_lookup.get(nid, test_news_lookup['PAD'])
                h_titles.append(d['title'])
                h_cats.append(d['category'])
                h_subcats.append(d['subcategory'])
            
            # 2. Replicate History for each Candidate
            # We want to score [User, Candidate_1], [User, Candidate_2]...
            # So we repeat the User History N times.
            # Shape: [num_candidates, history_len, title_len]
            batch_h_titles = torch.tensor(np.array(h_titles), dtype=torch.long).unsqueeze(0).repeat(num_candidates, 1, 1).to(device)
            batch_h_cats = torch.tensor(np.array(h_cats), dtype=torch.long).unsqueeze(0).repeat(num_candidates, 1).to(device)
            batch_h_subcats = torch.tensor(np.array(h_subcats), dtype=torch.long).unsqueeze(0).repeat(num_candidates, 1).to(device)
            
            # 3. Fetch Candidate Features
            c_titles, c_cats, c_subcats = [], [], []
            for nid in candidate_ids:
                d = test_news_lookup.get(nid, test_news_lookup['PAD'])
                c_titles.append(d['title'])
                c_cats.append(d['category'])
                c_subcats.append(d['subcategory'])
                
            batch_c_titles = torch.tensor(np.array(c_titles), dtype=torch.long).to(device)
            batch_c_cats = torch.tensor(np.array(c_cats), dtype=torch.long).to(device)
            batch_c_subcats = torch.tensor(np.array(c_subcats), dtype=torch.long).to(device)
            
            # --- Forward Pass ---
            logits = model(batch_h_titles, batch_h_cats, batch_h_subcats, 
                           batch_c_titles, batch_c_cats, batch_c_subcats)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # --- Format Output ---
            submission_row = {'id': impression_id}
            # Kaggle expects p1...p15. 
            # If there are fewer than 15 candidates, fill with 0.
            # If there are more, we just take the first 15 (though usually it's fixed).
            for i in range(15):
                submission_row[f'p{i+1}'] = probs[i] if i < len(probs) else 0.0
            
            results.append(submission_row)

    # Write output
    print(f"Writing {len(results)} rows to {SUBMISSION_PATH}...")
    submission_df = pd.DataFrame(results)
    cols = ['id'] + [f'p{i+1}' for i in range(15)]
    submission_df = submission_df[cols]
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()