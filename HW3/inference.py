import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
import os

# --- Configuration ---
TEST_NEWS_PATH = 'test/test_news.tsv'
TEST_BEHAVIORS_PATH = 'test/test_behaviors.tsv'
VOCAB_PATH = 'processed/vocabs.pkl'
MODEL_PATH = 'baseline_model.pt'
SUBMISSION_PATH = 'submission.csv'

# Same constraints as training
MAX_TITLE_LEN = 30
MAX_HISTORY_LEN = 50
EMBED_DIM = 128

# --- 1. Re-define the Model Classes ---
# (These must match your training script EXACTLY)

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, 
                 embed_dim=128, title_dim=32, cat_dim=32):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.title_reduce = nn.Linear(embed_dim, title_dim)
        self.cat_embedding = nn.Embedding(cat_vocab_size, cat_dim, padding_idx=0)
        self.subcat_embedding = nn.Embedding(subcat_vocab_size, cat_dim, padding_idx=0)
        self.final_layer = nn.Linear(title_dim + cat_dim + cat_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, title, category, subcategory):
        title_embed = self.word_embedding(title)
        title_vec = torch.mean(title_embed, dim=1)
        title_vec = self.relu(self.title_reduce(title_vec))
        cat_vec = self.cat_embedding(category)
        subcat_vec = self.subcat_embedding(subcategory)
        combined_vec = torch.cat([title_vec, cat_vec, subcat_vec], dim=1)
        return self.relu(self.final_layer(combined_vec))

class UserEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
    def forward(self, history_embeddings):
        return torch.mean(history_embeddings, dim=1)

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, embed_dim=128):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, cat_vocab_size, subcat_vocab_size, embed_dim)
        self.user_encoder = UserEncoder(embed_dim)
        
    def forward(self, hist_titles, hist_cats, hist_subcats, 
                cand_title, cand_cat, cand_subcat):
        cand_embedding = self.news_encoder(cand_title, cand_cat, cand_subcat)
        
        batch_size, history_len, title_len = hist_titles.shape
        hist_titles_flat = hist_titles.view(batch_size * history_len, title_len)
        hist_cats_flat = hist_cats.view(batch_size * history_len)
        hist_subcats_flat = hist_subcats.view(batch_size * history_len)
        
        history_embeddings_flat = self.news_encoder(hist_titles_flat, hist_cats_flat, hist_subcats_flat)
        history_embeddings = history_embeddings_flat.view(batch_size, history_len, -1)
        
        user_embedding = self.user_encoder(history_embeddings)
        score = torch.sum(user_embedding * cand_embedding, dim=1)
        return score

# --- 2. Helper: Process Test News ---
def process_test_news(news_path, vocabs, max_title_len=30):
    """
    Loads test news and converts to indices using the TRAINING vocab.
    Important: Does not add new words to vocab; uses UNK/PAD.
    """
    print("Processing test news...")
    news_df = pd.read_csv(
        news_path, sep='\t', header=None,
        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 't_ent', 'a_ent']
    )
    news_df['title'] = news_df['title'].fillna('')
    
    news_lookup = {}
    word_to_index = vocabs['word']
    cat_to_index = vocabs['category']
    subcat_to_index = vocabs['subcategory']
    
    # Pre-fetch special indices
    unk_idx = word_to_index['UNK']
    pad_cat = cat_to_index['PAD']
    pad_subcat = subcat_to_index['PAD']

    for _, row in tqdm(news_df.iterrows(), total=len(news_df)):
        # Title
        title_tokens = row['title'].lower().split()
        title_indices = [word_to_index.get(w, unk_idx) for w in title_tokens]
        
        padded_title = np.zeros(max_title_len, dtype=np.int32)
        L = min(len(title_indices), max_title_len)
        padded_title[:L] = title_indices[:L]
        
        news_lookup[row['news_id']] = {
            'title': padded_title,
            'category': cat_to_index.get(row['category'], pad_cat),
            'subcategory': subcat_to_index.get(row['subcategory'], pad_subcat)
        }
        
    # Ensure PAD exists
    news_lookup['PAD'] = {
        'title': np.zeros(max_title_len, dtype=np.int32),
        'category': pad_cat,
        'subcategory': pad_subcat
    }
    return news_lookup

# --- 3. Main Inference Function ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # A. Load Vocabs
    print(f"Loading vocabs from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'rb') as f:
        vocabs = pickle.load(f)
        
    # B. Process Test News (using training vocabs)
    test_news_lookup = process_test_news(TEST_NEWS_PATH, vocabs, MAX_TITLE_LEN)
    
    # C. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = BaselineModel(
        len(vocabs['word']), len(vocabs['category']), len(vocabs['subcategory']), EMBED_DIM
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Important: Set to eval mode!

    # D. Load Test Behaviors
    print(f"Loading test behaviors from {TEST_BEHAVIORS_PATH}...")
    behaviors_df = pd.read_csv(
        TEST_BEHAVIORS_PATH, sep='\t', header=None,
        names=['id', 'user_id', 'time', 'clicked_news', 'impressions'],
        engine='python'
    )
    
    # E. Prediction Loop
    print("Starting prediction...")
    results = []
    
    # We process row by row (user by user)
    with torch.no_grad():
        for idx, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df)):
            impression_id = row['id']
            
            # 1. Parse History
            history_ids = row['clicked_news']
            if pd.isna(history_ids):
                history_ids = []
            else:
                history_ids = history_ids.split()
            
            # Pad history
            padded_history = ['PAD'] * MAX_HISTORY_LEN
            L = min(len(history_ids), MAX_HISTORY_LEN)
            if L > 0:
                padded_history[-L:] = history_ids[-L:]
            
            # 2. Parse Candidates (Impressions)
            # Impressions in test might look like "N1234-0 N5678-0" or just "N1234 N5678"
            # We split by space, then strip any "-0" or "-1" if present
            raw_impressions = row['impressions'].split()
            candidate_ids = [imp.split('-')[0] for imp in raw_impressions]
            
            # According to specs, there should be 15 predictions.
            # If there are fewer candidates, we predict for what exists.
            num_candidates = len(candidate_ids)
            
            # 3. Build Batch for this User
            # We repeat the history 'num_candidates' times to create a batch
            
            # Get History Features (Single)
            h_titles, h_cats, h_subcats = [], [], []
            for nid in padded_history:
                d = test_news_lookup.get(nid, test_news_lookup['PAD'])
                h_titles.append(d['title'])
                h_cats.append(d['category'])
                h_subcats.append(d['subcategory'])
            
            # Replicate for batch
            batch_h_titles = torch.tensor(np.array(h_titles), dtype=torch.long).unsqueeze(0).repeat(num_candidates, 1, 1).to(device)
            batch_h_cats = torch.tensor(np.array(h_cats), dtype=torch.long).unsqueeze(0).repeat(num_candidates, 1).to(device)
            batch_h_subcats = torch.tensor(np.array(h_subcats), dtype=torch.long).unsqueeze(0).repeat(num_candidates, 1).to(device)
            
            # Get Candidate Features (Batch)
            c_titles, c_cats, c_subcats = [], [], []
            for nid in candidate_ids:
                d = test_news_lookup.get(nid, test_news_lookup['PAD'])
                c_titles.append(d['title'])
                c_cats.append(d['category'])
                c_subcats.append(d['subcategory'])
                
            batch_c_titles = torch.tensor(np.array(c_titles), dtype=torch.long).to(device)
            batch_c_cats = torch.tensor(np.array(c_cats), dtype=torch.long).to(device)
            batch_c_subcats = torch.tensor(np.array(c_subcats), dtype=torch.long).to(device)
            
            # 4. Forward Pass
            logits = model(batch_h_titles, batch_h_cats, batch_h_subcats, 
                           batch_c_titles, batch_c_cats, batch_c_subcats)
            
            # 5. Convert to Probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # 6. Store Result
            # Format: {'id': 1, 'p1': 0.5, 'p2': 0.1, ...}
            submission_row = {'id': impression_id}
            for i in range(15):
                col_name = f'p{i+1}'
                if i < len(probs):
                    submission_row[col_name] = probs[i]
                else:
                    # Fallback if fewer than 15 candidates (though spec says 15)
                    submission_row[col_name] = 0.0 
            results.append(submission_row)

    # F. Write Submission CSV
    print(f"Writing submission to {SUBMISSION_PATH}...")
    submission_df = pd.DataFrame(results)
    
    # Ensure column order
    cols = ['id'] + [f'p{i+1}' for i in range(15)]
    submission_df = submission_df[cols]
    
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Done! Upload 'submission.csv' to Kaggle.")

if __name__ == "__main__":
    main()