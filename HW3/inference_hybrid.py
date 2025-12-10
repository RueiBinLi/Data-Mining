import pandas as pd
import torch
import numpy as np
import pickle
from tqdm import tqdm
import os
import sys

# --- Import your Hybrid model ---
# Ensure model_roberta.py contains the 'HybridRobertaModel' class
from model_roberta import HybridRobertaModel 

# --- CONFIGURATION ---
BERT_DIM = 768          # Input size (RoBERTa base)
EMBED_DIM = 256         # Internal model dimension
MAX_HISTORY = 50        # Use FULL history (50) for best inference performance

# Paths
TEST_BEHAVIORS_PATH = 'test/test_behaviors.tsv'
SUBMISSION_PATH = 'submission.csv'
MODEL_PATH = 'roberta_model.pt' # Or 'roberta_best_model.pt' if you used validation

# Resources (Must match preprocessing)
PROCESSED_PATH = 'processed'
ROBERTA_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_roberta_embeddings.npy')
CAT_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_cat_subcat.npy')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Vocabs (For ID mapping & Model Initialization)
    print("Loading Vocabs...")
    if not os.path.exists(VOCAB_PATH):
        print("Error: vocabs.pkl not found.")
        return

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
        nid2idx = vocab['nid2idx']
        # We need vocab sizes to re-initialize the model correctly
        cat_vocab_size = len(vocab.get('cat2idx', {})) + 1
        subcat_vocab_size = len(vocab.get('subcat2idx', {})) + 1

    # 2. Load Global Matrices
    print("Loading Global Matrices...")
    
    # A. RoBERTa Vectors (Float32)
    if not os.path.exists(ROBERTA_MATRIX_PATH):
        print(f"Error: {ROBERTA_MATRIX_PATH} missing.")
        return
    r_matrix_np = np.load(ROBERTA_MATRIX_PATH)
    r_matrix = torch.tensor(r_matrix_np, dtype=torch.float32).to(device)
    
    # B. Category Indices (Int64)
    if not os.path.exists(CAT_MATRIX_PATH):
        print(f"Error: {CAT_MATRIX_PATH} missing.")
        return
    c_matrix_np = np.load(CAT_MATRIX_PATH)
    c_matrix = torch.tensor(c_matrix_np, dtype=torch.long).to(device)

    # 3. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    try:
        model = HybridRobertaModel(
            cat_vocab=cat_vocab_size, 
            subcat_vocab=subcat_vocab_size,
            roberta_dim=BERT_DIM, 
            embed_dim=EMBED_DIM
        ).to(device)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Read Test Behaviors
    print(f"Reading {TEST_BEHAVIORS_PATH}...")
    behaviors = []
    # Manual read to avoid pandas errors with bad lines
    with open(TEST_BEHAVIORS_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            if parts[0] == 'id': continue
            
            behaviors.append({
                'id': parts[0],
                'hist': parts[3],
                'imp': parts[4]
            })
    
    print(f"Found {len(behaviors)} test users.")
    
    # 5. Prediction Loop
    results = []
    print("Predicting...")
    
    with torch.no_grad():
        for row in tqdm(behaviors):
            impression_id = row['id']
            
            # --- Parse Indices ---
            # History
            hist_str = row['hist'].split()
            hist_ids = [nid2idx.get(x, 0) for x in hist_str]
            hist_ids = hist_ids[-MAX_HISTORY:] # Last N
            if len(hist_ids) < MAX_HISTORY:
                hist_ids = [0] * (MAX_HISTORY - len(hist_ids)) + hist_ids
            
            # Candidates
            imp_str = row['imp'].split()
            cand_nids = [x.split('-')[0] for x in imp_str]
            cand_ids = [nid2idx.get(x, 0) for x in cand_nids]
            
            if not cand_ids:
                res = {'id': impression_id}
                for i in range(15): res[f'p{i+1}'] = 0.0
                results.append(res)
                continue

            num_cands = len(cand_ids)
            
            # --- Prepare Batch (Replicate History for each Candidate) ---
            # Shape: [Num_Cands, MAX_HISTORY]
            batch_hist = torch.tensor([hist_ids] * num_cands, dtype=torch.long, device=device)
            # Shape: [Num_Cands]
            batch_cand = torch.tensor(cand_ids, dtype=torch.long, device=device)
            
            # --- MATRIX LOOKUP ---
            
            # 1. RoBERTa Vectors
            h_vecs = r_matrix[batch_hist] # [Num_Cands, 50, 768]
            c_vec = r_matrix[batch_cand]  # [Num_Cands, 768]
            
            # 2. Categories (Col 0) & Subcategories (Col 1)
            h_cats_all = c_matrix[batch_hist] # [Num_Cands, 50, 2]
            c_cats_all = c_matrix[batch_cand] # [Num_Cands, 2]
            
            h_cat = h_cats_all[:, :, 0]
            h_sub = h_cats_all[:, :, 1]
            
            c_cat = c_cats_all[:, 0]
            c_sub = c_cats_all[:, 1]
            
            # --- FORWARD PASS ---
            # Pass all 6 inputs to the model
            logits = model(h_vecs, h_cat, h_sub, c_vec, c_cat, c_sub)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # --- Formatting ---
            res = {'id': impression_id}
            for i in range(15):
                res[f'p{i+1}'] = probs[i] if i < num_cands else 0.0
            results.append(res)

    # 6. Save Output
    print(f"Writing {len(results)} rows to {SUBMISSION_PATH}...")
    df = pd.DataFrame(results)
    cols = ['id'] + [f'p{i+1}' for i in range(15)]
    df = df[cols]
    df.to_csv(SUBMISSION_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    main()