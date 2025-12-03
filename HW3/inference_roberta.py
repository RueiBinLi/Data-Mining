import pandas as pd
import torch
import numpy as np
import pickle
from tqdm import tqdm
import os
import sys

# --- CONFIGURATION ---
# Check your model.py class name!
# If using RoBERTa: from model import RobertaModel
# If using BERT:    from model import BertBaselineModel 
from model_roberta import RobertaBaselineModel 

BERT_DIM = 768          # Input size (Standard for BERT/RoBERTa base)
EMBED_DIM = 256         # Internal model dimension (Must match training!)
MAX_HISTORY = 20        # Must match what you used in train.py (20)

# Paths
TEST_BEHAVIORS_PATH = 'test/test_behaviors.tsv'
SUBMISSION_PATH = 'submission_roberta.csv'
MODEL_PATH = 'bert_model.pt'

# Resources (Must match your preprocessing)
NEWS_MATRIX_PATH = 'processed/news_roberta_embeddings.npy' 

VOCAB_PATH = 'processed/vocabs.pkl'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Resources (Global Matrix & Mappings)
    print("Loading Global Matrix & Vocabs...")
    
    if not os.path.exists(NEWS_MATRIX_PATH):
        print(f"CRITICAL ERROR: {NEWS_MATRIX_PATH} not found.")
        print("Did you run the global preprocessing script?")
        return
        
    # Load the float32 matrix (Pre-computed Vectors)
    matrix_np = np.load(NEWS_MATRIX_PATH)
    # Move to GPU immediately for fast lookup
    news_matrix = torch.tensor(matrix_np, dtype=torch.float32).to(device)
    
    # Load ID mapping (NewsID string -> Matrix Index int)
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
        nid2idx = vocab['nid2idx']

    # 2. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    try:
        model = RobertaBaselineModel(BERT_DIM, EMBED_DIM).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure 'model.py' defines the class 'RobertaBaselineModel' matching your saved weights.")
        return

    # 3. Read Test Behaviors
    print(f"Reading {TEST_BEHAVIORS_PATH}...")
    behaviors = []
    # Manual read to avoid pandas parsing errors with bad lines
    with open(TEST_BEHAVIORS_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            # Check format: id, uid, time, hist, imp
            if len(parts) < 5: continue
            
            # Handle potential header
            if parts[0] == 'id': continue
            
            behaviors.append({
                'id': parts[0],
                'hist': parts[3],
                'imp': parts[4]
            })
    
    print(f"Found {len(behaviors)} test users.")
    
    # 4. Prediction Loop
    results = []
    
    print("Predicting...")
    with torch.no_grad():
        for row in tqdm(behaviors):
            impression_id = row['id']
            
            # --- Parse History ---
            # Map NewsIDs to Indices
            hist_str = row['hist'].split()
            hist_ids = [nid2idx.get(x, 0) for x in hist_str]
            
            # Optimization: Take only the last N clicks (Same as training)
            hist_ids = hist_ids[-MAX_HISTORY:]
            
            # Pad if too short
            if len(hist_ids) < MAX_HISTORY:
                hist_ids = [0] * (MAX_HISTORY - len(hist_ids)) + hist_ids
            
            # --- Parse Candidates ---
            # Candidates string: "N1234 N5678" (or "N1234-0" if labeled)
            imp_str = row['imp'].split()
            cand_nids = [x.split('-')[0] for x in imp_str]
            cand_ids = [nid2idx.get(x, 0) for x in cand_nids]
            
            if not cand_ids:
                # Fallback for empty candidates (rare edge case)
                res = {'id': impression_id}
                for i in range(15): res[f'p{i+1}'] = 0.0
                results.append(res)
                continue

            # --- Batch Inference for 1 User ---
            num_cands = len(cand_ids)
            
            # Prepare Tensors on GPU
            # 1. Replicate History for every candidate
            # Shape: [Num_Cands, MAX_HISTORY]
            batch_hist = torch.tensor([hist_ids] * num_cands, dtype=torch.long, device=device)
            
            # 2. Candidates
            # Shape: [Num_Cands]
            batch_cand = torch.tensor(cand_ids, dtype=torch.long, device=device)
            
            # 3. LOOKUP VECTORS (The Speedup!)
            # Shape: [Num_Cands, MAX_HISTORY, 768]
            hist_vecs = news_matrix[batch_hist]
            # Shape: [Num_Cands, 768]
            cand_vecs = news_matrix[batch_cand]
            
            # 4. FORWARD PASS
            logits = model(hist_vecs, cand_vecs)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # --- Formatting for Kaggle ---
            res = {'id': impression_id}
            for i in range(15):
                if i < num_cands:
                    res[f'p{i+1}'] = probs[i]
                else:
                    res[f'p{i+1}'] = 0.0 # Pad with 0 if fewer than 15 cands
            results.append(res)

    # 5. Save Output
    print(f"Writing {len(results)} rows to {SUBMISSION_PATH}...")
    df = pd.DataFrame(results)
    
    # Ensure explicit column order
    cols = ['id'] + [f'p{i+1}' for i in range(15)]
    df = df[cols]
    
    df.to_csv(SUBMISSION_PATH, index=False)
    print("Done! Upload 'submission.csv' to Kaggle.")

if __name__ == "__main__":
    main()