import pandas as pd
import torch
import numpy as np
import pickle
from tqdm import tqdm
import os
import sys

# --- Import your model architecture ---
# Ensure model.py contains the BertBaselineModel (or RobertaModel) class
from model_BERT import BertBaselineModel 

# --- Configuration ---
# Match these with your train.py settings
BERT_DIM = 768          # Input vector size (BERT/RoBERTa base)
EMBED_DIM = 256         # Internal model dimension
MAX_HISTORY = 20        # Use the same history length you trained with (20 is faster/better)

# Paths
TEST_BEHAVIORS = 'test/test_behaviors.tsv'
SUBMISSION_PATH = 'submission.csv'

# Resources (Must match what you generated in preprocessing)
# If you used RoBERTa, change this to 'news_roberta_embeddings.npy'
NEWS_MATRIX_PATH = 'processed/news_bert_embeddings.npy' 
VOCAB_PATH = 'processed/vocabs.pkl'
MODEL_PATH = 'bert_model.pt'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Resources
    print("Loading Global Matrix & Vocabs...")
    
    # Load the big float32 matrix (Pre-computed BERT vectors)
    if not os.path.exists(NEWS_MATRIX_PATH):
        print(f"Error: {NEWS_MATRIX_PATH} not found. Did you run the global preprocess script?")
        return
        
    matrix_np = np.load(NEWS_MATRIX_PATH)
    news_matrix = torch.tensor(matrix_np, dtype=torch.float32).to(device)
    
    # Load ID mapping
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
        nid2idx = vocab['nid2idx']

    # 2. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    model = BertBaselineModel(BERT_DIM, EMBED_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Read Test Data
    print("Reading Test Behaviors...")
    # Parsing strictly to avoid pandas errors
    behaviors = []
    with open(TEST_BEHAVIORS, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            # Check format: id, uid, time, hist, imp
            if len(parts) < 5: continue
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
            # Split history string and map to Integers
            hist_str = row['hist'].split()
            hist_ids = [nid2idx.get(x, 0) for x in hist_str]
            
            # Take the last N clicks (must match training logic)
            hist_ids = hist_ids[-MAX_HISTORY:]
            
            # Pad if too short
            if len(hist_ids) < MAX_HISTORY:
                hist_ids = [0] * (MAX_HISTORY - len(hist_ids)) + hist_ids
            
            # --- Parse Candidates ---
            # format: "N1234 N5678" (test_behaviors usually doesn't have labels)
            # If it has labels "N1234-0", split by '-'
            imp_str = row['imp'].split()
            cand_nids = [x.split('-')[0] for x in imp_str]
            cand_ids = [nid2idx.get(x, 0) for x in cand_nids]
            
            if not cand_ids:
                # Fallback for empty candidates (rare)
                res = {'id': impression_id}
                for i in range(15): res[f'p{i+1}'] = 0.0
                results.append(res)
                continue

            # --- Batch Inference for 1 User ---
            # We want to score: (User) vs (Cand1), (User) vs (Cand2)...
            # So we repeat the User History vector for every candidate
            
            num_cands = len(cand_ids)
            
            # Prepare Tensors on GPU
            # History: [Num_Cands, MAX_HISTORY]
            batch_hist = torch.tensor([hist_ids] * num_cands, dtype=torch.long, device=device)
            # Candidate: [Num_Cands]
            batch_cand = torch.tensor(cand_ids, dtype=torch.long, device=device)
            
            # LOOKUP VECTORS (The Fast Part)
            # [Num_Cands, MAX_HISTORY, 768]
            hist_vecs = news_matrix[batch_hist]
            # [Num_Cands, 768]
            cand_vecs = news_matrix[batch_cand]
            
            # FORWARD PASS
            logits = model(hist_vecs, cand_vecs)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # --- Formatting ---
            res = {'id': impression_id}
            # Fill p1...p15
            for i in range(15):
                if i < num_cands:
                    res[f'p{i+1}'] = probs[i]
                else:
                    res[f'p{i+1}'] = 0.0 # Pad with 0 if fewer than 15 cands
            results.append(res)

    # 5. Save
    print(f"Writing submission to {SUBMISSION_PATH}...")
    df = pd.DataFrame(results)
    # Ensure correct column order
    cols = ['id'] + [f'p{i+1}' for i in range(15)]
    df = df[cols]
    df.to_csv(SUBMISSION_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    main()