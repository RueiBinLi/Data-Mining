import pandas as pd
import torch
import numpy as np
import pickle
from tqdm import tqdm
import os
from model_roberta import HybridRobertaModel 

# CONFIG
BERT_DIM = 768
EMBED_DIM = 256
MAX_HISTORY = 50  # Using full history for test!
MODELS_TO_ENSEMBLE = ['roberta_ep2.pt', 'roberta_ep3.pt', 'roberta_ep4.pt'] # Check if these exist!

TEST_BEHAVIORS = 'test/test_behaviors.tsv'
SUBMISSION_PATH = 'submission_ensemble.csv'
PROCESSED = 'processed'
R_MAT = f'{PROCESSED}/news_roberta_embeddings.npy'
C_MAT = f'{PROCESSED}/news_cat_subcat.npy'
VOCAB = f'{PROCESSED}/vocabs.pkl'

def main():
    device = torch.device('cuda')
    
    # 1. Load Resources
    print("Loading Resources...")
    r_matrix = torch.tensor(np.load(R_MAT), dtype=torch.float32).to(device)
    c_matrix = torch.tensor(np.load(C_MAT), dtype=torch.long).to(device)
    with open(VOCAB, 'rb') as f:
        v = pickle.load(f)
        nid2idx = v['nid2idx']
        c_len = len(v['cat2idx']) + 1
        s_len = len(v['subcat2idx']) + 1

    # 2. Load ALL Models
    models = []
    print(f"Loading {len(MODELS_TO_ENSEMBLE)} models for ensemble...")
    for path in MODELS_TO_ENSEMBLE:
        if not os.path.exists(path):
            print(f"Skipping {path} (Not found)")
            continue
        m = HybridRobertaModel(c_len, s_len, BERT_DIM, EMBED_DIM).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    # 3. Predict
    print("Reading Test Data...")
    behaviors = []
    with open(TEST_BEHAVIORS, 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p)<5 or p[0]=='id': continue
            behaviors.append({'id':p[0], 'h':p[3], 'i':p[4]})

    results = []
    print("Predicting with Ensemble...")
    with torch.no_grad():
        for row in tqdm(behaviors):
            # Parse
            h_ids = [nid2idx.get(x,0) for x in row['h'].split()][-MAX_HISTORY:]
            h_ids = [0]*(MAX_HISTORY-len(h_ids)) + h_ids
            
            imp_parts = row['i'].split()
            c_ids = [nid2idx.get(x.split('-')[0], 0) for x in imp_parts]
            n = len(c_ids)
            if n==0: 
                results.append({'id':row['id'], **{f'p{i+1}':0.0 for i in range(15)}})
                continue

            # Batch
            b_h = torch.tensor([h_ids]*n, device=device)
            b_c = torch.tensor(c_ids, device=device)
            
            # Lookup
            h_v, c_v = r_matrix[b_h], r_matrix[b_c]
            h_cats, c_cats = c_matrix[b_h], c_matrix[b_c]
            
            # --- AVERAGE PREDICTIONS ---
            avg_probs = np.zeros(n)
            for m in models:
                logits = m(h_v, h_cats[:,:,0], h_cats[:,:,1], c_v, c_cats[:,0], c_cats[:,1])
                probs = torch.sigmoid(logits).cpu().numpy()
                avg_probs += probs
            
            avg_probs /= len(models) # Average
            
            # Format
            res = {'id': row['id']}
            for i in range(15):
                res[f'p{i+1}'] = avg_probs[i] if i < n else 0.0
            results.append(res)

    pd.DataFrame(results)[['id']+[f'p{i+1}' for i in range(15)]].to_csv(SUBMISSION_PATH, index=False)
    print("Done! Upload submission_ensemble.csv")

if __name__ == "__main__":
    main()