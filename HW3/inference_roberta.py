import pandas as pd
import torch
import numpy as np
import pickle
from tqdm import tqdm
from model_roberta import RobertaModel

# Config
BATCH_SIZE = 512
TEST_BEHAVIORS = 'test/test_behaviors.tsv'
NEWS_MATRIX = 'processed/roberta_embeddings.npy'
VOCAB = 'processed/vocabs.pkl'
MODEL_PATH = 'roberta_model.pt'
OUTPUT = 'submission_roberta.csv'

def main():
    device = torch.device('cuda')
    
    # 1. Load Global Matrix & Mapping
    print("Loading Resources...")
    matrix = torch.tensor(np.load(NEWS_MATRIX), dtype=torch.float32).to(device)
    with open(VOCAB, 'rb') as f:
        nid2idx = pickle.load(f)['nid2idx']
    
    # 2. Load Model
    model = RobertaModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 3. Process Test File
    print("Processing Test Behaviors...")
    df = pd.read_csv(TEST_BEHAVIORS, sep='\t', header=None, 
                     names=['id','uid','time','hist','imp'])
    
    results = []
    
    # Process user by user (or batched if you want to optimize further)
    # Simple Loop:
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # History
            h_str = str(row['hist']).split()
            h_ids = [nid2idx.get(x, 0) for x in h_str[-50:]]
            h_ids = [0]*(50-len(h_ids)) + h_ids
            
            # Impressions
            imps = str(row['imp']).split()
            cand_nids = [x.split('-')[0] for x in imps]
            c_ids = [nid2idx.get(x, 0) for x in cand_nids]
            
            if not c_ids:
                results.append({'id': row['id'], **{f'p{i+1}':0 for i in range(15)}})
                continue
                
            # Create Batch for 1 User
            # H: [Num_Cands, 50]
            # C: [Num_Cands]
            n = len(c_ids)
            batch_h = torch.tensor([h_ids] * n, device=device) # Repeat history
            batch_c = torch.tensor(c_ids, device=device)
            
            # Lookup
            h_vecs = matrix[batch_h]
            c_vec = matrix[batch_c]
            
            # Score
            scores = torch.sigmoid(model(h_vecs, c_vec)).cpu().numpy()
            
            # Format
            res = {'id': row['id']}
            for i in range(15):
                res[f'p{i+1}'] = scores[i] if i < n else 0.0
            results.append(res)
            
    pd.DataFrame(results).to_csv(OUTPUT, index=False)
    print("Done.")

if __name__ == "__main__":
    main()