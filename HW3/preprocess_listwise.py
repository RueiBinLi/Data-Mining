import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import random

# --- Config ---
TRAIN_BEHAVIORS = 'train/train_behaviors.tsv'
PROCESSED_PATH = 'processed'
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')
OUTPUT_PATH = os.path.join(PROCESSED_PATH, 'train_data_listwise.pkl')

NEG_RATIO = 4 # For every 1 positive, pick 4 negatives

def main():
    if not os.path.exists(VOCAB_PATH):
        print("Error: Vocab not found. Run global preprocess first.")
        return

    print("Loading ID Mapping...")
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
        nid2idx = vocab['nid2idx']

    print("Reading Behaviors...")
    # Manual read to avoid pandas memory issues
    samples = []
    
    with open(TRAIN_BEHAVIORS, 'r') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 5: continue
            
            # History
            hist_str = parts[3]
            if pd.isna(hist_str):
                hist = [0] * 50
            else:
                hist = [nid2idx.get(x, 0) for x in hist_str.split()[-50:]]
                hist = [0]*(50-len(hist)) + hist
            
            # Impressions
            imp_str = parts[4]
            impressions = imp_str.split()
            
            positives = []
            negatives = []
            
            for imp in impressions:
                nid, label = imp.split('-')
                idx = nid2idx.get(nid, 0)
                if label == '1':
                    positives.append(idx)
                else:
                    negatives.append(idx)
            
            # Create Training Samples (1 Pos + 4 Negs)
            for pos in positives:
                if len(negatives) < NEG_RATIO:
                    # If not enough negatives, reuse random ones
                    negs = random.choices(negatives, k=NEG_RATIO) if negatives else [0]*NEG_RATIO
                else:
                    negs = random.sample(negatives, NEG_RATIO)
                
                samples.append({
                    'history': hist,
                    'pos': pos,
                    'negs': negs
                })

    print(f"Generated {len(samples)} listwise samples.")
    
    # Convert to Numpy for Speed
    print("Converting to Numpy...")
    history_matrix = np.array([s['history'] for s in samples], dtype=np.int32)
    pos_matrix = np.array([s['pos'] for s in samples], dtype=np.int32)
    neg_matrix = np.array([s['negs'] for s in samples], dtype=np.int32)
    
    print("Saving...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'history': history_matrix,
            'pos': pos_matrix,
            'negs': neg_matrix
        }, f)
    print("Done.")

if __name__ == "__main__":
    main()