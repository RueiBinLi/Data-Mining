import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
from model import BaselineModel

# --- Configuration ---
BATCH_SIZE = 512        # <--- INCREASED from 64. Critical for speed.
LEARNING_RATE = 0.001   # Slightly higher for larger batch size
EPOCHS = 8
EMBED_DIM = 256

# Paths
NEWS_MATRIX_PATH = 'processed/news_features.npy'
USER_DATA_PATH = 'processed/user_data.pkl'
VOCAB_PATH = 'processed/vocabs.pkl'
MODEL_SAVE_PATH = 'baseline_model.pt'

class FastNewsDataset(Dataset):
    """
    Super lightweight dataset. 
    Only returns INDICES (Integers), not full features.
    """
    def __init__(self, user_data_path):
        with open(user_data_path, 'rb') as f:
            data = pickle.load(f)
            
        # These are just giant numpy arrays of integers
        self.history = data['history']    # [N, 50] (News IDs)
        self.candidate = data['candidate'] # [N] (News ID)
        self.label = data['label']         # [N] (0 or 1)
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        # We perform NO heavy lifting here. Just return the ints.
        return (
            self.history[idx], 
            self.candidate[idx], 
            self.label[idx]
        )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Vocabs (for model init)
    print("Loading vocabs...")
    with open(VOCAB_PATH, 'rb') as f:
        vocabs = pickle.load(f)
    
    # 2. Load and Prepare News Matrix on GPU
    # This is the "Secret Sauce". The matrix is tiny (~12MB), so we put it on VRAM.
    print("Loading News Matrix to GPU...")
    news_matrix_np = np.load(NEWS_MATRIX_PATH)
    # Shape: [Num_News, 32] -> (Title=0-29, Cat=30, Subcat=31)
    news_matrix = torch.tensor(news_matrix_np, dtype=torch.long).to(device)
    
    # 3. Dataset & DataLoader
    print("Initializing DataLoader...")
    dataset = FastNewsDataset(USER_DATA_PATH)
    # num_workers=0 is often FASTER when simply passing integers from memory
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 4. Model
    model = BaselineModel(
        len(vocabs['word']), 
        len(vocabs['category']), 
        len(vocabs['subcategory']), 
        EMBED_DIM
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"--- Starting Training (Batch Size {BATCH_SIZE}) ---")
    
    TITLE_LEN = 30 # Must match your preprocessing
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        # Enable tqdm but update less frequently to save console I/O time
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_hist_ids, batch_cand_id, batch_labels in loop:
            # Move Indices to GPU
            batch_hist_ids = batch_hist_ids.to(device)   # [Batch, 50]
            batch_cand_id = batch_cand_id.to(device)     # [Batch]
            batch_labels = batch_labels.float().to(device)
            
            # --- GPU-SIDE LOOKUP ---
            # This is 100x faster than doing it in the Dataset
            
            # 1. Look up features for History
            # Result: [Batch, 50, 32]
            hist_feats = news_matrix[batch_hist_ids]
            
            # 2. Look up features for Candidate
            # Result: [Batch, 32]
            cand_feats = news_matrix[batch_cand_id]
            
            # 3. Unpack (Slicing on GPU is instant)
            # Features: Title (0-29), Cat (30), Subcat (31)
            
            h_titles = hist_feats[:, :, :TITLE_LEN]
            h_cats = hist_feats[:, :, TITLE_LEN]
            h_subcats = hist_feats[:, :, TITLE_LEN+1]
            
            c_title = cand_feats[:, :TITLE_LEN]
            c_cat = cand_feats[:, TITLE_LEN]
            c_subcat = cand_feats[:, TITLE_LEN+1]
            
            # --- Forward Pass ---
            optimizer.zero_grad()
            scores = model(h_titles, h_cats, h_subcats, c_title, c_cat, c_subcat)
            loss = criterion(scores, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()