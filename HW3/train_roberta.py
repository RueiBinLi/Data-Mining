import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
from model_roberta import RobertaBaselineModel

# --- Configuration ---
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 8
ROBERTA_DIM = 768 # Standard for base models
EMBED_DIM = 256

# Paths
NEWS_MATRIX_PATH = 'processed/news_roberta_embeddings.npy' 
USER_DATA_PATH = 'processed/user_data.pkl' # This stays the same!
MODEL_SAVE_PATH = 'roberta_model.pt'

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
    print("Loading RoBERTa Matrix...")
    news_matrix_np = np.load(NEWS_MATRIX_PATH)
    # Important: float32 for pre-computed vectors
    news_matrix = torch.tensor(news_matrix_np, dtype=torch.float32).to(device)
    
    # 3. Dataset & DataLoader
    print("Initializing DataLoader...")
    dataset = FastNewsDataset(USER_DATA_PATH)
    # num_workers=0 is often FASTER when simply passing integers from memory
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 4. Model
    model = RobertaBaselineModel(ROBERTA_DIM, EMBED_DIM).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"--- Starting Training (Batch Size {BATCH_SIZE}) ---")
    
    scaler = torch.cuda.amp.GradScaler() 

    # 2. Reduce History Length (Config)
    TRAIN_HISTORY_LEN = 20 # Use only last 20 clicks for training (Speed hack)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_hist_ids, batch_cand_id, batch_labels in loop:
            # Move Indices to GPU
            batch_hist_ids = batch_hist_ids.to(device)
            batch_cand_id = batch_cand_id.to(device)
            batch_labels = batch_labels.float().to(device)
            
            # --- OPTIMIZATION: Slice History ---
            # Instead of 50, we take the last 20. 
            # Since we padded with 0s at the start, slicing [-20:] keeps the real data.
            batch_hist_ids = batch_hist_ids[:, -TRAIN_HISTORY_LEN:]
            
            # Lookup Features (Same as before)
            hist_feats = news_matrix[batch_hist_ids]
            cand_feats = news_matrix[batch_cand_id]
            
            h_titles = hist_feats[:, :, :30]
            h_cats = hist_feats[:, :, 30]
            h_subcats = hist_feats[:, :, 31]
            
            c_title = cand_feats[:, :30]
            c_cat = cand_feats[:, 30]
            c_subcat = cand_feats[:, 31]
            
            optimizer.zero_grad()
            
            # --- OPTIMIZATION: Mixed Precision ---
            with torch.cuda.amp.autocast():
                scores = model(h_titles, h_cats, h_subcats, c_title, c_cat, c_subcat)
                loss = criterion(scores, batch_labels)
            
            # Scaled Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()