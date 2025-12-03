import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os

# --- Import your RoBERTa model class ---
# Ensure model.py has RobertaBaselineModel or RobertaModel defined
from model_roberta import RobertaBaselineModel 

# --- Config ---
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 8
EMBED_DIM = 256
ROBERTA_DIM = 768  # Input size

# Paths
NEWS_MATRIX_PATH = 'processed/news_roberta_embeddings.npy' 
USER_DATA_PATH = 'processed/user_data.pkl'
MODEL_SAVE_PATH = 'roberta_model.pt'

class FastDataset(Dataset):
    def __init__(self, user_data_path):
        with open(user_data_path, 'rb') as f:
            data = pickle.load(f)
        self.history = data['history']    
        self.candidate = data['candidate'] 
        self.label = data['label']        
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        return self.history[idx], self.candidate[idx], self.label[idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load RoBERTa Matrix (Float32)
    print("Loading RoBERTa Matrix...")
    if not os.path.exists(NEWS_MATRIX_PATH):
        print(f"Error: {NEWS_MATRIX_PATH} not found. Run preprocess first.")
        return
        
    news_matrix_np = np.load(NEWS_MATRIX_PATH)
    # Move huge matrix to GPU
    news_matrix = torch.tensor(news_matrix_np, dtype=torch.float32).to(device)
    
    # 2. DataLoader
    print("Loading User Data...")
    dataset = FastDataset(USER_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 3. Model
    # Note: We pass 768 (input) and 256 (internal dim)
    model = RobertaModel(ROBERTA_DIM, EMBED_DIM).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') # Updated for newer PyTorch
    
    print("--- Starting Training ---")
    
    TRAIN_HISTORY_LEN = 20 # Optimization
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_hist_ids, batch_cand_id, batch_labels in loop:
            batch_hist_ids = batch_hist_ids.to(device)
            batch_cand_id = batch_cand_id.to(device)
            batch_labels = batch_labels.float().to(device)
            
            # 1. Slice History (Optimization)
            # Take only the last 20 clicked items
            batch_hist_ids = batch_hist_ids[:, -TRAIN_HISTORY_LEN:]
            
            # 2. GPU Lookup (The Fix!)
            # Instead of slicing titles/cats, we grab the full 768-dim vector
            hist_vecs = news_matrix[batch_hist_ids] # [Batch, 20, 768]
            cand_vec = news_matrix[batch_cand_id]   # [Batch, 768]
            
            optimizer.zero_grad()
            
            # 3. Forward Pass (Only 2 inputs now)
            with torch.amp.autocast('cuda'):
                scores = model(hist_vecs, cand_vec)
                loss = criterion(scores, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model Saved.")

if __name__ == "__main__":
    main()