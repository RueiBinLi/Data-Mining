import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from model import BaselineModel # Import from your new model file

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 0.0001 # Lower LR for stability
EPOCHS = 8
EMBED_DIM = 256

TRAIN_DATA_PATH = 'processed/train_data.pkl'
NEWS_LOOKUP_PATH = 'processed/news_lookup.pkl'
VOCAB_PATH = 'processed/vocabs.pkl'
MODEL_SAVE_PATH = 'baseline_model.pt'

NEWS_MATRIX_PATH = 'processed/news_features.npy'
USER_DATA_PATH = 'processed/user_data.pkl'

class NewsDataset(Dataset):
    def __init__(self, news_matrix_path, user_data_path):
        # 1. Load the Big News Matrix (Int32)
        # Shape: [Num_News, 32]
        self.news_matrix = torch.tensor(
            np.load(news_matrix_path), dtype=torch.long
        )
        
        # 2. Load the User Behaviors (Indices)
        with open(user_data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.history = torch.tensor(data['history'], dtype=torch.long)   # [N, 50]
        self.candidate = torch.tensor(data['candidate'], dtype=torch.long) # [N]
        self.label = torch.tensor(data['label'], dtype=torch.float32)      # [N]
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        # --- INSTANT LOOKUP (No Loops) ---
        
        # 1. Get Indices
        hist_indices = self.history[idx] # [50]
        cand_index = self.candidate[idx] # Scalar
        
        # 2. Slice the News Matrix (Fast GPU/CPU operation)
        # Result: [50, 32] and [32]
        hist_data = self.news_matrix[hist_indices] 
        cand_data = self.news_matrix[cand_index]
        
        # 3. Unpack Features (Title=0-29, Cat=30, Subcat=31)
        # Note: Adjust 30 based on your MAX_TITLE_LEN
        TITLE_LEN = 30
        
        # History
        h_titles = hist_data[:, :TITLE_LEN]
        h_cats = hist_data[:, TITLE_LEN]
        h_subcats = hist_data[:, TITLE_LEN+1]
        
        # Candidate
        c_title = cand_data[:TITLE_LEN]
        c_cat = cand_data[TITLE_LEN]
        c_subcat = cand_data[TITLE_LEN+1]
        
        return h_titles, h_cats, h_subcats, c_title, c_cat, c_subcat, self.label[idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Vocabs
    with open(VOCAB_PATH, 'rb') as f:
        vocabs = pickle.load(f)
    
    # Dataset
    dataset = NewsDataset(NEWS_MATRIX_PATH, USER_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Model
    model = BaselineModel(
        len(vocabs['word']), len(vocabs['category']), len(vocabs['subcategory']), EMBED_DIM
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            batch = [x.to(device) for x in batch]
            *inputs, labels = batch
            
            optimizer.zero_grad()
            scores = model(*inputs)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()