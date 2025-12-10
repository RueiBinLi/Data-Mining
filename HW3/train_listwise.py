import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os
from model_roberta_best import HybridRobertaModel 

# --- Config ---
BATCH_SIZE = 128        # Smaller batch size because each sample is now 5x bigger (1+4 news)
LEARNING_RATE = 1e-4
EPOCHS = 5
EMBED_DIM = 256
ROBERTA_DIM = 768
TRAIN_HISTORY_LEN = 20

# Paths
PROCESSED_PATH = 'processed'
ROBERTA_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_roberta_embeddings.npy')
CAT_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_cat_subcat.npy')
LISTWISE_DATA = os.path.join(PROCESSED_PATH, 'train_data_listwise.pkl')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')
MODEL_SAVE_PATH = 'roberta_listwise.pt'

class ListwiseDataset(Dataset):
    def __init__(self):
        with open(LISTWISE_DATA, 'rb') as f:
            data = pickle.load(f)
        self.history = data['history']    
        self.pos = data['pos'] 
        self.negs = data['negs'] # [N, 4]       
        
    def __len__(self): return len(self.pos)
    
    def __getitem__(self, idx):
        return self.history[idx], self.pos[idx], self.negs[idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Vocabs
    with open(VOCAB_PATH, 'rb') as f:
        v = pickle.load(f)
        cat_len = len(v.get('cat2idx', {})) + 1
        subcat_len = len(v.get('subcat2idx', {})) + 1

    # Load Matrices
    print("Loading Matrices...")
    r_matrix = torch.tensor(np.load(ROBERTA_MATRIX_PATH), dtype=torch.float32).to(device)
    c_matrix = torch.tensor(np.load(CAT_MATRIX_PATH), dtype=torch.long).to(device)

    # DataLoader
    print("Loading Listwise Data...")
    dataset = ListwiseDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = HybridRobertaModel(cat_len, subcat_len, ROBERTA_DIM, EMBED_DIM).to(device)
    
    # CRITICAL CHANGE: CrossEntropyLoss for Ranking
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    print("--- Starting Listwise Training ---")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Ep {epoch+1}")
        
        for h_ids, pos_id, neg_ids in loop:
            # h_ids: [Batch, 50]
            # pos_id: [Batch]
            # neg_ids: [Batch, 4]
            
            h_ids = h_ids.to(device)[:, -TRAIN_HISTORY_LEN:]
            pos_id = pos_id.to(device)
            neg_ids = neg_ids.to(device)
            
            # 1. Combine Pos and Negs into one "Candidates" tensor
            # Shape: [Batch, 5] -> (Col 0 is Pos, Cols 1-4 are Negs)
            cands = torch.cat([pos_id.unsqueeze(1), neg_ids], dim=1) # [Batch, 5]
            
            # Flatten to feed into model: [Batch*5]
            batch_size = cands.shape[0]
            cands_flat = cands.view(-1)
            
            # 2. Replicate History
            # We need history for every candidate.
            # Shape: [Batch, 1, 20] -> [Batch, 5, 20] -> [Batch*5, 20]
            h_ids_expanded = h_ids.unsqueeze(1).repeat(1, 5, 1).view(-1, TRAIN_HISTORY_LEN)
            
            # 3. Lookup Features
            h_vecs = r_matrix[h_ids_expanded]
            c_vecs = r_matrix[cands_flat]
            
            h_cats = c_matrix[h_ids_expanded]
            c_cats = c_matrix[cands_flat]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # Get Scores for all 5 candidates
                # Output: [Batch*5]
                scores_flat = model(h_vecs, h_cats[:,:,0], h_cats[:,:,1], 
                                    c_vecs, c_cats[:,0], c_cats[:,1])
                
                # Reshape back to [Batch, 5]
                scores = scores_flat.view(batch_size, 5)
                
                # Target is always index 0 (because we put Positive at index 0)
                targets = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                loss = criterion(scores, targets)
            
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