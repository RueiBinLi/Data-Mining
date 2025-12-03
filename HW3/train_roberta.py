import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import os

# --- Import your RoBERTa model class ---
# Ensure model.py has HybridRobertaModel or RobertaModel defined
from model_roberta import HybridRobertaModel 

BATCH_SIZE = 512        # High batch size for speed (since vectors are pre-computed)
LEARNING_RATE = 0.001
EPOCHS = 8
EMBED_DIM = 256         # Internal model dimension
ROBERTA_DIM = 768       # Input dimension from RoBERTa-base

# Optimization
TRAIN_HISTORY_LEN = 20  # Use last 20 clicks for training (Faster & Sufficient)

# Paths
PROCESSED_PATH = 'processed'
ROBERTA_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_roberta_embeddings.npy')
CAT_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_cat_subcat.npy')
USER_DATA_PATH = os.path.join(PROCESSED_PATH, 'user_data.pkl')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')
MODEL_SAVE_PATH = 'roberta_model.pt'

class FastDataset(Dataset):
    """
    Efficient Dataset that only returns indices.
    All heavy lifting is done on the GPU.
    """
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
    
    # 1. Load Vocabularies (To get Category sizes for the model)
    print("Loading Vocabs...")
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: {VOCAB_PATH} not found.")
        return

    with open(VOCAB_PATH, 'rb') as f:
        v = pickle.load(f)
        # We need these sizes to initialize the Embedding layers
        cat_vocab_size = len(v.get('cat2idx', {})) + 1     # +1 for safety/padding
        subcat_vocab_size = len(v.get('subcat2idx', {})) + 1
        print(f"Vocab Sizes - Cat: {cat_vocab_size}, Subcat: {subcat_vocab_size}")

    # 2. Load Matrices to GPU
    # This is the "Speed Secret": Keep data on VRAM to avoid CPU-GPU transfer bottlenecks
    print("Loading Feature Matrices to GPU...")
    
    # Load RoBERTa Vectors (Float32)
    if not os.path.exists(ROBERTA_MATRIX_PATH):
        print("Error: RoBERTa matrix missing.")
        return
    r_matrix_np = np.load(ROBERTA_MATRIX_PATH)
    r_matrix = torch.tensor(r_matrix_np, dtype=torch.float32).to(device)
    
    # Load Category Indices (Int64/Long)
    if not os.path.exists(CAT_MATRIX_PATH):
        print("Error: Category matrix missing.")
        return
    c_matrix_np = np.load(CAT_MATRIX_PATH)
    # Shape: [Num_News, 2] -> Col 0: Cat, Col 1: Subcat
    c_matrix = torch.tensor(c_matrix_np, dtype=torch.long).to(device)

    # 3. Setup DataLoader
    print("Loading User Data...")
    dataset = FastDataset(USER_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 4. Initialize Model
    print("Initializing HybridRobertaModel...")
    model = HybridRobertaModel(
        cat_vocab=cat_vocab_size,
        subcat_vocab=subcat_vocab_size,
        roberta_dim=ROBERTA_DIM,
        embed_dim=EMBED_DIM
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Gradient Scaler for Mixed Precision (Faster training)
    scaler = torch.cuda.amp.GradScaler() 
    
    print("--- Starting Training ---")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_hist_ids, batch_cand_id, batch_labels in loop:
            # Move Indices to GPU
            batch_hist_ids = batch_hist_ids.to(device)
            batch_cand_id = batch_cand_id.to(device)
            batch_labels = batch_labels.float().to(device)
            
            # --- Optimization: Slice History ---
            # We only train on the last N clicks to save compute
            batch_hist_ids = batch_hist_ids[:, -TRAIN_HISTORY_LEN:]
            
            # --- GPU-Side Feature Lookup ---
            
            # A. Look up RoBERTa Vectors
            # History: [Batch, 20, 768]
            # Candidate: [Batch, 768]
            h_vecs = r_matrix[batch_hist_ids]
            c_vec = r_matrix[batch_cand_id]
            
            # B. Look up Categories
            # c_matrix is [Num_News, 2]. We slice it with IDs.
            h_cats_all = c_matrix[batch_hist_ids] # [Batch, 20, 2]
            c_cats_all = c_matrix[batch_cand_id]  # [Batch, 2]
            
            # Separate Category (Col 0) and Subcategory (Col 1)
            h_cat = h_cats_all[:, :, 0]
            h_sub = h_cats_all[:, :, 1]
            
            c_cat = c_cats_all[:, 0]
            c_sub = c_cats_all[:, 1]
            
            # --- Forward Pass ---
            optimizer.zero_grad()
            
            # Use Mixed Precision (Autocast)
            with torch.cuda.amp.autocast():
                # Pass all 6 features to the Hybrid Model
                scores = model(h_vecs, h_cat, h_sub, c_vec, c_cat, c_sub)
                loss = criterion(scores, batch_labels)
            
            # --- Backward Pass ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(dataloader):.4f}")
        
    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()