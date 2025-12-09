import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score

# --- Import Hybrid Model ---
from model_roberta import HybridRobertaModel 

# --- CONFIG ---
BATCH_SIZE = 512
LEARNING_RATE = 1e-4    # LOWER Learning Rate for fine-tuning stability
WEIGHT_DECAY = 1e-5     # Regularization to fight overfitting
EPOCHS = 5              # 8 is too many. 3-5 is usually the sweet spot.
EMBED_DIM = 256
ROBERTA_DIM = 768
TRAIN_HISTORY_LEN = 20

# Paths
PROCESSED_PATH = 'processed'
ROBERTA_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_roberta_embeddings.npy')
CAT_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_cat_subcat.npy')
USER_DATA_PATH = os.path.join(PROCESSED_PATH, 'user_data.pkl') # Must be the GLOBAL one
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')
MODEL_SAVE_PATH = 'roberta_best_model.pt'

class FastDataset(Dataset):
    def __init__(self, history, candidate, label):
        self.history = history
        self.candidate = candidate
        self.label = label
    def __len__(self): return len(self.label)
    def __getitem__(self, idx): return self.history[idx], self.candidate[idx], self.label[idx]

def load_data():
    with open(USER_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['history'], data['candidate'], data['label']

def compute_val_auc(model, loader, r_matrix, c_matrix, device):
    """
    Runs inference on the validation set and calculates the true AUC score.
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for h_ids, c_ids, labels in loader:
            h_ids = h_ids.to(device)[:, -TRAIN_HISTORY_LEN:]
            c_ids = c_ids.to(device)
            
            h_vecs = r_matrix[h_ids]
            c_vec = r_matrix[c_ids]
            
            h_cats = c_matrix[h_ids]
            c_cats = c_matrix[c_ids]
            
            scores = model(h_vecs, h_cats[:,:,0], h_cats[:,:,1], 
                           c_vec, c_cats[:,0], c_cats[:,1])
            
            # Apply Sigmoid for AUC calculation
            preds = torch.sigmoid(scores).cpu().numpy()
            
            all_scores.extend(preds)
            all_labels.extend(labels.numpy())
            
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5 # Edge case if only one class exists in batch
    return auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Vocabs
    with open(VOCAB_PATH, 'rb') as f:
        v = pickle.load(f)
        cat_len = len(v.get('cat2idx', {})) + 1
        subcat_len = len(v.get('subcat2idx', {})) + 1

    # 2. Load Matrices (GPU)
    print("Loading Matrices...")
    r_matrix = torch.tensor(np.load(ROBERTA_MATRIX_PATH), dtype=torch.float32).to(device)
    c_matrix = torch.tensor(np.load(CAT_MATRIX_PATH), dtype=torch.long).to(device)

    # 3. Prepare Splits (Train vs Validation)
    print("Splitting Data...")
    h, c, l = load_data()
    full_dataset = FastDataset(h, c, l)
    
    # 90% Train, 10% Validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    # 4. Model & Optimizer
    model = HybridRobertaModel(cat_len, subcat_len, ROBERTA_DIM, EMBED_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # ADDED WEIGHT DECAY (Regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    best_auc = 0.0
    
    print("--- Starting Training ---")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1} Train")
        
        for h_ids, c_ids, labels in loop:
            h_ids = h_ids.to(device)[:, -TRAIN_HISTORY_LEN:]
            c_ids = c_ids.to(device)
            labels = labels.float().to(device)
            
            # Lookups
            h_vecs = r_matrix[h_ids]
            c_vec = r_matrix[c_ids]
            
            h_cats = c_matrix[h_ids]
            c_cats = c_matrix[c_ids]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                scores = model(h_vecs, h_cats[:,:,0], h_cats[:,:,1], 
                               c_vec, c_cats[:,0], c_cats[:,1])
                loss = criterion(scores, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- VALIDATE ---
        print("Validating...")
        val_auc = compute_val_auc(model, val_loader, r_matrix, c_matrix, device)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        # --- SAVE BEST ONLY ---
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>> New Best Model Saved! (AUC: {best_auc:.4f})")
        else:
            print(f"No improvement (Best: {best_auc:.4f})")

if __name__ == "__main__":
    main()