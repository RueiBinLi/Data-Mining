import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score
from model_roberta import HybridRobertaModel 

# --- CONFIG ---
BATCH_SIZE = 512
LEARNING_RATE = 5e-5    # LOWER LR for better convergence
WEIGHT_DECAY = 1e-4     # Stronger regularization
EPOCHS = 4              # We only need 4 epochs
EMBED_DIM = 256
ROBERTA_DIM = 768
TRAIN_HISTORY_LEN = 20  # Keep 20 for training speed

# Paths
PROCESSED_PATH = 'processed'
ROBERTA_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_roberta_embeddings.npy')
CAT_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_cat_subcat.npy')
USER_DATA_PATH = os.path.join(PROCESSED_PATH, 'user_data.pkl')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')

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
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for h_ids, c_ids, labels in loader:
            h_ids = h_ids.to(device)[:, -TRAIN_HISTORY_LEN:]
            c_ids = c_ids.to(device)
            h_vecs, c_vec = r_matrix[h_ids], r_matrix[c_ids]
            h_cats, c_cats = c_matrix[h_ids], c_matrix[c_ids]
            
            scores = model(h_vecs, h_cats[:,:,0], h_cats[:,:,1], 
                           c_vec, c_cats[:,0], c_cats[:,1])
            all_scores.extend(torch.sigmoid(scores).cpu().numpy())
            all_labels.extend(labels.numpy())
    try: return roc_auc_score(all_labels, all_scores)
    except: return 0.5

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open(VOCAB_PATH, 'rb') as f:
        v = pickle.load(f)
        cat_len = len(v.get('cat2idx', {})) + 1
        subcat_len = len(v.get('subcat2idx', {})) + 1

    print("Loading Matrices...")
    r_matrix = torch.tensor(np.load(ROBERTA_MATRIX_PATH), dtype=torch.float32).to(device)
    c_matrix = torch.tensor(np.load(CAT_MATRIX_PATH), dtype=torch.long).to(device)

    print("Splitting Data...")
    h, c, l = load_data()
    full_dataset = FastDataset(h, c, l)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = HybridRobertaModel(cat_len, subcat_len, ROBERTA_DIM, EMBED_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    print("--- Starting Training (Saving ALL Epochs) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1} Train")
        
        for h_ids, c_ids, labels in loop:
            h_ids = h_ids.to(device)[:, -TRAIN_HISTORY_LEN:]
            c_ids = c_ids.to(device)
            labels = labels.float().to(device)
            
            h_vecs, c_vec = r_matrix[h_ids], r_matrix[c_ids]
            h_cats, c_cats = c_matrix[h_ids], c_matrix[c_ids]
            
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

        print("Validating...")
        val_auc = compute_val_auc(model, val_loader, r_matrix, c_matrix, device)
        print(f"Epoch {epoch+1} | Val AUC: {val_auc:.4f}")
        
        # SAVE EVERY MODEL
        filename = f"roberta_ep{epoch+1}.pt"
        torch.save(model.state_dict(), filename)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()