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
EPOCHS = 5
EMBED_DIM = 256

TRAIN_DATA_PATH = 'processed/train_data.pkl'
NEWS_LOOKUP_PATH = 'processed/news_lookup.pkl'
VOCAB_PATH = 'processed/vocabs.pkl'
MODEL_SAVE_PATH = 'baseline_model.pt'

class NewsDataset(Dataset):
    def __init__(self, train_data_path, news_lookup_path):
        self.train_data = pd.read_pickle(train_data_path)
        self.news_lookup = pickle.load(open(news_lookup_path, 'rb'))
        
    def __len__(self):
        return len(self.train_data)
        
    def __getitem__(self, idx):
        row = self.train_data.iloc[idx]
        label = row['label']
        
        # Candidate
        candidate_data = self.news_lookup.get(row['candidate_news'], self.news_lookup['PAD'])
        cand_title = torch.tensor(candidate_data['title'], dtype=torch.long)
        cand_cat = torch.tensor(candidate_data['category'], dtype=torch.long)
        cand_subcat = torch.tensor(candidate_data['subcategory'], dtype=torch.long)
        
        # History
        history_titles, history_cats, history_subcats = [], [], []
        for news_id in row['clicked_news']:
            d = self.news_lookup.get(news_id, self.news_lookup['PAD'])
            history_titles.append(d['title'])
            history_cats.append(d['category'])
            history_subcats.append(d['subcategory'])
        
        return (
            torch.tensor(np.array(history_titles), dtype=torch.long),
            torch.tensor(np.array(history_cats), dtype=torch.long),
            torch.tensor(np.array(history_subcats), dtype=torch.long),
            cand_title, cand_cat, cand_subcat,
            torch.tensor(label, dtype=torch.float32)
        )

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Vocabs
    with open(VOCAB_PATH, 'rb') as f:
        vocabs = pickle.load(f)
    
    # Dataset
    dataset = NewsDataset(TRAIN_DATA_PATH, NEWS_LOOKUP_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
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