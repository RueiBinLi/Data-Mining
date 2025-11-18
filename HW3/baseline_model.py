import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# --- 1. Define the Dataset Class ---

class NewsDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the preprocessed data.
    """
    def __init__(self, train_data_path, news_lookup_path):
        print(f"Loading data from {train_data_path}...")
        self.train_data = pd.read_pickle(train_data_path)
        
        print(f"Loading news lookup from {news_lookup_path}...")
        self.news_lookup = pickle.load(open(news_lookup_path, 'rb'))
        
    def __len__(self):
        return len(self.train_data)
        
    def __getitem__(self, idx):
        # 1. Get the row
        row = self.train_data.iloc[idx]
        
        # 2. Get label
        label = row['label']
        
        # 3. Get candidate news features
        candidate_news_id = row['candidate_news']
        candidate_data = self.news_lookup.get(candidate_news_id, self.news_lookup['PAD'])
        
        cand_title = torch.tensor(candidate_data['title'], dtype=torch.long)
        cand_cat = torch.tensor(candidate_data['category'], dtype=torch.long)
        cand_subcat = torch.tensor(candidate_data['subcategory'], dtype=torch.long)
        
        # 4. Get user history features
        history_news_ids = row['clicked_news'] # This is already padded
        
        history_titles = []
        history_cats = []
        history_subcats = []
        
        for news_id in history_news_ids:
            news_data = self.news_lookup.get(news_id, self.news_lookup['PAD'])
            history_titles.append(news_data['title'])
            history_cats.append(news_data['category'])
            history_subcats.append(news_data['subcategory'])
        
        hist_titles = torch.tensor(np.array(history_titles), dtype=torch.long)
        hist_cats = torch.tensor(np.array(history_cats), dtype=torch.long)
        hist_subcats = torch.tensor(np.array(history_subcats), dtype=torch.long)
        
        return (
            hist_titles, hist_cats, hist_subcats,
            cand_title, cand_cat, cand_subcat,
            torch.tensor(label, dtype=torch.float32)
        )
    
# --- 2. Define the Model Architecture ---

class AdditiveAttention(nn.Module):
    """
    A simple additive attention mechanism to weigh the user's history.
    """
    def __init__(self, query_dim, embed_dim):
        super().__init__()
        self.query_project = nn.Linear(query_dim, embed_dim, bias=False)
        self.context_project = nn.Linear(embed_dim, embed_dim, bias=False)
        self.final_project = nn.Linear(embed_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, context):
        # query: [batch, query_dim] -> (e.g., the candidate news embedding)
        # context: [batch, history_len, embed_dim] -> (e.g., the history embeddings)
        
        # Project query and context
        query_proj = self.query_project(query).unsqueeze(1) # [batch, 1, embed_dim]
        context_proj = self.context_project(context)       # [batch, history_len, embed_dim]
        
        # Calculate attention scores
        scores = self.final_project(self.tanh(query_proj + context_proj)) # [batch, history_len, 1]
        
        # Apply softmax to get weights
        weights = self.softmax(scores) # [batch, history_len, 1]
        
        # Calculate weighted sum
        weighted_context = torch.sum(weights * context, dim=1) # [batch, embed_dim]
        return weighted_context

class NewsEncoder(nn.Module):
    """
    Encodes a single news article into a vector.
    Follows the "Baseline Approach" .
    """
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, 
                 embed_dim=128, title_dim=32, cat_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Word embeddings for title
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # We will average the word embeddings, so we need a layer to reduce dim
        self.title_reduce = nn.Linear(embed_dim, title_dim)
        
        # Category embeddings
        self.cat_embedding = nn.Embedding(cat_vocab_size, cat_dim, padding_idx=0)
        
        # Subcategory embeddings
        self.subcat_embedding = nn.Embedding(subcat_vocab_size, cat_dim, padding_idx=0)
        
        # Final layer to combine features
        total_dim = title_dim + cat_dim + cat_dim
        self.final_layer = nn.Linear(total_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, title, category, subcategory):
        # title: [batch, max_title_len]
        # category: [batch]
        # subcategory: [batch]
        
        # 1. Title: Embed words and take the mean
        title_embed = self.word_embedding(title) # [batch, max_title_len, embed_dim]
        title_vec = torch.mean(title_embed, dim=1) # [batch, embed_dim]
        title_vec = self.relu(self.title_reduce(title_vec)) # [batch, title_dim]
        
        # 2. Category
        cat_vec = self.cat_embedding(category) # [batch, cat_dim]
        
        # 3. Subcategory
        subcat_vec = self.subcat_embedding(subcategory) # [batch, cat_dim]
        
        # 4. Concatenate and produce final embedding
        combined_vec = torch.cat([title_vec, cat_vec, subcat_vec], dim=1)
        news_embedding = self.relu(self.final_layer(combined_vec)) # [batch, embed_dim]
        
        return news_embedding

class UserEncoder(nn.Module):
    """
    Encodes a user's history into a vector.
    Follows the "Baseline Approach" .
    """
    def __init__(self, embed_dim):
        super().__init__()
        # We use simple averaging for the baseline
        # A better method (like the one commented out) uses attention
        pass

    def forward(self, history_embeddings):
        # history_embeddings: [batch, max_history_len, embed_dim]
        
        # Simple averaging
        user_embedding = torch.mean(history_embeddings, dim=1) # [batch, embed_dim]
        return user_embedding

class BaselineModel(nn.Module):
    """
    The main two-tower model.
    """
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, embed_dim=128):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, cat_vocab_size, subcat_vocab_size, embed_dim)
        self.user_encoder = UserEncoder(embed_dim)
        
    def forward(self, hist_titles, hist_cats, hist_subcats, 
                cand_title, cand_cat, cand_subcat):
        
        # --- Candidate News Tower ---
        # 1. Encode the candidate news
        # [batch, embed_dim]
        cand_embedding = self.news_encoder(cand_title, cand_cat, cand_subcat)
        
        # --- User History Tower ---
        # 2. Encode all news in the history
        batch_size, history_len, title_len = hist_titles.shape
        
        # We need to "flatten" the history to pass it through the news encoder
        hist_titles_flat = hist_titles.view(batch_size * history_len, title_len)
        hist_cats_flat = hist_cats.view(batch_size * history_len)
        hist_subcats_flat = hist_subcats.view(batch_size * history_len)
        
        # [batch * history_len, embed_dim]
        history_embeddings_flat = self.news_encoder(hist_titles_flat, hist_cats_flat, hist_subcats_flat)
        
        # Reshape back to [batch, history_len, embed_dim]
        history_embeddings = history_embeddings_flat.view(batch_size, history_len, -1)

        # 3. Encode the user's history
        # [batch, embed_dim]
        user_embedding = self.user_encoder(history_embeddings)
        
        # --- Prediction ---
        # 4. Calculate dot product similarity [cite: 105]
        # [batch]
        score = torch.sum(user_embedding * cand_embedding, dim=1)
        
        return score
    
# --- 3. Main Training Script ---

# --- Hyperparameters ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 3
EMBED_DIM = 128 # The main dimension for news and user embeddings

# --- File Paths ---
TRAIN_DATA_PATH = 'processed/train_data.pkl'
NEWS_LOOKUP_PATH = 'processed/news_lookup.pkl'
VOCAB_PATH = 'processed/vocabs.pkl'
MODEL_SAVE_PATH = 'baseline_model.pt'

def load_vocabs(vocab_path):
    """Loads the vocabulary file."""
    print(f"Loading vocabs from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocabs = pickle.load(f)
    print(f"Vocab sizes: Word={len(vocabs['word'])}, Cat={len(vocabs['category'])}, SubCat={len(vocabs['subcategory'])}")
    return vocabs

def main():
    # Set device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Vocabs
    vocabs = load_vocabs(VOCAB_PATH)
    vocab_size = len(vocabs['word'])
    cat_vocab_size = len(vocabs['category'])
    subcat_vocab_size = len(vocabs['subcategory'])
    
    # 2. Create Dataset and DataLoader
    dataset = NewsDataset(TRAIN_DATA_PATH, NEWS_LOOKUP_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 3. Initialize Model
    model = BaselineModel(
        vocab_size,
        cat_vocab_size,
        subcat_vocab_size,
        embed_dim=EMBED_DIM
    ).to(device)
    
    # 4. Define Loss and Optimizer
    # BCEWithLogitsLoss is perfect here. It takes raw scores (like our dot product)
    # and applies a sigmoid internally. It's more numerically stable.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        total_loss = 0.0
        
        # Use tqdm for a progress bar
        for i, (hist_titles, hist_cats, hist_subcats, 
                cand_title, cand_cat, cand_subcat, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            
            # Move all data to the device
            hist_titles = hist_titles.to(device)
            hist_cats = hist_cats.to(device)
            hist_subcats = hist_subcats.to(device)
            cand_title = cand_title.to(device)
            cand_cat = cand_cat.to(device)
            cand_subcat = cand_subcat.to(device)
            labels = labels.to(device)
            
            # Forward pass
            scores = model(hist_titles, hist_cats, hist_subcats, 
                           cand_title, cand_cat, cand_subcat)
            
            # Calculate loss
            loss = criterion(scores, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 1000 == 0: # Print loss every 1000 batches
                print(f"  Batch {i+1}/{len(dataloader)}, Avg. Loss: {total_loss / (i+1):.4f}")
        
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_epoch_loss:.4f}")

    # Save the trained model
    print(f"Training complete. Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()