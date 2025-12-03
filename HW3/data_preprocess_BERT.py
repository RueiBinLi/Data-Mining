import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import pickle

# --- Config ---
NEWS_TSV_PATH = 'train/train_news.tsv'
PROCESSED_PATH = 'processed'
OUTPUT_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_bert_embeddings.npy')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')

BATCH_SIZE = 128 # Inference batch size
MODEL_NAME = 'bert-base-uncased' # You can swap this for 'roberta-base'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    # 1. Load News Data
    print("Loading News...")
    news_df = pd.read_csv(
        NEWS_TSV_PATH, sep='\t', header=None,
        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
               'url', 'title_entities', 'abstract_entities']
    )
    news_df['title'] = news_df['title'].fillna('')
    
    # 2. Create NewsID Mapping (Same as before)
    print("Mapping News IDs...")
    news_ids = news_df['news_id'].values
    nid2idx = {nid: i+1 for i, nid in enumerate(news_ids)}
    nid2idx['PAD'] = 0
    
    # Save vocab for training/inference mapping
    # (We don't need word_vocab anymore, BERT handles that)
    with open(VOCAB_PATH, 'wb') as f:
        # We also save cat/subcat mappings if you want to add them later
        # For this BERT model, we will rely purely on Title for simplicity/speed
        pickle.dump({'nid2idx': nid2idx}, f)

    # 3. Load BERT
    print(f"Loading {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 4. Generate Embeddings
    print("Generating BERT Embeddings...")
    num_news = len(news_df)
    # BERT base outputs 768 dim vectors
    embedding_matrix = np.zeros((num_news + 1, 768), dtype=np.float32)
    
    titles = news_df['title'].tolist()
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, num_news, BATCH_SIZE)):
            batch_titles = titles[i : i + BATCH_SIZE]
            
            # Tokenize
            encoded = tokenizer(
                batch_titles,
                padding=True,
                truncation=True,
                max_length=30,
                return_tensors='pt'
            ).to(device)
            
            # Forward Pass
            outputs = model(**encoded)
            
            # Get CLS token (first token) as sentence embedding
            # Shape: [batch, 768]
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Store in matrix
            # Offsets: IDs start at 1, so row index corresponds to i+1
            start_idx = i + 1 
            end_idx = start_idx + len(batch_titles)
            embedding_matrix[start_idx : end_idx] = cls_embeddings

    print(f"Saving matrix to {OUTPUT_MATRIX_PATH}...")
    np.save(OUTPUT_MATRIX_PATH, embedding_matrix)
    print("Done!")

if __name__ == "__main__":
    main()