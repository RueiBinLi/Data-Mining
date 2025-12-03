import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import os
import pickle

# --- Config ---
NEWS_TSV_PATH = 'train/train_news.tsv'
PROCESSED_PATH = 'processed'
OUTPUT_MATRIX_PATH = os.path.join(PROCESSED_PATH, 'news_roberta_embeddings.npy')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')

BATCH_SIZE = 128
MODEL_NAME = 'roberta-base' # <--- The only big change

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    # 1. Load News
    print("Loading News...")
    news_df = pd.read_csv(
        NEWS_TSV_PATH, sep='\t', header=None,
        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
               'url', 'title_entities', 'abstract_entities']
    )
    news_df['title'] = news_df['title'].fillna('')
    
    # 2. Map News IDs
    print("Mapping News IDs...")
    news_ids = news_df['news_id'].values
    nid2idx = {nid: i+1 for i, nid in enumerate(news_ids)}
    nid2idx['PAD'] = 0
    
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({'nid2idx': nid2idx}, f)

    # 3. Load RoBERTa
    print(f"Loading {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 4. Generate Embeddings
    print("Generating RoBERTa Embeddings...")
    num_news = len(news_df)
    embedding_matrix = np.zeros((num_news + 1, 768), dtype=np.float32)
    
    titles = news_df['title'].tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, num_news, BATCH_SIZE)):
            batch_titles = titles[i : i + BATCH_SIZE]
            
            encoded = tokenizer(
                batch_titles,
                padding=True,
                truncation=True,
                max_length=30,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**encoded)
            
            # RoBERTa's "CLS" token is also at index 0 (technically <s>)
            # We use last_hidden_state[:, 0, :] just like BERT
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            start_idx = i + 1 
            end_idx = start_idx + len(batch_titles)
            embedding_matrix[start_idx : end_idx] = cls_embeddings

    print(f"Saving matrix to {OUTPUT_MATRIX_PATH}...")
    np.save(OUTPUT_MATRIX_PATH, embedding_matrix)
    print("Done!")

if __name__ == "__main__":
    main()