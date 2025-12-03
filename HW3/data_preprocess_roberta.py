import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import os
import pickle

# --- Config ---
TRAIN_NEWS = 'train/train_news.tsv'
TEST_NEWS = 'test/test_news.tsv'
TRAIN_BEHAVIORS = 'train/train_behaviors.tsv'

PROCESSED_PATH = 'processed'
OUTPUT_MATRIX = os.path.join(PROCESSED_PATH, 'news_roberta_embeddings.npy')
VOCAB_PATH = os.path.join(PROCESSED_PATH, 'vocabs.pkl')
USER_DATA_PATH = os.path.join(PROCESSED_PATH, 'user_data.pkl')

BATCH_SIZE = 128
MODEL_NAME = 'roberta-base'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    # 1. Load AND Merge News
    print("Reading News Files...")
    cols = ['news_id', 'cat', 'subcat', 'title', 'abs', 'url', 'te', 'ae']
    
    # Use manual parsing or strictly formatted read to avoid errors
    # quoting=3 is often safer for these TSVs
    df_train = pd.read_csv(TRAIN_NEWS, sep='\t', header=None, names=cols, quoting=3)
    df_test = pd.read_csv(TEST_NEWS, sep='\t', header=None, names=cols, quoting=3)
    
    # Concatenate and Drop Duplicates
    print("Merging News...")
    all_news = pd.concat([df_train, df_test]).drop_duplicates(subset=['news_id'])
    all_news['title'] = all_news['title'].fillna('')
    print(f"Total Unique News: {len(all_news)}")

    # 2. Create Global ID Mapping
    print("Creating Global ID Mapping...")
    news_ids = all_news['news_id'].values
    nid2idx = {nid: i+1 for i, nid in enumerate(news_ids)}
    nid2idx['PAD'] = 0
    
    # Save Mapping (Crucial for Inference!)
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({'nid2idx': nid2idx}, f)

    # 3. Generate RoBERTa Embeddings
    print(f"Loading {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    num_news = len(all_news)
    embedding_matrix = np.zeros((num_news + 1, 768), dtype=np.float32)
    
    titles = all_news['title'].tolist()
    
    print("Generating Embeddings (This takes time)...")
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
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Map to correct index (i+1 because 0 is PAD)
            start_idx = i + 1
            end_idx = start_idx + len(batch_titles)
            embedding_matrix[start_idx : end_idx] = cls_embeddings

    print(f"Saving Matrix to {OUTPUT_MATRIX}...")
    np.save(OUTPUT_MATRIX, embedding_matrix)

    # 4. Process Training Behaviors (Using the GLOBAL Mapping)
    print("Reprocessing Training Behaviors (Indices changed)...")
    behaviors = pd.read_csv(TRAIN_BEHAVIORS, sep='\t', header=None, 
                            names=['id','uid','time','hist','imp'], low_memory=False)
    
    # Parsing Helper
    def parse_hist(h):
        if pd.isna(h): return [0]*50
        ids = [nid2idx.get(x, 0) for x in h.split()[-50:]]
        return [0]*(50-len(ids)) + ids

    tqdm.pandas()
    behaviors['h_idx'] = behaviors['hist'].progress_apply(parse_hist)
    
    # Explode Impressions
    behaviors['imp'] = behaviors['imp'].fillna('').str.split()
    behaviors = behaviors.explode('imp').dropna(subset=['imp'])
    
    # Split Label
    split = behaviors['imp'].str.split('-', expand=True)
    behaviors['cand_nid'] = split[0]
    behaviors['label'] = split[1]
    
    # Drop Bad Rows
    behaviors = behaviors.dropna(subset=['label'])
    behaviors['label'] = behaviors['label'].astype(int)
    behaviors['c_idx'] = behaviors['cand_nid'].map(nid2idx).fillna(0).astype(int)
    
    # Save User Data
    print("Saving New User Data...")
    with open(USER_DATA_PATH, 'wb') as f:
        pickle.dump({
            'history': np.stack(behaviors['h_idx'].values),
            'candidate': behaviors['c_idx'].values,
            'label': behaviors['label'].values
        }, f)
        
    print("Done! Global Data Ready.")

if __name__ == "__main__":
    main()