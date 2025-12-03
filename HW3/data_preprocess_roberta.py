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

def load_news_manual(file_path):
    """
    Robustly loads news ID and Title, ignoring bad lines/extra tabs in abstracts.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}"):
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            # We strictly need ID (0) and Title (3)
            # 0=ID, 1=Cat, 2=Subcat, 3=Title
            if len(parts) >= 4:
                nid = parts[0]
                title = parts[3]
                data.append({'news_id': nid, 'title': title})
    return pd.DataFrame(data)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    # 1. Load AND Merge News (Using Manual Loader)
    print("Reading News Files...")
    
    # --- FIX: Use manual loader instead of read_csv ---
    df_train = load_news_manual(TRAIN_NEWS)
    df_test = load_news_manual(TEST_NEWS)
    
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
    
    # We use pandas here because behaviors usually don't have dirty text columns
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