import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import pickle

# --- Configuration ---
# These paths point to your dataset files [cite: 14]
NEWS_TSV_PATH = 'train/train_news.tsv'
BEHAVIORS_TSV_PATH = 'train/train_behaviors.tsv'
ENTITY_EMBEDDING_PATH = 'train_entity_embedding.vec' # [cite: 17, 51]

# These are for the processed output
PROCESSED_NEWS_PATH = 'processed/news_lookup.pkl'
PROCESSED_TRAIN_PATH = 'processed/train_data.pkl'
VOCAB_PATH = 'processed/vocabs.pkl'

# --- Model Hyperparameters ---
# You can adjust these
MAX_TITLE_LEN = 30     # Max words in a news title
MAX_HISTORY_LEN = 50   # Max news articles in a user's click history [cite: 25]

def preprocess_news_data(news_path, max_title_len):
    """
    Reads news.tsv, builds vocabularies, and creates a
    processed news lookup dictionary.
    """
    print("1. Loading news.tsv...")
    # Define column names based on the PDF
    news_df = pd.read_csv(
        news_path,
        sep='\\t',
        header=None,
        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
               'url', 'title_entities', 'abstract_entities']
    )
    
    # --- Build Vocabularies ---
    print("2. Building vocabularies...")
    
    # Category vocab
    categories = news_df['category'].unique()
    cat_to_index = {cat: i + 1 for i, cat in enumerate(categories)} # +1 for padding
    cat_to_index['PAD'] = 0
    
    # Subcategory vocab
    subcategories = news_df['subcategory'].unique()
    subcat_to_index = {subcat: i + 1 for i, subcat in enumerate(subcategories)}
    subcat_to_index['PAD'] = 0
    
    # Word vocab (from titles)
    word_counter = Counter()
    # We can fillna here for the counter
    for title in tqdm(news_df['title'].fillna('')):
        word_counter.update(title.lower().split())
        
    # Keep only common words (e.g., count > 1)
    word_vocab = [word for word, count in word_counter.items() if count > 1]
    word_to_index = {word: i + 2 for i, word in enumerate(word_vocab)} # +2 for PAD and UNK
    word_to_index['PAD'] = 0
    word_to_index['UNK'] = 1 # For unknown words

    # --- Create Processed News Lookup ---
    print(f"3. Processing news articles (padding to {max_title_len} tokens)...")
    
    # --- FIX 1: Fill NaNs on the whole column first ---
    # This ensures row['title'] will always be a string inside the loop.
    news_df['title'] = news_df['title'].fillna('')
    
    news_lookup = {}
    for _, row in tqdm(news_df.iterrows(), total=len(news_df)):
        
        # --- FIX 2: Remove .fillna() from inside the loop ---
        title_tokens = row['title'].lower().split()
        
        # Convert words to indices
        title_indices = [word_to_index.get(word, word_to_index['UNK']) for word in title_tokens]
        
        # Pad/Truncate title
        padded_title = np.zeros(max_title_len, dtype=np.int32)
        L = min(len(title_indices), max_title_len)
        padded_title[:L] = title_indices[:L]
        
        news_lookup[row['news_id']] = {
            'category': cat_to_index.get(row['category'], cat_to_index['PAD']),
            'subcategory': subcat_to_index.get(row['subcategory'], subcat_to_index['PAD']),
            'title': padded_title
        }
        
    # Add a 'PAD' news item for padding user histories
    news_lookup['PAD'] = {
        'category': cat_to_index['PAD'],
        'subcategory': subcat_to_index['PAD'],
        'title': np.zeros(max_title_len, dtype=np.int32)
    }

    vocabs = {
        'word': word_to_index,
        'category': cat_to_index,
        'subcategory': subcat_to_index
    }
    
    return news_lookup, vocabs

def preprocess_behaviors_data(behaviors_path, max_history_len):
    """
    Reads behaviors.tsv and explodes the impressions column
    to create training samples.
    """
    print(f"1. Loading behaviors.tsv (this may take a while)...")
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\\t',
        header=None,
        names=['id', 'user_id', 'time', 'clicked_news', 'impressions']
    )
    
    print("2. Preprocessing behaviors...")
    
    # Split the space-separated strings into lists
    behaviors_df['clicked_news'] = behaviors_df['clicked_news'].fillna('').str.split()
    behaviors_df['impressions'] = behaviors_df['impressions'].fillna('').str.split()
    
    # --- Pad/Truncate Click History ---
    # We pad the *history* (clicked_news) here [cite: 25]
    def pad_history(history):
        padded_history = ['PAD'] * max_history_len
        L = min(len(history), max_history_len)
        if L > 0:
            padded_history[-L:] = history[-L:] # Get most recent L items
        return padded_history
    
    print(f"3. Padding user histories (max {max_history_len} items)...")
    behaviors_df['clicked_news'] = behaviors_df['clicked_news'].apply(pad_history)
    
    # --- Explode Impressions ---
    # This is the key step: turn 1 row with 15 impressions into 15 rows
    print("4. Exploding impressions column...")
    behaviors_df = behaviors_df.explode('impressions').reset_index(drop=True)
    
    # --- Parse Impressions ---
    # Split "{news_id}-{clicked}" 
    print("5. Parsing impressions (news_id and label)...")
    
    # 1. Split "{news_id}-{clicked}"
    impression_split = behaviors_df['impressions'].str.split('-', expand=True)
    behaviors_df['candidate_news'] = impression_split[0]
    behaviors_df['label'] = impression_split[1] # Keep as object type for now

    # 2. DROP bad rows (where label is None)
    # This is the key fix
    behaviors_df = behaviors_df.dropna(subset=['label'])

    # 3. NOW it's safe to convert to integer
    behaviors_df['label'] = behaviors_df['label'].astype(int)
    
    # Select and rename final columns
    final_data = behaviors_df[[
        'id', 
        'user_id', 
        'clicked_news', 
        'candidate_news', 
        'label'
    ]]
    
    return final_data

def main():
    # --- Part 1: Process News ---
    print("--- Starting News Preprocessing ---")
    news_lookup, vocabs = preprocess_news_data(NEWS_TSV_PATH, MAX_TITLE_LEN)
    
    # Save the processed news and vocabs
    print(f"Saving news lookup to {PROCESSED_NEWS_PATH}")
    with open(PROCESSED_NEWS_PATH, 'wb') as f:
        pickle.dump(news_lookup, f)
        
    print(f"Saving vocabs to {VOCAB_PATH}")
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocabs, f)
        
    print(f"Vocab sizes: Word={len(vocabs['word'])}, Cat={len(vocabs['category'])}, SubCat={len(vocabs['subcategory'])}")

    # --- Part 2: Process Behaviors ---
    print("\n--- Starting Behaviors Preprocessing ---")
    train_data = preprocess_behaviors_data(BEHAVIORS_TSV_PATH, MAX_HISTORY_LEN)
    
    # Save the final training data
    print(f"Saving training data to {PROCESSED_TRAIN_PATH}")
    train_data.to_pickle(PROCESSED_TRAIN_PATH)
    
    print("\nPreprocessing complete!")
    print(f"Final training data has {len(train_data)} samples.")
    print("Head of training data:")
    print(train_data.head())

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs('processed', exist_ok=True)
    
    main()