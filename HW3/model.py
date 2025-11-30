import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """
    Learns to weight specific parts of the input (e.g., important words in a title,
    or important news in a user's history).
    """
    def __init__(self, embed_dim, hidden_dim=200):
        super().__init__()
        self.projection = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        
        # 1. Calculate Attention Scores
        # [batch, seq_len, hidden_dim]
        proj = self.activation(self.projection(x))
        # [batch, seq_len, 1]
        scores = self.context_vector(proj)
        
        # 2. Calculate Weights
        weights = self.softmax(scores)
        
        # 3. Weighted Sum
        # [batch, embed_dim]
        output = torch.sum(weights * x, dim=1)
        return output

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, 
                 embed_dim=128, title_dim=128, cat_dim=32):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # --- IMPROVEMENT 1: 1D CNN for Title ---
        # Captures local context (phrases) instead of just words
        self.cnn = nn.Conv1d(in_channels=embed_dim, 
                             out_channels=title_dim, 
                             kernel_size=3, 
                             padding=1)
        self.activation = nn.LeakyReLU(0.1)
        
        # --- IMPROVEMENT 2: Attention for Title ---
        # Decides which words/phrases are most important
        self.title_attention = AdditiveAttention(title_dim)
        
        self.cat_embedding = nn.Embedding(cat_vocab_size, cat_dim, padding_idx=0)
        self.subcat_embedding = nn.Embedding(subcat_vocab_size, cat_dim, padding_idx=0)
        
        self.final_layer = nn.Linear(title_dim + cat_dim + cat_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, title, category, subcategory):
        # 1. Embed Words [batch, 30, 128]
        word_vecs = self.word_embedding(title)
        
        # 2. CNN expects [batch, channels, seq_len] -> Permute
        word_vecs = word_vecs.permute(0, 2, 1)
        
        # 3. Apply CNN
        feature_map = self.activation(self.cnn(word_vecs))
        
        # 4. Permute back [batch, 30, 128]
        feature_map = feature_map.permute(0, 2, 1)
        
        # 5. Apply Attention to pool into a single vector
        title_vec = self.title_attention(feature_map)

        # 6. Categories
        cat_vec = self.cat_embedding(category)
        subcat_vec = self.subcat_embedding(subcategory)
        
        # 7. Concatenate and Project
        combined = torch.cat([title_vec, cat_vec, subcat_vec], dim=1)
        out = self.activation(self.layer_norm(self.final_layer(combined)))
        
        return out

class UserEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # --- IMPROVEMENT 3: Attention for User History ---
        # Learns which clicked news are relevant
        self.attention = AdditiveAttention(embed_dim)
        
    def forward(self, history_embeddings):
        # history_embeddings: [batch, 50, 128]
        return self.attention(history_embeddings)

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, embed_dim=128):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, cat_vocab_size, subcat_vocab_size, embed_dim)
        self.user_encoder = UserEncoder(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, hist_titles, hist_cats, hist_subcats, 
                cand_title, cand_cat, cand_subcat):
        
        # 1. Encode Candidate
        cand_embedding = self.news_encoder(cand_title, cand_cat, cand_subcat)
        
        # 2. Encode History
        batch_size, history_len, title_len = hist_titles.shape
        hist_titles_flat = hist_titles.view(batch_size * history_len, title_len)
        hist_cats_flat = hist_cats.view(batch_size * history_len)
        hist_subcats_flat = hist_subcats.view(batch_size * history_len)
        
        history_embeddings_flat = self.news_encoder(hist_titles_flat, hist_cats_flat, hist_subcats_flat)
        history_embeddings = history_embeddings_flat.view(batch_size, history_len, -1)
        
        # 3. Encode User
        user_embedding = self.user_encoder(history_embeddings)
        
        # 4. Dot Product
        score = torch.sum(user_embedding * cand_embedding, dim=1)
        return score