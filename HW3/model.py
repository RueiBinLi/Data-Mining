import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Verify dimension compatibility
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.W_O(output)

class AdditiveAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=200):
        super().__init__()
        self.projection = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        proj = self.activation(self.projection(x))
        scores = self.context_vector(proj)
        weights = torch.softmax(scores, dim=1)
        output = torch.sum(weights * x, dim=1)
        return output

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, 
                 embed_dim=256, cat_dim=64):
        super().__init__()
        
        # 1. Word Embedding (Learning from scratch needs size 256+)
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2) # Critical for larger models
        
        # 2. Main Title Encoder (Self-Attention)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads=16)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.additive_attn = AdditiveAttention(embed_dim)
        
        # 3. Category Features (The "Free Signal")
        self.cat_embedding = nn.Embedding(cat_vocab_size, cat_dim, padding_idx=0)
        self.subcat_embedding = nn.Embedding(subcat_vocab_size, cat_dim, padding_idx=0)
        
        # 4. Feature Fusion
        # Input: Title(256) + Cat(64) + Subcat(64) = 384
        # Output: 256 (To match User Encoder)
        self.final_proj = nn.Linear(embed_dim + cat_dim + cat_dim, embed_dim)
        self.final_act = nn.LeakyReLU(0.1)

    def forward(self, title, category, subcategory):
        # --- Title ---
        x = self.word_embedding(title)
        x = self.dropout(x)
        
        # Transformer-like block
        attn_out = self.self_attn(x)
        x = self.layer_norm(x + attn_out)
        
        title_vec = self.additive_attn(x)
        
        # --- Categories ---
        cat_vec = self.cat_embedding(category)
        subcat_vec = self.subcat_embedding(subcategory)
        
        # --- Combine ---
        combined = torch.cat([title_vec, cat_vec, subcat_vec], dim=1)
        out = self.final_proj(combined)
        out = self.final_act(out)
        
        return out

class UserEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads=16)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.additive_attn = AdditiveAttention(embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, history_embeddings):
        # history_embeddings: [batch, 50, 256]
        x = self.dropout(history_embeddings)
        
        attn_out = self.self_attn(x)
        x = self.layer_norm(x + attn_out)
        
        user_vec = self.additive_attn(x)
        return user_vec

class BaselineModel(nn.Module):
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, embed_dim=256):
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
        
    def forward(self, h_titles, h_cats, h_subcats, c_title, c_cat, c_subcat):
        cand_embed = self.news_encoder(c_title, c_cat, c_subcat)
        
        batch, h_len, t_len = h_titles.shape
        
        # Reshape to process all history news at once
        h_titles_flat = h_titles.view(-1, t_len)
        h_cats_flat = h_cats.view(-1)
        h_subcats_flat = h_subcats.view(-1)
        
        h_emb_flat = self.news_encoder(h_titles_flat, h_cats_flat, h_subcats_flat)
        h_emb = h_emb_flat.view(batch, h_len, -1)
        
        user_embed = self.user_encoder(h_emb)
        
        # Dot Product Similarity
        score = torch.sum(user_embed * cand_embed, dim=1)
        return score