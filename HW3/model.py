import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NativeMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16):
        super().__init__()
        # Use PyTorch's optimized implementation
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        # Native attention expects (query, key, value)
        # For self-attention, all three are x
        output, _ = self.attn(x, x, x)
        return output

class AdditiveAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=200):
        super().__init__()
        self.projection = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        proj = self.activation(self.projection(x))
        scores = self.context_vector(proj)
        weights = torch.softmax(scores, dim=1)
        output = torch.sum(weights * x, dim=1)
        return output

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, cat_vocab_size, subcat_vocab_size, 
                 embed_dim=256, cat_dim=64):
        super().__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        
        # USE NATIVE ATTENTION
        self.self_attn = NativeMultiHeadAttention(embed_dim, num_heads=16)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.additive_attn = AdditiveAttention(embed_dim)
        
        self.cat_embedding = nn.Embedding(cat_vocab_size, cat_dim, padding_idx=0)
        self.subcat_embedding = nn.Embedding(subcat_vocab_size, cat_dim, padding_idx=0)
        
        self.final_proj = nn.Linear(embed_dim + cat_dim + cat_dim, embed_dim)
        self.final_act = nn.LeakyReLU(0.1)

    def forward(self, title, category, subcategory):
        x = self.word_embedding(title)
        x = self.dropout(x)
        
        # Native Attention Block
        attn_out = self.self_attn(x)
        x = self.layer_norm(x + attn_out)
        
        title_vec = self.additive_attn(x)
        
        cat_vec = self.cat_embedding(category)
        subcat_vec = self.subcat_embedding(subcategory)
        
        combined = torch.cat([title_vec, cat_vec, subcat_vec], dim=1)
        out = self.final_proj(combined)
        out = self.final_act(out)
        return out

# UserEncoder and BaselineModel remain mostly the same, 
# but ensure UserEncoder also uses NativeMultiHeadAttention if you want speed there too.

class UserEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.self_attn = NativeMultiHeadAttention(embed_dim, num_heads=16)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.additive_attn = AdditiveAttention(embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, history_embeddings):
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

    def forward(self, h_titles, h_cats, h_subcats, c_title, c_cat, c_subcat):
        cand_embed = self.news_encoder(c_title, c_cat, c_subcat)
        
        batch, h_len, t_len = h_titles.shape
        h_titles_flat = h_titles.view(-1, t_len)
        h_cats_flat = h_cats.view(-1)
        h_subcats_flat = h_subcats.view(-1)
        
        h_emb_flat = self.news_encoder(h_titles_flat, h_cats_flat, h_subcats_flat)
        h_emb = h_emb_flat.view(batch, h_len, -1)
        
        user_embed = self.user_encoder(h_emb)
        score = torch.sum(user_embed * cand_embed, dim=1)
        return score