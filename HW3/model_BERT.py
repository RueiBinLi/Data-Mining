import torch
import torch.nn as nn
import numpy as np

# Re-use your Native Attention for the USER ENCODER
class NativeMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        output, _ = self.attn(x, x, x)
        return output

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

# --- UPDATED: Simpler NewsEncoder ---
class NewsEncoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=256):
        super().__init__()
        # Input is already a rich BERT vector (768)
        # We just project it down to match the User Encoder size
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, news_embeddings):
        # news_embeddings: [batch, 768]
        return self.projection(news_embeddings)

class UserEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Matches the output_dim of NewsEncoder
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

class BertBaselineModel(nn.Module):
    def __init__(self, bert_dim=768, embed_dim=256):
        super().__init__()
        self.news_encoder = NewsEncoder(bert_dim, embed_dim)
        self.user_encoder = UserEncoder(embed_dim)

    def forward(self, hist_bert_vecs, cand_bert_vec):
        # 1. Project News Vectors (768 -> 256)
        # Flatten history for efficient projection
        batch, h_len, bert_dim = hist_bert_vecs.shape
        
        hist_vecs_flat = hist_bert_vecs.view(-1, bert_dim)
        hist_emb_flat = self.news_encoder(hist_vecs_flat)
        hist_emb = hist_emb_flat.view(batch, h_len, -1)
        
        cand_emb = self.news_encoder(cand_bert_vec)
        
        # 2. Encode User
        user_emb = self.user_encoder(hist_emb)
        
        # 3. Score
        score = torch.sum(user_emb * cand_emb, dim=1)
        return score