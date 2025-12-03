import torch
import torch.nn as nn

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

class NewsEncoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=256):
        super().__init__()
        # Input: 768 (RoBERTa vector)
        # Output: 256 (Internal Model Dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, news_embeddings):
        return self.projection(news_embeddings)

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

class RobertaBaselineModel(nn.Module):
    def __init__(self, roberta_dim=768, embed_dim=256):
        super().__init__()
        self.news_encoder = NewsEncoder(roberta_dim, embed_dim)
        self.user_encoder = UserEncoder(embed_dim)

    def forward(self, hist_vecs, cand_vec):
        # Flatten history for efficient projection
        batch, h_len, dim = hist_vecs.shape
        
        hist_vecs_flat = hist_vecs.view(-1, dim)
        hist_emb_flat = self.news_encoder(hist_vecs_flat)
        hist_emb = hist_emb_flat.view(batch, h_len, -1)
        
        cand_emb = self.news_encoder(cand_vec)
        user_emb = self.user_encoder(hist_emb)
        
        score = torch.sum(user_emb * cand_emb, dim=1)
        return score