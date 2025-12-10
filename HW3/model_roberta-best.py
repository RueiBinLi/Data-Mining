import torch
import torch.nn as nn

class NativeMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out

class AdditiveAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=200):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.Tanh()
        self.ctx = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, x):
        scores = self.ctx(self.act(self.proj(x)))
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * x, dim=1)

class NewsEncoder(nn.Module):
    def __init__(self, cat_vocab, subcat_vocab, input_dim=768, output_dim=256, cat_dim=64):
        super().__init__()
        # 1. RoBERTa Projector
        self.roberta_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # 2. Category Embeddings
        self.cat_emb = nn.Embedding(cat_vocab, cat_dim, padding_idx=0)
        self.subcat_emb = nn.Embedding(subcat_vocab, cat_dim, padding_idx=0)
        
        # 3. Final Fusion
        # 256 (Text) + 64 (Cat) + 64 (Subcat) = 384
        self.fusion = nn.Linear(output_dim + cat_dim + cat_dim, output_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, roberta_vec, cat_id, subcat_id):
        text_vec = self.roberta_proj(roberta_vec) # [batch, 256]
        c_vec = self.cat_emb(cat_id)             # [batch, 64]
        s_vec = self.subcat_emb(subcat_id)       # [batch, 64]
        
        combined = torch.cat([text_vec, c_vec, s_vec], dim=1)
        return self.act(self.fusion(combined))

class UserEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.self_attn = NativeMultiHeadAttention(embed_dim, num_heads=16)
        self.norm = nn.LayerNorm(embed_dim)
        self.add_attn = AdditiveAttention(embed_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.dropout(x)
        x = self.norm(x + self.self_attn(x))
        return self.add_attn(x)

class HybridRobertaModel(nn.Module):
    def __init__(self, cat_vocab, subcat_vocab, roberta_dim=768, embed_dim=256):
        super().__init__()
        self.news_enc = NewsEncoder(cat_vocab, subcat_vocab, roberta_dim, embed_dim)
        self.user_enc = UserEncoder(embed_dim)
        
    def forward(self, h_vecs, h_cats, h_subcats, c_vec, c_cat, c_subcat):
        # Flatten History
        batch, seq, dim = h_vecs.shape
        h_vecs_flat = h_vecs.view(-1, dim)
        h_cats_flat = h_cats.view(-1)
        h_subcats_flat = h_subcats.view(-1)
        
        # Encode History News
        h_emb_flat = self.news_enc(h_vecs_flat, h_cats_flat, h_subcats_flat)
        h_emb = h_emb_flat.view(batch, seq, -1)
        
        # Encode Candidate News
        c_emb = self.news_enc(c_vec, c_cat, c_subcat)
        
        # Encode User
        user_emb = self.user_enc(h_emb)
        
        return torch.sum(user_emb * c_emb, dim=1)