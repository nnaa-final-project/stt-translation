import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalFF(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): return self.fc2(self.dropout(self.relu(self.fc1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(embed_dim, embed_dim) for _ in range(4))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        bs = query.shape[0]
        query, key, value = (proj(x).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2) for proj, x in
                             [(self.q_proj, query), (self.k_proj, key), (self.v_proj, value)])
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, value).transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)
        return self.out_proj(ctx)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = PositionalFF(embed_dim, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask):
        norm_src = self.norm1(src)
        src = src + self.dropout(self.self_attn(norm_src, norm_src, norm_src, mask))
        norm_src = self.norm2(src)
        src = src + self.dropout(self.feed_forward(norm_src))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.feed_forward = PositionalFF(embed_dim, d_ff, dropout)
        self.dropout =  nn.Dropout(dropout)

    def forward(self, tgt, mem, tgt_mask, mem_mask):
        norm_tgt = self.norm1(tgt)
        tgt = tgt + self.dropout(self.self_attn(norm_tgt, norm_tgt, norm_tgt, tgt_mask))
        norm_tgt = self.norm2(tgt)
        tgt = tgt + self.dropout(self.cross_attn(norm_tgt, mem, mem, mem_mask))
        norm_tgt = self.norm3(tgt)
        tgt = tgt + self.dropout(self.feed_forward(norm_tgt))
        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, embed_dim)
        pos, div = torch.arange(max_len).unsqueeze(1), torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SpeechToTextTranslationModel(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, embed_dim: int,
                 num_attn_heads: int, tgt_vocab_size: int, d_ff: int, dropout: float,
                 input_feat_dim: int = 80):
        super().__init__()

        self.config = {
            "num_heads": num_attn_heads,
            "tgt_vocab_size": tgt_vocab_size,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "embed_dim": embed_dim,
            "d_ff": d_ff,
            "dropout": dropout,
            "input_feat_dim": input_feat_dim
        }

        self.feature_projection = nn.Linear(input_feat_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)

        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(embed_dim, num_attn_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(embed_dim, num_attn_heads, d_ff, dropout) for _ in range(num_decoder_layers)])

        self.generator = nn.Linear(embed_dim, tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)  # Assume pad token index is 1

    def _make_pad_lookahead_target_mask(self, target, device):
        target_len = target.shape[1]
        target_pad_mask = (target != 1).unsqueeze(1).unsqueeze(2)
        target_lookahead_mask = torch.triu(torch.ones((target_len, target_len), device=device), diagonal=1).bool()
        return target_pad_mask & ~target_lookahead_mask

    def forward(self, input_features, labels=None, **kwargs):
        src = self.feature_projection(input_features) * math.sqrt(self.config["embed_dim"])
        src = self.pos_encoder(src)

        memory = src
        for layer in self.encoder_stack:
            memory = layer(memory, None)

        if labels is not None:
            tgt_mask = self._make_pad_lookahead_target_mask(labels, labels.device)
            tgt_emb = self.tgt_embedding(labels) * math.sqrt(self.config["embed_dim"])
            tgt_emb = self.pos_encoder(tgt_emb)
            dec_output = tgt_emb
            for layer in self.decoder_stack:
                dec_output = layer(dec_output, memory, tgt_mask, None)
            logits = self.generator(dec_output)
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return {"logits": logits, "loss": loss}

        return {"encoder_out": memory}
