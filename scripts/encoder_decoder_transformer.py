import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import config
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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, max_seq_len: int = 2048):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(embed_dim, embed_dim) for _ in range(4))
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):
        bs, seq_len = query.shape[0], query.shape[1]
        if seq_len > self.max_seq_len:
            query = query[:, :self.max_seq_len]
            key = key[:, :self.max_seq_len]
            value = value[:, :self.max_seq_len]
            seq_len = self.max_seq_len
            if mask is not None:
                mask = mask[:, :, :self.max_seq_len, :self.max_seq_len]

        query, key, value = (proj(x).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2) for proj, x in
                             [(self.q_proj, query), (self.k_proj, key), (self.v_proj, value)])

        if seq_len > 1024:
            return self._chunked_attention(query, key, value, mask)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, value).transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)
        return self.out_proj(ctx)


    def _chunked_attention(self, query, key, value, mask, chunk_size=512):
        bs, num_heads, seq_len, head_dim = query.shape
        output = torch.zeros_like(query)

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = query[:, :, i:end_i]

            scores = torch.matmul(q_chunk, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                if mask.shape[-1] != seq_len:
                    mask = mask[:, :, :seq_len, :seq_len]
                mask_chunk = mask[:, :, i:end_i, :]
                scores = scores.masked_fill(mask_chunk == 0, -1e9)

            attn = self.dropout(torch.softmax(scores, dim=-1))
            output[:, :, i:end_i] = torch.matmul(attn, value)

        return output.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)



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

        # Handle mask dimension for self-attention
        if tgt_mask is not None and tgt_mask.dim() == 4:
            # If mask has batch and head dimensions, use it as is
            self_attn_mask = tgt_mask
        elif tgt_mask is not None:
            # Expand mask to match attention head dimensions
            batch_size = tgt.shape[0]
            num_heads = self.self_attn.num_heads
            self_attn_mask = tgt_mask.unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        else:
            self_attn_mask = None

        tgt = tgt + self.dropout(self.self_attn(norm_tgt, norm_tgt, norm_tgt, self_attn_mask))
        norm_tgt = self.norm2(tgt)
        tgt = tgt + self.dropout(self.cross_attn(norm_tgt, mem, mem, mem_mask))
        norm_tgt = self.norm3(tgt)
        tgt = tgt + self.dropout(self.feed_forward(norm_tgt))
        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout, max_len=2048):  # Reduced from 12000
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        pe = torch.zeros(1, max_len, embed_dim)
        pos, div = torch.arange(max_len).unsqueeze(1), torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = min(x.size(1), self.max_len)
        return self.dropout(x[:, :seq_len] + self.pe[:, :seq_len])



class SpeechToTextTranslationModel(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, embed_dim: int,
                 num_attn_heads: int, tgt_vocab_size: int, d_ff: int, dropout: float,
                 input_feat_dim: int = 80):
        super().__init__()

        self.config = {
            "num_attn_heads": num_attn_heads,
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

        self.pad_token_id = 0
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, label_smoothing=config.LABEL_SMOOTHING)

    def _create_causal_mask(self, seq_len, device):
        """Create causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # True for allowed positions

    def _truncate_sequences(self, input_features, labels=None, max_audio_len=1024, max_text_len=256):
        """Truncate sequences to manageable lengths."""
        # Truncate audio features
        if input_features.shape[1] > max_audio_len:
            input_features = input_features[:, :max_audio_len]

        # Truncate labels if provided
        if labels is not None and labels.shape[1] > max_text_len:
            labels = labels[:, :max_text_len]

        return input_features, labels

    def compute_masked_loss(self, logits, labels):
        """Compute loss with proper masking for padding tokens."""
        mask = (labels != self.pad_token_id).float()
        logits_flat = logits.contiguous().view(-1, logits.shape[-1])
        labels_flat = labels.contiguous().view(-1)
        mask_flat = mask.contiguous().view(-1)

        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction='none',
            label_smoothing=0.1
        )
        masked_loss = loss * mask_flat
        return masked_loss.sum() / (mask_flat.sum() + 1e-8)

    def _make_pad_lookahead_target_mask(self, target, device):
        target_len = target.shape[1]
        target_pad_mask = (target != 1).unsqueeze(1).unsqueeze(2)
        target_lookahead_mask = torch.triu(torch.ones((target_len, target_len), device=device), diagonal=1).bool()
        return target_pad_mask & ~target_lookahead_mask

    def forward(self, input_features, labels=None, **kwargs):
        input_features, labels = self._truncate_sequences(input_features, labels)

        src = self.feature_projection(input_features) * math.sqrt(self.config["embed_dim"])
        src = self.pos_encoder(src)

        memory = src
        for layer in self.encoder_stack:
            memory = layer(memory, None)

        if labels is not None:
            seq_len = labels.shape[1]
            tgt_mask = self._create_causal_mask(seq_len, labels.device)

            pad_mask = self._make_pad_lookahead_target_mask(labels, labels.device)
            combined_mask = tgt_mask.unsqueeze(0).unsqueeze(0) & pad_mask
            tgt_emb = self.tgt_embedding(labels) * math.sqrt(self.config["embed_dim"])
            tgt_emb = self.pos_encoder(tgt_emb)

            dec_output = tgt_emb
            for layer in self.decoder_stack:
                dec_output = layer(dec_output, memory, combined_mask, None)

            logits = self.generator(dec_output)

            loss = self.compute_masked_loss(logits, labels)
            return {"logits": logits, "loss": loss}

        return {"encoder_out": memory}


