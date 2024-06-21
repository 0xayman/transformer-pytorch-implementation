import torch
import torch.nn as nn
from torch.nn import functional as F
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Hyperparameters
d_model = 32
h = 4
d_k = d_model // h
dropout = 0.1
n_layers = 6

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(d_k, d_k, bias=False)
        self.W_k = nn.Linear(d_k, d_k, bias=False)
        self.W_v = nn.Linear(d_k, d_k, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # perform scaled dot-production attention
        Q = self.W_q(q) # q @ Wq -> (seq_len, d_k) @ (d_k, d_k) => query (seq_len, d_k)
        K = self.W_k(k)   # k @ Wq -> (seq_len, d_k) @ (d_k, d_k) => key (seq_len, d_k)
        V = self.W_v(v) # v @ Wq -> (seq_len, d_k) @ (d_k, d_k) => value (seq_len, d_k)

        # The attention score will be calculated using the below formula
        # Attention(Q, K, V) = softmax((Q @ k.T) / sqrt(d_k)) @ V
        # Dimentions:
        # Q (seq_len, d_k) @ K.T (d_k, seq_len) -> (seq_len, seq_len) : will call this QK for reference
        # Apply the mask to QK to obtain masked_QK
        # attention(Q, K, V) = (masked_QK / sqrt(d_k)) @ V
        # The mask have dimentions (seq_len, seq_len)

        # Calculate (Q @ K.T) / sqrt(d_k)
        # print(Q.shape, K.shape)
        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        # print(attention_scores.shape, mask.shape)
        # Apply the mask
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float("-inf"))

        # Apply Softmax
        attention_scores = F.softmax(attention_scores, dim=-1)
        # calculate attention scores
        attention_scores = attention_scores @ V

        # apply dropout
        attention_scores = self.dropout(attention_scores)
        return attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead() for _ in range(h)]) # Create (h) heads, h = number of heads
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # Split q, k and v to (h) chunks to be sent in parallel to the (h) heads
        q_chunks = torch.chunk(q, h, dim=-1)
        k_chunks = torch.chunk(k, h, dim=-1)
        v_chunks = torch.chunk(v, h, dim=-1)
        attention_scores = torch.cat(
            [
                h(
                    q_chunks[idx],
                    k_chunks[idx],
                    v_chunks[idx],
                    mask)
                for idx, h in enumerate(self.attention_heads)],
            dim=-1
        ) # attention_scores has dimentions (seq_len, d_model)

        attention_scores = self.W_o(attention_scores) # attention_scores (seq_len, d_model) @ Wo (d_model, d_model) -> (seq_len, d_model)
        attention_scores = self.dropout(attention_scores) # Apply dropout
        return attention_scores
    


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

    

class EncoderBlock(nn.Module):
    """
    Implements a single encoder layer.
    Note that the Encoder consists of N identical layers.
    """
    def __init__(self):
        super().__init__()
        self.multi_head_self_attention = MultiHeadAttention()
        self.ffwd = FeedForwardNet()
        self.lnorm1 = nn.LayerNorm(d_model)
        self.lnorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.lnorm1(self.multi_head_self_attention(x, x, x, mask))
        x + x + self.lnorm2(self.ffwd(x))
        return x
     


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layers = nn.Sequential(*[EncoderBlock() for _ in range(n_layers)])
        self.lnorm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)

        x = self.lnorm(x)
        return x
    
    

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_self_attention = MultiHeadAttention()
        self.lnorm1 = nn.LayerNorm(d_model)
        self.multi_head_cross_attention = MultiHeadAttention()
        self.lnorm2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForwardNet()
        self.lnorm3 = nn.LayerNorm(d_model)

    def forward(self, o, encoder_output, src_mask, tgt_mask):
        self_attention_output = self.multi_head_self_attention(o, o, o, tgt_mask)
        o = o + self.lnorm1(self_attention_output)
        cross_attention_output = self.multi_head_cross_attention(o, encoder_output, encoder_output, src_mask) # query, key ,value
        o = o + self.lnorm2(cross_attention_output)
        ffwd_out = self.ffwd(o)
        o = o + self.lnorm3(ffwd_out)

        return o
    
    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_layers = nn.Sequential(*[DecoderBlock() for _ in range(n_layers)])
        self.lnorm = nn.LayerNorm(d_model)

    def forward(self, o, encoder_ouput, src_mask=None, tgt_mask=None):
        for decoder_layer in self.decoder_layers:
            o = decoder_layer(o, encoder_ouput, src_mask, tgt_mask)

        o = self.lnorm(o)
        return o
    
    

class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model) # (vocab_size, d_model)
        self.pos_embed = nn.Embedding(vocab_size, d_model) # (seq_len, d_model)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection_layer = nn.Linear(d_model, vocab_size)


    def encode(self, src, src_mask):
        # (batch_size, seq_len, d_model)
        src = self.token_embed(src) + self.pos_embed(src)
        encoder_output = self.encoder(src)
        return encoder_output

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # (batch_size, seq_len, d_model)
        tgt = self.token_embed(tgt) + self.pos_embed(tgt)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output

    def project(self, x):
        # (batch_size, seq_len, vocab_size)
        return self.projection_layer(x)

    

    