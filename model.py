import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
import numpy as np

"""
Embedding the input sequence
"""
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
    
"""
Positional Encoding Vector
"""
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pos_enc = torch.zeros(max_seq_length, embedding_dim) # positional encoding

        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pos_enc[pos, i] = math.sin(pos/(10000**(2*i/embedding_dim)))
                pos_enc[pos, i+1] = math.cos(pos/(10000**((2*i+1)/embedding_dim)))
    
        pos_enc = pos_enc.unsqueeze(0) # increase a dimension ([] -> [[]])
        self.register_buffer('pos_enc', pos_enc) # not as a parameter, register as a buffer

    def forward(self, x):
        x = x*math.sqrt(self.embedding_dim) # embedding_dim = d_model
        seq_length = x.size(1)
        pos_enc = Variable(self.pos_enc[:, :seq_length], requires_grad=False).to(x.device)

        x = x + pos_enc # add the pos_enc to the input embedding vector
        x = self.dropout(x)
        return x

"""
Scaled Dot-Product Attention layer (Self-attention)
"""   
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        k_dim = k.size(-1) # last dimension of a tensor
        attn = torch.matmul(q / np.sqrt(k_dim), k.transpose())
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9) # mask out with -inf (-1e9)

        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output

"""
Multi-head attention layer
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attn = ScaledDotProductAttention(dropout)
        self.num_heads = num_heads 
        self.dim_head = embedding_dim // num_heads # d_k = d_v = (d_model / h)

        # Setup Linear Projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        #self.dropout = nn.Dropout(dropout)
        

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Apply Linear Projection
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Reshape the input dimensions
        q = q.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        k = k.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)
        v = v.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1,2)

        # Calculate attention & reshape output
        scores = self.self_attn(q, k, v, mask)
        output = scores.transpose(1,2).contiguous().view(batch_size, -1, self.embedding_dim)

        # Apply projection to output
        output = self.out_proj(output)
        return output

"""
Normalization Layer
"""
class Normalize(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalize, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)
    

"""
Transformer Encoder Layer
"""
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim = 2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        
        # Point-wise Feedforward Networks
        self.ff_net = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim), 
            nn.ReLU(), 
            nn.Linear(ff_dim, embedding_dim)
        )

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Normalize(embedding_dim)
        self.norm_2 = Normalize(embedding_dim)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.self_attn(x2, x2, x2, mask)) # 1st residual connection

        x2 = self.norm_2(x) 
        x = x + self.dropout_2(self.ff_net(x2)) # 2nd residual connection 


"""
Transformer Decoder Layer 
"""
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim = 2048, dropout=0.1):
        super(DecoderLayer, self).__init__() 
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(embedding_dim, num_heads, dropout) # Third Sub-layer

        # Point-wise Feedforward Networks
        self.ff_net = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim), 
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.norm_1 = Normalize(embedding_dim)
        self.norm_2 = Normalize(embedding_dim)
        self.norm_3 = Normalize(embedding_dim)

    def forward(self, x, memory, source_mask, target_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.self_attn(x2, x2, x2, target_mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.enc_attn(x2, memory, memory, source_mask))

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff_net(x2))

        return x
    

"""
Encoder Parts
"""
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, 
                 max_seq_len, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, 
                                                  2048, dropout) for _ in range(num_layers)])
        self.norm = Normalize(embedding_dim)
        self.pos_emb = PositionalEncoder(embedding_dim, max_seq_len, dropout)

    def forward(self, source, source_mask):
        # Embed the source & Add positional embeddings
        x = self.embedding(source)
        x = self.pos_emb(x)

        # Propagate through sublayers + Add residual connections
        for layer in self.layers:
            x = layer(x, source_mask) # 1st layer: MultiheadAttn, 2nd layer: FFN
        
        # Normalize the layer
        x = self.norm(x)
        return x
    
"""
Decoder Parts
"""
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, 
                 max_seq_len, num_heads, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
        
        self.norm = Normalize(embedding_dim)
        self.pos_emb = PositionalEncoder(embedding_dim, max_seq_len, dropout)

    def forward(self, target, memory, source_mask, target_mask):
        x = self.embedding(target)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        x = self.norm(x)
        return x
    

"""
Transformer Architecture
"""
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, 
                 source_max_seq_len, target_max_seq_len, 
                 embedding_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = Encoder(source_vocab_size, embedding_dim, source_max_seq_len, 
                               num_heads, num_layers, dropout)
        self.decoder = Decoder(target_vocab_size, embedding_dim, target_max_seq_len, 
                               num_heads, num_layers, dropout)
        self.final_linear = nn.Linear(embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, target, source_mask, target_mask):
        memory = self.encoder(source, source_mask)
        output = self.decoder(target, memory, source_mask, target_mask)

        output = self.dropout(output)
        output = self.final_linear(output)
        return output
    
    def make_source_mask(self, source_ids, source_pad_id):
        return (source_ids != source_pad_id).unsqueeze(-2)
    
    def make_target_mask(self, target_ids):
        _, len_target = target_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=target_ids.device), diagonal=1)).bool()
        return subsequent_mask
        
