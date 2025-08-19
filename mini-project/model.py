"""
   ┌─     :::          :::   :::  ────────────────────────-───┐
   │     :+:         :+:+: :+:+:                              │
   │    +:+        +:+ +:+:+ +:+                              │
   │   +#+        +#+  +:+  +#+    Encoder implementation     │
   │  +#+        +#+       +#+    Louca Malerba (@malerbe)    │
   │ #+#        #+#       #+#    19/08/2025                   │
   │########## ###       ###    Educational Deep Dive         │
   └──────────────────────────────────────────────────────────┘
"""

#######################################
# Public libraries importations
import torch
import torch.nn as nn

# Private libraries importations


#######################################
# Following code is an exact copy from the implementation made in the explanations.ipynb notebook

class SelfAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads == embed_dim), "embed_dim must be divisible by num_heads"

        self.V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Q = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self,
                query,
                keys,
                values,
                mask=None):
        
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 1. Extract the embeddings from the input:
        Q = self.Q(query) # [N, query_len, embed_dim]
        K = self.K(keys) # [N, key_len, embed_dim]
        V = self.V(values) # [N, value_len, embed_dim]

        # 2. Split embeddings into multiple heads
        Queries = Q.reshape(N, query_len, self.num_heads, self.head_dim) # [N, query_len, num_heads, head_dim]
        Keys = K.reshape(N, key_len, self.num_heads, self.head_dim) # [N, key_len, num_heads, head_dim]
        Values = V.reshape(N, value_len, self.num_heads, self.head_dim) # [N, value_len, num_heads, head_dim]

        # 3. Compute the attention scores
        # matmul
        energy = torch.einsum("nqhd,nkhd->nhqk", [Queries, Keys])

        # scale
        energy = energy / (self.embed_dim ** (1/2))
        
        # apply mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # apply softmax to get attention weights
        attention = torch.softmax(energy, dim=3)

        # final matmul between attention weights with values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, Values]).reshape(N, query_len, self.num_heads * self.head_dim) # [N, query_len, num_heads, head_dim]

        out = self.fc_out(out)

        
        return out 
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            num_heads,
            device,
            forward_expansion,
            dropout,
            max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                ) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out