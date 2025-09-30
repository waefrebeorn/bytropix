import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 1. The Core Component: WuBu Sparse Attention - CORRECTED
# =============================================================================

class WuBuSparseAttention(nn.Module):
    """
    Implements the WuBu Memory concept.
    - Divides memory into a dense 'Working Memory' cache and a sparse 'Associative Memory' store.
    - Uses a lightweight 'RAS Indexer' to find relevant keys in the Associative Memory.
    - Combines outputs from both memory types for a context-rich representation.
    """
    def __init__(self, d_model: int, num_heads: int, k: int, working_memory_size: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.k = k  # Number of top keys to select from associative memory
        self.working_memory_size = working_memory_size

        # Standard linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # --- RAS Indexer ---
        indexer_dim = 64
        self.w_q_indexer = nn.Linear(d_model, indexer_dim)
        self.w_k_indexer = nn.Linear(d_model, indexer_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        if seq_len <= self.working_memory_size:
            k_work, v_work = k, v
            k_assoc, v_assoc = None, None
        else:
            k_work, v_work = k[:, :, -self.working_memory_size:, :], v[:, :, -self.working_memory_size:, :]
            k_assoc, v_assoc = k[:, :, :-self.working_memory_size, :], v[:, :, :-self.working_memory_size:, :]
        
        attn_scores_work = torch.matmul(q, k_work.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if k_assoc is not None:
            assoc_len = k_assoc.shape[2]
            
            q_idx = self.w_q_indexer(x) # (B, S, D_idx)
            k_idx = self.w_k_indexer(x[:, :assoc_len, :]) # (B, S_assoc, D_idx)
            
            indexer_scores = F.relu(torch.matmul(q_idx, k_idx.transpose(-1, -2))) # (B, S_query, S_assoc)
            
            _, top_k_indices = torch.topk(indexer_scores, min(self.k, assoc_len), dim=-1) # (B, S, k)

            # --- FIX STARTS HERE ---
            # We need to prepare k_assoc and v_assoc to be gathered from.
            # And prepare top_k_indices to do the gathering.
            
            # Expand indices for heads and d_head dimensions
            indices_for_gather = top_k_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, -1, self.d_head)
            
            # Expand k_assoc to have a dimension for queries
            k_assoc_expanded = k_assoc.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            
            # Gather keys. We gather along dim 3 (the assoc_len dimension).
            k_assoc_gathered = torch.gather(k_assoc_expanded, 3, indices_for_gather) # (B, H, S, k, d_head)
            
            # Repeat for values
            v_assoc_expanded = v_assoc.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            v_assoc_gathered = torch.gather(v_assoc_expanded, 3, indices_for_gather) # (B, H, S, k, d_head)
            
            # Now perform batched matmul between queries and gathered keys
            q_expanded = q.unsqueeze(3) # (B, H, S, 1, d_head)
            attn_scores_assoc = torch.matmul(q_expanded, k_assoc_gathered.transpose(-2, -1)).squeeze(3) / math.sqrt(self.d_head) # (B, H, S, k)

            all_attn_scores = torch.cat([attn_scores_assoc, attn_scores_work], dim=-1)
            
            # We need to combine values correctly too
            # v_work has shape (B, H, W, d_head). Expand for every query.
            v_work_expanded = v_work.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            all_v = torch.cat([v_assoc_gathered, v_work_expanded], dim=3) # (B, H, S, k+W, d_head)
            # --- FIX ENDS HERE ---
            
        else:
            all_attn_scores = attn_scores_work
            all_v = v_work
        
        if mask is not None:
            mask = mask[:, :, :seq_len, :all_attn_scores.shape[-1]]
            all_attn_scores = all_attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(all_attn_scores, dim=-1)

        # Final matmul needs to handle the expanded dimensions
        if len(all_v.shape) == 5: # Sparse path was taken
             output = torch.matmul(attn_weights.unsqueeze(3), all_v).squeeze(3)
        else: # Dense-only path
             output = torch.matmul(attn_weights, all_v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(output)


# =============================================================================
# 2. Standard Transformer Building Blocks
# =============================================================================

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, k: int, working_memory_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = WuBuSparseAttention(d_model, num_heads, k, working_memory_size)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Attention sub-layer with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward sub-layer with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# =============================================================================
# 3. The Full Model: WuBu Memory WBA (Decoder-Only Transformer)
# =============================================================================

class WuBuMemoryWBA_Model(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, 
                 k: int, working_memory_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model 
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, k, working_memory_size, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src_tokens: torch.Tensor):
        seq_len = src_tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(src_tokens.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        x = self.token_embedding(src_tokens) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
        logits = self.output_layer(x)
        return logits

# =============================================================================
# 4. Example Usage
# =============================================================================


if __name__ == "__main__":
    # --- Model Hyperparameters ---
    VOCAB_SIZE = 10000        # Size of the vocabulary
    MAX_SEQ_LEN = 4096        # Max context length the model can handle
    D_MODEL = 512             # Embedding dimension
    NUM_LAYERS = 6            # Number of Transformer blocks
    NUM_HEADS = 8             # Number of attention heads
    D_FF = 2048               # Hidden dimension of the feed-forward network
    DROPOUT = 0.1
    
    # --- WuBu Specific Hyperparameters ---
    K_SPARSE = 64             # Retrieve top 64 relevant items from long-term memory
    WORKING_MEMORY_SIZE = 256 # Last 256 tokens are treated as high-resolution working memory
    
    print("Initializing WuBu Memory WBA Model...")
    model = WuBuMemoryWBA_Model(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        k=K_SPARSE,
        working_memory_size=WORKING_MEMORY_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    )
    
    # --- Test with Dummy Data ---
    print("\n--- Running a test forward pass ---")
    batch_size = 2
    sequence_length = 1024 
    dummy_input = torch.randint(0, VOCAB_SIZE, (batch_size, sequence_length))
    
    print(f"Input tensor shape: {dummy_input.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    output_logits = model(dummy_input)
    
    print(f"Output logits shape: {output_logits.shape}")
    print("Test successful!")

    # --- DYNAMIC REPORTING BLOCK ---
    # This now uses the actual variables and tensor shapes from the run.
    
    # Check if the associative path was even used
    if sequence_length > WORKING_MEMORY_SIZE:
        assoc_memory_size = sequence_length - WORKING_MEMORY_SIZE
        
        print(f"\nThe output shape {output_logits.shape} corresponds to (batch_size, sequence_length, vocab_size).")
        
        print(f"\n--- How it worked for this sequence (length {sequence_length}) ---")
        print("1. The sequence was split into two parts for attention calculation:")
        print(f"   - Working Memory (Dense): The last {WORKING_MEMORY_SIZE} tokens.")
        print(f"   - Associative Memory (Sparse): The first {assoc_memory_size} tokens.")
        print(f"2. For each of the {sequence_length} tokens (queries):")
        print(f"   - It performed DENSE attention on all {WORKING_MEMORY_SIZE} tokens in Working Memory.")
        print(f"   - The RAS Indexer quickly scanned the {assoc_memory_size} Associative Memory tokens.")
        print(f"   - It performed SPARSE attention on only the top {K_SPARSE} most relevant tokens found by the indexer.")
        print("3. The results were combined, allowing the model to efficiently use both recent and important long-range context.")
    else:
        # Handle the case where the sequence is too short to use sparse attention
        print(f"\nSequence length ({sequence_length}) is less than or equal to Working Memory size ({WORKING_MEMORY_SIZE}).")
        print("Only DENSE attention was used across the entire sequence.")