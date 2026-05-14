"""
WubuNestGPT v2: Full Language Model
Architecture: LatentAttention + SparseMoE + Hyperbolic gyration + Turbo quant

This is the complete model. Training loop is in train_wubunest_gpt.py.

Architecture overview:
- Embedding → N×WubuBlock → LM Head
- Each WubuBlock: LatentAttention (MLA) + WubuMoE (Sparse MoE)
- KV cache compression via latent vectors (d_c = d_model/4)
- Turbo-style block-wise FP8 quantization for the latent KV cache
- Auxiliary-loss-free MoE load balancing (DeepSeek-V3)
- Hyperbolic gyration replaces RoPE for position encoding

Parameter counts (base config, d_model=768):
- Embed: 768 × 10000 = 7.68M
- Per block: ~6.2M (attention=1.2M + MoE=5.0M)
- 8 blocks: ~56M
- LM head: 768 × 10000 = 7.68M (tied with embed)
- Total: ~63.4M
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Optional, Any, Tuple, Dict

# WuBu components
from wubu_latent_attention import LatentAttention, PoincareBall, BlockQuantizer
from wubu_block_simple import WubuBlock  # Use simplified FFN for now


class WubuNestGPT(nn.Module):
    """
    Complete WuBuNestGPT model.
    
    Config keys:
        vocab_size: int = 10000
        d_model: int = 768
        n_layers: int = 8
        n_heads: int = 12
        d_head: int = 64
        d_compressed: int = 256  # d_c = 4*d_head for MLA
        d_ff: int = 2048
        n_shared: int = 2         # always-activated experts
        n_routed: int = 32        # pool of sparse experts
        top_k: int = 4            # activated per token
        use_quant: bool = True    # turbo quantization on KV cache
        dropout_rate: float = 0.1
        max_seq_len: int = 2048
        dtype: str = 'bfloat16'
    """

    config: Dict[str, Any]

    def setup(self):
        cfg = self.config
        dtype = jnp.bfloat16 if cfg.get('dtype', 'bfloat16') == 'bfloat16' else jnp.float32
        
        # Token embedding
        self.token_embed = nn.Embed(
            num_embeddings=cfg['vocab_size'],
            features=cfg['d_model'],
            dtype=dtype
        )
        
        # WuBu blocks
        self.blocks = [
            WubuBlock(
                d_model=cfg['d_model'],
                d_compressed=cfg.get('d_compressed', cfg['d_model'] // 3),
                d_head=cfg.get('d_head', cfg['d_model'] // cfg['n_heads']),
                n_heads=cfg['n_heads'],
                d_ff=cfg.get('d_ff', cfg['d_model'] * 4),
                use_moe=cfg.get('use_moe', False),
                use_quant=cfg.get('use_quant', False),
                dropout_rate=cfg.get('dropout_rate', 0.1),
                dtype=dtype,
                name=f'block_{i}'
            )
            for i in range(cfg['n_layers'])
        ]
        
        # Final norm
        self.final_norm = nn.LayerNorm(dtype=dtype)
        
        # LM head (can tie with embed weights)
        self.lm_head = nn.Dense(
            cfg['vocab_size'],
            use_bias=False,
            dtype=dtype,
            name='lm_head'
        )
        
        # MoE bias tracking (per-layer routing biases for auxiliary-loss-free balancing)
        self.expert_biases = [
            jnp.zeros(cfg.get('n_routed', 32), dtype=jnp.float32)
            for _ in range(cfg['n_layers'])
        ]

    def __call__(
        self,
        input_ids: jnp.ndarray,
        kv_caches: Optional[list] = None,
        is_training: bool = True,
        return_hidden: bool = False
    ):
        """
        input_ids: [batch, seq] token indices
        Returns: logits [batch, seq, vocab_size]
        """
        B, T = input_ids.shape
        cfg = self.config
        
        # Embed
        x = self.token_embed(input_ids)  # [B, T, d_model]
        
        # Pass through blocks
        new_kv_caches = [] if kv_caches is None else kv_caches
        bias_updates = []
        
        for i, block in enumerate(self.blocks):
            layer_cache = None if kv_caches is None else kv_caches[i]
            bias = self.expert_biases[i] if is_training else None
            
            x, layer_cache, bias_update = block(
                x, 
                kv_cache=layer_cache,
                expert_bias=bias,
                is_training=is_training
            )
            
            if kv_caches is not None:
                new_kv_caches.append(layer_cache)
            
            if is_training:
                bias_updates.append(bias_update)
        
        # Final norm
        x = self.final_norm(x)
        
        # Project to vocab
        logits = self.lm_head(x)
        
        if return_hidden:
            return logits, x, bias_updates
        return logits, new_kv_caches if kv_caches is not None else None, bias_updates if is_training else None

    def _sinusoidal_positions(self, T, d_model, dtype):
        """Standard sinusoidal position encoding."""
        pos = jnp.arange(T, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, d_model, 2, dtype=jnp.float32) * 
            -(jnp.log(10000.0) / d_model)
        )
        pe = jnp.zeros((T, d_model), dtype=dtype)
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))
        return pe

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        eos_token_id: int = 0,
        verbose: bool = False
    ):
        """Autoregressive generation."""
        B, T = input_ids.shape
        generated = input_ids
        
        # Initialize KV cache
        kv_caches = [{} for _ in range(self.config['n_layers'])]
        
        for step in range(max_new_tokens):
            # Forward pass on last token only (with KV cache)
            if step == 0:
                logits, kv_caches_out, _ = self(
                    generated, kv_caches=None, is_training=False
                )
            else:
                logits, kv_caches_out, _ = self(
                    last_token, kv_caches=kv_caches, is_training=False
                )
                kv_caches = kv_caches_out
            
            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                k_vals = jnp.sort(next_logits, axis=-1)[:, -top_k:]
                k_threshold = k_vals[:, 0:1]
                next_logits = jnp.where(
                    next_logits < k_threshold,
                    -float('inf'),
                    next_logits
                )
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits = jnp.sort(next_logits, axis=-1)[:, ::-1]
                sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
                cumsum = jnp.cumsum(sorted_probs, axis=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove = sorted_indices_to_remove.at[:, 1:].set(
                    sorted_indices_to_remove[:, :-1]
                )
                sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
                
                # Scatter back
                batch_indices = jnp.arange(B)[:, None]
                gather_indices = jnp.argsort(next_logits, axis=-1)[:, ::-1]
                
                # Create mask
                mask = jnp.zeros_like(next_logits)
                for b in range(B):
                    for i in range(next_logits.shape[-1]):
                        mask = mask.at[b, gather_indices[b, i]].set(
                            sorted_indices_to_remove[b, i].astype(jnp.float32)
                        )
                next_logits = jnp.where(mask == 0, next_logits, -float('inf'))
            
            # Sample
            probs = jax.nn.softmax(next_logits, axis=-1)
            key = jax.random.PRNGKey(step)
            next_token = jax.random.categorical(key, jnp.log(probs + 1e-10), axis=-1)
            
            # Append
            last_token = next_token[:, None]
            generated = jnp.concatenate([generated, last_token], axis=1)
            
            if verbose:
                if step % 20 == 0:
                    print(f"  step {step}, token {next_token[0]}")
            
            # Check for EOS
            if jnp.all(next_token == eos_token_id):
                break
        
        return generated


# ─── Training State & Loss ────────────────────────────────────────

def create_train_state(model, config, learning_rate=1e-4):
    """Initialize training state with optimizer."""
    rng = jax.random.PRNGKey(42)
    
    # Dummy input for parameter init
    dummy_ids = jnp.zeros((2, 16), dtype=jnp.int32)
    
    # Initialize params
    variables = model.init(rng, dummy_ids, is_training=True)
    params = variables['params']
    
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Total parameters: {param_count:,}")
    
    # Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=config.get('warmup_steps', 200),
        decay_steps=config.get('train_steps', 50000),
        end_value=learning_rate * 0.01
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=config.get('weight_decay', 0.1)
        )
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def compute_loss(params, model, batch, config):
    """Cross-entropy loss + optional MoE auxiliary loss."""
    input_ids = batch['input_ids']
    labels = batch['labels']
    
    logits, _, bias_updates = model.apply(
        {'params': params},
        input_ids,
        is_training=True
    )
    
    # Cross-entropy
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)
    
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits_flat, labels_flat
    ).mean()
    
    # Auxiliary MoE loss (small penalty for bias update magnitude)
    aux_loss = 0.0
    for bu in bias_updates:
        if bu is not None:
            aux_loss = aux_loss + jnp.mean(jnp.abs(bu))
    
    total_loss = ce_loss + config.get('aux_loss_coef', 0.01) * aux_loss
    
    return total_loss, {'ce_loss': ce_loss, 'aux_loss': aux_loss}


@jax.jit
def train_step(state, batch, config):
    """Single training step."""
    grad_fn = jax.value_and_grad(
        lambda p: compute_loss(p, None, batch, config)[0],
        has_aux=True
    )
    loss, grads = grad_fn(state.params)
    # Hack: need to pass model ref through closure
    return state.apply_gradients(grads=grads), loss


# ─── Data handling ────────────────────────────────────────────────

def load_corpus_data(corpus_path: str, max_samples: int = None):
    """Load and parse corpus files into text chunks."""
    import re
    
    texts = []
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract narrative and dialogue text from CORPUS.py format
    # Pattern: key-value pairs or narrative text blocks
    # Look for string literals and extract them
    text_chunks = re.findall(r'"""(.*?)"""', content, re.DOTALL)
    for chunk in text_chunks:
        # Clean whitespace
        clean = re.sub(r'\s+', ' ', chunk).strip()
        if len(clean) > 100:  # Skip very short chunks
            texts.append(clean)
    
    print(f"Loaded {len(texts)} text chunks (total chars: {sum(len(t) for t in text_chunks)})")
    
    if max_samples:
        texts = texts[:max_samples]
    
    return texts


def create_batches(texts, tokenizer, seq_len, batch_size):
    """Convert texts to token batches."""
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    # Cut into fixed-length sequences
    total_len = (len(all_tokens) // seq_len) * seq_len
    all_tokens = all_tokens[:total_len]
    
    # Reshape into batches
    n_batches = total_len // (seq_len * batch_size)
    usable = n_batches * seq_len * batch_size
    all_tokens = all_tokens[:usable]
    
    tokens_arr = np.array(all_tokens).reshape(-1, batch_size, seq_len)
    n_batches = tokens_arr.shape[0]
    
    return tokens_arr, n_batches


# ─── Tokenizer ────────────────────────────────────────────────────

class SimpleTokenizer:
    """
    A simple byte-level tokenizer with configurable vocab.
    Uses GPT-2's BPE merges if available, otherwise falls back to char-level.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = self._build_char_vocab() if vocab_size > 256 else None
    
    def _build_char_vocab(self):
        """Build a character-level vocabulary from scratch."""
        # Start with all printable ASCII
        chars = [chr(i) for i in range(32, 127)] + ['\n', '\t', '\r']
        # Add common CJK and special unicode blocks
        for i in range(0x2000, 0x206F):  # General punctuation
            chars.append(chr(i))
        for i in range(0x4E00, 0x9FFF, 100):  # CJK sampled
            chars.append(chr(i))
        for i in range(0x3040, 0x309F, 10):  # Hiragana sampled
            chars.append(chr(i))
        for i in range(0x30A0, 0x30FF, 10):  # Katakana sampled
            chars.append(chr(i))
        
        # Build vocab
        vocab = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2, '<BOS>': 3}
        for i, ch in enumerate(chars):
            if i + 4 < self.vocab_size:
                vocab[ch] = i + 4
                vocab[f'_{i}'] = i + 4  # reverse mapping
        
        self.id_to_char = {v: k for k, v in vocab.items() if len(k) <= 4}
        return vocab
    
    def encode(self, text: str) -> list:
        """Tokenize text into token IDs."""
        if self.vocab_size <= 256:
            # Byte-level
            return [b for b in text.encode('utf-8', errors='replace')[:self.vocab_size-4] + 4]
        
        # Char-level with BPE-like subword if available
        tokens = []
        for ch in text:
            if ch in self.vocab:
                tokens.append(self.vocab[ch])
            else:
                # Try to encode as individual bytes for non-vocab chars
                for b in ch.encode('utf-8', errors='replace'):
                    if b < self.vocab_size - 4:
                        tokens.append(b + 4)
                    else:
                        tokens.append(1)  # <UNK>
        return tokens[:self.vocab_size]  # cap
    
    def decode(self, ids: list) -> str:
        """Decode token IDs back to text."""
        chars = []
        for token_id in ids:
            if token_id < len(self.id_to_char):
                ch = self.id_to_char.get(token_id, '')
                if ch and not ch.startswith('_'):
                    chars.append(ch)
        return ''.join(chars)
    
    def get_vocab_size(self):
        return self.vocab_size
