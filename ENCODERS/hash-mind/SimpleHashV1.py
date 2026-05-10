
import numpy as np
import sys
import math
import time

# --- Part 1: The "Live Tokenizer" Engine (No changes here) ---
class SimplifiedASCIIConverter:
    def __init__(self):
        self.char_to_val, self.val_to_char = {}, {}
        self.char_to_idx, self.idx_to_char = {}, {}
        self.vocab_size = 0
        chars = ([str(i) for i in range(10)] +
                 [chr(ord('A') + i) for i in range(26)] +
                 [chr(ord('a') + i) for i in range(26)] +
                 [' ', '.', ',', '!', '?', '\n'])
        for char in sorted(list(set(chars))): self._add_char(char, ord(char))
    def _add_char(self, char, val):
        if char not in self.char_to_val:
            self.char_to_val[char], self.val_to_char[val] = val, char
            self.char_to_idx[char], self.idx_to_char[self.vocab_size] = self.vocab_size, char
            self.vocab_size += 1
    def convert(self, text): return [self.char_to_val.get(c, self.char_to_val[' ']) for c in text]
    def get_indices(self, text): return [self.char_to_idx.get(c, self.char_to_idx[' ']) for c in text]

class RollingHasher:
    def __init__(self, window_size, base=31, modulus=10**9 + 7):
        self.window_size, self.base, self.modulus = window_size, base, modulus
        self.precomputed_base = pow(self.base, self.window_size - 1, self.modulus)
    def hash_sequence(self, values):
        if len(values) < self.window_size: return []
        hashes, current_hash = [], 0
        for i in range(self.window_size): current_hash = (current_hash * self.base + values[i]) % self.modulus
        hashes.append(current_hash)
        for i in range(1, len(values) - self.window_size + 1):
            old_val, new_val = values[i-1], values[i+self.window_size-1]
            current_hash = ((current_hash - old_val * self.precomputed_base) * self.base + new_val) % self.modulus
            hashes.append(current_hash)
        return hashes

# --- Part 2: The Neural Network (V11 - Dual-Source Embedding) ---

class PositionalEncoding:
    def __init__(self, d_model, max_len=500):
        position = np.arange(max_len)[:, np.newaxis]; div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model)); pe[:, 0::2] = np.sin(position * div_term); pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[np.newaxis, ...]
    def forward(self, x): return x + self.pe[:, :x.shape[1], :]
    def backward(self, d_out): return d_out

class HashMindBlock:
    def __init__(self, d_model, n_heads):
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.params = {
            'qkv_proj': np.random.randn(d_model, d_model * 3) * np.sqrt(2./d_model), 'out_proj': np.random.randn(d_model, d_model) * np.sqrt(2./d_model),
            'ffn1': np.random.randn(d_model, d_model*4) * np.sqrt(2./d_model), 'ffn2': np.random.randn(d_model*4, d_model) * np.sqrt(2./(d_model*4)),
            'norm1_gamma': np.ones(d_model), 'norm1_beta': np.zeros(d_model), 'norm2_gamma': np.ones(d_model), 'norm2_beta': np.zeros(d_model)
        }
    def _layer_norm_forward(self, x, gamma, beta, eps=1e-5):
        mean=x.mean(axis=-1, keepdims=True); var=x.var(axis=-1, keepdims=True); x_hat=(x-mean)/np.sqrt(var+eps); out=gamma*x_hat+beta
        return out, (x, x_hat, mean, var, gamma, eps)
    def _layer_norm_backward(self, d_out, cache):
        x,x_hat,mean,var,gamma,eps=cache; B,T,D=x.shape; d_beta=d_out.sum(axis=(0,1)); d_gamma=(d_out*x_hat).sum(axis=(0,1)); dx_hat=d_out*gamma
        dvar=np.sum(dx_hat*(x-mean)*-0.5*(var+eps)**-1.5, axis=-1, keepdims=True); dmean=np.sum(dx_hat*-1/np.sqrt(var+eps), axis=-1, keepdims=True) + dvar*np.mean(-2.*(x-mean),axis=-1,keepdims=True)
        dx=(dx_hat/np.sqrt(var+eps))+(dvar*2.*(x-mean)/D)+(dmean/D); return dx, d_gamma, d_beta
    def _attention_forward(self, x_norm):
        B,T,C=x_norm.shape; qkv=x_norm@self.params['qkv_proj']; q,k,v=np.split(qkv,3,axis=-1); q,k,v=[y.reshape(B,T,self.n_heads,self.d_head).transpose(0,2,1,3) for y in (q,k,v)]
        attn_scores=(q@k.transpose(0,1,3,2))/math.sqrt(self.d_head); exp_scores=np.exp(attn_scores-np.max(attn_scores,axis=-1,keepdims=True)); attn_probs=exp_scores/exp_scores.sum(axis=-1,keepdims=True)
        attn_output=(attn_probs@v).transpose(0,2,1,3).reshape(B,T,C); out=attn_output@self.params['out_proj']; return out,(x_norm,q,k,v,attn_probs,attn_output)
    def _attention_backward(self,d_out,cache):
        x_norm,q,k,v,attn_probs,attn_output=cache; B,T,C=x_norm.shape; d_out_proj=attn_output.reshape(B*T,C).T@d_out.reshape(B*T,C); d_attn_output=d_out@self.params['out_proj'].T
        d_attn_output=d_attn_output.reshape(B,T,self.n_heads,self.d_head).transpose(0,2,1,3); d_attn_probs=d_attn_output@v.transpose(0,1,3,2); dv=attn_probs.transpose(0,1,3,2)@d_attn_output
        d_attn_scores=attn_probs*(d_attn_probs-np.sum(d_attn_probs*attn_probs,axis=-1,keepdims=True)); d_attn_scores/=math.sqrt(self.d_head); dq=d_attn_scores@k; dk=d_attn_scores.transpose(0,1,3,2)@q; dq,dk,dv=[y.transpose(0,2,1,3).reshape(B,T,C) for y in (dq,dk,dv)]
        d_qkv=np.concatenate([dq,dk,dv],axis=-1); d_qkv_proj=x_norm.reshape(B*T,C).T@d_qkv.reshape(B*T,3*C); dx_norm=d_qkv@self.params['qkv_proj'].T; return dx_norm, {'qkv_proj':d_qkv_proj, 'out_proj':d_out_proj}
    def forward(self,x):
        norm1_out,norm1_cache=self._layer_norm_forward(x,self.params['norm1_gamma'],self.params['norm1_beta']); attn_out,attn_cache=self._attention_forward(norm1_out); x_res1=x+attn_out
        norm2_out,norm2_cache=self._layer_norm_forward(x_res1,self.params['norm2_gamma'],self.params['norm2_beta']); ffn_intermediate=np.maximum(0,norm2_out@self.params['ffn1']); ffn_out=ffn_intermediate@self.params['ffn2']
        x_final=x_res1+ffn_out; return x_final, (x,x_res1,norm1_cache,attn_cache,norm2_out,norm2_cache,ffn_intermediate)
    def backward(self,d_out,cache):
        x_in,x_res1,norm1_cache,attn_cache,norm2_out,norm2_cache,ffn_intermediate=cache; grads={}; B,T,C=x_in.shape; d_ffn_out,d_x_res1_from_res=d_out,d_out; d_ffn2=ffn_intermediate.reshape(B*T,self.d_model*4).T@d_ffn_out.reshape(B*T,C)
        d_ffn_intermediate=d_ffn_out@self.params['ffn2'].T; d_ffn_intermediate[ffn_intermediate<=0]=0; d_ffn1=norm2_out.reshape(B*T,C).T@d_ffn_intermediate.reshape(B*T,self.d_model*4); d_norm2_out=d_ffn_intermediate@self.params['ffn1'].T
        grads.update({'ffn1':d_ffn1,'ffn2':d_ffn2}); d_x_res1_from_ffn,d_gamma2,d_beta2=self._layer_norm_backward(d_norm2_out,norm2_cache); grads.update({'norm2_gamma':d_gamma2,'norm2_beta':d_beta2})
        d_x_res1=d_x_res1_from_ffn+d_x_res1_from_res; d_attn_out,d_x_in_from_res=d_x_res1,d_x_res1; dx_norm,attn_grads=self._attention_backward(d_attn_out,attn_cache); grads.update(attn_grads)
        d_x_in_from_norm,d_gamma1,d_beta1=self._layer_norm_backward(dx_norm,norm1_cache); grads.update({'norm1_gamma':d_gamma1,'norm1_beta':d_beta1}); d_x_in=d_x_in_from_norm+d_x_in_from_res; return d_x_in,grads

class AdamOptimizer:
    def __init__(self, params_list, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params_list,self.lr,self.beta1,self.beta2,self.epsilon=params_list,lr,beta1,beta2,epsilon; self.m=[{k:np.zeros_like(p) for k,p in params.items()} for params in params_list]; self.v=[{k:np.zeros_like(p) for k,p in params.items()} for params in params_list]; self.t=0
    def step(self,grads_list):
        self.t+=1
        for i,(params,grads) in enumerate(zip(self.params_list,grads_list)):
            for k in params.keys():
                if k not in grads or grads[k] is None: continue
                np.clip(grads[k],-1.0,1.0,out=grads[k]); self.m[i][k]=self.beta1*self.m[i][k]+(1-self.beta1)*grads[k]; self.v[i][k]=self.beta2*self.v[i][k]+(1-self.beta2)*(grads[k]**2)
                m_hat,v_hat=self.m[i][k]/(1-self.beta1**self.t),self.v[i][k]/(1-self.beta2**self.t); params[k]-=self.lr*m_hat/(np.sqrt(v_hat)+self.epsilon)

class HashMind:
    # --- CHANGE 1: DUAL-SOURCE PARAMETERS ---
    def __init__(self, context_length, vocab_size, d_model=256, n_heads=4, n_layers=4, learning_rate=1e-4, modulus=10**9+7):
        self.context_length, self.vocab_size, self.d_model, self.modulus = context_length, vocab_size, d_model, modulus
        self.params = {
            'token_embedding': np.random.randn(vocab_size, d_model) * 0.02,
            'hash_projector': np.random.randn(1, d_model) * 0.02,
            'output_proj': np.random.randn(d_model, vocab_size) * np.sqrt(2. / d_model)
        }
        self.pos_encoder = PositionalEncoding(d_model, context_length)
        self.blocks = [HashMindBlock(d_model, n_heads) for _ in range(n_layers)]
        self.optimizer = AdamOptimizer([self.params] + [b.params for b in self.blocks], lr=learning_rate)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # --- CHANGE 2: DUAL-SOURCE TRAINING STEP ---
    def train_step(self, inputs_hashes, inputs_indices, targets_indices):
        B, T = inputs_hashes.shape
        # 1. Get the primary character embedding via lookup
        char_embed = self.params['token_embedding'][inputs_indices]
        # 2. Get the context hash embedding via linear projection
        x_scaled = inputs_hashes[..., np.newaxis] / self.modulus
        hash_embed = x_scaled @ self.params['hash_projector']
        # 3. BRIDGE: Combine them
        x = char_embed + hash_embed
        # 4. Add positional encoding
        x = self.pos_encoder.forward(x)

        block_caches = []
        for block in self.blocks: x, cache = block.forward(x); block_caches.append(cache)
        logits = x[:, -1, :] @ self.params['output_proj']
        probs = self._softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(len(targets_indices)), targets_indices.flatten()] + 1e-9))

        d_logits = probs; d_logits[np.arange(len(targets_indices)), targets_indices.flatten()] -= 1; d_logits /= len(targets_indices)
        grads = {}; grads['output_proj'] = x[:, -1, :].T @ d_logits
        d_block_out = np.zeros_like(x); d_block_out[:, -1, :] = d_logits @ self.params['output_proj'].T
        all_block_grads = []
        for i in range(len(self.blocks) - 1, -1, -1): d_block_out, block_grads_i = self.blocks[i].backward(d_block_out, block_caches[i]); all_block_grads.insert(0, block_grads_i)
        
        # Backpropagate to combined embedding and then to dual sources
        d_x = self.pos_encoder.backward(d_block_out)
        d_char_embed, d_hash_embed = d_x, d_x # Gradient is distributed
        
        # Gradient for hash_projector
        grads['hash_projector'] = np.sum(x_scaled.transpose(0, 2, 1) @ d_hash_embed, axis=0)
        
        # Gradient for token_embedding (using np.add.at for correctness)
        d_token_embedding = np.zeros_like(self.params['token_embedding'])
        np.add.at(d_token_embedding, inputs_indices, d_char_embed)
        grads['token_embedding'] = d_token_embedding
        
        self.optimizer.step([grads] + all_block_grads)
        return loss

    # --- CHANGE 3: DUAL-SOURCE GENERATION ---
    def generate(self, prompt, steps=50, temperature=0.75):
        text = prompt
        values = ascii_converter.convert(text)
        indices = ascii_converter.get_indices(text)

        for _ in range(steps):
            if len(values) < self.context_length + HASH_WINDOW -1: return "Prompt too short for dual-source generation."
            
            # Prepare both hash and index contexts
            context_hashes = hasher.hash_sequence(values)[-self.context_length:]
            # We need the index of the *last* char that contributes to each hash
            context_indices = [indices[i + HASH_WINDOW - 1] for i in range(len(indices) - self.context_length - HASH_WINDOW + 1, len(indices) - HASH_WINDOW + 1)]

            # Combine embeddings
            char_embed = self.params['token_embedding'][np.array([context_indices])]
            x_scaled = np.array([context_hashes])[..., np.newaxis] / self.modulus
            hash_embed = x_scaled @ self.params['hash_projector']
            x = char_embed + hash_embed
            x = self.pos_encoder.forward(x)

            for block in self.blocks: x, _ = block.forward(x)
            logits = x[:, -1, :] @ self.params['output_proj']
            
            if temperature > 0: logits_scaled = logits / temperature; probs = self._softmax(logits_scaled); next_idx = np.random.choice(self.vocab_size, p=probs.flatten())
            else: next_idx = np.argmax(logits, axis=-1)[0]
            
            next_char = ascii_converter.idx_to_char.get(next_idx, ' ')
            text += next_char
            values.append(ascii_converter.char_to_val.get(next_char, ord(' ')))
            indices.append(ascii_converter.char_to_idx.get(next_char, 0))

            if next_char in ['.', '\n'] or len(text) > 200: break
        return text

# --- Main Execution Block ---
if __name__ == "__main__":
    CONTEXT_LENGTH, HASH_WINDOW = 16, 3
    D_MODEL, N_HEADS, N_LAYERS = 64, 4, 4

    np.random.seed(42)
    ascii_converter = SimplifiedASCIIConverter()
    hasher = RollingHasher(window_size=HASH_WINDOW)
    model = HashMind(CONTEXT_LENGTH, ascii_converter.vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, learning_rate=5e-4)


    corpus_text = """
    # --- Section 1: Python Code Snippets ---
    # Goal: Force the model to learn rigid structural n-grams like 'def ', 'self.', ' for', ' in ', '):', ' = '.
    
    def calculate_loss(self, y_true, y_pred):
        # This is a standard loss function.
        return np.mean(np.square(y_true - y_pred))
    
    class SimpleNetwork:
        def __init__(self, input_size, output_size):
            self.weights = np.random.randn(input_size, output_size)
            self.bias = np.zeros(output_size)
    
        def forward(self, x):
            # A simple forward pass calculation.
            return np.dot(x, self.weights) + self.bias
    
    for i in range(100):
        if i % 10 == 0:
            print(f"Processing item number {i}")
    
    # --- Section 2: Technical and Encyclopedic Definitions ---
    # Goal: Learn common definitional phrases and subject-specific n-grams.
    
    A neural network is a network or circuit of biological neurons, or, in a modern sense, an artificial neural network, composed of artificial neurons or nodes.
    A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior.
    The Hypertext Transfer Protocol (HTTP) is an application protocol for distributed, collaborative, hypermedia information systems.
    An algorithm is a finite sequence of well-defined instructions, typically used to solve a class of specific problems or to perform a computation.
    
    # --- Section 3: Structured Data (Markdown-like) ---
    # Goal: Learn formatting patterns and key-value relationships.
    
    - Project Name: HashMind V11
    - Author: User
    - Language: Python
    - Key Features:
        - Dual-Source Embedding (Character + Hash)
        - Transformer Architecture
        - Adam Optimizer
    
    - Task: Predict the next character.
    - Objective: Learn from sequential patterns.
    
    # --- Section 4: Literary Prose and Famous Quotes ---
    # Goal: Learn grammar, punctuation, and common English word n-grams.
    
    To be or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them.
    
    It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.
    All the world's a stage, and all the men and women merely players. They have their exits and their entrances.
    The quick brown fox jumps over the lazy dog. A stitch in time saves nine. The early bird catches the worm.
    She sells seashells by the seashore. The shells she sells are surely seashells.
    Where there is a will, there is a way. All that glitters is not gold. A journey of a thousand miles begins with a single step.
    
    # --- Section 5: Repetitive Sentences ---
    # Goal: Test the model's ability to count and replicate simple, long-range patterns.
    
    The first rule of the club is you do not talk about the club.
    The second rule of the club is you do not talk about the club.
    The model that can learn is a useful model. We are teaching it to predict the next character.
    This is a test of the architecture. Learning patterns is the goal of this network.
    """


    
    values = ascii_converter.convert(corpus_text)
    hashes = hasher.hash_sequence(values)
    indices = ascii_converter.get_indices(corpus_text)

    # --- CHANGE 4: NEW DATA PREPARATION PIPELINE ---
    all_input_hashes, all_input_indices, all_targets = [], [], []
    # We need enough characters to form a full input (context_length + hash_window) and a target
    num_examples = len(indices) - CONTEXT_LENGTH - HASH_WINDOW
    for i in range(num_examples):
        # Hash sequence for context
        hash_sequence = hashes[i : i + CONTEXT_LENGTH]
        all_input_hashes.append(hash_sequence)
        
        # The character index corresponding to the *end* of each hash's window
        # For hash[i], the last char is at indices[i + HASH_WINDOW - 1]
        index_sequence = [indices[k + HASH_WINDOW - 1] for k in range(i, i + CONTEXT_LENGTH)]
        all_input_indices.append(index_sequence)

        # The target is the character immediately following the last one we used for input indices
        target_char_pos = i + CONTEXT_LENGTH + HASH_WINDOW -1
        all_targets.append([indices[target_char_pos]])

    if not all_input_hashes: raise ValueError("Corpus is too short for the given parameters.")
        
    all_input_hashes, all_input_indices, all_targets = np.array(all_input_hashes), np.array(all_input_indices), np.array(all_targets)
    
    print("--- Starting Training (V11: Dual-Source Embedding) ---")
    start_time = time.time()
    epochs = 1500
    batch_size = 666

    for epoch in range(epochs):
        epoch_loss = 0
        permutation = np.random.permutation(len(all_input_hashes))
        for i in range(0, len(all_input_hashes), batch_size):
            batch_indices = permutation[i:i+batch_size]
            if len(batch_indices) == 0: continue
            
            # --- CHANGE 5: FEEDING DUAL SOURCES TO THE MODEL ---
            batch_hashes = all_input_hashes[batch_indices]
            batch_indices_in = all_input_indices[batch_indices]
            batch_targets = all_targets[batch_indices]
            
            loss = model.train_step(batch_hashes, batch_indices_in, batch_targets)
            epoch_loss += loss

        if (epoch+1) % 50 == 0:
            num_batches = math.ceil(len(all_input_hashes) / batch_size)
            if num_batches > 0: print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss / num_batches:.4f}")
    
    print(f"Training finished in {time.time() - start_time:.2f}s\n")
    
    print(f"--- Generating from trained model ---")
    prompt = "A model that can learn" # Needs to be long enough now
    print(f"Prompt: '{prompt}'")
    print("Result:", model.generate(prompt, steps=100))
    
    prompt = "The early bird catches"
    print(f"Prompt: '{prompt}'")
    print("Result:", model.generate(prompt, steps=100))

    prompt = "Where there is a will"
    print(f"Prompt: '{prompt}'")
    print("Result:", model.generate(prompt, steps=100))