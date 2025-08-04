# Gemini: Okay, I see the output. The model learned that spaces are common but didn't grasp the deeper patterns.
# That's expected with a small dataset and greedy generation.
#
# Let's try two things:
# 1. I've expanded the training corpus to give it more to learn from.
# 2. I've changed the `generate` function to use temperature-based sampling. This encourages creativity
#    and prevents it from getting stuck on the most common character (the space).
#
# Let's see what it learns now.

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
        
        # Build a consistent vocabulary
        chars = ([str(i) for i in range(10)] +
                 [chr(ord('A') + i) for i in range(26)] +
                 [chr(ord('a') + i) for i in range(26)] +
                 [' ', '.', ',', '!', '?', '\n'])
        
        for char in sorted(list(set(chars))): # Ensure consistent ordering
            self._add_char(char, ord(char))

    def _add_char(self, char, val):
        if char not in self.char_to_val:
            self.char_to_val[char] = val
            self.val_to_char[val] = char
            self.char_to_idx[char] = self.vocab_size
            self.idx_to_char[self.vocab_size] = char
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

# --- Part 2: The Neural Network (V10.1 with Generation Fix) ---

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
        mean=x.mean(axis=-1, keepdims=True); var=x.var(axis=-1, keepdims=True)
        x_hat=(x-mean)/np.sqrt(var+eps); out=gamma*x_hat+beta
        return out, (x, x_hat, mean, var, gamma, eps)
    def _layer_norm_backward(self, d_out, cache):
        x,x_hat,mean,var,gamma,eps=cache; B,T,D=x.shape
        d_beta=d_out.sum(axis=(0,1)); d_gamma=(d_out*x_hat).sum(axis=(0,1)); dx_hat=d_out*gamma
        dvar=np.sum(dx_hat*(x-mean)*-0.5*(var+eps)**-1.5, axis=-1, keepdims=True)
        dmean=np.sum(dx_hat*-1/np.sqrt(var+eps), axis=-1, keepdims=True) + dvar*np.mean(-2.*(x-mean),axis=-1,keepdims=True)
        dx=(dx_hat/np.sqrt(var+eps))+(dvar*2.*(x-mean)/D)+(dmean/D)
        return dx, d_gamma, d_beta
    def _attention_forward(self, x_norm):
        B,T,C=x_norm.shape; qkv=x_norm@self.params['qkv_proj']; q,k,v=np.split(qkv,3,axis=-1)
        q,k,v=[y.reshape(B,T,self.n_heads,self.d_head).transpose(0,2,1,3) for y in (q,k,v)]
        attn_scores=(q@k.transpose(0,1,3,2))/math.sqrt(self.d_head)
        exp_scores=np.exp(attn_scores-np.max(attn_scores,axis=-1,keepdims=True))
        attn_probs=exp_scores/exp_scores.sum(axis=-1,keepdims=True)
        attn_output=(attn_probs@v).transpose(0,2,1,3).reshape(B,T,C)
        out=attn_output@self.params['out_proj']
        return out,(x_norm,q,k,v,attn_probs,attn_output)
    def _attention_backward(self,d_out,cache):
        x_norm,q,k,v,attn_probs,attn_output=cache; B,T,C=x_norm.shape
        d_out_proj=attn_output.reshape(B*T,C).T@d_out.reshape(B*T,C); d_attn_output=d_out@self.params['out_proj'].T
        d_attn_output=d_attn_output.reshape(B,T,self.n_heads,self.d_head).transpose(0,2,1,3)
        d_attn_probs=d_attn_output@v.transpose(0,1,3,2); dv=attn_probs.transpose(0,1,3,2)@d_attn_output
        d_attn_scores=attn_probs*(d_attn_probs-np.sum(d_attn_probs*attn_probs,axis=-1,keepdims=True)); d_attn_scores/=math.sqrt(self.d_head)
        dq=d_attn_scores@k; dk=d_attn_scores.transpose(0,1,3,2)@q; dq,dk,dv=[y.transpose(0,2,1,3).reshape(B,T,C) for y in (dq,dk,dv)]
        d_qkv=np.concatenate([dq,dk,dv],axis=-1); d_qkv_proj=x_norm.reshape(B*T,C).T@d_qkv.reshape(B*T,3*C)
        dx_norm=d_qkv@self.params['qkv_proj'].T
        return dx_norm, {'qkv_proj':d_qkv_proj, 'out_proj':d_out_proj}
    def forward(self,x):
        norm1_out,norm1_cache=self._layer_norm_forward(x,self.params['norm1_gamma'],self.params['norm1_beta'])
        attn_out,attn_cache=self._attention_forward(norm1_out); x_res1=x+attn_out
        norm2_out,norm2_cache=self._layer_norm_forward(x_res1,self.params['norm2_gamma'],self.params['norm2_beta'])
        ffn_intermediate=np.maximum(0,norm2_out@self.params['ffn1']); ffn_out=ffn_intermediate@self.params['ffn2']
        x_final=x_res1+ffn_out
        return x_final, (x,x_res1,norm1_cache,attn_cache,norm2_out,norm2_cache,ffn_intermediate)
    def backward(self,d_out,cache):
        x_in,x_res1,norm1_cache,attn_cache,norm2_out,norm2_cache,ffn_intermediate=cache; grads={}; B,T,C=x_in.shape
        d_ffn_out,d_x_res1_from_res=d_out,d_out; d_ffn2=ffn_intermediate.reshape(B*T,self.d_model*4).T@d_ffn_out.reshape(B*T,C)
        d_ffn_intermediate=d_ffn_out@self.params['ffn2'].T; d_ffn_intermediate[ffn_intermediate<=0]=0
        d_ffn1=norm2_out.reshape(B*T,C).T@d_ffn_intermediate.reshape(B*T,self.d_model*4)
        d_norm2_out=d_ffn_intermediate@self.params['ffn1'].T; grads.update({'ffn1':d_ffn1,'ffn2':d_ffn2})
        d_x_res1_from_ffn,d_gamma2,d_beta2=self._layer_norm_backward(d_norm2_out,norm2_cache); grads.update({'norm2_gamma':d_gamma2,'norm2_beta':d_beta2})
        d_x_res1=d_x_res1_from_ffn+d_x_res1_from_res; d_attn_out,d_x_in_from_res=d_x_res1,d_x_res1
        dx_norm,attn_grads=self._attention_backward(d_attn_out,attn_cache); grads.update(attn_grads)
        d_x_in_from_norm,d_gamma1,d_beta1=self._layer_norm_backward(dx_norm,norm1_cache); grads.update({'norm1_gamma':d_gamma1,'norm1_beta':d_beta1})
        d_x_in=d_x_in_from_norm+d_x_in_from_res
        return d_x_in,grads

class AdamOptimizer:
    def __init__(self, params_list, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params_list,self.lr,self.beta1,self.beta2,self.epsilon=params_list,lr,beta1,beta2,epsilon
        self.m=[{k:np.zeros_like(p) for k,p in params.items()} for params in params_list]; self.v=[{k:np.zeros_like(p) for k,p in params.items()} for params in params_list]
        self.t=0
    def step(self,grads_list):
        self.t+=1
        for i,(params,grads) in enumerate(zip(self.params_list,grads_list)):
            for k in params.keys():
                if k not in grads or grads[k] is None: continue
                # Gradient clipping
                np.clip(grads[k], -1.0, 1.0, out=grads[k])
                self.m[i][k]=self.beta1*self.m[i][k]+(1-self.beta1)*grads[k]; self.v[i][k]=self.beta2*self.v[i][k]+(1-self.beta2)*(grads[k]**2)
                m_hat,v_hat=self.m[i][k]/(1-self.beta1**self.t), self.v[i][k]/(1-self.beta2**self.t)
                params[k]-=self.lr*m_hat/(np.sqrt(v_hat)+self.epsilon)

class HashMind:
    def __init__(self, context_length, vocab_size, d_model=256, n_heads=4, n_layers=4, learning_rate=1e-4, modulus=10**9+7):
        self.context_length, self.vocab_size, self.d_model, self.modulus = context_length, vocab_size, d_model, modulus
        self.params = {
            'hash_embedding': np.random.randn(1, d_model) * 0.02,
            'output_proj': np.random.randn(d_model, vocab_size) * np.sqrt(2. / d_model)
        }
        self.pos_encoder = PositionalEncoding(d_model, context_length)
        self.blocks = [HashMindBlock(d_model, n_heads) for _ in range(n_layers)]
        self.optimizer = AdamOptimizer([self.params] + [b.params for b in self.blocks], lr=learning_rate)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def train_step(self, inputs_hashes, targets_indices):
        B, T = inputs_hashes.shape
        x_scaled = inputs_hashes[..., np.newaxis] / self.modulus
        x = x_scaled @ self.params['hash_embedding']
        x = self.pos_encoder.forward(x)
        block_caches = []
        for block in self.blocks:
            x, cache = block.forward(x)
            block_caches.append(cache)
        logits = x[:, -1, :] @ self.params['output_proj']
        probs = self._softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(len(targets_indices)), targets_indices.flatten()] + 1e-9))
        d_logits = probs
        d_logits[np.arange(len(targets_indices)), targets_indices.flatten()] -= 1
        d_logits /= len(targets_indices)
        grads = {}
        grads['output_proj'] = x[:, -1, :].T @ d_logits
        d_block_out = np.zeros_like(x)
        d_block_out[:, -1, :] = d_logits @ self.params['output_proj'].T
        all_block_grads = []
        for i in range(len(self.blocks) - 1, -1, -1):
            d_block_out, block_grads_i = self.blocks[i].backward(d_block_out, block_caches[i])
            all_block_grads.insert(0, block_grads_i)
        d_x = self.pos_encoder.backward(d_block_out)
        grads['hash_embedding'] = np.sum(x_scaled.transpose(0, 2, 1) @ d_x, axis=0)
        self.optimizer.step([grads] + all_block_grads)
        return loss

    # --- CHANGE 1: MODIFIED GENERATION WITH TEMPERATURE SAMPLING ---
    def generate(self, prompt, steps=50, temperature=0.75):
        text = prompt
        values = ascii_converter.convert(text)
        for _ in range(steps):
            hashes = hasher.hash_sequence(values)
            if not hashes: return "Prompt too short."

            context_hashes = ([0] * (self.context_length - len(hashes))) + hashes[-self.context_length:]

            x_scaled = np.array([context_hashes])[..., np.newaxis] / self.modulus
            x = x_scaled @ self.params['hash_embedding']
            x = self.pos_encoder.forward(x)
            for block in self.blocks:
                x, _ = block.forward(x)
            logits = x[:, -1, :] @ self.params['output_proj']

            # Instead of argmax, we sample from the distribution after adjusting for temperature
            if temperature > 0:
                logits_scaled = logits / temperature
                probs = self._softmax(logits_scaled)
                next_idx = np.random.choice(self.vocab_size, p=probs.flatten())
            else: # temperature=0 is equivalent to argmax
                next_idx = np.argmax(logits, axis=-1)[0]

            next_char = ascii_converter.idx_to_char.get(next_idx, ' ') # Safer lookup

            text += next_char
            values.append(ascii_converter.char_to_val.get(next_char, ord(' ')))
            # A more robust stop condition
            if next_char == '.' or next_char == '\n' or len(text) > 200:
                break
        return text

# --- Main Execution Block ---
if __name__ == "__main__":
    CONTEXT_LENGTH, HASH_WINDOW = 16, 3
    D_MODEL, N_HEADS, N_LAYERS = 64, 4, 4

    np.random.seed(42)
    ascii_converter = SimplifiedASCIIConverter()
    hasher = RollingHasher(window_size=HASH_WINDOW)
    model = HashMind(CONTEXT_LENGTH, ascii_converter.vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, learning_rate=3e-4)

    # --- CHANGE 2: EXPANDED CORPUS ---
    corpus_text = ("The quick brown fox jumps over the lazy dog. A stitch in time saves nine. "
                   "A model that can learn is a useful model. We are teaching it to predict the next character. "
                   "This is a test of the HashMind architecture. Learning from patterns is the goal of this network. "
                   "The rain in Spain stays mainly in the plain. How much wood would a woodchuck chuck? "
                   "She sells seashells by the seashore. Peter Piper picked a peck of pickled peppers. "
                   "To be or not to be, that is the question. The early bird catches the worm. "
                   "Where there is a will, there is a way. All that glitters is not gold. "
                   "An apple a day keeps the doctor away. A journey of a thousand miles begins with a single step.")

    values = ascii_converter.convert(corpus_text)
    hashes = hasher.hash_sequence(values)
    indices = ascii_converter.get_indices(corpus_text)

    inputs, targets = [], []
    num_examples = len(hashes) - CONTEXT_LENGTH
    for i in range(num_examples):
        input_sequence = hashes[i:i+CONTEXT_LENGTH]
        target_char_pos = i + CONTEXT_LENGTH + HASH_WINDOW - 1
        if target_char_pos < len(indices):
            inputs.append(input_sequence)
            targets.append([indices[target_char_pos]])

    if not inputs:
        raise ValueError("Corpus is too short for the given context length and hash window.")

    inputs, targets = np.array(inputs), np.array(targets)

    print("--- Starting Training (V10.1: More Data & Sampling) ---")
    start_time = time.time()
    # --- CHANGE 3: MORE TRAINING ---
    epochs = 600
    batch_size = 64 # Larger batch size for more stable gradients

    for epoch in range(epochs):
        epoch_loss = 0
        permutation = np.random.permutation(len(inputs))
        for i in range(0, len(inputs), batch_size):
            batch_indices = permutation[i:i+batch_size]
            if len(batch_indices) == 0: continue

            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]

            loss = model.train_step(batch_inputs, batch_targets)
            epoch_loss += loss

        if (epoch+1) % 50 == 0:
            num_batches = math.ceil(len(inputs) / batch_size)
            if num_batches > 0:
              avg_loss = epoch_loss / num_batches
              print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    print(f"Training finished in {time.time() - start_time:.2f}s\n")

    print(f"--- Generating from trained model ---")
    prompt = "A model that can"
    print(f"Prompt: '{prompt}'")
    print("Result:", model.generate(prompt, steps=100, temperature=0.75))

    prompt = "The quick brown"
    print(f"Prompt: '{prompt}'")
    print("Result:", model.generate(prompt, steps=100, temperature=0.75))
    
    prompt = "Peter Piper"
    print(f"Prompt: '{prompt}'")
    print("Result:", model.generate(prompt, steps=100, temperature=0.75))

    print("\nHow does this look? With more data and better sampling, it should be able to form words.")
    print("If it's better, we can try giving it even more complex data. If not, we might need to look at the model architecture itself.")
