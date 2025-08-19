import os

# --- Environment Setup for JAX/Flax on any hardware ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import random
import numpy as np
from tqdm import tqdm
import pickle
from typing import Any, Generator, Tuple, Dict, Optional
import sys
import argparse
from collections import deque
import signal
from dataclasses import dataclass
from functools import partial

# --- JAX Configuration for Performance and Stability ---
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_default_matmul_precision', 'bfloat16')
jax.config.update('jax_threefry_partitionable', True)

# --- Import Corpus and Tokenizer ---
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
except ImportError: print("[FATAL] `tokenizers` not found. `pip install tokenizers`."), sys.exit(1)
try:
    import CORPUS
except ImportError: print("[FATAL] CORPUS.py not found."), sys.exit(1)
try:
    from sklearn.neighbors import BallTree
except ImportError: print("[FATAL] `scikit-learn` not found. `pip install scikit-learn`."), sys.exit(1)

# --- XJDR's Metacognitive Sampler Logic (Integrated) ---
@dataclass(frozen=True)
class SamplerConfig:
  low_entropy_threshold = 0.3; high_entropy_threshold = 2.5
  low_varentropy_threshold = 1.2; high_varentropy_threshold = 2.5
  clarifying_question_token: int = 2
def get_entropy_metrics(logits: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logits_f32 = logits.astype(jnp.float32)
    probs=jax.nn.softmax(logits_f32); log_probs=jax.nn.log_softmax(logits_f32)
    entropy=-jnp.sum(probs*log_probs,axis=-1); varentropy=jnp.var(log_probs,axis=-1)
    return entropy, varentropy
@partial(jax.jit, static_argnames=("config",))
def xjdr_metacognitive_sample(key:jax.random.PRNGKey,logits:jnp.ndarray,config:SamplerConfig) -> jnp.ndarray:
    def _and(*args): return jnp.all(jnp.array(args))
    entropy,varentropy=get_entropy_metrics(logits)
    is_lelv=_and(entropy<config.low_entropy_threshold,varentropy<config.low_varentropy_threshold)
    is_helv=_and(entropy>config.high_entropy_threshold,varentropy<config.low_varentropy_threshold)
    is_lehv=_and(entropy<config.high_entropy_threshold,varentropy>config.high_varentropy_threshold)
    is_hehv=_and(entropy>config.high_entropy_threshold,varentropy>config.high_varentropy_threshold)
    case_index=jnp.argmax(jnp.array([is_lelv,is_helv,is_lehv,is_hehv,True]))
    def lelv_case(): return jax.random.categorical(key,logits)
    def helv_case(): return jnp.array(config.clarifying_question_token,dtype=jnp.int32)
    def lehv_case(): return jax.random.categorical(key,logits)
    def hehv_case():
        first_token=jax.random.categorical(key,logits); penalized_logits=logits.at[first_token].set(-jnp.inf)
        return jax.random.categorical(key,penalized_logits)
    def default_case(): return jax.random.categorical(key,logits)
    return jax.lax.switch(case_index,[lelv_case,helv_case,lehv_case,hehv_case,default_case])

# --- Helper Functions ---
def stream_text_from_corpus_data(data: Any) -> Generator[str, None, None]:
    if isinstance(data,str): yield data
    elif isinstance(data,dict):
        for v in data.values(): yield from stream_text_from_corpus_data(v)
    elif isinstance(data,list):
        for item in data: yield from stream_text_from_corpus_data(item)
class WubuTokenizer:
    def __init__(self,tokenizer_path:str): self.tokenizer_path=tokenizer_path; self.tokenizer=Tokenizer.from_file(tokenizer_path) if os.path.exists(tokenizer_path) else None
    def train(self,corpus_iterator,vocab_size): print("--- Training tokenizer... ---"); self.tokenizer=Tokenizer(models.BPE(unk_token="<UNK>")); self.tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=True); trainer=trainers.BpeTrainer(vocab_size=vocab_size,special_tokens=["<PAD>","<UNK>","<CQ>"]); self.tokenizer.train_from_iterator(corpus_iterator,trainer); self.tokenizer.save(self.tokenizer_path); print(f"--- Tokenizer trained. Vocab: {self.get_vocab_size()}. Saved to {self.tokenizer_path} ---")
    def get_vocab_size(self): return self.tokenizer.get_vocab_size() if self.tokenizer else 0
    def encode(self,text): return self.tokenizer.encode(text).ids if self.tokenizer else []
    def decode(self,ids): return self.tokenizer.decode(ids,skip_special_tokens=True) if self.tokenizer else ""
    @property
    def pad_id(self): return self.tokenizer.token_to_id("<PAD>")

class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def project(x):
        x_f32 = x.astype(jnp.float32)
        norm_sq = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
        max_norm = 1.0 - PoincareBall.EPS
        projected_f32 = jnp.where(norm_sq >= 1.0, x_f32 / jnp.sqrt(norm_sq + PoincareBall.EPS) * max_norm, x_f32)
        return projected_f32.astype(x.dtype)
    @staticmethod
    def mobius_add(x, y, c):
        x_f32, y_f32, c_f32 = x.astype(jnp.float32), y.astype(jnp.float32), c.astype(jnp.float32)
        x2, y2, xy = jnp.sum(x_f32*x_f32, -1, keepdims=True), jnp.sum(y_f32*y_f32, -1, keepdims=True), jnp.sum(x_f32*y_f32, -1, keepdims=True)
        num = (1 + 2 * c_f32 * xy + c_f32 * y2) * x_f32 + (1 - c_f32 * x2) * y_f32
        den = 1 + 2 * c_f32 * xy + c_f32**2 * x2 * y2
        return PoincareBall.project(num / den.clip(PoincareBall.EPS)).astype(x.dtype)
    @staticmethod
    def logmap0(y, c):
        y_f32, c_f32 = y.astype(jnp.float32), c.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c_f32).clip(PoincareBall.EPS)
        y_norm = jnp.linalg.norm(y_f32, axis=-1, keepdims=True)
        safe_y_norm = y_norm.clip(PoincareBall.EPS, 1.0 - PoincareBall.EPS)
        direction = y_f32 / safe_y_norm
        magnitude = jnp.arctanh(safe_y_norm) / sqrt_c
        result = jnp.where(y_norm < PoincareBall.EPS, jnp.zeros_like(y_f32), magnitude * direction)
        return result.astype(y.dtype)
    @staticmethod
    def expmap0(v, c):
        v_f32, c_f32 = v.astype(jnp.float32), c.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c_f32).clip(PoincareBall.EPS)
        v_norm = jnp.linalg.norm(v_f32, axis=-1, keepdims=True)
        safe_v_norm = v_norm.clip(PoincareBall.EPS)
        direction = v_f32 / safe_v_norm
        magnitude = jnp.tanh(sqrt_c * safe_v_norm) / sqrt_c
        result = jnp.where(v_norm < PoincareBall.EPS, jnp.zeros_like(v_f32), PoincareBall.project(magnitude * direction))
        return result.astype(v.dtype)
    @staticmethod
    def expmap_p(p, v, c):
        p_f32, v_f32, c_f32 = p.astype(jnp.float32), v.astype(jnp.float32), c.astype(jnp.float32)
        lambda_p = 2. / jnp.clip(1 - c_f32 * jnp.sum(p_f32 * p_f32, -1, keepdims=True), PoincareBall.EPS)
        expmapped_v = PoincareBall.expmap0(v_f32 * lambda_p, c_f32)
        return PoincareBall.mobius_add(p_f32, expmapped_v, c_f32).astype(p.dtype)

# --- Main Model Architectures (with Precision Control) ---
class ComplexEmbedding(nn.Module):
    vocab_size: int; features: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): real_embed=nn.Embed(self.vocab_size,self.features,name="real_embed",dtype=self.dtype)(x); imag_embed=nn.Embed(self.vocab_size,self.features,name="imag_embed",dtype=self.dtype)(x); return real_embed, imag_embed
class ComplexLayerNorm(nn.Module):
    dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x_complex: Tuple[jnp.ndarray, jnp.ndarray]): real,imag=x_complex; real_norm=nn.LayerNorm(dtype=self.dtype,name="real_ln")(real); imag_norm=nn.LayerNorm(dtype=self.dtype,name="imag_ln")(imag); return real_norm, imag_norm
class GRUCell(nn.Module):
    d_model_total: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, carry, x): xh=jnp.concatenate([x,carry],axis=-1); r=nn.sigmoid(nn.Dense(self.d_model_total,name="reset_gate_d",dtype=self.dtype)(xh)); u=nn.sigmoid(nn.Dense(self.d_model_total,name="update_gate_d",dtype=self.dtype)(xh)); c_in=jnp.concatenate([x,r*carry],axis=-1); c=nn.tanh(nn.Dense(self.d_model_total,name="candidate_gate_d",dtype=self.dtype)(c_in)); new_carry=(1-u)*carry+u*c; return new_carry, new_carry
class GalacticNavigator(nn.Module):
    d_model_total: int; vocab_size: int; dtype: Any = jnp.bfloat16
    def setup(self):
        self.d_model_comp=self.d_model_total//2; self.token_embed=ComplexEmbedding(self.vocab_size,self.d_model_comp,name="token_embed",dtype=self.dtype)
        self.gru=nn.scan(GRUCell,variable_broadcast='params',split_rngs={'params':False},in_axes=1,out_axes=1)(d_model_total=self.d_model_total,dtype=self.dtype)
        self.norm_syn=ComplexLayerNorm(dtype=self.dtype,name="norm_syn"); self.norm_sem=ComplexLayerNorm(dtype=self.dtype,name="norm_sem"); self.norm_exe=ComplexLayerNorm(dtype=self.dtype,name="norm_exe")
    def __call__(self, token_ids):
        initial_carry_real=jnp.zeros((token_ids.shape[0],self.d_model_total),dtype=self.dtype); token_embeds_r,token_embeds_i=self.token_embed(token_ids)
        xs_real=jnp.concatenate([token_embeds_r,token_embeds_i],axis=-1); _,hidden_states_real=self.gru(initial_carry_real,xs_real)
        h_r,h_i=jnp.split(hidden_states_real,2,axis=-1)
        return {'syn':self.norm_syn((h_r,h_i)), 'sem':self.norm_sem((h_r,h_i)), 'exe':self.norm_exe((h_r,h_i))}
class GalacticOracle(nn.Module):
    vocab_size: int; d_model: int
    @nn.compact
    def __call__(self, h_complex_dict: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]):
        h_r_syn,h_i_syn = h_complex_dict['syn']; h_r_sem,h_i_sem = h_complex_dict['sem']; h_r_exe,h_i_exe = h_complex_dict['exe']
        hidden_state_collapsed = jnp.concatenate([h_r_syn,h_i_syn, h_r_sem,h_i_sem, h_r_exe,h_i_exe], axis=-1).astype(jnp.bfloat16)
        x = nn.Dense(self.d_model*2,name="oracle_hidden",dtype=jnp.bfloat16)(hidden_state_collapsed); x = nn.gelu(x)
        return nn.Dense(self.vocab_size, name="oracle_output", dtype=jnp.float32)(x)

# --- Trend-Aware Q-Learning Controller ---
class JaxHakmemQController:
    def __init__(self,initial_lr:float,config:Dict[str,Any],logger_suffix:str=""):
        self.config=config; self.current_lr=initial_lr; self.logger_suffix=logger_suffix; self.q_table_size=int(self.config["q_table_size"]); self.num_actions=int(self.config["num_lr_actions"]); self.lr_change_factors=self.config["lr_change_factors"]; self.q_table=np.zeros((self.q_table_size,self.num_actions),dtype=np.float32); self.learning_rate_q=float(self.config["learning_rate_q"]); self.discount_factor_q=float(self.config["discount_factor_q"]); self.exploration_rate_q=float(self.config["exploration_rate_q"]); self.lr_min=float(self.config["lr_min"]); self.lr_max=float(self.config["lr_max"]); self.loss_history=deque(maxlen=int(self.config["metric_history_len"])); self.loss_min=float(self.config["loss_min"]); self.loss_max=float(self.config["loss_max"]); self.last_action_idx:Optional[int]=None; self.last_state_idx:Optional[int]=None; self.short_term_window=5
        print(f"--- HAKMEM Q-Controller ({self.logger_suffix}) initialized (Trend-Aware). LR: {self.current_lr:.2e}, Q-Table: {self.q_table.shape} ---")
    def _discretize_value(self,value:float) -> int:
        if value<=self.loss_min: return 0
        if value>=self.loss_max: return self.q_table_size-1
        bin_size=(self.loss_max-self.loss_min)/self.q_table_size; return min(int((value-self.loss_min)/bin_size),self.q_table_size-1)
    def _get_current_state_idx(self) -> int:
        if not self.loss_history: return self.q_table_size//2
        avg_loss=np.mean(list(self.loss_history)[-self.short_term_window:]); return self._discretize_value(avg_loss)
    def choose_action(self) -> float:
        self.last_state_idx=self._get_current_state_idx()
        if random.random()<self.exploration_rate_q: self.last_action_idx=random.randint(0,self.num_actions-1)
        else: self.last_action_idx=np.argmax(self.q_table[self.last_state_idx]).item()
        change_factor=self.lr_change_factors[self.last_action_idx]; self.current_lr=np.clip(self.current_lr*change_factor,self.lr_min,self.lr_max)
        return self.current_lr
    def update_q_value(self,current_loss:float):
        self.loss_history.append(current_loss)
        if self.last_state_idx is None or self.last_action_idx is None or len(self.loss_history)<self.loss_history.maxlen: return
        history_arr=np.array(self.loss_history); long_term_avg=np.mean(history_arr); short_term_avg=np.mean(history_arr[-self.short_term_window:])
        reward=long_term_avg-short_term_avg
        current_q=self.q_table[self.last_state_idx,self.last_action_idx]; next_state_idx=self._get_current_state_idx(); max_next_q=np.max(self.q_table[next_state_idx])
        new_q=current_q+self.learning_rate_q*(reward+self.discount_factor_q*max_next_q-current_q); self.q_table[self.last_state_idx,self.last_action_idx]=new_q
    def state_dict(self)->Dict[str,Any]: return {"current_lr":self.current_lr,"q_table":self.q_table.tolist(),"loss_history":list(self.loss_history),"last_action_idx":self.last_action_idx,"last_state_idx":self.last_state_idx}
    def load_state_dict(self,state_dict:Dict[str,Any]): self.current_lr=state_dict.get("current_lr",self.current_lr); self.q_table=np.array(state_dict.get("q_table",self.q_table.tolist()),dtype=np.float32); self.loss_history=deque(state_dict.get("loss_history",[]),maxlen=self.loss_history.maxlen); self.last_action_idx=state_dict.get("last_action_idx"); self.last_state_idx=state_dict.get("last_state_idx")

# --- The Galactic Funnel Cake System ---
class FunnelCakeConstructor:
    def __init__(self, config, tokenizer):
        self.config, self.tokenizer = config, tokenizer; self.d_model = config['d_model']; self.key = jax.random.PRNGKey(42)
        self.navigator = GalacticNavigator(d_model_total=self.d_model, vocab_size=tokenizer.get_vocab_size())
        self.oracle = GalacticOracle(vocab_size=tokenizer.get_vocab_size(), d_model=self.d_model)
        self.train_state = None; self.params = {}; self.q_controller = None
        self.H_sphere_points = {}; self.H_sphere_metadata = {}; self.ball_tree = {}
        self.manifolds = ['syn', 'sem', 'exe']
        for m in self.manifolds: self.H_sphere_points[m] = []; self.H_sphere_metadata[m] = []; self.ball_tree[m] = None
        self.should_shutdown = False; signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame): print("\n--- SIGINT received. Shutting down gracefully. ---"); self.should_shutdown = True

    def _init_models(self, training_mode: str):
        print(f"--- Initializing/Configuring Models for {training_mode}... ---")
        self.key, nav_key, oracle_key = jax.random.split(self.key, 3)
        dummy_tokens = jnp.zeros((2, self.config['train_chunk_size']), dtype=jnp.int32)
        if 'navigator' not in self.params:
            print("...Initializing new Navigator from scratch.")
            self.params['navigator'] = self.navigator.init(nav_key, dummy_tokens)['params']
        if 'oracle' not in self.params:
            print("...Initializing new Oracle from scratch.")
            dummy_h_states = self.navigator.apply({'params': self.params['navigator']}, dummy_tokens)
            self.params['oracle'] = self.oracle.init(oracle_key, dummy_h_states)['params']
        
        base_optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.config[f'learning_rate_{training_mode}'], weight_decay=0.01)
        tx = optax.chain(optax.clip_by_global_norm(1.0), base_optimizer)
        self.train_state = train_state.TrainState.create(
            apply_fn={'navigator': self.navigator.apply, 'oracle': self.oracle.apply}[training_mode],
            params=self.params[training_mode], tx=tx)

        self.q_controller = JaxHakmemQController(self.config[f'learning_rate_{training_mode}'], self.config, training_mode.capitalize())
        nav_pcount = sum(x.size for x in jax.tree.leaves(self.params['navigator'])); ora_pcount = sum(x.size for x in jax.tree.leaves(self.params['oracle']))
        print(f"--- Model Config Complete. Navigator: {nav_pcount:,} params | Oracle: {ora_pcount:,} params. ---")

    def train_navigator(self, token_file_path, epochs=3, batch_size=4096):
        self._init_models(training_mode="navigator")
        chunk_size = self.config['train_chunk_size']; tokens = np.memmap(token_file_path, dtype=np.int32, mode='r'); indices = np.arange(0, len(tokens) - (chunk_size + 10))
        @jax.jit
        def train_step(state, batch, margin):
            anchor, positive, negative = batch
            def loss_fn(params):
                h_anchor, h_pos, h_neg = [state.apply_fn({'params':params},x) for x in (anchor,positive,negative)]
                total_loss = 0.0
                for m in self.manifolds:
                    ah_r, ah_i = h_anchor[m][0][:,-1,:], h_anchor[m][1][:,-1,:]
                    ph_r, ph_i = h_pos[m][0][:,-1,:], h_pos[m][1][:,-1,:]
                    nh_r, nh_i = h_neg[m][0][:,-1,:], h_neg[m][1][:,-1,:]
                    dist_pos=jnp.sum((ah_r-ph_r)**2+(ah_i-ph_i)**2,axis=-1); dist_neg=jnp.sum((ah_r-nh_r)**2+(ah_i-nh_i)**2,axis=-1)
                    loss=jnp.mean(jnp.maximum(0,dist_pos-dist_neg+margin))
                    total_loss += loss * self.config['manifold_loss_weights'][m]
                return total_loss
            loss,grads=jax.value_and_grad(loss_fn)(state.params); return grads, loss
        
        for epoch in range(epochs):
            if self.should_shutdown: break
            margin=0.25+(epoch/max(1,epochs-1))*0.75; print(f"\n--- Starting Navigator Epoch {epoch+1}/{epochs} (Margin: {margin:.2f}) ---")
            np.random.shuffle(indices); pbar=tqdm(range(0,len(indices),batch_size),desc=f"Epoch {epoch+1}",total=len(indices)//batch_size)
            for i in pbar:
                if self.should_shutdown: break
                batch_indices=indices[i:i+batch_size];
                if len(batch_indices)<batch_size: continue
                new_lr=self.q_controller.choose_action()
                starts={'anchor':batch_indices,'positive':batch_indices+10,'negative':np.random.randint(0,len(tokens)-chunk_size,size=len(batch_indices))}
                batch_data={k:np.array([tokens[s:s+chunk_size] for s in v]) for k,v in starts.items()}
                grads,loss=train_step(self.train_state,tuple(batch_data.values()),margin)
                updates,new_opt_state=self.train_state.tx.update(grads,self.train_state.opt_state,self.train_state.params,hyperparams={'learning_rate':new_lr})
                new_params=optax.apply_updates(self.train_state.params,updates)
                self.train_state=self.train_state.replace(params=new_params,opt_state=new_opt_state)
                self.q_controller.update_q_value(loss.item()); pbar.set_postfix(avg_loss=f"{loss.item():.4f}",lr=f"{new_lr:.2e}")
        
        if not self.should_shutdown: print("\n--- Navigator Training Complete ---")
        self.params['navigator']=self.train_state.params; self.save_weights(self.config['basename'])

    def train_oracle(self, token_file_path, epochs=5, batch_size=896):
        self.load_weights(self.config['basename']); self._init_models(training_mode="oracle")
        chunk_size=self.config['train_chunk_size']; tokens=np.memmap(token_file_path,dtype=np.int32,mode='r'); indices=np.arange(0,len(tokens)-chunk_size)
        @jax.jit
        def train_step(nav_params, oracle_state, batch):
            inputs,labels=batch
            def loss_fn(oracle_params):
                hidden_states=self.navigator.apply({'params':nav_params},inputs); logits=oracle_state.apply_fn({'params':oracle_params},hidden_states)
                one_hot_labels=jax.nn.one_hot(labels,num_classes=self.tokenizer.get_vocab_size()); return optax.softmax_cross_entropy(logits,one_hot_labels).mean()
            loss,grads=jax.value_and_grad(loss_fn)(oracle_state.params); return grads, loss

        for epoch in range(epochs):
            if self.should_shutdown: break
            print(f"\n--- Starting Oracle Epoch {epoch+1}/{epochs} ---"); np.random.shuffle(indices)
            pbar=tqdm(range(0,len(indices),batch_size),desc=f"Epoch {epoch+1}",total=len(indices)//batch_size)
            for i in pbar:
                if self.should_shutdown: break
                batch_indices=indices[i:i+batch_size]
                if len(batch_indices)<batch_size: continue
                new_lr=self.q_controller.choose_action()
                input_batch=np.array([tokens[s:s+chunk_size-1] for s in batch_indices]); label_batch=np.array([tokens[s+1:s+chunk_size] for s in batch_indices])
                grads,loss=train_step(self.params['navigator'],self.train_state,(input_batch,label_batch))
                updates,new_opt_state=self.train_state.tx.update(grads,self.train_state.opt_state,self.train_state.params,hyperparams={'learning_rate':new_lr})
                new_params=optax.apply_updates(self.train_state.params,updates)
                self.train_state=self.train_state.replace(params=new_params,opt_state=new_opt_state)
                self.q_controller.update_q_value(loss.item()); pbar.set_postfix(avg_loss=f"{loss.item():.4f}",lr=f"{new_lr:.2e}")

        if not self.should_shutdown: print("\n--- Oracle Training Complete ---")
        self.params['oracle']=self.train_state.params; self.save_weights(self.config['basename'])

    def save_weights(self, basename):
        print(f"--- Saving model weights to {basename}.weights.pkl ---")
        with open(f"{basename}.weights.pkl", 'wb') as f:
            pickle.dump(jax.device_get(self.params), f)
        print("--- Weights saved. ---")

    def load_weights(self, basename):
        weight_file = f"{basename}.weights.pkl"
        if os.path.exists(weight_file):
            print(f"--- Loading weights from {weight_file} ---")
            with open(weight_file, 'rb') as f:
                self.params = pickle.load(f)

    def construct(self, token_file_path):
        self.load_weights(self.config['basename'])
        if 'navigator' not in self.params:
            print("[FATAL] Navigator must be trained."), sys.exit(1)

        print("--- Constructing Galactic Funnel Cake from memory-mapped tokens... ---")
        tokens = np.memmap(token_file_path, dtype=np.int32, mode='r')
        solidify_chunk_size = self.config['solidify_chunk_size']
        d_model = self.config['d_model']

        @partial(jax.jit, static_argnames=['solidify_chunk_size'])
        def process_token_chunk(params, token_chunk, solidify_chunk_size):
            num_tokens = token_chunk.shape[0]
            num_solid_points = num_tokens // solidify_chunk_size
            if num_solid_points == 0: return None
            
            truncated_tokens = token_chunk[:num_solid_points * solidify_chunk_size]
            states_dict = self.navigator.apply({'params': params}, truncated_tokens[None, :])
            
            tangent_vectors = {
                m: jnp.concatenate(states_dict[m], axis=-1)[0]
                   .reshape(num_solid_points, solidify_chunk_size, d_model)
                   .mean(axis=1)
                for m in self.manifolds
            }
            return tangent_vectors
        
        # --- FIX: WARM UP JIT COMPILATION to prevent hitching ---
        print("--- Warming up JIT compilation for construction... ---")
        warmup_tokens = jnp.zeros((solidify_chunk_size,), dtype=jnp.int32)
        _ = process_token_chunk(self.params['navigator'], warmup_tokens, solidify_chunk_size)
        print("--- JIT Warm-up complete. Starting construction. ---")
        # --- END FIX ---

        pbar = tqdm(total=len(tokens), unit='tok', unit_scale=True, desc="Constructing")
        chunk_size_in_bytes = self.config['construct_chunk_size_mb'] * 1024 * 1024
        chunk_size_in_tokens = chunk_size_in_bytes // 4

        for i in range(0, len(tokens), chunk_size_in_tokens):
            if self.should_shutdown: break
            
            # --- FIX: Load into NumPy (CPU RAM) first for smoother transfer ---
            token_chunk_np = np.array(tokens[i : i + chunk_size_in_tokens])
            if token_chunk_np.size == 0: continue
            
            tangent_vectors_dict = process_token_chunk(self.params['navigator'], token_chunk_np, solidify_chunk_size)
            if tangent_vectors_dict is None: continue

            for m in self.manifolds:
                c_val = jnp.array([self.config['manifold_curvatures'][m]], dtype=jnp.float32)
                
                # --- FIX: Correct vmap usage with in_axes ---
                vmapped_expmap0 = jax.vmap(PoincareBall.expmap0, in_axes=(0, None))
                new_points = np.array(vmapped_expmap0(tangent_vectors_dict[m], c_val))
                self.H_sphere_points[m].extend(new_points)

                num_new_points = new_points.shape[0]
                start_offset = i
                for k in range(num_new_points):
                    self.H_sphere_metadata[m].append({
                        'start_token_idx': start_offset + k * solidify_chunk_size,
                        'chunk_len': solidify_chunk_size
                    })

            pbar.update(len(token_chunk_np))
            pbar.set_postfix(solids=f"{len(self.H_sphere_points['sem']):,}")

        pbar.close()
        
        for m in self.manifolds:
            print(f"--- Manifold '{m}': {len(self.H_sphere_points[m])} solids. Building Ball Tree... ---")
            if self.H_sphere_points[m]:
                self.ball_tree[m] = BallTree(np.array(self.H_sphere_points[m], dtype=np.float32), leaf_size=40)
        self.save_cake(self.config['basename'])

    def save_cake(self, basename):
        print(f"--- Saving Funnel Cake to {basename}.cake ---")
        with open(f"{basename}.cake", 'wb') as f:
            pickle.dump({'config': self.config, 'ball_tree': self.ball_tree, 'H_sphere_metadata': self.H_sphere_metadata}, f)
        print("--- Cake saved. ---")

    def load_cake(self, basename):
        cake_file = f"{basename}.cake"; print(f"--- Loading Funnel Cake from {cake_file} ---")
        if not os.path.exists(cake_file): print(f"[FATAL] Cake file not found: {cake_file}"), sys.exit(1)
        with open(cake_file, 'rb') as f: state = pickle.load(f)
        self.config, self.ball_tree, self.H_sphere_metadata = state['config'], state['ball_tree'], state['H_sphere_metadata']
        if self.ball_tree.get('sem'): print(f"--- Funnel Cake loaded. Semantic manifold contains {self.ball_tree['sem'].data.shape[0]:,} solidified points. ---")

    def generate(self, prompt, max_new=200, momentum=0.8):
        self.load_weights(self.config['basename']); self.load_cake(self.config['basename'])
        if any(p not in self.params for p in ['navigator','oracle']) or not self.ball_tree.get('sem'): print("\n[ERROR] Models and Cake must be trained/constructed."), sys.exit(1)
        token_file_path=f"{self.config['basename']}.tokens.bin"; tokens_memmap=np.memmap(token_file_path,dtype=np.int32,mode='r')
        print(f"\n\033[1;32m{prompt}\033[0m",end='',flush=True)
        cq_token_id=self.tokenizer.tokenizer.token_to_id("<CQ>"); sampler_cfg=SamplerConfig(clarifying_question_token=cq_token_id if cq_token_id is not None else 2)
        @jax.jit
        def get_navigator_states(params,tokens): return self.navigator.apply({'params':params},tokens)
        @jax.jit
        def get_all_hidden_states(params,token_batch):
            h_dict=self.navigator.apply({'params':params},token_batch)
            return {m:jnp.concatenate(h_dict[m],axis=-1) for m in self.manifolds}
        @jax.jit
        def get_oracle_logits(params,h_dict): return self.oracle.apply({'params':params},h_dict)
        
        current_tokens=self.tokenizer.encode(prompt) or [self.tokenizer.pad_id]; states_dict=get_navigator_states(self.params['navigator'],jnp.array([current_tokens]));
        hidden_states_collapsed = {m:jnp.concatenate([states_dict[m][0][:,-1,:].squeeze(),states_dict[m][1][:,-1,:].squeeze()]) for m in self.manifolds}
        current_points = {}; velocity_vectors = {}
        for m in self.manifolds:
            _,start_indices=self.ball_tree[m].query(np.array(hidden_states_collapsed[m],dtype=np.float32).reshape(1,-1),k=1);
            current_points[m]=jnp.array(self.ball_tree[m].data[start_indices[0][0]]);
            velocity_vectors[m]=jnp.zeros_like(current_points[m])

        for _ in range(max_new):
            if self.should_shutdown: break
            self.key,subkey=jax.random.split(self.key);
            
            total_guidance_vector = jnp.zeros_like(velocity_vectors['sem'], dtype=jnp.float32)
            for m in self.manifolds:
                c_val=jnp.array([self.config['manifold_curvatures'][m]], dtype=jnp.float32)
                k=self.config['knn_sampling']; step_size=self.config['geodesic_step_size']
                current_anchor_state=hidden_states_collapsed[m]; intent_vector=PoincareBall.logmap0(current_points[m],c_val)-current_anchor_state
                _,indices=self.ball_tree[m].query(np.expand_dims(np.array(current_points[m]),0),k=k); neighbors=jnp.array([self.ball_tree[m].data[idx] for idx in indices.flatten()])
                local_flow=jnp.mean(jax.vmap(PoincareBall.logmap0,in_axes=(0,None))(neighbors,c_val),axis=0)
                guidance_vector=(intent_vector*0.8)+(local_flow*0.2); new_velocity=(velocity_vectors[m]*momentum)+(guidance_vector*(1-momentum)); velocity_vectors[m]=new_velocity/jnp.linalg.norm(new_velocity).clip(1e-6)
                current_points[m]=PoincareBall.expmap_p(current_points[m],velocity_vectors[m]*step_size,c_val)
                total_guidance_vector += velocity_vectors[m] * self.config['manifold_guidance_weights'][m]

            oracle_input_states = {}
            for m in self.manifolds:
                k_micro=self.config['micro_cake_neighbors']
                _,indices=self.ball_tree[m].query(np.expand_dims(np.array(current_points[m]),0),k=k_micro)
                micro_tokens=[];
                for idx in indices.flatten(): meta=self.H_sphere_metadata[m][idx]; start_idx,chunk_len=meta['start_token_idx'],meta['chunk_len']; micro_tokens.extend(tokens_memmap[start_idx:start_idx+chunk_len])
                if not micro_tokens: oracle_input_states[m]=hidden_states_collapsed[m]; continue
                micro_states=get_all_hidden_states(self.params['navigator'],jnp.array([micro_tokens]))[m][0]
                micro_ball_tree=BallTree(np.array(micro_states,dtype=np.float32))
                target_tangent=hidden_states_collapsed[m].astype(jnp.float32)+total_guidance_vector*self.config['micro_step_size']
                _,best_next_idx=micro_ball_tree.query(np.array(target_tangent).reshape(1,-1),k=1)
                oracle_input_states[m]=micro_states[best_next_idx[0][0]]

            oracle_input_h_dict={m:jnp.split(oracle_input_states[m],2) for m in self.manifolds}
            oracle_input_h_dict_batched={m:(r[None,None,:],i[None,None,:]) for m,(r,i) in oracle_input_h_dict.items()}
            final_logits=get_oracle_logits(self.params['oracle'],oracle_input_h_dict_batched)
            next_token_id=xjdr_metacognitive_sample(subkey,final_logits.squeeze(),sampler_cfg).item()
            decoded_token=self.tokenizer.tokenizer.decode([next_token_id],skip_special_tokens=False); print(decoded_token.replace('Ġ',' '),end='',flush=True)
            current_tokens.append(next_token_id)
            if len(current_tokens)>256: current_tokens.pop(0)
            states_dict=get_navigator_states(self.params['navigator'],jnp.array([current_tokens]));
            hidden_states_collapsed={m:jnp.concatenate([states_dict[m][0][:,-1,:].squeeze(),states_dict[m][1][:,-1,:].squeeze()]) for m in self.manifolds}
        print()

def main():
    parser=argparse.ArgumentParser(description="WubuMind Galactic Funnel Cake v25.2 (Performance Edition)")
    parser.add_argument('command',choices=['train_navigator','train_oracle','construct','generate'],help="The command to execute.")
    parser.add_argument('--basename',type=str,default="wubumind_v25_galactic",help="Basename for model files.")
    parser.add_argument('--epochs',type=int,default=5,help="Number of training epochs.")
    parser.add_argument('--batch-size',type=int,default=256,help="Batch size for training. Adjust based on VRAM.")
    args=parser.parse_args()
    MODEL_CONFIG={
        'd_model':256,'solidify_chunk_size':256,'geodesic_step_size':0.25,'knn_sampling':3,'basename':args.basename,
        'learning_rate_navigator':1e-4,'learning_rate_oracle':5e-5,'train_chunk_size':128,
        'construct_chunk_size_mb': 64, # New setting for faster construction
        'micro_cake_neighbors':5,'micro_step_size':0.05,
        'manifold_curvatures':{'syn':5.0,'sem':1.0,'exe':0.1},
        'manifold_loss_weights':{'syn':0.2,'sem':0.6,'exe':0.2},
        'manifold_guidance_weights':{'syn':0.2,'sem':0.5,'exe':0.3},
    }
    QLEARN_CONFIG={
        "q_table_size":10,"num_lr_actions":5,"lr_change_factors":[0.5,0.9,1.0,1.1,1.5],"learning_rate_q":0.1,
        "discount_factor_q":0.9,"exploration_rate_q":0.1,"lr_min":1e-7,"lr_max":1e-2,"metric_history_len":25,
        "loss_min":0.0,"loss_max":10.0
    }
    FULL_CONFIG = {**MODEL_CONFIG, **QLEARN_CONFIG}
    TOKENIZER_CONFIG={'vocab_size':8192,'tokenizer_path':f"{args.basename}_bpe.json"}
    TOKEN_FILE_PATH=f"{args.basename}.tokens.bin"

    print(f"--- WubuMind Galactic Funnel Cake Foundry v25.2 (Performance Edition) ---")
    tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
    if not tokenizer.tokenizer:
        print("--- No tokenizer found. Pre-tokenizing corpus first. ---")
        corpora=[getattr(CORPUS,n) for n in dir(CORPUS) if not n.startswith('_') and n.isupper()]
        tokenizer.train((chunk for corpus in corpora for chunk in stream_text_from_corpus_data(corpus)),TOKENIZER_CONFIG['vocab_size'])
        with open(TOKEN_FILE_PATH,'wb') as f_out:
            for text_chunk in stream_text_from_corpus_data(corpora):
                if token_ids:=tokenizer.encode(text_chunk): np.array(token_ids,dtype=np.int32).tofile(f_out)
        print("--- Pre-tokenization complete. Please run the desired command again. ---"); sys.exit(0)
    
    constructor=FunnelCakeConstructor(FULL_CONFIG,tokenizer)
    if not os.path.exists(TOKEN_FILE_PATH): print(f"[FATAL] Token file not found: {TOKEN_FILE_PATH}"), sys.exit(1)

    if args.command=='train_navigator': constructor.train_navigator(TOKEN_FILE_PATH,epochs=args.epochs,batch_size=args.batch_size)
    elif args.command=='train_oracle': constructor.train_oracle(TOKEN_FILE_PATH,epochs=args.epochs,batch_size=args.batch_size)
    elif args.command=='construct': constructor.construct(TOKEN_FILE_PATH)
    elif args.command=="generate":
        print("\n--- Galactic Oracle Command Console (v25.2) ---")
        while True:
            if constructor.should_shutdown: break
            try: prompt=input("\nYour Prompt> ")
            except EOFError: print("\n--- Exiting. ---"); break
            if prompt.lower() in ["exit","quit"]: break
            constructor.generate(prompt)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"); sys.exit(0)
