# %% PYTHON FILE: wubumind_galactic_core_v1.py
# The Autonomous Adversarial Core. Version 3.1
# NEW: Integrated the HAKMEMQController to autonomously manage Generator and
#      Discriminator learning rates, creating a self-tuning adversarial dynamic.
#
# The architecture is a Generative Adversarial Network (GAN). This setup forces
# the Generator to learn the deep, latent structure of the data stored in its
# hyperbolic spheres to fool the Discriminator.

import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import remat
from flax.training import train_state
from flax import serialization
import optax
from functools import partial
import numpy as np
import time
from tqdm import tqdm
import pickle
import json
from typing import Any, Sequence, Dict, Tuple, Deque
from collections import deque
import sys
import dataclasses
import signal
import traceback

# --- TOKENIZER & CORPUS IMPORTS ---
try:
    from tokenizers import Tokenizer; from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer; from tokenizers.pre_tokenizers import Whitespace
except ImportError: print("[FATAL] `tokenizers` not found. `pip install tokenizers`."), sys.exit(1)
try:
    import CORPUS
except ImportError: print("[FATAL] CORPUS.py not found."), sys.exit(1)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
jax.config.update("jax_debug_nans", False)

# --- Helper & Tokenizer (Unchanged) ---
def distill_text_from_corpus(data: Any) -> str:
    if isinstance(data, str): return data + "\n"
    if isinstance(data, dict): return "".join(distill_text_from_corpus(v) for v in data.values())
    if isinstance(data, list): return "".join(distill_text_from_corpus(item) for item in data)
    return ""
class WubuTokenizer:
    def __init__(self, tokenizer_path: str = "wubumind_bpe.json"):
        self.tokenizer_path = tokenizer_path
        if os.path.exists(tokenizer_path): self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else: self.tokenizer = None
    def train(self, text_corpus, vocab_size):
        print("--- Training tokenizer... ---"); self.tokenizer=Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer=Whitespace(); trainer=BpeTrainer(vocab_size=vocab_size,special_tokens=["<PAD>","<UNK>"])
        self.tokenizer.train_from_iterator([text_corpus],trainer); self.tokenizer.save(self.tokenizer_path)
        print(f"--- Tokenizer trained. Vocab: {self.get_vocab_size()}. Saved to {self.tokenizer_path} ---")
    def get_vocab_size(self): return self.tokenizer.get_vocab_size() if self.tokenizer else 0
    def encode(self, text): return self.tokenizer.encode(text).ids if self.tokenizer else []
    def decode(self, ids): return self.tokenizer.decode(ids) if self.tokenizer else ""
    @property
    def pad_id(self): return self.tokenizer.token_to_id("<PAD>")

# --- Geometric Primitives ---
class PoincareBall:
    EPS=1e-7
    @staticmethod
    def project(x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(PoincareBall.EPS)
        max_norm = 1.0 - PoincareBall.EPS
        return jnp.where(norm >= 1.0, x / norm * max_norm, x)

    @staticmethod
    def mobius_add(x, y, c):
        x2, y2, xy = jnp.sum(x*x, -1, keepdims=True), jnp.sum(y*y, -1, keepdims=True), jnp.sum(x*y, -1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return PoincareBall.project(num / den.clip(PoincareBall.EPS))

    @staticmethod
    def logmap0(y, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
        direction = y / y_norm.clip(PoincareBall.EPS)
        magnitude = jnp.arctanh(y_norm.clip(max=1.0-PoincareBall.EPS)) / sqrt_c
        return jnp.where(y_norm < PoincareBall.EPS, jnp.zeros_like(y), magnitude * direction)

    @staticmethod
    def expmap0(v, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        direction = v / v_norm.clip(PoincareBall.EPS)
        magnitude = jnp.tanh(sqrt_c * v_norm) / sqrt_c
        return PoincareBall.project(jnp.where(v_norm < PoincareBall.EPS, jnp.zeros_like(v), magnitude * direction))

    @staticmethod
    def dist(x, y, c):
        c_bcast = c[..., None]
        sqrt_c = jnp.sqrt(c_bcast).clip(PoincareBall.EPS)
        add_xy = PoincareBall.mobius_add(-x, y, c_bcast)
        add_norm = jnp.linalg.norm(add_xy, axis=-1)
        arg = jnp.minimum(sqrt_c.squeeze(-1) * add_norm, 1.0 - PoincareBall.EPS)
        return 2. * jnp.arctanh(arg) / sqrt_c.squeeze(-1)
        
# --- WuBu Core Block (Shared) ---
class HyperbolicAttention(nn.Module):
    dim:int;n_heads:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @staticmethod
    def apply_rotary_emb(x,freqs_cis): x_f32=x.astype(jnp.float32);x_r,x_i=jnp.split(x_f32,2,-1);x_c=jax.lax.complex(x_r,x_i);freqs_cis=freqs_cis.reshape(1,1,freqs_cis.shape[0],freqs_cis.shape[1]);x_rotated=x_c*freqs_cis; return jnp.concatenate([x_rotated.real,x_rotated.imag],-1).astype(x.dtype)
    @nn.compact
    def __call__(self,x_hyp,freqs_cis,c_sphere):
        B,N,_=x_hyp.shape;h_dim=self.dim//self.n_heads
        qkv_proj=nn.Dense(self.dim*3,name="qkv_proj",dtype=self.dtype,param_dtype=self.param_dtype);out_proj=nn.Dense(self.dim,name="out_proj",dtype=self.dtype,param_dtype=self.param_dtype)
        c_per_head_logits=self.param('c_per_head_logits',nn.initializers.zeros,(self.n_heads,),self.param_dtype);geo_scale=self.param('geo_scale',nn.initializers.ones,(1,self.n_heads,1,1),self.param_dtype)
        x_tangent=PoincareBall.logmap0(x_hyp,c_sphere);qkv=qkv_proj(x_tangent).reshape(B,N,3,self.n_heads,h_dim).transpose((2,0,3,1,4));q,k,v_euc=qkv[0],qkv[1],qkv[2]
        q_rot,k_rot=self.apply_rotary_emb(q,freqs_cis),self.apply_rotary_emb(k,freqs_cis);c_per_head=nn.softplus(c_per_head_logits).reshape(1,self.n_heads,1,1)
        q_hyp,k_hyp=PoincareBall.expmap0(q_rot,c_per_head),PoincareBall.expmap0(k_rot,c_per_head);dist=PoincareBall.dist(q_hyp[:,:,:,None,:],k_hyp[:,:,None,:,:],c_per_head)
        mask=nn.make_causal_mask(jnp.ones((B,N),dtype=bool));attn_scores=jnp.where(mask,-geo_scale*dist,-jnp.inf)
        attn_weights=nn.softmax(attn_scores.astype(jnp.float32),axis=-1).astype(self.dtype);attn_out_euc=(attn_weights@v_euc).transpose((0,2,1,3)).reshape(B,N,self.dim)
        return out_proj(attn_out_euc)
class HyperbolicFFN(nn.Module):
    dim:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @nn.compact
    def __call__(self,x_hyp,c_sphere): return nn.Sequential([nn.Dense(self.dim*4,dtype=self.dtype,param_dtype=self.param_dtype),nn.gelu,nn.Dense(self.dim,dtype=self.dtype,param_dtype=self.param_dtype)])(PoincareBall.logmap0(x_hyp,c_sphere))
class GalacticBlock(nn.Module):
    dim:int;n_heads:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @remat
    @nn.compact
    def __call__(self,states,curvatures,freqs_cis):
        x_euc,x_syn,x_sem,x_exe=states['euc'],states['syn'],states['sem'],states['exe'];c_syn,c_sem,c_exe=curvatures['syn'],curvatures['sem'],curvatures['exe']
        norm_euc_1=nn.LayerNorm(dtype=jnp.float32,name="norm_euc_1")(x_euc).astype(self.dtype);syn_informed=PoincareBall.mobius_add(x_syn,PoincareBall.expmap0(norm_euc_1,c_syn),c_syn)
        sem_informed=PoincareBall.mobius_add(x_sem,PoincareBall.expmap0(norm_euc_1,c_sem),c_sem);exe_informed=PoincareBall.mobius_add(x_exe,PoincareBall.expmap0(norm_euc_1,c_exe),c_exe)
        attn_syn=HyperbolicAttention(self.dim,self.n_heads,name="attn_syn",dtype=self.dtype,param_dtype=self.param_dtype)(syn_informed,freqs_cis,c_syn)
        attn_sem=HyperbolicAttention(self.dim,self.n_heads,name="attn_sem",dtype=self.dtype,param_dtype=self.param_dtype)(sem_informed,freqs_cis,c_sem)
        attn_exe=HyperbolicAttention(self.dim,self.n_heads,name="attn_exe",dtype=self.dtype,param_dtype=self.param_dtype)(exe_informed,freqs_cis,c_exe)
        x_euc_post_attn=x_euc+(attn_syn+attn_sem+attn_exe);norm_euc_2=nn.LayerNorm(dtype=jnp.float32,name="norm_euc_2")(x_euc_post_attn).astype(self.dtype)
        ffn_syn=HyperbolicFFN(self.dim,name="ffn_syn",dtype=self.dtype,param_dtype=self.param_dtype)(x_syn,c_syn);ffn_sem=HyperbolicFFN(self.dim,name="ffn_sem",dtype=self.dtype,param_dtype=self.param_dtype)(x_sem,c_sem)
        ffn_exe=HyperbolicFFN(self.dim,name="ffn_exe",dtype=self.dtype,param_dtype=self.param_dtype)(x_exe,c_exe)
        comm_sem_to_syn=nn.Dense(self.dim,name="comm_sem_to_syn",dtype=self.dtype,param_dtype=self.param_dtype)(ffn_sem);comm_exe_to_sem=nn.Dense(self.dim,name="comm_exe_to_sem",dtype=self.dtype,param_dtype=self.param_dtype)(ffn_exe)
        x_syn_final=PoincareBall.expmap0(PoincareBall.logmap0(x_syn,c_syn)+ffn_syn+comm_sem_to_syn,c_syn);x_sem_final=PoincareBall.expmap0(PoincareBall.logmap0(x_sem,c_sem)+ffn_sem+comm_exe_to_sem,c_sem)
        x_exe_final=PoincareBall.expmap0(PoincareBall.logmap0(x_exe,c_exe)+ffn_exe,c_exe);gate=nn.Dense(self.dim,name="chassis_gate",kernel_init=nn.initializers.zeros,dtype=self.dtype,param_dtype=self.param_dtype)(norm_euc_2)
        x_euc_final=x_euc_post_attn+(ffn_syn+ffn_sem+ffn_exe)*nn.sigmoid(gate)
        return{'euc':x_euc_final,'syn':x_syn_final,'sem':x_sem_final,'exe':x_exe_final}

# --- NEW: JAX-based HAKMEM Q-Controller ---
@dataclasses.dataclass
class HAKMEMState:
    q_table: jnp.ndarray
    key: jax.Array
    current_lr: float
    last_loss: float
    last_action_idx: int
    last_state_idx: int

class HAKMEMQController:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lr_change_factors = jnp.array(config['lr_change_factors'])
        print(f"--- HAKMEM Controller Initialized. Actions: {len(self.lr_change_factors)} ---")

    def init_state(self, key: jax.Array, initial_lr: float) -> HAKMEMState:
        q_key, action_key = jax.random.split(key)
        return HAKMEMState(
            q_table=jnp.zeros((self.config['q_table_size'], len(self.lr_change_factors))),
            key=action_key, current_lr=initial_lr, last_loss=0.0,
            last_action_idx=0, last_state_idx=0
        )
    
    @staticmethod
    @partial(jax.jit, static_argnames=['q_table_size', 'num_actions', 'exploration_rate', 'lr_min', 'lr_max', 'loss_min', 'loss_max', 'learning_rate_q', 'discount_factor_q'])
    def update(state: HAKMEMState, loss: float, lr_change_factors: jnp.ndarray,
               q_table_size: int, num_actions: int, exploration_rate: float,
               lr_min: float, lr_max: float, loss_min: float, loss_max: float,
               learning_rate_q: float, discount_factor_q: float) -> HAKMEMState:
        
        # 1. Calculate Reward from previous step
        reward = state.last_loss - loss  # Simple reward: negative change in loss

        # 2. Update Q-Table for the last action
        q_table = state.q_table
        if state.last_state_idx >= 0: # Skip update on first step
            current_q = q_table[state.last_state_idx, state.last_action_idx]
            
            # Discretize current loss to get next state
            bin_size = (loss_max - loss_min) / q_table_size
            next_state_idx = jnp.clip(jnp.floor((loss - loss_min) / bin_size).astype(jnp.int32), 0, q_table_size - 1)
            
            max_next_q = jnp.max(q_table[next_state_idx])
            new_q = current_q + learning_rate_q * (reward + discount_factor_q * max_next_q - current_q)
            q_table = q_table.at[state.last_state_idx, state.last_action_idx].set(new_q)
        
        # 3. Choose next action
        key, subkey = jax.random.split(state.key)
        
        def explore(q_table_ignored, key):
            return jax.random.randint(key, shape=(), minval=0, maxval=num_actions)

        def exploit(q_table_to_use, key_ignored):
            # Discretize current loss to get current state for action choice
            bin_size = (loss_max - loss_min) / q_table_size
            current_state_idx = jnp.clip(jnp.floor((loss - loss_min) / bin_size).astype(jnp.int32), 0, q_table_size - 1)
            return jnp.argmax(q_table_to_use[current_state_idx])

        action_idx = jax.lax.cond(
            jax.random.uniform(key) < exploration_rate,
            explore, exploit, q_table, subkey
        )
        
        # 4. Calculate new LR
        new_lr = jnp.clip(state.current_lr * lr_change_factors[action_idx], lr_min, lr_max)
        
        # 5. Return updated state
        return HAKMEMState(q_table=q_table, key=key, current_lr=new_lr, last_loss=loss,
                           last_action_idx=action_idx, last_state_idx=next_state_idx)

# --- Adversarial Model Definitions ---
@dataclasses.dataclass
class WubuMindCore(nn.Module):
    d_model:int;n_heads:int;n_layers:int;max_len:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @staticmethod
    def precompute_freqs_cis(dim,end,theta=10000.0): freqs=1.0/(theta**(jnp.arange(0,dim,2,dtype=jnp.float32)/dim)); return jnp.exp(1j*jnp.outer(jnp.arange(end),freqs))
    @nn.compact
    def __call__(self,base_euc):
        B,N,_=base_euc.shape;h_dim=self.d_model//self.n_heads
        c_syn,c_sem,c_exe=nn.softplus(self.param('c_syntactic',nn.initializers.constant(5.0),(1,))),nn.softplus(self.param('c_semantic',nn.initializers.constant(1.0),(1,))),nn.softplus(self.param('c_executive',nn.initializers.constant(0.1),(1,)))
        x_syn=PoincareBall.expmap0(nn.Dense(self.d_model,name="proj_syntactic",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc),c_syn)
        x_sem=PoincareBall.expmap0(nn.Dense(self.d_model,name="proj_semantic",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc),c_sem)
        x_exe=PoincareBall.expmap0(nn.Dense(self.d_model,name="proj_executive",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc),c_exe)
        x_euc_chassis=nn.LayerNorm(dtype=self.dtype,name="proj_chassis_norm")(base_euc);states={'euc':x_euc_chassis,'syn':x_syn,'sem':x_sem,'exe':x_exe}
        curvatures={'syn':c_syn,'sem':c_sem,'exe':c_exe};freqs_cis=self.precompute_freqs_cis(h_dim,self.max_len)[:N]
        for i in range(self.n_layers): states=GalacticBlock(dim=self.d_model,n_heads=self.n_heads,name=f"galaxy_{i}",dtype=self.dtype,param_dtype=self.param_dtype)(states,curvatures,freqs_cis)
        return states,curvatures
@dataclasses.dataclass
class Generator(nn.Module):
    vocab_size:int;d_model:int;n_heads:int;n_layers:int;max_len:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @nn.compact
    def __call__(self,indices):
        token_embed=nn.Embed(self.vocab_size,self.d_model,dtype=self.dtype,param_dtype=self.param_dtype,name="token_embed");base_euc=token_embed(indices)
        core=WubuMindCore(self.d_model,self.n_heads,self.n_layers,self.max_len,self.dtype,self.param_dtype,name="core");states,curvatures=core(base_euc)
        final_norm_euc=nn.LayerNorm(dtype=jnp.float32,name="final_norm")(states['euc']);output_proj=nn.Dense(self.vocab_size,dtype=jnp.float32,name="output_proj")
        final_logits=output_proj(final_norm_euc)
        tangent_syn=PoincareBall.logmap0(states['syn'],curvatures['syn']);tangent_sem=PoincareBall.logmap0(states['sem'],curvatures['sem']);tangent_exe=PoincareBall.logmap0(states['exe'],curvatures['exe'])
        return {'final_logits':final_logits,'drive_logits':{'syn':output_proj(tangent_syn),'sem':output_proj(tangent_sem),'exe':output_proj(tangent_exe)}}
@dataclasses.dataclass
class Discriminator(nn.Module):
    vocab_size:int;d_model:int;n_heads:int;n_layers:int;max_len:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @nn.compact
    def __call__(self,indices):
        token_embed=nn.Embed(self.vocab_size,self.d_model,dtype=self.dtype,param_dtype=self.param_dtype,name="token_embed");base_euc=token_embed(indices)
        core=WubuMindCore(self.d_model,self.n_heads,self.n_layers,self.max_len,self.dtype,self.param_dtype,name="core");states,_=core(base_euc)
        pooled_euc=jnp.mean(states['euc'],axis=1);real_fake_logit=nn.Dense(1,dtype=jnp.float32,name="disc_head")(pooled_euc)
        return real_fake_logit

# --- Data Prep & Checkpointing ---
def prepare_training_data(text_corpus,tokenizer,config):
    indices=np.array(tokenizer.encode(text_corpus),dtype=np.int32);cl,bs=config['context_length'],config['batch_size']
    num_samples=len(indices)-cl-1;strides=indices.strides[0]
    all_indices=np.lib.stride_tricks.as_strided(indices,shape=(num_samples,cl),strides=(strides,strides));all_targets=np.lib.stride_tricks.as_strided(indices[1:],shape=(num_samples,cl),strides=(strides,strides))
    num_batches=num_samples//bs
    if num_to_trim:=num_samples%bs:all_indices,all_targets=[arr[:-num_to_trim] for arr in(all_indices,all_targets)]
    all_indices_b,all_targets_b=[arr.reshape(num_batches,bs,cl) for arr in(all_indices,all_targets)]
    print(f"--- Data prep complete: {num_batches} batches. ---");return (all_indices_b,all_targets_b),num_batches
def save_checkpoint(g_state,d_state,g_q_state,d_q_state,basename):
    state_dict={'g':serialization.to_state_dict(g_state),'d':serialization.to_state_dict(d_state),'g_q':dataclasses.asdict(g_q_state),'d_q':dataclasses.asdict(d_q_state)}
    with open(f"{basename}.pkl",'wb') as f:pickle.dump(jax.device_get(state_dict),f)
    print(f"\n--- Checkpoint saved. Step: {g_state.step} ---")
def load_checkpoint(g_state,d_state,g_q_state,d_q_state,basename):
    filename=f"{basename}.pkl"
    if not os.path.exists(filename):print("--- No checkpoint found. ---");return g_state,d_state,g_q_state,d_q_state
    with open(filename,'rb') as f:saved_state_dict=pickle.load(f)
    print(f"--- Checkpoint found. Restoring... ---")
    g_state=serialization.from_state_dict(g_state,saved_state_dict['g']);d_state=serialization.from_state_dict(d_state,saved_state_dict['d'])
    g_q_state=g_q_state.__class__(**saved_state_dict['g_q']);d_q_state=d_q_state.__class__(**saved_state_dict['d_q'])
    print(f"--- States restored. Resuming from step: {g_state.step} ---");return g_state,d_state,g_q_state,d_q_state
def save_config(config,basename):
    with open(f"{basename}.json",'w') as f:json.dump(config,f,indent=4)
    print(f"--- Model config saved to {basename}.json ---")

# --- Autonomous Adversarial Training Manager ---
class AdversarialTrainingManager:
    def __init__(self,generator,discriminator,config,data,basename:str):
        self.g,self.d=generator,discriminator;self.config=config;self.data=data;self.basename=basename
        self.g_state,self.d_state,self.g_q_state,self.d_q_state,self.should_shutdown=None,None,None,None,False
        signal.signal(signal.SIGINT,self._handle_sigint)
    def _handle_sigint(self,s,f):
        if not self.should_shutdown:print("\n--- SIGINT received. Saving state... ---");self.should_shutdown=True
    @staticmethod
    @partial(jax.jit, static_argnames=['g_apply_fn', 'd_apply_fn', 'recon_weight', 'adv_weight'])
    def train_step(g_state,d_state,batch,g_apply_fn,d_apply_fn,recon_weight,adv_weight):
        real_indices,real_targets=batch
        def d_loss_fn(d_params):
            g_output=g_apply_fn({'params':g_state.params},real_indices);fake_indices=jnp.argmax(g_output['final_logits'],axis=-1)
            real_logits=d_apply_fn({'params':d_params},real_indices);fake_logits=d_apply_fn({'params':d_params},fake_indices)
            real_loss=optax.sigmoid_binary_cross_entropy(real_logits,jnp.ones_like(real_logits)).mean()
            fake_loss=optax.sigmoid_binary_cross_entropy(fake_logits,jnp.zeros_like(fake_logits)).mean()
            return(real_loss+fake_loss)/2, (real_loss,fake_loss)
        (d_loss,(d_real,d_fake)),d_grads=jax.value_and_grad(d_loss_fn,has_aux=True)(d_state.params)
        d_state=d_state.apply_gradients(grads=d_grads)
        def g_loss_fn(g_params):
            g_output=g_apply_fn({'params':g_params},real_indices);g_logits=g_output['final_logits']
            recon_loss=optax.softmax_cross_entropy_with_integer_labels(g_logits,real_targets).mean()
            fake_indices=jnp.argmax(g_logits,axis=-1);fake_d_logits=d_apply_fn({'params':d_state.params},fake_indices)
            adv_loss=optax.sigmoid_binary_cross_entropy(fake_d_logits,jnp.ones_like(fake_d_logits)).mean()
            return recon_weight*recon_loss+adv_weight*adv_loss, (recon_loss,adv_loss)
        (g_loss,(recon_loss,adv_loss)),g_grads=jax.value_and_grad(g_loss_fn,has_aux=True)(g_state.params)
        g_state=g_state.apply_gradients(grads=g_grads)
        metrics={'g_loss':g_loss,'d_loss':d_loss,'recon':recon_loss,'adv':adv_loss,'d_real':d_real,'d_fake':d_fake}
        return g_state,d_state,metrics
    def run(self):
        (i_all,t_all),num_batches=self.data;key=jax.random.PRNGKey(42);g_q_key,d_q_key,key=jax.random.split(key,3)
        dummy_batch=jax.device_put(i_all[0][:1]);g_params=self.g.init(jax.random.split(g_q_key)[0],dummy_batch)['params']
        print(f'--- Generator Initialized: {sum(x.size for x in jax.tree_util.tree_leaves(g_params)):,} params. ---')
        d_params=self.d.init(jax.random.split(d_q_key)[0],dummy_batch)['params']
        print(f'--- Discriminator Initialized: {sum(x.size for x in jax.tree_util.tree_leaves(d_params)):,} params. ---')

        # HAKMEM Controllers
        q_config = self.config['q_controller']
        g_q_ctrl = HAKMEMQController(q_config); self.g_q_state = g_q_ctrl.init_state(g_q_key, self.config['g_learning_rate'])
        d_q_ctrl = HAKMEMQController(q_config); self.d_q_state = d_q_ctrl.init_state(d_q_key, self.config['d_learning_rate'])
        
        # Optimizers now use LR from HAKMEM state
        g_tx=optax.adamw(learning_rate=self.g_q_state.current_lr,weight_decay=0.01); self.g_state=train_state.TrainState.create(apply_fn=self.g.apply,params=g_params,tx=g_tx)
        d_tx=optax.adamw(learning_rate=self.d_q_state.current_lr,weight_decay=0.01); self.d_state=train_state.TrainState.create(apply_fn=self.d.apply,params=d_params,tx=d_tx)
        
        save_config(self.config,self.basename)
        self.g_state,self.d_state,self.g_q_state,self.d_q_state = load_checkpoint(self.g_state,self.d_state,self.g_q_state,self.d_q_state,self.basename)
        start_step=self.g_state.step
        total_steps=self.config['epochs']*num_batches
        if start_step>=total_steps: print(f"--- Training already completed. ---"); return
        
        jit_train_step=partial(self.train_step,g_apply_fn=self.g.apply,d_apply_fn=self.d.apply,recon_weight=self.config['recon_weight'],adv_weight=self.config['adv_weight'])
        
        with tqdm(total=total_steps,initial=start_step,desc="GAN Training") as pbar:
            for step in range(start_step,total_steps):
                pbar.set_description(f"Epoch {(step//num_batches)+1}/{self.config['epochs']}")
                batch_host=(i_all[step%num_batches],t_all[step%num_batches]);batch_device=jax.device_put(batch_host)
                
                # Update LRs in optimizers before the step
                self.g_state.tx.learning_rate=self.g_q_state.current_lr; self.d_state.tx.learning_rate=self.d_q_state.current_lr
                
                self.g_state,self.d_state,metrics=jit_train_step(self.g_state,self.d_state,batch_device)
                
                # Update HAKMEM controllers with the new losses
                self.g_q_state = g_q_ctrl.update(self.g_q_state, metrics['g_loss'], g_q_ctrl.lr_change_factors, **q_config)
                self.d_q_state = d_q_ctrl.update(self.d_q_state, metrics['d_loss'], d_q_ctrl.lr_change_factors, **q_config)

                pbar.set_postfix(G_loss=f"{metrics['g_loss']:.2f}",D_loss=f"{metrics['d_loss']:.2f}",G_lr=f"{self.g_q_state.current_lr:.1e}",D_lr=f"{self.d_q_state.current_lr:.1e}")
                pbar.update(1)

                if self.should_shutdown or(step+1)%self.config['save_every']==0:
                    if self.should_shutdown:print("\n--- Interrupt honored. ---")
                    save_checkpoint(self.g_state,self.d_state,self.g_q_state,self.d_q_state,self.basename)
                    if self.should_shutdown:return

        print("\n--- Adversarial training complete. ---");save_checkpoint(self.g_state,self.d_state,self.g_q_state,self.d_q_state,self.basename)

# --- Main & Inference ---
def training_main(basename):
    MODEL_CONFIG = {'d_model': 384, 'n_heads': 6, 'n_layers': 6, 'max_len': 256} 
    TRAINING_CONFIG = {'epochs': 100, 'batch_size': 1, 'context_length': 256,
                     'g_learning_rate': 2e-4, 'd_learning_rate': 5e-5,
                     'recon_weight': 1.0, 'adv_weight': 0.1, 'save_every': 1000}
    Q_CONTROLLER_CONFIG = {"q_table_size": 10, "num_lr_actions": 5, "lr_change_factors": [0.5, 0.9, 1.0, 1.1, 1.5],
                         "learning_rate_q": 0.1, "discount_factor_q": 0.9, "exploration_rate_q": 0.1,
                         "lr_min": 1e-6, "lr_max": 1e-3, "loss_min": 0.0, "loss_max": 5.0}
    TOKENIZER_CONFIG = {'vocab_size': 32000, 'tokenizer_path': f"{basename}_bpe.json"}

    print(f"--- WubuMind Galactic Core Foundry v1 ---")
    print(f"--- Device: {jax.devices()[0].platform.upper()} ---")
    
    corpora = [getattr(CORPUS, n) for n in dir(CORPUS) if not n.startswith('_') and n.isupper()]
    if not corpora: print("[FATAL] No CORPUS vars."), sys.exit(1)
    
    corpus_text = distill_text_from_corpus(corpora)
    print(f"--- CORPUS Chars: {len(corpus_text):,} ---")
    
    tokenizer = WubuTokenizer(TOKENIZER_CONFIG['tokenizer_path'])
    if not tokenizer.tokenizer:
        tokenizer.train(corpus_text, TOKENIZER_CONFIG['vocab_size'])
        print("\n--- Tokenizer trained. Run again. ---")
        sys.exit(0)
    
    # --- THE FIX IS HERE ---
    # 1. Create a config specifically for the model architecture.
    arch_config = {**MODEL_CONFIG, 'vocab_size': tokenizer.get_vocab_size()}
    
    # 2. The full_config is for the training manager.
    full_config = {**arch_config, **TRAINING_CONFIG, 'q_controller': Q_CONTROLLER_CONFIG}
    
    data_bundle = prepare_training_data(corpus_text, tokenizer, full_config)
    
    # 3. Initialize the models using ONLY the architecture config.
    generator = Generator(**arch_config)
    discriminator = Discriminator(**arch_config)
    
    # 4. The training manager gets the full config.
    AdversarialTrainingManager(generator, discriminator, full_config, data_bundle, basename).run()
    
@partial(jax.jit,static_argnames=['model_apply_fn'])
def predict_step_fn(model_apply_fn,params,indices,key,temp,top_p,use_guidance,deviation_threshold):
    model_outputs=model_apply_fn({'params':params},indices);final_logits=model_outputs['final_logits'];drive_logits=model_outputs['drive_logits']
    def kl_divergence(logits_p,logits_q):log_p=jax.nn.log_softmax(logits_p,axis=-1)[:,-1,:];log_q=jax.nn.log_softmax(logits_q,axis=-1)[:,-1,:];return jnp.sum(jnp.exp(log_p)*(log_p-log_q),axis=-1)
    def power_modulator(final_logits,drive_logits):
        dev_sem_syn=kl_divergence(drive_logits['sem'],drive_logits['syn']);dev_exe_sem=kl_divergence(drive_logits['exe'],drive_logits['sem']);deviation=jnp.maximum(dev_sem_syn,dev_exe_sem)
        alpha=jnp.clip(1.0-(deviation/deviation_threshold),0.0,1.0);corrective_logits=drive_logits['exe']+drive_logits['syn']
        p_standard=jax.nn.softmax(final_logits,axis=-1);p_corrective=jax.nn.softmax(corrective_logits,axis=-1)
        p_mixed=(alpha[:,None,None]*p_standard)+((1.0-alpha[:,None,None])*p_corrective)
        return jnp.log(p_mixed.clip(1e-9))
    effective_logits=jax.lax.cond(use_guidance,lambda:power_modulator(final_logits,drive_logits),lambda:final_logits)
    initial_scaled=effective_logits[:,-1,:]/jnp.maximum(temp,1e-6)
    def apply_top_p(logits):
        sorted_indices=jnp.argsort(logits,axis=-1)[...,::-1];sorted_logits=jnp.take_along_axis(logits,sorted_indices,axis=-1)
        cum_probs=jnp.cumsum(nn.softmax(sorted_logits,axis=-1),axis=-1);sorted_to_remove=cum_probs>top_p
        sorted_to_remove=jnp.concatenate([jnp.zeros_like(sorted_to_remove[...,:1]),sorted_to_remove[...,:-1]],axis=-1)
        to_remove=jnp.zeros_like(sorted_to_remove).at[...,sorted_indices].set(sorted_to_remove);return jnp.where(to_remove,-jnp.inf,logits)
    final_scaled=jax.lax.cond(top_p<1.0,apply_top_p,lambda x:x,initial_scaled)
    return jax.random.categorical(key,final_scaled,axis=-1)

class WubuOracle:
    def __init__(self,model_basename):
        print("--- Oracle Awakens (Generator Mode) ---");self.basename=model_basename
        with open(f"{self.basename}.json",'r') as f:self.config=json.load(f)
        self.tokenizer=WubuTokenizer(f"{self.basename}_bpe.json");
        if not self.tokenizer.tokenizer:raise FileNotFoundError(f"Tokenizer not found")
        model_fields=[f.name for f in dataclasses.fields(Generator)];arch_config={k:v for k,v in self.config.items() if k in model_fields}
        self.model=Generator(**arch_config);print("--- Assimilating knowledge from checkpoint... ---")
        try:
            with open(f"{self.basename}.pkl",'rb') as f:saved_state_dict=pickle.load(f)
        except FileNotFoundError:print(f"[ERROR] Checkpoint not found."),sys.exit(1)
        if'generator'not in saved_state_dict or'params'not in saved_state_dict['generator']:raise ValueError("Checkpoint invalid.")
        g_saved_state=saved_state_dict['generator'];dummy_params=self.model.init(jax.random.PRNGKey(0),jnp.ones((1,1),dtype=jnp.int32))['params']
        self.params=serialization.from_state_dict(dummy_params,g_saved_state['params']);step=g_saved_state.get('step','unknown')
        print(f"--- Oracle assimilated Generator knowledge from step {step}. ---");self.use_guidance=False;self.deviation_threshold=5.0;self.jit_compiled=False
    def generate(self,prompt,max_new=500,temp=0.7,top_p=0.95):
        if not self.jit_compiled:print("--- JIT compiling Generator... ---",flush=True)
        key=jax.random.PRNGKey(int(time.time()));indices=self.tokenizer.encode(prompt);decoded_text=prompt
        sys.stdout.write(f"\n\033[1;32m{prompt}\033[0m");sys.stdout.flush()
        for _ in range(max_new):
            current_indices=indices[-self.config['context_length']:];pad_len=self.config['context_length']-len(current_indices)
            if pad_len>0:current_indices=[self.tokenizer.pad_id]*pad_len+current_indices
            i_batch=jax.device_put(np.array(current_indices,dtype=np.int32)[None,:]);key,subkey=jax.random.split(key)
            next_idx_array=predict_step_fn(self.model.apply,self.params,i_batch,subkey,temp,top_p,self.use_guidance,self.deviation_threshold)
            if not self.jit_compiled:next_idx_array.block_until_ready();self.jit_compiled=True
            new_idx=int(next_idx_array.item())
            if new_idx==self.tokenizer.pad_id:break
            indices.append(new_idx);full_decoded_text=self.tokenizer.decode(indices);new_chunk=full_decoded_text[len(decoded_text):]
            sys.stdout.write(new_chunk);sys.stdout.flush();decoded_text=full_decoded_text
        print()

def interactive_mode(model_basename):
    try:oracle=WubuOracle(model_basename)
    except Exception as e:traceback.print_exc();return
    print("\n--- Oracle Command Console (WIGS v2) ---");print("  Commands: /guidance on|off, /threshold N, /exit")
    while True:
        try:
            prompt=input("\nYour Prompt> ")
            if prompt.lower()in["exit","quit"]:break
            if prompt.lower().startswith('/guidance on'):oracle.use_guidance=True;print("\n\033[1;33m--- Guidance Modulator ENGAGED ---\033[0m");continue
            if prompt.lower().startswith('/guidance off'):oracle.use_guidance=False;print("\n\033[1;33m--- Guidance Modulator DISENGAGED ---\033[0m");continue
            if prompt.lower().startswith('/threshold'):
                try:oracle.deviation_threshold=float(prompt.split()[1]);print(f"\n\033[1;33m--- Modulation threshold set to: {oracle.deviation_threshold:.2f} ---\033[0m")
                except:print("\n\033[1;31mInvalid. Usage: /threshold 5.0\033[0m")
                continue
            oracle.generate(prompt)
        except KeyboardInterrupt:print("\n-- Exiting. --");break
        except Exception as e:print(f"\nAn error occurred: {e}")

def main():
    BASENAME="wubumind_adversarial_core_v3"
    if len(sys.argv)<2 or sys.argv[1] not in["train","infer"]:print(f"Usage: python {sys.argv[0]} [train|infer]");sys.exit(1)
    if sys.argv[1]=="train":training_main(BASENAME)
    elif sys.argv[1]=="infer":interactive_mode(BASENAME)

if __name__=="__main__":main()
