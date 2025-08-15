# WubuMind Adversarial Synthesizer v1.0
# An implementation of the Galactic Core architecture for generative audio.
# This model learns from raw audio streams in an unsupervised, adversarial manner,
# leveraging multiple, interacting geometric spaces to generate coherent, complex,
# and non-repetitive audio.

import os

# --- Environment Setup for JAX/Flax on any hardware ---
# Set robust, universally compatible environment variables for memory management.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

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
from typing import Any, Sequence, Dict, Tuple, Optional
import sys
import dataclasses
import signal
import traceback
import argparse
import shutil
import random
from collections import deque
from pathlib import Path

# --- Core JAX Configuration ---
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
jax.config.update('jax_threefry_partitionable', True)

from jax import tree, profiler, tree_util, vmap
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, Mesh

# --- Audio Processing Dependencies ---
# These are essential for handling the audio data stream.
try:
    import torch
    import torchaudio
    from encodec import EncodecModel
    import soundfile as sf
except ImportError:
    print("[FATAL] Audio dependencies not found. Please install them:")
    print("`pip install torch torchaudio soundfile encodec`")
    sys.exit(1)

# --- The Audio Codec: The "Tokenizer" for Sound ---
# This class replaces the text tokenizer. It converts continuous audio waveforms
# into discrete integer tokens that the model can process.

class WubuAudioCodec:
    """A wrapper for EnCodec, serving as the audio tokenizer."""
    def __init__(self, sample_rate=24000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"--- Initializing WubuAudioCodec on device: {device} ---")
        self.model = EncodecModel.encodec_model_24khz().to(device)
        self.model.set_target_bandwidth(6.0) # Quality setting
        self.device = device
        self.sample_rate = self.model.sample_rate
        if sample_rate != self.sample_rate:
            print(f"[WARNING] Requested sample rate {sample_rate} differs from model's native {self.model.sample_rate}.")
            self.resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.model.sample_rate)
        else:
            self.resampler = None
        
        # The 'vocabulary' of the audio model is the number of entries in the VQ codebook.
        self.vocab_size = self.model.quantizer.bins
        self.pad_id = self.vocab_size # Use the next available integer as the padding token.

    @torch.no_grad()
    def encode(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """Encodes a waveform tensor into a sequence of integer tokens."""
        if self.resampler:
            audio_tensor = self.resampler(audio_tensor.cpu()).to(self.device)
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.dim() == 2:
             audio_tensor = audio_tensor.unsqueeze(1) # Add channel dimension
        
        # Encodec expects [Batch, Channels, Samples]
        encoded_frames = self.model.encode(audio_tensor)
        # We use only the first codebook for simplicity: [Batch, Num_Quantizers, Tokens] -> [Batch, Tokens]
        codes = encoded_frames[0][0][:, 0, :].cpu().numpy()
        return codes

    @torch.no_grad()
    def decode(self, codes_array: np.ndarray) -> torch.Tensor:
        """Decodes a sequence of integer tokens back into a waveform tensor."""
        # Add back the quantizer dimension and convert to torch tensor
        codes_tensor = torch.from_numpy(codes_array).long().unsqueeze(1).to(self.device)
        # Create the full frame structure Encodec expects
        encoded_frames = [(codes_tensor, None)]
        return self.model.decode(encoded_frames).squeeze(0).cpu()

# --- Hyperbolic Geometry Core ---
# This remains the mathematical heart of the model.

class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def project(x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True).clip(PoincareBall.EPS)
        max_norm = 1.0 - PoincareBall.EPS
        cond = norm >= 1.0
        return jnp.where(cond, x / norm * max_norm, x)
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
        safe_y_norm = y_norm.clip(PoincareBall.EPS)
        direction = y / safe_y_norm
        magnitude = jnp.arctanh(y_norm.clip(max=1.0 - PoincareBall.EPS)) / sqrt_c
        return jnp.where(y_norm < PoincareBall.EPS, jnp.zeros_like(y), magnitude * direction)
    @staticmethod
    def expmap0(v, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        safe_v_norm = v_norm.clip(PoincareBall.EPS)
        direction = v / safe_v_norm
        magnitude = jnp.tanh(sqrt_c * safe_v_norm) / sqrt_c
        return PoincareBall.project(jnp.where(v_norm < PoincareBall.EPS, jnp.zeros_like(v), magnitude * direction))
    @staticmethod
    def dist(x, y, c):
        sqrt_c = jnp.sqrt(c).clip(PoincareBall.EPS)
        add_xy = PoincareBall.mobius_add(-x, y, c)
        add_norm = jnp.linalg.norm(add_xy, axis=-1)
        arg = jnp.clip(sqrt_c * add_norm, max=1.0 - PoincareBall.EPS)
        return 2. * jnp.arctanh(arg) / sqrt_c

# --- Model Architecture ---
# The same powerful geometric architecture, now applied to audio tokens.
# CRITICAL: Hyperbolic operations are performed in float32 for precision,
# while the Euclidean chassis can use bfloat16 for speed.

class HyperbolicAttention(nn.Module):
    dim:int;n_heads:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @staticmethod
    def apply_rotary_emb(x,freqs_cis):
        x_f32=x.astype(jnp.float32);x_r,x_i=jnp.split(x_f32,2,-1);x_c=jax.lax.complex(x_r,x_i)
        freqs_cis=freqs_cis.reshape(1,1,freqs_cis.shape[0],freqs_cis.shape[1]);x_rotated=x_c*freqs_cis
        return jnp.concatenate([x_rotated.real,x_rotated.imag],-1)
    @nn.compact
    def __call__(self,x_hyp,freqs_cis,c_sphere):
        B,N,_=x_hyp.shape;h_dim=self.dim//self.n_heads
        # --- PRECISION CONTROL: Perform hyperbolic math in float32 ---
        x_hyp_f32=x_hyp.astype(jnp.float32);c_sphere_f32=c_sphere.astype(jnp.float32)
        qkv_proj=nn.Dense(self.dim*3,name="qkv_proj",dtype=self.dtype,param_dtype=self.param_dtype);out_proj=nn.Dense(self.dim,name="out_proj",dtype=self.dtype,param_dtype=self.param_dtype)
        c_per_head_logits=self.param('c_per_head_logits',nn.initializers.zeros,(self.n_heads,),self.param_dtype);geo_scale=self.param('geo_scale',nn.initializers.ones,(1,self.n_heads,1,1),self.param_dtype)
        x_tangent=PoincareBall.logmap0(x_hyp_f32, c_sphere_f32)
        qkv=qkv_proj(x_tangent.astype(self.dtype)).reshape(B,N,3,self.n_heads,h_dim).transpose((2,0,3,1,4));q,k,v_euc=qkv[0],qkv[1],qkv[2]
        q_rot,k_rot=self.apply_rotary_emb(q,freqs_cis),self.apply_rotary_emb(k,freqs_cis);c_per_head=nn.softplus(c_per_head_logits.astype(jnp.float32))
        q_hyp=PoincareBall.expmap0(q_rot,c_per_head[:,None,None]);k_hyp=PoincareBall.expmap0(k_rot,c_per_head[:,None,None])
        def compute_dist_matrix(q_seq, k_seq, c_val): return vmap(lambda q_vec: vmap(lambda k_vec: PoincareBall.dist(q_vec, k_vec, c_val))(k_seq))(q_seq)
        dist=vmap(lambda q_b, k_b: vmap(compute_dist_matrix, in_axes=(0, 0, 0))(q_b, k_b, c_per_head))(q_hyp, k_hyp)
        mask=nn.make_causal_mask(jnp.ones((B,N),dtype=bool));attn_scores=jnp.where(mask,-geo_scale.astype(jnp.float32)*dist,-jnp.inf)
        attn_weights=nn.softmax(attn_scores,axis=-1);attn_out_euc=(attn_weights.astype(self.dtype)@v_euc).transpose((0,2,1,3)).reshape(B,N,self.dim)
        return out_proj(attn_out_euc)

class HyperbolicFFN(nn.Module):
    dim:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @nn.compact
    def __call__(self,x_hyp,c_sphere):
        # --- PRECISION CONTROL: Logmap in float32, FFN in bfloat16 ---
        x_hyp_f32=x_hyp.astype(jnp.float32);c_sphere_f32=c_sphere.astype(jnp.float32);x_tangent=PoincareBall.logmap0(x_hyp_f32,c_sphere_f32)
        ffn_output=nn.Sequential([nn.Dense(self.dim*4,dtype=self.dtype,param_dtype=self.param_dtype),nn.gelu,nn.Dense(self.dim,dtype=self.dtype,param_dtype=self.param_dtype)])(x_tangent.astype(self.dtype))
        return ffn_output

class GalacticBlock(nn.Module):
    dim:int;n_heads:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    @remat
    @nn.compact
    def __call__(self,states,curvatures,freqs_cis): 
        x_euc=states['euc'] # bfloat16
        # --- Unpack states, ensuring hyperbolic ones are ready for float32 ops ---
        x_syn, x_sem, x_exe = states['syn'], states['sem'], states['exe']
        c_syn, c_sem, c_exe = curvatures['syn'], curvatures['sem'], curvatures['exe']
        
        # --- Euclidean -> Hyperbolic Projection (float32 math) ---
        norm_euc_1=nn.LayerNorm(dtype=jnp.float32,name="norm_euc_1")(x_euc).astype(self.dtype)
        norm_euc_1_f32=norm_euc_1.astype(jnp.float32)
        exp_map_euc_syn=PoincareBall.expmap0(norm_euc_1_f32, c_syn.astype(jnp.float32))
        exp_map_euc_sem=PoincareBall.expmap0(norm_euc_1_f32, c_sem.astype(jnp.float32))
        exp_map_euc_exe=PoincareBall.expmap0(norm_euc_1_f32, c_exe.astype(jnp.float32))
        syn_informed=PoincareBall.mobius_add(x_syn.astype(jnp.float32),exp_map_euc_syn,c_syn.astype(jnp.float32))
        sem_informed=PoincareBall.mobius_add(x_sem.astype(jnp.float32),exp_map_euc_sem,c_sem.astype(jnp.float32))
        exe_informed=PoincareBall.mobius_add(x_exe.astype(jnp.float32),exp_map_euc_exe,c_exe.astype(jnp.float32))

        # --- Hyperbolic Attention (dtype passed internally) ---
        attn_syn=HyperbolicAttention(self.dim,self.n_heads,name="attn_syn",dtype=self.dtype,param_dtype=self.param_dtype)(syn_informed,freqs_cis,c_syn)
        attn_sem=HyperbolicAttention(self.dim,self.n_heads,name="attn_sem",dtype=self.dtype,param_dtype=self.param_dtype)(sem_informed,freqs_cis,c_sem)
        attn_exe=HyperbolicAttention(self.dim,self.n_heads,name="attn_exe",dtype=self.dtype,param_dtype=self.param_dtype)(exe_informed,freqs_cis,c_exe)
        
        # --- Update Euclidean Chassis (bfloat16) ---
        x_euc_post_attn=x_euc+(attn_syn+attn_sem+attn_exe)
        norm_euc_2=nn.LayerNorm(dtype=jnp.float32,name="norm_euc_2")(x_euc_post_attn).astype(self.dtype)

        # --- Hyperbolic FFNs (dtype passed internally) ---
        ffn_syn=HyperbolicFFN(self.dim,name="ffn_syn",dtype=self.dtype,param_dtype=self.param_dtype)(x_syn,c_syn)
        ffn_sem=HyperbolicFFN(self.dim,name="ffn_sem",dtype=self.dtype,param_dtype=self.param_dtype)(x_sem,c_sem)
        ffn_exe=HyperbolicFFN(self.dim,name="ffn_exe",dtype=self.dtype,param_dtype=self.param_dtype)(x_exe,c_exe)
        
        # --- Inter-space Communication (float32 tangent space updates) ---
        comm_sem_to_syn=nn.Dense(self.dim,name="comm_sem_to_syn",dtype=self.dtype,param_dtype=self.param_dtype)(ffn_sem)
        comm_exe_to_sem=nn.Dense(self.dim,name="comm_exe_to_sem",dtype=self.dtype,param_dtype=self.param_dtype)(ffn_exe)
        x_syn_f32, x_sem_f32, x_exe_f32 = x_syn.astype(jnp.float32), x_sem.astype(jnp.float32), x_exe.astype(jnp.float32)
        c_syn_f32, c_sem_f32, c_exe_f32 = c_syn.astype(jnp.float32), c_sem.astype(jnp.float32), c_exe.astype(jnp.float32)
        ffn_syn_f32, ffn_sem_f32, ffn_exe_f32 = ffn_syn.astype(jnp.float32), ffn_sem.astype(jnp.float32), ffn_exe.astype(jnp.float32)
        comm_sem_to_syn_f32=comm_sem_to_syn.astype(jnp.float32);comm_exe_to_sem_f32=comm_exe_to_sem.astype(jnp.float32)
        
        # --- Final Hyperbolic State Updates (float32) ---
        x_syn_final=PoincareBall.expmap0(PoincareBall.logmap0(x_syn_f32,c_syn_f32)+ffn_syn_f32+comm_sem_to_syn_f32,c_syn_f32)
        x_sem_final=PoincareBall.expmap0(PoincareBall.logmap0(x_sem_f32,c_sem_f32)+ffn_sem_f32+comm_exe_to_sem_f32,c_sem_f32)
        x_exe_final=PoincareBall.expmap0(PoincareBall.logmap0(x_exe_f32,c_exe_f32)+ffn_exe_f32,c_exe_f32)
        
        # --- Final Euclidean State Update (bfloat16) ---
        gate=nn.Dense(self.dim,name="chassis_gate",kernel_init=nn.initializers.zeros,dtype=self.dtype,param_dtype=self.param_dtype)(norm_euc_2)
        x_euc_final=x_euc_post_attn+(ffn_syn+ffn_sem+ffn_exe)*nn.sigmoid(gate)
        
        return {
            'euc': x_euc_final.astype(self.dtype),
            'syn': x_syn_final.astype(self.dtype), # Store as bfloat16 to save memory, cast back to float32 on next block entry
            'sem': x_sem_final.astype(self.dtype),
            'exe': x_exe_final.astype(self.dtype)
        }

@dataclasses.dataclass
class WubuMindCore(nn.Module):
    d_model:int;n_heads:int;n_layers:int;max_len:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    
    @staticmethod
    def precompute_freqs_cis(dim,end,theta=10000.0):
        freqs=1.0/(theta**(jnp.arange(0,dim,2,dtype=jnp.float32)/dim))
        return jnp.exp(1j*jnp.outer(jnp.arange(end),freqs))

    @nn.compact
    def __call__(self,base_euc):
        B,N,_=base_euc.shape;h_dim=self.d_model//self.n_heads

        # --- Learnable Curvatures for Audio Spaces ---
        # Syntactic (Timbre/Rhythm): High curvature, hierarchical
        # Semantic (Genre/Context): Medium curvature, relational
        # Executive (Flow/Narrative): Low curvature, sequential
        c_syn=nn.softplus(self.param('c_syntactic',nn.initializers.constant(5.0),(1,))).astype(jnp.float32)
        c_sem=nn.softplus(self.param('c_semantic',nn.initializers.constant(1.0),(1,))).astype(jnp.float32)
        c_exe=nn.softplus(self.param('c_executive',nn.initializers.constant(0.1),(1,))).astype(jnp.float32)
        
        proj_syn=nn.Dense(self.d_model,name="proj_syntactic",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc)
        proj_sem=nn.Dense(self.d_model,name="proj_semantic",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc)
        proj_exe=nn.Dense(self.d_model,name="proj_executive",dtype=self.dtype,param_dtype=self.param_dtype)(base_euc)
        
        # --- Project into Hyperbolic Spaces (float32) ---
        x_syn=PoincareBall.expmap0(proj_syn.astype(jnp.float32),c_syn)
        x_sem=PoincareBall.expmap0(proj_sem.astype(jnp.float32),c_sem)
        x_exe=PoincareBall.expmap0(proj_exe.astype(jnp.float32),c_exe)
        
        x_euc_chassis=nn.LayerNorm(dtype=self.dtype,name="proj_chassis_norm")(base_euc)
        
        states={
            'euc':x_euc_chassis.astype(self.dtype),
            'syn':x_syn.astype(self.dtype),
            'sem':x_sem.astype(self.dtype),
            'exe':x_exe.astype(self.dtype)
        }
        curvatures={'syn':c_syn,'sem':c_sem,'exe':c_exe}
        freqs_cis=self.precompute_freqs_cis(h_dim,self.max_len)[:N]
        
        for i in range(self.n_layers):
            states=GalacticBlock(dim=self.d_model,n_heads=self.n_heads,name=f"galaxy_{i}",dtype=self.dtype,param_dtype=self.param_dtype)(states,curvatures,freqs_cis)
        
        return states,curvatures

@dataclasses.dataclass
class Generator(nn.Module):
    vocab_size:int;d_model:int;n_heads:int;n_layers:int;max_len:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    
    @nn.compact
    def __call__(self,indices):
        token_embed_layer=nn.Embed(self.vocab_size + 1, self.d_model,dtype=self.dtype,param_dtype=self.param_dtype,name="token_embed")
        base_euc=token_embed_layer(indices)
        
        core=WubuMindCore(self.d_model,self.n_heads,self.n_layers,self.max_len,self.dtype,self.param_dtype,name="core")
        states,curvatures=core(base_euc)
        
        final_norm_euc=nn.LayerNorm(dtype=jnp.float32,name="final_norm")(states['euc'])
        # ### FIX: Output layer must also know about the +1 for the padding token ###
        output_proj=nn.Dense(self.vocab_size + 1, dtype=jnp.float32,name="output_proj")
        final_logits=output_proj(final_norm_euc)
        
        # --- Project from each space to get "drive" logits for guidance ---
        tangent_syn=PoincareBall.logmap0(states['syn'].astype(jnp.float32),curvatures['syn'].astype(jnp.float32))
        tangent_sem=PoincareBall.logmap0(states['sem'].astype(jnp.float32),curvatures['sem'].astype(jnp.float32))
        tangent_exe=PoincareBall.logmap0(states['exe'].astype(jnp.float32),curvatures['exe'].astype(jnp.float32))
        drive_logits={
            'syn':output_proj(tangent_syn),
            'sem':output_proj(tangent_sem),
            'exe':output_proj(tangent_exe)
        }
        
        return {
            'final_logits':final_logits,
            'embedding_matrix':token_embed_layer.embedding,
            'embeddings':base_euc,
            'drive_logits':drive_logits
        }

@dataclasses.dataclass
class Discriminator(nn.Module):
    d_model:int;n_heads:int;n_layers:int;max_len:int;dtype:Any=jnp.bfloat16;param_dtype:Any=jnp.float32
    
    @nn.compact
    def __call__(self,embeddings):
        core=WubuMindCore(self.d_model,self.n_heads,self.n_layers,self.max_len,self.dtype,self.param_dtype,name="core")
        states,_=core(embeddings)
        pooled_euc=jnp.mean(states['euc'].astype(jnp.float32),axis=1)
        real_fake_logit=nn.Dense(1,dtype=jnp.float32,name="disc_head")(pooled_euc)
        return real_fake_logit

# --- Data Preparation for Audio Streams ---
def prepare_audio_data(audio_dir: str, audio_codec: WubuAudioCodec, config: Dict[str, Any]):
    print("--- Preparing audio data... This may take a while. ---")
    audio_files = list(Path(audio_dir).glob('**/*.wav')) + list(Path(audio_dir).glob('**/*.mp3'))
    if not audio_files:
        print(f"[FATAL] No audio files found in {audio_dir}")
        sys.exit(1)
        
    all_codes = []
    chunk_len_samples = int(config['chunk_len_sec'] * config['sample_rate'])
    
    # ### FIX: Get the target device from the audio codec ###
    device = audio_codec.device
    
    for audio_file in tqdm(audio_files, desc="Processing Audio Files"):
        try:
            waveform, sr = torchaudio.load(audio_file)
            
            # ### FIX: Move waveform to the correct device immediately after loading ###
            waveform = waveform.to(device)
            
            if sr != config['sample_rate']:
                # The resampler should also be on the correct device. Let's ensure it is.
                # (The Codec's internal resampler is already on the correct device, this is for safety)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config['sample_rate']).to(device)
                waveform = resampler(waveform)
            waveform = waveform.mean(dim=0) # to mono
            
            # Pad to be a multiple of chunk length
            if waveform.shape[0] % chunk_len_samples != 0:
                 pad_amount = chunk_len_samples - (waveform.shape[0] % chunk_len_samples)
                 waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

            chunks = waveform.split(chunk_len_samples)
            for chunk in chunks:
                if chunk.shape[0] == chunk_len_samples:
                    # Now the 'chunk' tensor is already on the GPU, so encode will work.
                    codes = audio_codec.encode(chunk)[0] # shape [Tokens]
                    all_codes.append(codes)
        except Exception as e:
            print(f"Skipping file {audio_file} due to error: {e}")

    if not all_codes:
         print(f"[FATAL] No valid audio chunks could be processed.")
         sys.exit(1)
         
    all_codes_np = np.concatenate(all_codes).astype(np.int32)
    
    cl = config['context_length']
    num_samples = len(all_codes_np) - cl - 1
    strides = all_codes_np.strides[0]
    
    all_indices = np.lib.stride_tricks.as_strided(all_codes_np, shape=(num_samples, cl), strides=(strides, strides))
    all_targets = np.lib.stride_tricks.as_strided(all_codes_np[1:], shape=(num_samples, cl), strides=(strides, strides))
    
    rng = np.random.default_rng(42)
    perm = rng.permutation(num_samples)
    all_indices, all_targets = all_indices[perm], all_targets[perm]
    
    micro_bs = config['batch_size']
    num_micro_batches = num_samples // micro_bs
    
    if num_to_trim := num_samples % micro_bs:
        all_indices, all_targets = [arr[:-num_to_trim] for arr in (all_indices, all_targets)]
        
    all_indices_b = all_indices.reshape(num_micro_batches, micro_bs, cl)
    all_targets_b = all_targets.reshape(num_micro_batches, micro_bs, cl)
    
    print(f"--- Audio prep complete: {num_micro_batches} micro-batches created. ---")
    return (all_indices_b, all_targets_b), num_micro_batches

# --- Q-Learning Controller and Training Manager (largely unchanged) ---
# ... (The JaxHakmemQController, save/load, and GAN step logic is identical) ...
class JaxHakmemQController:
    """A JAX-compatible Q-learning controller for hyperparameter tuning, managed in Python runtime."""
    def __init__(self, initial_lr: float, config: Dict[str, Any], logger_suffix: str = ""):
        self.config = config
        self.current_lr = initial_lr
        self.logger_suffix = logger_suffix
        self.q_table_size = int(self.config["q_table_size"])
        self.num_actions = int(self.config["num_lr_actions"])
        self.lr_change_factors = self.config["lr_change_factors"]
        self.q_table = np.zeros((self.q_table_size, self.num_actions), dtype=np.float32)
        self.learning_rate_q = float(self.config["learning_rate_q"])
        self.discount_factor_q = float(self.config["discount_factor_q"])
        self.exploration_rate_q = float(self.config["exploration_rate_q"])
        self.lr_min = float(self.config["lr_min"])
        self.lr_max = float(self.config["lr_max"])
        self.loss_history = deque(maxlen=int(self.config["metric_history_len"]))
        self.loss_min = float(self.config["loss_min"])
        self.loss_max = float(self.config["loss_max"])
        self.last_action_idx: Optional[int] = None
        self.last_state_idx: Optional[int] = None
        print(f"--- HAKMEM Q-Controller ({self.logger_suffix}) initialized. LR: {self.current_lr:.2e}, Q-Table: {self.q_table.shape} ---")
    def _discretize_value(self, value: float) -> int:
        if value <= self.loss_min: return 0
        if value >= self.loss_max: return self.q_table_size - 1
        bin_size = (self.loss_max - self.loss_min) / self.q_table_size
        return min(int((value - self.loss_min) / bin_size), self.q_table_size - 1)
    def _get_current_state_idx(self, current_loss: Optional[float]) -> int:
        if current_loss is not None: return self._discretize_value(current_loss)
        return self.q_table_size // 2
    def choose_action(self, current_loss: Optional[float]) -> float:
        self.last_state_idx = self._get_current_state_idx(current_loss)
        if random.random() < self.exploration_rate_q:
            self.last_action_idx = random.randint(0, self.num_actions - 1)
        else:
            self.last_action_idx = np.argmax(self.q_table[self.last_state_idx]).item()
        change_factor = self.lr_change_factors[self.last_action_idx]
        self.current_lr = np.clip(self.current_lr * change_factor, self.lr_min, self.lr_max)
        return self.current_lr
    def log_reward(self, reward: float, current_loss: Optional[float]):
        if self.last_state_idx is None or self.last_action_idx is None: return
        current_q = self.q_table[self.last_state_idx, self.last_action_idx]
        next_state_idx = self._get_current_state_idx(current_loss)
        max_next_q = np.max(self.q_table[next_state_idx])
        new_q = current_q + self.learning_rate_q * (reward + self.discount_factor_q * max_next_q - current_q)
        self.q_table[self.last_state_idx, self.last_action_idx] = new_q
        if current_loss is not None: self.loss_history.append(current_loss)
    def state_dict(self) -> Dict[str, Any]:
        return {"current_lr": self.current_lr, "q_table": self.q_table.tolist(), "loss_history": list(self.loss_history), "last_action_idx": self.last_action_idx, "last_state_idx": self.last_state_idx}
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_lr = state_dict.get("current_lr", self.current_lr)
        self.q_table = np.array(state_dict.get("q_table", self.q_table.tolist()), dtype=np.float32)
        self.loss_history = deque(state_dict.get("loss_history", []), maxlen=self.loss_history.maxlen)
        self.last_action_idx = state_dict.get("last_action_idx")
        self.last_state_idx = state_dict.get("last_state_idx")

def save_checkpoint(g_state, d_state, g_q_controller, d_q_controller, basename):
    state_dict = {'g': jax.device_get(serialization.to_state_dict(g_state)), 'd': jax.device_get(serialization.to_state_dict(d_state)), 'g_q_ctrl': g_q_controller.state_dict(), 'd_q_ctrl': d_q_controller.state_dict()}
    with open(f"{basename}.pkl", 'wb') as f: pickle.dump(state_dict, f)
    print(f"\n--- Checkpoint saved. Step: {g_state.step} ---")

def load_checkpoint(g_state, d_state, g_q_controller, d_q_controller, basename):
    filename = f"{basename}.pkl"
    if not os.path.exists(filename):
        print("--- No checkpoint found. ---")
        return g_state, d_state, g_q_controller, d_q_controller
    with open(filename, 'rb') as f: saved_state_dict = pickle.load(f)
    print(f"--- Checkpoint found. Restoring... ---")
    g_state = serialization.from_state_dict(g_state, saved_state_dict['g'])
    d_state = serialization.from_state_dict(d_state, saved_state_dict['d'])
    g_q_controller.load_state_dict(saved_state_dict['g_q_ctrl'])
    d_q_controller.load_state_dict(saved_state_dict['d_q_ctrl'])
    print(f"--- States and Q-Controllers restored. Resuming from step: {g_state.step} ---")
    return g_state, d_state, g_q_controller, d_q_controller

def save_config(config,basename):
    with open(f"{basename}.json",'w') as f:json.dump(config,f,indent=4)
    print(f"--- Model config saved to {basename}.json ---")

@partial(jax.jit, static_argnames=['g_apply_fn', 'd_apply_fn', 'recon_weight', 'adv_weight'], inline=False)
def gan_train_step(g_params, d_params, batch, key, g_apply_fn, d_apply_fn, recon_weight, adv_weight):
    real_indices, real_targets = batch
    g_output = g_apply_fn({'params': g_params}, real_indices)
    real_embeds, g_logits, g_embedding_matrix = g_output['embeddings'], g_output['final_logits'], g_output['embedding_matrix']
    gumbel_noise = jax.random.gumbel(key, g_logits.shape, dtype=jnp.float32)
    fake_probs = nn.softmax((g_logits + gumbel_noise) / 0.5)
    fake_embeds = fake_probs @ g_embedding_matrix.astype(jnp.float32)
    def d_loss_fn(d_params_inner):
        real_logits_d = d_apply_fn({'params': d_params_inner}, real_embeds)
        fake_logits_d = d_apply_fn({'params': d_params_inner}, jax.lax.stop_gradient(fake_embeds))
        real_loss = optax.sigmoid_binary_cross_entropy(real_logits_d, jnp.ones_like(real_logits_d)).mean()
        fake_loss = optax.sigmoid_binary_cross_entropy(fake_logits_d, jnp.zeros_like(fake_logits_d)).mean()
        return (real_loss + fake_loss) / 2.0
    d_loss, d_grads = jax.value_and_grad(d_loss_fn)(d_params)
    def g_loss_fn(g_params_inner):
        g_output_inner = g_apply_fn({'params': g_params_inner}, real_indices)
        g_logits_inner, g_embedding_matrix_inner = g_output_inner['final_logits'], g_output_inner['embedding_matrix']
        fake_probs_inner = nn.softmax((g_logits_inner + gumbel_noise) / 0.5)
        fake_embeds_inner = fake_probs_inner @ g_embedding_matrix_inner.astype(jnp.float32)
        recon_loss = optax.softmax_cross_entropy_with_integer_labels(g_logits_inner, real_targets).mean()
        fake_d_logits = d_apply_fn({'params': d_params}, fake_embeds_inner)
        adv_loss = optax.sigmoid_binary_cross_entropy(fake_d_logits, jnp.ones_like(fake_d_logits)).mean()
        return recon_weight * recon_loss + adv_weight * adv_loss
    g_loss, g_grads = jax.value_and_grad(g_loss_fn)(g_params)
    g_grads = tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4), g_grads)
    d_grads = tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4), d_grads)
    metrics = {'g_loss': g_loss, 'd_loss': d_loss}
    return g_grads, d_grads, metrics

class AdversarialTrainingManager:
    def __init__(self,generator,discriminator,config: Dict[str, Any],data,basename:str):
        self.g, self.d = generator, discriminator; self.config = config; self.data, self.basename = data, basename
        self.g_state, self.d_state = None, None; self.g_q_controller, self.d_q_controller = None, None
        self.should_shutdown = False; signal.signal(signal.SIGINT, self._handle_sigint)
    def _handle_sigint(self,s,f):
        if not self.should_shutdown: print("\n--- SIGINT received. Saving state... ---"); self.should_shutdown=True
    def run(self):
        (i_all, t_all), num_micro_batches = self.data; key = jax.random.PRNGKey(42); key, g_key, d_key = jax.random.split(key, 3)
        num_devices = jax.device_count()
        if num_devices > 1:
             print(f"--- Sharding data across {num_devices} devices... ---")
             device_mesh = mesh_utils.create_device_mesh((num_devices,)); mesh = Mesh(device_mesh, axis_names=('batch',))
             data_sharding = PositionalSharding(device_mesh).reshape(num_devices, 1, 1)
             i_all, t_all = jax.device_put(i_all, data_sharding), jax.device_put(t_all, data_sharding)
        else: print("--- Single device detected. Running without sharding. ---"); i_all, t_all = jax.device_put(i_all), jax.device_put(t_all)
        dummy_indices = i_all[0,:1]; g_params = self.g.init(g_key, dummy_indices)['params']
        dummy_embeds = self.g.apply({'params': g_params}, dummy_indices)['embeddings']
        print(f'--- Generator Initialized: {sum(x.size for x in tree.leaves(g_params)):,} params. ---')
        d_params = self.d.init(d_key, dummy_embeds)['params']
        print(f'--- Discriminator Initialized: {sum(x.size for x in tree.leaves(d_params)):,} params. ---')
        def tx_factory(learning_rate): return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=learning_rate, weight_decay=0.01))
        g_tx = optax.inject_hyperparams(tx_factory)(learning_rate=self.config['g_learning_rate'])
        d_tx = optax.inject_hyperparams(tx_factory)(learning_rate=self.config['d_learning_rate'])
        self.g_state = train_state.TrainState.create(apply_fn=self.g.apply, params=g_params, tx=g_tx)
        self.d_state = train_state.TrainState.create(apply_fn=self.d.apply, params=d_params, tx=d_tx)
        self.g_q_controller = JaxHakmemQController(self.config['g_learning_rate'], self.config, "Generator")
        self.d_q_controller = JaxHakmemQController(self.config['d_learning_rate'], self.config, "Discriminator")
        save_config(self.config, self.basename)
        self.g_state, self.d_state, self.g_q_controller, self.d_q_controller = load_checkpoint(self.g_state, self.d_state, self.g_q_controller, self.d_q_controller, self.basename)
        num_global_steps = self.config['epochs'] * num_micro_batches; start_step = self.g_state.step
        if start_step >= num_global_steps: print(f"--- Training already completed. ---"); return
        print("--- Compiling and warming up training functions... ---")
        jit_train_step = partial(gan_train_step, g_apply_fn=self.g.apply, d_apply_fn=self.d.apply, recon_weight=self.config['recon_weight'], adv_weight=self.config['adv_weight'])
        warmup_key, key = jax.random.split(key)
        g_grads_warmup, d_grads_warmup, _ = jit_train_step(self.g_state.params, self.d_state.params, (dummy_indices, t_all[0,:1]), warmup_key)
        jax.block_until_ready((g_grads_warmup, d_grads_warmup)); print("--- Compilation complete. Starting training. ---")
        last_g_loss, last_d_loss = 20.0, 20.0 
        with tqdm(total=num_global_steps, initial=start_step, desc="GAN Training") as pbar:
            for step in range(start_step, num_global_steps):
                new_g_lr = self.g_q_controller.choose_action(last_g_loss); new_d_lr = self.d_q_controller.choose_action(last_d_loss)
                grad_key, key = jax.random.split(key); micro_batch_idx = step % num_micro_batches
                batch_device = (i_all[micro_batch_idx], t_all[micro_batch_idx])
                g_grads, d_grads, metrics = jit_train_step(self.g_state.params, self.d_state.params, batch_device, grad_key)
                g_updates, g_new_opt_state = self.g_state.tx.update(g_grads, self.g_state.opt_state, self.g_state.params, hyperparams={'learning_rate': new_g_lr})
                g_new_params = optax.apply_updates(self.g_state.params, g_updates)
                self.g_state = self.g_state.replace(step=self.g_state.step + 1, params=g_new_params, opt_state=g_new_opt_state)
                d_updates, d_new_opt_state = self.d_state.tx.update(d_grads, self.d_state.opt_state, self.d_state.params, hyperparams={'learning_rate': new_d_lr})
                d_new_params = optax.apply_updates(self.d_state.params, d_updates)
                self.d_state = self.d_state.replace(params=d_new_params, opt_state=d_new_opt_state)
                current_g_loss, current_d_loss = metrics['g_loss'].item(), metrics['d_loss'].item()
                g_reward = (last_g_loss - current_g_loss); d_reward = (last_d_loss - current_d_loss) 
                self.g_q_controller.log_reward(g_reward, current_g_loss); self.d_q_controller.log_reward(d_reward, current_d_loss)
                last_g_loss, last_d_loss = current_g_loss, current_d_loss
                pbar.set_description(f"Epoch {int(step/num_micro_batches)+1}/{self.config['epochs']}")
                pbar.set_postfix(G_loss=f"{current_g_loss:.3f}",D_loss=f"{current_d_loss:.3f}",G_lr=f"{new_g_lr:.1e}",D_lr=f"{new_d_lr:.1e}")
                pbar.update(1)
                if self.should_shutdown:
                    save_checkpoint(self.g_state, self.d_state, self.g_q_controller, self.d_q_controller, self.basename); print("\n--- Interrupt honored. Training halted. ---"); return
        print("\n--- Adversarial training complete. ---"); save_checkpoint(self.g_state, self.d_state, self.g_q_controller, self.d_q_controller, self.basename)
        
# --- Inference and Generation Logic ---
@partial(jax.jit,static_argnames=['model_apply_fn', 'use_guidance', 'top_p'])
def predict_step_fn(model_apply_fn,params,indices,key,temp,top_p,use_guidance,deviation_threshold):
    model_outputs=model_apply_fn({'params':params},indices)
    final_logits,drive_logits=model_outputs['final_logits'],model_outputs['drive_logits']
    def kl_divergence(logits_p,logits_q):
        log_p=jax.nn.log_softmax(logits_p,axis=-1)[:,-1,:];log_q=jax.nn.log_softmax(logits_q,axis=-1)[:,-1,:]
        return jnp.sum(jnp.exp(log_p)*(log_p-log_q),axis=-1)
    def power_modulator(fl,dl): 
        dev_sem_syn=kl_divergence(dl['sem'],dl['syn']);dev_exe_sem=kl_divergence(dl['exe'],dl['sem'])
        deviation=jnp.maximum(dev_sem_syn,dev_exe_sem);alpha=jnp.clip(1.0-(deviation/deviation_threshold),0.0,1.0)
        corrective_logits=dl['exe']+dl['syn'];p_standard=jax.nn.softmax(fl,axis=-1)
        p_corrective=jax.nn.softmax(corrective_logits,axis=-1)
        p_mixed=(alpha[:,None,None]*p_standard)+((1.0-alpha[:,None,None])*p_corrective)
        return jnp.log(p_mixed.clip(1e-9))
    effective_logits=jax.lax.cond(use_guidance,power_modulator,lambda fl,dl:fl,final_logits,drive_logits)
    scaled = effective_logits[:,-1,:]/jnp.maximum(temp,1e-6)
    def apply_top_p(logits): 
        sorted_indices=jnp.argsort(logits,axis=-1)[...,::-1];sorted_logits=jnp.take_along_axis(logits,sorted_indices,axis=-1)
        cum_probs=jnp.cumsum(nn.softmax(sorted_logits,axis=-1),axis=-1);sorted_to_remove=cum_probs>top_p
        sorted_to_remove=jnp.concatenate([jnp.zeros_like(sorted_to_remove[...,:1]),sorted_to_remove[...,:-1]],axis=-1)
        to_remove=jnp.zeros_like(sorted_to_remove).at[...,sorted_indices].set(sorted_to_remove);return jnp.where(to_remove,-jnp.inf,logits)
    final_scaled=jax.lax.cond(top_p<1.0,apply_top_p,lambda x:x,scaled)
    return jax.random.categorical(key,final_scaled,axis=-1)

class WubuSynth:
    """The inference engine for the audio synthesizer."""
    def __init__(self, model_basename):
        print("--- WubuSynth Awakens ---");self.basename=model_basename
        with open(f"{self.basename}.json",'r') as f: self.config=json.load(f)
        
        self.audio_codec = WubuAudioCodec(sample_rate=self.config['sample_rate'])
        self.config['vocab_size'] = self.audio_codec.vocab_size

        self.model = Generator(
            vocab_size=self.config['vocab_size'], d_model=self.config['d_model'], n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers'], max_len=self.config['max_len']
        )
        print("--- Assimilating knowledge from checkpoint... ---")
        try:
            with open(f"{self.basename}.pkl",'rb') as f: saved_state_dict=pickle.load(f)
        except FileNotFoundError: print(f"[ERROR] Checkpoint not found."),sys.exit(1)
        
        g_saved_state = saved_state_dict.get('g') or saved_state_dict.get('generator')
        if not g_saved_state or 'params' not in g_saved_state: raise ValueError("Checkpoint invalid.")
        
        dummy_params=self.model.init(jax.random.PRNGKey(0),jnp.ones((1,1),dtype=jnp.int32))['params']
        self.params=serialization.from_state_dict(dummy_params,g_saved_state['params'])
        step=g_saved_state.get('step','unknown')
        print(f"--- WubuSynth assimilated knowledge from step {step}. ---")
        self.jit_compiled=False
        
    def generate(self, prompt_wav:str, output_wav:str, gen_len_sec:int, temp:float, top_p:float, use_guidance:bool, deviation_threshold:float):
        if not self.jit_compiled: print("--- JIT compiling Generator... ---",flush=True)
        key = jax.random.PRNGKey(int(time.time()))
        
        # --- Process Prompt Audio ---
        try:
            prompt_waveform, sr = torchaudio.load(prompt_wav)
            if sr != self.config['sample_rate']:
                 resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.config['sample_rate'])
                 prompt_waveform = resampler(prompt_waveform)
            prompt_waveform = prompt_waveform.mean(dim=0) # mono
            indices = self.audio_codec.encode(prompt_waveform)[0]
        except Exception as e:
            print(f"Could not load prompt audio: {e}. Starting from scratch.")
            indices = np.array([], dtype=np.int32)
        
        # --- Autoregressive Generation Loop ---
        tokens_per_second = self.audio_codec.model.frame_rate
        num_new_tokens = int(gen_len_sec * tokens_per_second)
        
        pbar = tqdm(total=num_new_tokens, desc="Synthesizing Audio")
        for _ in range(num_new_tokens):
            current_indices = indices[-self.config['context_length']:]
            pad_len = self.config['context_length'] - len(current_indices)
            if pad_len > 0:
                padded_indices = np.pad(current_indices, (pad_len, 0), 'constant', constant_values=self.audio_codec.pad_id)
            else:
                padded_indices = current_indices

            i_batch = jax.device_put(np.array(padded_indices,dtype=np.int32)[None,:])
            key,subkey = jax.random.split(key)
            next_idx_array = predict_step_fn(
                self.model.apply, self.params, i_batch, subkey, temp, top_p, use_guidance, deviation_threshold
            )
            if not self.jit_compiled:
                next_idx_array.block_until_ready()
                self.jit_compiled=True

            new_idx = int(next_idx_array.item())
            indices = np.append(indices, new_idx)
            pbar.update(1)
        pbar.close()

        # --- Decode and Save Output ---
        print("\n--- Decoding generated tokens back to waveform... ---")
        generated_waveform = self.audio_codec.decode(indices[None, :])
        sf.write(output_wav, generated_waveform.numpy().T, self.config['sample_rate'])
        print(f"--- Synthesis complete. Audio saved to {output_wav} ---")


# --- Main Application Logic ---
def main():
    parser = argparse.ArgumentParser(description="WubuMind Adversarial Synthesizer")
    parser.add_argument('command', choices=['train', 'infer'], help="Command: 'train' a new model or 'infer' with an existing one.")
    parser.add_argument('--basename', type=str, default="wubumind_adversarial_synth_v1", help="Base name for model files.")
    parser.add_argument('--audio-dir', type=str, default="./audio_corpus", help="Directory with training audio files (.wav, .mp3).")
    parser.add_argument('--prompt', type=str, default=None, help="Path to prompt .wav file for inference.")
    parser.add_argument('--output', type=str, default="generated_audio.wav", help="Path to save the generated .wav file.")
    parser.add_argument('--len', type=int, default=10, help="Length of audio to generate in seconds.")
    parser.add_argument('--temp', type=float, default=0.9, help="Generation temperature.")
    parser.add_argument('--top-p', type=float, default=0.95, help="Top-p (nucleus) sampling probability.")
    parser.add_argument('--guidance', action='store_true', help="Enable Power Modulator guidance during inference.")
    parser.add_argument('--threshold', type=float, default=5.0, help="KL-divergence threshold for guidance.")
    
    args = parser.parse_args()

    if args.command == "train":
        # --- Configuration Dictionaries ---
        TRAINING_CONFIG = {
            'epochs': 20, 'batch_size': 1, 'context_length': 256, # In audio tokens
            'g_learning_rate': 3e-4, 'd_learning_rate': 5e-6,
            'recon_weight': 1.0, 'adv_weight': 0.1,
            'sample_rate': 24000, 'chunk_len_sec': 5.0, # for data prep
        }
        MODEL_CONFIG = {
            'd_model': 256, 'n_heads': 4, 'n_layers': 4,
            'max_len': TRAINING_CONFIG['context_length']
        }
        QLEARN_CONFIG = {
            "q_table_size": 10, "num_lr_actions": 5, "lr_change_factors": [0.5, 0.9, 1.0, 1.1, 1.5],
            "learning_rate_q": 0.1, "discount_factor_q": 0.9, "exploration_rate_q": 0.1,
            "lr_min": 1e-7, "lr_max": 1e-2, "metric_history_len": 10,
            "loss_min": 0.0, "loss_max": 20.0
        }
        
        print(f"--- WubuMind Synthesizer Foundry ---")
        print(f"--- Device: {jax.devices()[0].platform.upper()} ({jax.device_count()} devices) ---")
        
        # Ensure audio directory exists
        if not os.path.isdir(args.audio_dir):
            print(f"[FATAL] Audio directory not found: {args.audio_dir}")
            print("Please create it and add audio files for training.")
            sys.exit(1)

        audio_codec = WubuAudioCodec(sample_rate=TRAINING_CONFIG['sample_rate'])
        
        d_arch_config = {k: v for k, v in MODEL_CONFIG.items()}
        g_arch_config = {**MODEL_CONFIG, 'vocab_size': audio_codec.vocab_size}
        full_config = {**g_arch_config, **TRAINING_CONFIG, **QLEARN_CONFIG}
        
        data_bundle, num_micro_batches = prepare_audio_data(args.audio_dir, audio_codec, TRAINING_CONFIG)
        generator = Generator(**g_arch_config)
        discriminator = Discriminator(**d_arch_config)
        
        AdversarialTrainingManager(generator, discriminator, full_config, (data_bundle, num_micro_batches), args.basename).run()

    elif args.command == "infer":
        try:
            synth = WubuSynth(args.basename)
            synth.generate(
                prompt_wav=args.prompt, output_wav=args.output, gen_len_sec=args.len,
                temp=args.temp, top_p=args.top_p, use_guidance=args.guidance,
                deviation_threshold=args.threshold
            )
        except Exception as e:
            print("\n--- An error occurred during synthesis ---")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()