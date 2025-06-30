# wubu_diffusion.py
# Denoising Diffusion Probabilistic Model (DDPM) using a true Hyperbolic Geometric Attention U-Net (HGA-UNet).
# v4.4: Final fix for time embedding dimension mismatch in U-Net.
# CPU-only, NO CNNs, NO explicit DFT/DCT.
# Based on the refined principle: "The geometry IS the architecture."

import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
import numpy as np
import math, random, argparse, logging, time, os
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
from collections import deque
from pathlib import Path
import torchvision.transforms as T

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio unavailable. Dummy video creation and some loading might fail.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# --- Constants ---
EPS = 1e-8

# --- Basic Logging Setup ---
logger_wubu_diffusion = logging.getLogger("WuBuDiffusionHGA")
if not logger_wubu_diffusion.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_wubu_diffusion.addHandler(handler)
    logger_wubu_diffusion.setLevel(logging.INFO)

# --- Global TQDM setup ---
_TQDM_INITIAL_WRAPPER = None
try: from tqdm import tqdm as _tqdm_imported_module; _TQDM_INITIAL_WRAPPER = _tqdm_imported_module
except ImportError:
    class _TqdmDummyMissing:
        def __init__(self,iterable=None,*a,**kw): self.iterable=iterable or []
        def __iter__(self): return iter(self.iterable)
        def __enter__(self): return self
        def __exit__(self,*e): pass
        def set_postfix(self,*a,**kw): pass
        def update(self,*a,**kw): pass
        def close(self): pass
    _TQDM_INITIAL_WRAPPER = _TqdmDummyMissing
tqdm = _TQDM_INITIAL_WRAPPER

# --- Utility Functions ---
def init_weights_general(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if getattr(m, 'elementwise_affine', True):
            if hasattr(m, 'weight') and m.weight is not None: nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

# --- Hyperbolic Geometry Utilities ---
class PoincareBall:
    @staticmethod
    def expmap0(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if (c <= 0).all(): return v
        sqrt_c = torch.sqrt(c).unsqueeze(-1)
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(EPS)
        lam = (1. / (sqrt_c * v_norm)) * torch.tanh(sqrt_c * v_norm)
        return lam * v
    @staticmethod
    def logmap0(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if (c <= 0).all(): return p
        sqrt_c = torch.sqrt(c).unsqueeze(-1)
        p_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp_min(EPS)
        lam = (1. / (sqrt_c * p_norm)) * torch.atanh(sqrt_c * p_norm)
        return lam * p
    @staticmethod
    def dist(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if (c <= 0).all(): return torch.norm(x - y, p=2, dim=-1)
        sqrt_c = torch.sqrt(c).squeeze()
        diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
        x_norm_sq = torch.sum(x ** 2, dim=-1)
        y_norm_sq = torch.sum(y ** 2, dim=-1)
        num = 2 * c * diff_norm_sq
        den = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
        arg = (1 + num / (den + EPS)).clamp(min=1.0)
        return (1.0 / sqrt_c) * torch.acosh(arg)

# --- Dataset (Unchanged) ---
class VideoFrameDatasetCPU(Dataset):
    def __init__(self, video_path, image_size, data_fraction=1.0, val_fraction=0.0, mode='train', seed=42):
        super().__init__()
        self.video_path=video_path; self.image_size=image_size; self.mode=mode.lower(); self.logger=logger_wubu_diffusion.getChild(f"Dataset_{self.mode.upper()}")
        if not os.path.isfile(self.video_path): raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self.video_frames_in_ram=None;
        if IMAGEIO_AVAILABLE and imageio:
            try:
                reader=imageio.get_reader(self.video_path,'ffmpeg'); frames_list=[]
                for frame_np in reader:
                    if frame_np.ndim==3 and frame_np.shape[-1] in [3,4]: frames_list.append(torch.from_numpy(np.transpose(frame_np[...,:3],(2,0,1)).copy()))
                if frames_list: self.video_frames_in_ram=torch.stack(frames_list).contiguous()
                reader.close()
            except Exception as e: self.logger.error(f"imageio failed to read {self.video_path}: {e}")
        if self.video_frames_in_ram is None: raise RuntimeError(f"Failed to load video '{self.video_path}'.")
        self.resize_transform=T.Resize(self.image_size, antialias=True); self.normalize_transform=T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        all_indices=list(range(self.video_frames_in_ram.shape[0])); random.Random(seed).shuffle(all_indices)
        if 0.0<val_fraction<1.0:
            num_val=int(len(all_indices)*val_fraction)
            self.indices=all_indices[num_val:] if self.mode=='train' else all_indices[:num_val]
        else: self.indices=all_indices
        if self.mode=='train' and data_fraction<1.0: self.indices=self.indices[:max(1,int(len(self.indices)*data_fraction))]
        self.logger.info(f"Dataset({self.mode.upper()}) loaded {len(self.indices)} frames from {self.video_path}.")
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, idx: int) -> torch.Tensor:
        frame_idx=self.indices[idx]; frame_tensor_uint8=self.video_frames_in_ram[frame_idx]
        return self.normalize_transform(self.resize_transform(frame_tensor_uint8).float()/255.0)

# --- Diffusion Components (Cleaned and Fixed) ---
def _extract(a, t, x_shape):
    res = a.to(t.device)[t].float()
    while len(res.shape) < len(x_shape):
        res = res.unsqueeze(-1)
    return res.expand(x_shape)
class DiffusionProcess:
    def __init__(self, timesteps, beta_schedule='cosine', device='cpu'):
        self.timesteps=timesteps; self.device=device
        if beta_schedule=="linear": self.betas=torch.linspace(1e-4,0.02,timesteps,device=device)
        elif beta_schedule=="cosine":
            s=0.008; x=torch.linspace(0,timesteps,timesteps+1,device=device)
            ac=torch.cos(((x/timesteps)+s)/(1+s)*math.pi*0.5)**2
            ac = ac / ac[0]
            self.betas=torch.clip(1-(ac[1:]/ac[:-1]),0.0001,0.9999)
        self.alphas=1.-self.betas; self.alphas_cumprod=torch.cumprod(self.alphas,axis=0); self.alphas_cumprod_prev=F.pad(self.alphas_cumprod[:-1],(1,0),value=1.)
        self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod); self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod=torch.sqrt(1./self.alphas_cumprod); self.sqrt_recipm1_alphas_cumprod=torch.sqrt(1./self.alphas_cumprod - 1)
        self.posterior_variance=self.betas*(1.-self.alphas_cumprod_prev)/(1.-self.alphas_cumprod)
        self.posterior_mean_coef1=self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.-self.alphas_cumprod)
        self.posterior_mean_coef2=(1.-self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1.-self.alphas_cumprod)
    def q_sample(self,x_start,t,noise=None): noise=torch.randn_like(x_start) if noise is None else noise; return _extract(self.sqrt_alphas_cumprod,t,x_start.shape)*x_start+_extract(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape)*noise
    def _predict_xstart_from_eps(self,x_t,t,eps): return _extract(self.sqrt_recip_alphas_cumprod,t,x_t.shape)*x_t - _extract(self.sqrt_recipm1_alphas_cumprod,t,x_t.shape)*eps
    def q_posterior(self,x_start,x_t,t): post_mean=_extract(self.posterior_mean_coef1,t,x_t.shape)*x_start+_extract(self.posterior_mean_coef2,t,x_t.shape)*x_t; return post_mean, _extract(self.posterior_variance,t,x_t.shape)
    def p_sample(self,pred_noise,x,t): x0_pred=self._predict_xstart_from_eps(x,t,pred_noise); model_mean,_=self.q_posterior(x0_pred,x,t); noise=torch.randn_like(x) if t[0]>0 else 0; return model_mean+torch.sqrt(_extract(self.posterior_variance,t,x.shape))*noise
    @torch.no_grad()
    def sample(self, model_callable, batch_size, latent_shape, show_progress=True):
        x_t=torch.randn(batch_size,*latent_shape,device=self.device)
        iterable=tqdm(reversed(range(self.timesteps)),desc="Sampling",total=self.timesteps,disable=not show_progress)
        for i in iterable: t=torch.full((batch_size,),i,device=self.device,dtype=torch.long); predicted_noise=model_callable(x_t,t); x_t=self.p_sample(predicted_noise,x_t,t)
        return x_t

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, time): device=time.device; half_dim=self.dim//2; embeddings=math.log(10000)/(half_dim-1); embeddings=torch.exp(torch.arange(half_dim,device=device)*-embeddings); embeddings=time[:,None]*embeddings[None,:]; embeddings=torch.cat((embeddings.sin(),embeddings.cos()),dim=-1); return embeddings

# --- Image Encoder/Decoder (Robust) ---
class ImagePatchEncoder(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__(); self.patch_size=patch_size; self.proj=nn.Linear(in_channels*patch_size*patch_size,embed_dim)
    def forward(self, x):
        B,C,H,W=x.shape; x=x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size)
        x=x.permute(0,2,3,1,4,5).contiguous().view(B,-1,C*self.patch_size*self.patch_size); return self.proj(x)
class ImagePatchDecoder(nn.Module):
    def __init__(self, patch_size, out_channels, embed_dim, num_patches_h, num_patches_w):
        super().__init__(); self.patch_size=patch_size; self.out_channels=out_channels; self.num_patches_h=num_patches_h; self.num_patches_w=num_patches_w
        patch_dim=out_channels*patch_size*patch_size; self.proj=nn.Sequential(nn.Linear(embed_dim,patch_dim),nn.Tanh())
    def forward(self, x):
        B,N,D=x.shape; x=self.proj(x).view(B,self.num_patches_h,self.num_patches_w,self.out_channels,self.patch_size,self.patch_size)
        return x.permute(0,3,1,4,2,5).reshape(B,self.out_channels,self.num_patches_h*self.patch_size,self.num_patches_w*self.patch_size)

# --- Hyperbolic Geometric Attention (HGA) U-Net Architecture ---
class HyperbolicPooling(nn.Module):
    def __init__(self, dim_in):
        super().__init__(); self.pool_proj = nn.Linear(dim_in, dim_in*2)
    def forward(self, features, positions, c):
        B, N, _ = features.shape; H = W = int(N**0.5)
        features_2d = features.view(B, H, W, -1).permute(0, 3, 1, 2)
        pooled_features = F.avg_pool2d(features_2d, 2).permute(0, 2, 3, 1).reshape(B, N//4, -1)
        pooled_features = self.pool_proj(pooled_features)
        tangent_pos = PoincareBall.logmap0(positions, c)
        tangent_pos_2d = tangent_pos.view(B, H, W, -1).permute(0, 3, 1, 2)
        pooled_tangent = F.avg_pool2d(tangent_pos_2d, 2).permute(0, 2, 3, 1).reshape(B, N//4, -1)
        pooled_positions = PoincareBall.expmap0(pooled_tangent, c)
        return pooled_features, pooled_positions
class HyperbolicUnpooling(nn.Module):
    def __init__(self, dim_in):
        super().__init__(); self.unpool_proj = nn.Linear(dim_in, dim_in//2)
        offset_scale = 0.01; offsets = torch.tensor([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,1.]]) * offset_scale
        self.register_buffer('offsets', offsets)
    def forward(self, features, positions, c):
        B, N_coarse, _ = features.shape; D_pos = positions.shape[-1]
        features_up = self.unpool_proj(features).unsqueeze(2).repeat(1,1,4,1).view(B,N_coarse*4,-1)
        tangent_pos = PoincareBall.logmap0(positions, c)
        padded_offsets = F.pad(self.offsets, (0, D_pos-self.offsets.shape[-1])) if self.offsets.shape[-1]<D_pos else self.offsets
        new_tangent_pos = tangent_pos.unsqueeze(2) + padded_offsets
        positions_up = PoincareBall.expmap0(new_tangent_pos.view(B, N_coarse*4, -1), c)
        return features_up, positions_up
class kNNHyperbolicAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads, k):
        super().__init__(); self.dim=dim; self.n_heads=n_heads; self.k=k; self.h_dim=dim//n_heads
        self.q_proj=nn.Linear(dim,dim); self.k_proj=nn.Linear(dim,dim); self.v_proj=nn.Linear(dim,dim); self.out_proj=nn.Linear(dim,dim)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
        self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim); self.log_tau = nn.Parameter(torch.tensor(0.0))
    def forward(self, x, positions, c, time_emb=None):
        B, N, _ = x.shape; x_res = x; x_norm1 = self.norm1(x)
        if time_emb is not None: x_norm1 = x_norm1 + time_emb.unsqueeze(1)
        q=self.q_proj(x_norm1).view(B,N,self.n_heads,self.h_dim).transpose(1,2)
        k=self.k_proj(x_norm1).view(B,N,self.n_heads,self.h_dim).transpose(1,2)
        v=self.v_proj(x_norm1).view(B,N,self.n_heads,self.h_dim).transpose(1,2)
        with torch.no_grad(): dist_matrix=PoincareBall.dist(positions.unsqueeze(1),positions.unsqueeze(2),c); attn_dists,top_k_indices=torch.topk(dist_matrix,self.k,dim=-1,largest=False)
        k_for_gather = k.unsqueeze(3).expand(-1, -1, -1, self.k, -1)
        v_for_gather = v.unsqueeze(3).expand(-1, -1, -1, self.k, -1)
        indices = top_k_indices.unsqueeze(1).unsqueeze(4).expand(-1, self.n_heads, -1, -1, self.h_dim)
        k_gathered = torch.gather(k_for_gather, 2, indices)
        v_gathered = torch.gather(v_for_gather, 2, indices)
        feature_scores = torch.matmul(q.unsqueeze(3), k_gathered.transpose(-1,-2)).squeeze(3) / math.sqrt(self.h_dim)
        tau=torch.exp(self.log_tau)+EPS; geometric_scores = -(attn_dists.unsqueeze(1))/tau
        combined_scores = feature_scores + geometric_scores; attn_probs = F.softmax(combined_scores,dim=-1)
        attn_output=torch.matmul(attn_probs.unsqueeze(3),v_gathered).squeeze(3).transpose(1,2).reshape(B,N,-1)
        x = x_res + self.out_proj(attn_output); x = x + self.ffn(self.norm2(x)); return x
class HGA_UNet_Block(nn.Module):
    def __init__(self, num_layers, dim, n_heads, k):
        super().__init__(); self.layers=nn.ModuleList([kNNHyperbolicAttentionLayer(dim,n_heads,k) for _ in range(num_layers)])
    def forward(self, x, pos, c, time_emb):
        for layer in self.layers: x=layer(x,pos,c,time_emb)
        return x

class WuBuDiffusionModel(nn.Module):
    def __init__(self, args):
        super().__init__(); self.args = args; self.dim = args.hga_dim
        self.num_patches_h = args.image_h // args.hga_patch_size; self.num_patches_w = args.image_w // args.hga_patch_size
        self.patch_encoder = ImagePatchEncoder(args.hga_patch_size, args.num_channels, self.dim)
        if args.hga_learnable_geometry: self.initial_tangent_grid = nn.Parameter(self._create_tangent_grid())
        else: self.register_buffer('initial_tangent_grid', self._create_tangent_grid())
        if args.hga_learnable_curvature: self.log_c = nn.Parameter(torch.tensor(math.log(args.hga_poincare_c)))
        else: self.register_buffer('log_c', torch.tensor(math.log(args.hga_poincare_c)))
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(self.dim),nn.Linear(self.dim,self.dim*4),nn.GELU(),nn.Linear(self.dim*4,self.dim))
        self.down_blocks=nn.ModuleList(); self.up_blocks=nn.ModuleList(); self.down_projs=nn.ModuleList(); self.up_projs=nn.ModuleList(); self.merge_layers=nn.ModuleList()
        self.down_time_mlps = nn.ModuleList(); self.up_time_mlps = nn.ModuleList() # TIME EMBEDDING FIX
        current_dim=self.dim; dims=[]
        for _ in range(args.hga_unet_depth):
            self.down_blocks.append(HGA_UNet_Block(args.hga_num_layers_per_block,current_dim,args.hga_n_heads,args.hga_knn_k)); dims.append(current_dim)
            self.down_time_mlps.append(nn.Sequential(nn.GELU(), nn.Linear(self.dim, current_dim)))
            self.down_projs.append(HyperbolicPooling(current_dim)); current_dim*=2
        self.bottleneck = HGA_UNet_Block(args.hga_num_layers_per_block,current_dim,args.hga_n_heads,args.hga_knn_k)
        self.mid_time_mlp = nn.Sequential(nn.GELU(), nn.Linear(self.dim, current_dim))
        for _ in range(args.hga_unet_depth):
            self.up_projs.append(HyperbolicUnpooling(current_dim)); current_dim//=2
            self.up_time_mlps.append(nn.Sequential(nn.GELU(), nn.Linear(self.dim, current_dim)))
            self.merge_layers.append(nn.Linear(current_dim+dims.pop(),current_dim))
            self.up_blocks.append(HGA_UNet_Block(args.hga_num_layers_per_block,current_dim,args.hga_n_heads,args.hga_knn_k))
        self.final_proj = nn.Linear(self.dim, self.dim)
        self.patch_decoder = ImagePatchDecoder(args.hga_patch_size, args.num_channels, self.dim, self.num_patches_h, self.num_patches_w)
        self.apply(init_weights_general)
    def _create_tangent_grid(self):
        gh=torch.linspace(-.9,.9,self.num_patches_h); gw=torch.linspace(-.9,.9,self.num_patches_w)
        grid=torch.stack(torch.meshgrid(gh,gw,indexing='ij'),dim=-1).reshape(-1,2)
        if self.args.hga_pos_dim>2: grid=F.pad(grid,(0,self.args.hga_pos_dim-2))
        return grid
    def predict_noise_from_latent(self, x_t_latent, t):
        B=x_t_latent.shape[0]; c=torch.exp(self.log_c); positions=PoincareBall.expmap0(self.initial_tangent_grid,c)
        base_time_emb=self.time_mlp(t); x,pos_current=x_t_latent,positions.expand(B,-1,-1); skips=[]
        for i in range(self.args.hga_unet_depth):
            level_time_emb = self.down_time_mlps[i](base_time_emb)
            x = self.down_blocks[i](x, pos_current, c, level_time_emb)
            skips.append((x, pos_current)); x, pos_current = self.down_projs[i](x, pos_current, c)
        mid_time_emb = self.mid_time_mlp(base_time_emb)
        x = self.bottleneck(x, pos_current, c, mid_time_emb)
        for i in range(self.args.hga_unet_depth):
            x, pos_current_up = self.up_projs[i](x, pos_current, c)
            skip_x, skip_pos = skips.pop(); pos_current = skip_pos
            x=torch.cat([x,skip_x],dim=-1); x=self.merge_layers[i](x)
            level_time_emb = self.up_time_mlps[i](base_time_emb)
            x=self.up_blocks[i](x,pos_current,c,level_time_emb)
        return self.final_proj(x)
    def forward(self, x0_image, t, diffusion):
        x0_features=self.patch_encoder(x0_image); noise=torch.randn_like(x0_features)
        x_t_features=diffusion.q_sample(x0_features, t, noise); pred_noise=self.predict_noise_from_latent(x_t_features,t)
        return pred_noise, noise

# --- HAKMEM Q-Controller and AdamW Wrapper ---
class HAKMEMQController:
    def __init__(self,lr_scale_options=None,beta1_scale_options=None,beta2_scale_options=None,**kwargs):
        self.q_table={}; self.alpha=kwargs.get('q_lr',0.02); self.gamma=kwargs.get('q_discount',0.97); self.epsilon=kwargs.get('q_epsilon',0.15); self.min_epsilon=0.02; self.epsilon_decay=0.9995
        self.prev_loss=self.prev_state=self.prev_action=None; self.logger = logger_wubu_diffusion
        if lr_scale_options is None: lr_scale_options=[0.9,0.97,1.0,1.03,1.1]
        if beta1_scale_options is None: beta1_scale_options=[0.98,0.99,1.0,1.005,1.01]
        if beta2_scale_options is None: beta2_scale_options=[0.99,0.995,1.0,1.0005,1.001]
        self.action_ranges={'lr_scale':np.array(lr_scale_options),'beta1_scale':np.array(beta1_scale_options),'beta2_scale':np.array(beta2_scale_options)}
        self.loss_window=deque(maxlen=10);self.grad_norm_window=deque(maxlen=10);self.stable_steps=0
    def get_state(self,lr,beta1,beta2,grad_norm,loss):
        if loss is None or grad_norm is None: return None
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm)
        if len(self.loss_window)<3: return None
        try: loss_trend=np.polyfit(range(3),list(self.loss_window)[-3:],1)[0]/(np.mean(list(self.loss_window)[-3:])+1e-6)
        except np.linalg.LinAlgError: loss_trend = 0.0
        loss_trend_bin=np.digitize(loss_trend,bins=[-0.05,-0.005,0.005,0.05])
        grad_norm_bin=np.digitize(np.log10(np.mean(list(self.grad_norm_window))+1e-9),bins=[-4,-2,0,2])
        return (loss_trend_bin,grad_norm_bin)
    def compute_reward(self,current_loss,prev_loss,grad_norm):
        if current_loss is None or prev_loss is None: return 0.0
        reward=np.tanh((prev_loss-current_loss)/(prev_loss+1e-6)*5)
        if reward>0.1: self.stable_steps+=1; reward+=min(0.1,0.02*math.log1p(self.stable_steps))
        else: self.stable_steps=0
        return float(np.clip(reward,-1.0,1.0))
    def choose_action(self,state):
        if state is None: return None
        if state not in self.q_table: self.q_table[state]={p:np.zeros(len(s)) for p,s in self.action_ranges.items()}
        action={}; current_epsilon=max(self.min_epsilon,self.epsilon*(self.epsilon_decay**self.stable_steps))
        for param,space in self.action_ranges.items():
            if random.random()<current_epsilon: action[param]=float(np.random.choice(space))
            else: action[param]=float(space[np.argmax(self.q_table[state][param])])
        return action
    def update(self,state,action,reward,next_state):
        if state is None or next_state is None or action is None: return
        if next_state not in self.q_table: self.q_table[next_state]={p:np.zeros(len(s)) for p,s in self.action_ranges.items()}
        for param,val in action.items():
            action_idx=np.abs(self.action_ranges[param]-val).argmin()
            old_q=self.q_table[state][param][action_idx]
            future_q=np.max(self.q_table[next_state][param])
            self.q_table[state][param][action_idx]=old_q+self.alpha*(reward+self.gamma*future_q-old_q)
class AdamW_QControlled(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-2,q_learning_config={}):
        self.adam_optimizer=torch.optim.AdamW(params,lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(self.adam_optimizer.param_groups,self.adam_optimizer.defaults); self.state=self.adam_optimizer.state
        self.q_controller=HAKMEMQController(**q_learning_config); self.current_loss=None
    def zero_grad(self,set_to_none=False): self.adam_optimizer.zero_grad(set_to_none=set_to_none)
    def set_current_loss(self,loss): self.current_loss=loss
    @torch.no_grad()
    def step(self,closure=None):
        if self.current_loss is None: raise RuntimeError("Call set_current_loss() before step().")
        group=self.param_groups[0]; current_lr, (current_b1,current_b2) = group['lr'], group['betas']
        grad_norm=_get_average_grad_norm(self.param_groups)
        q_state=self.q_controller.get_state(current_lr,current_b1,current_b2,grad_norm,self.current_loss)
        if self.q_controller.prev_state and self.q_controller.prev_action and q_state:
            reward=self.q_controller.compute_reward(self.current_loss,self.q_controller.prev_loss,grad_norm)
            self.q_controller.update(self.q_controller.prev_state,self.q_controller.prev_action,reward,q_state)
        q_action=self.q_controller.choose_action(q_state)
        if q_action:
            for g in self.param_groups:
                base_lr=g['lr']/g.get('prev_lr_scale',1.0); g['lr']=np.clip(base_lr*q_action.get('lr_scale',1.0),1e-8,0.1); g['prev_lr_scale']=q_action.get('lr_scale',1.0)
                b1,b2=g['betas']; base_b1=b1/g.get('prev_b1_scale',1.0); base_b2=b2/g.get('prev_b2_scale',1.0)
                g['betas']=(np.clip(base_b1*q_action.get('beta1_scale',1.0),0.5,0.999),np.clip(base_b2*q_action.get('beta2_scale',1.0),0.9,0.9999))
                g['prev_b1_scale']=q_action.get('beta1_scale',1.0); g['prev_b2_scale']=q_action.get('beta2_scale',1.0)
        self.q_controller.prev_state=q_state; self.q_controller.prev_action=q_action; self.q_controller.prev_loss=self.current_loss
        return self.adam_optimizer.step(closure)
def _get_average_grad_norm(param_groups):
    total_norm_sq=0.0; num_params=0
    for group in param_groups:
        for p in group['params']:
            if p.grad is None: continue
            param_norm=p.grad.detach().data.norm(2); total_norm_sq+=param_norm.item()**2; num_params+=1
    return (total_norm_sq/num_params)**0.5 if num_params>0 else 0.0

# --- Trainer ---
class CPUDiffusionTrainer:
    def __init__(self,model,diffusion,args):
        self.model=model; self.diffusion=diffusion; self.args=args; self.device=torch.device("cpu"); self.logger=logger_wubu_diffusion.getChild("Trainer"); self.global_step=0
        if args.optimizer=='adamw': self.optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
        elif args.optimizer=='adamw_qcontrolled': self.optimizer=AdamW_QControlled(model.parameters(),lr=args.learning_rate,q_learning_config={'q_lr':args.q_lr,'q_epsilon':args.q_epsilon})
        else: raise ValueError(f"Unknown optimizer: {args.optimizer}")
    def _setup_data(self):
        video_path=self.args.video_file_path; self.train_ds=VideoFrameDatasetCPU(video_path,(self.args.image_h,self.args.image_w),val_fraction=self.args.val_fraction,mode='train'); self.train_loader=DataLoader(self.train_ds,batch_size=self.args.batch_size,shuffle=True,drop_last=True,num_workers=self.args.num_workers)
        if self.args.val_fraction>0: self.val_loader=DataLoader(VideoFrameDatasetCPU(video_path,(self.args.image_h,self.args.image_w),val_fraction=self.args.val_fraction,mode='val'),batch_size=self.args.val_batch_size,num_workers=self.args.num_workers)
    def train(self):
        self._setup_data()
        for epoch in range(self.args.epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}");
            for x0_img in pbar:
                self.optimizer.zero_grad(); x0_img=x0_img.to(self.device); t=torch.randint(0,self.diffusion.timesteps,(x0_img.shape[0],),device=self.device).long()
                pred_noise, target_noise = self.model(x0_img, t, self.diffusion)
                loss=F.mse_loss(pred_noise,target_noise); loss.backward()
                if isinstance(self.optimizer,AdamW_QControlled): self.optimizer.set_current_loss(loss.item())
                self.optimizer.step(); self.global_step+=1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
                if self.args.wandb and WANDB_AVAILABLE and self.global_step%self.args.log_interval==0:
                    log_data={"train/loss":loss.item()}; group=self.optimizer.param_groups[0]
                    log_data.update({"hparams/lr":group['lr'],"hparams/beta1":group['betas'][0],"hparams/beta2":group['betas'][1]})
                    wandb.log(log_data,step=self.global_step)
            if (epoch+1)%self.args.save_interval==0: self._save_checkpoint(epoch)
    def _save_checkpoint(self,epoch): os.makedirs(self.args.checkpoint_dir,exist_ok=True); torch.save({'model_state_dict':self.model.state_dict()},f"{self.args.checkpoint_dir}/hga_ep{epoch+1}.pt"); self.logger.info(f"Saved ckpt ep{epoch+1}")

def parse_args():
    p=argparse.ArgumentParser(description="HGA-UNet Diffusion Model"); p.add_argument('--video_data_path',type=str,default="vid_data"); p.add_argument('--checkpoint_dir',type=str,default='hga_checkpoints')
    p.add_argument('--image_h',type=int,default=64); p.add_argument('--image_w',type=int,default=64); p.add_argument('--num_channels',type=int,default=3); p.add_argument('--num_workers',type=int,default=0)
    p.add_argument('--epochs',type=int,default=200); p.add_argument('--batch_size',type=int,default=8); p.add_argument('--learning_rate',type=float,default=2e-4)
    p.add_argument('--diffusion_timesteps',type=int,default=1000); p.add_argument('--optimizer',type=str,default='adamw',choices=['adamw','adamw_qcontrolled'])
    p.add_argument('--hga_patch_size',type=int,default=8); p.add_argument('--hga_dim',type=int,default=256); p.add_argument('--hga_pos_dim',type=int,default=2)
    p.add_argument('--hga_poincare_c',type=float,default=1.0); p.add_argument('--hga_n_heads',type=int,default=8)
    p.add_argument('--hga_unet_depth',type=int,default=2); p.add_argument('--hga_num_layers_per_block',type=int,default=2)
    p.add_argument('--hga_knn_k',type=int,default=16); p.add_argument('--hga_learnable_geometry',action='store_true'); p.add_argument('--hga_learnable_curvature',action='store_true')
    p.add_argument('--q_lr',type=float,default=0.02); p.add_argument('--q_epsilon',type=float,default=0.15)
    p.add_argument('--wandb',action='store_true'); p.add_argument('--wandb_project',type=str,default='HGA-Diffusion-v4')
    p.add_argument('--log_interval',type=int,default=50); p.add_argument('--save_interval',type=int,default=10)
    p.add_argument('--val_fraction',type=float,default=0.1); p.add_argument('--val_batch_size',type=int,default=8)
    args=p.parse_args()
    if (args.image_h//args.hga_patch_size)%(2**args.hga_unet_depth)!=0 or (args.image_w//args.hga_patch_size)%(2**args.hga_unet_depth)!=0: raise ValueError("Num patches H/W must be divisible by 2**depth.")
    return args

def main():
    args=parse_args(); logger_wubu_diffusion.info(f"--- HGA-UNet v4.4 --- \nArgs: {vars(args)}")
    os.makedirs(args.checkpoint_dir,exist_ok=True); os.makedirs(args.video_data_path,exist_ok=True)
    video_files=[f for f in os.listdir(args.video_data_path) if f.endswith(('.mp4', '.mov', '.avi', '.mkv'))];
    if not video_files and IMAGEIO_AVAILABLE:
        print(f"No video found in {args.video_data_path}, creating dummy video.");
        try:
            imageio.mimsave(os.path.join(args.video_data_path, 'dummy.mp4'), [np.random.randint(0,255, (args.image_h, args.image_w, 3), 'u8') for _ in range(100)], fps=30)
            video_files = ['dummy.mp4']
        except Exception as e:
            logger_wubu_diffusion.error(f"Could not create dummy video: {e}")
            sys.exit(1)
    elif not video_files:
        logger_wubu_diffusion.error(f"No video found in {args.video_data_path} and imageio is not available.")
        sys.exit(1)
    args.video_file_path = os.path.join(args.video_data_path, video_files[0])
    if args.wandb and WANDB_AVAILABLE: wandb.init(project=args.wandb_project,config=args,name=f"run_{datetime.now().strftime('%y%m%d_%H%M')}")
    model=WuBuDiffusionModel(args); diffusion=DiffusionProcess(args.diffusion_timesteps); trainer=CPUDiffusionTrainer(model,diffusion,args)
    trainer.train()

if __name__ == "__main__":
    main()