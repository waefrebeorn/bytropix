# =================================================================================================
# ADVANCED ENTROPIC SAMPLER (DSlider) - Self-Contained Integration
# Source: Entropix, adapted for high-performance image generation.
# =================================================================================================
from dataclasses import dataclass, field, fields

# --- [THE DEFINITIVE SOLUTION] ---
# Explicit PyTree registration correctly separates dynamic JAX arrays from static, hashable Python values.

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DSThreshold:
    # Explicitly define fields in the order they will be unflattened
    bilinear: jnp.ndarray
    linear_state_ent: jnp.ndarray
    linear_state_std: jnp.ndarray
    weight: float
    bias: float
    linear_naked_ent: float
    linear_naked_varent: float

    def tree_flatten(self):
        # Children are JAX arrays (dynamic data)
        children = (self.bilinear, self.linear_state_ent, self.linear_state_std)
        # Aux_data are Python primitives (static data)
        aux_data = (self.weight, self.bias, self.linear_naked_ent, self.linear_naked_varent)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        bilinear, linear_state_ent, linear_state_std = children
        weight, bias, linear_naked_ent, linear_naked_varent = aux_data
        return cls(bilinear, linear_state_ent, linear_state_std, weight, bias, linear_naked_ent, linear_naked_varent)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DSConfig:
    # --- Dynamic Children (JAX arrays or other PyTrees) ---
    dirichlet_support: jnp.ndarray
    outlier_threshold: DSThreshold
    argmax_threshold: DSThreshold
    dirichlet_threshold: DSThreshold
    target_entropy: DSThreshold
    # --- Static Aux Data (Python primitives) ---
    outlier_topk: int
    noise_floor: float
    emwa_ent_naked_coeff: float
    emwa_varent_naked_coeff: float
    emwa_topk_ent_naked_coeff: float
    emwa_temp_coeff: float
    emwa_logp_base: float
    emwa_logp_exp_factor: float
    emwa_dir_ent_coeff: float
    emwa_ent_scaffold_coeff: float
    emwa_varent_scaffold_coeff: float
    token_cross_ent_naked_coeff: float
    token_cross_ent_scaffold_coeff: float
    token_cross_var_naked_coeff: float
    token_cross_var_scaffold_coeff: float
    perturb_base_coeff: float
    perturb_exp_coeff: float

    def tree_flatten(self):
        children = (self.dirichlet_support, self.outlier_threshold, self.argmax_threshold, self.dirichlet_threshold, self.target_entropy)
        aux_data = (self.outlier_topk, self.noise_floor, self.emwa_ent_naked_coeff, self.emwa_varent_naked_coeff, self.emwa_topk_ent_naked_coeff,
                    self.emwa_temp_coeff, self.emwa_logp_base, self.emwa_logp_exp_factor, self.emwa_dir_ent_coeff, self.emwa_ent_scaffold_coeff,
                    self.emwa_varent_scaffold_coeff, self.token_cross_ent_naked_coeff, self.token_cross_ent_scaffold_coeff,
                    self.token_cross_var_naked_coeff, self.token_cross_var_scaffold_coeff, self.perturb_base_coeff, self.perturb_exp_coeff)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (dirichlet_support, outlier_threshold, argmax_threshold, dirichlet_threshold, target_entropy) = children
        (outlier_topk, noise_floor, emwa_ent_naked_coeff, emwa_varent_naked_coeff, emwa_topk_ent_naked_coeff,
         emwa_temp_coeff, emwa_logp_base, emwa_logp_exp_factor, emwa_dir_ent_coeff, emwa_ent_scaffold_coeff,
         emwa_varent_scaffold_coeff, token_cross_ent_naked_coeff, token_cross_ent_scaffold_coeff,
         token_cross_var_naked_coeff, token_cross_var_scaffold_coeff, perturb_base_coeff, perturb_exp_coeff) = aux_data
        return cls(dirichlet_support, outlier_threshold, argmax_threshold, dirichlet_threshold, target_entropy,
                   outlier_topk, noise_floor, emwa_ent_naked_coeff, emwa_varent_naked_coeff, emwa_topk_ent_naked_coeff,
                   emwa_temp_coeff, emwa_logp_base, emwa_logp_exp_factor, emwa_dir_ent_coeff, emwa_ent_scaffold_coeff,
                   emwa_varent_scaffold_coeff, token_cross_ent_naked_coeff, token_cross_ent_scaffold_coeff,
                   token_cross_var_naked_coeff, token_cross_var_scaffold_coeff, perturb_base_coeff, perturb_exp_coeff)


def DEFAULT_DS_CONFIG():
    return DSConfig(
        outlier_topk=16,
        dirichlet_support=jnp.arange(1, 257),
        noise_floor=-18.42068,
        emwa_ent_naked_coeff=0.01,
        emwa_varent_naked_coeff=0.01,
        emwa_topk_ent_naked_coeff=0.01,
        emwa_temp_coeff=0.01,
        emwa_logp_base=2.0,
        emwa_logp_exp_factor=1.0,
        emwa_dir_ent_coeff=0.01,
        emwa_ent_scaffold_coeff=0.01,
        emwa_varent_scaffold_coeff=0.01,
        token_cross_ent_naked_coeff=0.01,
        token_cross_ent_scaffold_coeff=0.01,
        token_cross_var_naked_coeff=0.01,
        token_cross_var_scaffold_coeff=0.01,
        perturb_base_coeff=0.5,
        perturb_exp_coeff=0.1,
        outlier_threshold=DSThreshold(bilinear=jnp.zeros((4,4)), linear_state_ent=jnp.zeros(4), linear_state_std=jnp.zeros(4), weight=1.0, bias=0.5, linear_naked_ent=0.0, linear_naked_varent=0.0),
        argmax_threshold=DSThreshold(bilinear=jnp.zeros((4,4)), linear_state_ent=jnp.zeros(4), linear_state_std=jnp.zeros(4), weight=1.0, bias=-0.5, linear_naked_ent=0.0, linear_naked_varent=0.0),
        dirichlet_threshold=DSThreshold(bilinear=jnp.zeros((4,4)), linear_state_ent=jnp.zeros(4), linear_state_std=jnp.zeros(4), weight=1.0, bias=-0.5, linear_naked_ent=0.0, linear_naked_varent=0.0),
        target_entropy=DSThreshold(bilinear=jnp.zeros((4,4)), linear_state_ent=jnp.array([0., 0., 0., 1.0]), linear_state_std=jnp.zeros(4), weight=0.0, bias=0.5, linear_naked_ent=0.0, linear_naked_varent=0.0)
    )

@dataclass
class SamplerLogicConfig:
  low_naked_entropy_threshold = 0.3
  high_naked_entropy_threshold = 2.5
  low_naked_varentropy_threshold = 1.2
  high_naked_varentropy_threshold = 2.5

# --- Core State and Math Kernels (unchanged) ---
EPS = 1e-8
MIN_TEMP = 0.1
MAX_TEMP = 10.0

class DSState(NamedTuple):
  emwa_dir: jnp.ndarray; emwa_logp_on_supp: jnp.ndarray; emwa_temp: jnp.ndarray
  emwa_ent_scaffold: jnp.ndarray; emwa_ent_naked: jnp.ndarray; emwa_varent_scaffold: jnp.ndarray
  emwa_varent_naked: jnp.ndarray; token_cross_ent_scaffold: jnp.ndarray
  token_cross_ent_naked: jnp.ndarray; token_cross_var_scaffold: jnp.ndarray
  token_cross_var_naked: jnp.ndarray; emwa_dir_ent: jnp.ndarray; emwa_topk_ent_naked: jnp.ndarray

@jax.jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  p = jnp.exp(logp); ent = -jnp.sum(p * logp, axis=-1)
  diff = logp + ent[..., None]; varent = jnp.sum(p * diff**2, axis=-1)
  return ent, varent

@jax.jit
def normalize_logits(logits: jnp.ndarray, noise_floor: float) -> jnp.ndarray:
  shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
  normalized = shifted - jax.nn.logsumexp(shifted + EPS, axis=-1, keepdims=True)
  return jnp.where(normalized < noise_floor, jnp.log(EPS), normalized)

@jax.jit
def dirichlet_log_likelihood_from_logprob(logprobs: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
  return (jnp.sum((alpha - 1.0) * logprobs, axis=-1) - jax.scipy.special.gammaln(jnp.sum(alpha, axis=-1)) + jnp.sum(jax.scipy.special.gammaln(alpha), axis=-1))

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def fit_dirichlet(target_values, init_alpha=None, initial_lr=1.2, decay_alpha=0.1, decay_beta=2.0, decay_gamma=0.25, decay_nu=0.75, max_iters=140, tol=1e-4, dtype: jnp.dtype = jnp.bfloat16):
  batch_shape=target_values.shape[:-1]; n=target_values.shape[-1]; min_lr=1e-8
  target_values=target_values.astype(jnp.float32)
  if init_alpha is None: init_alpha=jnp.ones((*batch_shape, n), dtype=jnp.float32)
  def halley_update(alpha,target_values):
    p1=jax.scipy.special.polygamma(1,alpha);p2=jax.scipy.special.polygamma(2,alpha);S=jnp.sum(alpha,axis=-1,keepdims=True);s1=jax.scipy.special.polygamma(1,S);s2=jax.scipy.special.polygamma(2,S);p1_inv=1./p1;sum_p1_inv=jnp.sum(p1_inv,axis=-1,keepdims=True);denom=jnp.where(jnp.abs(1.-s1*sum_p1_inv)<1e-12,1e-12,1.-s1*sum_p1_inv);coeff=s1/denom;error=jax.scipy.special.digamma(alpha)-jax.scipy.special.digamma(S)-target_values;temp=p1_inv*error;sum_temp=jnp.sum(temp,axis=-1,keepdims=True);J_inv_error=temp+coeff*sum_temp*p1_inv;sum_J_inv_error=jnp.sum(J_inv_error,axis=-1,keepdims=True);H_J_inv_error=p2*J_inv_error-s2*sum_J_inv_error;temp2=p1_inv*H_J_inv_error;sum_temp2=jnp.sum(temp2,axis=-1,keepdims=True);J_inv_H_J_inv_error=temp2+coeff*sum_temp2*p1_inv
    return -J_inv_error+.5*J_inv_H_J_inv_error
  def scan_body(carry, _):
    alpha,converged,error_norm,step=carry;S=jnp.sum(alpha,axis=-1,keepdims=True);error=jax.scipy.special.digamma(alpha)-jax.scipy.special.digamma(S)-target_values;error_norm=jnp.linalg.norm(error,axis=-1);new_converged=converged|(error_norm<tol);lr=jnp.maximum(initial_lr*jnp.exp(-decay_alpha*(step**decay_nu))*jnp.abs(jnp.cos(decay_beta/(step**decay_gamma))),min_lr);delta_alpha=jnp.clip(lr[...,None]*halley_update(alpha,target_values),-.5*alpha,.5*alpha);new_alpha=jnp.where(new_converged[...,None],alpha,jnp.maximum(alpha+delta_alpha,alpha/2))
    return (new_alpha,new_converged,error_norm,step+1),None
  init_state=(init_alpha,jnp.zeros(batch_shape,dtype=jnp.bool_),jnp.full(batch_shape,jnp.inf),jnp.ones(batch_shape,dtype=jnp.int32));(final_alpha,final_converged,_,final_step),_=jax.lax.scan(scan_body,init_state,None,length=max_iters)
  return final_alpha.astype(dtype),final_step-1,final_converged

@partial(jax.jit, static_argnames=("bsz", "dtype"))
def initialize_state(logits: jax.Array, bsz: int, config: DSConfig, dtype=jnp.bfloat16) -> DSState:
    # --- [THE DEFINITIVE FIX] ---
    # The initial logits from the model have a sequence length dimension (L=1),
    # e.g., shape (B, 1, V). This extra dimension corrupts the shape of all
    # initial state metrics (e.g., entropy becomes (B, 1) instead of (B,)).
    # We must squeeze this dimension out *before* any calculations.
    if logits.ndim == 3:
        logits = logits.squeeze(1)
    # Now logits has the correct shape (B, V).

    logprobs = normalize_logits(logits, config.noise_floor)
    ent, varent = ent_varent(logprobs) # ent and varent will now have the correct shape (B,)

    topk_logits, topk_indices = jax.lax.top_k(logprobs, config.outlier_topk)
    topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
    topk_ent, _ = ent_varent(topk_logprobs)
    logprobs_on_supp = normalize_logits(logits[..., config.dirichlet_support], config.noise_floor)
    initial_dir, _, _ = fit_dirichlet(jnp.mean(logprobs_on_supp, axis=0, keepdims=True))
    avg_dir_ent = dirichlet_log_likelihood_from_logprob(logprobs_on_supp, initial_dir).mean()
    topk_token_logprobs = jnp.take_along_axis(logprobs, topk_indices, axis=-1)
    
    # All metrics are now correctly shaped, so the initial state will be correct.
    single_state = DSState(
        emwa_dir=initial_dir, 
        emwa_logp_on_supp=jnp.mean(logprobs_on_supp, axis=0, keepdims=True), 
        emwa_temp=jnp.ones((1,), dtype=dtype), 
        emwa_ent_scaffold=ent, 
        emwa_ent_naked=ent, 
        emwa_varent_scaffold=jnp.zeros((1,), dtype=dtype), 
        emwa_varent_naked=varent, 
        token_cross_ent_scaffold=ent, 
        token_cross_ent_naked=-topk_token_logprobs.mean(), 
        token_cross_var_scaffold=jnp.zeros((1,), dtype=dtype), 
        token_cross_var_naked=topk_token_logprobs.var(), 
        emwa_dir_ent=avg_dir_ent, 
        emwa_topk_ent_naked=topk_ent
    )
    return jax.tree_util.tree_map(lambda x: x.repeat(bsz, axis=0), single_state)

@jax.jit
def update_emwa(new: jax.Array, old: jax.Array, coeff: float | jax.Array) -> jax.Array:
  return coeff * new + (1 - coeff) * old

@jax.jit
def adaptive_dirichlet_step(key: jax.random.PRNGKey, state: DSState, logits: jnp.ndarray, config: DSConfig):
    dtype = logits.dtype; bsz, vsz = logits.shape; output_tokens = jnp.zeros(bsz, dtype=jnp.int32)
    naked_log_probs = normalize_logits(logits, config.noise_floor)
    naked_ent, naked_varent = ent_varent(naked_log_probs)
    new_emwa_ent_naked = update_emwa(naked_ent, state.emwa_ent_naked, config.emwa_ent_naked_coeff)
    new_emwa_varent_naked = update_emwa(naked_varent, state.emwa_varent_naked, config.emwa_varent_naked_coeff)
    topk_logits, topk_indices = jax.lax.top_k(naked_log_probs, config.outlier_topk)
    topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
    naked_topk_ent, _ = ent_varent(topk_logprobs)
    new_emwa_topk_ent_naked = update_emwa(naked_topk_ent, state.emwa_topk_ent_naked, config.emwa_topk_ent_naked_coeff)
    argmax_threshold = config.argmax_threshold.weight * state.emwa_topk_ent_naked + config.argmax_threshold.bias
    argmax_mask = (naked_topk_ent < argmax_threshold)
    argmax_indices = jnp.argmax(topk_logprobs, axis=-1)
    argmax_tokens = jnp.take_along_axis(topk_indices, argmax_indices[:, None], axis=-1).squeeze(1)
    output_tokens = jnp.where(argmax_mask, argmax_tokens, output_tokens)
    inlier_sampling_mask = ~argmax_mask
    inlier_sampling_temp = jnp.ones_like(state.emwa_temp)
    inlier_choices = jax.random.categorical(key, topk_logprobs / inlier_sampling_temp[:, None])
    inlier_tokens = jnp.take_along_axis(topk_indices, inlier_choices[:, None], axis=-1).squeeze(1)
    output_tokens = jnp.where(inlier_sampling_mask, inlier_tokens, output_tokens)
    scaffold_ent, scaffold_varent = naked_ent, naked_varent
    naked_token_logprob = jnp.take_along_axis(naked_log_probs, output_tokens[:,None], axis=-1).squeeze(-1)
    scaffold_token_logprob = naked_token_logprob
    new_state = state._replace(emwa_ent_naked=new_emwa_ent_naked, emwa_varent_naked=new_emwa_varent_naked, emwa_topk_ent_naked=new_emwa_topk_ent_naked)
    return new_state, output_tokens, naked_ent, naked_varent, scaffold_ent, scaffold_varent, naked_token_logprob, scaffold_token_logprob

@jax.jit
def dslider_sampler_step(key: jax.random.PRNGKey, state: DSState, logits: jnp.ndarray, config: DSConfig):
  cfg = SamplerLogicConfig()
  main_key, resample_key = jax.random.split(key)

  # --- Step 1: Propose initial tokens for all items in the batch ---
  (proposed_state, proposed_token, naked_ent, naked_varent, *_) = adaptive_dirichlet_step(main_key, state, logits, config)

  # --- Step 2: Identify which items need resampling (High Entropy High Variance) ---
  is_hehv = (naked_ent > cfg.high_naked_entropy_threshold) & (naked_varent > cfg.high_naked_varentropy_threshold)

  # --- Step 3: *Unconditionally* perform resampling for all items ---
  # This is more JIT-friendly than lax.cond with a non-static predicate.
  # The .at[] operation correctly handles batching by creating a new masked array.
  masked_logits = logits.at[jnp.arange(logits.shape[0]), proposed_token].set(-1e9)
  (resampled_state, resampled_token, *_) = adaptive_dirichlet_step(resample_key, proposed_state, masked_logits, config)

  # --- Step 4: Selectively combine the results using the `is_hehv` mask ---
  # jnp.where correctly handles selecting between the two pre-computed results.
  final_token = jnp.where(is_hehv, resampled_token, proposed_token)

  # For the state, we must also use `where`. We broadcast the 1D `is_hehv` mask 
  # to match the shape of each leaf in the state PyTree.
  final_state = jax.tree_util.tree_map(
      lambda original, resampled: jnp.where(is_hehv.reshape(-1, *([1] * (original.ndim - 1))), resampled, original),
      proposed_state,
      resampled_state
  )

  return final_token, final_state

