

from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from entropix.dslider import DSState, adaptive_dirichlet_step, initialize_state
from entropix.dslider_config import DSConfig, DEFAULT_DS_CONFIG


MAX_K = 256
LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


@dataclass
class SamplerConfig:
  # Naked (logits) entropy thresholds
  low_naked_entropy_threshold = 0.3  # Captures most observed LELV cases
  medium_naked_entropy_threshold = 1.2  # Separates medium from high entropy cases
  high_naked_entropy_threshold = 2.5  # Above this we see clear high entropy cases

  # Naked (logits) varentropy thresholds
  low_naked_varentropy_threshold = 1.2  # Most LELV cases are below this
  high_naked_varentropy_threshold = 2.5  # Clear separation for high variance cases

  # Scaffold (attention) metrics thresholds
  # These don't appear in logs, keeping unchanged
  low_scaffold_entropy_threshold = 1.0
  high_scaffold_entropy_threshold = 2.0
  low_scaffold_varentropy_threshold = 0.3
  high_scaffold_varentropy_threshold = 0.8


@partial(jax.jit, static_argnames=("config",))
def sample(
  state: DSState,
  logits: jnp.ndarray,
  config: DSConfig,
  clarifying_question_token: int = 2564,
  key=jax.random.PRNGKey(1337),
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
  cfg = SamplerConfig()
  bsz = logits.shape[0]
  (
    new_state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
    naked_token_logprob,
    scaffold_token_logprob,
  ) = adaptive_dirichlet_step(key, state, logits, config)
  new_token = new_token.reshape((bsz, 1))

  def _and(*args):
    res = True
    for a in args:
      res = jax.lax.bitwise_and(res, a)
    return res

  def sample_one(
    idx,
    logit,
    state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
    loops=0,
  ):
    LELV = _and(
      naked_ent < cfg.low_naked_entropy_threshold,
      naked_varent < cfg.low_naked_varentropy_threshold,
      # scaffold_ent < cfg.low_scaffold_entropy_threshold,
      # scaffold_varent < cfg.low_scaffold_varentropy_threshold,
    ).astype(float)

    HELV = _and(
      naked_ent > cfg.high_naked_entropy_threshold,
      naked_varent < cfg.low_naked_varentropy_threshold,
      # scaffold_ent < cfg.low_scaffold_entropy_threshold,
      # scaffold_varent < cfg.low_scaffold_varentropy_threshold,
    ).astype(float)

    LEHV = _and(
      naked_ent < cfg.high_naked_entropy_threshold,
      naked_varent > cfg.high_naked_varentropy_threshold,
      # scaffold_ent < cfg.low_scaffold_entropy_threshold,
      # scaffold_varent > cfg.high_scaffold_varentropy_threshold,
    ).astype(float)

    HEHV = _and(
      naked_ent > cfg.medium_naked_entropy_threshold,
      naked_varent > cfg.high_naked_varentropy_threshold,
      # scaffold_ent > cfg.high_scaffold_entropy_threshold,
      # scaffold_varent > cfg.high_scaffold_varentropy_threshold,
    ).astype(float)

    case = jnp.argmax(jnp.hstack([LELV, HELV, LEHV, HEHV]))

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    def lelv():
      # jax.debug.print("LELV Naked Ent: {}", naked_ent)
      # jax.debug.print("LELV Naked Varent: {}", naked_varent)
      # jax.debug.print("LELV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("LELV Scaffold Varent: {}\n", scaffold_varent)
      return new_token, state

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    def helv():
      # jax.debug.print("HELV Naked Ent: {}", naked_ent)
      # jax.debug.print("HELV Naked Varent: {}", naked_varent)
      # jax.debug.print("HELV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("HELV Scaffold Varent: {}\n", scaffold_varent)
      return jnp.array([2564]), state

    # Low Entropy, High Varentropy: "exploring forks in the path"
    def lehv():
      # jax.debug.print("LEHV Naked Ent: {}", naked_ent)
      # jax.debug.print("LEHV Naked Varent: {}", naked_varent)
      # jax.debug.print("LEHV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("LEHV Scaffold Varent: {}\n", scaffold_varent)
      # TODO(xjdr): We need to do a differnt version of tree search here with constant return dimensions
      return new_token, state

    # High Entropy, High Varentropy: "resampling in the mist"
    def hehv():
      # jax.debug.print("HEHV Naked Ent: {}", naked_ent)
      # jax.debug.print("HEHV Naked Varent: {}", naked_varent)
      # jax.debug.print("HEHV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("HEHV Scaffold Varent: {}\n", scaffold_varent)
      plogit = logit.at[new_token].set(float("-inf"))

      # Run ADS with single batch
      (
        new_state,
        resampled_token,
        *_,  # Other metrics
      ) = adaptive_dirichlet_step(
        key,
        jax.tree_map(lambda x: x[None, ...], state),
        plogit[None, ...],  # Shape (1, vocab)
        DEFAULT_DS_CONFIG,
      )
      return resampled_token, jax.tree_map(lambda x: jnp.bfloat16(x[-1]), new_state)

    def default():
      # jax.debug.print("Default Naked Ent: {}", naked_ent)
      # jax.debug.print("Default Naked Varent: {}", naked_varent)
      return new_token, state

    return jax.lax.switch(case, (lelv, helv, lehv, hehv, default))

  result, new_state = jax.vmap(sample_one)(
    jnp.arange(bsz),
    logits,
    state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
  )
  return result.reshape((bsz, 1)), new_state
  
  from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from entropix.dslider_config import EPS, MAX_TEMP, MIN_TEMP, DSConfig
from entropix.dslider_utils import *


@jax.jit
def kl_divergence(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
  """Compute KL divergence between two log probability distributions."""
  p = jnp.exp(logp)
  return jnp.sum(jnp.where(p > 0, p * (logp - logq), 0.0), axis=-1)


@jax.jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute entropy and varentropy from log probabilities."""
  p = jnp.exp(logp)
  ent = -jnp.sum(p * logp, axis=-1)
  diff = logp + ent[..., None]
  varent = jnp.sum(p * diff**2, axis=-1)
  return ent, varent


@jax.jit
def normalize_logits(logits: jnp.ndarray, noise_floor: float) -> jnp.ndarray:
  """Normalize logits to log probabilities with noise floor truncation."""
  shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
  normalized = shifted - jax.nn.logsumexp(shifted + EPS, axis=-1, keepdims=True)
  # noise floor calculated for bfloat16
  return jnp.where(normalized < noise_floor, jnp.log(EPS), normalized)


class DSState(NamedTuple):
  emwa_dir: jnp.ndarray
  emwa_logp_on_supp: jnp.ndarray
  emwa_temp: jnp.ndarray
  emwa_ent_scaffold: jnp.ndarray
  emwa_ent_naked: jnp.ndarray
  emwa_varent_scaffold: jnp.ndarray
  emwa_varent_naked: jnp.ndarray
  token_cross_ent_scaffold: jnp.ndarray
  token_cross_ent_naked: jnp.ndarray
  token_cross_var_scaffold: jnp.ndarray
  token_cross_var_naked: jnp.ndarray
  emwa_dir_ent: jnp.ndarray
  emwa_topk_ent_naked: jnp.ndarray


@partial(jax.jit, static_argnames=("bsz", "config", "dtype"))
def initialize_state(
  logits: jax.Array, bsz: int, config: DSConfig, dtype=jnp.bfloat16
) -> DSState:
  _, seqlen, _ = logits.shape
  logprobs = normalize_logits(logits, config.noise_floor)
  ent, varent = ent_varent(logprobs)
  avg_ent, avg_varent = ent.mean(axis=-1), varent.mean(axis=-1)

  topk_logits, topk_indices = jax.lax.top_k(logprobs, config.outlier_topk)
  topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
  topk_ent, _ = ent_varent(topk_logprobs)
  avg_topk_ent = topk_ent.mean(axis=-1)

  logprobs_on_supp = normalize_logits(
    logits[..., config.dirichlet_support], config.noise_floor
  )
  avg_logprobs_on_supp = jnp.mean(logprobs_on_supp, axis=1)

  initial_dir, _, _ = fit_dirichlet(avg_logprobs_on_supp)
  avg_dir_ent = dirichlet_log_likelihood_from_logprob(
    logprobs_on_supp, initial_dir[:, None, :]
  ).mean(axis=-1)

  topk_token_logprobs = jnp.take_along_axis(logprobs, topk_indices, axis=-1)
  initial_cross_ent_naked = -topk_token_logprobs.mean(axis=(1, 2))
  initial_cross_var_naked = topk_token_logprobs.var(axis=(1, 2))

  state = DSState(
    emwa_dir=initial_dir.repeat(bsz, axis=0),
    emwa_logp_on_supp=avg_logprobs_on_supp.repeat(bsz, axis=0),
    emwa_temp=jnp.ones((bsz,), dtype=dtype),
    emwa_ent_scaffold=avg_ent.repeat(bsz, axis=0),
    emwa_ent_naked=avg_ent.repeat(bsz, axis=0),
    emwa_varent_scaffold=jnp.zeros((bsz,), dtype=dtype),
    emwa_varent_naked=avg_varent.repeat(bsz, axis=0),
    token_cross_ent_scaffold=avg_ent.repeat(bsz, axis=0),
    token_cross_ent_naked=initial_cross_ent_naked.repeat(bsz, axis=0),
    token_cross_var_scaffold=jnp.zeros((bsz,), dtype=dtype),
    token_cross_var_naked=initial_cross_var_naked.repeat(bsz, axis=0),
    emwa_dir_ent=avg_dir_ent.repeat(bsz, axis=0),
    emwa_topk_ent_naked=avg_topk_ent.repeat(bsz, axis=0),
  )
  return state


@partial(jax.jit, static_argnames=("config", "wild"))
def adaptive_dirichlet_step(
  key: jax.random.PRNGKey,
  state: DSState,
  logits: jnp.ndarray,
  config: DSConfig,
  wild: bool = True,
) -> Tuple[DSState, jnp.ndarray]:
  dtype = logits.dtype
  bsz, vsz = logits.shape
  output_tokens = jnp.zeros(bsz, dtype=jnp.int32)
  EPS = jnp.array(1e-8, dtype=dtype)
  naked_log_probs = normalize_logits(logits, config.noise_floor)
  # update naked entropy rate
  naked_ent, naked_varent = ent_varent(naked_log_probs)
  # fix shape issue!
  new_emwa_ent_naked = update_emwa(
    naked_ent, state.emwa_ent_naked, config.emwa_ent_naked_coeff
  )
  new_emwa_varent_naked = update_emwa(
    naked_varent, state.emwa_varent_naked, config.emwa_varent_naked_coeff
  )
  # entropy and varentropy vectors - shape (bsz, 4)
  state_ent = jnp.array(
    [
      state.token_cross_ent_scaffold,
      state.token_cross_ent_naked,
      state.emwa_ent_scaffold,
      state.emwa_ent_naked,
    ]
  ).T  # TODO(doomslide): add dirichlet expected entropy...
  state_std = jnp.sqrt(
    jnp.array(
      [
        state.token_cross_var_scaffold,
        state.token_cross_var_naked,
        state.emwa_varent_scaffold,
        state.emwa_varent_naked,
      ]
    )
  ).T  # TODO(doomslide): add dirichlet expected std...
  outlier_threshold = compute_outlier_threshold(
    state_ent, state_std, naked_ent, naked_varent, config
  )
  outlier_mask = outlier_threshold > 0
  # update emwa topk entropy
  topk_logits, topk_indices = jax.lax.top_k(naked_log_probs, config.outlier_topk)
  topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
  naked_topk_ent, _ = ent_varent(topk_logprobs)
  new_emwa_topk_ent_naked = update_emwa(
    naked_topk_ent, state.emwa_topk_ent_naked, config.emwa_topk_ent_naked_coeff
  )
  """
  argmax policy for concentrated inliers
  """
  argmax_threshold = (
    config.argmax_threshold.weight * state.emwa_topk_ent_naked
    + config.argmax_threshold.bias
  )
  argmax_mask = ~outlier_mask & (naked_topk_ent < argmax_threshold)
  argmax_indices = jnp.argmax(topk_logprobs, axis=-1)
  argmax_tokens = jnp.take_along_axis(
    topk_indices, argmax_indices[:, None], axis=-1
  ).squeeze(1)
  output_tokens = jnp.where(argmax_mask, argmax_tokens, output_tokens)
  """
  topk temperature tuning policy for dispersed inliers
  """
  inlier_sampling_indices = ~outlier_mask & ~argmax_mask
  inlier_sampling_temp, _, _ = temp_tune(topk_logprobs, state.emwa_topk_ent_naked)
  sampling_inlier_choices = jax.random.categorical(
    key, topk_logprobs / inlier_sampling_temp[:, None]
  )
  sampling_inlier_tokens = jnp.take_along_axis(
    topk_indices, sampling_inlier_choices[:, None], axis=-1
  ).squeeze(1)
  output_tokens = jnp.where(
    inlier_sampling_indices, sampling_inlier_tokens, output_tokens
  )
  """
  tune temperature of outliers to match target entropy
  """
  target_entropy = (
    jnp.dot(state_ent, config.target_entropy.linear)
    + jnp.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, axis=-1)
    + config.target_entropy.bias
  )
  temp, _, _ = temp_tune(naked_log_probs.astype(jnp.float32), target_entropy)
  new_emwa_temp = update_emwa(temp, state.emwa_temp, config.emwa_temp_coeff)
  tuned_logprobs = normalize_logits(
    naked_log_probs / jnp.clip(temp[:, None], MIN_TEMP, MAX_TEMP), config.noise_floor
  )
  """
  update emwa logp (on dirichlet support)
  """
  logprobs_on_supp = normalize_logits(
    tuned_logprobs[:, config.dirichlet_support], config.noise_floor
  )
  kl = jnp.sum(
    jnp.exp(logprobs_on_supp) * (logprobs_on_supp - state.emwa_logp_on_supp), axis=-1
  )
  emwa_logp_coeff = config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
  new_emwa_logp_on_supp = update_emwa(
    logprobs_on_supp, state.emwa_logp_on_supp, emwa_logp_coeff[..., None]
  )
  new_emwa_dir, _, _ = fit_dirichlet(new_emwa_logp_on_supp)
  """
  update dirichlet and compute threshold
  """
  dir_log_likelihood = dirichlet_log_likelihood_from_logprob(
    logprobs_on_supp, state.emwa_dir
  )
  new_emwa_dir_ent = update_emwa(
    -dir_log_likelihood, state.emwa_dir_ent, config.emwa_dir_ent_coeff
  )
  dirichlet_threshold = (
    config.dirichlet_threshold.weight * state.emwa_dir_ent
    + config.dirichlet_threshold.bias
  )
  use_dirichlet = outlier_mask & (-dir_log_likelihood < dirichlet_threshold)
  if wild:  # if wild, sample from dirichlet, else use expectation
    dir_probs = sample_dirichlet(key, new_emwa_dir)
  else:
    dir_probs = dirichlet_expectation(new_emwa_dir)
  """
  below dirichlet threshold, interpolate and sample (can improve this in the future)
  """
  kl = jnp.sum(dir_probs * (jnp.log(dir_probs + EPS) - logprobs_on_supp), axis=-1)
  perturb_coeff = 1 - jnp.pow(
    config.perturb_base_coeff, -config.perturb_exp_coeff * (1 / (kl + EPS))
  )
  interpolated_probs = perturb_coeff[:, None] * dir_probs + (
    1 - perturb_coeff[:, None]
  ) * jnp.exp(logprobs_on_supp)
  # in use_dirichlet case take argmax of the slided probs
  dicihlet_choices = jnp.argmax(interpolated_probs, axis=-1)
  dirichlet_tokens = jnp.take(config.dirichlet_support, dicihlet_choices)
  output_tokens = jnp.where(use_dirichlet, dirichlet_tokens, output_tokens)
  """
  above dirichlet threshold youre ngmi
  """
  ood_choices = jax.random.categorical(key, jnp.log(dir_probs + EPS))
  ood_tokens = jnp.take(config.dirichlet_support, ood_choices)
  output_tokens = jnp.where(outlier_mask & ~use_dirichlet, ood_tokens, output_tokens)
  # update scaffold entropy rate
  scaffold_ent, scaffold_varent = ent_varent(jnp.log(interpolated_probs + EPS))
  new_emwa_ent_scaffold = update_emwa(
    scaffold_ent, state.emwa_ent_scaffold, config.emwa_ent_scaffold_coeff
  )
  new_emwa_varent_scaffold = update_emwa(
    scaffold_varent, state.emwa_varent_scaffold, config.emwa_varent_scaffold_coeff
  )
  # update token cross entropies
  batch_indices = jnp.arange(bsz)
  scaffold_token_logprob = jnp.log(
    interpolated_probs[batch_indices, output_tokens] + EPS
  )
  naked_token_logprob = jnp.log(naked_log_probs[batch_indices, output_tokens] + EPS)
  (
    new_token_cross_ent_scaffold,
    new_token_cross_ent_naked,
    new_token_cross_var_scaffold,
    new_token_cross_var_naked,
  ) = update_token_cross_entropies(
    state, scaffold_token_logprob, naked_token_logprob, config
  )
  # assemble new state
  new_state = DSState(
    emwa_dir=new_emwa_dir,
    emwa_logp_on_supp=new_emwa_logp_on_supp,
    emwa_temp=new_emwa_temp,
    emwa_ent_scaffold=new_emwa_ent_scaffold,
    emwa_ent_naked=new_emwa_ent_naked,
    emwa_varent_scaffold=new_emwa_varent_scaffold,
    emwa_varent_naked=new_emwa_varent_naked,
    token_cross_ent_scaffold=new_token_cross_ent_scaffold,
    token_cross_ent_naked=new_token_cross_ent_naked,
    token_cross_var_scaffold=new_token_cross_var_scaffold,
    token_cross_var_naked=new_token_cross_var_naked,
    emwa_dir_ent=new_emwa_dir_ent,
    emwa_topk_ent_naked=new_emwa_topk_ent_naked,
  )
  return (
    new_state,
    output_tokens,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
    naked_token_logprob,
    scaffold_token_logprob,
  )


@jax.jit
def update_emwa(new: jax.Array, old: jax.Array, coeff: float | jax.Array) -> jax.Array:
  return coeff * new + (1 - coeff) * old


@partial(jax.jit, static_argnames=("config",))
def compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config):
  return (
    jnp.einsum("bi,ij,bj->b", state_ent, config.outlier_threshold.bilinear, state_std)
    + jnp.einsum("bi,i->b", state_ent, config.outlier_threshold.linear_state_ent)
    + jnp.einsum("bi,i->b", state_std, config.outlier_threshold.linear_state_std)
    + naked_ent * config.outlier_threshold.linear_naked_ent
    + naked_varent * config.outlier_threshold.linear_naked_varent
    + config.outlier_threshold.bias
  )


@partial(jax.jit, static_argnames=("config",))
def update_dirichlet_params(tuned_logprobs_on_supp, state, config):
  kl = kl_divergence(tuned_logprobs_on_supp, state.emwa_logp_on_supp)
  emwa_logp_coeff = (
    config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
  )[:, None]
  new_emwa_logp_dir_sup = (
    emwa_logp_coeff * tuned_logprobs_on_supp
    + (1 - emwa_logp_coeff) * state.emwa_logp_on_supp
  )
  new_dir_params, _, _ = fit_dirichlet(new_emwa_logp_dir_sup)
  return new_dir_params, new_emwa_logp_dir_sup


@jax.jit
def update_token_cross_entropies(
  state: DSState,
  scaffold_token_logprob: jnp.ndarray,
  naked_token_logprob: jnp.ndarray,
  config: DSConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Update token cross entropy statistics."""
  token_cross_ent_naked = (
    config.token_cross_ent_naked_coeff * (-naked_token_logprob)
    + (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
  )
  token_cross_ent_scaffold = (
    config.token_cross_ent_scaffold_coeff * (-scaffold_token_logprob)
    + (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
  )
  token_cross_var_naked = (
    config.token_cross_var_naked_coeff
    * (token_cross_ent_naked - naked_token_logprob) ** 2
    + (1 - config.token_cross_var_naked_coeff) * state.token_cross_var_naked
  )
  token_cross_var_scaffold = (
    config.token_cross_var_scaffold_coeff
    * (token_cross_ent_scaffold - scaffold_token_logprob) ** 2
    + (1 - config.token_cross_var_scaffold_coeff) * state.token_cross_var_scaffold
  )
  return (
    token_cross_ent_scaffold,
    token_cross_ent_naked,
    token_cross_var_scaffold,
    token_cross_var_naked,
  )
  
  
  from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp


@jax.jit
def sample_dirichlet(key: jax.random.PRNGKey, alpha: jnp.ndarray) -> jnp.ndarray:
  """Sample from a Dirichlet distribution."""
  gamma_samples = jax.random.gamma(key, alpha, shape=alpha.shape)
  return gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)


@jax.jit
def dirichlet_log_likelihood_from_logprob(
  logprobs: jnp.ndarray, alpha: jnp.ndarray
) -> jnp.ndarray:
  """
  Computes Dirichlet log likelihood:

  log Dir(p|α) = ln Γ(α₀) - ∑ᵢln Γ(αᵢ) + ∑ᵢ(αᵢ-1)ln(pᵢ)

  where:
  - α₀ = ∑ᵢαᵢ is the sum of all parameters
  - Γ(x) is the gamma function
  - pᵢ are probabilities (passed as logprobs)
  """
  return (
    jnp.sum((alpha - 1.0) * logprobs, axis=-1)
    - jsp.gammaln(jnp.sum(alpha, axis=-1))
    + jnp.sum(jsp.gammaln(alpha), axis=-1)
  )


@jax.jit
def dirichlet_expectation(alpha: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the expectation of p ~ Dir(α):

  E[p] = αᵢ/∑ⱼαⱼ

  where:
  - αᵢ is the i-th parameter
  - ∑ⱼαⱼ is the sum of all parameters
  """
  alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
  return alpha / alpha_sum


@jax.jit
def dirichlet_expected_entropy(alpha: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the expected entropy of p ~ Dir(α):

  E[H(p)] = ln B(α) + (α₀ - K)ψ(α₀) - ∑ⱼ(αⱼ - 1)ψ(αⱼ)

  where:
  - B(α) is the multivariate beta function
  - K is the dimension
  - ψ(x) is the digamma function
  """
  alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)  # alpha_0
  K = alpha.shape[-1]
  # ln B(α) term
  log_beta = jnp.sum(jsp.gammaln(alpha), axis=-1) - jsp.gammaln(alpha_sum.squeeze())

  # (α₀ - K)ψ(α₀) term
  digamma_sum = jsp.digamma(alpha_sum)
  second_term = (alpha_sum.squeeze() - K) * digamma_sum.squeeze()

  # -sum((αⱼ - 1)ψ(αⱼ)) term
  digamma_alpha = jsp.digamma(alpha)
  third_term = -jnp.sum((alpha - 1) * digamma_alpha, axis=-1)

  return log_beta + second_term + third_term


@jax.jit
def dirichlet_expected_varentropy(alpha: jnp.ndarray) -> jnp.ndarray:
  """Compute the expected varentropy of p ~ Dir(α):

  E[∑ᵢ ln(pᵢ)² * pᵢ] = ∑ᵢ (αᵢ/α₀) * (ψ(αᵢ)² + ψ₁(αᵢ))

  where:
  - α₀ = ∑ᵢαᵢ is the sum of all parameters
  - ψ(x) is the digamma function
  - ψ₁(x) is the trigamma function (first derivative of digamma)
  """
  alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)  # α₀
  # E[Xᵢ] = αᵢ/α₀
  expected_x = alpha / alpha_sum
  # ψ(αᵢ)² + ψ₁(αᵢ) term
  digamma_alpha = jsp.digamma(alpha)
  trigamma_alpha = jsp.polygamma(1, alpha)
  squared_plus_deriv = digamma_alpha**2 + trigamma_alpha
  # ∑ᵢ (αᵢ/α₀) * (ψ₁(αᵢ) + ψ(αᵢ)²)
  return jnp.sum(expected_x * squared_plus_deriv, axis=-1)


@jax.jit
def halley_update(alpha, target_values):
  """
  Compute the Halley's method update direction for the function
  """
  p1 = jsp.polygamma(1, alpha)
  p2 = jsp.polygamma(2, alpha)
  S = jnp.sum(alpha, axis=-1, keepdims=True)
  s1 = jsp.polygamma(1, S)
  s2 = jsp.polygamma(2, S)
  p1_inv = 1.0 / p1
  sum_p1_inv = jnp.sum(p1_inv, axis=-1, keepdims=True)
  denom = 1.0 - s1 * sum_p1_inv
  denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
  coeff = s1 / denom
  error = jsp.digamma(alpha) - jsp.digamma(S) - target_values
  temp = p1_inv * error
  sum_temp = jnp.sum(temp, axis=-1, keepdims=True)
  J_inv_error = temp + coeff * sum_temp * p1_inv
  sum_J_inv_error = jnp.sum(J_inv_error, axis=-1, keepdims=True)
  H_J_inv_error = p2 * J_inv_error - s2 * sum_J_inv_error
  temp2 = p1_inv * H_J_inv_error
  sum_temp2 = jnp.sum(temp2, axis=-1, keepdims=True)
  J_inv_H_J_inv_error = temp2 + coeff * sum_temp2 * p1_inv
  return -J_inv_error + 0.5 * J_inv_H_J_inv_error


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def fit_dirichlet(
  target_values,
  init_alpha=None,
  initial_lr=1.2,
  decay_alpha=0.1,
  decay_beta=2.0,
  decay_gamma=0.25,
  decay_nu=0.75,
  max_iters=140,
  tol=1e-4,
  dtype: jnp.dtype = jnp.bfloat16,
):
  """
  Estimates Dirichlet parameters (alpha) from target logprobs.
  """
  batch_shape = target_values.shape[:-1]
  n = target_values.shape[-1]
  min_lr = 1e-8
  target_values = target_values.astype(
    jnp.float32
  )  # for large vocab size needs float64
  if init_alpha is None:
    init_alpha = jnp.ones((*batch_shape, n), dtype=jnp.float32)

  def scan_body(carry, _):
    alpha, converged, error_norm, step = carry
    S = jnp.sum(alpha, axis=-1, keepdims=True)
    digamma_alpha = jsp.digamma(alpha)
    psi_S = jsp.digamma(S)
    error = digamma_alpha - psi_S - target_values
    error_norm = jnp.linalg.norm(error, axis=-1)
    new_converged = converged | (error_norm < tol)
    exp_factor = jnp.exp(-decay_alpha * (step**decay_nu))
    cos_factor = jnp.abs(jnp.cos(decay_beta / (step**decay_gamma)))
    lr = initial_lr * exp_factor * cos_factor
    lr = jnp.maximum(lr, min_lr)
    delta_alpha = halley_update(alpha, target_values)
    scaled_delta_alpha = lr[..., None] * delta_alpha
    max_delta = 0.5 * alpha
    scaled_delta_alpha = jnp.clip(scaled_delta_alpha, -max_delta, max_delta)
    new_alpha = jnp.where(
      new_converged[..., None],
      alpha,
      jnp.maximum(alpha + scaled_delta_alpha, alpha / 2),
    )
    return (new_alpha, new_converged, error_norm, step + 1), None

  init_state = (
    init_alpha,
    jnp.zeros(batch_shape, dtype=jnp.bool_),
    jnp.full(batch_shape, jnp.inf),
    jnp.ones(batch_shape, dtype=jnp.int32),
  )
  (final_alpha, final_converged, _, final_step), _ = jax.lax.scan(
    scan_body, init_state, None, length=max_iters
  )

  return final_alpha.astype(dtype), final_step - 1, final_converged


@jax.jit
def ent_grad_hess(
  logits: jnp.ndarray, T: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  p = jax.nn.softmax(logits / T[..., None], axis=-1)
  log_p = jax.nn.log_softmax(logits / T[..., None], axis=-1)
  mu1 = jnp.sum(p * log_p, axis=-1)
  diff = log_p - mu1[..., None]
  mu2 = jnp.sum(p * diff**2, axis=-1)
  mu3 = jnp.sum(p * diff**3, axis=-1)
  return -mu1, mu2 / T, -(2 * mu3 + 3 * mu2) / (T * T)


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def temp_tune(
  logits: jnp.ndarray,
  target_ent: jnp.ndarray,
  T_init: float = 1.0,
  lr: float = 0.1,
  max_iters: int = 10,
  tol: float = 1e-6,
  dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  batch_size = logits.shape[0]
  logits = logits.astype(jnp.float32)

  def scan_body(carry, _):
    T, iters, converged = carry
    ent, grad, hess = ent_grad_hess(logits, T)
    error = ent - target_ent
    new_converged = converged | (jnp.abs(error) < tol)
    denominator = 2 * grad * grad - error * hess
    halley_step = jnp.where(
      jnp.abs(denominator) > 1e-8,
      2 * error * grad / denominator,
      jnp.full_like(T, jnp.inf),
    )
    newton_step = jnp.where(
      jnp.abs(grad) > 1e-8, error / grad, jnp.full_like(T, jnp.inf)
    )
    grad_step = jnp.where(error > 0, lr * T, -lr * T)

    delta_T = jnp.where(
      jnp.abs(grad) < 1e-8,
      grad_step,
      jnp.where(jnp.abs(denominator) < 1e-8, newton_step, halley_step),
    )
    delta_T = jnp.clip(delta_T, -0.5 * T, 0.5 * T)
    new_T = jnp.where(new_converged, T, jnp.maximum(T - delta_T, T / 2))
    return (new_T, iters + 1, new_converged), None

  init_state = (
    jnp.full((batch_size,), T_init, dtype=jnp.float32),
    jnp.zeros(batch_size, dtype=jnp.int32),
    jnp.zeros(batch_size, dtype=jnp.bool_),
  )
  (final_T, final_iters, final_converged), _ = jax.lax.scan(
    scan_body, init_state, None, length=max_iters
  )
  return final_T.astype(dtype), final_iters, final_converged
  
