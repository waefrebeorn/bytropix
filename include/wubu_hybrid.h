/*
 * wubu_hybrid.h -- the hybrid-attention frontier (JA). C11.
 * Agnostic: a hybrid-config table (attention/SSM layer ratios,
 * per-head position schemes) + the hybrid engine ops. Covers
 * Falcon-H1, Hymba, Qwen3-Next, SSM at scale, Pareto analysis,
 * recall compensation, layer design, receptive fields, KV budget,
 * decode scheduling, prefill speed, accuracy parity, energy model,
 * streaming, stability, reasoning, co-training, quantization,
 * unified cache, speculative decode.
 */
#ifndef WUBU_HYBRID_H
#define WUBU_HYBRID_H

#include <stdint.h>

/* JA01: Falcon-H1 parallel hybrid (attention + Mamba2). */
int wubu_hyb_falcon(const float *attn_out, const float *ssm_out,
                         int d, float alpha, float *out);

/* JA02: Hymba hybrid-head (attention + SSM in one layer). */
int wubu_hyb_hymba(const float *x, int d, int n_heads,
                        float *attn_out, float *ssm_out);

/* JA03: Qwen3-Next GDN + gated alternation. */
int wubu_hyb_qwen(const float *x, int d, float gdn_scale,
                        float *out);

/* JA04: SSM at-scale energy (57K -> 370J). */
float wubu_hyb_ssm_energy(long ctx, float base_j);

/* JA05: hybrid Pareto frontier. */
int wubu_hyb_pareto(float acc, float ttft, float *pareto_score);

/* JA06: SSM recall compensation (attention for precise recall). */
int wubu_hyb_recall_comp(float ssm_recall, float attn_recall,
                             float *compensated);

/* JA07: hybrid layer-position design. */
int wubu_hyb_layer_pos(int layer, int n_layers);

/* JA08: SSM local + attention global (receptive fields). */
int wubu_hyb_receptive(int ssm_local, int attn_global);

/* JA09: hybrid KV budget (attention keeps KV, SSM doesn't). */
long wubu_hyb_kv_budget(long attn_kv, long ssm_kv, long total);

/* JA10: hybrid decode scheduling (per-layer dispatch). */
int wubu_hyb_decode_sched(int layer, int n_layers, float attn_frac);

/* JA11: SSM prefill speed advantage. */
float wubu_hyb_prefill_speed(long ctx, float attn_t, float ssm_t);

/* JA12: hybrid accuracy parity evaluation. */
int wubu_hyb_parity(float hybrid_acc, float attn_acc);

/* JA13: SSM energy model at scale. */
float wubu_hyb_energy_model(long ctx, float j_per_token);

/* JA14: hybrid streaming (SSM constant + attention window). */
int wubu_hyb_stream(long ssm_state, int attn_window, long total);

/* JA15: gated-attention long-context stability. */
int wubu_hyb_stability(float attn_norm, float ssm_norm, float th);

/* JA16: hybrid reasoning accuracy (long-context). */
float wubu_hyb_reasoning(float hybrid_acc, long ctx);

/* JA17: SSM + attention co-training recipe. */
int wubu_hyb_cotrain(float attn_lr, float ssm_lr, float ratio);

/* JA18: hybrid quantization (both mechanisms). */
int wubu_hyb_quant(const float *w, int n, int bits);

/* JA19: SSM state + KV unified cache. */
int wubu_hyb_unified_cache(long ssm_state, long kv, long *total);

/* JA20: hybrid speculative decode (SSM drafter + attention verifier). */
int wubu_hyb_spec_decode(const float *draft, const float *verify, int n);

#endif
