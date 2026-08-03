/*
 * wubu_vision.h -- the multimodal vision frontier (JB). C11.
 * Agnostic: a multimodal-config table (per-modality budgets,
 * routing weights) + the vision ops. Covers VisionSelector,
 * visual-text efficiency, image/video/audio token compression,
 * cross-modal alignment, redundancy detection, modality-aware
 * KV, sparsity, survey, importance scoring, fusion, budget
 * planning, encoder efficiency, eviction, prefix, streaming,
 * energy, dedup, routing.
 */
#ifndef WUBU_VISION_H
#define WUBU_VISION_H

#include <stdint.h>

/* JB01: learnable visual-token selection. */
int wubu_vision_selector(const float *scores, int n, float th, int *keep);

/* JB02: text-as-pixels token efficiency. */
float wubu_vision_text_eff(int text_tokens, int pixel_tokens);

/* JB03: image token compression (patch merging). */
int wubu_vision_img_compress(int patches, int merge_factor, int *out);

/* JB04: video token compression (temporal redundancy). */
int wubu_vision_vid_compress(int frames, int fps, float redundancy, int *out);

/* JB05: audio token compression (spectral redundancy). */
int wubu_vision_audio_compress(int spec_bins, float redundancy, int *out);

/* JB06: cross-modal token alignment (CLIP-style). */
int wubu_vision_clip_align(const float *vis, const float *txt, int d,
                                float *sim);

/* JB07: visual redundancy detection (similar-patch dedup). */
int wubu_vision_redundancy(const float *patches, int n, int d,
                                float th, int *keep);

/* JB08: modality-aware KV budgets. */
int wubu_vision_kv_budget(int vis_kv, int txt_kv, int total, int *alloc);

/* JB09: multimodal attention sparsity. */
int wubu_vision_sparse(const float *attn, int n, float th, int *keep);

/* JB11: visual token importance scoring. */
int wubu_vision_importance(const float *features, int n, int k, int *topk);

/* JB12: audio-visual fusion compression. */
int wubu_vision_av_fusion(const float *audio, const float *vis, int n,
                               float *fused);

/* JB13: multimodal token budget planner. */
int wubu_vision_budget_plan(int vis_tok, int txt_tok, long total_budget,
                                 int *vis_alloc, int *txt_alloc);

/* JB14: vision encoder efficiency (ViT patch efficiency). */
float wubu_vision_enc_eff(int patches, int d_model);

/* JB15: multimodal eviction (low-salience tokens). */
int wubu_vision_evict(const float *salience, int n, float th, int *evict);

/* JB16: cross-modal prefix (shared multimodal prefix). */
int wubu_vision_prefix(const float *vis_prefix, const float *txt_prefix,
                            int d, float *shared);

/* JB17: visual token streaming. */
int wubu_vision_stream(const float *tokens, int n, int d, int window,
                            float *out);

/* JB18: per-modality energy. */
float wubu_vision_energy(int modality, long tokens, float j_per_token);

/* JB19: visual token dedup (repeated-region suppression). */
int wubu_vision_dedup(const float *tokens, int n, int d, float th, int *keep);

/* JB20: modality routing (which modality matters per task). */
int wubu_vision_route(const float *task_vec, int n, float *weights);

#endif