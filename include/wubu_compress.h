/*
 * wubu_compress.h -- the context-compression frontier (IY). C11.
 * Agnostic: a compressor-state + policy table, the caller picks
 * the method. Covers LLMLingua, RECOMP, Doc2Atom, Cartridges,
 * LaMR, SES-RAG, GRC, EPC, LIM mitigation, density budgeting,
 * tool-schema compression, autoencoder, distillation, paged
 * attention, ratio governor, fidelity audit.
 */
#ifndef WUBU_COMPRESS_H
#define WUBU_COMPRESS_H

#include <stdint.h>

/* IY01: LLMLingua perplexity-gated drop. */
int wubu_comp_llmlingua(const float *perplexities, int n, float th,
                            int *keep);

/* IY02: LLMLingua-2 token classification. */
int wubu_comp_llmlingua2(const float *scores, int n, float th,
                             int *keep);

/* IY03: question-aware reordering. */
int wubu_comp_reorder(const int *is_question, int n, int *order);

/* IY04: self-information pruning. */
int wubu_comp_self_info(const float *info, int n, float keep_frac,
                            int *keep);

/* IY05: RECOMP extractive+abstractive compression. */
int wubu_comp_recmp(const float *scores, int n, float ext_th,
                        float abs_th, int *keep);

/* IY06: Doc2Atom knowledge atoms. */
int wubu_comp_doc2atom(const float *embeddings, int n, int d,
                           float th, int *atoms);

/* IY07: Cartridges KV cache manager. */
int wubu_comp_cartridge(long kv_used, long kv_cap, long *evict);

/* IY08: LaMR code-context pruning. */
int wubu_comp_lamr(const float *sem_score, const float *dep_score,
                       int n, float w_sem, float w_dep);

/* IY09: SES-RAG semantic segmentation. */
int wubu_comp_sesrag(const float *densities, int n, float th,
                         int *segments);

/* IY10: GRC meta latent tokens. */
int wubu_comp_grc(const float *tokens, int n, int k, int *meta);

/* IY11: EPC write-time retention. */
int wubu_comp_epc(float predicted_relevance, float cur_retention);

/* IY12: LIM mitigation (reorder important context). */
int wubu_comp_lim(const float *importance, int n, int *order);

/* IY13: lexical-density-aware budgeting. */
long wubu_comp_budget(long tokens, float density, long base_budget);

/* IY14: tool-schema compression. */
int wubu_comp_tool_schema(const char *schema, int len, float ratio);

/* IY15: in-context autoencoder. */
int wubu_comp_autoenc(const float *ctx, int n, int d,
                          float *latent, int k);

/* IY16: Doc-to-LoRA distillation. */
int wubu_comp_distill(const float *doc_emb, int n, int d,
                          float *lora_weights);

/* IY17: latent-memory generation. */
int wubu_comp_latent_mem(const float *kv, int n, int d,
                             float *memory);

/* IY18: hybrid paged attention. */
int wubu_comp_paged(const float *attn, int n, int page_size,
                        int *pages);

/* IY19: compression-ratio governor. */
int wubu_comp_governor(float ratio, float target, float quality);

/* IY20: compressed-prompt fidelity audit. */
float wubu_comp_fidelity(const float *orig, const float *recon, int n);

#endif
