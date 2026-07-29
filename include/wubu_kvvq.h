/*
 * wubu_kvvq.h -- data-independent RESIDUAL subvector VQ for KV-cache vectors
 * (doc 014: CommVQ / TurboQuant / Residual-VQ convergent idea).
 *
 * WHY: after A01 (Q8_0) / A02 (KIVI) KV quant, the next halving is SUB-4-bit.
 * CommVQ (Apple, ICML'25) and TurboQuant (Google, ICRL'26) show 1-3 bit KV
 * with minimal loss using a CODEBOOK that is DATA-INDEPENDENT -- fixed, not
 * trained on your KV distribution -- so you quantize each K/V vector ONLINE
 * during decode with zero calibration. Complement to FlashDecoding (doc 015):
 * shrinks KV *storage/bandwidth* while FlashDecoding parallelizes *compute*.
 *
 * SCHEME (own-C, data-independent): RESIDUAL subvector (product) VQ.
 *   - Split head_dim into n_sub subvectors of sub_dim = head_dim/n_sub.
 *   - For each subvector, run n_stages stages of VQ: stage s quantizes the
 *     RESIDUAL left by stage s-1 against that stage's fixed codebook.
 *     This is standard Residual VQ (used by EnCodec / TurboQuant) and reaches
 *     high fidelity with few bits per stage (2-3). Total bits/vec =
 *     n_sub * n_stages * bits (kept sub-4-bit-per-element in practice).
 *   - All codebooks are built from a FIXED (seeded) Gaussian + L2-normalize,
 *     so they are identical on every machine/model -- truly data-independent.
 *
 * bits in [1,3] (sub-4-bit per stage). n_stages typically 2-4.
 */
#ifndef WUBU_KVVQ_H
#define WUBU_KVVQ_H

#include <stdint.h>
#include <stddef.h>

#define WUBU_KVVQ_MAX_BITS 3
#define WUBU_KVVQ_MAX_STAGES 8

/* One stage's codebook at one subvector position: 2^bits codewords x sub_dim. */
typedef struct {
    int bits, sub_dim, n_codewords;
    float *codebook;       /* [n_codewords * sub_dim] */
} wubu_kvvq_subcb_t;

typedef struct {
    int bits, head_dim, n_sub, sub_dim, n_stages;
    wubu_kvvq_subcb_t *sub;  /* [n_sub * n_stages] (stage-major) */
} wubu_kvvq_codebook_t;

/* head_dim must be divisible by n_sub. Returns 0 on success. */
int  wubu_kvvq_codebook_init(wubu_kvvq_codebook_t *cb, int bits, int head_dim, int n_sub, int n_stages);
void wubu_kvvq_codebook_free(wubu_kvvq_codebook_t *cb);

/* Quantize one head_dim vector -> n_sub*n_stages indices. */
void wubu_kvvq_quantize_vec(const float *vec, const wubu_kvvq_codebook_t *cb, int *indices);
/* Dequantize n_sub*n_stages indices -> out[head_dim]. */
void wubu_kvvq_dequant_vec(const int *indices, const wubu_kvvq_codebook_t *cb, float *out);

/* Packed buffer helpers. Total indices per vector = n_sub * n_stages. */
int  wubu_kvvq_packed_bytes(int n_vecs, int n_sub, int n_stages, int bits);
void wubu_kvvq_pack(const int *indices, int n_vecs, int n_sub, int n_stages, int bits, uint8_t *out);
void wubu_kvvq_unpack(const uint8_t *buf, int n_vecs, int n_sub, int n_stages, int bits, int *indices);

#endif /* WUBU_KVVQ_H */
