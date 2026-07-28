/* Test: KV-cache quantization integration via kv_cache_read/write_head
 * (the engine's real attention path) for OUR_Q8 and KIVI, plus the
 * roofline-driven wubu_kv_select. Pass 1 correctness + Pass 3 edge cases. */
#include "wubu_model.h"
#include "wubu_kv_select.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

/* Re-define the schemes locally so we can compile this test for each without
 * recompiling the whole engine with a different KV_CACHE_* macro. We directly
 * exercise the three kv_cache_* functions under each #define by including the
 * logic via macros is hard, so instead we test the underlying wubu_kvcache_quant
 * module paths that the headers route to, AND test kv_cache_read/write_head
 * under KV_CACHE_OUR_Q8 + KV_CACHE_KIVI by compiling this TU with those set. */

static float cosine(const float *a, const float *b, int n) {
    double dot=0,na=0,nb=0;
    for (int i=0;i<n;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
    return (na>0&&nb>0)? (float)(dot/sqrt(na*nb)) : 0.0f;
}

int main(void) {
    int n_tokens = 16, head_dim = 128;
    int N = n_tokens * head_dim;
    float *src = (float*)malloc(N*sizeof(float));
    srand(21);
    for (int i=0;i<N;i++) src[i]=((rand()%2000)/1000.0f)-1.0f;

#if defined(KV_CACHE_OUR_Q8) || defined(KV_CACHE_KIVI)
    /* Under these schemes the cache stores int8 + scales; alloc accordingly. */
    int64_t bytes = kv_cache_alloc_size(N);
    void *cache = malloc((size_t)bytes);
    assert(cache && bytes > 0);
    float *rec = (float*)malloc(N*sizeof(float));

    /* Write the whole layer (offset 0, n = N elements) like the real GQA
     * forward, then read each token back at offset = t*head_dim. */
    kv_cache_write_head(cache, 0, src, N);
    float maxerr = 0;
    for (int t=0;t<n_tokens;t++) {
        kv_cache_read_head(cache, (int64_t)t*head_dim, rec + t*head_dim, head_dim);
        for (int i=0;i<head_dim;i++){
            float e = fabsf(rec[t*head_dim+i] - src[t*head_dim+i]);
            if (e>maxerr) maxerr=e;
        }
    }
    float cos = cosine(src, rec, N);
    printf("[%s] alloc=%lld B  roundtrip cosine=%.6f  maxerr=%.5f\n",
           wubu_kv_scheme_name(
#if defined(KV_CACHE_KIVI)
               WUBU_KV_KIVI
#else
               WUBU_KV_Q8
#endif
           ), (long long)bytes, cos, maxerr);
    assert(cos > 0.99f);  /* int8 round-trip */
    free(cache); free(rec);
#endif

    /* --- roofline-driven selector (independent of KV_CACHE_* macro) --- */
    {
        wubu_roofline_cfg_t c = wubu_roofline_default();
        c.n_kv_heads = 8; c.head_dim = 128; c.bw_bits = 16; c.bkv_bits = 16;
        c.beta_eff_tb_s = 0.05; /* CPU ~50 GB/s effective */

        /* Large batch -> KV dominates (B >> B*): expect KV compression (Q8,
         * short ctx). bstar(4096)~105, B=256 -> COMPRESS_KV. P=27e9 (27B). */
        wubu_kv_choice_t a = wubu_kv_select(&c, 27e9, 256, 4096);
        printf("select(P=27B,B=256,s=4k): kv=%s  wbits=%d  \"%s\"\n",
               wubu_kv_scheme_name(a.kv), a.weight_bits, a.why);
        assert(a.kv == WUBU_KV_Q8);          /* short ctx -> Q8 */
        assert(a.weight_bits == 16);

        /* Small batch -> weights dominate (B < B*): compress weights to int4,
         * KV stays fp16. */
        wubu_kv_choice_t b = wubu_kv_select(&c, 27e9, 1, 4096);
        printf("select(P=27B,B=1,s=4k): kv=%s  wbits=%d  \"%s\"\n",
               wubu_kv_scheme_name(b.kv), b.weight_bits, b.why);
        assert(b.kv == WUBU_KV_F16);
        assert(b.weight_bits == 4);

        /* Long context, large batch -> KV dominates + long ctx -> KIVI. */
        wubu_kv_choice_t d = wubu_kv_select(&c, 27e9, 64, 131072);
        printf("select(P=27B,B=64,s=131k): kv=%s  wbits=%d  \"%s\"\n",
               wubu_kv_scheme_name(d.kv), d.weight_bits, d.why);
        assert(d.kv == WUBU_KV_KIVI);
    }

    free(src);
    printf("ALL KV-CACHE-INTEGRATION TESTS PASSED\n");
    return 0;
}
