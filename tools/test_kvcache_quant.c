/* Test: wubu_kvcache_quant (Roofline/DB/KIVI convergence -- KV quant).
 * Pass 1 (correctness): q8_0 + KIVI round-trip within int8 resolution.
 * Pass 3 (robustness): n=0 / head_dim=1 / all-zero must not crash / NaN. */
#include "wubu_kvcache_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static float cosine(const float *a, const float *b, int n) {
    double dot=0,na=0,nb=0;
    for (int i=0;i<n;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
    return (na>0&&nb>0)? (float)(dot/sqrt(na*nb)) : 0.0f;
}

int main(void) {
    int n_tokens = 32, head_dim = 128;
    int N = n_tokens * head_dim;
    float *K = (float*)malloc(N*sizeof(float));
    float *V = (float*)malloc(N*sizeof(float));
    srand(11);
    for (int i=0;i<N;i++){ K[i]=((rand()%2000)/1000.0f)-1.0f; V[i]=((rand()%2000)/1000.0f)-1.0f; }

    /* ---- Q8_0 block-32 round trip ---- */
    {
        int8_t *q = (int8_t*)malloc(N*sizeof(int8_t));
        float *sc = (float*)malloc(((N+31)/32)*sizeof(float));
        float *out = (float*)malloc(N*sizeof(float));
        for (int b=0;b<N;b+=32){
            int bn = (b+32<=N)?32:(N-b);
            wubu_kvq_q8_quant(K+b, q+b, sc+(b/32), bn);
            wubu_kvq_q8_dequant(q+b, sc[b/32], out+b, bn);
        }
        float cos = cosine(K, out, N);
        int finite=1; for(int i=0;i<N;i++) if(!isfinite(out[i])) finite=0;
        printf("q8_0  cosine=%.6f finite=%d  bytes/elem=%.3f\n", cos, finite,
               wubu_kvq_bytes_per_elem(WUBU_KVQ_Q8_0));
        assert(finite);
        assert(cos > 0.99f);  /* int8 round-trip, K has full dynamic range */
        free(q); free(sc); free(out);
    }

    /* ---- KIVI: K per-channel, V per-token ---- */
    {
        int8_t *qK=(int8_t*)malloc(N), *qV=(int8_t*)malloc(N);
        float *sK=(float*)malloc(head_dim*sizeof(float));     /* per channel */
        float *sV=(float*)malloc(n_tokens*sizeof(float));      /* per token  */
        float *oK=(float*)malloc(N*sizeof(float)), *oV=(float*)malloc(N*sizeof(float));
        wubu_kvq_kivi_quant_K(K, qK, sK, n_tokens, head_dim);
        wubu_kvq_kivi_dequant_K(qK, sK, oK, n_tokens, head_dim);
        wubu_kvq_kivi_quant_V(V, qV, sV, n_tokens, head_dim);
        wubu_kvq_kivi_dequant_V(qV, sV, oV, n_tokens, head_dim);
        float cosK=cosine(K,oK,N), cosV=cosine(V,oV,N);
        printf("kiviK cosine=%.6f  kiviV cosine=%.6f  bytes/elem=%.3f\n",
               cosK, cosV, wubu_kvq_bytes_per_elem(WUBU_KVQ_KIVI));
        assert(cosK>0.99f && cosV>0.99f);
        free(qK);free(qV);free(sK);free(sV);free(oK);free(oV);
    }

    /* ---- Pass 3: edge cases ---- */
    {
        /* n=0 must not crash / write OOB */
        int8_t q0; float s0;
        wubu_kvq_q8_quant(NULL, &q0, &s0, 0);
        assert(s0==0.0f);
        /* all-zero input -> scale 0, all q=0, dequant 0, finite */
        float *z=(float*)calloc(N,sizeof(float));
        int8_t *qz=(int8_t*)malloc(N); float *sz=(float*)calloc(((N+31)/32),sizeof(float));
        float *oz=(float*)malloc(N*sizeof(float));
        for(int b=0;b<N;b+=32){ int bn=(b+32<=N)?32:(N-b);
            wubu_kvq_q8_quant(z+b,qz+b,sz+(b/32),bn);
            wubu_kvq_q8_dequant(qz+b,sz[b/32],oz+b,bn); }
        int zok=1; for(int i=0;i<N;i++) if(oz[i]!=0.0f) zok=0;
        assert(zok);
        /* head_dim=1 (degenerate) */
        float kv1[4]={0.5f,-0.3f,0.1f,-0.9f};
        int8_t q1[4]; float s1[4];
        wubu_kvq_kivi_quant_V(kv1,q1,s1,4,1);
        float o1[4]; wubu_kvq_kivi_dequant_V(q1,s1,o1,4,1);
        assert(isfinite(o1[0])&&isfinite(o1[3]));
        free(z);free(qz);free(sz);free(oz);
        printf("edge cases (n=0, all-zero, head_dim=1): OK\n");
    }

    free(K); free(V);
    printf("ALL KV-CACHE-QUANT TESTS PASSED\n");
    return 0;
}
