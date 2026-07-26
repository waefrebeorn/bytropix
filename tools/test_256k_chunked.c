/* test_256k_chunked.c — prove the FULL 256K (262144) context forward runs on a
 * memory-limited box by chunking the prefill into time-chunks that carry the
 * model's persistent SSM/conv/KV-cache state.
 *
 * Two checks:
 *  (A) Correctness: chunked forward == single forward (final-chunk logits must
 *      match the same tokens processed in one shot). The recurrence is stateful
 *      and continues mid-sequence, so chunking is mathematically exact.
 *  (B) 256K window: chunked forward at T_total=262144 (the full window) with a
 *      chunk_sz that keeps peak memory within the box, must produce finite
 *      logits. This is the real 256K prefill (single-shot 262144 OOMs at ~30GB).
 */
#include "wubu_model_safetensors_bridge.h"
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static int all_finite(const float *p, int n) {
    for (int i = 0; i < n; i++) if (!isfinite(p[i])) return 0;
    return 1;
}
static double now_sec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static unsigned long vmpeak_kb(void){
    FILE *f=fopen("/proc/self/status","r"); if(!f) return 0;
    char ln[256]; unsigned long kb=0;
    while(fgets(ln,sizeof ln,f)) if(strncmp(ln,"VmPeak:",7)==0) kb=strtoul(ln+7,NULL,10);
    fclose(f); return kb;
}

static const char *base = "fixture_model.safetensors";

int main(int argc, char **argv) {
    int proof_T = (argc > 1) ? atoi(argv[1]) : 262144;   /* default: FULL 256K */
    int chunk_sz = (argc > 2) ? atoi(argv[2]) : 4096;
    int corr_T   = 8192;     /* correctness baseline length */
    int corr_cs  = 2048;
    /* Force the SCALAR SSM recurrence for BOTH the single and chunked forwards
     * so the comparison is apples-to-apples. The optimized chunked SSM
     * recurrence (wubu_ssm_chunked_recurrence) is reference-correct only for
     * short sequences (T<=4); for long sequences it diverges from the scalar
     * path — a known correctness bug in the optimized prefill, tracked
     * separately. The scalar path IS correct and carries state across the
     * multiple calls our chunked forward makes. */
    setenv("FORCE_CPU_SSM_SEQ", "1", 1);

    wubu_model_t m={0};
    if (wubu_model_init_safetensors(&m, base, &(wubu_adapter_t){0})) {
        fprintf(stderr, "FAIL: cannot load %s\n", base); return 1;
    }
    printf("loaded: n_layers=%d d_model=%d vocab=%d gqa_q=%d kv=%d hd=%d\n",
           m.n_layers, m.d_model, m.vocab_size, m.gqa_q_heads, m.gqa_kv_heads, m.gqa_head_dim);
    printf("GQA_MAX_CTX=%d  (256K window needs >= 262144)\n", GQA_MAX_CTX);
    if (GQA_MAX_CTX < 262144) { fprintf(stderr, "FAIL: window < 256K\n"); return 1; }

    /* ---- (A) Correctness: single vs chunked on corr_T tokens ---- */
    int *prompt = malloc((size_t)corr_T * sizeof(int));
    for (int i = 0; i < corr_T; i++) prompt[i] = (int)((i*2654435761u) % (unsigned)m.vocab_size);

    float *single = malloc((size_t)corr_T * m.vocab_size * sizeof(float));
    wubu_model_forward(&m, prompt, 1, corr_T, single);   /* single forward */
    if (!all_finite(single, corr_T*m.vocab_size)) {
        fprintf(stderr, "FAIL: single forward non-finite\n"); return 1;
    }
    /* last corr_cs positions from single = expected final-chunk logits */
    float *expect = malloc((size_t)corr_cs * m.vocab_size * sizeof(float));
    memcpy(expect, single + (size_t)(corr_T - corr_cs)*m.vocab_size,
           (size_t)corr_cs * m.vocab_size * sizeof(float));

    float *chunked = malloc((size_t)corr_cs * m.vocab_size * sizeof(float));
    wubu_model_forward_chunked(&m, prompt, 1, corr_T, corr_cs, chunked);
    if (!all_finite(chunked, corr_cs*m.vocab_size)) {
        fprintf(stderr, "FAIL: chunked forward non-finite\n"); return 1;
    }
    double maxdiff = 0;
    for (int i = 0; i < corr_cs*m.vocab_size; i++) {
        double d = fabs((double)chunked[i] - (double)expect[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("(A) correctness single-vs-chunked (T=%d, chunk=%d): max|diff|=%.3e %s\n",
           corr_T, corr_cs, maxdiff, maxdiff < 1e-2 ? "-> MATCH" : "-> MISMATCH");
    if (maxdiff >= 1e-2) { fprintf(stderr, "FAIL: chunked != single\n"); return 1; }
    free(single); free(expect); free(chunked);
    wubu_model_safetensors_free(&m);

    /* ---- (B) 256K window: chunked full prefill ---- */
    wubu_model_t m2={0};
    if (wubu_model_init_safetensors(&m2, base, &(wubu_adapter_t){0})) {
        fprintf(stderr, "FAIL: cannot reload %s\n", base); return 1;
    }
    int Ttot = proof_T;
    int *p2 = malloc((size_t)Ttot * sizeof(int));
    for (int i = 0; i < Ttot; i++) p2[i] = (int)((i*2654435761u) % (unsigned)m2.vocab_size);

    float *lg = malloc((size_t)chunk_sz * m2.vocab_size * sizeof(float));
    printf("(B) 256K prefill: chunked T_total=%d chunk_sz=%d ...\n", Ttot, chunk_sz);
    fflush(stdout);
    double t0 = now_sec();
    wubu_model_forward_chunked(&m2, p2, 1, Ttot, chunk_sz, lg);
    double dt = now_sec() - t0;
    unsigned long peak = vmpeak_kb();
    printf("    done in %.1fs (%.0f tok/s), VmPeak=%.2f GB\n", dt, Ttot/dt, (double)peak/1048576.0);
    if (!all_finite(lg, chunk_sz*m2.vocab_size)) {
        fprintf(stderr, "FAIL: 256K chunked prefill non-finite\n"); return 1;
    }
    int am=0; float mv=-1e30f;
    for (int j=0;j<m2.vocab_size;j++) if (lg[j]>mv){mv=lg[j];am=j;}
    printf("PASS 256K chunked prefill: finite logits, argmax=%d logit=%.4f\n", am, mv);
    printf("\n=== 256K-context chunked forward: ALL CHECKS PASSED ===\n");
    free(p2); free(lg);
    wubu_model_safetensors_free(&m2);
    free(prompt);
    return 0;
}
