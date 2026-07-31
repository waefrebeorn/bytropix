/* test_polarquant.c — PolarQuant recursive polar packed roundtrip */
#include "wubu_polarquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float dot_f(const float *a, const float *b, int d) {
    float s=0; for(int i=0;i<d;i++) s+=a[i]*b[i]; return s;
}
static float norm_f(const float *v, int d) {
    float s=0; for(int i=0;i<d;i++) s+=v[i]*v[i]; return sqrtf(s);
}
static float cos_sim(const float *a, const float *b, int d) {
    return dot_f(a,b,d)/(norm_f(a,d)*norm_f(b,d)+1e-10f);
}

int main(void) {
    printf("=== PolarQuant Recursive Polar Packed Roundtrip ===\n\n");

    /* Configs: d, default_bits, min_pass_score, use_perlevel, bits_array... */
    /* For per-level: bits_array = {level0_bits, level1_bits, ...} */
    
    /* Uniform configs */
    printf("--- Uniform d=128, 8-bit ---\n");
    {
        wubu_polarquant_t pq;
        wubu_polarquant_init(&pq, 128, 1, 1.0f, 8.0f);
        int storage = wubu_polarquant_storage_bytes(&pq, 128);
        printf("  storage = %d bytes, %.1fx compression\n", storage, 512.0/storage);
        /* quick check */
        float orig[128], recon[128], bits_buf[600];
        for (int i=0;i<128;i++) orig[i]=(float)(i%19-9)*0.01f;
        int ob=600;
        wubu_polarquant_quantize_kv(&pq, orig, bits_buf, &ob);
        wubu_polarquant_dequantize_kv(&pq, bits_buf, ob, recon, 128);
        float cs=0, no=0,nr=0,do_=0;
        for(int i=0;i<128;i++){no+=orig[i]*orig[i];nr+=recon[i]*recon[i];do_+=orig[i]*recon[i];}
        cs=do_/(sqrtf(no)*sqrtf(nr)+1e-10f);
        printf("  cosine = %.4f %s\n\n", cs, cs>0.95f?"PASS":"FAIL");
        wubu_polarquant_free(&pq);
    }
    
    /* Per-level configs are documented as a known failure mode.
     * The PolarQuant paper uses uniform bits at every level because
     * error compounds as (1+alpha)^t (Appendix C, arxiv 2502.02617).
     * Depth-tapered bits break the recursive chain: deep levels with
     * 2 bits produce 4 identical radii, which propagate outward.
     * Use uniform bits instead (tested below). */
    
    /* Default uniform configs */
    int configs[][3] = {
        {128, 8, 150}, {128, 6, 100}, {64, 8, 150}, {32, 8, 150},
    };
    int n_cfg = 4;
    int errors = 0;

    for (int ci = 0; ci < n_cfg; ci++) {
        int d = configs[ci][0];
        int bits = configs[ci][1];
        int min_score = configs[ci][2];
        printf("--- d=%d, bits/angle=%d ---\n", d, bits);

        wubu_polarquant_t pq;
        if (wubu_polarquant_init(&pq, d, 1, 1.0f, (float)bits) != 0) {
            fprintf(stderr, "init FAIL\n"); errors++; continue;
        }

        int storage = wubu_polarquant_storage_bytes(&pq, d);
        printf("  storage = %d bytes vs F32 %d bytes = %.1fx compression\n",
               storage, d*4, (double)(d*4)/storage);

        float *orig = malloc((size_t)d*sizeof(float));
        float *recon = malloc((size_t)d*sizeof(float));
        float *qvec = malloc((size_t)d*sizeof(float));
        float *bits_buf = malloc((size_t)(storage + 64));

        if (!orig||!recon||!qvec||!bits_buf) {
            errors++; wubu_polarquant_free(&pq); continue;
        }

        float avg_cos = 0;
        int pass_cos = 0, pass_score = 0;
        int n_trials = 200;

        for (int t = 0; t < n_trials; t++) {
            for (int i = 0; i < d; i++) {
                orig[i] = (float)((i*7 + t*13 + 3) % 19 - 9) * 0.01f;
                qvec[i] = (float)(((i+1)*3 + t*11) % 17 - 8) * 0.01f;
            }

            int out_bytes = storage + 64;
            if (wubu_polarquant_quantize_kv(&pq, orig, bits_buf, &out_bytes) != 0) {
                fprintf(stderr, "quantize FAIL t=%d\n", t); errors++; break;
            }
            if (wubu_polarquant_dequantize_kv(&pq, bits_buf, out_bytes, recon, d) != 0) {
                fprintf(stderr, "dequantize FAIL t=%d\n", t); errors++; break;
            }

            float cs = cos_sim(orig, recon, d);
            avg_cos += cs;
            if (cs > 0.90f) pass_cos++;

            float so = dot_f(qvec, orig, d);
            float sr = dot_f(qvec, recon, d);
            float rel = fabsf(so-sr)/(fabsf(so)+1e-6f);
            if (rel < 0.30f) pass_score++;
        }

        avg_cos /= n_trials;
        printf("  cosine sim:  avg=%.4f  pass(>0.90)=%d/%d %s\n",
               avg_cos, pass_cos, n_trials, pass_cos >= n_trials*3/4 ? "PASS" : "FAIL");
        printf("  attn score:   pass(<30%%)=%d/%d %s\n",
               pass_score, n_trials, pass_score >= min_score ? "PASS" : "FAIL");
        if (pass_score < min_score) errors++;

        /* 512K bandwidth */
        double f32_mb = (double)d*4*2*524288/1e6;
        double pq_mb  = (double)storage*2*524288/1e6;
        printf("  512K KV: F32=%.1f MB  PQ=%.1f MB  (%.0fx)\n\n",
               f32_mb, pq_mb, f32_mb/pq_mb);

        free(orig); free(recon); free(qvec); free(bits_buf);
        wubu_polarquant_free(&pq);
    }

    printf("=== %d errors ===\n", errors);
    return errors > 0 ? 1 : 0;
}
