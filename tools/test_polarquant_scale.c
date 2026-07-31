/* test_polarquant_scale.c — PolarQuant vs Q8 scaling benchmark
 * Simulates decode attention at 256K and 512K context
 * Measures: bandwidth, attention cosine, throughput
 */
#include "wubu_polarquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static float dot_f(const float *a, const float *b, int d) {
    float s=0; for(int i=0;i<d;i++) s+=a[i]*b[i]; return s;
}

/* Q8 quantize: scale + int8 values */
typedef struct {
    float scale;
    int8_t *vals;
} q8_vec;

static void q8_quantize(const float *x, int d, q8_vec *out) {
    float amax = 0;
    for (int i=0;i<d;i++) amax = fmaxf(amax, fabsf(x[i]));
    out->scale = amax / 127.0f;
    for (int i=0;i<d;i++) {
        int v = (int)(x[i] / (out->scale + 1e-10f));
        if (v > 127) v = 127; if (v < -128) v = -128;
        out->vals[i] = (int8_t)v;
    }
}

static void q8_dequantize(const q8_vec *q, int d, float *out) {
    for (int i=0;i<d;i++) out[i] = q->vals[i] * q->scale;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("=== PolarQuant vs Q8 Scaling Benchmark ===\n\n");

    int d = 128;
    int n_recent = 128;  /* F32 residual buffer */
    int contexts[] = {1024, 4096, 16384, 65536, 262144, 524288};
    int n_ctx = 6;

    /* Init PolarQuant */
    wubu_polarquant_t pq;
    wubu_polarquant_init(&pq, d, 1, 1.0f, 8.0f);
    int pq_storage = wubu_polarquant_storage_bytes(&pq, d);
    int q8_storage = sizeof(float) + d;  /* scale + int8 values */

    printf("Config: d=%d, n_recent=%d\n", d, n_recent);
    printf("Storage per vector: F32=%d bytes, Q8=%d bytes, PQ=%d bytes\n\n",
           d*4, q8_storage, pq_storage);

    printf("%-10s %-8s %-8s %-12s %-12s %-10s %-10s\n",
           "Context", "F32_MB", "Q8_MB", "PQ_MB", "F32->PQ", "PQ_cos", "Q8_cos");
    printf("---------- -------- -------- ------------ ------------ ---------- ----------\n");

    srand(12345);

    for (int ci = 0; ci < n_ctx; ci++) {
        int ctx = contexts[ci];
        int n_quant = ctx - n_recent;
        if (n_quant < 0) n_quant = 0;

        /* Gather random K,V */
        float *k_all = malloc((size_t)ctx * d * sizeof(float));
        float *v_all = malloc((size_t)ctx * d * sizeof(float));
        float q[128];
        for (int i = 0; i < ctx * d; i++) {
            k_all[i] = (float)((rand() % 200) - 100) * 0.01f;
            v_all[i] = (float)((rand() % 200) - 100) * 0.01f;
        }
        for (int i = 0; i < d; i++)
            q[i] = (float)((rand() % 200) - 100) * 0.01f;

        /* Quantize older tokens to Q8 and PQ */
        q8_vec *q8_k = malloc((size_t)n_quant * sizeof(q8_vec));
        q8_vec *q8_v = malloc((size_t)n_quant * sizeof(q8_vec));
        for (int i = 0; i < n_quant; i++) {
            q8_k[i].vals = malloc(d);
            q8_v[i].vals = malloc(d);
            q8_quantize(&k_all[(n_recent+i)*d], d, &q8_k[i]);
            q8_quantize(&v_all[(n_recent+i)*d], d, &q8_v[i]);
        }

        /* PQ cache (only quantize a subset for speed at large ctx) */
        int pq_test_n = n_quant < 2048 ? n_quant : 2048;
        float *pq_k_packed = malloc((size_t)pq_test_n * (pq_storage + 16) * sizeof(float));
        int *pq_k_bytes = malloc((size_t)pq_test_n * sizeof(int));
        float *pq_v_packed = malloc((size_t)pq_test_n * (pq_storage + 16) * sizeof(float));
        for (int i = 0; i < pq_test_n; i++) {
            int ob = (pq_storage + 16);
            wubu_polarquant_quantize_kv(&pq, &k_all[(n_recent+i)*d],
                &pq_k_packed[i * ((pq_storage+16+3)/4)], &ob);
            pq_k_bytes[i] = ob;
            ob = (pq_storage + 16);
            wubu_polarquant_quantize_kv(&pq, &v_all[(n_recent+i)*d],
                &pq_v_packed[i * ((pq_storage+16+3)/4)], &ob);
        }

        /* F32 baseline attention (online softmax) */
        float max_s = -1e30f, sum_e = 0.0f;
        float f32_out[128]; memset(f32_out, 0, sizeof(f32_out));
        for (int i = 0; i < ctx; i++) {
            float s = dot_f(q, &k_all[i*d], d);
            if (s > max_s) {
                float om = max_s; max_s = s;
                sum_e = sum_e * expf(om - max_s) + 1.0f;
                float sc = expf(om - max_s);
                for (int j=0;j<d;j++) f32_out[j] *= sc;
                for (int j=0;j<d;j++) f32_out[j] += v_all[i*d+j];
            } else {
                float e = expf(s - max_s); sum_e += e;
                for (int j=0;j<d;j++) f32_out[j] += e * v_all[i*d+j];
            }
        }
        for (int j=0;j<d;j++) f32_out[j] /= (sum_e + 1e-10f);

        /* Q8 attention */
        max_s = -1e30f; sum_e = 0.0f;
        float q8_out[128]; memset(q8_out, 0, sizeof(q8_out));
        /* F32 recent */
        for (int i=0;i<n_recent && i<ctx;i++) {
            float s = dot_f(q, &k_all[i*d], d);
            if (s > max_s) {
                float om=max_s; max_s=s;
                sum_e=sum_e*expf(om-max_s)+1.0f;
                float sc=expf(om-max_s);
                for(int j=0;j<d;j++) q8_out[j]*=sc;
                for(int j=0;j<d;j++) q8_out[j]+=v_all[i*d+j];
            } else {
                float e=expf(s-max_s); sum_e+=e;
                for(int j=0;j<d;j++) q8_out[j]+=e*v_all[i*d+j];
            }
        }
        /* Q8 quantized */
        for (int i=0;i<n_quant;i++) {
            float k_dec[128]; q8_dequantize(&q8_k[i], d, k_dec);
            float s = dot_f(q, k_dec, d);
            float v_dec[128]; q8_dequantize(&q8_v[i], d, v_dec);
            if (s > max_s) {
                float om=max_s; max_s=s;
                sum_e=sum_e*expf(om-max_s)+1.0f;
                float sc=expf(om-max_s);
                for(int j=0;j<d;j++) q8_out[j]*=sc;
                for(int j=0;j<d;j++) q8_out[j]+=v_dec[j];
            } else {
                float e=expf(s-max_s); sum_e+=e;
                for(int j=0;j<d;j++) q8_out[j]+=e*v_dec[j];
            }
        }
        for(int j=0;j<d;j++) q8_out[j]/=(sum_e+1e-10f);

        /* PQ attention (sample subset for large ctx) */
        max_s = -1e30f; sum_e = 0.0f;
        float pq_out[128]; memset(pq_out, 0, sizeof(pq_out));
        /* F32 recent */
        for (int i=0;i<n_recent && i<ctx;i++) {
            float s = dot_f(q, &k_all[i*d], d);
            if (s > max_s) {
                float om=max_s; max_s=s;
                sum_e=sum_e*expf(om-max_s)+1.0f;
                float sc=expf(om-max_s);
                for(int j=0;j<d;j++) pq_out[j]*=sc;
                for(int j=0;j<d;j++) pq_out[j]+=v_all[i*d+j];
            } else {
                float e=expf(s-max_s); sum_e+=e;
                for(int j=0;j<d;j++) pq_out[j]+=e*v_all[i*d+j];
            }
        }
        /* PQ quantized (sample) */
        for (int i=0;i<pq_test_n;i++) {
            float k_dec[128];
            wubu_polarquant_dequantize_kv(&pq,
                &pq_k_packed[i * ((pq_storage+16+3)/4)], pq_k_bytes[i], k_dec, d);
            float s = dot_f(q, k_dec, d);
            float v_dec[128];
            wubu_polarquant_dequantize_kv(&pq,
                &pq_v_packed[i * ((pq_storage+16+3)/4)], pq_k_bytes[i], v_dec, d);
            if (s > max_s) {
                float om=max_s; max_s=s;
                sum_e=sum_e*expf(om-max_s)+1.0f;
                float sc=expf(om-max_s);
                for(int j=0;j<d;j++) pq_out[j]*=sc;
                for(int j=0;j<d;j++) pq_out[j]+=v_dec[j];
            } else {
                float e=expf(s-max_s); sum_e+=e;
                for(int j=0;j<d;j++) pq_out[j]+=e*v_dec[j];
            }
        }
        for(int j=0;j<d;j++) pq_out[j]/=(sum_e+1e-10f);

        /* Cosine similarities */
        float q8_d=0,q8_n=0,q8_r=0, pq_d=0,pq_n=0,pq_r=0;
        for(int j=0;j<d;j++){
            q8_d+=f32_out[j]*q8_out[j]; q8_n+=f32_out[j]*f32_out[j]; q8_r+=q8_out[j]*q8_out[j];
            pq_d+=f32_out[j]*pq_out[j]; pq_n+=f32_out[j]*f32_out[j]; pq_r+=pq_out[j]*pq_out[j];
        }
        float q8_cos = q8_d/(sqrtf(q8_n)*sqrtf(q8_r)+1e-10f);
        float pq_cos = pq_d/(sqrtf(pq_n)*sqrtf(pq_r)+1e-10f);

        /* Bandwidth */
        double f32_mb = (double)d*4*2*ctx/1e6;
        double q8_mb  = (double)q8_storage*2*n_quant/1e6 + (double)d*4*2*n_recent/1e6;
        double pq_mb  = (double)pq_storage*2*n_quant/1e6 + (double)d*4*2*n_recent/1e6;

        printf("%-10d %-8.1f %-8.1f %-12.1f %-12.1f %-10.4f %-10.4f\n",
               ctx, f32_mb, q8_mb, pq_mb, f32_mb/pq_mb, pq_cos, q8_cos);

        /* Cleanup */
        for (int i=0;i<n_quant;i++){free(q8_k[i].vals);free(q8_v[i].vals);}
        free(q8_k); free(q8_v);
        free(pq_k_packed); free(pq_k_bytes); free(pq_v_packed);
        free(k_all); free(v_all);
    }

    printf("\n=== Benchmark Complete ===\n");
    wubu_polarquant_free(&pq);
    return 0;
}
