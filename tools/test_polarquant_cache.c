/* test_polarquant_cache.c — PolarQuant mixed-precision KV cache benchmark
 * Tests: roundtrip accuracy, fused dot, attention vs F32 baseline,
 * bandwidth comparison at 256K/512K context.
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

int main(void) {
    printf("=== PolarQuant Mixed-Precision KV Cache Benchmark ===\n\n");

    int d = 128;
    int n_recent = 32;
    int capacity = 256;
    int bits = 8;

    /* Init quantizer */
    wubu_polarquant_t pq;
    wubu_polarquant_init(&pq, d, 1, 1.0f, (float)bits);

    /* Init cache */
    wubu_polar_cache_t cache;
    wubu_polar_cache_init(&cache, &pq, d, n_recent, capacity);

    int storage = wubu_polarquant_storage_bytes(&pq, d);
    printf("Config: d=%d, bits=%d, n_recent=%d, capacity=%d\n", d, bits, n_recent, capacity);
    printf("PolarQuant storage: %d bytes/vector (%.1fx vs F32)\n\n", storage, (float)(d*4)/storage);

    /* Push random K,V pairs */
    srand(42);
    float *k_all = malloc((size_t)capacity * d * sizeof(float));
    float *v_all = malloc((size_t)capacity * d * sizeof(float));
    float *q = malloc((size_t)d * sizeof(float));

    for (int i = 0; i < capacity * d; i++) {
        k_all[i] = (float)((rand() % 200) - 100) * 0.01f;
        v_all[i] = (float)((rand() % 200) - 100) * 0.01f;
    }
    for (int i = 0; i < d; i++)
        q[i] = (float)((rand() % 200) - 100) * 0.01f;

    for (int i = 0; i < capacity; i++)
        wubu_polar_cache_push(&cache, &k_all[i*d], &v_all[i*d]);

    printf("Cache filled: %d tokens (%d F32 + %d quantized)\n\n",
           cache.n_filled, n_recent, cache.n_filled - n_recent);

    /* Compute attention with PolarQuant cache */
    float pq_out[128];
    wubu_polar_cache_attention(&cache, q, pq_out, 1.0f);

    /* Compute F32 baseline attention (online softmax) */
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float f32_out[128];
    memset(f32_out, 0, sizeof(f32_out));

    for (int i = 0; i < capacity; i++) {
        float score = dot_f(q, &k_all[i*d], d);
        if (score > max_score) {
            float old_max = max_score;
            max_score = score;
            sum_exp = sum_exp * expf(old_max - max_score) + 1.0f;
            float scale = expf(old_max - max_score);
            for (int j = 0; j < d; j++) f32_out[j] *= scale;
            for (int j = 0; j < d; j++) f32_out[j] += v_all[i*d + j];
        } else {
            float e = expf(score - max_score);
            sum_exp += e;
            for (int j = 0; j < d; j++) f32_out[j] += e * v_all[i*d + j];
        }
    }
    for (int j = 0; j < d; j++) f32_out[j] /= (sum_exp + 1e-10f);

    /* Compare */
    float max_err = 0, avg_err = 0;
    for (int i = 0; i < d; i++) {
        float err = fabsf(pq_out[i] - f32_out[i]);
        if (err > max_err) max_err = err;
        avg_err += err;
    }
    avg_err /= d;

    /* Cosine similarity of attention outputs */
    float dot = 0, no = 0, nr = 0;
    for (int i = 0; i < d; i++) {
        dot += f32_out[i] * pq_out[i];
        no += f32_out[i] * f32_out[i];
        nr += pq_out[i] * pq_out[i];
    }
    float cos_attn = dot / (sqrtf(no) * sqrtf(nr) + 1e-10f);

    printf("--- Attention Output Comparison (PolarQuant vs F32) ---\n");
    printf("  cosine similarity: %.6f\n", cos_attn);
    printf("  max error:          %.6f\n", max_err);
    printf("  avg error:          %.6f\n", avg_err);
    printf("  %s\n\n", cos_attn > 0.95f ? "PASS" : "FAIL");

    /* Bandwidth comparison at 256K and 512K context */
    printf("--- Bandwidth at Scale ---\n");
    for (int ctx = 256*1024; ctx <= 512*1024; ctx *= 2) {
        double f32_mb = (double)d * 4 * 2 * ctx / 1e6;
        double pq_mb  = (double)storage * 2 * (ctx - 32) / 1e6
                       + (double)d * 4 * 2 * 32 / 1e6;
        printf("  %dK: F32=%.1f MB  PQ=%.1f MB  (%.1fx compression)\n",
               ctx/1024, f32_mb, pq_mb, f32_mb / pq_mb);
    }
    printf("\n");

    /* Fused dot vs decode+dot accuracy */
    printf("--- Fused Dot Accuracy ---\n");
    float max_dot_err = 0;
    int n_test = 100;
    for (int t = 0; t < n_test; t++) {
        int idx = rand() % (capacity - n_recent);
        const float *k_packed = &cache.quant_k[idx * cache.max_bytes_per_token];
        int k_bytes = cache.quant_bytes[idx];
        float fused = wubu_polarquant_fused_dot(&pq, q, k_packed, k_bytes);
        /* Decode then dot manually */
        float k_dec[128];
        wubu_polarquant_dequantize_kv(&pq, k_packed, k_bytes, k_dec, d);
        float manual = dot_f(q, k_dec, d);
        float err = fabsf(fused - manual);
        if (err > max_dot_err) max_dot_err = err;
    }
    printf("  max fused vs manual dot error: %.8f %s\n\n", max_dot_err,
           max_dot_err < 1e-5f ? "PASS" : "FAIL");

    int errors = (cos_attn <= 0.95f) || (max_dot_err >= 1e-5f);
    printf("=== %d errors ===\n", errors);

    /* Cleanup */
    free(k_all); free(v_all); free(q);
    wubu_polar_cache_free(&cache);
    wubu_polarquant_free(&pq);
    return errors;
}
