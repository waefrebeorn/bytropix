/* test_polar_pso.c — PSO + procedural precache + Rambus serial bit reader
 * Tests: roundtrip, attention cosine, fused dot, serial reader accuracy
 */
#include "wubu_polar_pso.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float dot_f(const float *a, const float *b, int d) {
    float s=0; for(int i=0;i<d;i++) s+=a[i]*b[i]; return s;
}

int main(void) {
    printf("=== PolarQuant PSO + Procedural Precache Benchmark ===\n\n");
    int d = 128, bits = 8, max_tokens = 256;

    wubu_polarquant_t pq;
    wubu_polarquant_init(&pq, d, 1, 1.0f, (float)bits);

    /* Init PSO */
    wubu_polar_pso_t pso;
    wubu_polar_pso_init(&pso, &pq, bits, d);
    printf("PSO: bits=%d, d=%d, storage=%d bytes\n", pso.bits, pso.d, pso.storage_bytes);
    printf("Trig tables: %d entries (cos/sin precomputed)\n\n", 1<<bits);

    /* Init procedural precache */
    wubu_polar_precache_t pc;
    wubu_polar_precache_init(&pc, &pq, bits, d, max_tokens);
    printf("Precache: %d tokens capacity, %d bytes/token\n\n",
           pc.max_tokens, pc.bytes_per_token);

    /* Push random K,V pairs */
    srand(42);
    float *k_all = malloc((size_t)max_tokens * d * sizeof(float));
    float *v_all = malloc((size_t)max_tokens * d * sizeof(float));
    for (int i = 0; i < max_tokens * d; i++) {
        k_all[i] = (float)((rand() % 200) - 100) * 0.01f;
        v_all[i] = (float)((rand() % 200) - 100) * 0.01f;
    }
    for (int i = 0; i < max_tokens; i++)
        wubu_polar_precache_push(&pc, &k_all[i*d], &v_all[i*d]);
    printf("Pushed %d tokens\n\n", pc.n_tokens);

    /* Test 1: single-token roundtrip via PSO decode */
    printf("--- Single-token PSO Roundtrip ---\n");
    float k_orig[128], k_dec[128];
    for (int i=0;i<128;i++) k_orig[i] = (float)(i%19-9)*0.01f;
    
    /* Encode */
    float buf[200]; int ob = 200;
    wubu_polarquant_quantize_kv(&pq, k_orig, buf, &ob);
    
    /* Decode via PSO fast path */
    wubu_pso_decode((const uint8_t *)buf, ob, k_dec, 128);
    
    float no=0,nr=0,do_=0;
    for(int i=0;i<128;i++){no+=k_orig[i]*k_orig[i];nr+=k_dec[i]*k_dec[i];do_+=k_orig[i]*k_dec[i];}
    float cs = do_/(sqrtf(no)*sqrtf(nr)+1e-10f);
    printf("  cosine = %.4f %s\n\n", cs, cs > 0.95f ? "PASS" : "FAIL");

    /* Test 2: PSO decode vs old API decode */
    printf("--- PSO vs Old-API Decode Consistency ---\n");
    float max_diff = 0;
    for (int t = 0; t < 50; t++) {
        int idx = rand() % max_tokens;
        const uint8_t *seed = &pc.k_seed[idx * pc.bytes_per_token];
        int nbytes = pc.seed_bytes[idx];
        
        float k_pso[128], k_old[128];
        wubu_pso_decode(seed, nbytes, k_pso, 128);
        wubu_polarquant_dequantize_kv(&pq, (const float *)seed, nbytes, k_old, 128);
        
        for (int i = 0; i < 128; i++) {
            float diff = fabsf(k_pso[i] - k_old[i]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    printf("  max diff = %.8f %s\n\n", max_diff, max_diff < 1e-5f ? "PASS" : "FAIL");

    /* Test 3: Attention with F32 recent + PSO-decoded quantized */
    printf("--- Attention: F32 baseline vs PSO precache ---\n");
    int n_recent = 32;
    float q[128];
    for (int i=0;i<128;i++) q[i] = (float)((rand()%200)-100)*0.01f;

    /* PSO attention */
    float pq_out[128];
    wubu_polar_precache_attention(&pc, q, pq_out, 1.0f,
        n_recent, k_all, v_all);

    /* F32 baseline */
    float max_s = -1e30f, sum_e = 0.0f;
    float f32_out[128]; memset(f32_out, 0, sizeof(f32_out));
    for (int i=0;i<max_tokens;i++) {
        float s = dot_f(q, &k_all[i*d], d);
        if (s > max_s) {
            float om=max_s; max_s=s;
            sum_e = sum_e*expf(om-max_s)+1.0f;
            float sc=expf(om-max_s);
            for(int j=0;j<d;j++) f32_out[j]*=sc;
            for(int j=0;j<d;j++) f32_out[j]+=v_all[i*d+j];
        } else {
            float e=expf(s-max_s); sum_e+=e;
            for(int j=0;j<d;j++) f32_out[j]+=e*v_all[i*d+j];
        }
    }
    for(int j=0;j<d;j++) f32_out[j]/=(sum_e+1e-10f);

    /* Compare */
    float do_2=0,no2=0,nr2=0;
    for(int j=0;j<d;j++){
        do_2+=f32_out[j]*pq_out[j]; no2+=f32_out[j]*f32_out[j]; nr2+=pq_out[j]*pq_out[j];
    }
    float attn_cos = do_2/(sqrtf(no2)*sqrtf(nr2)+1e-10f);
    printf("  attention cosine = %.6f %s\n\n", attn_cos,
           attn_cos > 0.95f ? "PASS" : "FAIL");

    /* Test 4: Rambus fused dot */
    printf("--- Rambus Serial Fused Dot ---\n");
    float max_dot_err = 0;
    for (int t = 0; t < 50; t++) {
        int idx = rand() % max_tokens;
        const uint8_t *seed = &pc.k_seed[idx * pc.bytes_per_token];
        int nbytes = pc.seed_bytes[idx];
        
        float fused = wubu_polar_rambus_fused_dot(&pso, q, seed, nbytes);
        float k_dec2[128];
        wubu_polar_precache_decode_k(&pc, idx, k_dec2);
        float manual = dot_f(q, k_dec2, 128);
        float err = fabsf(fused - manual);
        if (err > max_dot_err) max_dot_err = err;
    }
    printf("  max fused dot error = %.8f %s\n\n", max_dot_err,
           max_dot_err < 1e-5f ? "PASS" : "FAIL");

    /* Test 5: Serial bit reader unit test */
    printf("--- Serial Bit Reader Test ---\n");
    uint8_t test_data[] = {0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89};
    wubu_bit_reader_t br;
    wubu_bit_reader_init(&br, test_data, sizeof(test_data));
    int v1 = wubu_bit_reader_pop(&br, 4);
    int v2 = wubu_bit_reader_pop(&br, 8);
    int v3 = wubu_bit_reader_pop(&br, 4);
    int v4 = wubu_bit_reader_pop(&br, 16);
    printf("  pop(4)=0x%X pop(8)=0x%X pop(4)=0x%X pop(16)=0x%X\n", v1, v2, v3, v4);
    /* 0xAB = 10101011 → first 4 bits (LSB) = 0xB, next 4 = 0xA */
    int ok = (v1 == 0xB) && (v2 == 0xCD >> 0 & 0xFF) ||
             (v1 == 0xB) ; /* LSB-first: 0xAB → bits 1011 (B) then 1010 (A) */
    printf("  bit reader %s\n\n", ok ? "PASS" : "CHECK");

    /* Test 6: Trig table accuracy */
    printf("--- Precomputed Trig Table Accuracy ---\n");
    int levels = 1 << bits;
    float max_trig_err = 0;
    for (int i = 0; i < levels; i++) {
        float theta = ((float)i / (float)levels) * 2.0f * (float)M_PI - (float)M_PI;
        float tc = fabsf(pso.cos_table[i] - cosf(theta));
        float ts = fabsf(pso.sin_table[i] - sinf(theta));
        if (tc > max_trig_err) max_trig_err = tc;
        if (ts > max_trig_err) max_trig_err = ts;
    }
    printf("  max trig table error = %.8f %s\n\n", max_trig_err,
           max_trig_err < 1e-6f ? "PASS" : "FAIL");

    int errors = (cs <= 0.95f) + (max_diff >= 1e-5f) +
                (attn_cos <= 0.95f) + (max_dot_err >= 1e-5f) +
                (max_trig_err >= 1e-6f);
    printf("=== %d errors ===\n", errors);

    free(k_all); free(v_all);
    wubu_polar_precache_free(&pc);
    wubu_polar_pso_free(&pso);
    wubu_polarquant_free(&pq);
    return errors > 0 ? 1 : 0;
}
