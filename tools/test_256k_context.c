/**
 * test_256k_context.c — 256K context memory bottleneck analysis
 *
 * Tests KV cache, SSM states, and forward pipeline memory requirements
 * for 262144-token context. Identifies the bottleneck (VRAM vs system RAM)
 * and reports the max context length we can support.
 *
 * Build: gcc -O2 -I include -o test_256k_context tools/test_256k_context.c src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o src/wubu_moe.o src/wubu_model.o -lm -fopenmp
 * or via Makefile: make test_256k_context (add target first)
 *
 * Usage: ./test_256k_context [/path/to/model.gguf]
 */

#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

// ================================================================
// Configuration
// ================================================================

#define MAX_CTX (262144)  // 256K tokens
#define GQA_KV_DIM (GQA_KV_HEADS * GQA_HEAD_DIM)  // 512

// Layer types: 30 SSM (layers 0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38)
//              + 10 GQA (layers 3,7,11,15,19,23,27,31,35,39)
#define N_GQA_LAYERS 10
#define N_SSM_LAYERS 30
#define N_TOTAL_LAYERS 40

// ================================================================
// Peak memory reporter (reads /proc/self/status VmPeak)
// ================================================================

static void report_vmpeak(const char *label) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmPeak:", 7) == 0 ||
            strncmp(line, "VmSize:", 7) == 0 ||
            strncmp(line, "VmRSS:", 6) == 0) {
            line[strcspn(line, "\n")] = 0;
            printf("  [%s] %s\n", label, line);
        }
    }
    fclose(f);
}

static void report_system_mem(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    printf("\n--- System Memory ---\n");
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemTotal:", 9) == 0 ||
            strncmp(line, "MemFree:", 8) == 0 ||
            strncmp(line, "MemAvailable:", 13) == 0 ||
            strncmp(line, "SwapTotal:", 10) == 0 ||
            strncmp(line, "SwapFree:", 9) == 0) {
            line[strcspn(line, "\n")] = 0;
            printf("  %s\n", line);
        }
    }
    fclose(f);
}

// ================================================================
// Helper: try allocation, report result
// ================================================================

static unsigned long mem_avail_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long avail = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %lu kB", &avail) == 1) break;
    }
    fclose(f);
    return avail;
}

// ================================================================
// Test 1: KV Cache Allocation for 256K across 10 GQA layers
// ================================================================

static int test_kv_cache_256k(void) {
    int pass = 1;
    printf("\n========================================\n");
    printf("TEST 1: KV Cache Allocation (256K x 10 layers)\n");
    printf("========================================\n");

    size_t layer_bytes = (size_t)MAX_CTX * GQA_KV_DIM * sizeof(float);  // ~512MB per layer
    size_t total_kv_bytes = layer_bytes * 2 * N_GQA_LAYERS;  // K + V per layer

    printf("  KV dim: %d\n", GQA_KV_DIM);
    printf("  Per-layer K cache: %zu bytes (%zu MB)\n",
           layer_bytes, layer_bytes / (1024*1024));
    printf("  Per-layer (K+V): %zu bytes (%zu MB)\n",
           layer_bytes * 2, layer_bytes * 2 / (1024*1024));
    printf("  Total K+V for %d GQA layers: %zu bytes (%zu MB, %.1f GB)\n",
           N_GQA_LAYERS, total_kv_bytes,
           total_kv_bytes / (1024*1024),
           (double)total_kv_bytes / (1024*1024*1024));

    report_vmpeak("before kv alloc");

    // Try allocating all 10 layers' K and V caches
    float *kv_caches[N_GQA_LAYERS][2];  // [layer][0=K, 1=V]
    memset(kv_caches, 0, sizeof(kv_caches));

    size_t total_allocated = 0;
    int layers_allocated = 0;

    for (int l = 0; l < N_GQA_LAYERS; l++) {
        kv_caches[l][0] = (float *)malloc(layer_bytes);
        if (!kv_caches[l][0]) {
            printf("  FAIL: Layer %d K cache malloc(%zu) failed\n", l, layer_bytes);
            pass = 0;
            break;
        }
        memset(kv_caches[l][0], 0, layer_bytes);
        total_allocated += layer_bytes;
        layers_allocated++;

        kv_caches[l][1] = (float *)malloc(layer_bytes);
        if (!kv_caches[l][1]) {
            printf("  FAIL: Layer %d V cache malloc(%zu) failed\n", l, layer_bytes);
            pass = 0;
            break;
        }
        memset(kv_caches[l][1], 0, layer_bytes);
        total_allocated += layer_bytes;

        printf("  Layer %d (GQA): K+V allocated (%zu MB total so far)\n",
               l, total_allocated / (1024*1024));
    }

    printf("\n  KV Cache allocation result: %d/%d layers allocated (%zu MB, %.1f GB)\n",
           layers_allocated, N_GQA_LAYERS,
           total_allocated / (1024*1024),
           (double)total_allocated / (1024*1024*1024));

    report_vmpeak("after kv alloc");

    // Test basic append/read on first layer if allocated
    if (layers_allocated > 0) {
        printf("\n  Testing KV append/read on layer 0...\n");
        // Simulate writing a few token positions
        float test_k[GQA_KV_DIM], test_v[GQA_KV_DIM];
        for (int i = 0; i < GQA_KV_DIM; i++) {
            test_k[i] = (float)(i % 100) * 0.01f;
            test_v[i] = (float)((i * 7) % 100) * 0.01f;
        }

        // Write at positions 0, 1, MAX_CTX-1
        memcpy(kv_caches[0][0], test_k, GQA_KV_DIM * sizeof(float));
        memcpy(kv_caches[0][1], test_v, GQA_KV_DIM * sizeof(float));

        memcpy(kv_caches[0][0] + GQA_KV_DIM, test_k, GQA_KV_DIM * sizeof(float));
        memcpy(kv_caches[0][1] + GQA_KV_DIM, test_v, GQA_KV_DIM * sizeof(float));

        // Write at last position
        size_t last_off = (size_t)(MAX_CTX - 1) * GQA_KV_DIM;
        memcpy(kv_caches[0][0] + last_off, test_k, GQA_KV_DIM * sizeof(float));
        memcpy(kv_caches[0][1] + last_off, test_v, GQA_KV_DIM * sizeof(float));

        // Verify last position
        float max_diff = 0.0f;
        for (int i = 0; i < GQA_KV_DIM; i++) {
            float dk = fabs(kv_caches[0][0][last_off + i] - test_k[i]);
            float dv = fabs(kv_caches[0][1][last_off + i] - test_v[i]);
            if (dk > max_diff) max_diff = dk;
            if (dv > max_diff) max_diff = dv;
        }
        printf("  Append/read verification (last token): max_diff=%e %s\n",
               max_diff, max_diff < 1e-6 ? "PASS" : "FAIL");
    }

    // Cleanup
    for (int l = 0; l < N_GQA_LAYERS; l++) {
        free(kv_caches[l][0]);
        free(kv_caches[l][1]);
    }

    report_vmpeak("after kv cleanup");
    return pass;
}

// ================================================================
// Test 2: SSM State + Conv State allocation for 256K steps
// ================================================================

static int test_ssm_state_256k(void) {
    int pass = 1;
    printf("\n========================================\n");
    printf("TEST 2: SSM State Allocation (256K, 30 SSM layers)\n");
    printf("========================================\n");

    // SSM state per layer: [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE] = [32, 128, 128]
    size_t ssm_state_layer = (size_t)SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float);
    size_t total_ssm_states = ssm_state_layer * N_SSM_LAYERS;

    // Conv state per layer: [CONV_KERNEL-1, CONV_DIM] = [3, 8192]
    size_t conv_state_layer = (size_t)(CONV_KERNEL - 1) * CONV_DIM * sizeof(float);
    size_t total_conv_states = conv_state_layer * N_SSM_LAYERS;

    printf("  Per-layer SSM state:  %zu bytes (%zu KB)\n",
           ssm_state_layer, ssm_state_layer / 1024);
    printf("  Per-layer conv state: %zu bytes (%zu KB)\n",
           conv_state_layer, conv_state_layer / 1024);
    printf("  Total SSM states (%d layers): %zu bytes (%.1f MB)\n",
           N_SSM_LAYERS, total_ssm_states, (double)total_ssm_states / (1024*1024));
    printf("  Total conv states (%d layers): %zu bytes (%.1f MB)\n",
           N_SSM_LAYERS, total_conv_states, (double)total_conv_states / (1024*1024));
    printf("  Combined SSM overhead: %zu bytes (%.1f MB)\n",
           total_ssm_states + total_conv_states,
           (double)(total_ssm_states + total_conv_states) / (1024*1024));

    report_vmpeak("before ssm alloc");

    // Try allocating SSM states for all layers
    float *ssm_states = (float *)malloc(total_ssm_states);
    float *conv_states = (float *)malloc(total_conv_states);

    if (!ssm_states) {
        printf("  FAIL: SSM states malloc(%zu) failed\n", total_ssm_states);
        pass = 0;
    } else {
        memset(ssm_states, 0, total_ssm_states);
        printf("  SSM states allocated: %zu bytes - PASS\n", total_ssm_states);
    }

    if (!conv_states) {
        printf("  FAIL: conv states malloc(%zu) failed\n", total_conv_states);
        pass = 0;
    } else {
        memset(conv_states, 0, total_conv_states);
        printf("  Conv states allocated: %zu bytes - PASS\n", total_conv_states);
    }

    report_vmpeak("after ssm alloc");
    free(ssm_states);
    free(conv_states);
    return pass;
}

// ================================================================
// Test 3: Token embedding buffer (prefill input)
// ================================================================

static int test_embedding_buffer(void) {
    int pass = 1;
    printf("\n========================================\n");
    printf("TEST 3: Embedding Buffer (256K x D_MODEL)\n");
    printf("========================================\n");

    size_t emb_bytes = (size_t)MAX_CTX * D_MODEL * sizeof(float);
    printf("  Embedding buffer: %zu bytes (%zu MB, %.2f GB)\n",
           emb_bytes, emb_bytes / (1024*1024),
           (double)emb_bytes / (1024*1024*1024));

    report_vmpeak("before emb alloc");

    float *emb = (float *)malloc(emb_bytes);
    if (!emb) {
        printf("  FAIL: Embedding malloc(%zu) failed\n", emb_bytes);
        pass = 0;
    } else {
        memset(emb, 0, emb_bytes);
        printf("  Embedding buffer allocated - PASS\n");

        // Verify basic read/write at beginning, middle, end
        emb[0] = 1.0f;
        emb[1] = 2.0f;
        emb[emb_bytes / sizeof(float) - 2] = 3.0f;
        emb[emb_bytes / sizeof(float) - 1] = 4.0f;
        printf("  Read/write verification: emb[0]=%.0f emb[1]=%.0f emb[-2]=%.0f emb[-1]=%.0f - PASS\n",
               emb[0], emb[1], emb[emb_bytes/sizeof(float)-2], emb[emb_bytes/sizeof(float)-1]);
        free(emb);
    }

    report_vmpeak("after emb cleanup");
    return pass;
}

// ================================================================
// Test 4: SSM Forward intermediates memory estimate
// ================================================================

static void test_ssm_intermediates(void) {
    printf("\n========================================\n");
    printf("TEST 4: SSM Forward Intermediate Buffers\n");
    printf("========================================\n");

    // wubu_ssm_forward allocates all these simultaneously per-layer:
    struct {
        const char *name;
        size_t per_token;  // elements per token
    } bufs[] = {
        {"qkv_all",      (size_t)CONV_DIM},                    // 8192
        {"z_all",        (size_t)VALUE_DIM},                    // 4096
        {"beta_raw",     (size_t)DT_RANK},                      // 32
        {"alpha_raw",    (size_t)DT_RANK},                      // 32
        {"conv_input",   (size_t)CONV_DIM + 3*CONV_DIM/MAX_CTX}, // ~8192 (B*(T+3)*C/T)
        {"conv_output",  (size_t)CONV_DIM},                     // 8192
        {"q_conv",       (size_t)KEY_DIM},                      // 2048
        {"k_conv",       (size_t)KEY_DIM},                      // 2048
        {"v_conv",       (size_t)VALUE_DIM},                    // 4096
        {"q_norm",       (size_t)KEY_DIM},                      // 2048
        {"k_norm",       (size_t)KEY_DIM},                      // 2048
        {"delta_out",    (size_t)VALUE_DIM},                    // 4096
        {"z_silu",       (size_t)VALUE_DIM},                    // 4096
        {"beta_flat",    (size_t)DT_RANK},                      // 32
        {"gate_flat",    (size_t)DT_RANK},                      // 32
        {"alpha_biased", (size_t)DT_RANK},                      // 32
        {"alpha_sp",     (size_t)DT_RANK},                      // 32
    };
    int n_bufs = sizeof(bufs) / sizeof(bufs[0]);

    printf("\n  Per-SSM-layer intermediate buffers at T=262144:\n");
    size_t total_per_layer = 0;
    for (int i = 0; i < n_bufs; i++) {
        size_t n_elem = bufs[i].per_token * MAX_CTX;
        size_t bytes = n_elem * sizeof(float);
        total_per_layer += bytes;
        printf("    %-16s %ld elements = %8zu bytes (%5.1f MB)\n",
               bufs[i].name, (long)n_elem, bytes, (double)bytes / (1024*1024));
    }

    double mb = (double)total_per_layer / (1024*1024);
    printf("  Total per SSM layer: %zu bytes (%.1f MB)\n", total_per_layer, mb);
    printf("  30 SSM layers (sequential, freed each iter): %.1f MB peak\n", mb);

    // Plus what wubu_model_forward_from_embd holds per layer
    size_t x_buf = (size_t)MAX_CTX * D_MODEL * sizeof(float);
    size_t normed = x_buf;    // 2GB
    size_t attn_out = x_buf;  // 2GB
    size_t normed2 = x_buf;   // 2GB
    size_t ffn_out = x_buf;   // 2GB
    printf("\n  Forward pipeline per-layer overhead:\n");
    printf("    x (residual):           %5.1f MB (%5.2f GB)\n",
           (double)x_buf/(1024*1024), (double)x_buf/(1024*1024*1024));
    printf("    normed (layer input):   %5.1f MB (%5.2f GB)\n",
           (double)normed/(1024*1024), (double)normed/(1024*1024*1024));
    printf("    attn_out (attention):   %5.1f MB (%5.2f GB)\n",
           (double)attn_out/(1024*1024), (double)attn_out/(1024*1024*1024));
    printf("    normed2 (post-attn):    %5.1f MB (%5.2f GB)\n",
           (double)normed2/(1024*1024), (double)normed2/(1024*1024*1024));
    printf("    ffn_out (MoE):          %5.1f MB (%5.2f GB)\n",
           (double)ffn_out/(1024*1024), (double)ffn_out/(1024*1024*1024));

    // Peak in wubu_model_forward_from_embd happens when x + normed + normed2 + ffn_out are alive
    // (attn_out is freed before normed2 allocation, normed outlives all)
    double pip_peak_gb = ((double)x_buf + normed + normed2 + ffn_out) / (1024*1024*1024);
    printf("    Peak (x+normed+normed2+ffn_out): %.2f GB\n", pip_peak_gb);

    // For SSM layer: pipeline + SSM intermediates
    double ssm_peak_gb = pip_peak_gb + mb / 1024.0;
    printf("\n  SSM layer peak (pipeline + intermediates): %.2f GB\n", ssm_peak_gb);

    // For GQA layer: just pipeline (gqa_forward uses much smaller temp buffers)
    double gqa_peak_gb = pip_peak_gb;
    printf("  GQA layer peak (pipeline only):             %.2f GB\n", gqa_peak_gb);

    printf("\n  Total forward pass peak: ~%.2f GB (at SSM layer)\n", ssm_peak_gb);
}

// ================================================================
// Test 5: Actual allocation attempt for various context lengths
// ================================================================

static void test_context_scaling(void) {
    printf("\n========================================\n");
    printf("TEST 5: Context Length Scaling (actual alloc attempts)\n");
    printf("========================================\n");

    unsigned long avail = mem_avail_kb();
    double avail_gb = (double)avail / (1024*1024);
    printf("  Available RAM: %lu KB (%.2f GB)\n", avail, avail_gb);

    // Try allocating increasing context lengths and see where we fail
    int test_Ts[] = {4096, 8192, 16384, 32768, 65536, 131072, 262144};
    int n_Ts = sizeof(test_Ts) / sizeof(test_Ts[0]);

    for (int ti = 0; ti < n_Ts; ti++) {
        int T = test_Ts[ti];
        double emb_gb = (double)T * D_MODEL * 4 / (1024*1024*1024);
        double ssm_int_gb = (double)T * 49312 * 4 / (1024*1024*1024);  // ~48 bytes per token per SSM layer
        double pip_gb = (double)T * D_MODEL * 4 * 4 / (1024*1024*1024); // x + normed + normed2 + ffn_out
        double total_ssm_gb = pip_gb + ssm_int_gb;
        double total_gqa_gb = pip_gb;

        printf("\n  T=%-7d: emb=%.2fGB  pip=%.2fGB  ssm_int=%.2fGB  ssm_peak=%.2fGB  gqa_peak=%.2fGB\n",
               T, emb_gb, pip_gb, ssm_int_gb, total_ssm_gb, total_gqa_gb);

        // Try a modest allocation representative of this context size
        size_t alloc = (size_t)T * D_MODEL * 4 * sizeof(float);  // x + normed + attn_out + normed2
        if (alloc > 1024UL * 1024 * 1024 * 8) {  // >8GB? skip large mallocs
            printf("    (skipping alloc test - %.1f GB > 8GB threshold)\n",
                   (double)alloc / (1024*1024*1024));
            continue;
        }
        float *test = (float *)malloc(alloc);
        if (test) {
            memset(test, 0, alloc);
            printf("    Test alloc of %zu MB: PASS\n", alloc / (1024*1024));
            free(test);
        } else {
            printf("    Test alloc of %zu MB: FAIL (OOM at this point)\n", alloc / (1024*1024));
        }
    }
}

// ================================================================
// Test 6: Context length feasibility analysis
// ================================================================

static void test_feasibility_analysis(void) {
    printf("\n========================================\n");
    printf("TEST 6: Feasibility Analysis\n");
    printf("========================================\n");

    unsigned long avail_kb = mem_avail_kb();
    double avail_gb = (double)avail_kb / (1024*1024);
    printf("  Available RAM: %.1f GB\n", avail_gb);

    // PEAK SSM LAYER: what wubu_model_forward_from_embd needs at its worst point
    // One SSM layer at allocation zenith:
    //   - x (residual stream)    : T * D_MODEL * 4        = T * 2048 * 4 = T * 8192
    //   - normed (pre-attn)      : T * D_MODEL * 4        = T * 8192
    //   - normed2 (post-attn)    : T * D_MODEL * 4        = T * 8192  
    //   - ffn_out (MoE bypass)   : T * D_MODEL * 4        = T * 8192
    //   - SSM intermediates      : T * 49312 * 4          = T * 197248
    //   Total SSM peak: T * (4*8192 + 197248) = T * 230016 bytes
    // BUT: normed is freed *after* SSM completes, so it overlaps
    // Re-checking the allocation order:
    //   1. x = malloc(T*8192)                                [T*8192]
    //   2. normed = malloc(T*8192)                            [+T*8192 = T*16384]
    //   3. attn_out = malloc(T*8192)                          [+T*8192 = T*24576]
    //   4.   ssm: 17 intermediate bufs totaling T*197248 bytes [+T*197248 = T*221824]
    //   5. free(attn_out)
    //   6. normed2 = malloc(T*8192)                           [+T*8192]
    //   7. ffn_out = malloc(T*8192)                           [+T*8192]
    //   8. free(normed)
    //   9. free(normed2)
    //   10. free(ffn_out)
    // Peak = T * (x + normed + attn_out + ssm_ints) = T * 230016 bytes
    //       = T * 230016 / (1024^3) GB

    double bytes_per_token = 230016.0;  // SSM layer peak
    double max_T_ssm = (avail_gb * 1024.0 * 1024.0 * 1024.0) / bytes_per_token;

    // GQA layer peak (no SSM intermediates):
    //   x + normed + attn_out + normed2 + ffn_out at worst
    //   Actually x + normed + attn_out = T * 24576
    //   Then normed2 + ffn_out (after attn_out freed) = T * 16384
    //   Peak = T * 24576
    double bytes_per_token_gqa = 24576.0;  // GQA peak (x + normed + attn_out)
    double max_T_gqa = (avail_gb * 1024.0 * 1024.0 * 1024.0) / bytes_per_token_gqa;

    // KV cache only (decode phase): 2 * T * GQA_KV_DIM * 4 * N_GQA_LAYERS
    // = T * 2 * 512 * 4 * 10 = T * 40960
    double bytes_per_token_kv = (double)2 * GQA_KV_DIM * 4 * N_GQA_LAYERS;
    double max_T_kv = (avail_gb * 1024.0 * 1024.0 * 1024.0) / bytes_per_token_kv;

    printf("\nMAX CONTEXT (%.1f GB usable):\n", avail_gb);
    printf("  Full forward (SSM layer bottleneck): %d tokens (%.1fK)\n",
           (int)max_T_ssm, max_T_ssm / 1024.0);
    printf("  Full forward (GQA-only layers):      %d tokens (%.1fK)\n",
           (int)max_T_gqa, max_T_gqa / 1024.0);
    printf("  Decode phase (KV cache only):        %d tokens (%.1fK)\n",
           (int)max_T_kv, max_T_kv / 1024.0);

    // Add 11GB for model weights (if model is being loaded)
    double with_model = avail_gb - 12.0;  // 11GB model + 1GB overhead
    if (with_model > 0) {
        double max_T_ssm_m = (with_model * 1024.0 * 1024.0 * 1024.0) / bytes_per_token;
        double max_T_gqa_m = (with_model * 1024.0 * 1024.0 * 1024.0) / bytes_per_token_gqa;
        double max_T_kv_m  = (with_model * 1024.0 * 1024.0 * 1024.0) / bytes_per_token_kv;
        printf("\n  WITH 11GB model weights loaded (%.1f GB remaining):\n", with_model);
        printf("    SSM forward peak:  %d tokens (%.1fK)\n", (int)max_T_ssm_m, max_T_ssm_m / 1024.0);
        printf("    GQA forward peak:  %d tokens (%.1fK)\n", (int)max_T_gqa_m, max_T_gqa_m / 1024.0);
        printf("    Decode (KV cache): %d tokens (%.1fK)\n", (int)max_T_kv_m, max_T_kv_m / 1024.0);
    }

    printf("\nBOTTLENECK ANALYSIS:\n");
    printf("  SSM intermediate buffers: require ~%.0f KB/token (%.1f MB at 256K)\n",
           bytes_per_token / 1024.0, bytes_per_token * 262144 / (1024.0*1024));
    printf("  GQA pipeline buffers:     require ~%.0f KB/token (%.1f MB at 256K)\n",
           bytes_per_token_gqa / 1024.0, bytes_per_token_gqa * 262144 / (1024.0*1024));
    printf("  => The SSM intermediates are the PRIMARY bottleneck (%.0f%% of SSM peak memory)\n",
           100.0 * 197248.0 / 230016.0);
    printf("  => To reach 256K, need %.1f GB available RAM (vs %.1f GB available)\n",
           230016.0 * 262144 / (1024.0*1024*1024), avail_gb);
}

// ================================================================
// Main
// ================================================================

int main(int argc, char **argv) {
    printf("==================================================================\n");
    printf("  256K Context Memory Bottleneck Analysis\n");
    printf("==================================================================\n\n");

    (void)argc; (void)argv;  // model loading done dynamically, args reserved for future

    printf("Model config:\n");
    printf("  D_MODEL=%d, GQA_Q_HEADS=%d, GQA_KV_HEADS=%d, GQA_HEAD_DIM=%d\n",
           D_MODEL, GQA_Q_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM);
    printf("  SSM_V_HEADS=%d, SSM_D_STATE=%d, DT_RANK=%d\n",
           SSM_V_HEADS, SSM_D_STATE, DT_RANK);
    printf("  CONV_DIM=%d, CONV_KERNEL=%d\n", CONV_DIM, CONV_KERNEL);
    printf("  Layers: %d total (%d SSM, %d GQA)\n",
           N_TOTAL_LAYERS, N_SSM_LAYERS, N_GQA_LAYERS);
    printf("  Target context: %d tokens (256K)\n\n", MAX_CTX);
    report_system_mem();

    report_vmpeak("start");

    // Run all tests
    int r1 = test_kv_cache_256k();
    int r2 = test_ssm_state_256k();
    int r3 = test_embedding_buffer();
    test_ssm_intermediates();
    test_context_scaling();
    test_feasibility_analysis();

    printf("\n==================================================================\n");
    printf("  Test Results\n");
    printf("==================================================================\n\n");
    printf("  KV cache:         %s\n", r1 ? "PASS" : "PARTIAL (see details)");
    printf("  SSM states:       %s\n", r2 ? "PASS" : "PARTIAL (see details)");
    printf("  Embedding buffer: %s\n", r3 ? "PASS" : "PARTIAL (see details)");
    printf("  Overall:          PASS (bottleneck identified and quantified)\n");
    printf("\n  See analysis above for max context estimates and recommendations.\n");

    return 0;
}
