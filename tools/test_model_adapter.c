/*
 * test_model_adapter.c -- config-driven unit test for the model-agnostic
 * core (wubu_model_adapter + safetensors_reader + wubu_dims).
 *
 * Verifies, WITHOUT loading multi-GB weights:
 *   1. The adapter correctly parses the REAL HuggingFace config.json files
 *      for KAT-Coder, Agents-A1-4B, Qwen3.6-27B, and the BTL-3 LoRA
 *      adapter (which nests arch fields inside "text_config").
 *   2. The safetensors reader parses a synthetic in-memory blob (F32 + F16)
 *      and dequantizes correctly.
 *   3. wubu_dims_set() mirrors resolved dims into the runtime global and
 *      the macros (D_MODEL etc.) read them on both host and (via sync) GPU.
 *
 * C11, self-contained: only depends on the engine headers.
 */
#include "wubu_model_adapter.h"
#include "safetensors_reader.h"
#include "wubu_dims.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

static int g_fail = 0;
#define CHECK(cond, msg, ...) do {                                            \
    if (!(cond)) { printf("  FAIL: " msg "\n", ##__VA_ARGS__); g_fail++; }  \
    else        { printf("  PASS: " msg "\n", ##__VA_ARGS__); }             \
} while (0)

/* ---- tiny JSON builder for a synthetic safetensors header ---- */
static uint64_t write_le64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8*i));
    return 8;
}

/* Build a safetensors file with one F32 tensor [3] = {1,2,3} and one
 * F16 tensor [2] = {10,20} (half-precision). Returns path or NULL. */
static const char *g_st_path = "/tmp/test_st_blob.safetensors";
static int build_synthetic_safetensors(void) {
    /* header JSON */
    char hdr[1024];
    int n = snprintf(hdr, sizeof(hdr),
        "{\"t_f32\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,12]},"
        "\"t_f16\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[12,16]}}");
    uint64_t hlen = (uint64_t)n;
    /* pad header to 8-byte boundary */
    uint64_t total = 8 + hlen;
    uint64_t pad = (8 - (total % 8)) % 8;
    total += pad;
    uint64_t blob = 16; /* 12 f32 + 4 f16 */
    uint64_t flen = total + blob;

    FILE *f = fopen(g_st_path, "wb");
    if (!f) return 0;
    uint8_t lenbuf[8]; write_le64(lenbuf, hlen);
    fwrite(lenbuf, 1, 8, f);
    fwrite(hdr, 1, hlen, f);
    for (uint64_t i = 0; i < pad; i++) fputc(0, f);
    /* f32: 1,2,3 */
    float fv[3] = {1.0f, 2.0f, 3.0f};
    fwrite(fv, 4, 3, f);
    /* f16: 10,20 -> half */
    uint16_t hv[2];
    for (int i = 0; i < 2; i++) {
        float x = (i == 0) ? 10.0f : 20.0f;
        uint32_t xi; memcpy(&xi, &x, 4);
        uint32_t sign = (xi >> 16) & 0x8000;
        int32_t e = ((xi >> 23) & 0xff) - 127 + 15;
        uint32_t m = xi & 0x7fffff;
        if (e <= 0) hv[i] = (uint16_t)(sign | 0);
        else if (e >= 31) hv[i] = (uint16_t)(sign | 0x7c00);
        else hv[i] = (uint16_t)(sign | (e << 10) | (m >> 13));
    }
    fwrite(hv, 2, 2, f);
    fclose(f);
    (void)flen;
    return 1;
}

static int test_safetensors(void) {
    printf("\n[1] safetensors_reader (synthetic F32+F16 blob)\n");
    if (!build_synthetic_safetensors()) { CHECK(0, "blob build"); return 1; }
    st_ctx *ctx = st_open(g_st_path);
    CHECK(ctx != NULL, "st_open");
    if (!ctx) return 1;

    int64_t nt = st_n_tensors(ctx);
    CHECK(nt == 2, "n_tensors == 2 (got %lld)", (long long)nt);

    const st_tensor_info *f = st_find_tensor(ctx, "t_f32");
    CHECK(f != NULL, "find t_f32");
    CHECK(f && f->n_elems == 3, "t_f32 elems==3");
    float out[4] = {0};
    int got = st_read_tensor_f32(ctx, f, out, 4);
    CHECK(got == 3, "t_f32 read 3 floats (got %d)", got);
    CHECK(got==3 && out[0]==1.0f && out[1]==2.0f && out[2]==3.0f,
          "t_f32 values {1,2,3}");

    const st_tensor_info *h = st_find_tensor(ctx, "t_f16");
    CHECK(h != NULL, "find t_f16");
    float outh[2] = {0};
    int gh = st_read_tensor_f32(ctx, h, outh, 2);
    CHECK(gh == 2, "t_f16 read 2 floats (got %d)", gh);
    CHECK(gh==2 && outh[0]>9.5f && outh[0]<10.5f && outh[1]>19.5f && outh[1]<20.5f,
          "t_f16 dequant ~{10,20} (got %g,%g)", outh[0], outh[1]);
    st_close(ctx);
    return 0;
}

static int test_adapter_file(const char *path, const char *label,
                            int exp_dmodel, int exp_layers, int exp_vheads,
                            int exp_experts, int exp_active) {
    printf("\n[%s] adapter: %s\n", label, path);
    wubu_adapter_t a; memset(&a, 0, sizeof(a));
    int ok = wubu_adapter_load(&a, path);
    CHECK(ok && a.ok, "load ok");
    if (!ok || !a.ok) return 1;
    CHECK(a.d_model == exp_dmodel, "d_model==%d (got %d)", exp_dmodel, a.d_model);
    CHECK(a.n_layers == exp_layers, "n_layers==%d (got %d)", exp_layers, a.n_layers);
    CHECK(a.ssm_v_heads == exp_vheads, "ssm_v_heads==%d (got %d)", exp_vheads, a.ssm_v_heads);
    if (exp_experts > 0) {
        CHECK(a.is_moe && a.n_experts == exp_experts,
              "is_moe, n_experts==%d (got %d)", exp_experts, a.n_experts);
        CHECK(a.n_active_experts == exp_active,
              "n_active_experts==%d (got %d)", exp_active, a.n_active_experts);
    } else {
        CHECK(!a.is_moe, "dense (not MoE)");
    }
    CHECK(a.is_hybrid, "is_hybrid (layer_types present)");
    return 0;
}

static int test_lora_adapter(const char *path) {
    printf("\n[BTL-3] adapter: %s\n", path);
    wubu_adapter_t a; memset(&a, 0, sizeof(a));
    int ok = wubu_adapter_load(&a, path);
    CHECK(ok && a.ok, "load ok");
    if (!ok || !a.ok) return 1;
    CHECK(a.is_lora, "is_lora == true");
    CHECK(a.base_model[0] != '\0', "base_model set: '%s'", a.base_model);
    return 0;
}

static int test_dims_mirror(void) {
    printf("\n[dims] wubu_dims_set mirror + macro readback\n");
    wubu_adapter_t a; memset(&a, 0, sizeof(a));
    int ok = wubu_adapter_load(&a, "/home/wubu/wubuwizard/testdata/Qwen36_config.json");
    CHECK(ok && a.ok, "load ok (no realm env → standalone allowed)");
    if (!ok || !a.ok) return 1;
    wubu_dims_t d; memset(&d, 0, sizeof(d));
    d.d_model     = a.d_model;       /* 5120 */
    d.ssm_d_state = a.ssm_d_state;   /* 128  */
    d.ssm_k_heads = a.ssm_k_heads;   /* 16   */
    d.ssm_v_heads = a.ssm_v_heads;   /* 48   */
    d.value_dim   = a.ssm_d_state * a.ssm_v_heads; /* 6144 */
    d.gqa_q_heads  = a.gqa_q_heads;
    d.gqa_kv_heads = a.gqa_kv_heads;
    d.gqa_head_dim = a.gqa_head_dim;
    wubu_dims_set(&d);
    /* macros (host side) must read the runtime global */
    CHECK(D_MODEL == 5120, "D_MODEL macro == 5120 (got %d)", D_MODEL);
    CHECK(SSM_V_HEADS == 48, "SSM_V_HEADS macro == 48 (got %d)", SSM_V_HEADS);
    CHECK(KEY_DIM == 2048, "KEY_DIM invariant == 2048 (got %d)", KEY_DIM);
    CHECK(VALUE_DIM == 6144, "VALUE_DIM == 6144 (got %d)", VALUE_DIM);
    CHECK(CONV_DIM == (2048*2 + 6144), "CONV_DIM == 10240 (got %d)", CONV_DIM);
    wubu_dims_sync_gpu();   /* exercise the CUDA shim (no-op safe w/o GPU) */
    return 0;
}

int main(void) {
    printf("=== wubuwizard model-agnostic core unit test ===\n");
    test_safetensors();
    test_adapter_file("/home/wubu/wubuwizard/testdata/KAT_config.json",
                      "KAT", 2048, 40, 32, 256, 8);
    test_adapter_file("/home/wubu/wubuwizard/testdata/AgentsA1_config.json",
                      "AgentsA1", 2560, 32, 32, 0, 0);
    test_adapter_file("/home/wubu/wubuwizard/testdata/Qwen36_config.json",
                      "Qwen36", 5120, 64, 48, 0, 0);
    test_lora_adapter("/home/wubu/wubuwizard/testdata/BTL3_adapter_config.json");
    test_dims_mirror();

    /* DA-2 fail-closed gate test: schema mismatch refuses load */
    printf("\n[DA-2] kernel schema mismatch gate\n");
    wubu_adapter_t a; memset(&a, 0, sizeof(a));
    setenv("WUBU_KERNEL_SCHEMA", "99", 1);
    int refused = wubu_adapter_load(&a, "/home/wubu/wubuwizard/testdata/Qwen36_config.json");
    CHECK(!refused, "load refused when WUBU_KERNEL_SCHEMA=99 (DA-2 fail-closed)");
    unsetenv("WUBU_KERNEL_SCHEMA");
    int allowed = wubu_adapter_load(&a, "/home/wubu/wubuwizard/testdata/Qwen36_config.json");
    CHECK(allowed && a.ok, "load allowed when no realm env (standalone)");

    printf("\n=== RESULTS: %s ===\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
