/* test_gguf_tq.c — TurboQuant loader unit test (Q2_0 remap + TQ3_1S dequant)
 * Builds a synthetic GGUF v3 by hand with two tensors:
 *   t0: 64 elems, type 42 (legacy Q2_0 alias -> must remap to 47)
 *   t1: 32 elems, type 45 (TQ3_1S, all indices 0, d0=d1=2.0)
 * Verifies: remap works, sizes are exact, dequant outputs match the
 * reference math (constant -> inverse-WHT -> delta at index 0).
 * PASS criterion: all asserts + "ALL TQ TESTS PASSED".
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void put_u32(unsigned char *p, uint32_t v){ memcpy(p,&v,4); }
static void put_i32(unsigned char *p, int32_t v){ memcpy(p,&v,4); }
static void put_u64(unsigned char *p, uint64_t v){ memcpy(p,&v,8); }
static void put_i64(unsigned char *p, int64_t v){ memcpy(p,&v,8); }

int main(void){
    /* ---- craft the synthetic GGUF ----
       header: magic GGUF, ver 3, 2 tensors, 0 KV, alignment 32
       then 2 tensor infos (name, ndims=1, dims, type, data_offset)
       then data blob: t0 at 0 (18B), t1 at 32 (16B)                 */
    unsigned char buf[4096];
    memset(buf, 0, sizeof(buf));
    size_t o = 0;
    memcpy(buf+o, "GGUF", 4); o += 4;
    put_u32(buf+o, 3); o += 4;          /* version */
    put_u64(buf+o, 2); o += 8;          /* n_tensors */
    put_u64(buf+o, 0); o += 8;          /* n_kv (reader hardcodes alignment=32) */

    /* t0: "t0.q2" 64 elems, type 42 (legacy Q2_0) */
    const char *n0 = "t0.q2"; 
    put_u64(buf+o, strlen(n0)); o += 8; memcpy(buf+o, n0, strlen(n0)); o += strlen(n0);
    put_u32(buf+o, 1); o += 4;          /* ndims */
    put_i64(buf+o, 64); o += 8;         /* dims[0] */
    put_i32(buf+o, 42); o += 4;         /* type: legacy Q2_0 alias */
    put_u64(buf+o, 0); o += 8;          /* data_offset */
    /* t1: "t1.tq3" 32 elems, type 45 (TQ3_1S) */
    const char *n1 = "t1.tq3";
    put_u64(buf+o, strlen(n1)); o += 8; memcpy(buf+o, n1, strlen(n1)); o += strlen(n1);
    put_u32(buf+o, 1); o += 4;
    put_i64(buf+o, 32); o += 8;
    put_i32(buf+o, 45); o += 4;
    put_u64(buf+o, 32); o += 8;         /* data_offset (aligned) */

    /* pad data start to 32 */
    while (o % 32) o++;
    /* t0 data: d = 0.5 (fp16 0x3800), qs byte0 = 0b00111001 (q: 01,10,11,00) */
    unsigned char *d0 = buf + o;
    put_u16_bits: { uint16_t h = 0x3800; memcpy(d0, &h, 2); }
    d0[2] = 0x39;
    /* t1 data at o+32: d0=d1=2.0 (fp16 0x4000), all qs bytes 0 (idx 0) */
    unsigned char *d1b = buf + o + 32;
    { uint16_t h = 0x4000; memcpy(d1b, &h, 2); memcpy(d1b+2, &h, 2); }
    size_t file_len = o + 48;
    FILE *f = fopen("/tmp/synth_tq.gguf", "wb");
    fwrite(buf, 1, file_len, f);
    fclose(f);

    gguf_ctx *ctx = gguf_open("/tmp/synth_tq.gguf");
    if (!ctx) { fprintf(stderr, "FAIL: gguf_open\n"); return 1; }
    gguf_tensor_info *t0 = gguf_find_tensor(ctx, "t0.q2");
    gguf_tensor_info *t1 = gguf_find_tensor(ctx, "t1.tq3");
    if (!t0 || !t1) { fprintf(stderr, "FAIL: find tensors\n"); return 1; }

    /* 1. remap check: type 42 must have been remapped to 47 */
    if (t0->ggml_type != 47) { fprintf(stderr, "FAIL: remap 42->47 (got %d)\n", t0->ggml_type); return 1; }
    printf("ok: legacy type 42 remapped to Q2_0 (47)\n");

    /* 2. Q2_0 dequant: 64 elems; byte0 -> q=[01,10,11,00] -> [0,+1,+2,-1]*0.5 */
    float *q = (float*)calloc(64, sizeof(float));
    int n = gguf_read_tensor_f32(ctx, t0, q, 64);
    if (n != 64) { fprintf(stderr, "FAIL: q2_0 read %d\n", n); return 1; }
    float exp0[4] = { 0.0f, 0.5f, 1.0f, -0.5f };
    for (int j = 0; j < 4; j++)
        if (fabsf(q[j] - exp0[j]) > 1e-4f) { fprintf(stderr, "FAIL: q2_0[%d]=%f want %f\n", j, q[j], exp0[j]); return 1; }
    /* remaining 60 elems: q=0 -> -0.5 */
    for (int j = 4; j < 64; j++)
        if (fabsf(q[j] + 0.5f) > 1e-4f) { fprintf(stderr, "FAIL: q2_0[%d]=%f want -0.5\n", j, q[j]); return 1; }
    printf("ok: Q2_0 dequant exact (d=0.5, {-1,0,+1,+2}*d)\n");

    /* 3. TQ3_1S dequant: all idx 0, d0=d1=2.0 -> constant c = centroid[0]*2
          inverse-WHT of a constant c -> [c*sqrt(32), 0, ...] */
    float *t = (float*)calloc(32, sizeof(float));
    n = gguf_read_tensor_f32(ctx, t1, t, 32);
    if (n != 32) { fprintf(stderr, "FAIL: tq3 read %d\n", n); return 1; }
    float c = -1.996684f * 2.0f;
    float want0 = c * (float)sqrt(32.0);
    if (fabsf(t[0] - want0) > 1e-3f) { fprintf(stderr, "FAIL: tq3[0]=%f want %f\n", t[0], want0); return 1; }
    for (int j = 1; j < 32; j++)
        if (fabsf(t[j]) > 1e-3f) { fprintf(stderr, "FAIL: tq3[%d]=%f want 0\n", j, t[j]); return 1; }
    printf("ok: TQ3_1S inverse-RHT math exact (constant -> delta * sqrt(32))\n");

    /* 4. raw sizes match the offset-derived spans (no fallback needed) */
    int64_t rs0 = gguf_raw_size(t0->ggml_type, 64);
    int64_t rs1 = gguf_raw_size(t1->ggml_type, 32);
    if (rs0 != 18 || rs1 != 16) { fprintf(stderr, "FAIL: raw sizes %ld %ld\n", (long)rs0, (long)rs1); return 1; }
    if (ctx->tensor_raw_bytes[0] != 32 || ctx->tensor_raw_bytes[1] != 16) {
        fprintf(stderr, "FAIL: spans %ld %ld\n", (long)ctx->tensor_raw_bytes[0], (long)ctx->tensor_raw_bytes[1]); return 1;
    }
    printf("ok: Q2_0 18B/64 + TQ3_1S 16B/32 sizes, spans match\n");

    free(q); free(t); gguf_close(ctx);
    printf("ALL TQ TESTS PASSED\n");
    return 0;
}
