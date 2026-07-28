/*
 * wubu_gemv_tune.c -- Roofline-driven GEMV auto-tuner (see header).
 * Pure C, routes through wubu_roofline for the B*-ridge decision.
 */
#include "wubu_gemv_tune.h"
#include "wubu_roofline.h"
#include <stdio.h>
#include <math.h>

static int cpu_has_avx512(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned a, b, c, d;
    __asm__ __volatile__("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "0"(7),"2"(0));
    if ((b & (1u<<16)) == 0) return 0;
    __asm__ __volatile__("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "0"(1));
    return ((c & (1u<<28)) && (c & (1u<<12))) ? 1 : 0;
#else
    return 0;
#endif
}

wubu_gemv_tile_t wubu_gemv_detect(void) {
    wubu_gemv_tile_t t;
    t.avx512 = cpu_has_avx512();
    t.k_unroll = t.avx512 ? 16 : 8;   /* match SIMD lane count */
    t.use_int8 = 0;
    return t;
}

wubu_gemv_tile_t wubu_gemv_autotune(int M, int K, double beta_eff_tb_s) {
    wubu_gemv_tile_t t = wubu_gemv_detect();

    /* Work per output element = 2*K flops, weight traffic = K * wbits/8 bytes.
     * The arithmetic intensity of one GEMV column is AI = (2K) / (K*wbits/8)
     * = 16/wbits flop/byte -- INDEPENDENT of K! So a single GEMV column is
     * always far below the machine ridge (~10-30 flop/byte for AVX2/AVX512
     * with good BW), i.e. always BW-bound. The int8 lever therefore ALWAYS
     * halves traffic and wins -- PROVIDED the requant cost (2K flops for the
     * absmax+quant of one weight row) is amortized. Requant is amortized when
     * M is large OR the same weight matrix is reused across tokens.
     *
     * Decision: use_int8 when M is large enough that the per-row requant
     * (2K flops) is dwarfed by the per-token GEMV work (2*M*K flops) -- i.e.
     * requant overhead < ~2% of the GEMV cost. That is M*K >> K  =>  M >> 1,
     * which is true for every real projection (M = d_model >= 512). The
     * roofline ridge only gates whether the *extra* int32->f32 dequant tail
     * matters; with AVX512 + good BW it does not. So enable int8 for any
     * realistic projection. */
    (void)beta_eff_tb_s; /* BW is in the ridge already; GEMV AI is BW-bound by
                          * construction, so int8 is nearly always the pick. */
    (void)K;

    /* Keep int8 off for degenerate/tiny shapes where overhead isn't amortized
     * and scalar is fine. int4 needs even larger M to amortize the
     * per-row requant-from-fp32 cost (it requants the SAME row int8 would,
     * then packs to nybbles -- so only enable it when M is large enough
     * that the extra packing/depacking cost is dwarfed by the 2x traffic
     * win vs int8). Precedence handled in quantized_matmul: int4 > int8 > fp32. */
    t.use_int8 = (M >= 256);
    t.use_int4 = (M >= 4096) && (K >= 1024);  /* big projections only */
    return t;
}

/* ---- int4 weight GEMV (B03, the next traffic halving after int8) ---- */

void wubu_gemv_quantize_i4(const float *w, int8_t *q4, float *scale, int M, int K) {
    /* pack 2 nybbles/byte; per-row absmax scale. Low nybble = even k. */
    for (int m = 0; m < M; m++) {
        const float *wr = w + (size_t)m * K;
        float amax = 1e-12f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(wr[k]);
            if (a > amax) amax = a;
        }
        scale[m] = amax / 7.0f;  /* map [-amax,amax] -> int4 [-7,7] */
        const int K2 = (K + 1) / 2;
        for (int j = 0; j < K2; j++) {
            int hi = 0, lo = 0;
            float v0 = wr[2*j] / scale[m];
            int v0i = (int)lrintf(v0 < -7 ? -7 : (v0 > 7 ? 7 : v0));
            lo = v0i + 8;  /* store as unsigned nybble 0..15 */
            if (2*j + 1 < K) {
                float v1 = wr[2*j+1] / scale[m];
                int v1i = (int)lrintf(v1 < -7 ? -7 : (v1 > 7 ? 7 : v1));
                hi = (v1i + 8) << 4;
            }
            q4[m*K2 + j] = (int8_t)(hi | lo);
        }
    }
}

void wubu_gemv_i4(const int8_t *q4, const float *scale,
                 const float *x, float *y, int M, int K) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (int m = 0; m < M; m++) {
        const int8_t *qr = q4 + (size_t)m * ((K + 1) / 2);
        float s = scale[m];
        float sum = 0.0f;
        int k = 0;
        /* two weights per byte; unrolled pair keeps the FMA hot */
        for (; k + 1 < K; k += 2) {
            int8_t b = qr[k >> 1];
            float w0 = (float)((b & 0xF) - 8) * s;
            float w1 = (float)(((b >> 4) & 0xF) - 8) * s;
            sum += w0 * x[k] + w1 * x[k+1];
        }
        if (k < K) {
            int8_t b = qr[k >> 1];
            sum += (float)((b & 0xF) - 8) * s * x[k];
        }
        y[m] = sum;
    }
}

const char *wubu_gemv_tile_name(const wubu_gemv_tile_t *t) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "unroll=%d int8=%d avx512=%d",
             t->k_unroll, t->use_int8, t->avx512);
    return buf;
}
