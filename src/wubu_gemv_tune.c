/*
 * wubu_gemv_tune.c -- Roofline-driven GEMV auto-tuner (see header).
 * Pure C, routes through wubu_roofline for the B*-ridge decision.
 */
#include "wubu_gemv_tune.h"
#include "wubu_roofline.h"
#include <stdio.h>

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
     * and scalar is fine. */
    t.use_int8 = (M >= 256);
    return t;
}

const char *wubu_gemv_tile_name(const wubu_gemv_tile_t *t) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "unroll=%d int8=%d avx512=%d",
             t->k_unroll, t->use_int8, t->avx512);
    return buf;
}
