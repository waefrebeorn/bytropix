/*
 * wubu_kv_runtime.c -- runtime KV-cache scheme dispatch.
 *
 * The KV cache read/write/alloc functions in wubu_model.h dispatch on the
 * global g_kv_scheme instead of a compile-time #if, so the engine can pick the
 * precision per-model at load time via the Roofline auto-selector
 * (wubu_kv_select). This is the runtime half of the Kevin-Bacon convergence:
 * the scheme is no longer frozen at compile time.
 */
#include "wubu_kv_runtime.h"
#include "wubu_roofline.h"

/* Default scheme from the compile-time macros (keeps existing builds working
 * when nothing calls wubu_kv_set_scheme). */
#if defined(KV_CACHE_Q4_0)
  #define KV_SCHEME_DEFAULT WUBU_KV_Q4_0
#elif defined(KV_CACHE_OUR_Q8)
  #define KV_SCHEME_DEFAULT WUBU_KV_Q8
#elif defined(KV_CACHE_KIVI)
  #define KV_SCHEME_DEFAULT WUBU_KV_KIVI
#elif defined(KV_CACHE_F16)
  #define KV_SCHEME_DEFAULT WUBU_KV_F16
#else
  #define KV_SCHEME_DEFAULT WUBU_KV_F32
#endif

int g_kv_scheme = KV_SCHEME_DEFAULT;
int g_kv_head_dim = 0;
int g_use_q8_cache = 0; /* set to 1 when fast-attn Q8 decode should be used */

void wubu_kv_set_scheme(int scheme) { g_kv_scheme = scheme; }
int  wubu_kv_get_scheme(void)        { return g_kv_scheme; }

/* Pick + apply the KV scheme from real model params + detected bandwidth.
 * P_params in absolute param count (e.g. 27e9); s = target context.
 * Returns the chosen scheme. */
int wubu_kv_autoselect(double P_params, int n_layers, int n_kv_heads,
                        int head_dim, double beta_eff_tb_s, int s) {
    wubu_roofline_cfg_t c = wubu_roofline_default();
    c.n_layers   = n_layers;
    c.n_kv_heads = n_kv_heads;
    c.head_dim   = head_dim;
    c.beta_eff_tb_s = beta_eff_tb_s > 0 ? beta_eff_tb_s : 0.05;
    wubu_kv_choice_t ch = wubu_kv_select(&c, P_params, 1, s);
    /* A04: For long-context decode (s >= 4096), force Q8 KV cache regardless
     * of Roofline crossover — Q8 is near-lossless (0.075 MSE) and 2x faster
     * than F16 on bandwidth-bound attention scan at 512K. */
    if (s >= 4096 && ch.kv != WUBU_KV_Q8 && ch.kv != WUBU_KV_KIVI
        && ch.kv != WUBU_KV_Q4_0 && ch.kv != WUBU_KV_4KV) {
        /* Only override if currently F32/F16 (not already compressed) */
        ch.kv = WUBU_KV_Q8;
        ch.kv_bits = 8;
        ch.why = "512K decode: Q8 KV cache (near-lossless, 2x faster)";
    }
    /* WUBU_FORCE_Q8_KV env overrides everything */
    const char *fq8 = getenv("WUBU_FORCE_Q8_KV");
    if (fq8 && atoi(fq8)) {
        ch.kv = WUBU_KV_Q8;
        ch.kv_bits = 8;
        ch.why = "WUBU_FORCE_Q8_KV env override";
    }
    g_kv_scheme = (int)ch.kv;
    g_kv_head_dim = head_dim;
    return g_kv_scheme;
}
