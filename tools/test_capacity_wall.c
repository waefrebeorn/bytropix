/*
 * test_capacity_wall.c -- N12/N11 decode-I/O capacity + roofline predictor.
 * Verifies: KV bytes formula, fits-ram (512K gate), B* crossover, TPOT,
 * and edge cases (cap==0, seq==0, bits==0, beta==0).
 */
#include "wubu_capacity_wall.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_capacity_wall (N12 capacity-wall / N11 TPOT) ===\n");

    /* Llama-3-8B-ish: L=32, n_kv=8, d_h=128, FP16 (16 bits). */
    int L = 32, n_kv = 8, d_h = 128, b_kv = 16;
    double params = 8e9;

    /* KV bytes/tok/layer = 2*8*128*(16/8)=4096; total @4k,batch1 = 4096*32*4096 */
    double kv = wubu_kv_cache_bytes(L, n_kv, d_h, b_kv, 1, 4096);
    CHECK(fabs(kv - 536870912.0) < 1.0, "KV bytes @4k matches 512MB/seq");

    /* fits in 12GB at 4k? 512MB -> yes */
    CHECK(wubu_kv_fits_ram(L, n_kv, d_h, b_kv, 1, 4096, 12e9) == 1, "fits 12GB @4k");
    /* at 1M ctx, KV ~ 4096*32*1e6 = 131GB -> no */
    CHECK(wubu_kv_fits_ram(L, n_kv, d_h, b_kv, 1, 1000000, 12e9) == 0, "no fit 12GB @1M");

    /* B* crossover: W=8e9*2=16GB; K_seq=4096*32*4096 ~ 536MB; B*=16e9/536e6 ~ 29.8 */
    double bs = wubu_b_star(params, 16, L, n_kv, d_h, b_kv, 4096);
    CHECK(fabs(bs - 29.8) < 1.0, "B* ~ 29.8 @4k");
    /* at 1M ctx B* drops below 1 -> always KV-bound */
    double bs1 = wubu_b_star(params, 16, L, n_kv, d_h, b_kv, 1000000);
    CHECK(bs1 < 1.0, "B* < 1 at 1M ctx (always KV-bound)");

    /* TPOT: beta_eff ~ 50 GB/s (WSL DRAM). W=16GB + K=0.5GB -> 16.5GB/50e9
     * = 0.33s/tok -> ~3 tok/s. Just check monotonic sense. */
    double beta = 50e9;
    double t_4k = wubu_tpot(params, 16, L, n_kv, d_h, b_kv, 1, 4096, beta);
    double t_1m = wubu_tpot(params, 16, L, n_kv, d_h, b_kv, 1, 1000000, beta);
    CHECK(t_1m > t_4k, "TPOT grows with context");
    double tps = wubu_tok_per_sec(params, 16, L, n_kv, d_h, b_kv, 1, 4096, beta);
    CHECK(tps > 0.0 && fabs(tps - 1.0 / t_4k) < 1e-9, "tok/s = 1/TPOT");

    /* edge cases */
    CHECK(wubu_kv_cache_bytes(L, n_kv, d_h, b_kv, 0, 100) == 0.0, "batch 0 -> 0");
    CHECK(wubu_kv_cache_bytes(L, n_kv, d_h, b_kv, 1, 0) == 0.0, "seq 0 -> 0");
    CHECK(wubu_weight_bytes(0, 16) == 0.0, "params 0 -> 0");
    CHECK(wubu_tpot(params, 16, L, n_kv, d_h, b_kv, 1, 4096, 0) == 0.0, "beta 0 -> 0");
    CHECK(wubu_kv_fits_ram(L, n_kv, d_h, 0, 1, 100, 12e9) == 1, "bits 0 -> fits (0 KV)");
    CHECK(wubu_b_star(params, 16, L, n_kv, d_h, 0, 4096) < 0.0, "KV bits 0 -> never KV-bound (-1)");
    CHECK(wubu_oom_risk(params, 16, L, n_kv, d_h, b_kv, 1, 4096, 12e9, 0.9) == 0, "no OOM risk @4k/12GB");
    CHECK(wubu_oom_risk(params, 16, L, n_kv, d_h, b_kv, 1, 1000000, 12e9, 0.9) == 1, "OOM risk trips @1M/12GB");
    CHECK(wubu_regime(50.0, 1, 1.5) == 0, "b*>>batch -> WEIGHT_BOUND");
    CHECK(wubu_regime(0.5, 100, 1.5) == 2, "b*<<batch -> KV_BOUND");
    CHECK(wubu_regime(30.0, 30, 1.5) == 1, "b*~batch -> BALANCED");

    if (failures == 0) { printf("ALL CAPACITY-WALL TESTS PASSED\n"); return 0; }
    printf("%d CAPACITY-WALL TEST(S) FAILED\n", failures);
    return 1;
}
