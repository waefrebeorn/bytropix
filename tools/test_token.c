/* test_token.c -- Theme IT batch 1: the tokenization frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_token.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_token (IT batch 1) ===\n");

    /* IT01: bit-level BPE cost */
    CHECK(wubu_tok_bit_bpe_cost(10, 0) == 80, "plain bytes");
    CHECK(wubu_tok_bit_bpe_cost(10, 2) > 80, "symbol bits add");

    /* IT02: tokenizer-free UTF-8 embedding */
    {
        float out[8];
        const unsigned char s[] = "héllo";
        int n = wubu_tok_utf8_embed(s, 6, out, 8);
        CHECK(n == 6, "bytes embedded");
        float e = 0;
        for (int i = 0; i < 8; i++) e += fabsf(out[i]);
        CHECK(e > 0, "embedding non-zero");
    }

    /* IT04: entropy merge score -- uniform is higher entropy */
    {
        uint32_t c1[4] = { 25, 25, 25, 25 }, c2[4] = { 97, 1, 1, 1 };
        NEAR(wubu_tok_entropy_merge(c1, 4), 2.0f, 1e-4f);
        CHECK(wubu_tok_entropy_merge(c2, 4) < 0.5f, "skewed is low-entropy");
    }

    /* IT05: lexical density -> window */
    {
        int w_lo = wubu_tok_density_window(100, 0.2f, 512);
        int w_hi = wubu_tok_density_window(100, 0.9f, 512);
        CHECK(w_hi < w_lo, "denser context -> shorter effective window");
        CHECK(wubu_tok_density_window(100, 0.2f, 512) <= 512, "capped");
    }

    /* IT06: merge cache */
    {
        wubu_tok_cache_t c = { 0, 0, 0 };
        CHECK(wubu_tok_cache_get(&c, 7, 3) == 3, "miss -> fallback");
        wubu_tok_cache_put(&c, 7, 11);
        CHECK(wubu_tok_cache_get(&c, 7, 3) == 11, "hit -> cached");
    }

    /* IT07: vocab pruning */
    {
        int used[5] = { 1, 0, 1, 0, 1 }, remap[5], kept = 0;
        wubu_tok_prune(used, 5, remap, &kept);
        CHECK(kept == 3, "three kept");
        CHECK(remap[0] == 0 && remap[2] == 1 && remap[4] == 2, "remapped dense");
        CHECK(remap[1] == -1, "dropped -> -1");
    }

    /* IT08: roundtrip fidelity */
    {
        const unsigned char s[] = "abc", back[] = "abc", bad[] = "abd";
        CHECK(wubu_tok_roundtrip(s, 3, back, 3) == 1, "identical roundtrip");
        CHECK(wubu_tok_roundtrip(s, 3, bad, 3) == 0, "drift detected");
    }

    /* IT12: token efficiency */
    NEAR(wubu_tok_efficiency(100, 400), 4.0f, 1e-5f);

    /* IT16: OOV handling */
    CHECK(wubu_tok_oov(5, 100, 3) == 5, "in-vocab kept");
    CHECK(wubu_tok_oov(500, 100, 3) == 3, "OOV falls back");
    CHECK(wubu_tok_oov(500, 100, -1) == -1, "no fallback -> -1");

    /* IT10: entropy coding size */
    {
        uint32_t counts[4] = { 50, 50, 0, 0 };
        /* 100 tokens at 1 bit each = 100 bits = 12.5 bytes */
        CHECK(wubu_tok_entropy_size(counts, 4, 100) == 12, "1 bit/token");
    }

    if (failures == 0) printf("ALL TOKEN TESTS PASSED\n");
    else printf("%d TOKEN FAILURES\n", failures);
    return failures ? 1 : 0;
}
