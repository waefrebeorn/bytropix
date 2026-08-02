/* test_token2.c -- Theme IT complete: the tokenization frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_token2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_token2 (IT complete) ===\n");
    NEAR(wubu_tok2_bench(100, 400), 4.0f, 1e-5f);
    {
        int old[3] = { 1, 0, 2 }, map[3] = { 10, 20, 30 }, out[3];
        CHECK(wubu_tok2_remap(old, 3, map, out) == 3, "remapped");
        CHECK(out[0] == 20 && out[1] == 10 && out[2] == 30, "ids swapped");
    }
    {
        uint32_t c[3] = { 50, 30, 20 }, r[3] = { 40, 40, 20 };
        CHECK(wubu_tok2_shift(c, 3, r, 0.1f) == 1, "shift detected");
    }
    NEAR(wubu_tok2_pair_score(10, 5, 5), 0.4f, 1e-5f);
    {
        wubu_tok2_cache_t c = { 0, 0, 0 };
        CHECK(wubu_tok2_cache_get(&c, 5, 3) == 3, "miss");
        wubu_tok2_cache_put(&c, 5, 9);
        CHECK(wubu_tok2_cache_get(&c, 5, 3) == 9, "hit");
    }
    CHECK(wubu_tok2_norm_guard((const unsigned char *)"héllo", 6, 0) == 1,
          "valid utf8");
    CHECK(wubu_tok2_norm_guard((const unsigned char[]){ 'a', 0x80, 'b', 0 }, 3, 0) == 0,
          "stray continuation rejected");
    CHECK(wubu_tok2_len_reg(90, 100) == 1, "within cap");
    CHECK(wubu_tok2_len_reg(120, 100) == 0, "over cap");
    {
        int ok = 1;
        CHECK(wubu_tok2_byte_fallback((const unsigned char *)"abc", 3, &ok) == 3,
              "clean bytes");
        CHECK(wubu_tok2_byte_fallback((const unsigned char *)"a\x80", 2, &ok) == 1 &&
              ok == 0, "corrupt flagged");
    }
    {
        uint32_t freq[8] = { 0 };
        wubu_tok2_pair_freq(freq, 8, 1, 2);
        CHECK(freq[(1 * 8 + 2) % 8] == 1, "pair counted");
    }
    NEAR(wubu_tok2_density(100, 800), 8.0f, 1e-5f);
    {
        uint32_t a[3] = { 1, 2, 3 }, b[3] = { 1, 2, 3 };
        CHECK(wubu_tok2_deterministic(a, b, 3) == 1, "deterministic");
    }
    CHECK(wubu_tok2_budget_plan(100, 0.5f, 1000) == 150, "planned");
    CHECK(wubu_tok2_budget_plan(1000, 0.5f, 1000) == 1000, "capped");
    CHECK(wubu_tok2_entity_align(2, 5, 10) == 1, "entity aligned");
    CHECK(wubu_tok2_entity_align(5, 2, 10) == 0, "inverted rejected");
    {
        wubu_tok2_stream_t s = { 0, 0 };
        uint32_t t = 0;
        for (int i = 0; i < 4; i++) wubu_tok2_stream(&s, (unsigned char)(i + 1), &t);
        CHECK(t == 0x01020304u, "stream assembled 4 bytes");
    }
    {
        uint32_t ids[5] = { 1, 2, 3, 4, 5 }, out[5];
        int k = wubu_tok2_dropout(ids, 5, 0.0f, out);
        CHECK(k == 5, "no dropout at p=0");
    }
    CHECK(wubu_tok2_byte_rope(0.5f, 100, 10000.0f) > 0, "rope angle");
    {
        uint32_t ids[5] = { 1, 2, 3, 4, 5 }, out[3];
        CHECK(wubu_tok2_next_n(ids, 5, 3, out) == 3, "next-N");
        CHECK(out[0] == 3 && out[2] == 5, "the last N");
    }
    {
        uint32_t ids[5] = { 9, 9, 4, 9, 9 };
        int depth = -1;
        CHECK(wubu_tok2_trie(ids, 5, 4, &depth) == 1 && depth == 2, "trie hit");
    }
    {
        uint32_t vocab[2] = { 0x01020304, 0xAABBCCDD };
        uint8_t buf[8];
        CHECK(wubu_tok2_serialize(vocab, 2, buf, 8) == 8, "serialized");
        CHECK(buf[0] == 4 && buf[7] == 0xAA, "little-endian bytes laid out");
    }
    NEAR(wubu_tok2_pair_health(50, 100), 0.5f, 1e-6f);
    CHECK(wubu_tok2_skip_redundant(100, 0.3f) == 70, "redundant skipped");
    {
        uint32_t out[4];
        CHECK(wubu_tok2_fallback((const unsigned char *)"ab", 2, out) == 2,
              "byte fallback");
        CHECK(out[0] == 'a', "byte ids");
    }
    NEAR(wubu_tok2_coverage(95, 100), 0.95f, 1e-6f);

    if (failures == 0) printf("ALL TOKEN2 TESTS PASSED\n");
    else printf("%d TOKEN2 FAILURES\n", failures);
    return failures ? 1 : 0;
}
