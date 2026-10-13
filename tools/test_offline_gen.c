/* test_offline_gen.c -- UX #241 tests: companion offline fallback gen_text. */
#include "wubu_offline_gen.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)
#define CHECK_STR(s, expected) do { if (strcmp((s),(expected))!=0) { \
    printf("FAIL: %s == \"%s\" expected \"%s\" (line %d)\n", #s, (s), (expected), __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_offline_gen_t og;
    char out[WUBU_OG_OUT_MAX];
    wubu_og_src_t src = (wubu_og_src_t)99;

    CHECK(wubu_og_init(&og) == 0);
    CHECK(og.mode == WUBU_OG_MODE_AUTO);
    CHECK(og.have_key == 0);

    /* no key, no remote -> local */
    CHECK(wubu_og_decide(&og, "hi") == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_generate(&og, "hello world", out, sizeof(out), &src) > 0);
    CHECK(src == WUBU_OG_SRC_LOCAL);
    CHECK_STR(out, "[offline] hello world");
    CHECK(wubu_og_n_local(&og) == 1);
    CHECK(wubu_og_n_remote(&og) == 0);
    CHECK(wubu_og_n_fallbacks(&og) == 0);

    /* with key + healthy remote -> remote */
    wubu_og_set_key(&og, 1);
    wubu_og_remote_health(&og, 1);
    CHECK(wubu_og_decide(&og, "hi") == WUBU_OG_SRC_REMOTE);
    CHECK(wubu_og_generate(&og, "remote prompt", out, sizeof(out), &src) > 0);
    CHECK(src == WUBU_OG_SRC_REMOTE);
    CHECK_STR(out, "[remote] remote prompt");
    CHECK(wubu_og_n_remote(&og) == 1);

    /* key present but remote unhealthy -> auto falls back to local */
    wubu_og_remote_health(&og, 0);
    CHECK(wubu_og_decide(&og, "hi") == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_generate(&og, "fallback now", out, sizeof(out), &src) > 0);
    CHECK(src == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_n_fallbacks(&og) == 1);
    CHECK(wubu_og_n_remote(&og) >= 1);   /* counter monotonic */
    CHECK_STR(out, "[offline] fallback now");

    /* explicit offline override -> local even with key+healthy remote */
    wubu_og_remote_health(&og, 1);
    wubu_og_set_offline(&og, 1);
    CHECK(wubu_og_decide(&og, "hi") == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_generate(&og, "offline mode", out, sizeof(out), &src) > 0);
    CHECK(src == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_n_local(&og) >= 2);
    wubu_og_set_offline(&og, 0);

    /* ONLINE-only mode with bad remote -> local fallback */
    og.mode = WUBU_OG_MODE_ONLINE;
    wubu_og_remote_health(&og, 0);     /* mark remote unhealthy for this case */
    CHECK(wubu_og_decide(&og, "hi") == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_generate(&og, "online-fail", out, sizeof(out), &src) > 0);
    CHECK(src == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_n_fallbacks(&og) >= 2);
    /* ONLINE + healthy remote -> remote */
    wubu_og_remote_health(&og, 1);
    CHECK(wubu_og_decide(&og, "hi") == WUBU_OG_SRC_REMOTE);

    /* OFFLINE mode: always local regardless of key/remote */
    og.mode = WUBU_OG_MODE_OFFLINE;
    CHECK(wubu_og_decide(&og, "hi") == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_generate(&og, "forced local", out, sizeof(out), &src) > 0);
    CHECK(src == WUBU_OG_SRC_LOCAL);

    /* local truncates long prompts to fill buffer */
    {
        wubu_offline_gen_t g; wubu_og_init(&g);
        char big[32];
        int n = wubu_og_local(&g, "this is a very long prompt that exceeds buffers", big, sizeof(big));
        CHECK(n > 0);
        CHECK((size_t)n == sizeof(big) - 1);   /* filled to capacity, NUL-terminated */
        CHECK(big[sizeof(big)-1] == '\0');
    }

    /* output cap check, bad args */
    CHECK(wubu_og_generate(&og, NULL, out, sizeof(out), &src) == -1);
    CHECK(wubu_og_generate(&og, "x", NULL, 0, &src) == -1);
    CHECK(wubu_og_decide(NULL, "x") == WUBU_OG_SRC_LOCAL);
    CHECK(wubu_og_init(NULL) == -1);

    /* long prompt is rejected (prompt-truncation guard) */
    {
        char longp[WUBU_OG_PROMPT_MAX + 10];
        memset(longp, 'a', sizeof(longp)-1); longp[sizeof(longp)-1] = '\0';
        CHECK(wubu_og_generate(&og, longp, out, sizeof(out), &src) == -1);
    }

    if (fails == 0) printf("ALL OFFLINE_GEN TESTS PASSED\n");
    else printf("%d OFFLINE_GEN TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
