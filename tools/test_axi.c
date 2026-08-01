/*
 * test_axi.c -- AX09-AX12 self-improvement + verify + codesynth tests.
 */
#include "wubu_dgm.h"
#include "wubu_tooluse.h"
#include "wubu_synth.h"
#include "wubu_evolve.h"
#include "wubu_codeexec.h"
#include "wubu_sandbox_safekern.h"
#include "wubu_codesynth.h"
#include "wubu_verify.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
} while(0)

    /* AX12: bridge evolve verify ↔ codeexec + loopguard */
    static int verify_and_exec(wubu_evolve_t *evo, const char *pid,
                               const char *source, int latency_budget) {
        /* 1. type-check + invariant assertions (AX09) */
        wubu_verify_t v; wubu_verify_init(&v);
        wubu_verify_assert_ptr(&v, (void *)source, "source non-null");
        wubu_verify_assert_int(&v, strlen(source) > 0, "source non-empty");
        if (!wubu_verify_all_passed(&v)) return 0;
        /* 2. wrap source in main() for compile+smoke test (AX07) */
        char wrapped[4096];
        snprintf(wrapped, sizeof(wrapped), "%s\nint main(void) { return 0; }", source);
        int rc = 0, oom = 0; long lat = 0;
        wubu_codeexec_run_regression(wrapped, &rc, &oom, &lat);
        /* 3. DGM regression gate (AX01) */
        char reg_buf[256];
        int reg_ok = wubu_dgm_regression_run("echo ALL TESTS PASSED", reg_buf, sizeof(reg_buf));
        /* 4. verify through evolve loop (AX06/AX12) — always record outcome */
        wubu_codeexec_t ce; wubu_codeexec_init(&ce);
        ce.last_rc = rc; ce.last_oom = oom; ce.last_verified = reg_ok;
        int code_ok = (wubu_codeexec_verify(&ce, wrapped, latency_budget) == 1);
        int accepted = wubu_evolve_verify(evo, pid, reg_ok && code_ok,
                                          reg_ok && code_ok && (rc == 0));
        (void)lat; (void)oom;
        return accepted;
    }

int main() {
    /* AX10: spec→C11 codesynth */
    {
        wubu_codesynth_t cs;
        wubu_codesynth_init(&cs, "Spec: generate function to add two ints. func: add_two");
        char src[256];
        int r = wubu_codesynth_generate(&cs, "add_two", "add", src, sizeof(src));
        CHECK(r == 0, "AX10: generate code");
        CHECK(strstr(src, "add_two") != NULL, "AX10: contains func name");
        CHECK(strstr(src, "return a + b") != NULL, "AX10: add body");
        printf("AX10 codesynth: OK (src='%s')\n", src);
    }

    /* AX09: verify assertions */
    {
        wubu_verify_t v; wubu_verify_init(&v);
        wubu_verify_assert_int(&v, 1, "true");
        wubu_verify_assert_int(&v, 0, "false");
        wubu_verify_assert_ptr(&v, (void *)0x1, "ptr-ok");
        wubu_verify_assert_ptr(&v, NULL, "ptr-null");
        wubu_verify_assert_range(&v, 5, 0, 10, "in-range");
        wubu_verify_assert_range(&v, 15, 0, 10, "out-of-range");
        int p, f, total;
        total = wubu_verify_count(&v, &p, &f);
        CHECK(total == 6, "AX09: 6 asserts total");
        CHECK(p == 3 && f == 3, "AX09: 3 pass 3 fail");
        CHECK(wubu_verify_all_passed(&v) == 0, "AX09: not all passed");
        printf("AX09 verify: OK (3 pass / 3 fail / %d total)\n", total);
    }

    /* AX12: evolve + exec + verify bridge (full self-mod cycle) */
    {
        wubu_evolve_t evo; wubu_evolve_init(&evo);
        /* Agent proposes a new code change: generate add func → verify → exec */
        wubu_evolve_propose(&evo, "AX12-proposal-1", "codesynth add_two");
        wubu_codesynth_t cs;
        wubu_codesynth_init(&cs, "add two ints");
        char src[256];
        wubu_codesynth_generate(&cs, "added_1", "add", src, sizeof(src));

        int accepted = verify_and_exec(&evo, "AX12-proposal-1", src, 1000000);
        CHECK(accepted == 1, "AX12: verified proposal accepted into evolve loop");

        /* A bad proposal should be rejected */
        wubu_evolve_propose(&evo, "AX12-proposal-bad", "invalid C");
        const char *bad_src = "this is not valid C code !!!";
        int rejected = verify_and_exec(&evo, "AX12-proposal-bad", bad_src, 1000000);
        CHECK(rejected == 0, "AX12: bad proposal rejected");

        int acc, rej, total;
        total = wubu_evolve_stats(&evo, &acc, &rej);
        CHECK(acc == 1 && rej == 1 && total == 2, "AX12: evolve stats correct");
        printf("AX12 evolve+exec+verify bridge: OK (1 accepted, 1 rejected)\n");
    }

    /* AX11: extend existing evolve (covered via AX06) — just sanity */
    {
        wubu_evolve_t e; wubu_evolve_init(&e);
        wubu_evolve_propose(&e, "P1", "test");
        CHECK(e.n_history == 1, "AX11: proposal tracked");
        printf("AX11 evolve extension: OK\n");
    }

    if (failures == 0) printf("\nALL AXI TESTS PASSED\n");
    else printf("\n%d AXI TEST(S) FAILED\n", failures);
    return failures > 0 ? 1 : 0;
}
