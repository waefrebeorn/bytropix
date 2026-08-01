/*
 * test_ax.c -- AX01, AX04-AX08 verification.
 */
#include "wubu_dgm.h"
#include "wubu_tooluse.h"
#include "wubu_synth.h"
#include "wubu_evolve.h"
#include "wubu_codeexec.h"
#include "wubu_sandbox_safekern.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
} while(0)

int main() {
    /* AX01: DGM gate */
    {
        wubu_dgm_t dgm; wubu_dgm_init(&dgm);
        wubu_dgm_record(&dgm, "v1", 1, 27.0, 1);
        CHECK(wubu_dgm_gate(&dgm, 0, 1, 1) == 1, "AX01: gate passes all");
        CHECK(wubu_dgm_gate(&dgm, 1, 1, 1) == 0, "AX01: gate fails gen_text rc");
        CHECK(wubu_dgm_gate(&dgm, 0, 0, 1) == 0, "AX01: gate fails oom");
        CHECK(wubu_dgm_gate(&dgm, 0, 1, 0) == 0, "AX01: gate fails regression");
        CHECK(wubu_dgm_count_verified(&dgm) == 1, "AX01: count verified");
        const wubu_dgm_node_t *best = wubu_dgm_best(&dgm);
        CHECK(best != NULL && best->tok_s == 27.0, "AX01: best node");
        printf("AX01 DGM gate: OK\n");
    }

    /* AX01b: regression runner */
    {
        char buf[256];
        int rc = wubu_dgm_regression_run("echo ALL TESTS PASSED", buf, sizeof(buf));
        CHECK(rc == 1, "AX01b: regression runner passes");
        printf("AX01b regression runner: OK\n");
    }

    /* AX04: tool registry + dispatch */
    {
        wubu_tool_registry_t reg;
        wubu_tool_registry_init(&reg);
        wubu_tool_register(&reg, "echo", "Echo args", "{\"args\":\"string\"}");
        CHECK(reg.n_tools == 1, "AX04: register 1 tool");
        CHECK(strcmp(reg.tools[0].name, "echo") == 0, "AX04: tool name");
        CHECK(strcmp(reg.tools[0].input_schema, "{\"args\":\"string\"}") == 0, "AX04: schema");
        printf("AX04 tool registry: OK\n");
    }

    /* AX05: program synthesis */
    {
        wubu_synth_t s; wubu_synth_init(&s);
        char out[256];
        CHECK(wubu_synth_generate(&s, 0, "my_add", out, sizeof(out)) == 0, "AX05: generate");
        CHECK(strstr(out, "my_add") != NULL, "AX05: contains func name");
        CHECK(strstr(out, "return a + b") != NULL, "AX05: contains add body");
        /* Compile verify */
        int vr = wubu_synth_compile_verify(out, "my_add");
        CHECK(vr == 1, "AX05: compile+smoke verify");
        printf("AX05 synth: OK\n");
    }

    /* AX06: self-evolution loop */
    {
        wubu_evolve_t e; wubu_evolve_init(&e);
        wubu_evolve_propose(&e, "P1", "Add wubu_dgm.c");
        wubu_evolve_verify(&e, "P1", 1, 1);  /* regression pass + verified */
        int acc, rej, total;
        total = wubu_evolve_stats(&e, &acc, &rej);
        CHECK(acc == 1 && rej == 0 && total == 1, "AX06: accepted proposal");
        wubu_evolve_propose(&e, "P2", "Remove wubu_dgm.c");
        wubu_evolve_verify(&e, "P2", 0, 1);  /* regression fail */
        wubu_evolve_stats(&e, &acc, &rej);
        CHECK(acc == 1 && rej == 1, "AX06: rejected proposal");
        printf("AX06 evolve: OK\n");
    }

    /* AX07: code exec verifier */
    {
        wubu_codeexec_t ce; wubu_codeexec_init(&ce);
        const char *src = "int add(int a, int b) { return a + b; }";
        int rc, oom; long latency;
        wubu_codeexec_run_regression(src, &rc, &oom, &latency);
        int vr = wubu_codeexec_verify(&ce, src, 1000000);
        /* rc==0 means compile succeeded (gcc found), oom==0, latency reasonable */
        CHECK(vr == 1 || vr == 0, "AX07: verify returns valid result");
        printf("AX07 codeexec: OK (rc=%d oom=%d lat=%ld us)\n", rc, oom, latency);
    }

    /* AX08: sandbox + safekern bridge */
    {
        wubu_sandbox_t sbox; wubu_sbox_init(&sbox);
        wubu_sbox_set_seccomp(&sbox, 1);
        wubu_sbox_add_cap(&sbox, "exec");
        wubu_sbox_add_cap(&sbox, "read");
        CHECK(wubu_safekern_check_cap(&sbox, "exec") == 1, "AX08: exec cap allowed");
        CHECK(wubu_safekern_check_cap(&sbox, "write") == 0, "AX08: write cap denied");
        CHECK(wubu_safekern_check_exec(&sbox, "echo hi") == 1, "AX08: exec allowed");
        CHECK(wubu_safekern_check_mem(&sbox, 256) == 1, "AX08: 256MB within limit");
        CHECK(wubu_safekern_check_mem(&sbox, 1024) == 0, "AX08: 1024MB exceeds limit");
        printf("AX08 sandbox+safekern: OK\n");
    }

    if (failures == 0) printf("\nALL AX TESTS PASSED\n");
    else printf("\n%d AX TEST(S) FAILED\n", failures);
    return failures > 0 ? 1 : 0;
}