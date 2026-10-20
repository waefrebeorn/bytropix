/* test_harness2.c -- assertions for the C11 Harness^2 loop. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "wubu_harness2.h"

static int fails = 0;

#define CHECK(cond) do { \
    if (cond) printf("  PASS  %s\n", #cond); \
    else { fails++; printf("  FAIL  %s (line %d)\n", #cond, __LINE__); } \
} while (0)

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1);
    if (!s) { fclose(f); return NULL; }
    size_t rd = fread(s, 1, (size_t)n, f);
    s[rd] = '\0';
    fclose(f);
    return s;
}

int main(void) {
    char dir[] = "/tmp/wubu_h2_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 2; }

    /* ---- rules ---- */
    wubu_h2_harness_t *h = wubu_h2_harness_new();
    CHECK(h != NULL);
    CHECK(wubu_h2_harness_rule_count(h) == 0);
    CHECK(wubu_h2_harness_add_rule(h, "run the check first") == 0);
    CHECK(wubu_h2_harness_add_rule(h, "quote the check output") == 0);
    CHECK(wubu_h2_harness_rule_count(h) == 2);
    CHECK(wubu_h2_harness_add_rule(h, "run the check first") == -1); /* duplicate */
    CHECK(wubu_h2_harness_add_rule(h, "") == -1);
    CHECK(wubu_h2_harness_has_rule(h, "quote the check output") == 1);
    CHECK(wubu_h2_harness_has_rule(h, "absent") == 0);

    char longrule[WUBU_H2_RULE_LEN + 8];
    memset(longrule, 'x', sizeof longrule - 1);
    longrule[sizeof longrule - 1] = '\0';
    CHECK(wubu_h2_harness_add_rule(h, longrule) == -1);

    for (int i = 0; i < WUBU_H2_MAX_RULES; i++) {
        char r[64];
        snprintf(r, sizeof r, "filler rule %d", i);
        (void)wubu_h2_harness_add_rule(h, r);
    }
    CHECK(wubu_h2_harness_rule_count(h) == WUBU_H2_MAX_RULES);
    CHECK(wubu_h2_harness_add_rule(h, "one too many") == -1);
    wubu_h2_harness_free(h);

    /* ---- render + stage ---- */
    h = wubu_h2_harness_new();
    wubu_h2_harness_add_rule(h, "alpha");
    wubu_h2_harness_add_rule(h, "beta");
    char a[4096], b[4096];
    int na = wubu_h2_harness_render(h, a, sizeof a);
    int nb = wubu_h2_harness_render(h, b, sizeof b);
    CHECK(na > 0 && na == nb);
    CHECK(strcmp(a, b) == 0);                       /* deterministic */
    CHECK(strstr(a, "- alpha") != NULL);
    CHECK(strstr(a, "- beta") != NULL);
    CHECK(strstr(a, "check command exits 0") != NULL);
    CHECK(wubu_h2_harness_render(h, a, 16) == -1);  /* too small */

    CHECK(wubu_h2_harness_stage(h, dir) == 0);
    char path[512];
    snprintf(path, sizeof path, "%s/AGENTS.md", dir);
    char *staged = slurp(path);
    CHECK(staged != NULL);
    if (staged) {
        CHECK(strcmp(staged, b) == 0);              /* bytes == render */
        free(staged);
    }
    wubu_h2_harness_free(h);

    /* ---- probes: the check decides, not the executor ---- */
    wubu_h2_harness_t *empty = wubu_h2_harness_new();
    wubu_h2_probe_t p0, p1, p2;
    CHECK(wubu_h2_probe_run(&p0, 0, empty, "exit 0", 5) == 0);
    CHECK(p0.check_rc == 0);
    CHECK(wubu_h2_probe_run(&p1, 1, empty, "exit 7", 5) == 7);
    CHECK(p1.check_rc == 7);
    CHECK(wubu_h2_probe_run(&p2, 2, empty, "sleep 3", 1) == 124);
    CHECK(p2.check_rc == 124);                       /* timeout != success */
    CHECK(wubu_h2_probe_run(&p2, 3, empty, "true", 0) == -1); /* bad timeout */
    CHECK(wubu_h2_probe_run(&p2, 3, empty, "", 5) == -1);     /* empty command */

    /* ---- composition ---- */
    wubu_h2_probe_t set[3];
    memset(set, 0, sizeof set);
    set[0].probe_index = 0; set[0].check_rc = 1;
    set[1].probe_index = 1; set[1].check_rc = 0;
    set[2].probe_index = 2; set[2].check_rc = 0;
    CHECK(wubu_h2_compose(set, 3) == 1);             /* first passing wins */
    set[0].check_rc = 0;
    CHECK(wubu_h2_compose(set, 3) == 0);             /* baseline can win */
    set[0].check_rc = 1; set[1].check_rc = 1; set[2].check_rc = 1;
    CHECK(wubu_h2_compose(set, 3) == -1);            /* nothing passed */
    CHECK(wubu_h2_compose(NULL, 0) == -1);

    set[1].rules_count = 2;
    char adopted[4][WUBU_H2_RULE_LEN];
    CHECK(wubu_h2_adopt(set, 3, 1, adopted, 4) == 2);
    CHECK(wubu_h2_adopt(set, 3, 9, adopted, 4) == -1);
    CHECK(wubu_h2_adopt(set, 3, -1, adopted, 4) == 0);

    /* ---- ledger ---- */
    char ledger[512];
    snprintf(ledger, sizeof ledger, "%s/ledger.jsonl", dir);
    set[0].check_rc = 1; set[1].check_rc = 0; set[2].check_rc = 3;
    CHECK(wubu_h2_ledger_append(ledger, "task-a", set, 3, 1) == 0);
    CHECK(wubu_h2_ledger_append(ledger, "task-b", set, 3, -1) == 0);
    CHECK(wubu_h2_ledger_count(ledger) == 2);
    CHECK(wubu_h2_ledger_append(ledger, "bad\"id", set, 3, 1) == -1);
    CHECK(wubu_h2_ledger_count(ledger) == 2);        /* refused line absent */
    char *body = slurp(ledger);
    CHECK(body != NULL);
    if (body) {
        CHECK(strstr(body, "\"task\":\"task-a\"") != NULL);
        CHECK(strstr(body, "\"adopted\":1") != NULL);
        CHECK(strstr(body, "\"check_rc\":3") != NULL);
        free(body);
    }

    /* ---- executor argv ---- */
    char *argv[WUBU_H2_ARGV_MAX];
    int k = wubu_h2_executor_argv(argv, WUBU_H2_ARGV_MAX, "/usr/bin/hermes",
                                  "do it", "/tmp/ws", "model-x");
    CHECK(k == 8);
    CHECK(strcmp(argv[0], "/usr/bin/hermes") == 0);
    CHECK(strcmp(argv[1], "-z") == 0);
    CHECK(strcmp(argv[2], "do it") == 0);
    CHECK(strcmp(argv[3], "--in") == 0);
    CHECK(strcmp(argv[4], "/tmp/ws") == 0);
    CHECK(strcmp(argv[5], "-m") == 0);
    CHECK(strcmp(argv[6], "model-x") == 0);
    CHECK(strcmp(argv[7], "--pass-session-id") == 0);
    CHECK(argv[8 - 1] != NULL && argv[8] == NULL);
    k = wubu_h2_executor_argv(argv, WUBU_H2_ARGV_MAX, "/h", "p", "/w", NULL);
    CHECK(k == 6);
    CHECK(wubu_h2_executor_argv(argv, 4, "/h", "p", "/w", NULL) == -1);
    CHECK(wubu_h2_executor_argv(NULL, 8, "/h", "p", "/w", NULL) == -1);

    wubu_h2_harness_free(empty);

    /* cleanup: the staged tree is under mkdtemp, remove the files we made */
    unlink(path);
    unlink(ledger);
    rmdir(dir);

    printf("\n=== test_harness2: %s (%d failure(s)) ===\n",
           fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}