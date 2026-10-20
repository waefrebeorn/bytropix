/* harness2_c11 -- run one Harness^2 probe cycle from the shell.
 *
 * Baseline probe (no extra rules) plus one probe per candidate rule, each
 * scored by the check command's exit status; the first passing harness is
 * adopted and one JSONL record is appended. No model is invoked here: this is
 * the controller, and its evidence is the check command.
 *
 * usage: harness2_c11 <task-id> <ledger-path> <check-cmd> [rule ...]
 * exit 0 when some probe passed, 1 when none did, 2 on usage error.
 */
#include <stdio.h>
#include <string.h>

#include "wubu_harness2.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <task-id> <ledger-path> <check-cmd> [rule ...]\n", argv[0]);
        return 2;
    }
    const char *task = argv[1];
    const char *ledger = argv[2];
    const char *check = argv[3];

    wubu_h2_probe_t probes[WUBU_H2_MAX_PROBES];
    wubu_h2_harness_t *keep[WUBU_H2_MAX_PROBES];
    int n = 0;

    wubu_h2_harness_t *base = wubu_h2_harness_new();
    if (!base) return 2;
    keep[n] = base;
    probes[n].probe_index = 0;
    wubu_h2_probe_run(&probes[n], 0, base, check, 30);
    printf("probe 0 (baseline, 0 rules): check_rc=%d\n", probes[n].check_rc);
    n++;

    for (int i = 4; i < argc && n < WUBU_H2_MAX_PROBES; i++) {
        wubu_h2_harness_t *h = wubu_h2_harness_new();
        if (!h) return 2;
        if (wubu_h2_harness_add_rule(h, argv[i]) != 0) {
            fprintf(stderr, "refused rule (duplicate, empty or too long): %s\n", argv[i]);
            wubu_h2_harness_free(h);
            continue;
        }
        keep[n] = h;   /* adoption reads the harness, so it must outlive the probe */
        wubu_h2_probe_run(&probes[n], n, h, check, 30);
        printf("probe %d (%d rule): check_rc=%d  %s\n",
               n, wubu_h2_harness_rule_count(h), probes[n].check_rc, argv[i]);
        n++;
    }

    int winner = wubu_h2_compose(probes, n);
    if (winner < 0) {
        printf("compose: no probe passed; nothing adopted\n");
        (void)wubu_h2_ledger_append(ledger, task, probes, n, -1);
        for (int i = 0; i < n; i++) wubu_h2_harness_free(keep[i]);
        return 1;
    }
    printf("compose: adopted probe %d\n", winner);
    char adopted[WUBU_H2_MAX_RULES][WUBU_H2_RULE_LEN];
    int na = wubu_h2_adopt(probes, n, winner, adopted, WUBU_H2_MAX_RULES);
    for (int i = 0; i < na; i++) printf("  adopted rule: %s\n", adopted[i]);
    if (wubu_h2_ledger_append(ledger, task, probes, n, winner) != 0) {
        fprintf(stderr, "ledger append failed: %s\n", ledger);
        return 2;
    }
    printf("ledger: %s now holds %d record(s)\n", ledger, wubu_h2_ledger_count(ledger));
    for (int i = 0; i < n; i++) wubu_h2_harness_free(keep[i]);
    return 0;
}