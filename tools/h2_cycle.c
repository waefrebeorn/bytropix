/* h2_cycle -- one full Harness^2 cycle with an external editor command.
 *
 *   baseline probe -> (if it fails) editor -> one candidate per proposed rule
 *   -> probe each -> compose by VERDICT -> adopt / keep -> library + ledger
 *
 * The editor is any command whose stdout is a bullet list ("- rule ..."). Using
 * a command (not a model API) is what lets tools/test_cycle.sh prove the whole
 * loop without spending a token; set --editor to `hermes -z ...` for the real
 * thing.
 *
 * usage: h2_cycle --task ID --check CMD --root DIR [--cwd DIR] [--editor CMD]
 *                 [--timeout S] [--max-rules N]
 * exit: 0 adopted-or-nothing-to-do, 1 no improvement, 2 usage/IO error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_harness2.h"

static int run_editor(const char *cmd, char *out, size_t cap) {
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    size_t got = fread(out, 1, cap - 1, p);
    out[got] = 0;
    int rc = pclose(p);
    return rc == 0 ? (int)got : -1;
}

static void ledger(const char *root, const char *task,
                   const wubu_h2_probe_t *probes, int n, int adopted) {
    char path[WUBU_H2_PATH_LEN];
    if (snprintf(path, sizeof path, "%s/LEDGER.jsonl", root) < (int)sizeof path)
        (void)wubu_h2_ledger_append(path, task, probes, n, adopted);
}

int main(int argc, char **argv) {
    const char *task = NULL, *check = NULL, *root = NULL, *cwd = NULL;
    const char *editor = getenv("H2_EDITOR_CMD");
    int timeout_s = 300, max_rules = 4;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--task") && i + 1 < argc) task = argv[++i];
        else if (!strcmp(argv[i], "--check") && i + 1 < argc) check = argv[++i];
        else if (!strcmp(argv[i], "--root") && i + 1 < argc) root = argv[++i];
        else if (!strcmp(argv[i], "--cwd") && i + 1 < argc) cwd = argv[++i];
        else if (!strcmp(argv[i], "--editor") && i + 1 < argc) editor = argv[++i];
        else if (!strcmp(argv[i], "--timeout") && i + 1 < argc) timeout_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-rules") && i + 1 < argc) max_rules = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }
    if (!task || !check || !root) {
        fprintf(stderr, "usage: %s --task ID --check CMD --root DIR [--cwd DIR] "
                        "[--editor CMD] [--timeout S] [--max-rules N]\n", argv[0]);
        return 2;
    }
    if (max_rules < 1 || max_rules > WUBU_H2_MAX_PROBES - 1) max_rules = 4;
    if (timeout_s < 1) timeout_s = 300;

    wubu_h2_probe_t probes[WUBU_H2_MAX_PROBES];
    wubu_h2_harness_t *keep[WUBU_H2_MAX_PROBES];
    int n = 0;

    /* 1. baseline */
    wubu_h2_harness_t *base = wubu_h2_harness_new();
    if (!base) return 2;
    {
        char dir[WUBU_H2_PATH_LEN];
        if (wubu_h2_harness_stage_named(base, root, "base", dir, sizeof dir) != 0) {
            fprintf(stderr, "cannot stage baseline under %s\n", root);
            return 2;
        }
    }
    keep[n] = base;
    wubu_h2_probe_run_in(&probes[n], 0, base, check, timeout_s, cwd);
    printf("baseline: check_rc=%d\n", probes[n].check_rc);
    n++;

    if (probes[0].check_rc == 0) {
        printf("baseline already passes; nothing to adopt\n");
        (void)ledger(root, task, probes, n, -1);
        wubu_h2_harness_free(base);
        return 0;
    }

    if (!editor || !editor[0]) {
        printf("no editor configured (--editor or H2_EDITOR_CMD); baseline only\n");
        (void)ledger(root, task, probes, n, -1);
        wubu_h2_harness_free(base);
        return 1;
    }

    /* 2. propose */
    char text[8192];
    if (run_editor(editor, text, sizeof text) < 0) {
        printf("editor failed; nothing adopted\n");
        (void)ledger(root, task, probes, n, -1);
        wubu_h2_harness_free(base);
        return 1;
    }
    char rules[WUBU_H2_MAX_PROBES][WUBU_H2_RULE_LEN];
    int nr = wubu_h2_parse_proposals(text, rules, max_rules);
    if (nr <= 0) {
        printf("editor proposed %d usable rule(s); nothing adopted\n", nr);
        (void)ledger(root, task, probes, n, -1);
        wubu_h2_harness_free(base);
        return 1;
    }
    for (int i = 0; i < nr; i++) printf("proposal %d: %s\n", i, rules[i]);

    /* 3. probe one candidate per rule: one edit per candidate keeps attribution
     *    clean, so a pass is not confounded. */
    for (int i = 0; i < nr && n < WUBU_H2_MAX_PROBES; i++) {
        wubu_h2_harness_t *h = wubu_h2_harness_new();
        if (!h) return 2;
        if (wubu_h2_harness_add_rule(h, rules[i]) != 0) { wubu_h2_harness_free(h); continue; }
        char label[32], dir[WUBU_H2_PATH_LEN];
        snprintf(label, sizeof label, "cand-%d", i);
        if (wubu_h2_harness_stage_named(h, root, label, dir, sizeof dir) != 0) {
            wubu_h2_harness_free(h);
            continue;
        }
        keep[n] = h;
        wubu_h2_probe_run_in(&probes[n], n, h, check, timeout_s, cwd);
        printf("cand-%d: check_rc=%d  %s\n", i, probes[n].check_rc, rules[i]);
        n++;
    }

    /* 4. compose */
    int winner = wubu_h2_compose_evidence(probes, n);
    char adopted[WUBU_H2_MAX_RULES][WUBU_H2_RULE_LEN];
    int na = winner >= 0 ? wubu_h2_adopt(probes, n, winner, adopted, WUBU_H2_MAX_RULES) : 0;

    if (winner < 0) {
        printf("compose: keep the current harness (no attributable improvement)\n");
        (void)ledger(root, task, probes, n, -1);
    } else {
        printf("compose: adopted probe %d with %d rule(s)\n", winner, na);
        for (int i = 0; i < na; i++) printf("  rule: %s\n", adopted[i]);
        (void)ledger(root, task, probes, n, winner);
        if (na > 0) {
            char lib[WUBU_H2_PATH_LEN];
            snprintf(lib, sizeof lib, "%s/library", root);
            const wubu_h2_harness_t *wh = NULL;
            for (int i = 0; i < n; i++) if (probes[i].probe_index == winner) wh = probes[i].harness;
            if (wh) {
                char ev[128];
                snprintf(ev, sizeof ev, "baseline rc=%d, candidate rc=%d",
                         probes[0].check_rc, probes[winner].check_rc);
                (void)wubu_h2_library_record(lib, task, wh, adopted[0], adopted[0], ev);
                printf("library: %s/%s/EXPERIENCE.md\n", lib, task);
            }
        }
    }

    for (int i = 0; i < n; i++) wubu_h2_harness_free(keep[i]);
    return winner >= 0 ? 0 : 1;
}