/* wubu_pacman.c -- OMARCHY #272 Arch base: pacman package manager driver.
 * See wubu_pacman.h. Parsing/validation/dispatch policy; exec is delegated. */
#include "wubu_pacman.h"
#include <string.h>

const char *wubu_pac_op_name(wubu_pac_op_t op) {
    switch (op) {
        case WUBU_PAC_QUERY:    return "query";
        case WUBU_PAC_SYNC:     return "sync";
        case WUBU_PAC_REMOVE:   return "remove";
        case WUBU_PAC_DATABASE: return "database";
        case WUBU_PAC_FILES:    return "files";
        case WUBU_PAC_UPGRADE:  return "upgrade";
        case WUBU_PAC_DEPTEST:  return "deptest";
        default:                return "invalid";
    }
}

int wubu_pacman_parse(const char *const *argv, int argc, wubu_pac_cmd_t *cmd) {
    if (!argv || !cmd || argc < 1) return -1;
    memset(cmd, 0, sizeof *cmd);
    cmd->op = WUBU_PAC_INVALID;

    int i = 1;   /* skip argv[0] = program name */
    int op_seen = 0;
    for (; i < argc; i++) {
        const char *a = argv[i];
        if (!a) return -1;
        if (a[0] == '-' && a[1] != '\0') {
            /* an option cluster, e.g. -Qi, -Syu, --noconfirm */
            if (a[1] == '-') {
                /* long option */
                if (strcmp(a, "--noconfirm") == 0) cmd->flags |= WUBU_PAC_FLAG_NOCONF;
                else if (strcmp(a, "--needed") == 0) cmd->flags |= WUBU_PAC_FLAG_NEEDED;
                else return -1;   /* unknown long option: fail-closed */
                continue;
            }
            /* short cluster: each char is a flag; the first of Q/S/R/D/F/U/T
             * sets the operation, the rest refine it. */
            for (const char *c = a + 1; *c; c++) {
                switch (*c) {
                    case 'Q': cmd->op = WUBU_PAC_QUERY;    op_seen = 1; break;
                    case 'S': cmd->op = WUBU_PAC_SYNC;     op_seen = 1; break;
                    case 'R': cmd->op = WUBU_PAC_REMOVE;   op_seen = 1; break;
                    case 'D': cmd->op = WUBU_PAC_DATABASE; op_seen = 1; break;
                    case 'F': cmd->op = WUBU_PAC_FILES;    op_seen = 1; break;
                    case 'U': cmd->op = WUBU_PAC_UPGRADE;  op_seen = 1; break;
                    case 'T': cmd->op = WUBU_PAC_DEPTEST;  op_seen = 1; break;
                    case 'i': cmd->flags |= WUBU_PAC_FLAG_INFO;    break;
                    case 'l': cmd->flags |= WUBU_PAC_FLAG_LIST;    break;
                    case 'y': cmd->flags |= WUBU_PAC_FLAG_REFRESH; break;
                    case 'u': cmd->flags |= WUBU_PAC_FLAG_SYSUP;   break;
                    case 'e': /* explicitly-installed (query filter) */ break;
                    case 'n': /* download only / no extract */ break;
                    case 's': /* search (files/query) */ break;
                    default:  return -1;   /* unknown flag: fail-closed */
                }
            }
        } else {
            /* a target (package name or file) */
            if (cmd->n_targets >= WUBU_PAC_MAX_ARGS) return -1;
            size_t tl = strlen(a);
            if (tl == 0 || tl >= 64) return -1;
            strncpy(cmd->targets[cmd->n_targets++], a, 63);
        }
    }
    if (!op_seen) return -1;   /* no operation flag: fail-closed */
    return 0;
}

int wubu_pacman_validate(const wubu_pac_cmd_t *cmd) {
    if (!cmd || cmd->op == WUBU_PAC_INVALID) return -1;
    switch (cmd->op) {
        case WUBU_PAC_SYNC:
            /* -S with no targets and no -u is a no-op/refresh-only; -Syu is
             * a full upgrade. Either is valid. -S with targets installs them. */
            return 0;
        case WUBU_PAC_REMOVE:
        case WUBU_PAC_UPGRADE:
        case WUBU_PAC_DEPTEST:
            /* these require at least one target */
            return cmd->n_targets > 0 ? 0 : -1;
        case WUBU_PAC_QUERY:
        case WUBU_PAC_DATABASE:
        case WUBU_PAC_FILES:
            /* targets optional */
            return 0;
        default:
            return -1;
    }
}

int wubu_pacman_needs_network(const wubu_pac_cmd_t *cmd) {
    if (!cmd) return 0;
    if (cmd->op == WUBU_PAC_SYNC) return 1;         /* sync always hits repos */
    if (cmd->op == WUBU_PAC_UPGRADE) return 0;      /* local file */
    if (cmd->op == WUBU_PAC_QUERY || cmd->op == WUBU_PAC_FILES ||
        cmd->op == WUBU_PAC_REMOVE || cmd->op == WUBU_PAC_DATABASE ||
        cmd->op == WUBU_PAC_DEPTEST) return 0;
    return 0;
}
