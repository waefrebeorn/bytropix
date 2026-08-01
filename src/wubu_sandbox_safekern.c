/*
 * wubu_sandbox_safekern.c -- Sandbox capability bridge → safekern (AX08). C11.
 *
 * Convergence (sandbox + safekern 7-hop):
 *   - AX08: bridge between sandbox isolation and safekern capabilities.
 *     When code is exec'd in a sandbox, safekern checks the capability
 *     token before allowing the sandboxed code to access host resources.
 */
#include "wubu_sandbox_safekern.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WUBU_SBOX_MAX_CAPS 16

int wubu_sbox_init(wubu_sandbox_t *s) {
    if (!s) return -1;
    s->n_caps = 0;
    s->seccomp_enabled = 0;
    s->network_allowed = 0;
    s->filesystem_ro = 1;
    s->max_memory_mb = 512;
    return 0;
}

int wubu_sbox_add_cap(wubu_sandbox_t *s, const char *cap) {
    if (!s || !cap) return -1;
    if (s->n_caps >= WUBU_SBOX_MAX_CAPS) return -1;
    if (strlen(cap) >= WUBU_SBOX_MAX_CAP) return -1;
    snprintf(s->caps[s->n_caps++], WUBU_SBOX_MAX_CAP, "%s", cap);
    return 0;
}

int wubu_sbox_set_seccomp(wubu_sandbox_t *s, int enabled) {
    if (!s) return -1;
    s->seccomp_enabled = enabled ? 1 : 0;
    return 0;
}

int wubu_sbox_set_network(wubu_sandbox_t *s, int allowed) {
    if (!s) return -1;
    s->network_allowed = allowed ? 1 : 0;
    return 0;
}

int wubu_sbox_set_fs_ro(wubu_sandbox_t *s, int ro) {
    if (!s) return -1;
    s->filesystem_ro = ro ? 1 : 0;
    return 0;
}

/* ---- Safekern bridge (AX08) ---- */
int wubu_safekern_check_cap(const wubu_sandbox_t *sbox, const char *cap) {
    if (!sbox || !cap) return 0;  /* default-deny */
    if (!sbox->seccomp_enabled) return 0;  /* seccomp must be on */
    for (int i = 0; i < sbox->n_caps; i++) {
        if (strcmp(sbox->caps[i], cap) == 0) return 1;
    }
    return 0;  /* not in allowlist */
}

int wubu_safekern_check_exec(const wubu_sandbox_t *sbox, const char *cmd) {
    if (!sbox || !cmd) return 0;
    if (!sbox->seccomp_enabled) return 0;
    /* Default-deny: only explicitly allowed caps can exec. */
    return wubu_safekern_check_cap(sbox, "exec");
}

int wubu_safekern_check_mem(const wubu_sandbox_t *sbox, int mb_requested) {
    if (!sbox) return 0;
    return (mb_requested <= sbox->max_memory_mb) ? 1 : 0;
}