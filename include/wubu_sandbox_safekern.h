/*
 * wubu_sandbox_safekern.h -- Sandbox capability bridge → safekern (AX08).
 */
#ifndef WUBU_SANDBOX_SAFEKERN_H
#define WUBU_SANDBOX_SAFEKERN_H

#define WUBU_SBOX_MAX_CAPS 16
#define WUBU_SBOX_MAX_CAP 64

typedef struct {
    char caps[WUBU_SBOX_MAX_CAPS][WUBU_SBOX_MAX_CAP];
    int n_caps;
    int seccomp_enabled;
    int network_allowed;
    int filesystem_ro;
    int max_memory_mb;
} wubu_sandbox_t;

int wubu_sbox_init(wubu_sandbox_t *s);
int wubu_sbox_add_cap(wubu_sandbox_t *s, const char *cap);
int wubu_sbox_set_seccomp(wubu_sandbox_t *s, int enabled);
int wubu_sbox_set_network(wubu_sandbox_t *s, int allowed);
int wubu_sbox_set_fs_ro(wubu_sandbox_t *s, int ro);

/* AX08: safekern bridge */
int wubu_safekern_check_cap(const wubu_sandbox_t *sbox,
                                       const char *cap);
int wubu_safekern_check_exec(const wubu_sandbox_t *sbox,
                                       const char *cmd);
int wubu_safekern_check_mem(const wubu_sandbox_t *sbox,
                                       int mb_requested);

#endif