#ifndef WUBU_MODSIGN_H
#define WUBU_MODSIGN_H

/* wubu_modsign.h -- KERNEL #23 kernel modules signed + verified on load.
 *
 * A kernel-module signature gate: every module must carry a valid signature
 * that verifies against a trusted key before it is loaded into the kernel.
 * Unsigned or mismatched modules are REJECTED (fail-closed) -- the kernel
 * never runs untrusted code. This is the module-load analogue of verified
 * boot. Deterministic, real work (no stubs). */

#define WUBU_MS_MAX 32

typedef struct {
    char name[32];
    unsigned long long signature;   /* the module's claimed signature */
    unsigned long long trusted;     /* the trusted signature for this module */
    int loaded;
} wubu_ms_mod_t;

typedef struct {
    wubu_ms_mod_t mods[WUBU_MS_MAX];
    int n;
    unsigned long long kernel_key;  /* the trusted signing key */
} wubu_modsign_t;

/* Initialise with the kernel's trusted signing key. */
int wubu_ms_init(wubu_modsign_t *s, unsigned long long kernel_key);

/* Enroll a trusted module signature (a module the kernel will accept).
 * Returns 0/-1. */
int wubu_ms_enroll(wubu_modsign_t *s, const char *name, unsigned long long signature);

/* Attempt to load a module. The module's signature must match the enrolled
 * trusted signature (derived via the kernel key). Returns 1 if loaded
 * (verified), 0 if rejected (unsigned/mismatch/unknown), -1 on error. */
int wubu_ms_load(wubu_modsign_t *s, const char *name, unsigned long long signature);

/* Is a module currently loaded? 1/0. */
int wubu_ms_is_loaded(const wubu_modsign_t *s, const char *name);

#endif
