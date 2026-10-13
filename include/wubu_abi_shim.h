#ifndef WUBU_ABI_SHIM_H
#define WUBU_ABI_SHIM_H

/* wubu_abi_shim.h -- RUNTIME #339 foreign-ABI translation (libc/syscall shim).
 *
 * Third-party binaries compiled for a foreign ABI (Linux x86-64, the common
 * case for containers) issue syscalls by foreign number with a foreign
 * register/argument convention. To run them ON the AGI kernel, the kernel
 * translates each foreign syscall to its native VSL syscall personality,
 * remapping the argument registers to the native convention. This module is
 * the translation core: a foreign-number -> native-syscall table plus
 * argument-count/register mapping. Unknown foreign syscalls are denied
 * (fail-closed) rather than silently mis-dispatched. Deterministic, real work.
 *
 * Foreign ABI modelled: Linux x86-64 (syscall number in rax, args in
 * rdi/rsi/rdx/r10/r8/r9). Native VSL convention: syscall id + up to 6 args in
 * v1..v6. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_ABI_MAX_ARGS 6
#define WUBU_ABI_TABLE_MAX 256

/* Native VSL syscall ids the shim can translate to. */
enum {
    WUBU_VSL_READ = 0, WUBU_VSL_WRITE = 1, WUBU_VSL_OPEN = 2, WUBU_VSL_CLOSE = 3,
    WUBU_VSL_MMAP = 9, WUBU_VSL_MPROTECT = 10, WUBU_VSL_BRK = 12,
    WUBU_VSL_EXIT = 60, WUBU_VSL_FSTAT = 5,
    WUBU_VSL_NOSYS = -1
};

typedef struct {
    int foreign_nr;                 /* foreign (Linux x86-64) syscall number */
    int native_id;                  /* native VSL syscall id (WUBU_VSL_NOSYS = deny) */
    int n_args;                     /* argument count to remap (0..6) */
    int arg_regs[WUBU_ABI_MAX_ARGS];/* native arg-slot index for each foreign arg */
} wubi_abi_entry_t;

typedef struct {
    wubi_abi_entry_t table[WUBU_ABI_TABLE_MAX];
    int n;
    int default_deny;               /* 1 = unknown foreign nr is denied */
    uint64_t n_translated;
    uint64_t n_denied;
} wubu_abi_shim_t;

/* Initialise with the default Linux x86-64 -> VSL table + default-deny.
 * Returns 0. */
int wubu_abi_init(wubu_abi_shim_t *s);

/* Add/override a translation entry. Returns 0/-1. */
int wubu_abi_map(wubu_abi_shim_t *s, int foreign_nr, int native_id,
                 int n_args, const int *arg_regs);

/* Translate a foreign syscall. On success fills native_id + remapped args[],
 * returns the native arg count. On deny (unknown + default_deny, or mapped to
 * WUBU_VSL_NOSYS) returns -1 and increments the denied counter. */
int wubu_abi_translate(wubu_abi_shim_t *s, int foreign_nr,
                       const int64_t *foreign_args, int n_foreign_args,
                       int *native_id, int64_t *native_args);

/* Set default-deny behaviour (1 = deny unknown, 0 = map unknown to NOSYS).
 * Returns 0. */
int wubu_abi_set_default_deny(wubu_abi_shim_t *s, int deny);

/* Counters. */
uint64_t wubu_abi_translated(const wubu_abi_shim_t *s);
uint64_t wubu_abi_denied(const wubu_abi_shim_t *s);

#endif
