/* wubu_abi_shim.c -- RUNTIME #339 foreign-ABI translation (libc/syscall shim).
 * See wubu_abi_shim.h. Real, deterministic Linux x86-64 -> VSL translation. */

#include "wubu_abi_shim.h"
#include <string.h>

/* Linux x86-64 syscall numbers. */
enum {
    LNX_READ = 0, LNX_WRITE = 1, LNX_OPEN = 2, LNX_CLOSE = 3, LNX_FSTAT = 5,
    LNX_MMAP = 9, LNX_MPROTECT = 10, LNX_BRK = 12, LNX_EXIT = 60
};

int wubu_abi_init(wubu_abi_shim_t *s)
{
    if (!s) return -1;
    memset(s, 0, sizeof(*s));
    s->default_deny = 1;
    /* identity-ish mapping: same arg count, args pass through in order.
     * arg_regs[i] = i means foreign arg i -> native slot i. */
    int id6[6] = {0,1,2,3,4,5};
    wubu_abi_map(s, LNX_READ,     WUBU_VSL_READ,     3, id6);
    wubu_abi_map(s, LNX_WRITE,    WUBU_VSL_WRITE,    3, id6);
    wubu_abi_map(s, LNX_OPEN,     WUBU_VSL_OPEN,     3, id6);
    wubu_abi_map(s, LNX_CLOSE,    WUBU_VSL_CLOSE,    1, id6);
    wubu_abi_map(s, LNX_FSTAT,    WUBU_VSL_FSTAT,    2, id6);
    wubu_abi_map(s, LNX_MMAP,     WUBU_VSL_MMAP,     6, id6);
    wubu_abi_map(s, LNX_MPROTECT, WUBU_VSL_MPROTECT, 3, id6);
    wubu_abi_map(s, LNX_BRK,      WUBU_VSL_BRK,      1, id6);
    wubu_abi_map(s, LNX_EXIT,     WUBU_VSL_EXIT,     1, id6);
    return 0;
}

int wubu_abi_map(wubu_abi_shim_t *s, int foreign_nr, int native_id,
                 int n_args, const int *arg_regs)
{
    if (!s) return -1;
    if (n_args < 0 || n_args > WUBU_ABI_MAX_ARGS) return -1;
    /* override existing entry for this foreign_nr */
    for (int i = 0; i < s->n; i++) {
        if (s->table[i].foreign_nr == foreign_nr) {
            s->table[i].native_id = native_id;
            s->table[i].n_args = n_args;
            for (int a = 0; a < n_args; a++)
                s->table[i].arg_regs[a] = arg_regs ? arg_regs[a] : a;
            return 0;
        }
    }
    if (s->n >= WUBU_ABI_TABLE_MAX) return -1;
    wubi_abi_entry_t *e = &s->table[s->n++];
    e->foreign_nr = foreign_nr;
    e->native_id = native_id;
    e->n_args = n_args;
    for (int a = 0; a < n_args; a++)
        e->arg_regs[a] = arg_regs ? arg_regs[a] : a;
    return 0;
}

int wubu_abi_set_default_deny(wubu_abi_shim_t *s, int deny)
{
    if (!s) return -1;
    s->default_deny = deny ? 1 : 0;
    return 0;
}

int wubu_abi_translate(wubu_abi_shim_t *s, int foreign_nr,
                       const int64_t *foreign_args, int n_foreign_args,
                       int *native_id, int64_t *native_args)
{
    if (!s || !native_id || !native_args) return -1;
    const wubi_abi_entry_t *e = NULL;
    for (int i = 0; i < s->n; i++)
        if (s->table[i].foreign_nr == foreign_nr) { e = &s->table[i]; break; }
    if (!e) {
        /* unknown foreign syscall */
        s->n_denied++;
        return -1;   /* fail-closed regardless of default_deny: an unmapped
                      * foreign syscall has no safe native target. */
    }
    if (e->native_id == WUBU_VSL_NOSYS) {
        s->n_denied++;
        return -1;   /* explicitly mapped to "no such syscall" */
    }
    *native_id = e->native_id;
    /* remap each native arg slot from the foreign arg at arg_regs[i] */
    for (int i = 0; i < e->n_args; i++) {
        int fr = e->arg_regs[i];
        if (fr >= 0 && fr < n_foreign_args && fr < WUBU_ABI_MAX_ARGS)
            native_args[i] = foreign_args[fr];
        else
            native_args[i] = 0;   /* missing foreign arg -> 0 */
    }
    s->n_translated++;
    return e->n_args;
}

uint64_t wubu_abi_translated(const wubu_abi_shim_t *s) { return s ? s->n_translated : 0; }
uint64_t wubu_abi_denied(const wubu_abi_shim_t *s) { return s ? s->n_denied : 0; }
