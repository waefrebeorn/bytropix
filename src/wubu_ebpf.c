/* wubu_ebpf.c -- KERNEL #17 eBPF probe points attached from /n.
 * See wubu_ebpf.h. Real, deterministic mini eBPF VM + probe registry. */

#include "wubu_ebpf.h"
#include <string.h>

int wubu_ebpf_init(wubu_ebpf_t *e)
{
    if (!e) return -1;
    memset(e, 0, sizeof(*e));
    return 0;
}

static int find_probe(const wubu_ebpf_t *e, const char *name)
{
    if (!e || !name) return -1;
    for (int i = 0; i < e->n; i++)
        if (e->probes[i].attached && strcmp(e->probes[i].name, name) == 0) return i;
    return -1;
}

int wubu_ebpf_attach(wubu_ebpf_t *e, const char *name, const char *point,
                     const wubu_ebpf_insn_t *prog, int prog_len)
{
    if (!e || !name || !point || !prog) return -1;
    if (prog_len <= 0 || prog_len > WUBU_EBPF_PROG_LEN) return -1;
    if (find_probe(e, name) >= 0) return -1;   /* dup name */
    if (e->n >= WUBU_EBPF_MAX_PROBES) return -1;
    wubu_ebpf_probe_t *p = &e->probes[e->n];
    memset(p, 0, sizeof(*p));
    strncpy(p->name, name, WUBU_EBPF_NAME_MAX - 1);
    strncpy(p->point, point, WUBU_EBPF_NAME_MAX - 1);
    for (int i = 0; i < prog_len; i++) p->prog[i] = prog[i];
    p->prog_len = prog_len;
    p->attached = 1;
    p->enabled = 1;
    return e->n++;
}

int wubu_ebpf_detach(wubu_ebpf_t *e, const char *name)
{
    if (!e) return -1;
    int i = find_probe(e, name);
    if (i < 0) return -1;
    e->probes[i].attached = 0;
    e->probes[i].enabled = 0;
    return 0;
}

int wubu_ebpf_set_enabled(wubu_ebpf_t *e, const char *name, int enabled)
{
    if (!e) return -1;
    int i = find_probe(e, name);
    if (i < 0) return -1;
    e->probes[i].enabled = enabled ? 1 : 0;
    return 0;
}

int64_t wubu_ebpf_run(const wubu_ebpf_probe_t *p, int64_t arg)
{
    if (!p) return 0;
    int64_t r[10] = {0};
    int flag = 0;
    int pc = 0;
    int guard = 0;
    while (pc >= 0 && pc < p->prog_len && guard++ < 1000) {
        const wubu_ebpf_insn_t *in = &p->prog[pc];
        switch (in->op) {
        case EBPF_MOV: if (in->dst>=0 && in->dst<10) r[in->dst] = in->imm; pc++; break;
        case EBPF_LD:  r[0] = arg; pc++; break;   /* r0 = event arg (working value) */
        case EBPF_ADD: if (in->dst>=0 && in->dst<10 && in->src>=0 && in->src<10) r[in->dst] += r[in->src]; pc++; break;
        case EBPF_MUL: if (in->dst>=0 && in->dst<10) r[in->dst] *= in->imm; pc++; break;
        case EBPF_CMP: if (in->dst>=0 && in->dst<10) flag = (r[in->dst] > in->imm) - (r[in->dst] < in->imm); pc++; break;
        case EBPF_JGT: if (flag > 0) pc += (int)in->imm; else pc++; break;
        case EBPF_RET: return r[0];
        default: pc++; break;
        }
    }
    return r[0];
}

static void log_event(wubu_ebpf_t *e, const char *probe, const char *point,
                      int64_t arg, int64_t result)
{
    wubu_ebpf_event_t *ev = &e->log[e->log_head];
    ev->seq = ++e->seq;
    strncpy(ev->probe, probe, WUBU_EBPF_NAME_MAX - 1);
    ev->probe[WUBU_EBPF_NAME_MAX - 1] = '\0';
    strncpy(ev->point, point, WUBU_EBPF_NAME_MAX - 1);
    ev->point[WUBU_EBPF_NAME_MAX - 1] = '\0';
    ev->arg = arg;
    ev->result = result;
    e->log_head = (e->log_head + 1) % WUBU_EBPF_LOG_MAX;
    if (e->log_count < WUBU_EBPF_LOG_MAX) e->log_count++;
}

int wubu_ebpf_fire(wubu_ebpf_t *e, const char *point, int64_t arg)
{
    if (!e || !point) return -1;
    int fired = 0;
    for (int i = 0; i < e->n; i++) {
        wubu_ebpf_probe_t *p = &e->probes[i];
        if (!p->attached || !p->enabled) continue;
        if (strcmp(p->point, point) != 0) continue;
        int64_t res = wubu_ebpf_run(p, arg);
        p->fires++;
        p->last_result = res;
        log_event(e, p->name, p->point, arg, res);
        fired++;
    }
    return fired;
}

int wubu_ebpf_log_get(const wubu_ebpf_t *e, int i, wubu_ebpf_event_t *out)
{
    if (!e || !out) return -1;
    if (i < 0 || i >= e->log_count) return -1;
    int start = (e->log_head - e->log_count + WUBU_EBPF_LOG_MAX) % WUBU_EBPF_LOG_MAX;
    *out = e->log[(start + i) % WUBU_EBPF_LOG_MAX];
    return 0;
}

int wubu_ebpf_log_count(const wubu_ebpf_t *e)
{
    return e ? e->log_count : 0;
}
