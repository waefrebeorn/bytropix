#ifndef WUBU_EBPF_H
#define WUBU_EBPF_H

/* wubu_ebpf.h -- KERNEL #17 eBPF probe points attached from /n.
 *
 * eBPF-style probe points attached/detached from the kernel via 9P control
 * (/n/kernel/... or /n/svc/<name>/ebpf). A probe is a tiny register-machine
 * program attached to a named probe point (a kernel event site); when the
 * event fires, the kernel runs the probe over an event context and records the
 * result. Probes can be attached, detached, enabled/disabled, and their firing
 * history is kept in a bounded log. Deterministic, real work (no stubs).
 *
 * The mini eBPF VM: registers r0..r9 (r0 = return/result). Instructions:
 *   MOV ri, imm        ri = imm
 *   LD arg             r1 = event arg (the fired event's value)
 *   ADD ri, rj         ri += rj
 *   MUL ri, imm        ri *= imm
 *   CMP ri, imm        sets flag (ri - imm)
 *   JGT off            if flag>0, pc += off (skip)
 *   RET                return r0
 * A probe program is an array of these ops, max 32. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_EBPF_MAX_PROBES 32
#define WUBU_EBPF_PROG_LEN 32
#define WUBU_EBPF_NAME_MAX 48
#define WUBU_EBPF_LOG_MAX 64

typedef enum {
    EBPF_MOV = 0,   /* MOV ri, imm */
    EBPF_LD  = 1,   /* LD -> r1 = event arg */
    EBPF_ADD = 2,   /* ADD ri, rj */
    EBPF_MUL = 3,   /* MUL ri, imm */
    EBPF_CMP = 4,   /* CMP ri, imm */
    EBPF_JGT = 5,   /* JGT off (skip if r_i > imm) */
    EBPF_RET = 6    /* RET (return r0) */
} wubu_ebpf_op_t;

typedef struct {
    wubu_ebpf_op_t op;
    int dst;        /* register index for MOV/ADD/MUL/CMP/JGT */
    int src;        /* register index for ADD */
    int64_t imm;    /* immediate / jump offset */
} wubu_ebpf_insn_t;

typedef struct {
    char name[WUBU_EBPF_NAME_MAX];     /* probe name */
    char point[WUBU_EBPF_NAME_MAX];    /* probe point (event site) it's on */
    wubu_ebpf_insn_t prog[WUBU_EBPF_PROG_LEN];
    int prog_len;
    int attached;   /* 1 if attached to its point */
    int enabled;    /* 1 if it fires (attached + enabled) */
    uint64_t fires; /* times it has run */
    int64_t last_result;
} wubu_ebpf_probe_t;

typedef struct {
    uint64_t seq;
    char probe[WUBU_EBPF_NAME_MAX];
    char point[WUBU_EBPF_NAME_MAX];
    int64_t arg;
    int64_t result;
} wubu_ebpf_event_t;

typedef struct {
    wubu_ebpf_probe_t probes[WUBU_EBPF_MAX_PROBES];
    int n;
    wubu_ebpf_event_t log[WUBU_EBPF_LOG_MAX];
    int log_head;
    int log_count;
    uint64_t seq;
} wubu_ebpf_t;

/* Initialise. Returns 0. */
int wubu_ebpf_init(wubu_ebpf_t *e);

/* Define + attach a probe program to a probe point. Returns probe index/-1. */
int wubu_ebpf_attach(wubu_ebpf_t *e, const char *name, const char *point,
                     const wubu_ebpf_insn_t *prog, int prog_len);

/* Detach a probe by name (stops firing, frees the slot for reuse). 0/-1. */
int wubu_ebpf_detach(wubu_ebpf_t *e, const char *name);

/* Enable/disable a probe (attached but disabled does not fire). 0/-1. */
int wubu_ebpf_set_enabled(wubu_ebpf_t *e, const char *name, int enabled);

/* Fire an event at a probe point with an argument value. Runs every enabled,
 * attached probe on that point over the arg, records each result in the log.
 * Returns the number of probes that fired, or -1 on bad args. */
int wubu_ebpf_fire(wubu_ebpf_t *e, const char *point, int64_t arg);

/* Run one probe program over an arg (the mini VM). Returns r0. */
int64_t wubu_ebpf_run(const wubu_ebpf_probe_t *p, int64_t arg);

/* Read log entry i (0 = oldest). Returns 0/-1. */
int wubu_ebpf_log_get(const wubu_ebpf_t *e, int i, wubu_ebpf_event_t *out);

/* Number of log entries retained. */
int wubu_ebpf_log_count(const wubu_ebpf_t *e);

#endif
