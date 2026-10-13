/* test_ebpf.c -- KERNEL #17 tests: eBPF probe points attached from /n. */
#include "wubu_ebpf.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

/* helper to build an insn */
static wubu_ebpf_insn_t I(wubu_ebpf_op_t op, int dst, int src, int64_t imm) {
    wubu_ebpf_insn_t x; x.op=op; x.dst=dst; x.src=src; x.imm=imm; return x;
}

int main(void)
{
    wubu_ebpf_t e;
    wubu_ebpf_event_t ev;

    CHECK(wubu_ebpf_init(&e) == 0);

    /* probe "count": r0 = arg * 2  (LD; MUL r0,2; RET) */
    wubu_ebpf_insn_t p1[] = { I(EBPF_LD,0,0,0), I(EBPF_MUL,0,0,2), I(EBPF_RET,0,0,0) };
    CHECK(wubu_ebpf_attach(&e, "count", "sys_enter", p1, 3) == 0);
    CHECK(wubu_ebpf_attach(&e, "count", "sys_enter", p1, 3) == -1);  /* dup name */
    CHECK(wubu_ebpf_attach(&e, "bad", "x", p1, 0) == -1);           /* prog_len 0 */

    /* run the VM directly: arg=21 -> 42 */
    CHECK(wubu_ebpf_run(&e.probes[0], 21) == 42);
    CHECK(wubu_ebpf_run(&e.probes[0], 0) == 0);

    /* fire the event: only "count" is on sys_enter */
    CHECK(wubu_ebpf_fire(&e, "sys_enter", 5) == 1);
    CHECK(e.probes[0].fires == 1);
    CHECK(e.probes[0].last_result == 10);   /* 5*2 */
    CHECK(wubu_ebpf_fire(&e, "sys_enter", 7) == 1);
    CHECK(e.probes[0].last_result == 14);

    /* firing a different point fires nothing */
    CHECK(wubu_ebpf_fire(&e, "sys_exit", 100) == 0);
    CHECK(e.probes[0].fires == 2);          /* unchanged */

    /* log records both fires */
    CHECK(wubu_ebpf_log_count(&e) == 2);
    CHECK(wubu_ebpf_log_get(&e, 0, &ev) == 0);
    CHECK(ev.arg == 5 && ev.result == 10 && strcmp(ev.probe, "count") == 0);
    CHECK(wubu_ebpf_log_get(&e, 1, &ev) == 0);
    CHECK(ev.arg == 7 && ev.result == 14);

    /* a second probe on the same point: both fire */
    wubu_ebpf_insn_t p2[] = { I(EBPF_LD,0,0,0), I(EBPF_MOV,1,0,1), I(EBPF_ADD,0,1,0), I(EBPF_RET,0,0,0) };
    CHECK(wubu_ebpf_attach(&e, "inc", "sys_enter", p2, 4) == 1);
    CHECK(wubu_ebpf_fire(&e, "sys_enter", 10) == 2);   /* count + inc */
    CHECK(e.probes[0].last_result == 20);   /* count: 10*2 */
    CHECK(e.probes[1].last_result == 11);   /* inc: 10+1 */

    /* disable "count": only "inc" fires */
    CHECK(wubu_ebpf_set_enabled(&e, "count", 0) == 0);
    CHECK(wubu_ebpf_fire(&e, "sys_enter", 10) == 1);
    CHECK(e.probes[1].last_result == 11);
    CHECK(e.probes[0].fires == 3);          /* unchanged since disabled */
    CHECK(wubu_ebpf_set_enabled(&e, "count", 1) == 0);

    /* detach "count": only "inc" fires */
    CHECK(wubu_ebpf_detach(&e, "count") == 0);
    CHECK(wubu_ebpf_fire(&e, "sys_enter", 10) == 1);
    CHECK(wubu_ebpf_detach(&e, "count") == -1);   /* already detached */
    CHECK(wubu_ebpf_set_enabled(&e, "count", 1) == -1);  /* detached */

    /* CMP/JGT: probe that returns the arg when > 100, else 0. LD loads arg
     * into r0, so CMP/JGT operate on r0 directly. */
    wubu_ebpf_insn_t p3[] = {
        I(EBPF_LD,0,0,0),       /* r0 = arg */
        I(EBPF_CMP,0,0,100),    /* flag = arg > 100 ? */
        I(EBPF_JGT,0,0,2),      /* if big, skip the MOV r0,0 (pc+=2 -> RET) */
        I(EBPF_MOV,0,0,0),      /* r0 = 0 (small) */
        I(EBPF_RET,0,0,0)
    };
    CHECK(wubu_ebpf_attach(&e, "big", "sys_exit", p3, 5) == 2);
    CHECK(wubu_ebpf_run(&e.probes[2], 50) == 0);    /* <= 100 -> 0 */
    CHECK(wubu_ebpf_run(&e.probes[2], 150) == 150); /* > 100 -> arg */
    CHECK(wubu_ebpf_run(&e.probes[2], 100) == 0);   /* not > 100 -> 0 */

    /* bad args */
    CHECK(wubu_ebpf_fire(&e, NULL, 1) == -1);
    CHECK(wubu_ebpf_fire(NULL, "x", 1) == -1);
    CHECK(wubu_ebpf_log_get(&e, 999, &ev) == -1);

    if (fails == 0) printf("ALL EBPF TESTS PASSED\n");
    else printf("%d EBPF TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
