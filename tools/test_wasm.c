/* test_wasm.c -- RUNTIME #348 WASM execution leg.
 * Proves: validate magic, run const/add/sub/mul, locals, drop, return,
 * stack underflow -> error, bad opcode -> error, fail-closed on NULL. */
#include "wubu_wasm.h"
#include <stdio.h>
#include <limits.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

static wubu_wasm_mod_t mk(const uint8_t *code, int len) {
    wubu_wasm_mod_t m;
    m.magic[0]=0x00; m.magic[1]='a'; m.magic[2]='s'; m.magic[3]='m';
    memset(m.code, 0, sizeof m.code);
    memcpy(m.code, code, len);
    m.code_len = len;
    return m;
}
#define RUN(c) ({ wubu_wasm_mod_t _m = mk((c), sizeof(c)); wubu_wasm_run(&_m); })

int main(void) {
    /* validate */
    uint8_t c0[] = { W_CONST, 42, W_END };
    wubu_wasm_mod_t m = mk(c0, sizeof c0);
    CHECK(wubu_wasm_validate(&m) == 1, "valid magic");
    m.magic[0] = 0xFF;
    CHECK(wubu_wasm_validate(&m) == 0, "bad magic rejected");

    /* const 42 */
    wubu_wasm_mod_t m1 = mk(c0, sizeof c0);
    CHECK(wubu_wasm_run(&m1) == 42, "const 42 -> 42");

    /* (3 + 4) * 2 = 14 : const3 const4 add const2 mul end */
    uint8_t c2[] = { W_CONST,3, W_CONST,4, W_ADD, W_CONST,2, W_MUL, W_END };
    CHECK(RUN(c2) == 14, "(3+4)*2 = 14");

    /* (10 - 3) = 7 */
    uint8_t c3[] = { W_CONST,10, W_CONST,3, W_SUB, W_END };
    CHECK(RUN(c3) == 7, "10-3 = 7");

    /* locals: local.set 0 (5); local.get 0; local.get 0; add -> 10 */
    uint8_t c4[] = { W_CONST,5, W_LOCAL_SET,0, W_LOCAL_GET,0, W_LOCAL_GET,0, W_ADD, W_END };
    CHECK(RUN(c4) == 10, "local set/get: 5+5 = 10");

    /* drop: const 1 const 2 drop -> 1 */
    uint8_t c5[] = { W_CONST,1, W_CONST,2, W_DROP, W_END };
    CHECK(RUN(c5) == 1, "drop discards top (1)");

    /* negative const (LEB signed): -5 */
    uint8_t c6[] = { W_CONST, 0x7B, W_END };  /* 0x7B = -5 in signed LEB */
    CHECK(RUN(c6) == -5, "const -5 (LEB signed)");

    /* stack underflow -> error */
    uint8_t c7[] = { W_ADD, W_END };
    CHECK(RUN(c7) == INT32_MIN, "add on empty -> error");

    /* bad opcode -> error */
    uint8_t c8[] = { 0xEE, W_END };
    CHECK(RUN(c8) == INT32_MIN, "bad opcode -> error");

    /* invalid module -> error */
    wubu_wasm_mod_t bad = mk(c0, sizeof c0);
    bad.magic[1] = 'X';
    CHECK(wubu_wasm_run(&bad) == INT32_MIN, "invalid module -> error");

    /* fail-closed on NULL */
    CHECK(wubu_wasm_run(NULL) == INT32_MIN, "NULL run -> error");
    CHECK(wubu_wasm_validate(NULL) == 0, "NULL validate -> 0");

    printf("\n=== test_wasm: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
