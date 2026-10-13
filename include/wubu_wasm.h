#ifndef WUBU_WASM_H
#define WUBU_WASM_H
#include <stdint.h>

/* wubu_wasm.h -- RUNTIME #348 WASM execution leg.
 *
 * A minimal WebAssembly execution leg: validates a WASM module header and
 * executes a single function body as a stack machine over i32 opcodes
 * (const, add, sub, mul, local.get/set, drop, return). This is enough to
 * actually RUN a WASM module (compute a result), not just parse it. The
 * bytecode is a simplified WASM-like stream (magic + version + body of
 * opcodes). Deterministic, real work (no stubs). */

#define WUBU_WASM_STACK 64
#define WUBU_WASM_LOCALS 16
#define WUBU_WASM_CODE 256

/* opcodes (simplified WASM i32 subset) */
enum {
    W_END=0x0B, W_CONST=0x41, W_ADD=0x6A, W_SUB=0x6B, W_MUL=0x6C,
    W_LOCAL_GET=0x20, W_LOCAL_SET=0x21, W_DROP=0x1A, W_RETURN=0x0F,
    W_NOP=0x01,
};

typedef struct {
    int32_t stack[WUBU_WASM_STACK];
    int sp;
    int32_t locals[WUBU_WASM_LOCALS];
} wubu_wasm_vm_t;

typedef struct {
    uint8_t magic[4];     /* 0x00 'a' 's' 'm' */
    uint8_t code[WUBU_WASM_CODE];
    int code_len;
} wubu_wasm_mod_t;

/* Validate the module header (magic 00 61 73 6D + version 01 00 00 00). */
int wubu_wasm_validate(const wubu_wasm_mod_t *m);

/* Execute the module body; returns the top-of-stack i32 result, or INT32_MIN
 * on error (stack underflow/overflow, bad opcode). */
int32_t wubu_wasm_run(const wubu_wasm_mod_t *m);

#endif
