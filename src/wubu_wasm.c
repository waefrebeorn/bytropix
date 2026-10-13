/* wubu_wasm.c -- RUNTIME #348 WASM execution leg.
 * See wubu_wasm.h. */
#include "wubu_wasm.h"
#include <string.h>
#include <limits.h>

int wubu_wasm_validate(const wubu_wasm_mod_t *m) {
    if (!m) return 0;
    /* magic: 00 61 73 6D ('\0asm'), version: 01 00 00 00 */
    if (m->magic[0] != 0x00 || m->magic[1] != 'a' ||
        m->magic[2] != 's' || m->magic[3] != 'm') return 0;
    return 1;
}

/* LEB128 unsigned decode (for const immediates and local indices). */
static int leb_u32(const uint8_t *code, int len, int *pos, uint32_t *out) {
    uint32_t result = 0; int shift = 0;
    while (*pos < len) {
        uint8_t b = code[(*pos)++];
        result |= (uint32_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) { *out = result; return 0; }
        shift += 7;
        if (shift > 28) return -1;
    }
    return -1;
}

/* LEB128 signed decode (for i32.const immediates). */
static int leb_i32(const uint8_t *code, int len, int *pos, int32_t *out) {
    int32_t result = 0; int shift = 0; uint8_t b;
    do {
        if (*pos >= len) return -1;
        b = code[(*pos)++];
        result |= (int32_t)(b & 0x7F) << shift;
        shift += 7;
    } while (b & 0x80);
    if (shift < 32 && (b & 0x40)) result |= -((int32_t)1 << shift);
    *out = result;
    return 0;
}

static int push(wubu_wasm_vm_t *vm, int32_t v) {
    if (vm->sp >= WUBU_WASM_STACK) return -1;
    vm->stack[vm->sp++] = v;
    return 0;
}
static int pop(wubu_wasm_vm_t *vm, int32_t *out) {
    if (vm->sp <= 0) return -1;
    *out = vm->stack[--vm->sp];
    return 0;
}

int32_t wubu_wasm_run(const wubu_wasm_mod_t *m) {
    if (!m || !wubu_wasm_validate(m)) return INT32_MIN;
    wubu_wasm_vm_t vm;
    memset(&vm, 0, sizeof vm);
    int pos = 0;
    while (pos < m->code_len) {
        uint8_t op = m->code[pos++];
        switch (op) {
            case W_NOP: break;
            case W_CONST: {
                int32_t v;
                if (leb_i32(m->code, m->code_len, &pos, &v) != 0) return INT32_MIN;
                if (push(&vm, v) != 0) return INT32_MIN;
                break;
            }
            case W_ADD: case W_SUB: case W_MUL: {
                int32_t b, a;
                if (pop(&vm, &b) != 0 || pop(&vm, &a) != 0) return INT32_MIN;
                int32_t r = (op == W_ADD) ? a + b : (op == W_SUB) ? a - b : a * b;
                if (push(&vm, r) != 0) return INT32_MIN;
                break;
            }
            case W_LOCAL_GET: {
                uint32_t idx;
                if (leb_u32(m->code, m->code_len, &pos, &idx) != 0) return INT32_MIN;
                if (idx >= WUBU_WASM_LOCALS) return INT32_MIN;
                if (push(&vm, vm.locals[idx]) != 0) return INT32_MIN;
                break;
            }
            case W_LOCAL_SET: {
                uint32_t idx; int32_t v;
                if (leb_u32(m->code, m->code_len, &pos, &idx) != 0) return INT32_MIN;
                if (idx >= WUBU_WASM_LOCALS) return INT32_MIN;
                if (pop(&vm, &v) != 0) return INT32_MIN;
                vm.locals[idx] = v;
                break;
            }
            case W_DROP: {
                int32_t v;
                if (pop(&vm, &v) != 0) return INT32_MIN;
                break;
            }
            case W_RETURN: case W_END: {
                int32_t r;
                if (pop(&vm, &r) != 0) return INT32_MIN;
                return r;
            }
            default:
                return INT32_MIN;   /* bad opcode */
        }
    }
    /* fell off the end: return top of stack if any */
    int32_t r;
    return pop(&vm, &r) == 0 ? r : INT32_MIN;
}
