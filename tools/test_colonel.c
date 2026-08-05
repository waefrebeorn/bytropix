/*
 * test_colonel.c — tests for the Colonel dispatcher.
 *
 * Port of WuBuOS/src/runtime/tests/test_colonel.c.
 * Uses the wubu_colonel API (opaque struct + accessors).
 *
 * C11, no external deps.
 */
#include "wubu_colonel.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(c, m) do { \
    checks++; \
    if (!(c)) { printf("  FAIL: %s\n", m); failures++; } \
    else { printf("  PASS: %s\n", m); } \
} while (0)

/* The test eval: a REAL HolyC-style arithmetic evaluator substitute.
 * Returns 6 for "1+2+3", 42 for anything else. */
static int64_t fake_eval(const char *src) {
    if (src && strcmp(src, "1+2+3") == 0) return 6;
    return 42;
}

int main(void) {
    printf("=== test_colonel ===\n");

    /* ---- Test 1: parse: command classes ---- */
    printf("\n--- Test 1: parse ---\n");
    wubu_colonel_t c;
    CHECK(wubu_colonel_parse("run calc", &c) == WUBU_COLONEL_OK, "run parses");
    CHECK(wubu_colonel_get_class(&c) == WUBU_COL_CMD_APP, "run class");
    CHECK(strcmp(wubu_colonel_get_cmd(&c), "calc") == 0, "run name");

    CHECK(wubu_colonel_parse("eval 1+2+3", &c) == WUBU_COLONEL_OK, "eval parses");
    CHECK(wubu_colonel_get_class(&c) == WUBU_COL_CMD_EVAL, "eval class");
    CHECK(strcmp(wubu_colonel_get_arg(&c), "1+2+3") == 0, "eval arg");

    CHECK(wubu_colonel_parse("os shutdown", &c) == WUBU_COLONEL_OK, "os parses");
    CHECK(wubu_colonel_get_class(&c) == WUBU_COL_CMD_OS, "os class");

    CHECK(wubu_colonel_parse("sys reboot", &c) == WUBU_COLONEL_OK, "sys parses");
    CHECK(wubu_colonel_get_class(&c) == WUBU_COL_CMD_SYS, "sys class");

    CHECK(wubu_colonel_parse("agi close-gap", &c) == WUBU_COLONEL_OK, "agi parses");
    CHECK(wubu_colonel_get_class(&c) == WUBU_COL_CMD_AGI, "agi class");

    CHECK(wubu_colonel_parse("load wasm /tmp/app.wasm", &c) == WUBU_COLONEL_OK, "load parses");
    CHECK(wubu_colonel_get_class(&c) == WUBU_COL_CMD_LOAD, "load class");
    CHECK(strcmp(wubu_colonel_get_cmd(&c), "wasm") == 0, "load format");
    CHECK(strcmp(wubu_colonel_get_arg(&c), "/tmp/app.wasm") == 0, "load path");

    CHECK(wubu_colonel_parse("calc", &c) == WUBU_COLONEL_OK, "bare token parses");
    CHECK(wubu_colonel_get_class(&c) == WUBU_COL_CMD_APP, "bare -> app class");
    CHECK(strcmp(wubu_colonel_get_cmd(&c), "calc") == 0, "bare -> calc");

    CHECK(wubu_colonel_parse("   ", &c) == WUBU_COLONEL_EMPTY, "empty string");
    CHECK(wubu_colonel_parse("", &c) == WUBU_COLONEL_EMPTY, "zero-length");
    CHECK(wubu_colonel_parse(NULL, &c) == WUBU_COLONEL_BAD, "NULL line");
    CHECK(wubu_colonel_parse("run calc", NULL) == WUBU_COLONEL_BAD, "NULL result");

    /* ---- Test 2: dispatch ---- */
    printf("\n--- Test 2: dispatch ---\n");
    CHECK(wubu_colonel_dispatch("eval 1+2+3", &c, fake_eval) == WUBU_COLONEL_OK,
          "eval dispatched");
    CHECK(wubu_colonel_get_value(&c) == 6, "eval result routed");

    CHECK(wubu_colonel_dispatch("eval 9*9", &c, fake_eval) == WUBU_COLONEL_OK,
          "eval fallback dispatched");
    CHECK(wubu_colonel_get_value(&c) == 42, "fallback value");

    CHECK(wubu_colonel_dispatch("eval 9*9", &c, NULL) == WUBU_COLONEL_BAD,
          "eval without callback -> BAD");

    /* ---- Test 3: app registry ---- */
    printf("\n--- Test 3: app registry ---\n");
    CHECK(wubu_colonel_app_known("calc") == 1, "calc known");
    CHECK(wubu_colonel_app_known("bonzi") == 1, "bonzi known");
    CHECK(wubu_colonel_app_known("comfy") == 1, "comfy known");
    CHECK(wubu_colonel_app_known("not-a-real-app") == 0, "unknown rejected");

    CHECK(wubu_colonel_dispatch("run not-a-real-app", &c, NULL) == WUBU_COLONEL_UNKNOWN,
          "unknown app -> UNKNOWN");
    CHECK(wubu_colonel_dispatch("run calc", &c, NULL) == WUBU_COLONEL_OK,
          "known app -> OK (GUI launches)");
    CHECK(wubu_colonel_dispatch("calc", &c, NULL) == WUBU_COLONEL_OK,
          "bare calc -> OK");

    /* ---- Test 4: verb classes route ---- */
    printf("\n--- Test 4: verb routing ---\n");
    CHECK(wubu_colonel_dispatch("os shutdown", &c, NULL) == WUBU_COLONEL_OK,
          "os verb routes");
    CHECK(wubu_colonel_dispatch("load wasm /tmp/app.wasm", &c, NULL) == WUBU_COLONEL_OK,
          "load routes");
    CHECK(wubu_colonel_dispatch("os ", &c, NULL) == WUBU_COLONEL_UNKNOWN,
          "empty os verb -> UNKNOWN");
    CHECK(wubu_colonel_dispatch("agi close-gap", &c, NULL) == WUBU_COLONEL_OK,
          "agi verb routes");
    CHECK(wubu_colonel_dispatch("sys reboot", &c, NULL) == WUBU_COLONEL_OK,
          "sys verb routes");

    /* ---- Test 5: NULL safety ---- */
    printf("\n--- Test 5: NULL safety ---\n");
    CHECK(wubu_colonel_dispatch(NULL, &c, NULL) == WUBU_COLONEL_BAD, "NULL line dispatch");
    CHECK(wubu_colonel_dispatch("run calc", NULL, NULL) == WUBU_COLONEL_BAD, "NULL result dispatch");
    CHECK(wubu_colonel_app_known(NULL) == 0, "NULL app name");
    CHECK(wubu_colonel_app_known("") == 0, "empty app name");
    CHECK(wubu_colonel_get_class(NULL) == 0, "get_class(NULL)");
    CHECK(wubu_colonel_get_cmd(NULL) == NULL, "get_cmd(NULL)");

    printf("\n=== Results: %d checks, %d failures ===\n", checks, failures);
    if (failures == 0) {
        printf("ALL COLONEL TESTS PASSED\n");
        return 0;
    }
    return 1;
}
