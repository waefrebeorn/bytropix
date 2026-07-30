/*
 * test_decoder_paths.c -- regression harness comparing Qwen/VLM decode variants.
 *
 * Build: make test_decoder_paths
 *
 * Not yet wired into live decode; this harness is the bag boundary for
 * offline step artifacts only.
 */

#include <stdio.h>
#include <string.h>

typedef enum {
    WUBU_DECODE_VLM     = 0,
    WUBU_DECODE_QWEN    = 1,
    WUBU_DECODE_HYBRID  = 2,
    WUBU_DECODE_COUNT
} wubu_decode_path_t;

static const char *wubu_decode_path_name(wubu_decode_path_t t) {
    switch (t) {
        case WUBU_DECODE_VLM:    return "VLM";
        case WUBU_DECODE_QWEN:   return "QWEN";
        case WUBU_DECODE_HYBRID: return "HYBRID";
        default:                 return "UNKNOWN";
    }
}

static wubu_decode_path_t wubu_decode_path_resolve(const char *name) {
    if (!name || !*name) return WUBU_DECODE_VLM;
    if (strstr(name, "qwen"))  return WUBU_DECODE_QWEN;
    if (strstr(name, "hybrid")) return WUBU_DECODE_HYBRID;
    return WUBU_DECODE_VLM;
}

int main(int argc, char **argv) {
    const char *path_name = argc > 1 ? argv[1] : "qwen";
    wubu_decode_path_t p = wubu_decode_path_resolve(path_name);
    printf("decode paths supported:\n");
    for (int i = 0; i < WUBU_DECODE_COUNT; i++)
        printf("  %d %s%s\n", i, wubu_decode_path_name((wubu_decode_path_t)i),
               i == (int)p ? " (resolved)" : "");
    return 0;
}
