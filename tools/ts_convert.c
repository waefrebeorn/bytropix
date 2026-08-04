/*
 * ts_convert.c — one-liner model-format interchange via the tensor store.
 *
 *   ts_convert <in> <out> [safetensors|gguf|st|q8|mixed]
 *
 * Opens ANY native format (safetensors / GGUF / .st dump) as a catalog,
 * streams the export one tensor at a time (bounded RAM — never
 * load-all-then-save). Default target: safetensors.
 */
#include "wubu_tensor_store.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in> <out> [safetensors|gguf|st|q8|mixed]\n",
                argv[0]);
        return 2;
    }
    const char *in  = argv[1];
    const char *out = argv[2];
    const char *tgt = (argc > 3) ? argv[3] : "safetensors";

    wubu_ts_fmt src = wubu_ts_sniff(in);
    if (src == WUBU_TS_UNKNOWN) {
        fprintf(stderr, "cannot sniff format of %s\n", in);
        return 1;
    }
    const char *src_name = src == WUBU_TS_SAFETENSORS ? "safetensors"
                         : src == WUBU_TS_GGUF       ? "gguf"
                         :                              ".st dump";
    printf("open %s as %s catalog\n", in, src_name);

    wubu_tensor_store_t *ts = wubu_ts_open(in);
    if (!ts) { fprintf(stderr, "open failed\n"); return 1; }
    printf("  %d tensors\n", wubu_ts_count(ts));

    int rc = -1;
    if (!strcmp(tgt, "safetensors")) {
        rc = wubu_ts_export(ts, WUBU_TS_SAFETENSORS, out);
    } else if (!strcmp(tgt, "gguf")) {
        rc = wubu_ts_export(ts, WUBU_TS_GGUF, out);
    } else if (!strcmp(tgt, "st")) {
        rc = wubu_ts_export(ts, WUBU_TS_STDUMP, out);
    } else if (!strcmp(tgt, "q8")) {
        rc = wubu_ts_export_q8(ts, out);
    } else if (!strcmp(tgt, "mixed")) {
        rc = wubu_ts_export_mixed(ts, out);
    } else {
        fprintf(stderr, "unknown target '%s'\n", tgt);
    }
    wubu_ts_close(ts);
    if (rc != 0) { fprintf(stderr, "export failed (%d)\n", rc); return 1; }
    printf("exported -> %s (%s)\n", out, tgt);
    return 0;
}
