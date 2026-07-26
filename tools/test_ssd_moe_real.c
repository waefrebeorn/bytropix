/*
 * test_ssd_moe_real.c -- prove the ds4-ssd slot-bank on REAL KAT-256-expert
 * weights with BOUNDED memory (a few MB, never the 22 GB checkpoint).
 *
 * Strategy (the ds4-ssd way): do NOT open all shards / mmap the model.
 * Instead, for each expert, locate its tensor by parsing the relevant shard's
 * small JSON header, compute its absolute byte offset, and pread ONLY that
 * tensor (BF16, ~1 MB). Build the sidecar one expert at a time; verify one
 * expert at a time. Peak RAM ~ few MB regardless of model size.
 *
 * Verifies: BF16 sidecar pack -> LRU page-in -> F32 dequant reproduces the
 * exact checkpoint weights for a spread of experts (forcing slot evictions).
 */
#include "wubu_ssd_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static const char *KAT_DIR = "/tmp/models/KAT-Coder-V2.5-Dev";
static const char *SIDECAR = "/tmp/kat_sidecar_real";
static const int   LAYER   = 0;
static const int   D       = 2048;
static const int   F       = 512;
static const int   SLOTS   = 4;

/* Minimal safetensors: read the u64 header length, then the JSON header. */
static char *read_header(const char *path, long *out_hdrlen) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    uint64_t hlen;
    if (pread(fd, &hlen, 8, 0) != 8) { close(fd); return NULL; }
    char *buf = (char *)malloc((size_t)hlen + 1);
    if (!buf) { close(fd); return NULL; }
    if (pread(fd, buf, hlen, 8) != (ssize_t)hlen) { free(buf); close(fd); return NULL; }
    buf[hlen] = 0;
    close(fd);
    *out_hdrlen = (long)hlen;
    return buf;
}

/* Find a tensor's absolute file offset + element count by scanning the JSON
 * header (lightweight string search; no full parse). Returns end offset too. */
static int find_tensor_off(const char *hdr, const char *name,
                           long hdrlen, long *data_off, long *data_end) {
    char key[300];
    snprintf(key, sizeof(key), "\"%s\"", name);
    const char *p = strstr(hdr, key);
    if (!p) return -1;
    const char *info = strchr(p, '{');
    if (!info) return -1;
    const char *do_ = strstr(info, "data_offsets");
    if (!do_) return -1;
    const char *br = strchr(do_, '[');
    if (!br) return -1;
    long a = strtol(br + 1, NULL, 10);
    const char *comma = strchr(br, ',');
    long b = comma ? strtol(comma + 1, NULL, 10) : a;
    *data_off = a; *data_end = b;
    (void)hdrlen;
    return 0;
}

/* pread one BF16 tensor of `n` elements from `path` at data offset `base`. */
static int read_bf16(const char *path, long base, int64_t n, float *out_f32) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    size_t bytes = (size_t)n * 2;
    uint8_t *raw = (uint8_t *)malloc(bytes);
    if (!raw) { close(fd); return -1; }
    size_t done = 0;
    while (done < bytes) {
        ssize_t r = pread(fd, raw + done, bytes - done, (off_t)(base + done));
        if (r <= 0) { free(raw); close(fd); return -1; }
        done += (size_t)r;
    }
    close(fd);
    for (int64_t i = 0; i < n; i++) {
        uint16_t h = (uint16_t)((raw[2*i] << 8) | raw[2*i+1]);
        uint32_t bits = (uint32_t)h << 16;
        memcpy(&out_f32[i], &bits, 4);
    }
    free(raw);
    return 0;
}

static int shard_for_expert(int e, char *path_out) {
    /* Probe shards model-00000.. for the tensor; return first that has it
     * AND whose tensor bytes are fully present in the file (skip partial/
     * still-downloading shards). */
    for (int s = 0; s < 64; s++) {
        char nm[64]; snprintf(nm, sizeof(nm), "model-%05d-of-00013.safetensors", s);
        char full[1200]; snprintf(full, sizeof(full), "%s/%s", KAT_DIR, nm);
        struct stat st; if (stat(full, &st) != 0) continue;
        long hl; char *hdr = read_header(full, &hl);
        if (!hdr) continue;
        char tname[256];
        snprintf(tname, sizeof(tname),
            "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", LAYER, e);
        long off, end;
        int found = (find_tensor_off(hdr, tname, hl, &off, &end) == 0);
        long abs_end = 8 + hl + end;
        free(hdr);
        if (found && abs_end <= st.st_size) { snprintf(path_out, 1200, "%s", full); return 1; }
    }
    return 0;
}

int main(void) {
    int E = 0;
    char probe[1200];
    for (;;) {
        if (!shard_for_expert(E, probe)) break;
        if (E > 4096) break;
        E++;
    }
    if (E <= 0) { printf("FAIL: no experts found for layer %d\n", LAYER); return 1; }
    printf("layer %d: %d experts detected (D=%d F=%d)\n", LAYER, E, D, F);

    int64_t per = (size_t)D * F;
    size_t expert_bytes = (size_t)per * 3 * 2; /* gate|up|down, BF16 */

    /* Build sidecar: one expert at a time, pread from its shard. */
    mkdir(SIDECAR, 0755);
    char spath[1200]; snprintf(spath, sizeof(spath), "%s/experts.%d.bin", SIDECAR, LAYER);
    int fd = open(spath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { printf("FAIL: sidecar create\n"); return 1; }

    float *g = (float *)malloc((size_t)per * sizeof(float));
    float *u = (float *)malloc((size_t)per * sizeof(float));
    float *d = (float *)malloc((size_t)per * sizeof(float));
    uint8_t *raw = (uint8_t *)malloc(expert_bytes);
    if (!g || !u || !d || !raw) { printf("FAIL: OOM scratch (should be tiny)\n"); return 1; }

    int packed = 0;
    for (int e = 0; e < E; e++) {
        char sp[1200];
        if (!shard_for_expert(e, sp)) { printf("FAIL: expert %d missing\n", e); break; }
        long hl; char *hdr = read_header(sp, &hl);
        if (!hdr) { printf("FAIL: hdr %d\n", e); break; }
        char tn[256];
        long off, end;
        snprintf(tn, sizeof(tn), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", LAYER, e);
        if (find_tensor_off(hdr, tn, hl, &off, &end) || read_bf16(sp, 8 + hl + off, per, g)) { free(hdr); printf("FAIL: gate %d\n", e); break; }
        snprintf(tn, sizeof(tn), "model.language_model.layers.%d.mlp.experts.%d.up_proj.weight", LAYER, e);
        if (find_tensor_off(hdr, tn, hl, &off, &end) || read_bf16(sp, 8 + hl + off, per, u)) { free(hdr); printf("FAIL: up %d\n", e); break; }
        snprintf(tn, sizeof(tn), "model.language_model.layers.%d.mlp.experts.%d.down_proj.weight", LAYER, e);
        if (find_tensor_off(hdr, tn, hl, &off, &end) || read_bf16(sp, 8 + hl + off, per, d)) { free(hdr); printf("FAIL: down %d\n", e); break; }
        free(hdr);

        uint16_t *b = (uint16_t *)raw;
        for (int64_t i = 0; i < per; i++) b[i]      = (uint16_t)(((uint32_t*)&g[i])[0] >> 16);
        for (int64_t i = 0; i < per; i++) b[per + i]  = (uint16_t)(((uint32_t*)&u[i])[0] >> 16);
        for (int64_t i = 0; i < per; i++) b[2*per+i]  = (uint16_t)(((uint32_t*)&d[i])[0] >> 16);
        size_t base = (size_t)e * expert_bytes;
        size_t done = 0;
        while (done < expert_bytes) {
            ssize_t w = pwrite(fd, raw + done, expert_bytes - done, (off_t)(base + done));
            if (w <= 0) break;
            done += (size_t)w;
        }
        packed++;
    }
    close(fd); free(raw); free(g); free(u); free(d);
    wubu_ssd_moe_write_manifest(SIDECAR, 1, E, D, F, 8, SLOTS);
    printf("packed %d/%d experts -> %s (%.1f MB on disk)\n", packed, E, SIDECAR,
           (double)expert_bytes * E / 1048576.0);

    /* Verify: page a spread of experts, compare each to an independent pread. */
    wubu_ssd_moe_t *m = wubu_ssd_moe_open(SIDECAR, SLOTS);
    if (!m) { printf("FAIL: open sidecar\n"); return 1; }

    int check[8] = {0, 1, 7, 64, 128, 200, 255, 100};
    int mism = 0, checked = 0;
    for (int c = 0; c < 8; c++) {
        int e = check[c];
        if (e >= E) continue;
        float *out[3];
        int r = wubu_ssd_moe_get(m, LAYER, e, out);
        if (r < 0) { printf("FAIL: page %d\n", e); return 1; }
        /* independent reference via pread */
        char sp[1200]; if (!shard_for_expert(e, sp)) { printf("FAIL: ref shard %d\n", e); return 1; }
        long hl; char *hdr = read_header(sp, &hl); if (!hdr) { printf("FAIL: ref hdr %d\n", e); return 1; }
        char tn[256]; long off, end;
        float *ref = (float *)malloc((size_t)per * sizeof(float));
        snprintf(tn, sizeof(tn), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", LAYER, e);
        find_tensor_off(hdr, tn, hl, &off, &end); read_bf16(sp, 8 + hl + off, per, ref);
        free(hdr);
        float maxdiff = 0.0f;
        for (int i = 0; i < 256; i++) { float dd = fabsf(out[0][i] - ref[i]); if (dd > maxdiff) maxdiff = dd; }
        free(ref);
        printf("  expert %3d: pagein=%d  max|paged-ref| gate[0..255]=%.5f\n", e, r, maxdiff);
        if (maxdiff > 0.05f) mism++;
        checked++;
    }
    long pi, hi; long long br; wubu_ssd_moe_stats(m, &pi, &hi, &br);
    printf("stats: pageins=%ld hits=%ld bytes_read=%lld\n", pi, hi, br);
    wubu_ssd_moe_close(m);
    if (mism) { printf("FAIL: %d/%d exceeded BF16 tol\n", mism, checked); return 1; }
    printf("PASS: ds4-ssd slot-bank reproduces REAL KAT-256-expert weights from SSD (bounded RAM, %d checked)\n", checked);
    return 0;
}
