/*
 * merge_gguf.c — Merge GGUF shards (e.g. -00001-of-00003 ...) into ONE file.
 *
 * wubuwizard's gguf_open() reads a single file, so multi-shard DeepSeek-V4
 * Config-I (3 x ~32 GB) cannot load. This tool concatenates the shards into a
 * single valid GGUF: it reuses gguf_open() (which tolerates TurboQuant types
 * 45/46/47), copies each tensor's raw bytes from the mmap'd blob, and rewrites
 * the header with cumulative data_offsets + split.count patched to 1.
 *
 * Usage:  merge_gguf <shard1.gguf> <out.gguf>
 *         (shard count + sibling paths are derived from split.count / filename)
 */

#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Minimal KV helper: locate the byte range of the KV section in shard 1 so we
 * can copy it verbatim, and patch split.count / split.no / split.tensors.count. */

static int patch_split_kv(uint8_t *kv, size_t kv_len) {
    const char *keys[] = {"split.count", "split.no", "split.tensors.count"};
    uint64_t vals[] = {1, 0, 0}; /* count=1, no=0, tensors.count=0 (single file) */
    int patched = 0;
    for (int k = 0; k < 3; k++) {
        const char *key = keys[k];
        size_t klen = strlen(key);
        for (size_t i = 0; i + klen + 12 < kv_len; i++) {
            if (memcmp(kv + i, key, klen) == 0) {
                size_t p = i + klen;
                uint32_t kind = (uint32_t)kv[p] | ((uint32_t)kv[p+1]<<8) |
                                ((uint32_t)kv[p+2]<<16) | ((uint32_t)kv[p+3]<<24);
                if (kind == 6) { /* uint64 */
                    size_t vp = p + 4;
                    uint64_t v = vals[k];
                    kv[vp]   = (uint8_t)(v & 0xFF);
                    kv[vp+1] = (uint8_t)((v>>8)&0xFF);
                    kv[vp+2] = (uint8_t)((v>>16)&0xFF);
                    kv[vp+3] = (uint8_t)((v>>24)&0xFF);
                    kv[vp+4] = (uint8_t)((v>>32)&0xFF);
                    kv[vp+5] = (uint8_t)((v>>40)&0xFF);
                    kv[vp+6] = (uint8_t)((v>>48)&0xFF);
                    kv[vp+7] = (uint8_t)((v>>56)&0xFF);
                    patched++;
                    break;
                }
            }
        }
    }
    return patched;
}

static inline int64_t ne_of(const gguf_tensor_info *ti) {
    int64_t ne = 1; for (int d = 0; d < ti->n_dims; d++) ne *= ti->dims[d];
    return ne;
}

/* Given the true byte size of a tensor (from file offsets) and its element
   count, find the ggml_type whose gguf_raw_size() matches. This corrects
   corrupted type labels in the source GGUF (e.g. a tensor tagged F32 whose
   bytes are actually Q2_0 / TurboQuant). Returns the corrected type, or the
   original type if none matches. */
static int ggml_type_for_bytes(int64_t n_elems, int64_t bytes) {
    static const int candidates[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_Q8_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ1_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS,
        GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F64,
        GGML_TYPE_IQ1_M, GGML_TYPE_BF16, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
        GGML_TYPE_TQ3_1S, GGML_TYPE_TQ4_1S, GGML_TYPE_Q2_0
    };
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        int64_t rb = gguf_raw_size(candidates[i], n_elems);
        if (rb > 0 && rb == bytes) return candidates[i];
    }
    return -1; /* no match */
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <shard1.gguf> <out.gguf>\n", argv[0]);
        return 1;
    }
    const char *shard1 = argv[1];
    const char *outpath = argv[2];

    gguf_ctx *c1 = gguf_open(shard1);
    if (!c1) { fprintf(stderr, "Failed to open %s\n", shard1); return 1; }
    int corrected_any = 0;
    long n_corrected = 0;

    /* Determine shard count + sibling paths. */
    int n_shards = 1;
    char siblings[16][1024];
    strncpy(siblings[0], shard1, 1023);
    /* Derive from filename -NNNNN-of-NNNNN */
    char *of = strstr(shard1, "-of-");
    if (of) {
        int total = atoi(of + 4);
        if (total > 1 && total <= 16) {
            n_shards = total;
            /* replace the -NNNNN- before -of- with 1..total */
            char *dash = of;
            while (dash > shard1 && dash[-1] != '-') dash--;
            size_t prefix_len = dash - shard1;
            char prefix[1024]; memcpy(prefix, shard1, prefix_len); prefix[prefix_len]=0;
            for (int s = 1; s < n_shards; s++) {
                snprintf(siblings[s], 1024, "%s%05d-of-%05d.gguf", prefix, s+1, total);
            }
        }
    }
    fprintf(stderr, "[merge] %d shards detected\n", n_shards);
    for (int s = 0; s < n_shards; s++) fprintf(stderr, "  [%d] %s\n", s, siblings[s]);

    /* Collect all tensors across shards. */
    typedef struct { gguf_tensor_info info; int shard; uint64_t abs_off; uint64_t raw; } mt_t;
    mt_t *all = NULL; size_t cap = 0, cnt = 0;
    uint64_t merged_data = 0;
    for (int s = 0; s < n_shards; s++) {
        gguf_ctx *c = (s == 0) ? c1 : gguf_open(siblings[s]);
        if (!c) { fprintf(stderr, "Failed to open shard %d: %s\n", s, siblings[s]); return 1; }
        if (!gguf_buffer_data(c)) { fprintf(stderr, "Failed to mmap blob for shard %d\n", s); return 1; }
        for (int64_t t = 0; t < c->n_tensors; t++) {
            if (cnt >= cap) { cap = cap ? cap*2 : 1024; all = realloc(all, cap*sizeof(mt_t)); }
            gguf_tensor_info *ti = &c->tensors[t];
            all[cnt].info = *ti;
            all[cnt].shard = s;
            all[cnt].abs_off = c->data_blob_offset + ti->data_offset;
            /* Actual byte span from source file offsets (ground truth). */
            int64_t actual_raw;
            if (t + 1 < c->n_tensors)
                actual_raw = c->tensors[t+1].data_offset - ti->data_offset;
            else
                actual_raw = (int64_t)c->data_blob_size - ti->data_offset;
            /* Correct corrupted type labels: if declared type's byte size does
               not match the actual file span, relabel to the quant type that does. */
            int64_t declared_raw = gguf_raw_size(ti->ggml_type, ne_of(ti));
            if (declared_raw != actual_raw) {
                int corr = ggml_type_for_bytes(ne_of(ti), actual_raw);
                if (corr >= 0 && corr != ti->ggml_type) {
                    all[cnt].info.ggml_type = corr;
                    if (!corrected_any) { corrected_any = 1; }
                    n_corrected++;
                }
            }
            all[cnt].raw = (actual_raw > 0) ? (uint64_t)actual_raw : (uint64_t)declared_raw;
            cnt++;
        }
        if (s != 0) gguf_close(c);
    }
    fprintf(stderr, "[merge] total tensors: %lld\n", (long long)cnt);
    if (n_corrected > 0)
        fprintf(stderr, "[merge] CORRECTED %ld tensor type labels (declared size != actual)\n", n_corrected);

    /* Locate KV bytes in shard 1 (between header-start+24 and data_blob_offset minus tensor-info size). */
    /* Compute tensor-info section size for shard1. */
    uint64_t ti_size = 0;
    for (int64_t t = 0; t < c1->n_tensors; t++) {
        gguf_tensor_info *ti = &c1->tensors[t];
        ti_size += 8 + strlen(ti->name) + 4 + 8 + 8 * ti->n_dims;
    }
    uint64_t kv_start = 24; /* magic(4)+version(4)+n_tensors(8)+n_kv(8) for v3 */
    uint64_t kv_end = c1->data_blob_offset - ti_size;
    uint64_t kv_len = kv_end - kv_start;
    fprintf(stderr, "[merge] KV bytes: %llu..%llu (%llu)\n",
            (unsigned long long)kv_start, (unsigned long long)kv_end, (unsigned long long)kv_len);

    FILE *fin = fopen(shard1, "rb");
    if (!fin) { fprintf(stderr, "Cannot reopen %s\n", shard1); return 1; }
    uint8_t *kvbuf = malloc(kv_len);
    fseek(fin, (long)kv_start, SEEK_SET);
    if (fread(kvbuf, 1, kv_len, fin) != kv_len) { fprintf(stderr, "KV read fail\n"); return 1; }
    fclose(fin);
    int np = patch_split_kv(kvbuf, (size_t)kv_len);
    fprintf(stderr, "[merge] patched %d split.* KV entries\n", np);

    /* Write merged file. */
    FILE *out = fopen(outpath, "wb");
    if (!out) { fprintf(stderr, "Cannot write %s\n", outpath); return 1; }
    uint32_t magic = 0x46554747; /* 'GGUF' */
    uint32_t version = c1->version;
    uint64_t n_tensors = cnt;
    uint64_t n_kv = c1->n_kv;
    fwrite(&magic, 4, 1, out);
    fwrite(&version, 4, 1, out);
    fwrite(&n_tensors, 8, 1, out);
    fwrite(&n_kv, 8, 1, out);
    fwrite(kvbuf, 1, kv_len, out);
    free(kvbuf);

    /* Tensor infos with cumulative data_offset; track data section write pos. */
    uint64_t data_pos = 0;
    uint8_t *buf = malloc(1 << 20); (void)buf;
    fprintf(stderr, "[merge] writing %llu tensor infos...\n", (unsigned long long)cnt); fflush(stderr);
    for (size_t t = 0; t < cnt; t++) {
        if ((t % 200) == 0) { fprintf(stderr, "  ti %llu\n", (unsigned long long)t); fflush(stderr); }
        mt_t *m = &all[t];
        uint64_t nm = strlen(m->info.name);
        uint64_t off = data_pos;
        fwrite(&nm, 8, 1, out);
        fwrite(m->info.name, 1, nm, out);
        uint32_t gt = (uint32_t)m->info.ggml_type;
        fwrite(&gt, 4, 1, out);
        int nd = m->info.n_dims > 4 ? 4 : m->info.n_dims;
        fwrite(&nd, 8, 1, out);
        for (int d = 0; d < nd; d++) fwrite(&m->info.dims[d], 8, 1, out);
        fwrite(&off, 8, 1, out);
        data_pos += m->raw;
    }
    fprintf(stderr, "[merge] tensor infos done, data section = %llu bytes\n", (unsigned long long)data_pos); fflush(stderr);

    /* Pad header so the data section begins on a 32-byte boundary (GGUF alignment). */
    long hdr_pos = ftell(out);
    long pad = (32 - (hdr_pos % 32)) % 32;
    if (pad) { uint8_t zero = 0; for (long i = 0; i < pad; i++) fwrite(&zero, 1, 1, out); }
    fprintf(stderr, "[merge] header at %ld, padded %ld -> data starts at %ld\n", hdr_pos, pad, ftell(out)); fflush(stderr);

    /* Write data sections, copying raw bytes from each shard's mmap blob. */
    for (int s = 0; s < n_shards; s++) {
        gguf_ctx *c = (s == 0) ? c1 : gguf_open(siblings[s]);
        if (!c) { fprintf(stderr, "reopen fail %s\n", siblings[s]); return 1; }
        if (!gguf_buffer_data(c)) { fprintf(stderr, "re-mmap fail %s\n", siblings[s]); return 1; }
        for (int64_t t = 0; t < c->n_tensors; t++) {
            int64_t ne = 1; for (int d = 0; d < c->tensors[t].n_dims; d++) ne *= c->tensors[t].dims[d];
            int64_t rb = gguf_raw_size(c->tensors[t].ggml_type, ne);
            uint64_t raw = (rb > 0) ? (uint64_t)rb
                : (t+1 < c->n_tensors) ? (uint64_t)(c->tensors[t+1].data_offset - c->tensors[t].data_offset)
                                      : (uint64_t)(c->data_blob_size - c->tensors[t].data_offset);
            const uint8_t *src = (const uint8_t *)c->data_blob + c->tensors[t].data_offset;
            uint64_t left = raw;
            while (left) {
                size_t chunk = left > (1<<20) ? (1<<20) : (size_t)left;
                fwrite(src, 1, chunk, out);
                src += chunk; left -= chunk;
            }
        }
        if (s != 0) gguf_close(c);
    }
    fclose(out);
    fprintf(stderr, "[merge] wrote %s (%llu tensors)\n", outpath, (unsigned long long)cnt);
    free(all);
    gguf_close(c1);
    return 0;
}
