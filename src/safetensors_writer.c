/*
 * safetensors_writer.c -- write safetensors files (F32).
 *
 * Layout:
 *   [ uint64 LE header_len ][ header JSON ][ padding to 8 ][ raw blob ]
 * The header maps tensor-name -> {dtype, shape, data_offsets} where
 * data_offsets are relative to the raw blob start.
 */
#include "safetensors_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t json_escape(char *out, size_t cap, const char *s)
{
    size_t k = 0;
    for (const char *p = s; *p && k + 6 < cap; p++) {
        if (*p == '"' || *p == '\\') { out[k++] = '\\'; out[k++] = *p; }
        else if (*p == '\n') { out[k++] = '\\'; out[k++] = 'n'; }
        else if (*p == '\r') { out[k++] = '\\'; out[k++] = 'r'; }
        else if ((unsigned char)*p < 0x20) {
            k += (size_t)snprintf(out + k, cap - k, "\\u%04x", *p);
        } else out[k++] = *p;
    }
    out[k] = 0;
    return k;
}

int st_write_f32(const char *path, const st_writer_tensor_t *tensors,
                 int n_tensors)
{
    if (!path || (!tensors && n_tensors > 0)) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* --- build the header JSON first (its length prefixes the file) --- */
    /* worst-case header size: name escapes + ~90 bytes per tensor */
    size_t hcap = 512;
    for (int i = 0; i < n_tensors; i++)
        hcap += strlen(tensors[i].name) * 2 + 128;
    char *hdr = (char *)malloc(hcap);
    if (!hdr) { fclose(f); return -1; }
    size_t k = 0;
    k += (size_t)snprintf(hdr + k, hcap - k, "{");
    uint64_t blob_off = 0;
    for (int i = 0; i < n_tensors; i++) {
        const st_writer_tensor_t *t = &tensors[i];
        char name_esc[256];
        json_escape(name_esc, sizeof(name_esc), t->name);
        if (i > 0) k += (size_t)snprintf(hdr + k, hcap - k, ",");
        k += (size_t)snprintf(hdr + k, hcap - k,
                              "\"%s\":{\"dtype\":\"F32\",\"shape\":[", name_esc);
        for (int d = 0; d < t->n_dims; d++) {
            if (d > 0) k += (size_t)snprintf(hdr + k, hcap - k, ",");
            k += (size_t)snprintf(hdr + k, hcap - k, "%lld",
                                  (long long)t->dims[d]);
        }
        uint64_t bytes = (uint64_t)t->n_elems * 4;
        k += (size_t)snprintf(hdr + k, hcap - k,
                              "],\"data_offsets\":[%llu,%llu]}",
                              (unsigned long long)blob_off,
                              (unsigned long long)(blob_off + bytes));
        blob_off += bytes;
    }
    k += (size_t)snprintf(hdr + k, hcap - k, "}");
    if (k >= hcap) { free(hdr); fclose(f); return -1; }

    /* --- write: header_len (uint64 LE), header (PADDED to 8), blob ---
     * The safetensors spec: the header itself is padded with spaces so
     * that (8 + header_len) is a multiple of 8 -- the reader computes
     * the data start as exactly 8 + header_len, no extra padding pass.
     * The reference python writer confirms: header_len INCLUDES the
     * trailing spaces. Our first attempt padded AFTER the header, which
     * the reader never sees -> misaligned data (the HF "file not fully
     * covered" error). The DA check caught this. */
    size_t pad = (8 - ((8 + k) % 8)) % 8;
    for (size_t i = 0; i < pad; i++) hdr[k++] = ' ';
    hdr[k] = 0;
    uint64_t hlen = (uint64_t)k;
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, k, f);
    for (int i = 0; i < n_tensors; i++)
        fwrite(tensors[i].data, 4, (size_t)tensors[i].n_elems, f);

    free(hdr);
    fclose(f);
    return 0;
}
