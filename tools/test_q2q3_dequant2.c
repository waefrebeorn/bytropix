// Test Q2_K and Q3_K dequant with known reference values
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef QK_K
#define QK_K 256
#endif

static float ggml_half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) f = (sign << 31) | (0x7f - 15 + 1) << 23 | (mant << 13);
    else if (exp == 31) f = (sign << 31) | 0x7f800000 | (mant << 13);
    else f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

void dequantize_q2_K_row(const uint8_t *data, float *output, int64_t n_elems) {
    int nb = (int)((n_elems + QK_K - 1) / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t *block = data + i * 84;
        float d = ggml_half_to_float(*(const uint16_t*)(block + 80));
        float min = ggml_half_to_float(*(const uint16_t*)(block + 82));
        const uint8_t *scales = block;
        const uint8_t *q = block + 16;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) *output++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) *output++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_q3_K_row(const uint8_t *data, float *output, int64_t n_elems) {
    int nb = (int)((n_elems + QK_K - 1) / QK_K);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t *scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {
        const uint8_t *block = data + i * 110;
        float d_all = ggml_half_to_float(*(const uint16_t*)(block + 108));
        const uint8_t *q = block + 32;
        const uint8_t *hm = block;

        memcpy(aux, block + 96, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *output++ = dl * ((int8_t)((q[l+0] >> shift) & 3) - ((hm[l+0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *output++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
            hm += 32;
        }
    }
}

// Declare external dequant functions from gguf_reader.c
extern void dequantize_q5_K_row(const uint8_t *data, float *output, int64_t n_elems);
extern void dequantize_q6_K_row(const uint8_t *data, float *output, int64_t n_elems);
extern void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);

int main() {
    
    // Test: set all 6-bit scales to 33 (which gives scale = 33-32 = 1)
    // scales[i] = 33 = 0b100001
    // low 4 bits = 0x01, high 2 bits = 0x02
    
    // The 12 bytes pack 16 × 6-bit values:
    // bytes 0-3: low 4 bits of scales[0..7] (interleaved: [0lo][1lo] in byte0, [2lo][3lo] in byte1, etc.)
    // Wait no - the reference reads aux[0] = *(uint32_t*)(scales) which is bytes 0-3
    // aux[0] = byte0 | byte1<<8 | byte2<<16 | byte3<<32
    // Each byte contains two 4-bit low values: byte0 = scales[1]<<4 | scales[0]
    // So aux[0] after reshuffle becomes the scale values themselves
    
    // Let me simplify: I'll test against the ACTUAL MTP model's blk.40.ffn_down_exps
    // by comparing our dequant to what llama.cpp produces
    
    // Simple test: verify a real tensor
    const char *model_path = "/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) { printf("FAIL: can't open model\n"); return 1; }
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.40.ffn_gate_shexp.weight");
    if (!t) { printf("FAIL: can't find test tensor\n"); gguf_close(ctx); return 1; }
    
    int64_t n_elems = t->dims[0] * t->dims[1];
    printf("Test tensor: %s type=%d dims=[%ld,%ld] n_elems=%ld\n", 
           t->name, t->ggml_type, (long)t->dims[0], (long)t->dims[1], (long)n_elems);
    
    // Read as F32 via our reader
    float *f32_ref = (float *)malloc(n_elems * sizeof(float));
    int ok = gguf_read_tensor_f32(ctx, t, f32_ref, n_elems);
    if (!ok) { printf("FAIL: can't read test tensor as F32\n"); free(f32_ref); gguf_close(ctx); return 1; }
    
    // Now read raw Q5_K bytes and dequant with Q5_K
    int64_t raw_sz = gguf_raw_size(t->ggml_type, n_elems);
    uint8_t *raw = (uint8_t *)malloc(raw_sz);
    FILE *mf = fopen(model_path, "rb");
    if (!mf) { printf("FAIL: can't reopen model\n"); free(raw); free(f32_ref); gguf_close(ctx); return 1; }
    
    // Find data start: after tensor info, need alignment
    // Read GGUF header to find data start
    fseek(mf, 0, SEEK_SET);
    char magic[4]; fread(magic, 1, 4, mf);
    uint32_t version; fread(&version, 4, 1, mf);
    uint64_t nt, nkv; fread(&nt, 8, 1, mf); fread(&nkv, 8, 1, mf);
    
    // Skip KV pairs
    for (uint64_t ki = 0; ki < nkv; ki++) {
        uint64_t klen; fread(&klen, 8, 1, mf); fseek(mf, klen, SEEK_CUR);
        uint32_t vtype; fread(&vtype, 4, 1, mf);
        switch (vtype) {
            case 0: case 1: case 2: fseek(mf, 4, SEEK_CUR); break;
            case 3: fseek(mf, 1, SEEK_CUR); break;
            case 4: { uint64_t sl; fread(&sl, 8, 1, mf); fseek(mf, sl, SEEK_CUR); break; }
            case 5: { uint32_t at; fread(&at, 4, 1, mf); uint64_t al; fread(&al, 8, 1, mf); 
                      for (uint64_t ai = 0; ai < al; ai++) { 
                          if (at == 4) { uint64_t sl; fread(&sl, 8, 1, mf); fseek(mf, sl, SEEK_CUR); }
                          else fseek(mf, 4, SEEK_CUR);
                      } break; }
            case 7: fseek(mf, 8, SEEK_CUR); break;
            case 8: fseek(mf, 8, SEEK_CUR); break;
            case 12: fseek(mf, 8, SEEK_CUR); break;
            default: fseek(mf, 4, SEEK_CUR); break;
        }
    }
    
    // Skip tensor info  
    fseek(mf, nt * (8 + 256 + 4 + 4*8 + 4 + 8), SEEK_CUR);
    // Align to 32
    long pos = ftell(mf);
    long pad = (32 - (pos % 32)) % 32;
    if (pad) fseek(mf, pad, SEEK_CUR);
    
    printf("Data start: %ld, data_offset=%llu\n", ftell(mf), (unsigned long long)t->data_offset);
    
    fseek(mf, ftell(mf) + t->data_offset, SEEK_SET);
    size_t nread = fread(raw, 1, raw_sz, mf);
    fclose(mf);
    
    if (nread != (size_t)raw_sz) { printf("FAIL: read %zu/%lld\n", nread, (long long)raw_sz); free(raw); free(f32_ref); gguf_close(ctx); return 1; }
    
    // Dequant using our Q5_K dequant
    float *f32_deq = (float *)malloc(n_elems * sizeof(float));
    switch (t->ggml_type) {
        case GGML_TYPE_Q5_K: dequantize_q5_K_row(raw, f32_deq, n_elems); break;
        case GGML_TYPE_Q6_K: dequantize_q6_K_row(raw, f32_deq, n_elems); break;
        case GGML_TYPE_IQ2_XXS: dequantize_iq2_xxs_row(raw, f32_deq, n_elems); break;
        case GGML_TYPE_IQ3_XXS: dequantize_iq3_xxs_row(raw, f32_deq, n_elems); break;
        default: printf("Unsupported type %d\n", t->ggml_type); break;
    }
    
    // Compare
    double max_err = 0, sum = 0;
    for (int64_t i = 0; i < n_elems; i++) {
        double err = fabs((double)f32_ref[i] - (double)f32_deq[i]);
        if (err > max_err) max_err = err;
        sum += err;
    }
    printf("F32 vs dequant: max_err=%f, avg_err=%f, first=%f vs %f\n", 
           max_err, sum/n_elems, f32_ref[0], f32_deq[0]);
    
    // Now test Q2_K/Q3_K dequant on actual model data
    // Find a Q2_K tensor (blk.40.ffn_gate_exps)
    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_exps.weight");
    if (t) {
        n_elems = t->dims[0] * t->dims[1];  // per-expert
        raw_sz = gguf_raw_size(t->ggml_type, n_elems);
        // Need to read from blob data
        if (!ctx->data_blob) gguf_buffer_data(ctx);
        if (ctx->data_blob) {
            const uint8_t *src = (const uint8_t *)ctx->data_blob + t->data_offset;
            // Dequant first expert
            float *deq_out = (float *)malloc(n_elems * sizeof(float));
            dequantize_q2_K_row(src, deq_out, n_elems);
            double sum2 = 0;
            for (int64_t i = 0; i < n_elems; i++) sum2 += fabs((double)deq_out[i]);
            printf("Q2_K dequant (expert 0, 1st 5):");
            for (int i = 0; i < 5 && i < n_elems; i++) printf(" %f", deq_out[i]);
            printf(" ... avg_mag=%f\n", sum2/n_elems);
            free(deq_out);
        }
    }
    
    // Test Q3_K on blk.40.ffn_down_exps
    t = gguf_find_tensor(ctx, "blk.40.ffn_down_exps.weight");
    if (t) {
        n_elems = t->dims[0] * t->dims[1];  // per-expert
        if (!ctx->data_blob) gguf_buffer_data(ctx);
        if (ctx->data_blob) {
            const uint8_t *src = (const uint8_t *)ctx->data_blob + t->data_offset;
            float *deq_out = (float *)malloc(n_elems * sizeof(float));
            dequantize_q3_K_row(src, deq_out, n_elems);
            double sum2 = 0;
            for (int64_t i = 0; i < n_elems; i++) sum2 += fabs((double)deq_out[i]);
            printf("Q3_K dequant (expert 0, 1st 5):");
            for (int i = 0; i < 5 && i < n_elems; i++) printf(" %f", deq_out[i]);
            printf(" ... avg_mag=%f\n", sum2/n_elems);
            free(deq_out);
        }
    }
    
    // Test Q8_0 on eh_proj
    t = gguf_find_tensor(ctx, "blk.40.nextn.eh_proj.weight");
    if (t) {
        n_elems = t->dims[0] * t->dims[1];
        printf("eh_proj: type=%d dims=[%ld,%ld] n_elems=%ld\n", t->ggml_type, (long)t->dims[0], (long)t->dims[1], (long)n_elems);
        if (!ctx->data_blob) gguf_buffer_data(ctx);
        if (ctx->data_blob) {
            const uint8_t *src = (const uint8_t *)ctx->data_blob + t->data_offset;
            float *deq_out = (float *)malloc(n_elems * sizeof(float));
            // Q8_0: 32 elements per block, 34 bytes each
            int64_t n_blocks = (n_elems + 31) / 32;
            for (int64_t b = 0; b < n_blocks; b++) {
                uint16_t d_bits; memcpy(&d_bits, src + b*34, 2);
                float d = ggml_half_to_float(d_bits);
                const int8_t *qs = (const int8_t *)(src + b*34 + 2);
                for (int j = 0; j < 32 && b*32 + j < n_elems; j++)
                    deq_out[b*32 + j] = d * (float)qs[j];
            }
            double sum2 = 0;
            for (int64_t i = 0; i < n_elems; i++) sum2 += fabs((double)deq_out[i]);
            printf("Q8_0 dequant (1st 5):");
            for (int i = 0; i < 5 && i < n_elems; i++) printf(" %f", deq_out[i]);
            printf(" ... avg_mag=%f\n", sum2/n_elems);
            free(deq_out);
        }
    }
    
    free(raw); free(f32_ref); free(f32_deq);
    gguf_close(ctx);
    return 0;
}
