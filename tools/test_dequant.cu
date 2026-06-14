/* test_dequant.c — Compare CPU vs GPU Q4_K dequant for first block */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define Q4K_BLOCK_SIZE 144
#define Q4K_N_ELEMS 256

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        uint32_t normal_f32 = (sign << 31) | ((1 + 112) << 23) | (mant << 13);
        float normal_val;
        memcpy(&normal_val, &normal_f32, 4);
        if (sign) return normal_val + 6.103515625e-5f;
        else return normal_val - 6.103515625e-5f;
    }
    if (exp == 31) {
        uint32_t f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
        float result;
        memcpy(&result, &f32, 4);
        return result;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

static __host__ __device__ inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j+4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

static void cpu_dequant(const uint8_t *data, float *output) {
    const uint8_t *block = data;
    float d = f16_to_f32(*(const uint16_t*)block);
    float dmin = f16_to_f32(*(const uint16_t*)(block+2));
    const uint8_t *scales = block + 4;
    const uint8_t *qs = block + 16;

    printf("  CPU d=%.6f dmin=%.6f\n", d, dmin);
    printf("  CPU scales raw: ");
    for (int i = 0; i < 12; i++) printf("%02x ", scales[i]);
    printf("\n");
    printf("  CPU qs raw: ");
    for (int i = 0; i < 16; i++) printf("%02x ", qs[i]);
    printf("\n");

    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, &sc, &m);
        float d1 = d * sc; float m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        float d2 = d * sc; float m2 = dmin * m;

        const uint8_t *bq = qs + j/2;
        for (int l = 0; l < 32; l++)
            output[j + l] = d1 * (bq[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++)
            output[j + 32 + l] = d2 * (bq[l] >> 4) - m2;
        is += 2;
    }

    printf("  CPU output[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", output[i]);
    printf("\n");
}

__global__ void gpu_dequant_kernel(const uint8_t *W_q, float *W_f32) {
    int tid = threadIdx.x;
    if (tid >= 256) return;
    const uint8_t *blk = W_q;
    float d, dmin;

    // fp16_to_f32_dev (matching CPU)
    uint16_t h;
    memcpy(&h, blk, 2);
    uint32_t sign = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x03FF;
    if (e == 0) { uint32_t nf = (sign << 31) | ((1+112) << 23) | (m << 13); float nv; memcpy(&nv, &nf, 4); d = sign ? nv + 6.103515625e-5f : nv - 6.103515625e-5f; }
    else if (e == 31) { uint32_t f32 = (sign << 31) | (0xFF << 23) | (m << 13); memcpy(&d, &f32, 4); }
    else { uint32_t f32 = (sign << 31) | ((e+112) << 23) | (m << 13); memcpy(&d, &f32, 4); }

    memcpy(&h, blk+2, 2);
    sign = (h >> 15) & 1; e = (h >> 10) & 0x1F; m = h & 0x03FF;
    if (e == 0) { uint32_t nf = (sign << 31) | ((1+112) << 23) | (m << 13); float nv; memcpy(&nv, &nf, 4); dmin = sign ? nv + 6.103515625e-5f : nv - 6.103515625e-5f; }
    else if (e == 31) { uint32_t f32 = (sign << 31) | (0xFF << 23) | (m << 13); memcpy(&dmin, &f32, 4); }
    else { uint32_t f32 = (sign << 31) | ((e+112) << 23) | (m << 13); memcpy(&dmin, &f32, 4); }

    const uint8_t *sc = blk + 4, *qs = blk + 16;
    int chunk = tid / 64, sub = (tid % 64) / 32, pos = tid % 32;
    uint8_t scv, mnv;
    get_scale_min_k4(chunk * 2 + sub, sc, &scv, &mnv);
    uint8_t qv = qs[chunk * 32 + pos];
    int nib = (sub == 0) ? (qv & 0xF) : (qv >> 4);
    W_f32[tid] = d * (float)scv * (float)nib - dmin * (float)mnv;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    // Read first Q4_K block from the model
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }

    // Read GGUF header
    char magic[4]; fread(magic, 1, 4, f);
    uint32_t version; fread(&version, 4, 1, f);
    int64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    printf("GGUF v%u, %ld tensors, %ld KV\n", version, (long)n_tensors, (long)n_kv);

    // Skip KV pairs
    for (int64_t i = 0; i < n_kv; i++) {
        uint64_t kl; fread(&kl, 8, 1, f);
        fseek(f, kl, SEEK_CUR);
        int32_t typ; fread(&typ, 4, 1, f);
        switch(typ) {
            case 8: { uint64_t vl; fread(&vl,8,1,f); fseek(f,vl,SEEK_CUR); break; }
            case 4: case 5: case 6: fseek(f,4,SEEK_CUR); break;
            case 7: fseek(f,1,SEEK_CUR); break;
            case 10: case 11: fseek(f,8,SEEK_CUR); break;
            case 9: { int32_t at; uint64_t al; fread(&at,4,1,f); fread(&al,8,1,f); fseek(f,al*(at==8?1:4),SEEK_CUR); break; }
            default: fseek(f,4,SEEK_CUR);
        }
    }

    // Find first Q4_K tensor
    int64_t data_offset = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        char name[256]; uint32_t name_len;
        // Read name length as uint64 in GGUF v3
        uint64_t nl; fread(&nl, 8, 1, f);
        size_t r = fread(name, 1, nl < 255 ? nl : 255, f);
        name[r] = 0;
        if (nl > 255) fseek(f, nl - 255, SEEK_CUR);

        uint32_t nd; fread(&nd, 4, 1, f);
        int64_t dims[4] = {0};
        for (int d2 = 0; d2 < nd && d2 < 4; d2++) fread(&dims[d2], 8, 1, f);
        int32_t gtype; fread(&gtype, 4, 1, f);
        uint64_t doff; fread(&doff, 8, 1, f);

        if (gtype == 12) {  // Q4_K = type 12
            printf("Found Q4_K tensor: %s, dims=[%lld,%lld], offset=%lu\n",
                   name, (long)dims[0], (long)dims[1], (unsigned long)doff);
            data_offset = doff;
            break;
        }
        if (i < 5) printf("  tensor %ld: %s type=%d\n", (long)i, name, gtype);
    }
    fclose(f);

    if (!data_offset) { fprintf(stderr, "No Q4_K tensor found\n"); return 1; }

    // Re-open and read the data blob
    f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read header to find data blob offset
    fseek(f, 4+4+8+8, SEEK_SET);  // skip magic+version+n_tensors+n_kv
    for (int64_t i = 0; i < n_kv; i++) {
        uint64_t kl; fread(&kl, 8, 1, f);
        fseek(f, kl, SEEK_CUR);
        int32_t typ; fread(&typ, 4, 1, f);
        if (typ == 8) { uint64_t vl; fread(&vl,8,1,f); fseek(f,vl,SEEK_CUR); }
        else fseek(f, (typ==10||typ==11)?8:4, SEEK_CUR);
    }
    for (int64_t i = 0; i < n_tensors; i++) {
        uint64_t nl; fread(&nl, 8, 1, f);
        fseek(f, nl, SEEK_CUR);
        uint32_t nd; fread(&nd, 4, 1, f);
        fseek(f, nd * 8 + 4 + 8, SEEK_CUR);
    }
    long data_start = ftell(f);
    long pad = (32 - (data_start % 32)) % 32;
    long blob_offset = data_start + pad;
    printf("Data blob at offset %ld\n", blob_offset);

    // Read first Q4_K block
    uint8_t block[Q4K_BLOCK_SIZE];
    fseek(f, blob_offset + data_offset, SEEK_SET);
    fread(block, 1, Q4K_BLOCK_SIZE, f);
    fclose(f);

    printf("First block raw: ");
    for (int i = 0; i < 18; i++) printf("%02x ", block[i]);
    printf("\n");

    // CPU dequant
    float cpu_out[256];
    printf("\n=== CPU dequant ===\n");
    cpu_dequant(block, cpu_out);

    // GPU dequant
    uint8_t *d_block; float *d_out;
    cudaMalloc((void**)&d_block, Q4K_BLOCK_SIZE);
    cudaMalloc((void**)&d_out, 256 * sizeof(float));
    cudaMemcpy(d_block, block, Q4K_BLOCK_SIZE, cudaMemcpyHostToDevice);
    gpu_dequant_kernel<<<1, 256>>>(d_block, d_out);
    float gpu_out[256];
    cudaMemcpy(gpu_out, d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== GPU dequant ===\n");
    printf("  GPU output[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", gpu_out[i]);
    printf("\n");

    // Compare
    printf("\n=== Comparison ===\n");
    int mismatches = 0;
    for (int i = 0; i < 256; i++) {
        float diff = fabsf(cpu_out[i] - gpu_out[i]);
        if (diff > 0.01f) {
            if (mismatches < 10)
                printf("  MISMATCH [%d]: CPU=%.4f GPU=%.4f diff=%.4f\n", i, cpu_out[i], gpu_out[i], diff);
            mismatches++;
        }
    }
    printf("Total mismatches: %d / 256\n", mismatches);

    cudaFree(d_block);
    cudaFree(d_out);
    return 0;
}
