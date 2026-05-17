/**
 * dequant_ref.c — Read raw GGUF data and dequantize using ggml.
 * Build: g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/ggml/include \
 *   -o dequant_ref tools/dequant_ref.c \
 *   -L /home/wubu/llama.cpp/build/bin -lggml-base -lggml \
 *   -lm -lstdc++ -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * Usage: ./dequant_ref model.gguf tensor_name expert_id
 */
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

// Minimal GGUF header parsing
struct gguf_header {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

struct gguf_tensor_info_t {
    char name[256];
    int n_dims;
    uint64_t ne[4];
    int type;
    uint64_t offset; // data offset from start of data section
};

static uint64_t read_uint64(FILE *f) {
    uint64_t v; fread(&v, 8, 1, f); return v;
}
static uint32_t read_uint32(FILE *f) {
    uint32_t v; fread(&v, 4, 1, f); return v;
}

int main(int argc, char **argv) {
    if (argc < 4) return 1;
    const char *path = argv[1];
    const char *tname = argv[2];
    int expert_id = atoi(argv[3]);
    
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    
    // Read header
    char magic[4];
    fread(magic, 1, 4, f);
    uint32_t version = read_uint32(f);
    uint64_t n_tensors = read_uint64(f);
    uint64_t n_kv = read_uint64(f);
    fprintf(stderr, "GGUF v%u, %llu tensors, %llu KV\n", version, 
            (unsigned long long)n_tensors, (unsigned long long)n_kv);
    
    // Skip KV metadata (simplified: just skip all values)
    for (uint64_t k = 0; k < n_kv; k++) {
        // Key
        uint64_t key_len = read_uint64(f);
        fseek(f, key_len, SEEK_CUR);
        // Value type
        uint32_t val_type = read_uint32(f);
        switch (val_type) {
            case 0: { // uint8
                uint8_t v; fread(&v, 1, 1, f); break;
            }
            case 1: { // int8
                int8_t v; fread(&v, 1, 1, f); break;
            }
            case 2: { // uint16
                uint16_t v; fread(&v, 2, 1, f); break;
            }
            case 3: { // int16
                int16_t v; fread(&v, 2, 1, f); break;
            }
            case 4: { // uint32
                uint32_t v; fread(&v, 4, 1, f); break;
            }
            case 5: { // int32
                int32_t v; fread(&v, 4, 1, f); break;
            }
            case 6: { // float32
                float v; fread(&v, 4, 1, f); break;
            }
            case 7: { // bool
                uint8_t v; fread(&v, 1, 1, f); break;
            }
            case 8: { // string
                uint64_t len = read_uint64(f);
                fseek(f, len, SEEK_CUR);
                break;
            }
            case 9: { // array
                uint32_t arr_type = read_uint32(f);
                uint64_t arr_len = read_uint64(f);
                for (uint64_t a = 0; a < arr_len; a++) {
                    // Re-use the type switch for array elements
                    // Each element type is determined by arr_type
                    // For simplicity, skip based on known types
                    if (arr_type == 8) { // array of strings
                        uint64_t slen = read_uint64(f);
                        fseek(f, slen, SEEK_CUR);
                    } else if (arr_type == 4 || arr_type == 5) {
                        fseek(f, 4, SEEK_CUR);
                    } else {
                        fseek(f, 1, SEEK_CUR); // skip 1 byte
                    }
                }
                break;
            }
            default:
                fprintf(stderr, "Unknown KV type %d\n", val_type);
                fclose(f);
                return 1;
        }
    }
    
    // Read tensor info
    std::vector<gguf_tensor_info_t> tensors;
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info_t ti;
        // Name
        uint64_t name_len = read_uint64(f);
        fread(ti.name, 1, name_len < 255 ? name_len : 255, f);
        ti.name[name_len < 255 ? name_len : 255] = 0;
        if (name_len > 255) fseek(f, name_len - 255, SEEK_CUR);
        
        // Dims
        ti.n_dims = read_uint32(f);
        for (int d = 0; d < ti.n_dims; d++)
            ti.ne[d] = read_uint64(f);
        for (int d = ti.n_dims; d < 4; d++)
            ti.ne[d] = 1;
        
        // Type
        ti.type = read_uint32(f);
        
        // Offset (relative to data start)
        ti.offset = read_uint64(f);
        
        tensors.push_back(ti);
    }
    
    // Find target tensor
    gguf_tensor_info_t *target = nullptr;
    for (auto &ti : tensors) {
        if (strcmp(ti.name, tname) == 0) {
            target = &ti;
            break;
        }
    }
    if (!target) {
        fprintf(stderr, "Tensor '%s' not found\n", tname);
        fclose(f);
        return 1;
    }
    
    int64_t ne_per_exp = 1;
    for (int d = 0; d < target->n_dims - 1; d++)
        ne_per_exp *= target->ne[d];
    
    int type = target->type;
    
    fprintf(stderr, "Tensor '%s': dims=%d [", tname, target->n_dims);
    for (int d = 0; d < target->n_dims; d++)
        fprintf(stderr, "%llu%s", (unsigned long long)target->ne[d], d+1<target->n_dims?",":"");
    fprintf(stderr, "] type=%d\n", type);
    fprintf(stderr, "  Elements per expert: %lld\n", (long long)ne_per_exp);
    fprintf(stderr, "  Expert %d requested\n", expert_id);
    
    // Find data offset
    // Tensor info ends at some position, find it by computing
    // The offset in the GGUF is from the start of the DATA section
    // The data section starts after all tensor info
    long data_section_start = ftell(f);
    
    // Calculate alignment padding
    long long data_start = (data_section_start + 31) & ~31; // 32-byte align
    
    // Read raw data for the expert
    // First calculate the raw size for ne_per_exp elements of this type
    int64_t n_blocks = (ne_per_exp + 255) / 256;
    int bytes_per_block = 0;
    switch (type) {
        case 16: bytes_per_block = 66; break;  // IQ2_XXS
        case 18: bytes_per_block = 116; break; // IQ3_XXS
        case 19: bytes_per_block = 144; break; // IQ4_XS? check...
        default: fprintf(stderr, "Unknown type %d\n", type); fclose(f); return 1;
    }
    int64_t raw_per_exp = n_blocks * bytes_per_block;
    
    // Expert data starts at: data_start + target->offset + expert_id * raw_per_exp
    long long expert_offset = data_start + target->offset + (long long)expert_id * raw_per_exp;
    fseek(f, expert_offset, SEEK_SET);
    
    uint8_t *raw = (uint8_t *)malloc(raw_per_exp);
    fread(raw, 1, raw_per_exp, f);
    
    // Dequantize using ggml
    float *deq = (float *)malloc(ne_per_exp * sizeof(float));
    
    // Use ggml_type_traits
    const auto *traits = ggml_get_type_traits((ggml_type)type);
    if (traits && traits->to_float) {
        traits->to_float(raw, deq, ne_per_exp);
        fprintf(stderr, "Dequantized using ggml type traits\n");
    } else {
        fprintf(stderr, "No to_float in ggml type traits for type %d\n", type);
        memset(deq, 0, ne_per_exp * sizeof(float));
    }
    
    // Compute stats
    double sum=0, sumsq=0;
    float mn=1e30f, mx=-1e30f;
    for (int64_t i = 0; i < ne_per_exp && i < 100000; i++) {
        sum += deq[i]; sumsq += deq[i]*deq[i];
        if (deq[i] < mn) mn = deq[i];
        if (deq[i] > mx) mx = deq[i];
    }
    int64_t check = ne_per_exp < 100000 ? ne_per_exp : 100000;
    fprintf(stderr, "  Stats (first %lld): mean=%.8f rms=%.8f min=%.6f max=%.6f\n",
            (long long)check, (float)(sum/check), (float)sqrt(sumsq/check), mn, mx);
    
    // Save to file for comparison
    FILE *out = fopen("/tmp/ggml_exp0_gate_deq.bin", "wb");
    if (out) {
        fwrite(deq, sizeof(float), 256, out);
        fclose(out);
        fprintf(stderr, "  Saved to /tmp/ggml_exp0_gate_deq.bin (first 256 values)\n");
    }
    
    free(raw);
    free(deq);
    fclose(f);
    return 0;
}
