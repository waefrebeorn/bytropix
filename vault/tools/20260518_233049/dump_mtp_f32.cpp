/**
 * dump_mtp_f32.cpp — Dump MTP head weights as F32 using libllama.so dequant.
 * Opens MTP GGUF, finds tensors by name, dequants via libllama, saves as .f32.
 *
 * Build:
 *   g++ -std=c++17 -O2 -I/home/wubu/llama.cpp/include -I/home/wubu/llama.cpp/ggml/include \
 *       -o dump_mtp_f32 dump_mtp_f32.cpp \
 *       -L/home/wubu/llama.cpp/build/bin \
 *       -lllama -lggml-base -lggml-cpu -lggml \
 *       -lm -fopenmp -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * Usage: ./dump_mtp_f32 /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf /tmp/mtp_weights/
 *
 * This tool duplicates the tensor lookup logic from gguf_reader's perspective
 * BUT uses libllama.so's dequant functions to handle all quant types.
 */
#include "ggml.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

// Reproduce GGUF header parsing to find tensor offsets
// We read the GGUF file directly for tensor metadata, then use ggml dequant

typedef struct {
    char name[256];
    int n_dims;
    int64_t dims[4];
    int ggml_type;
    uint64_t data_offset;
} tensor_info_t;

static std::vector<tensor_info_t> read_gguf_tensors(const char *path, uint64_t *data_start) {
    std::vector<tensor_info_t> tensors;
    
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", path); return tensors; }
    
    // GGUF header
    char magic[4]; fread(magic, 1, 4, f);
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    
    fprintf(stderr, "GGUF v%u, %llu tensors, %llu KV\n", version,
            (unsigned long long)n_tensors, (unsigned long long)n_kv);
    
    // Skip KV pairs
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; fread(&klen, 8, 1, f);
        fseek(f, klen, SEEK_CUR);
        uint32_t vtype; fread(&vtype, 4, 1, f);
        // Skip value based on type
        switch (vtype) {
            case 0:  fseek(f, 4, SEEK_CUR); break;   // uint32
            case 1:  fseek(f, 4, SEEK_CUR); break;   // int32
            case 2:  fseek(f, 4, SEEK_CUR); break;   // float32
            case 3:  fseek(f, 1, SEEK_CUR); break;   // bool
            case 4: { // string
                uint64_t slen; fread(&slen, 8, 1, f); fseek(f, slen, SEEK_CUR);
                break;
            }
            case 5: { // array
                uint32_t atype; fread(&atype, 4, 1, f);
                uint64_t alen; fread(&alen, 8, 1, f);
                for (uint64_t j = 0; j < alen; j++) {
                    if (atype == 4) { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); }
                    else if (atype == 0 || atype == 1) fseek(f, 4, SEEK_CUR);
                    else if (atype == 2) fseek(f, 4, SEEK_CUR);
                    else if (atype == 5) { fseek(f, 4, SEEK_CUR); uint64_t tl; fread(&tl, 8, 1, f); fseek(f, tl*8, SEEK_CUR); }
                    else if (atype == 8) fseek(f, 8, SEEK_CUR);
                    else if (atype == 12) fseek(f, 8, SEEK_CUR);
                    else fprintf(stderr, "Unknown array elem type %u\n", atype);
                }
                break;
            }
            case 6:  fseek(f, 8, SEEK_CUR); break;   // uint32 (?)

            case 7:  fseek(f, 8, SEEK_CUR); break;   // float64
            case 8:  fseek(f, 8, SEEK_CUR); break;   // uint64
            case 12: fseek(f, 8, SEEK_CUR); break;   // int64
            default:
                fprintf(stderr, "Unknown KV type %u at offset %ld\n", vtype, ftell(f)-4);
                // Try to skip 4 bytes and continue
                fseek(f, 4, SEEK_CUR);
                break;
        }
    }
    
    // Get alignment
    uint32_t alignment = 32;
    
    // Align to tensor info start
    long info_start = ftell(f);
    long pad = (alignment - (info_start % alignment)) % alignment;
    if (pad) fseek(f, pad, SEEK_CUR);
    
    fprintf(stderr, "Tensor info at offset %ld\n", ftell(f));
    
    // Read tensor info
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info_t ti;
        memset(&ti, 0, sizeof(ti));
        
        uint64_t nlen; fread(&nlen, 8, 1, f);
        if (nlen >= 256) { fprintf(stderr, "Tensor name too long: %llu\n", (unsigned long long)nlen); break; }
        fread(ti.name, 1, nlen, f);
        ti.name[nlen] = '\0';
        
        fread(&ti.n_dims, 4, 1, f);
        for (int d = 0; d < ti.n_dims; d++) fread(&ti.dims[d], 8, 1, f);
        int raw_type; fread(&raw_type, 4, 1, f);
        ti.ggml_type = raw_type;
        fread(&ti.data_offset, 8, 1, f);
        
        tensors.push_back(ti);
        
        // Print blk.40 and nextn tensors
        if (strstr(ti.name, "blk.40") || strstr(ti.name, "nextn") || i < 3) {
            fprintf(stderr, "  [%3llu] %-40s type=%d dims=[", (unsigned long long)i, ti.name, ti.ggml_type);
            for (int d = 0; d < ti.n_dims; d++) fprintf(stderr, "%s%lld", d?",":"", (long long)ti.dims[d]);
            fprintf(stderr, "] offset=%llu\n", (unsigned long long)ti.data_offset);
        }
    }
    
    // Find data start (after all tensor infos)
    long after_info = ftell(f);
    long pad2 = (alignment - (after_info % alignment)) % alignment;
    *data_start = after_info + pad2;
    fprintf(stderr, "Data blob at offset %llu\n", (unsigned long long)*data_start);
    
    fclose(f);
    return tensors;
}

// Calculate raw_size for ggml types — needed to compute per-expert offsets for MoE
static int64_t raw_size(int gtype, int64_t n_elems) {
    // Use ggml's internal function
    return ggml_row_size((enum ggml_type)gtype, n_elems);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf output_dir/\n", argv[0]);
        return 1;
    }
    
    const char *model_path = argv[1];
    const char *out_dir = argv[2];
    
    // Create output dir
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
    system(cmd);
    
    // Read GGUF tensor info
    uint64_t data_start = 0;
    auto tensors = read_gguf_tensors(model_path, &data_start);
    if (tensors.empty()) { fprintf(stderr, "No tensors found\n"); return 1; }
    
    // Load model via llama.cpp (for dequant)
    ggml_backend_load_all();
    auto mparams = llama_model_default_params();
    mparams.use_mmap = true;  // mmap prevents loading weights again
    
    struct llama_model *llm = llama_model_load_from_file(model_path, mparams);
    if (!llm) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llm));
    int n_embd = llama_model_n_embd(llm);
    fprintf(stderr, "Model: n_embd=%d, n_vocab=%d\n", n_embd, n_vocab);
    
    // We need a context to get the actual tensor data
    // Actually, we can access tensors through the model's internal data
    // The model is mmap'd, so tensor data is directly accessible
    
    // Focus on blk.40 + nextn tensors
    const char *targets[] = {
        "blk.40.attn_norm.weight",
        "blk.40.attn_q.weight",
        "blk.40.attn_k.weight",
        "blk.40.attn_v.weight",
        "blk.40.attn_output.weight",
        "blk.40.attn_q_norm.weight",
        "blk.40.attn_k_norm.weight",
        "blk.40.ffn_gate_inp.weight",
        "blk.40.ffn_gate_inp_shexp.weight",
        "blk.40.ffn_gate_exps.weight",
        "blk.40.ffn_gate_shexp.weight",
        "blk.40.ffn_up_exps.weight",
        "blk.40.ffn_up_shexp.weight",
        "blk.40.ffn_down_exps.weight",
        "blk.40.ffn_down_shexp.weight",
        "blk.40.post_attention_norm.weight",
        "blk.40.nextn.hnorm.weight",
        "blk.40.nextn.enorm.weight",
        "blk.40.nextn.eh_proj.weight",
        "blk.40.nextn.shared_head_norm.weight",
        nullptr
    };
    
    // For each target: find tensor, compute total elements, dequant, save
    FILE *mf = fopen(model_path, "rb");
    if (!mf) { fprintf(stderr, "Failed to reopen model for raw reading\n"); return 1; }
    
    for (const char **tp = targets; *tp; tp++) {
        const char *tname = *tp;
        
        // Find tensor info
        tensor_info_t *ti = nullptr;
        for (auto &t : tensors) {
            if (strcmp(t.name, tname) == 0) { ti = &t; break; }
        }
        if (!ti) {
            fprintf(stderr, "  %-40s NOT FOUND\n", tname);
            continue;
        }
        
        // Compute total elements
        int64_t n_elems = 1;
        for (int d = 0; d < ti->n_dims; d++) n_elems *= ti->dims[d];
        
        // Compute raw size
        int64_t raw_sz = raw_size(ti->ggml_type, n_elems);
        if (raw_sz <= 0) {
            fprintf(stderr, "  %-40s unsupported type %d\n", tname, ti->ggml_type);
            continue;
        }
        
        // Read raw data from file
        std::vector<uint8_t> raw(raw_sz);
        fseek(mf, data_start + ti->data_offset, SEEK_SET);
        size_t nread = fread(raw.data(), 1, raw_sz, mf);
        if (nread != (size_t)raw_sz) {
            fprintf(stderr, "  %-40s read %zu/%lld bytes\n", tname, nread, (long long)raw_sz);
            continue;
        }
        
        // Dequantize using ggml API
        // Use ggml_type_name to verify type, then dequant
        std::vector<float> f32(n_elems);
        
        // Use ggml's built-in type dequant function
        ggml_type gtype = (ggml_type)ti->ggml_type;
        
        // For types we know are F32 or can just memcpy
        if (gtype == GGML_TYPE_F32) {
            memcpy(f32.data(), raw.data(), n_elems * sizeof(float));
            fprintf(stderr, "  %-40s %lld elems (F32) -> ", tname, (long long)n_elems);
        } else if (gtype == GGML_TYPE_F16) {
            uint16_t *fh = (uint16_t *)raw.data();
            for (int64_t i = 0; i < n_elems; i++) {
                f32[i] = ggml_fp16_to_fp32(fh[i]);
            }
            fprintf(stderr, "  %-40s %lld elems (F16) -> ", tname, (long long)n_elems);
        } else {
            // For quantized types, use ggml's dequant
            // ggml doesn't expose a generic "dequant_row" API directly.
            // But we can use ggml_type_info which has .to_float
            const struct ggml_type_traits *traits = ggml_get_type_traits(gtype);
            if (!traits || !traits->to_float) {
                fprintf(stderr, "  %-40s no dequant fn for type %d\n", tname, ti->ggml_type);
                continue;
            }
            traits->to_float(raw.data(), f32.data(), n_elems);
            fprintf(stderr, "  %-40s %lld elems (type=%d) -> ", tname, (long long)n_elems, ti->ggml_type);
        }
        
        // Save as F32 binary
        char out_path[512];
        snprintf(out_path, sizeof(out_path), "%s/%s.f32", out_dir, tname);
        FILE *of = fopen(out_path, "wb");
        if (of) {
            fwrite(f32.data(), sizeof(float), n_elems, of);
            fclose(of);
            fprintf(stderr, "%s\n", out_path);
        }
    }
    
    fclose(mf);
    llama_model_free(llm);
    fprintf(stderr, "\nDone. Weights saved to %s/\n", out_dir);
    return 0;
}
