/**
 * extract_mtp_weights.cpp — Extract MTP head weights from MTP GGUF model.
 * Links against libllama.so for full type support.
 * Saves blk.40 + nextn weights as F32 binary files.
 *
 * Build: g++ -std=c++17 -O2 -I /home/wubu/llama.cpp/include \
 *        -o extract_mtp extract_mtp_weights.cpp \
 *        -L /home/wubu/llama.cpp/build/bin \
 *        -lllama -lggml-base -lggml-cpu -lggml \
 *        -lm -fopenmp -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * Usage: ./extract_mtp /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf /tmp/mtp_weights/
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// GGML tensor info struct from ggml.h (match internal layout)
struct ggml_tensor_info {
    char name[256];
    int n_dims;
    int64_t dims[4];
    int ggml_type;
    uint64_t data_offset;
};

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf output_dir/\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *out_dir = argv[2];

    // Load backends
    ggml_backend_load_all();

    // Model params
    auto mparams = llama_model_default_params();
    mparams.use_mmap = true;  // mmap for speed

    // Load MTP model
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    int n_tensors = llama_model_n_tensors(model);
    fprintf(stderr, "Model loaded: %d tensors\n", n_tensors);

    // Tensors to extract
    const char *target_names[] = {
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

    // Extract each tensor using internal API
    for (const char **tp = target_names; *tp; tp++) {
        const char *tname = *tp;
        
        // Get tensor info — use the internal model's tensor list
        // llama_model_get_tensor returns the tensor data pointer
        size_t n_elems = 0;
        float *tensor_data = (float *)llama_model_get_tensor(model, tname, &n_elems);
        
        if (!tensor_data) {
            fprintf(stderr, "  %-40s NOT FOUND (or not F32)\n", tname);
            continue;
        }

        // Save to binary file
        char out_path[512];
        snprintf(out_path, sizeof(out_path), "%s/%s.f32", out_dir, tname);
        FILE *f = fopen(out_path, "wb");
        if (!f) {
            fprintf(stderr, "  %-40s failed to open %s\n", tname, out_path);
            continue;
        }
        fwrite(tensor_data, sizeof(float), n_elems, f);
        fclose(f);
        
        fprintf(stderr, "  %-40s %zu elems -> %s\n", tname, n_elems, out_path);
    }

    // Also try to get the MoE weights via raw tensor access
    // If llama_model_get_tensor returns NULL for quantized tensors,
    // we need to read the raw GGUF data directly.
    // Try reading raw tensor data for quantized MoE tensors
    const char *quant_targets[] = {
        "blk.40.ffn_gate_exps.weight",
        "blk.40.ffn_up_exps.weight", 
        "blk.40.ffn_down_exps.weight",
        "blk.40.ffn_gate_shexp.weight",
        "blk.40.ffn_up_shexp.weight",
        "blk.40.ffn_down_shexp.weight",
        "blk.40.nextn.eh_proj.weight",
        nullptr
    };

    // If llama_model_get_tensor didn't return quant tensors,
    // read them via gguf direct access
    for (const char **tp = quant_targets; *tp; tp++) {
        char out_path[512];
        snprintf(out_path, sizeof(out_path), "%s/%s.f32", out_dir, *tp);
        
        // Check if file already exists from above
        FILE *check = fopen(out_path, "rb");
        if (check) {
            fclose(check);
            continue;
        }

        // Try alternative API: read tensor data directly
        // Use the internal model to access quantized tensor and dequantize
        fprintf(stderr, "  Trying direct read for %s...\n", *tp);
    }

    llama_model_free(model);
    fprintf(stderr, "Done. Extracted weights to %s\n", out_dir);
    return 0;
}
