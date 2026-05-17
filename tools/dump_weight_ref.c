/**
 * dump_weight_ref.c — Dump a model tensor via llama.cpp.
 * Build: g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/include \
 *   -I /home/wubu/llama.cpp/ggml/include -o dump_weight_ref tools/dump_weight_ref.c \
 *   -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
 *   -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 * Usage: ./dump_weight_ref model.gguf tensor_name
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s model.gguf tensor_name\n", argv[0]); return 1; }
    
    ggml_backend_load_all();
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed model\n"); return 1; }
    
    // Find tensor
    ggml_tensor *t = llama_model_get_tensor(model, argv[2]);
    if (!t) {
        // Try "model." prefix
        char prefixed[256];
        snprintf(prefixed, sizeof(prefixed), "model.%s", argv[2]);
        t = llama_model_get_tensor(model, prefixed);
    }
    if (!t) {
        fprintf(stderr, "Tensor '%s' not found\n", argv[2]);
        llama_model_free(model);
        return 1;
    }
    
    int64_t n_elems = 1;
    for (int d = 0; d < ggml_n_dims(t); d++) n_elems *= t->ne[d];
    
    fprintf(stderr, "Tensor '%s': dims=%d [", argv[2], ggml_n_dims(t));
    for (int d = 0; d < ggml_n_dims(t); d++)
        fprintf(stderr, "%lld%s", (long long)t->ne[d], d+1<ggml_n_dims(t) ? "," : "");
    fprintf(stderr, "] type=%d\n", t->type);
    
    // Read data — llama.cpp stores in host memory
    float *data = (float *)malloc(n_elems * sizeof(float));
    
    // Dequantize using ggml
    ggml_backend_tensor_get(t, data, 0, n_elems * sizeof(float));
    
    // Actually, for CPU tensors, we can just read the data directly
    // The tensor should be in CPU memory
    const float *raw = (const float *)t->data;
    if (t->type == 0) { // F32
        memcpy(data, raw, n_elems * sizeof(float));
    } else {
        // Need to dequantize using ggml API
        // The tensor data is quantized, we need to convert
        // Use ggml_backend_tensor_get which handles this
        ggml_backend_tensor_get(t, data, 0, n_elems * sizeof(float));
    }
    
    // Stats
    double sum=0, sumsq=0;
    float minv = 1e30f, maxv = -1e30f;
    int64_t check = n_elems < 10000000 ? n_elems : 10000000;
    for (int64_t i = 0; i < check; i++) {
        sum += data[i];
        sumsq += data[i] * data[i];
        if (data[i] < minv) minv = data[i];
        if (data[i] > maxv) maxv = data[i];
    }
    fprintf(stderr, "Stats (first %lld elems): mean=%.6f rms=%.6f min=%.6f max=%.6f\n",
           (long long)check, (float)(sum/check), (float)sqrt(sumsq/check), minv, maxv);
    
    // Dump first 16384 elements
    char fn[256];
    // Replace dots with underscores in tensor name
    const char *base = strchr(argv[2], '.');
    base = base ? base + 1 : argv[2];
    snprintf(fn, sizeof(fn), "/tmp/ref_weight_%s.bin", base);
    for (char *p = fn; *p; p++) if (*p == '.') *p = '_';
    FILE *f = fopen(fn, "wb");
    if (f) {
        int64_t dump_n = n_elems < 16384 ? n_elems : 16384;
        fwrite(data, sizeof(float), dump_n, f);
        fclose(f);
        fprintf(stderr, "Saved to %s (first %lld elems)\n", fn, (long long)dump_n);
    }
    
    free(data);
    llama_model_free(model);
    return 0;
}
